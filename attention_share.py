"""Mechanistic smoke test: where does the answer position look?

For each composite image we ask the dish question and the context question and
measure, at every LLM layer, what share of the last token's attention to the
image goes to PLATE tokens vs BACKGROUND tokens.

If the model routes evidence by question, the plate share should be higher for
the dish question than for the context question. The dashed line in the plot
is the plate's share of image AREA (what you'd get by attending uniformly).

This is a coarse first look, not a causal test. Attention knockout / activation
patching come later.

Usage:
  python attention_share.py --pairs pairs/meta.jsonl --n 30 --out attn
"""

import argparse
import csv
import json
import random

import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from common import COUNTRY_NAME, pretty
from run_eval import CTX_PROMPT, DISH_PROMPT, ctx_options, dish_options, fmt


def load(model_id, max_pixels):
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, device_map="auto",
        attn_implementation="eager")  # eager is needed to return attention maps
    model.eval()
    proc = AutoProcessor.from_pretrained(model_id, min_pixels=128 * 28 * 28,
                                         max_pixels=max_pixels)
    return model, proc


@torch.no_grad()
def plate_share(model, proc, image_path, mask_path, prompt):
    img = Image.open(image_path).convert("RGB")
    messages = [{"role": "user", "content": [{"type": "image"},
                                             {"type": "text", "text": prompt}]}]
    text = proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = proc(text=[text], images=[img], return_tensors="pt").to(model.device)
    out = model(**inputs, output_attentions=True)

    img_tok = proc.tokenizer.convert_tokens_to_ids("<|image_pad|>")
    pos = (inputs["input_ids"][0] == img_tok).nonzero().squeeze(-1)

    # image tokens are the merged patch grid in row-major order
    _, gh, gw = inputs["image_grid_thw"][0].tolist()
    m = proc.image_processor.merge_size
    gh, gw = gh // m, gw // m
    assert len(pos) == gh * gw, (len(pos), gh, gw)
    mask = Image.open(mask_path).convert("L").resize((gw, gh), Image.BOX)
    plate = torch.from_numpy(np.array(mask) / 255.0 > 0.5).flatten().to(model.device)

    shares = []
    for att in out.attentions:                  # [1, heads, seq, seq] per layer
        a = att[0, :, -1, pos].float().mean(0)  # last token -> image tokens, head-avg
        shares.append((a[plate].sum() / a.sum()).item())
    return shares, plate.float().mean().item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-3B-Instruct")
    ap.add_argument("--max_pixels", type=int, default=512 * 28 * 28)
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--out", default="attn")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    recs = [json.loads(l) for l in open(args.pairs)]
    recs = [r for r in recs if r["condition"] == "mismatched"]
    random.Random(args.seed).shuffle(recs)
    recs = recs[:args.n]
    model, proc = load(args.model, args.max_pixels)

    rows, dish_all, ctx_all, areas = [], [], [], []
    for r in recs:
        rng = random.Random(f"{args.seed}-{r['id']}")
        d_opts, _ = dish_options(r, rng)
        c_opts = ctx_options(r, rng)
        s_d, area = plate_share(model, proc, r["image"], r["mask"],
                                DISH_PROMPT.format(opts=fmt(d_opts, pretty)))
        s_c, _ = plate_share(model, proc, r["image"], r["mask"],
                             CTX_PROMPT.format(opts=fmt(c_opts, COUNTRY_NAME.get)))
        dish_all.append(s_d)
        ctx_all.append(s_c)
        areas.append(area)
        for L, (a, b) in enumerate(zip(s_d, s_c)):
            rows.append({"id": r["id"], "layer": L, "plate_share_dish_q": a,
                         "plate_share_context_q": b, "plate_area": area})

    with open(f"{args.out}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    d = np.array(dish_all).mean(0)
    c = np.array(ctx_all).mean(0)
    area = float(np.mean(areas))
    print(f"plate area share: {area:.3f}")
    print("layer  dish_q  context_q")
    for L in range(len(d)):
        print(f"{L:5d}  {d[L]:.3f}   {c[L]:.3f}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(d, label="dish question")
    ax.plot(c, label="context question")
    ax.axhline(area, ls="--", c="gray", label="plate area (uniform attention)")
    ax.set_xlabel("LLM layer")
    ax.set_ylabel("share of image attention on plate")
    ax.set_title(f"Where the answer looks ({args.model.split('/')[-1]}, n={len(recs)})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{args.out}.png", dpi=150)
    print(f"saved {args.out}.csv and {args.out}.png")


if __name__ == "__main__":
    main()

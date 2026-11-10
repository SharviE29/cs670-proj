"""Smoke test: does the background change the VLM's answer, and in the right way?

Paired design: for each (food, background) record we ask the SAME multiple-choice
question, with the SAME options in the SAME order, on several images:

  Dish question  ("What dish is on the plate?")
     on: composite (food + country background) | neutral (food + plain bg) | original photo
     options: true dish, a dish from the BACKGROUND's country ("leak" distractor), 2 others
     -> want: same answer everywhere (INVARIANCE), no pull toward the leak dish

  Context question ("Ignore the food. Which country is the background from?")
     on: composite | background only
     options: true background country, the FOOD's country, 2 others
     -> want: correct on both (SENSITIVITY), the food shouldn't pull the answer

Answers are read from next-token probabilities of the letters A-D (no free-text parsing).

Usage:
  python run_eval.py --pairs pairs/meta.jsonl --out results.csv
  python run_eval.py --pairs pairs/meta.jsonl --out results.csv --model Qwen/Qwen2.5-VL-7B-Instruct
  python run_eval.py --summarize_only results.csv
  python run_eval.py --pairs pairs/meta.jsonl --out dry.csv --dry_run   # no GPU, random answers
"""

import argparse
import csv
import json
import math
import random

from common import DISH_COUNTRY, COUNTRY_NAME, COUNTRIES, LETTERS, pretty

DISH_PROMPT = "What dish is shown on the plate?\n{opts}\nAnswer with the option's letter only."
CTX_PROMPT = ("Ignore the food. In which country was the background scene most likely "
              "photographed?\n{opts}\nAnswer with the option's letter only.")


# ---------------------------------------------------------------- options
def dish_options(rec, rng):
    true, fc, bc = rec["dish"], rec["food_country"], rec["bg_country"]
    opts, leak = [true], ""
    if bc != fc:
        leak = rng.choice([d for d, c in DISH_COUNTRY.items() if c == bc])
        opts.append(leak)
    pool = [d for d, c in DISH_COUNTRY.items() if c not in (fc, bc)]
    opts += rng.sample(pool, 4 - len(opts))
    rng.shuffle(opts)
    return opts, leak


def ctx_options(rec, rng):
    fc, bc = rec["food_country"], rec["bg_country"]
    opts = [bc] + ([fc] if fc != bc else [])
    opts += rng.sample([c for c in COUNTRIES if c not in opts], 4 - len(opts))
    rng.shuffle(opts)
    return opts


def fmt(opts, namer):
    return "\n".join(f"{LETTERS[i]}. {namer(o)}" for i, o in enumerate(opts))


# ---------------------------------------------------------------- model
class Scorer:
    def __init__(self, model_id, max_pixels, dry_run=False, seed=0):
        self.dry = dry_run
        self.rng = random.Random(seed)
        if dry_run:
            return
        import torch
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
        self.torch = torch
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id, torch_dtype=torch.bfloat16, device_map="auto")
        self.model.eval()
        self.proc = AutoProcessor.from_pretrained(
            model_id, min_pixels=128 * 28 * 28, max_pixels=max_pixels)
        tok = self.proc.tokenizer
        self.letter_ids = [tok.encode(L, add_special_tokens=False)[0] for L in LETTERS]

    def probs(self, image_path, prompt):
        """Probability over A-D for the first answer token."""
        if self.dry:
            w = [self.rng.random() for _ in LETTERS]
            s = sum(w)
            return [x / s for x in w]
        from PIL import Image
        torch = self.torch
        img = Image.open(image_path).convert("RGB")
        messages = [{"role": "user", "content": [
            {"type": "image"}, {"type": "text", "text": prompt}]}]
        text = self.proc.apply_chat_template(messages, tokenize=False,
                                             add_generation_prompt=True)
        inputs = self.proc(text=[text], images=[img], return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            logits = self.model(**inputs).logits[0, -1]
        p = torch.softmax(logits[self.letter_ids].float(), 0)
        return p.tolist()


def pick(opts, p):
    return opts[max(range(len(p)), key=lambda i: p[i])]


# ---------------------------------------------------------------- evaluation
def evaluate(pairs_path, scorer, out_csv, limit=None, seed=0):
    recs = [json.loads(l) for l in open(pairs_path)]
    if limit:
        recs = recs[:limit]
    rows = []
    for k, r in enumerate(recs):
        rng = random.Random(f"{seed}-{r['id']}")  # same options every run
        d_opts, leak = dish_options(r, rng)
        d_prompt = DISH_PROMPT.format(opts=fmt(d_opts, pretty))
        p_comp = scorer.probs(r["image"], d_prompt)
        p_neu = scorer.probs(r["neutral"], d_prompt)
        p_orig = scorer.probs(r["original"], d_prompt)

        c_opts = ctx_options(r, rng)
        c_prompt = CTX_PROMPT.format(opts=fmt(c_opts, COUNTRY_NAME.get))
        pc_comp = scorer.probs(r["image"], c_prompt)
        pc_bg = scorer.probs(r["bg_only"], c_prompt)

        ti = d_opts.index(r["dish"])
        li = d_opts.index(leak) if leak else None
        rows.append({
            "id": r["id"], "food_id": r["food_id"], "dish": r["dish"],
            "food_country": r["food_country"], "bg_country": r["bg_country"],
            "condition": r["condition"], "dish_options": json.dumps(d_opts), "leak": leak,
            "dish_pred_comp": pick(d_opts, p_comp),
            "dish_pred_neutral": pick(d_opts, p_neu),
            "dish_pred_original": pick(d_opts, p_orig),
            "p_true_comp": p_comp[ti], "p_true_neutral": p_neu[ti], "p_true_original": p_orig[ti],
            "p_leak_comp": p_comp[li] if li is not None else "",
            "p_leak_neutral": p_neu[li] if li is not None else "",
            "ctx_options": json.dumps(c_opts),
            "ctx_pred_comp": pick(c_opts, pc_comp), "ctx_pred_bgonly": pick(c_opts, pc_bg),
            "p_bg_comp": pc_comp[c_opts.index(r["bg_country"])],
            "p_bg_bgonly": pc_bg[c_opts.index(r["bg_country"])],
        })
        if (k + 1) % 20 == 0:
            print(f"  {k + 1}/{len(recs)}")
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} rows to {out_csv}")
    return rows


# ---------------------------------------------------------------- metrics
def prop(xs):
    """Mean of booleans with a 95% normal-approx confidence half-width."""
    n = len(xs)
    if n == 0:
        return float("nan"), float("nan"), 0
    p = sum(xs) / n
    return p, 1.96 * math.sqrt(max(p * (1 - p), 1e-12) / n), n


def line(name, xs):
    p, ci, n = prop(xs)
    return f"  {name:<52s} {100 * p:5.1f}% ± {100 * ci:4.1f}  (n={n})"


def summarize(rows):
    mis = [r for r in rows if r["condition"] == "mismatched"]
    mat = [r for r in rows if r["condition"] == "matched"]
    out = ["\n=== DISH QUESTION (should NOT depend on background) ==="]
    out.append(line("accuracy, original photo", [r["dish_pred_original"] == r["dish"] for r in rows]))
    out.append(line("accuracy, neutral background (paste control)",
                    [r["dish_pred_neutral"] == r["dish"] for r in rows]))
    out.append(line("accuracy, matched-culture background",
                    [r["dish_pred_comp"] == r["dish"] for r in mat]))
    out.append(line("accuracy, mismatched-culture background",
                    [r["dish_pred_comp"] == r["dish"] for r in mis]))
    out.append(line("INVARIANCE: same answer as neutral (mismatched)",
                    [r["dish_pred_comp"] == r["dish_pred_neutral"] for r in mis]))
    leak_c = [r["dish_pred_comp"] == r["leak"] for r in mis]
    leak_n = [r["dish_pred_neutral"] == r["leak"] for r in mis]
    out.append(line("LEAKAGE: picks background-country dish, cultural bg", leak_c))
    out.append(line("LEAKAGE: picks same distractor, neutral bg", leak_n))
    if mis:
        dp = [float(r["p_leak_comp"]) - float(r["p_leak_neutral"]) for r in mis]
        out.append(f"  mean change in P(leak dish) from adding background:  {sum(dp) / len(dp):+.3f}")

    out.append("\n=== CONTEXT QUESTION (SHOULD depend on background) ===")
    out.append(line("SENSITIVITY: accuracy, background only",
                    [r["ctx_pred_bgonly"] == r["bg_country"] for r in rows]))
    out.append(line("SENSITIVITY: accuracy, background + food",
                    [r["ctx_pred_comp"] == r["bg_country"] for r in rows]))
    out.append(line("reverse leakage: picks FOOD's country (mismatched)",
                    [r["ctx_pred_comp"] == r["food_country"] for r in mis]))
    out.append(line("  ...same option, background only (mismatched)",
                    [r["ctx_pred_bgonly"] == r["food_country"] for r in mis]))

    # verdict
    acc_neu = prop([r["dish_pred_neutral"] == r["dish"] for r in mis])[0]
    acc_mis = prop([r["dish_pred_comp"] == r["dish"] for r in mis])[0]
    lc, ln = prop(leak_c)[0], prop(leak_n)[0]
    out.append("\n=== SMOKE-TEST VERDICT ===")
    out.append(f"  dish accuracy drop (neutral -> mismatched bg): {100 * (acc_neu - acc_mis):+.1f} pts")
    out.append(f"  leakage increase from background:              {100 * (lc - ln):+.1f} pts")
    if acc_neu - acc_mis >= 0.05 or lc - ln >= 0.05:
        out.append("  -> Background effect is visible. Proceed to training baselines.")
    else:
        out.append("  -> Effect is small. Try --plate_frac 0.25, more items, or a weaker model "
                   "before concluding (see README).")
    text = "\n".join(out)
    print(text)
    return text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs")
    ap.add_argument("--out", default="results.csv")
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-3B-Instruct")
    ap.add_argument("--max_pixels", type=int, default=512 * 28 * 28)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--summarize_only")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if args.summarize_only:
        rows = list(csv.DictReader(open(args.summarize_only)))
    else:
        scorer = Scorer(args.model, args.max_pixels, args.dry_run, args.seed)
        rows = evaluate(args.pairs, scorer, args.out, args.limit, args.seed)
        rows = list(csv.DictReader(open(args.out)))  # same types as summarize_only
    summary = summarize(rows)
    if not args.summarize_only:
        with open(args.out.replace(".csv", "_summary.txt"), "w") as f:
            f.write(summary)


if __name__ == "__main__":
    main()

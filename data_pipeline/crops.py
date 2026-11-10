"""Step 2: dish cutouts from the Food-101 TEST split.

    python -m data_pipeline.crops extract [--classes pho tacos ...] [--limit_per_class N]
    python -m data_pipeline.crops qwen-check        # crop-only accuracy of Qwen2.5-VL-3B (logged exception)
    python -m data_pipeline.crops finalize          # apply per-class rules, write class_status.json
    python -m data_pipeline.crops report            # rejection stats + contact sheets

Per photo: GroundingDINO("plate of food.") -> best box -> SAM2 box prompt -> mask (whole plate
+ food) -> filters, each rejection logged with a reason in crops/rejections.jsonl:
  no_detection            no box above box_threshold
  mask_area_out_of_range  mask area / image area outside crops.mask_area_frac
  fragmented_mask         largest connected component < min_dominant_cc_frac of the mask
  clip_not_top1           CLIP zero-shot over the 50 class names (cutout on a gray canvas)
                          does not rank the true class first
  class_full              the class already has target_max_per_class accepted cutouts
                          (not a quality rejection; logged so counts add up)
Saved per accepted photo: crops/<class>/<id>.png (RGBA: RGB = original photo pixels in the
mask's bbox, A = hard mask) and crops/<class>/<id>_mask.png (mask in source-image coords).
The crop id is "<class>/<Food-101 image id>", i.e. the source photo id (one cutout/photo).
CLIP is not part of the evaluation panel (selection-bias rule). The Qwen check is the one
deliberate exception and is logged as such.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import label as cc_label

from .classes import CLASSES, pretty
from .utils import (Config, derive_seed, get_logger, load_config, pick_device,
                    read_jsonl, seed_everything, write_json, write_jsonl)


# ---------------------------------------------------------------- sources
def iter_food101(cfg: Config, classes: list[str], rng_seed: int, limit: int):
    """Yield (class, image_id, PIL image) for the TEST split, per class in a seeded order."""
    fc = cfg["food101"]
    assert fc["split"] == "test", "spec: use the Food-101 test split only"
    if fc["source"] == "local":
        root = cfg.path("food101_root")
        lines = (root / "meta" / "test.txt").read_text().split()
        by_cls = defaultdict(list)
        for ln in lines:
            c, _ = ln.split("/")
            by_cls[c].append(ln)
        for c in classes:
            ids = sorted(by_cls[c])
            random.Random(derive_seed(rng_seed, c)).shuffle(ids)
            for iid in ids[:limit]:
                yield c, iid, Image.open(root / "images" / f"{iid}.jpg").convert("RGB")
    elif fc["source"] == "hf":
        # NOTE: HF parquet has no file names. IDs become "<class>/hfval<index>" (index in the
        # HF validation split). Fine for local demos; use the official tarball for the release.
        from datasets import load_dataset

        ds = load_dataset(fc["hf_name"], split=fc["hf_split"], streaming=True,
                          cache_dir=cfg["paths"].get("hf_cache"))
        names = ds.features["label"].names
        want = set(classes)
        seen = Counter()
        for idx, ex in enumerate(ds):
            c = names[ex["label"]]
            if c in want and seen[c] < limit:
                seen[c] += 1
                yield c, f"{c}/hfval{idx:05d}", ex["image"].convert("RGB")
            if all(seen[k] >= limit for k in want):
                break
    elif fc["source"] == "parquet":
        # Local copies of the HF validation shards, listed IN ORDER starting at shard 0 so the
        # running row index equals the HF validation index. Same ID caveat as "hf".
        import pyarrow.parquet as pq
        from io import BytesIO

        names = None
        offset = 0
        want = set(classes)
        seen = Counter()
        for f in fc["parquet_files"]:
            t = pq.read_table(f)
            if names is None:
                meta = json.loads(t.schema.metadata[b"huggingface"])
                names = meta["info"]["features"]["label"]["names"]
            labels = t.column("label").to_pylist()
            for i, lab in enumerate(labels):
                c = names[lab]
                if c in want and seen[c] < limit:
                    seen[c] += 1
                    img = t.column("image")[i].as_py()
                    yield c, f"{c}/hfval{offset + i:05d}", Image.open(BytesIO(img["bytes"])).convert("RGB")
            offset += len(labels)
    else:
        raise ValueError(fc["source"])


# ---------------------------------------------------------------- models
class Segmenter:
    def __init__(self, cfg: Config, device: str):
        import torch
        from transformers import (AutoModelForZeroShotObjectDetection, AutoProcessor, Sam2Model,
                                  Sam2Processor)

        cc = cfg["crops"]
        self.torch, self.device, self.cc = torch, device, cc
        self.dp = AutoProcessor.from_pretrained(cc["detector_model"])
        self.dm = AutoModelForZeroShotObjectDetection.from_pretrained(cc["detector_model"]).to(device).eval()
        self.sp = Sam2Processor.from_pretrained(cc["segmenter_model"])
        self.sm = Sam2Model.from_pretrained(cc["segmenter_model"]).to(device).eval()

    def detect(self, img: Image.Image):
        torch = self.torch
        inp = self.dp(images=img, text=self.cc["detector_prompt"], return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.dm(**inp)
        r = self.dp.post_process_grounded_object_detection(
            out, inp.input_ids, threshold=self.cc["box_threshold"],
            text_threshold=self.cc["text_threshold"], target_sizes=[img.size[::-1]])[0]
        if len(r["scores"]) == 0:
            return None, 0.0
        i = int(r["scores"].argmax())
        return [float(v) for v in r["boxes"][i].tolist()], float(r["scores"][i])

    def segment(self, img: Image.Image, box):
        """SAM2 multimask on the box; choose per crops.sam_mask_select."""
        torch = self.torch
        si = self.sp(images=img, input_boxes=[[box]], return_tensors="pt").to(self.device)
        with torch.no_grad():
            so = self.sm(**si, multimask_output=True)
        masks = self.sp.post_process_masks(so.pred_masks.cpu(), si["original_sizes"].cpu())[0][0].numpy()
        ious = so.iou_scores[0, 0].float().cpu().numpy()
        ok = [k for k in range(len(ious)) if ious[k] >= self.cc["sam_min_iou"]] or [int(ious.argmax())]
        if self.cc["sam_mask_select"] == "largest":
            k = max(ok, key=lambda j: masks[j].sum())
        else:
            k = max(ok, key=lambda j: ious[j])
        return masks[k].astype(bool), float(ious[k])


class ClipClassifier:
    def __init__(self, model_id: str, labels: list[str], prompt: str, device: str):
        import torch
        from transformers import CLIPModel, CLIPProcessor

        self.torch, self.device, self.labels = torch, device, labels
        self.proc = CLIPProcessor.from_pretrained(model_id)
        self.model = CLIPModel.from_pretrained(model_id).to(device).eval()
        txt = self.proc(text=[prompt.format(pretty(l)) for l in labels], return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            t = self.model.get_text_features(**txt)
            t = t if isinstance(t, torch.Tensor) else t.pooler_output
        self.text = t / t.norm(dim=-1, keepdim=True)

    def probs(self, img: Image.Image) -> np.ndarray:
        torch = self.torch
        inp = self.proc(images=img, return_tensors="pt").to(self.device)
        with torch.no_grad():
            v = self.model.get_image_features(**inp)
            v = v if isinstance(v, torch.Tensor) else v.pooler_output
            v = v / v.norm(dim=-1, keepdim=True)
            logits = self.model.logit_scale.exp() * v @ self.text.T
        return logits.softmax(-1)[0].float().cpu().numpy()


# ---------------------------------------------------------------- mask utils
def dominant_component(mask: np.ndarray) -> tuple[np.ndarray, float]:
    lab, n = cc_label(mask)
    if n == 0:
        return mask, 0.0
    sizes = np.bincount(lab.ravel())[1:]
    k = int(sizes.argmax()) + 1
    return lab == k, float(sizes.max() / sizes.sum())


def make_cutout(img: Image.Image, mask: np.ndarray, pad: int) -> Image.Image:
    ys, xs = np.where(mask)
    H, W = mask.shape
    y0, y1 = max(0, ys.min() - pad), min(H, ys.max() + 1 + pad)
    x0, x1 = max(0, xs.min() - pad), min(W, xs.max() + 1 + pad)
    rgba = img.crop((x0, y0, x1, y1)).convert("RGBA")
    rgba.putalpha(Image.fromarray((mask[y0:y1, x0:x1] * 255).astype(np.uint8)))
    return rgba


def on_gray(rgba: Image.Image, size: int, gray: int, fit: int | None = None) -> Image.Image:
    """Cutout centred on a uniform gray square canvas (used by CLIP and Qwen checks)."""
    fit = fit or size
    s = min(fit / rgba.width, fit / rgba.height)
    im = rgba.resize((max(1, round(rgba.width * s)), max(1, round(rgba.height * s))), Image.LANCZOS)
    canvas = Image.new("RGB", (size, size), (gray,) * 3)
    canvas.paste(im, ((size - im.width) // 2, (size - im.height) // 2), im)
    return canvas


# ---------------------------------------------------------------- extract
def extract(cfg: Config, classes: list[str], limit: int | None = None, fresh: bool = False) -> dict:
    """Resumable: every processed photo is appended to crops/_progress.jsonl immediately, so a
    crashed / disconnected run continues where it stopped (pass fresh=True to start over).
    index_raw.jsonl and rejections.jsonl are rebuilt from the progress log at the end."""
    log = get_logger("crops", cfg)
    cc = cfg["crops"]
    seed = derive_seed(cfg["seed"], "crops")
    seed_everything(seed, log)
    device = pick_device(cc["device"])
    log.info("device=%s detector=%s segmenter=%s clip=%s", device, cc["detector_model"],
             cc["segmenter_model"], cc["class_check"]["model"])
    out_dir = cfg.path("crops_dir")
    out_dir.mkdir(parents=True, exist_ok=True)
    prog_path = out_dir / "_progress.jsonl"
    if fresh and prog_path.exists():
        prog_path.unlink()
    done = {r["id"]: r for r in read_jsonl(prog_path)} if prog_path.exists() else {}
    accepted = Counter(r["row"]["class"] for r in done.values() if "row" in r)
    if done:
        log.info("resuming: %d photos already processed, %d accepted", len(done), sum(accepted.values()))
    seg = Segmenter(cfg, device)
    ccheck = cc["class_check"]
    clip = ClipClassifier(ccheck["model"], CLASSES, ccheck["prompt"], device)
    prog = open(prog_path, "a")

    def rej(iid, reason, **details):
        prog.write(json.dumps({"id": iid, "reason": reason, **details}, sort_keys=True) + "\n")
        prog.flush()

    tried = Counter()
    lo, hi = cc["mask_area_frac"]
    for cls, iid, img in iter_food101(cfg, classes, seed, limit or cc["max_photos_per_class"]):
        tried[cls] += 1
        if iid in done:
            continue
        if accepted[cls] >= cc["target_max_per_class"]:
            rej(iid, "class_full")
            continue
        box, dscore = seg.detect(img)
        if box is None:
            rej(iid, "no_detection")
            continue
        mask, siou = seg.segment(img, box)
        area = float(mask.mean())
        dom, frac = dominant_component(mask)
        if not lo <= area <= hi:
            rej(iid, "mask_area_out_of_range", area=round(area, 4), det_score=round(dscore, 3))
            continue
        if frac < cc["min_dominant_cc_frac"]:
            rej(iid, "fragmented_mask", dominant_cc_frac=round(frac, 4))
            continue
        rgba = make_cutout(img, dom, cc["box_pad_px"])
        p = clip.probs(on_gray(rgba, ccheck["canvas_size"], ccheck["canvas_gray"]))
        top = CLASSES[int(p.argmax())]
        if ccheck["require_top1"] and top != cls:
            rej(iid, "clip_not_top1", clip_top1=top, clip_p_true=round(float(p[CLASSES.index(cls)]), 4))
            continue
        rel = Path(f"{iid}.png")
        (out_dir / rel).parent.mkdir(parents=True, exist_ok=True)
        rgba.save(out_dir / rel)
        Image.fromarray((dom * 255).astype(np.uint8)).save(out_dir / f"{iid}_mask.png")
        accepted[cls] += 1
        row = {
            "crop_id": iid, "class": cls, "source_id": iid,
            "source": "food101-test" if cfg["food101"]["source"] == "local" else "food101-test(hf-index)",
            "rgba_path": str(rel), "mask_path": f"{iid}_mask.png",
            "det_box": [round(v, 1) for v in box], "det_score": round(dscore, 4), "sam_iou": round(siou, 4),
            "mask_area_frac": round(area, 4), "dominant_cc_frac": round(frac, 4),
            "clip_p_true": round(float(p.max()), 4), "cutout_size": list(rgba.size),
        }
        prog.write(json.dumps({"id": iid, "row": row}, sort_keys=True) + "\n")
        prog.flush()
        if sum(tried.values()) % 50 == 0:
            log.info("processed %d photos this run; accepted so far %s", sum(tried.values()), dict(accepted))
    prog.close()
    return rebuild_extract_outputs(cfg, seed)


def rebuild_extract_outputs(cfg: Config, seed: int | None = None) -> dict:
    """index_raw.jsonl, rejections.jsonl and extract_stats.json from crops/_progress.jsonl."""
    log = get_logger("crops", cfg)
    out_dir = cfg.path("crops_dir")
    recs = list({r["id"]: r for r in read_jsonl(out_dir / "_progress.jsonl")}.values())
    rows = sorted((r["row"] for r in recs if "row" in r), key=lambda r: r["crop_id"])
    rejs = sorted((r for r in recs if "reason" in r), key=lambda r: r["id"])
    write_jsonl(out_dir / "index_raw.jsonl", rows)
    write_jsonl(cfg.path("crops_rejections"), rejs)
    stats = {"processed": len(recs), "accepted": dict(Counter(r["class"] for r in rows)),
             "rejections": dict(Counter(r["reason"] for r in rejs)), "seed": seed}
    write_json(out_dir / "extract_stats.json", stats)
    log.info("accepted %d / processed %d; rejections %s", len(rows), len(recs), stats["rejections"])
    return stats


# ---------------------------------------------------------------- qwen check (logged exception)
def qwen_check(cfg: Config) -> dict:
    """Crop-only accuracy of Qwen2.5-VL-3B per class, 4-option MC, cutout on a gray canvas at the
    render slot size. DELIBERATE EXCEPTION to the selection-bias rule (the spec requires it);
    recorded as such in the output file and in DATASHEET.md."""
    import torch
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    log = get_logger("crops", cfg)
    qc = cfg["crops"]["qwen_check"]
    seed = derive_seed(cfg["seed"], "qwen_check", qc["seed_offset"])
    rows = read_jsonl(cfg.path("crops_dir") / "index_raw.jsonl")
    device = pick_device(cfg["crops"]["device"])
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(qc["model"], dtype=getattr(torch, qc["dtype"])).to(device).eval()
    proc = AutoProcessor.from_pretrained(qc["model"], max_pixels=qc["max_pixels"])
    letters = cfg["questions"]["letters"][: qc["n_options"]]
    letter_ids = [proc.tokenizer.encode(l, add_special_tokens=False)[0] for l in letters]
    size = cfg["render"]["size"]
    x0, y0, x1, y1 = cfg["render"]["slots"]["left"]
    per_cls = defaultdict(lambda: [0, 0])
    items = []
    for r in rows:
        rng = random.Random(derive_seed(seed, r["crop_id"]))
        opts = [r["class"]] + rng.sample([c for c in CLASSES if c != r["class"]], qc["n_options"] - 1)
        rng.shuffle(opts)
        img = on_gray(Image.open(cfg.path("crops_dir") / r["rgba_path"]).convert("RGBA"), size,
                      cfg["crops"]["class_check"]["canvas_gray"], fit=max(x1 - x0, y1 - y0))
        prompt = "\n".join(["What dish is on the plate?"] + [f"{l}. {pretty(o)}" for l, o in zip(letters, opts)]
                           + [cfg["questions"]["answer_suffix"]])
        msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
        text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inp = proc(text=[text], images=[img], return_tensors="pt").to(device)
        with torch.no_grad():
            logits = model(**inp).logits[0, -1]
        pred = opts[int(logits[letter_ids].argmax())]
        ok = pred == r["class"]
        per_cls[r["class"]][0] += ok
        per_cls[r["class"]][1] += 1
        items.append({"crop_id": r["crop_id"], "options": opts, "pred": pred, "correct": ok})
    acc = {c: a / n for c, (a, n) in sorted(per_cls.items())}
    out = {
        "NOTE": "Deliberate exception to the selection-bias rule: Qwen2.5-VL-3B (an evaluation-panel "
                "model) is used to require >= min_class_accuracy crop-only accuracy per kept class.",
        "model": qc["model"], "seed": seed, "min_class_accuracy": qc["min_class_accuracy"],
        "accuracy": acc, "n": {c: n for c, (_, n) in sorted(per_cls.items())},
        "failing_classes": sorted(c for c, a in acc.items() if a < qc["min_class_accuracy"]),
        "items": items,
    }
    write_json(cfg.path("crops_qwen_check"), out)
    log.info("qwen check: failing classes %s", out["failing_classes"])
    return out


# ---------------------------------------------------------------- finalize + report
def finalize(cfg: Config, require_qwen: bool = True) -> dict:
    """Keep a class only if >= target_min_per_class cutouts survive (and Qwen acc passes).
    Writes crops/index.jsonl (kept classes only) and crops/class_status.json."""
    log = get_logger("crops", cfg)
    cc = cfg["crops"]
    rows = read_jsonl(cfg.path("crops_dir") / "index_raw.jsonl")
    n = Counter(r["class"] for r in rows)
    qpath = cfg.path("crops_qwen_check")
    qacc = json.load(open(qpath))["accuracy"] if qpath.exists() else None
    if require_qwen and qacc is None:
        raise SystemExit("run `crops qwen-check` first (or pass --no_qwen for a dev run)")
    status = {}
    for c in CLASSES:
        reasons = []
        if n[c] < cc["target_min_per_class"]:
            reasons.append(f"only {n[c]} cutouts (< {cc['target_min_per_class']})")
        if qacc is not None and qacc.get(c, 0.0) < cc["qwen_check"]["min_class_accuracy"]:
            reasons.append(f"qwen crop-only acc {qacc.get(c, 0.0):.2f} < {cc['qwen_check']['min_class_accuracy']}")
        status[c] = {"kept": not reasons, "n_cutouts": n[c], "reasons": reasons,
                     "qwen_acc": None if qacc is None else qacc.get(c)}
    kept = {c for c, s in status.items() if s["kept"]}
    write_jsonl(cfg.path("crops_index"), [r for r in rows if r["class"] in kept])
    write_json(cfg.path("crops_dir") / "class_status.json", status)
    dropped = sorted(set(CLASSES) - kept)
    log.info("kept %d/%d classes; dropped %s", len(kept), len(CLASSES), dropped)
    return status


def contact_sheet(cfg: Config, rows: list[dict], out: Path, per_row: int = 20, thumb: int = 96) -> None:
    by = defaultdict(list)
    for r in rows:
        by[r["class"]].append(r)
    classes = sorted(by)
    lw = 150
    sheet = Image.new("RGB", (lw + per_row * thumb, max(1, len(classes)) * thumb), (128, 128, 128))
    d = ImageDraw.Draw(sheet)
    for i, c in enumerate(classes):
        d.text((6, i * thumb + thumb // 2 - 6), f"{c} ({len(by[c])})", fill="white")
        for j, r in enumerate(by[c][:per_row]):
            im = Image.open(cfg.path("crops_dir") / r["rgba_path"]).convert("RGBA")
            im.thumbnail((thumb - 4, thumb - 4))
            sheet.paste(im, (lw + j * thumb + (thumb - im.width) // 2, i * thumb + (thumb - im.height) // 2), im)
    sheet.save(out, quality=90)


def report(cfg: Config) -> None:
    root = cfg.path("crops_dir")
    rows = read_jsonl(root / "index_raw.jsonl")
    rej = read_jsonl(cfg.path("crops_rejections"))
    by_cls = defaultdict(Counter)
    for r in rej:
        by_cls[r["id"].split("/")[0]][r["reason"]] += 1
    acc = Counter(r["class"] for r in rows)
    reasons = sorted({r["reason"] for r in rej})
    print(f"{'class':22s} {'accepted':>8s} " + " ".join(f"{x[:14]:>14s}" for x in reasons))
    for c in sorted(set(acc) | set(by_cls)):
        print(f"{c:22s} {acc[c]:8d} " + " ".join(f"{by_cls[c][x]:14d}" for x in reasons))
    tot = Counter(r["reason"] for r in rej)
    print(f"{'TOTAL':22s} {sum(acc.values()):8d} " + " ".join(f"{tot[x]:14d}" for x in reasons))
    contact_sheet(cfg, rows, root / "contact_sheet.jpg")
    print("contact sheet:", root / "contact_sheet.jpg")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["extract", "qwen-check", "finalize", "report"])
    ap.add_argument("--config")
    ap.add_argument("--classes", nargs="*", help="subset of classes (default: all 50)")
    ap.add_argument("--limit_per_class", type=int)
    ap.add_argument("--no_qwen", action="store_true", help="finalize without the Qwen check (dev only)")
    ap.add_argument("--fresh", action="store_true", help="extract: ignore crops/_progress.jsonl and start over")
    args = ap.parse_args()
    cfg = load_config(args.config)
    if args.cmd == "extract":
        classes = args.classes or CLASSES
        unknown = set(classes) - set(CLASSES)
        assert not unknown, f"unknown classes {unknown}"
        extract(cfg, classes, args.limit_per_class, fresh=args.fresh)
    elif args.cmd == "qwen-check":
        qwen_check(cfg)
    elif args.cmd == "finalize":
        finalize(cfg, require_qwen=not args.no_qwen)
    else:
        report(cfg)


if __name__ == "__main__":
    main()

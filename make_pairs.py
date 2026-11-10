"""Build paired images: the SAME food plate pasted onto different backgrounds.

For every food photo we save:
  original   - the food photo itself (center crop)
  neutral    - plate pasted on a plain / wood-texture background (controls for paste artifacts)
  <country>  - plate pasted on a background photo from that country
  bg_only    - that background photo without the plate
plus a binary mask of where the plate is (used later for attention analysis).

Usage:
  # food photos from Food-101 on Hugging Face (streams, no full download)
  python make_pairs.py --from_hf --bg_dir backgrounds --out pairs --n_per_dish 5

  # or your own photos: food_dir/<dish_name>/*.jpg (dish names as in common.py)
  python make_pairs.py --food_dir my_food --bg_dir backgrounds --out pairs

backgrounds/ must look like: backgrounds/japan/*.jpg, backgrounds/italy/*.jpg, ...
"""

import argparse
import json
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from common import DISH_COUNTRY, COUNTRIES

BG_W, BG_H = 1024, 768
IMG_EXT = {".jpg", ".jpeg", ".png", ".webp"}


# ---------------------------------------------------------------- image utils
def cover_resize(img, W=BG_W, H=BG_H):
    w, h = img.size
    s = max(W / w, H / h)
    img = img.resize((round(w * s), round(h * s)), Image.LANCZOS)
    w, h = img.size
    l, t = (w - W) // 2, (h - H) // 2
    return img.crop((l, t, l + W, t + H))


def center_square(img, size=512):
    w, h = img.size
    s = min(w, h)
    l, t = (w - s) // 2, (h - s) // 2
    return img.crop((l, t, l + s, t + s)).resize((size, size), Image.LANCZOS)


def ellipse_alpha(size, feather=6):
    """Crude 'plate cut-out': an ellipse. Food-101 photos are mostly plate-centred."""
    hard = Image.new("L", (size, size), 0)
    pad = int(0.04 * size)
    ImageDraw.Draw(hard).ellipse((pad, pad, size - pad, size - pad), fill=255)
    soft = hard.filter(ImageFilter.GaussianBlur(feather)) if feather > 0 else hard
    return hard, soft


def rembg_alpha(img):
    """Better cut-out using rembg (pip install rembg). Downloads a small model once."""
    from rembg import remove
    a = remove(img).split()[-1]
    hard = a.point(lambda v: 255 if v > 127 else 0)
    return hard, a


def neutral_bg(rng):
    kind = rng.choice(["gray", "beige", "wood"])
    if kind == "gray":
        return Image.new("RGB", (BG_W, BG_H), (128, 128, 128)), kind
    if kind == "beige":
        return Image.new("RGB", (BG_W, BG_H), (222, 206, 180)), kind
    nrng = np.random.default_rng(rng.randint(0, 10**9))
    y = np.arange(BG_H)[:, None]
    x = np.arange(BG_W)[None, :]
    base = 0.5 + 0.5 * np.sin(y / 9.0 + 2 * np.sin(x / 150.0))
    v = np.clip(base * 0.3 + 0.55 + nrng.normal(0, 0.05, (BG_H, BG_W)), 0, 1)
    rgb = np.stack([v * 190, v * 130, v * 80], -1).astype(np.uint8)
    return Image.fromarray(rgb), kind


def composite(bg, food, alpha, plate_frac):
    """Paste food (with alpha) bottom-centre, like a plate on a table.
    Same size and position for every background, so only the background changes."""
    size = min(int(plate_frac * BG_W), int(0.9 * BG_H))
    f = food.resize((size, size), Image.LANCZOS)
    hard, soft = (a.resize((size, size)) for a in alpha)
    x, y = (BG_W - size) // 2, BG_H - size - int(0.04 * BG_H)
    out = bg.copy()
    out.paste(f, (x, y), soft)
    mask = Image.new("L", (BG_W, BG_H), 0)
    mask.paste(hard, (x, y))
    return out, mask


# ---------------------------------------------------------------- data loading
def foods_from_dir(food_dir, n_per_dish, rng):
    out = {}
    for dish in DISH_COUNTRY:
        d = Path(food_dir) / dish
        if not d.is_dir():
            continue
        files = sorted(p for p in d.iterdir() if p.suffix.lower() in IMG_EXT)
        rng.shuffle(files)
        out[dish] = [Image.open(p).convert("RGB") for p in files[:n_per_dish]]
    return out


def foods_from_hf(n_per_dish, split="validation"):
    from datasets import load_dataset
    ds = None
    for name in ("ethz/food101", "food101"):
        try:
            ds = load_dataset(name, split=split, streaming=True)
            break
        except Exception as e:  # noqa: BLE001
            print(f"could not load {name}: {e}")
    if ds is None:
        raise SystemExit("Food-101 not reachable; use --food_dir instead.")
    names = ds.features["label"].names
    want = {i: n for i, n in enumerate(names) if n in DISH_COUNTRY}
    out = {n: [] for n in want.values()}
    for ex in ds:
        name = want.get(ex["label"])
        if name and len(out[name]) < n_per_dish:
            out[name].append(ex["image"].convert("RGB"))
            if all(len(v) >= n_per_dish for v in out.values()):
                break
    return out


def load_backgrounds(bg_dir):
    bgs = {}
    for c in COUNTRIES:
        d = Path(bg_dir) / c
        if d.is_dir():
            files = sorted(p for p in d.iterdir() if p.suffix.lower() in IMG_EXT)
            if files:
                bgs[c] = files
    missing = [c for c in COUNTRIES if c not in bgs]
    if missing:
        print(f"[warn] no backgrounds for: {missing} (those countries are skipped)")
    if len(bgs) < 2:
        raise SystemExit("Need backgrounds for at least 2 countries.")
    return bgs


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--from_hf", action="store_true")
    src.add_argument("--food_dir")
    ap.add_argument("--bg_dir", required=True)
    ap.add_argument("--out", default="pairs")
    ap.add_argument("--n_per_dish", type=int, default=5)
    ap.add_argument("--plate_frac", type=float, default=0.45,
                    help="plate width as a fraction of image width (smaller = harder)")
    ap.add_argument("--use_rembg", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    out = Path(args.out)
    (out / "img").mkdir(parents=True, exist_ok=True)

    foods = (foods_from_hf(args.n_per_dish) if args.from_hf
             else foods_from_dir(args.food_dir, args.n_per_dish, rng))
    bgs = load_backgrounds(args.bg_dir)
    print("dishes:", {k: len(v) for k, v in foods.items()})

    records = []
    for dish, imgs in foods.items():
        fc = DISH_COUNTRY[dish]
        for i, raw in enumerate(imgs):
            fid = f"{dish}_{i}"
            food = center_square(raw)
            alpha = rembg_alpha(food) if args.use_rembg else ellipse_alpha(food.size[0])

            orig_p = out / "img" / f"{fid}__original.jpg"
            food.save(orig_p, quality=95)

            nbg, nkind = neutral_bg(rng)
            neu, neu_mask = composite(nbg, food, alpha, args.plate_frac)
            neu_p = out / "img" / f"{fid}__neutral.jpg"
            neu_mask_p = out / "img" / f"{fid}__neutral_mask.png"
            neu.save(neu_p, quality=95)
            neu_mask.save(neu_mask_p)

            for country, files in bgs.items():
                bg = cover_resize(Image.open(rng.choice(files)).convert("RGB"))
                comp, mask = composite(bg, food, alpha, args.plate_frac)
                stem = f"{fid}__{country}"
                comp_p = out / "img" / f"{stem}.jpg"
                mask_p = out / "img" / f"{stem}_mask.png"
                bgo_p = out / "img" / f"{stem}_bgonly.jpg"
                comp.save(comp_p, quality=95)
                mask.save(mask_p)
                bg.save(bgo_p, quality=95)
                records.append({
                    "id": stem, "food_id": fid, "dish": dish,
                    "food_country": fc, "bg_country": country,
                    "condition": "matched" if country == fc else "mismatched",
                    "image": str(comp_p), "mask": str(mask_p), "bg_only": str(bgo_p),
                    "neutral": str(neu_p), "neutral_mask": str(neu_mask_p),
                    "neutral_kind": nkind, "original": str(orig_p),
                })

    meta = out / "meta.jsonl"
    with open(meta, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    n_mis = sum(r["condition"] == "mismatched" for r in records)
    print(f"wrote {len(records)} paired records ({n_mis} mismatched) to {meta}")


if __name__ == "__main__":
    main()

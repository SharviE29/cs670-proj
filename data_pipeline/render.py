"""Step 4: deterministic layer compositor.  [FIXED semantics]

    image, masks = renderer.render(bg_id, left_crop_id, right_crop_id, mode="paste")

Guarantees (tested in tests/test_pixel_preservation.py and audit.py):
  * Output is 448x448 RGB uint8; identical inputs -> byte-identical output.
  * A plate is scaled to fit its slot footprint (aspect preserved) and centred in it.
  * The feathered alpha ramps INWARD from the hard mask edge, so a plate's pixels never
    leave its hard mask, and the hard mask never leaves its slot box. Hence swapping one
    dish can only change pixels inside that slot box, and in "paste" mode swapping the
    background cannot change any pixel where a plate is fully opaque (alpha == 255).
  * Same feather width and the same scaling rule for every image.

Masks returned per slot ("left"/"right"):
  <slot>_slot     the footprint box (the slot region; superset of everything below)
  <slot>          plate support, alpha > 0 (the "pixel mask of the slot")
  <slot>_core     fully opaque plate pixels, alpha == 255

"harmonized" mode applies a Reinhard-style colour transfer of each plate toward the
background ring around its slot. It depends on the background, so background swaps change
plate pixels in that mode by design. Dish swaps are still slot-local.

CLI:
  python -m data_pipeline.render --split train            # render all groups of a split
  python -m data_pipeline.render --demo out.png           # render first group side by side
"""

from __future__ import annotations

import argparse
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import binary_dilation, distance_transform_edt

from .utils import Config, get_logger, load_config, read_jsonl

RESAMPLE = {"lanczos": Image.LANCZOS, "bicubic": Image.BICUBIC, "bilinear": Image.BILINEAR}
SLOTS = ("left", "right")


# ---------------------------------------------------------------- pure functions
def cover_resize(img: Image.Image, size: int, resample=Image.LANCZOS) -> np.ndarray:
    """Scale to cover size x size, then centre-crop (as in the pilot's cover_resize)."""
    img = img.convert("RGB")
    w, h = img.size
    s = max(size / w, size / h)
    nw, nh = max(size, round(w * s)), max(size, round(h * s))
    img = img.resize((nw, nh), resample)
    l, t = (nw - size) // 2, (nh - size) // 2
    return np.asarray(img.crop((l, t, l + size, t + size)), dtype=np.uint8).copy()


def fit_plate(rgba: Image.Image, box: tuple[int, int, int, int], feather_px: int,
              resample=Image.LANCZOS) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
    """Scale an RGBA cutout to fit `box` (aspect preserved) and build its feathered alpha.

    Returns (rgb uint8 [h,w,3], alpha uint8 [h,w], (x0, y0) top-left in image coords)."""
    x0, y0, x1, y1 = box
    bw, bh = x1 - x0, y1 - y0
    w, h = rgba.size
    s = min(bw / w, bh / h)
    nw, nh = min(bw, max(1, round(w * s))), min(bh, max(1, round(h * s)))
    rgb = np.asarray(rgba.convert("RGB").resize((nw, nh), resample), dtype=np.uint8)
    a_src = rgba.getchannel("A").resize((nw, nh), resample)
    hard = np.asarray(a_src) >= 128
    # distance (px) from each inside pixel to the nearest outside pixel; the patch border
    # counts as outside so the ramp also applies where the mask touches the crop edge.
    dist = distance_transform_edt(np.pad(hard, 1))[1:-1, 1:-1]
    alpha = np.clip(dist / (feather_px + 1), 0.0, 1.0)
    alpha = np.round(alpha * 255).astype(np.uint8)
    alpha[~hard] = 0
    px, py = x0 + (bw - nw) // 2, y0 + (bh - nh) // 2
    return rgb, alpha, (px, py)


def harmonize(rgb: np.ndarray, alpha: np.ndarray, bg: np.ndarray, pos: tuple[int, int],
              strength: float, ring_px: int) -> np.ndarray:
    """Reinhard mean/std transfer (per RGB channel) of the plate toward its background ring."""
    h, w = alpha.shape
    px, py = pos
    H, W = bg.shape[:2]
    support = np.zeros((H, W), bool)
    support[py:py + h, px:px + w] = alpha > 0
    ring = binary_dilation(support, iterations=ring_px) & ~support
    fg = rgb[alpha > 0].astype(np.float64)
    bgp = bg[ring].astype(np.float64)
    if len(fg) == 0 or len(bgp) == 0:
        return rgb
    mu_f, sd_f = fg.mean(0), fg.std(0) + 1e-6
    mu_b, sd_b = bgp.mean(0), bgp.std(0) + 1e-6
    mu_t = mu_f + strength * (mu_b - mu_f)
    sd_t = sd_f + strength * (sd_b - sd_f)
    out = (rgb.astype(np.float64) - mu_f) / sd_f * sd_t + mu_t
    return np.clip(np.round(out), 0, 255).astype(np.uint8)


def paste(canvas: np.ndarray, rgb: np.ndarray, alpha: np.ndarray, pos: tuple[int, int]) -> None:
    """In-place integer alpha blend: out = (fg*a + bg*(255-a) + 127) // 255."""
    h, w = alpha.shape
    px, py = pos
    region = canvas[py:py + h, px:px + w].astype(np.uint32)
    a = alpha.astype(np.uint32)[..., None]
    blended = (rgb.astype(np.uint32) * a + region * (255 - a) + 127) // 255
    canvas[py:py + h, px:px + w] = blended.astype(np.uint8)


def box_mask(box, size) -> np.ndarray:
    m = np.zeros((size, size), bool)
    x0, y0, x1, y1 = box
    m[y0:y1, x0:x1] = True
    return m


def validate_slots(slots: dict, size: int) -> None:
    for name, (x0, y0, x1, y1) in slots.items():
        assert 0 <= x0 < x1 <= size and 0 <= y0 < y1 <= size, f"slot {name} out of bounds"
    (a, b) = (slots["left"], slots["right"])
    overlap = not (a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1])
    assert not overlap, "slot boxes must not overlap"


# ---------------------------------------------------------------- renderer
class Renderer:
    def __init__(self, cfg: Config, crop_paths: dict[str, Path] | None = None,
                 bg_paths: dict[str, Path] | None = None):
        r = cfg["render"]
        self.cfg = cfg
        self.size = int(r["size"])
        self.slots = {k: tuple(v) for k, v in r["slots"].items()}
        validate_slots(self.slots, self.size)
        self.feather = int(r["feather_px"])
        self.bg_resample = RESAMPLE[r["bg_resample"]]
        self.plate_resample = RESAMPLE[r["plate_resample"]]
        self.harm = r["harmonize"]
        self.crop_paths = crop_paths if crop_paths is not None else self._crop_index(cfg)
        self.bg_paths = bg_paths if bg_paths is not None else self._bg_index(cfg)
        self._bg = lru_cache(maxsize=256)(self._load_bg)
        self._plate = lru_cache(maxsize=1024)(self._load_plate)
        self.slot_masks = {k: box_mask(v, self.size) for k, v in self.slots.items()}

    @staticmethod
    def _crop_index(cfg: Config) -> dict[str, Path]:
        p = cfg.path("crops_index")
        if not p.exists():
            return {}
        root = cfg.path("crops_dir")
        return {r["crop_id"]: root / r["rgba_path"] for r in read_jsonl(p)}

    @staticmethod
    def _bg_index(cfg: Config) -> dict[str, Path]:
        p = cfg.path("backgrounds_index")
        if not p.exists():
            return {}
        root = cfg.path("backgrounds_dir")
        return {r["bg_id"]: root / r["path"] for r in read_jsonl(p)}

    def _load_bg(self, bg_id: str) -> np.ndarray:
        arr = cover_resize(Image.open(self.bg_paths[bg_id]), self.size, self.bg_resample)
        arr.setflags(write=False)
        return arr

    def _load_plate(self, crop_id: str, slot: str):
        rgba = Image.open(self.crop_paths[crop_id]).convert("RGBA")
        rgb, alpha, pos = fit_plate(rgba, self.slots[slot], self.feather, self.plate_resample)
        rgb.setflags(write=False)
        alpha.setflags(write=False)
        return rgb, alpha, pos

    def background(self, bg_id: str) -> np.ndarray:
        return self._bg(bg_id)

    def _plate_pixels(self, crop_id: str, slot: str, bg: np.ndarray, mode: str):
        rgb, alpha, pos = self._plate(crop_id, slot)
        if mode == "harmonized":
            rgb = harmonize(rgb, alpha, bg, pos, float(self.harm["strength"]), int(self.harm["ring_px"]))
        elif mode != "paste":
            raise ValueError(f"unknown render mode {mode!r}")
        return rgb, alpha, pos

    def _plate_masks(self, slot: str, alpha: np.ndarray, pos) -> dict[str, np.ndarray]:
        h, w = alpha.shape
        px, py = pos
        sup = np.zeros((self.size, self.size), bool)
        core = np.zeros((self.size, self.size), bool)
        sup[py:py + h, px:px + w] = alpha > 0
        core[py:py + h, px:px + w] = alpha == 255
        return {slot: sup, f"{slot}_core": core, f"{slot}_slot": self.slot_masks[slot]}

    def render(self, bg_id: str, left_crop_id: str | None, right_crop_id: str | None,
               mode: str = "paste") -> tuple[np.ndarray, dict[str, np.ndarray]]:
        """Composite background + left plate + right plate. A crop id of None leaves the slot empty."""
        bg = self.background(bg_id)
        img = bg.copy()
        masks: dict[str, np.ndarray] = {}
        for slot, cid in (("left", left_crop_id), ("right", right_crop_id)):
            if cid is None:
                masks.update({slot: np.zeros_like(self.slot_masks[slot]),
                              f"{slot}_core": np.zeros_like(self.slot_masks[slot]),
                              f"{slot}_slot": self.slot_masks[slot]})
                continue
            rgb, alpha, pos = self._plate_pixels(cid, slot, bg, mode)
            paste(img, rgb, alpha, pos)
            masks.update(self._plate_masks(slot, alpha, pos))
        return img, masks

    def edit_slot(self, img: np.ndarray, bg_id: str, slot: str, new_crop_id: str | None,
                  mode: str = "paste") -> np.ndarray:
        """Incremental edit of an existing render: restore the slot box from the background,
        then paste `new_crop_id` there. Touches no pixel outside the slot box."""
        bg = self.background(bg_id)
        out = img.copy()
        x0, y0, x1, y1 = self.slots[slot]
        out[y0:y1, x0:x1] = bg[y0:y1, x0:x1]
        if new_crop_id is not None:
            rgb, alpha, pos = self._plate_pixels(new_crop_id, slot, bg, mode)
            paste(out, rgb, alpha, pos)
        return out


# ---------------------------------------------------------------- group helpers
LAYERS = {  # key -> (left, right, background) as recipe field names
    "x": ("t", "o", "s"), "t": ("t_prime", "o", "s"), "o": ("t", "o_prime", "s"),
    "b": ("t", "o", "s_prime"), "to": ("t_prime", "o_prime", "s"),
    "tb": ("t_prime", "o", "s_prime"), "ob": ("t", "o_prime", "s_prime"),
}


def group_layers(rec: dict, key: str) -> tuple[str, str, str]:
    l, r, b = LAYERS[key]
    return rec[l], rec[r], rec[b]


def render_group(renderer: Renderer, rec: dict, mode: str | None = None):
    """Render all 7 images of a group. Returns {key: (image, masks)}."""
    mode = mode or rec.get("render_mode", "paste")
    out = {}
    for key in LAYERS:
        left, right, bg = group_layers(rec, key)
        out[key] = renderer.render(bg, left, right, mode)
    return out


def image_relpath(split: str, group_id: str, key: str, mode: str = "paste") -> str:
    suffix = "" if mode == "paste" else f"_{mode}"
    return f"{split}/{group_id}/{key}{suffix}.png"


def save_png(arr: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr, "RGB").save(path, format="PNG", optimize=False, compress_level=6)


def side_by_side(images: dict[str, np.ndarray], labels: bool = True, pad: int = 6) -> Image.Image:
    from PIL import ImageDraw

    keys = list(images)
    s = next(iter(images.values())).shape[0]
    top = 22 if labels else 0
    sheet = Image.new("RGB", (len(keys) * (s + pad) + pad, s + top + pad), "white")
    d = ImageDraw.Draw(sheet)
    for i, k in enumerate(keys):
        x = pad + i * (s + pad)
        sheet.paste(Image.fromarray(images[k]), (x, top))
        if labels:
            d.text((x + 4, 4), k, fill="black")
    return sheet


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    ap.add_argument("--split", help="render every group in groups/<split>.jsonl")
    ap.add_argument("--controls", action="store_true", help="also render groups/controls.jsonl")
    ap.add_argument("--demo", help="write the first group of --demo_split side by side to this path")
    ap.add_argument("--demo_split", default="train")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num_shards", type=int, default=1, help="for Slurm array jobs")
    args = ap.parse_args()
    cfg = load_config(args.config)
    log = get_logger("render", cfg)
    renderer = Renderer(cfg)

    if args.demo:
        rec = read_jsonl(cfg.path("groups_dir") / f"{args.demo_split}.jsonl")[0]
        imgs = {k: v[0] for k, v in render_group(renderer, rec).items()}
        side_by_side(imgs).save(args.demo)
        log.info("wrote %s (group %s)", args.demo, rec["group_id"])
        return

    if args.split:
        out_root = cfg.path("renders_dir")
        recs = read_jsonl(cfg.path("groups_dir") / f"{args.split}.jsonl")
        recs = recs[args.shard::args.num_shards]
        for i, rec in enumerate(recs):
            mode = rec.get("render_mode", "paste")
            for key, (img, _) in render_group(renderer, rec, mode).items():
                save_png(img, out_root / image_relpath(rec["split"], rec["group_id"], key, mode))
            if (i + 1) % 100 == 0:
                log.info("%s: %d/%d groups", args.split, i + 1, len(recs))
        log.info("rendered %d groups of %s (shard %d/%d)", len(recs), args.split, args.shard, args.num_shards)


if __name__ == "__main__":
    main()

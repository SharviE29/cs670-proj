"""Test fixtures.

Default: a small SYNTHETIC dataset (random textured "plates" with elliptical masks, random
smooth-noise backgrounds) in the same on-disk format as the real one, so the logic is
tested without GPUs or downloads.

Full dataset:  CIRA_TEST_FULL=1 [CIRA_CONFIG=...] [CIRA_DATA_ROOT=...] pytest tests/
runs the same tests over every group of the real, generated dataset.
"""

from __future__ import annotations

import os
import random

import numpy as np
import pytest
from PIL import Image, ImageDraw

from data_pipeline.classes import CLASSES, CLASS_REGION, REGIONS
from data_pipeline.utils import load_config, write_jsonl

FULL = os.environ.get("CIRA_TEST_FULL") == "1"

SMALL = {
    "splits": {"group_counts": {"train": 40, "val": 10, "test_id": 20, "test_unseen": 15},
               "heldout_class_region_pairs": 10, "heldout_dish_swap_pairs": 10,
               "heldout_region_swap_pairs": 5},
    "controls": {"identity_repaste": 5, "wrong_entity_swap": 5, "harmonized_groups": 3, "blank_groups": 5},
    "human_audit": {"crops_per_class": 1, "backgrounds_per_region": 1},
}


def smooth_noise(rng: np.random.Generator, w: int, h: int, cells: int = 6) -> Image.Image:
    small = rng.integers(0, 256, (cells, cells, 3), dtype=np.uint8)
    img = Image.fromarray(small).resize((w, h), Image.BICUBIC)
    fine = rng.normal(0, 8, (h, w, 3))
    return Image.fromarray(np.clip(np.asarray(img) + fine, 0, 255).astype(np.uint8))


def make_synthetic(root, n_crops=8, n_bgs=12, seed=0):
    rng = np.random.default_rng(seed)
    prng = random.Random(seed)
    crops, bgs = [], []
    for cls in CLASSES:
        for i in range(n_crops):
            w, h = prng.randint(120, 260), prng.randint(120, 260)
            img = smooth_noise(rng, w, h, cells=prng.randint(3, 9))
            mask = Image.new("L", (w, h), 0)
            pad = prng.randint(0, 10)
            ImageDraw.Draw(mask).ellipse((pad, pad, w - 1 - pad, h - 1 - pad), fill=255)
            rgba = img.convert("RGBA")
            rgba.putalpha(mask)
            cid = f"{cls}/{100000 + i}"
            p = root / "crops" / f"{cid}.png"
            p.parent.mkdir(parents=True, exist_ok=True)
            rgba.save(p)
            mask.save(root / "crops" / f"{cid}_mask.png")
            crops.append({"crop_id": cid, "class": cls, "source_id": cid, "rgba_path": f"{cid}.png",
                          "mask_path": f"{cid}_mask.png"})
    for reg in REGIONS:
        for i in range(n_bgs):
            bid = f"wm_{reg}_{i:03d}"
            w, h = prng.choice([(640, 480), (500, 500), (480, 640), (800, 450)])
            p = root / "backgrounds" / "img" / f"{bid}.png"
            p.parent.mkdir(parents=True, exist_ok=True)
            smooth_noise(rng, w, h, cells=prng.randint(4, 10)).save(p)
            bgs.append({"bg_id": bid, "region": reg, "path": f"img/{bid}.png"})
    write_jsonl(root / "crops" / "index.jsonl", crops)
    write_jsonl(root / "backgrounds" / "index.jsonl", bgs)
    assert all(CLASS_REGION[c["class"]] for c in crops)


@pytest.fixture(scope="session")
def cfg(tmp_path_factory):
    if FULL:
        return load_config()
    root = tmp_path_factory.mktemp("cira_synth")
    make_synthetic(root)
    os.environ["CIRA_DATA_ROOT"] = str(root)
    c = load_config(overrides=SMALL)
    from data_pipeline import groups, splits

    splits.assign(c)
    groups.generate_all(c)
    return c


@pytest.fixture(scope="session")
def all_groups(cfg):
    from data_pipeline.splits import load_groups

    return load_groups(cfg)


@pytest.fixture(scope="session")
def renderer(cfg):
    from data_pipeline.render import Renderer

    return Renderer(cfg)

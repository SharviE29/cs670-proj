"""Pixel preservation, commutativity and determinism over EVERY group (synthetic by default,
full dataset with CIRA_TEST_FULL=1). Mutation tests make sure the checks can fail."""

from __future__ import annotations

import io

import numpy as np
import pytest
from PIL import Image

from data_pipeline.audit import changed_layers, check_commutativity, check_group_pixels, check_pair
from data_pipeline.render import LAYERS, Renderer, fit_plate, render_group, validate_slots


def _all(all_groups):
    return [r for recs in all_groups.values() for r in recs]


def test_pixel_preservation_all_groups(renderer, all_groups):
    recs = _all(all_groups)
    assert recs
    errs = []
    for rec in recs:
        errs += check_group_pixels(renderer, rec)
    assert not errs, f"{len(errs)} violations, e.g. {errs[:5]}"


def test_single_layer_pairs_are_tight(renderer, all_groups):
    """One-layer swaps: every changed pixel lies in that slot's box; bg swaps leave opaque
    plate pixels untouched AND do change pixels somewhere (the swap really happened)."""
    rec = _all(all_groups)[0]
    imgs = render_group(renderer, rec)
    one_layer = [(a, b) for a in LAYERS for b in LAYERS if a < b and len(changed_layers(rec, a, b)) == 1]
    assert len(one_layer) == 9  # x-t, x-o, x-b, t-to, o-to, t-tb, b-tb, o-ob, b-ob
    for a, b in one_layer:
        (ia, ma), (ib, mb) = imgs[a], imgs[b]
        diff = np.any(ia != ib, axis=2)
        assert diff.any(), f"{a}->{b}: nothing changed"
        (layer,) = changed_layers(rec, a, b)
        if layer == "background":
            core = ma["left_core"] | ma["right_core"]
            assert not (diff & core).any()
            assert np.array_equal(ma["left_core"], mb["left_core"])
        else:
            assert not (diff & ~ma[f"{layer}_slot"]).any()


def test_commutativity_all_groups(renderer, all_groups):
    errs = []
    for rec in _all(all_groups):
        errs += check_commutativity(renderer, rec)
    assert not errs, errs[:5]


def test_determinism_bytes(cfg, all_groups):
    rec = _all(all_groups)[0]
    r1, r2 = Renderer(cfg), Renderer(cfg)  # fresh caches
    for key in LAYERS:
        a = render_group(r1, rec)[key][0]
        b = render_group(r2, rec)[key][0]
        assert np.array_equal(a, b)
        bufs = []
        for arr in (a, b):
            buf = io.BytesIO()
            Image.fromarray(arr).save(buf, format="PNG", optimize=False, compress_level=6)
            bufs.append(buf.getvalue())
        assert bufs[0] == bufs[1]


def test_output_shape(cfg, renderer, all_groups):
    rec = _all(all_groups)[0]
    img, masks = renderer.render(rec["s"], rec["t"], rec["o"])
    s = cfg["render"]["size"]
    assert img.shape == (s, s, 3) and img.dtype == np.uint8
    for slot in ("left", "right"):
        assert masks[slot].any()
        assert not (masks[slot] & ~masks[f"{slot}_slot"]).any(), "plate leaves its slot box"
        assert not (masks[f"{slot}_core"] & ~masks[slot]).any()


def test_plate_fit_and_centred():
    rgba = Image.new("RGBA", (300, 150), (200, 10, 10, 255))
    box = (24, 240, 204, 420)
    rgb, alpha, (px, py) = fit_plate(rgba, box, feather_px=3)
    h, w = alpha.shape
    assert w == 180 and h == 90                      # aspect preserved, fits the footprint
    assert px == 24 and py == 240 + (180 - 90) // 2  # centred
    assert alpha[h // 2, 0] < 255 and alpha[h // 2, w // 2] == 255  # inward feather


def test_feather_never_exceeds_hard_mask():
    m = Image.new("L", (100, 100), 0)
    from PIL import ImageDraw
    ImageDraw.Draw(m).ellipse((10, 10, 89, 89), fill=255)
    rgba = Image.new("RGBA", (100, 100), (0, 200, 0, 0))
    rgba.putalpha(m)
    _, alpha, _ = fit_plate(rgba, (0, 0, 100, 100), feather_px=4)
    hard = np.asarray(m) >= 128
    assert not ((alpha > 0) & ~hard).any()
    assert (alpha == 255).any() and ((alpha > 0) & (alpha < 255)).any()


def test_slot_overlap_rejected():
    with pytest.raises(AssertionError):
        validate_slots({"left": [0, 0, 200, 200], "right": [150, 0, 350, 200]}, 448)


def test_harmonized_dish_edits_stay_local(renderer, all_groups):
    rec = {**_all(all_groups)[0], "render_mode": "harmonized"}
    assert not check_group_pixels(renderer, rec)
    assert not check_commutativity(renderer, rec)


# ---------------------------------------------------------------- mutation tests
def test_detects_pixel_leak(renderer, all_groups):
    rec = _all(all_groups)[0]
    imgs = render_group(renderer, rec)
    (ix, mx), (it, mt) = imgs["x"], imgs["t"]
    bad = it.copy()
    bad[5, 5] ^= 1  # one pixel in the top-left corner, far from the left slot
    assert check_pair(ix, mx, bad, mt, {"left"}, "paste")
    assert not check_pair(ix, mx, it, mt, {"left"}, "paste")


def test_detects_background_bleeding_into_plate(renderer, all_groups):
    rec = _all(all_groups)[0]
    imgs = render_group(renderer, rec)
    (ix, mx), (ib, mb) = imgs["x"], imgs["b"]
    ys, xs = np.where(mx["right_core"])
    bad = ib.copy()
    bad[ys[0], xs[0]] ^= 1
    assert check_pair(ix, mx, bad, mb, {"background"}, "paste")


def test_detects_noncommuting_editor(cfg, all_groups):
    class Leaky(Renderer):
        def edit_slot(self, img, bg_id, slot, new_crop_id, mode="paste"):
            out = super().edit_slot(img, bg_id, slot, new_crop_id, mode)
            out[0, 0] = (out[0, 0].astype(np.int32) + 1) % 256  # touches a pixel outside the slot
            return out

    rec = _all(all_groups)[0]
    assert check_commutativity(Leaky(cfg), rec)

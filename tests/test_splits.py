"""Split tests: no source photo / background shared across splits, held-out classes only in
test_unseen, held-out pairings absent from train, near-duplicates never cross splits."""

from __future__ import annotations

import copy
import json

import numpy as np
import pytest
from PIL import Image

from data_pipeline.classes import HELDOUT_CLASSES
from data_pipeline.questions import crop_class
from data_pipeline.splits import group_pairings, near_dup_clusters, phash, verify


def _load(cfg, key):
    with open(cfg.path(key)) as f:
        return json.load(f)


def test_verify_passes(cfg, all_groups):
    stats = verify(cfg, all_groups)
    for split, n in cfg["splits"]["group_counts"].items():
        assert stats["groups"][split] == n


def test_unseen_classes(all_groups):
    for split, recs in all_groups.items():
        for rec in recs:
            dishes = {crop_class(rec[k]) for k in ("t", "t_prime", "o", "o_prime")}
            if split == "test_unseen":
                assert dishes <= set(HELDOUT_CLASSES)
            else:
                assert not dishes & set(HELDOUT_CLASSES)
                assert not (set(rec["options"]["q_L"]) | set(rec["options"]["q_R"])) & set(HELDOUT_CLASSES)


def test_heldout_pairings_not_in_train(cfg, all_groups):
    held = {k: {tuple(p) for p in v} for k, v in _load(cfg, "heldout_pairings").items()}
    assert all(held.values())
    for rec in all_groups["train"]:
        gp = group_pairings(rec)
        assert all(not gp[k] & held[k] for k in held)


def test_detects_cross_split_leak(cfg, all_groups):
    bad = copy.deepcopy(all_groups)
    bad["val"][0]["t"] = bad["train"][0]["t"]
    with pytest.raises(AssertionError):
        verify(cfg, bad)
    bad = copy.deepcopy(all_groups)
    bad["test_id"][0]["s_prime"] = bad["train"][0]["s"]
    with pytest.raises(AssertionError):
        verify(cfg, bad)


def test_detects_near_duplicate_across_splits(cfg, all_groups):
    a = copy.deepcopy(_load(cfg, "splits_file"))
    ids = sorted(a["backgrounds"])
    x = next(i for i in ids if a["backgrounds"][i] == "train")
    y = next(i for i in ids if a["backgrounds"][i] == "val")
    a["phash"]["backgrounds"][y] = a["phash"]["backgrounds"][x]
    with pytest.raises(AssertionError, match="near-duplicate"):
        verify(cfg, all_groups, assignment=a)


def test_phash_near_duplicates():
    rng = np.random.default_rng(1)
    base = Image.fromarray(rng.integers(0, 256, (8, 8, 3), dtype=np.uint8)).resize((400, 300), Image.BICUBIC)
    shifted = base.resize((380, 285), Image.BICUBIC).convert("RGB")       # rescaled copy
    other = Image.fromarray(rng.integers(0, 256, (8, 8, 3), dtype=np.uint8)).resize((400, 300), Image.BICUBIC)
    h = [phash(base), phash(shifted), phash(other)]
    clusters = near_dup_clusters(["a", "b", "c"], h, max_ham=6)
    assert ["a", "b"] in clusters and ["c"] in clusters


def test_assign_deterministic(cfg):
    from data_pipeline import splits

    from conftest import FULL

    if FULL:
        pytest.skip("assign is not re-run on the real dataset (it would rewrite the split file)")
    before = _load(cfg, "splits_file")
    again = splits.assign(cfg)
    assert json.loads(json.dumps(again, sort_keys=True)) == json.loads(json.dumps(before, sort_keys=True))

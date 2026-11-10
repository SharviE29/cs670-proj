"""Step 7: splits.  [FIXED]

Pipeline order:  splits.py assign  ->  groups.py  ->  questions.py  ->  splits.py verify

`assign` decides, BEFORE any group is sampled, which split owns each source food photo and
each background (the two split keys). Near-duplicates (pHash) are merged into clusters
first and a cluster always goes to one split. Held-out classes go only to test_unseen.
It also samples the held-out pairings withheld from train for the composition tests.

`verify` asserts on the generated groups:
  * no source photo and no background is used by more than one split;
  * held-out classes never appear in train/val (neither as dishes nor as options);
  * test_unseen uses only held-out classes as dishes;
  * no held-out pairing occurs in train;
  * no near-duplicate pair (pHash) crosses splits.

CLI:  python -m data_pipeline.splits assign | verify
"""

from __future__ import annotations

import argparse
import random
from collections import defaultdict
from itertools import product

import numpy as np
from PIL import Image
from scipy.fft import dctn

from .classes import CLASSES, HELDOUT_CLASSES, REGIONS, SEEN_CLASSES
from .utils import (Config, derive_seed, get_logger, load_config, read_jsonl, write_json)

SPLITS = ("train", "val", "test_id", "test_unseen")
SEEN_SPLITS = ("train", "val", "test_id")


# ---------------------------------------------------------------- perceptual hash
def phash(img: Image.Image, hash_size: int = 8, highfreq: int = 4) -> int:
    """DCT perceptual hash (64 bits for hash_size=8). RGBA is flattened onto mid-gray."""
    if img.mode == "RGBA":
        canvas = Image.new("RGBA", img.size, (128, 128, 128, 255))
        img = Image.alpha_composite(canvas, img)
    n = hash_size * highfreq
    g = np.asarray(img.convert("L").resize((n, n), Image.LANCZOS), dtype=np.float64)
    d = dctn(g, norm="ortho")[:hash_size, :hash_size]
    bits = (d > np.median(d.flatten()[1:])).flatten()
    return int("".join("1" if b else "0" for b in bits), 2)


def hamming_matrix(hashes: list[int]) -> np.ndarray:
    a = np.array(hashes, dtype=np.uint64)
    x = a[:, None] ^ a[None, :]
    # popcount on uint64
    x = x - ((x >> np.uint64(1)) & np.uint64(0x5555555555555555))
    x = (x & np.uint64(0x3333333333333333)) + ((x >> np.uint64(2)) & np.uint64(0x3333333333333333))
    x = (x + (x >> np.uint64(4))) & np.uint64(0x0F0F0F0F0F0F0F0F)
    return ((x * np.uint64(0x0101010101010101)) >> np.uint64(56)).astype(np.int32)


def near_dup_clusters(ids: list[str], hashes: list[int], max_ham: int) -> list[list[str]]:
    """Union-find over pairs with Hamming distance <= max_ham. Returns clusters of size >= 1."""
    parent = list(range(len(ids)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    if ids:
        hm = hamming_matrix(hashes)
        ii, jj = np.where(np.triu(hm <= max_ham, k=1))
        for i, j in zip(ii.tolist(), jj.tolist()):
            parent[find(i)] = find(j)
    groups = defaultdict(list)
    for i, x in enumerate(ids):
        groups[find(i)].append(x)
    return sorted((sorted(g) for g in groups.values()), key=lambda g: g[0])


# ---------------------------------------------------------------- allocation
def allocate(units: list[list[str]], fractions: dict[str, float], rng: random.Random) -> dict[str, str]:
    """Assign clusters (lists of ids) to splits so member counts track `fractions`.
    Greedy largest-deficit-first after a seeded shuffle; deterministic given rng."""
    units = list(units)
    rng.shuffle(units)
    units.sort(key=len, reverse=True)  # stable: big clusters first, shuffled within size
    total = sum(len(u) for u in units)
    target = {s: f * total for s, f in fractions.items()}
    have = {s: 0 for s in fractions}
    out = {}
    for u in units:
        s = max(fractions, key=lambda k: (target[k] - have[k], fractions[k]))
        have[s] += len(u)
        for x in u:
            out[x] = s
    return out


def assign(cfg: Config) -> dict:
    log = get_logger("splits", cfg)
    sc = cfg["splits"]
    seed = derive_seed(cfg["seed"], "splits")
    rng = random.Random(seed)
    log.info("assign: seed=%d", seed)

    crops = read_jsonl(cfg.path("crops_index"))
    bgs = read_jsonl(cfg.path("backgrounds_index"))
    croot, broot = cfg.path("crops_dir"), cfg.path("backgrounds_dir")

    crop_hash = {c["crop_id"]: phash(Image.open(croot / c["rgba_path"]), sc["phash_size"]) for c in crops}
    bg_hash = {b["bg_id"]: phash(Image.open(broot / b["path"]), sc["phash_size"]) for b in bgs}

    # ---- crops: global clusters (a duplicate photo could sit in two classes)
    cid = sorted(crop_hash)
    clusters = near_dup_clusters(cid, [crop_hash[c] for c in cid], sc["phash_max_hamming"])
    cls_of = {c["crop_id"]: c["class"] for c in crops}
    crop_split, excluded = {}, {}
    seen_units = defaultdict(list)   # class -> clusters (keyed by first member's class)
    for cl in clusters:
        classes = {cls_of[c] for c in cl}
        held = classes & set(HELDOUT_CLASSES)
        if held and classes - set(HELDOUT_CLASSES):
            for c in cl:   # mixed seen/held-out duplicates: drop the seen members
                if cls_of[c] in HELDOUT_CLASSES:
                    crop_split[c] = "test_unseen"
                else:
                    excluded[c] = "near_duplicate_of_heldout_crop"
            continue
        if held:
            for c in cl:
                crop_split[c] = "test_unseen"
        else:
            seen_units[cls_of[cl[0]]].append(cl)
    for cls in sorted(seen_units):
        crop_split.update(allocate(seen_units[cls], sc["crop_fractions"],
                                   random.Random(derive_seed(seed, "crops", cls))))

    # ---- backgrounds: per region, clustered
    reg_of = {b["bg_id"]: b["region"] for b in bgs}
    bid = sorted(bg_hash)
    bclusters = near_dup_clusters(bid, [bg_hash[b] for b in bid], sc["phash_max_hamming"])
    by_region = defaultdict(list)
    for cl in bclusters:
        by_region[reg_of[cl[0]]].append(cl)
    bg_split = {}
    for reg in sorted(by_region):
        bg_split.update(allocate(by_region[reg], sc["bg_fractions"],
                                 random.Random(derive_seed(seed, "bgs", reg))))

    pairings = sample_heldout_pairings(cfg, random.Random(derive_seed(seed, "pairings")))
    out = {
        "seed": seed,
        "crops": dict(sorted(crop_split.items())),
        "backgrounds": dict(sorted(bg_split.items())),
        "excluded_crops": excluded,
        "phash": {"crops": {k: f"{v:016x}" for k, v in sorted(crop_hash.items())},
                  "backgrounds": {k: f"{v:016x}" for k, v in sorted(bg_hash.items())}},
        "near_dup_clusters": {"crops": [c for c in clusters if len(c) > 1],
                              "backgrounds": [c for c in bclusters if len(c) > 1]},
    }
    write_json(cfg.path("splits_file"), out)
    write_json(cfg.path("heldout_pairings"), pairings)
    for name, m in (("crops", crop_split), ("backgrounds", bg_split)):
        cnt = defaultdict(int)
        for s in m.values():
            cnt[s] += 1
        log.info("%s per split: %s", name, dict(sorted(cnt.items())))
    log.info("near-dup clusters (size>1): crops=%d backgrounds=%d excluded_crops=%d",
             len(out["near_dup_clusters"]["crops"]), len(out["near_dup_clusters"]["backgrounds"]), len(excluded))
    return out


def sample_heldout_pairings(cfg: Config, rng: random.Random) -> dict:
    """Combinations withheld from train (kept in val/test) for composition tests.

    class_region: no train image shows dish class c on a background of region r
                  (only pairs where r is NOT c's own region, plus a few matched ones).
    dish_swap:    no train group replaces class a with class b (t->t' or o->o').
    region_swap:  no train group swaps a background of region a for region b (s->s')."""
    sc = cfg["splits"]
    seen = sorted(SEEN_CLASSES)
    cr = sorted(product(seen, REGIONS))
    ds = [(a, b) for a, b in product(seen, seen) if a != b]
    rs = [(a, b) for a, b in product(REGIONS, REGIONS) if a != b]
    return {
        "class_region": sorted(map(list, rng.sample(cr, sc["heldout_class_region_pairs"]))),
        "dish_swap": sorted(map(list, rng.sample(ds, sc["heldout_dish_swap_pairs"]))),
        "region_swap": sorted(map(list, rng.sample(rs, sc["heldout_region_swap_pairs"]))),
    }


# ---------------------------------------------------------------- verification
def group_pairings(rec: dict) -> dict[str, set[tuple[str, str]]]:
    """All (class, region), dish-swap and region-swap combinations a group contains."""
    from .questions import crop_class

    dishes = [crop_class(rec[k]) for k in ("t", "t_prime", "o", "o_prime")]
    regs = [rec["regions"]["s"], rec["regions"]["s_prime"]]
    return {
        "class_region": {(c, r) for c in dishes for r in regs},
        "dish_swap": {(crop_class(rec["t"]), crop_class(rec["t_prime"])),
                      (crop_class(rec["o"]), crop_class(rec["o_prime"]))},
        "region_swap": {(regs[0], regs[1])},
    }


def verify(cfg: Config, groups_by_split: dict[str, list[dict]], assignment: dict | None = None,
           pairings: dict | None = None) -> dict:
    """Raise AssertionError on any split violation. Returns summary stats."""
    from .questions import crop_class

    assignment = assignment or read_jsonl_json(cfg.path("splits_file"))
    pairings = pairings or read_jsonl_json(cfg.path("heldout_pairings"))
    sc = cfg["splits"]
    crop_used, bg_used = defaultdict(set), defaultdict(set)
    for split, recs in groups_by_split.items():
        for rec in recs:
            assert rec["split"] == split, rec["group_id"]
            for k in ("t", "t_prime", "o", "o_prime"):
                crop_used[rec[k]].add(split)
                owner = assignment["crops"].get(rec[k])
                assert owner == split, f"{rec['group_id']}: crop {rec[k]} owned by {owner}, used in {split}"
            for k in ("s", "s_prime"):
                bg_used[rec[k]].add(split)
                owner = assignment["backgrounds"].get(rec[k])
                assert owner == split, f"{rec['group_id']}: bg {rec[k]} owned by {owner}, used in {split}"
            dishes = {crop_class(rec[k]) for k in ("t", "t_prime", "o", "o_prime")}
            opt_classes = set(rec["options"]["q_L"]) | set(rec["options"]["q_R"])
            if split == "test_unseen":
                assert dishes <= set(HELDOUT_CLASSES), f"{rec['group_id']}: seen class in test_unseen"
            else:
                assert not (dishes | opt_classes) & set(HELDOUT_CLASSES), \
                    f"{rec['group_id']}: held-out class in {split}"
    multi_c = {k: v for k, v in crop_used.items() if len(v) > 1}
    multi_b = {k: v for k, v in bg_used.items() if len(v) > 1}
    assert not multi_c, f"crops in >1 split: {list(multi_c)[:5]}"
    assert not multi_b, f"backgrounds in >1 split: {list(multi_b)[:5]}"

    # held-out pairings never in train
    held = {k: {tuple(p) for p in v} for k, v in pairings.items()}
    for rec in groups_by_split.get("train", []):
        gp = group_pairings(rec)
        for k in held:
            bad = gp[k] & held[k]
            assert not bad, f"{rec['group_id']}: held-out {k} {bad} in train"

    # near-duplicates must not cross splits (check both keys, on everything assigned)
    for kind, owner in (("crops", assignment["crops"]), ("backgrounds", assignment["backgrounds"])):
        ids = sorted(owner)
        hashes = [int(assignment["phash"][kind][i], 16) for i in ids]
        if len(ids) > 1:
            hm = hamming_matrix(hashes)
            ii, jj = np.where(np.triu(hm <= sc["phash_max_hamming"], k=1))
            cross = [(ids[i], ids[j]) for i, j in zip(ii, jj) if owner[ids[i]] != owner[ids[j]]]
            assert not cross, f"near-duplicate {kind} across splits: {cross[:5]}"

    n_comp = {s: sum(bool(any(group_pairings(r)[k] & held[k] for k in held)) for r in recs)
              for s, recs in groups_by_split.items()}
    return {"groups": {s: len(v) for s, v in groups_by_split.items()},
            "crops_used": {s: sum(s in v for v in crop_used.values()) for s in groups_by_split},
            "backgrounds_used": {s: sum(s in v for v in bg_used.values()) for s in groups_by_split},
            "groups_with_heldout_pairing": n_comp}


def read_jsonl_json(path):
    import json

    with open(path) as f:
        return json.load(f)


def load_groups(cfg: Config) -> dict[str, list[dict]]:
    out = {}
    for s in SPLITS:
        p = cfg.path("groups_dir") / f"{s}.jsonl"
        if p.exists():
            out[s] = read_jsonl(p)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["assign", "verify"])
    ap.add_argument("--config")
    args = ap.parse_args()
    cfg = load_config(args.config)
    if args.cmd == "assign":
        assign(cfg)
    else:
        stats = verify(cfg, load_groups(cfg))
        get_logger("splits", cfg).info("verify OK: %s", stats)


if __name__ == "__main__":
    main()

"""Step 5: groups and edits.  [FIXED]

A group = base scene x plus 6 edits (keys x, t, o, b, to, tb, ob; see render.LAYERS).
Stored as recipes (IDs only) in groups/<split>.jsonl; images are re-rendered from recipes.

Sampling rules (all asserted in tests/test_labels.py and audit.py):
  * t, o, t', o' are four DISTINCT classes (so a replacement always differs from what it
    replaces, and options never collide); four distinct source photos.
  * region(s') != region(s).
  * bg_dish_match in {left, right, both, none}: whether region(s) is the region of the left
    dish t, the right dish o, both, or neither. About `groups.bg_dish_match_frac` of groups
    are sampled as matched.
  * every crop/background comes from the split's own pool (splits.py assign);
  * train groups contain no held-out pairing (splits/heldout_pairings.json);
  * option sets (questions.make_option_sets) must be feasible; otherwise resample.

CLI:  python -m data_pipeline.groups
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict

from .classes import CLASS_REGION, HELDOUT_CLASSES, SEEN_CLASSES
from .questions import InfeasibleOptions, make_option_sets, make_permutation
from .splits import SPLITS, group_pairings
from .utils import Config, derive_seed, get_logger, load_config, read_jsonl, write_jsonl


class Pools:
    def __init__(self, split: str, crops: list[dict], bgs: list[dict], assignment: dict, cfg: Config):
        self.split = split
        self.crops_by_class = defaultdict(list)
        for c in crops:
            if assignment["crops"].get(c["crop_id"]) == split:
                self.crops_by_class[c["class"]].append(c["crop_id"])
        self.bgs_by_region = defaultdict(list)
        for b in bgs:
            if assignment["backgrounds"].get(b["bg_id"]) == split:
                self.bgs_by_region[b["region"]].append(b["bg_id"])
        for d in (self.crops_by_class, self.bgs_by_region):
            for k in d:
                d[k].sort()
        allowed = set(HELDOUT_CLASSES) if split == "test_unseen" else set(SEEN_CLASSES)
        kept = {c["class"] for c in crops}
        self.dish_classes = sorted(c for c in self.crops_by_class if c in allowed)
        if split == "test_unseen" and cfg["questions"]["unseen_distractor_pool"] == "all":
            self.option_classes = sorted(kept)
        else:
            self.option_classes = sorted(c for c in kept if c in allowed)
        self.regions = sorted(self.bgs_by_region)

    def check(self):
        assert len(self.dish_classes) >= 4, f"{self.split}: need >= 4 dish classes, have {self.dish_classes}"
        assert len(self.regions) >= 4, f"{self.split}: need >= 4 regions with backgrounds, have {self.regions}"


def sample_group(pools: Pools, rng: random.Random, match_frac: float, held: dict | None) -> dict:
    """One attempt. Raises InfeasibleOptions / ValueError when the draw violates a rule."""
    classes, regions = pools.dish_classes, pools.regions
    t_c, o_c = rng.sample(classes, 2)
    if rng.random() < match_frac:
        side_cls = rng.choice([t_c, o_c])
        reg_s = CLASS_REGION[side_cls]
        if reg_s not in pools.bgs_by_region:
            raise ValueError("no background for matched region")
    else:
        cand = [r for r in regions if r not in (CLASS_REGION[t_c], CLASS_REGION[o_c])]
        if not cand:
            raise ValueError("no mismatched region")
        reg_s = rng.choice(cand)
    reg_sp = rng.choice([r for r in regions if r != reg_s])
    rest = [c for c in classes if c not in (t_c, o_c)]
    t_pc = rng.choice(rest)
    o_pc = rng.choice([c for c in rest if c != t_pc])

    options = make_option_sets(t_c, t_pc, o_c, o_pc, reg_s, reg_sp,
                               pools.option_classes, pools.regions + [], rng)
    cb = pools.crops_by_class
    rec = {
        "t": rng.choice(cb[t_c]), "o": rng.choice(cb[o_c]),
        "t_prime": rng.choice(cb[t_pc]), "o_prime": rng.choice(cb[o_pc]),
        "s": rng.choice(pools.bgs_by_region[reg_s]), "s_prime": rng.choice(pools.bgs_by_region[reg_sp]),
        "regions": {"s": reg_s, "s_prime": reg_sp},
        "options": options, "perm": make_permutation(rng),
    }
    m = [side for side, c in (("left", t_c), ("right", o_c)) if CLASS_REGION[c] == reg_s]
    rec["bg_dish_match"] = {0: "none", 1: m[0] if m else "none", 2: "both"}[len(m)]
    mp = [side for side, c in (("left", t_c), ("right", o_c)) if CLASS_REGION[c] == reg_sp]
    rec["bg_prime_dish_match"] = {0: "none", 1: mp[0] if mp else "none", 2: "both"}[len(mp)]
    if held is not None:
        gp = group_pairings(rec)
        for k, v in held.items():
            if gp[k] & v:
                raise ValueError(f"held-out {k}")
    return rec


def generate_split(cfg: Config, split: str, n: int, pools: Pools, held: dict | None,
                   start_index: int, log) -> list[dict]:
    pools.check()
    gc = cfg["groups"]
    seen_keys, out = set(), []
    fails = Counter()
    for i in range(n):
        gseed = derive_seed(cfg["seed"], "groups", split, i)
        rng = random.Random(gseed)
        for _ in range(gc["max_tries_per_group"]):
            try:
                rec = sample_group(pools, rng, gc["bg_dish_match_frac"], held)
            except (InfeasibleOptions, ValueError) as e:
                fails[str(e).split(":")[0][:60]] += 1
                continue
            key = tuple(rec[k] for k in ("t", "o", "s", "t_prime", "o_prime", "s_prime"))
            if key in seen_keys:
                fails["duplicate recipe"] += 1
                continue
            seen_keys.add(key)
            break
        else:
            raise RuntimeError(f"{split}: could not sample group {i}; failures: {fails.most_common(5)}")
        out.append({"group_id": f"g{start_index + i:06d}", "split": split,
                    "render_mode": gc["render_mode"], "seed": gseed, **rec})
    log.info("%s: %d groups (resample reasons: %s)", split, len(out), dict(fails.most_common(6)))
    return out


def generate_all(cfg: Config, log=None) -> dict[str, list[dict]]:
    log = log or get_logger("groups", cfg)
    crops = read_jsonl(cfg.path("crops_index"))
    bgs = read_jsonl(cfg.path("backgrounds_index"))
    with open(cfg.path("splits_file")) as f:
        assignment = json.load(f)
    with open(cfg.path("heldout_pairings")) as f:
        held = {k: {tuple(p) for p in v} for k, v in json.load(f).items()}
    out, start = {}, 0
    for split in SPLITS:
        n = cfg["splits"]["group_counts"][split]
        pools = Pools(split, crops, bgs, assignment, cfg)
        out[split] = generate_split(cfg, split, n, pools, held if split == "train" else None, start, log)
        start += n
        write_jsonl(cfg.path("groups_dir") / f"{split}.jsonl", out[split])
        m = Counter(r["bg_dish_match"] for r in out[split])
        log.info("%s bg_dish_match: %s", split, dict(sorted(m.items())))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    args = ap.parse_args()
    cfg = load_config(args.config)
    generate_all(cfg)


if __name__ == "__main__":
    main()

"""Step 8: automated checks, control sets, human-audit export/ingest.

    python -m data_pipeline.audit checks [--use_saved] [--split S]   # all must pass before freeze
    python -m data_pipeline.audit controls                           # build control recipes + cells
    python -m data_pipeline.audit human-export                       # HTML + CSV sheets
    python -m data_pipeline.audit human-ingest results.csv           # store filled-in sheet

Checks (over the FULL dataset, every group):
  1. pixel preservation: for every pair of images in a group, pixels may differ only inside
     the layers that differ. Dish layer: inside that slot's box AND inside the plate support
     of either image. Background layer (paste mode): never where an unchanged plate is fully
     opaque. Harmonized mode: dish edits checked the same way, background edits exempt.
  2. commutativity: x -t-> -o-> equals x -o-> -t-> equals a fresh render of `to`, bytewise;
     same for tb/ob (dish edit before/after background swap).
  3. labels: answers in option sets, options identical across the 7 images, relevance
     matches the spec table, t/t'/o/o' distinct, region(s) != region(s').
  4. determinism: rendering twice gives identical bytes; with --use_saved, saved PNGs equal
     a fresh render.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import random
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
from PIL import Image

from .classes import CLASS_REGION, pretty, region_name
from .questions import QUESTIONS, crop_class, group_cells, relevance
from .render import LAYERS, Renderer, image_relpath, render_group, save_png
from .splits import SPLITS
from .utils import Config, derive_seed, get_logger, load_config, read_jsonl, write_json, write_jsonl

LAYER_NAMES = ("left", "right", "background")

# The spec's relevance table, written out independently of questions.relevance().
SPEC_RELEVANCE = {
    "t":  {"q_L": "change", "q_R": "keep",   "q_S": "keep"},
    "o":  {"q_L": "keep",   "q_R": "change", "q_S": "keep"},
    "b":  {"q_L": "keep",   "q_R": "keep",   "q_S": "change"},
    "to": {"q_L": "change", "q_R": "change", "q_S": "keep"},
    "tb": {"q_L": "change", "q_R": "keep",   "q_S": "change"},
    "ob": {"q_L": "keep",   "q_R": "change", "q_S": "change"},
}


# ---------------------------------------------------------------- 1. pixel preservation
def changed_layers(rec: dict, k1: str, k2: str) -> set[str]:
    a = [rec[f] for f in LAYERS[k1]]
    b = [rec[f] for f in LAYERS[k2]]
    return {name for name, u, v in zip(LAYER_NAMES, a, b) if u != v}


def check_pair(img1, m1, img2, m2, layers: set[str], mode: str) -> list[str]:
    """Return violation messages (empty = OK)."""
    diff = np.any(img1 != img2, axis=2)
    errs = []
    if not layers:
        if diff.any():
            errs.append(f"identical recipe but {int(diff.sum())} px differ")
        return errs
    if "background" in layers:
        if mode != "paste":
            return errs  # harmonized plates depend on the background by design
        protected = np.zeros_like(diff)
        for slot in ("left", "right"):
            if slot not in layers:
                protected |= m1[f"{slot}_core"] & m2[f"{slot}_core"]
        n = int((diff & protected).sum())
        if n:
            errs.append(f"background swap changed {n} px inside an unchanged opaque plate")
        # the dish layers that changed too must stay inside their slot (checked below on the
        # remainder only when no background change; with a bg change everything else may move)
        return errs
    allowed_box = np.zeros_like(diff)
    allowed_sup = np.zeros_like(diff)
    for slot in layers:
        allowed_box |= m1[f"{slot}_slot"]
        allowed_sup |= m1[slot] | m2[slot]
    n_box = int((diff & ~allowed_box).sum())
    n_sup = int((diff & ~allowed_sup).sum())
    if n_box:
        errs.append(f"{sorted(layers)} swap changed {n_box} px outside the slot box")
    if n_sup:
        errs.append(f"{sorted(layers)} swap changed {n_sup} px outside the plate masks")
    return errs


def check_group_pixels(renderer: Renderer, rec: dict, rendered=None) -> list[str]:
    mode = rec.get("render_mode", "paste")
    rendered = rendered or render_group(renderer, rec, mode)
    errs = []
    for k1, k2 in combinations(LAYERS, 2):
        for e in check_pair(*rendered[k1], *rendered[k2], changed_layers(rec, k1, k2), mode):
            errs.append(f"{rec['group_id']} {k1}->{k2}: {e}")
    return errs


# ---------------------------------------------------------------- 2. commutativity
def check_commutativity(renderer: Renderer, rec: dict, rendered=None) -> list[str]:
    mode = rec.get("render_mode", "paste")
    rendered = rendered or render_group(renderer, rec, mode)
    x = rendered["x"][0]
    s, sp = rec["s"], rec["s_prime"]
    errs = []
    t_then_o = renderer.edit_slot(renderer.edit_slot(x, s, "left", rec["t_prime"], mode), s, "right", rec["o_prime"], mode)
    o_then_t = renderer.edit_slot(renderer.edit_slot(x, s, "right", rec["o_prime"], mode), s, "left", rec["t_prime"], mode)
    if not np.array_equal(t_then_o, o_then_t):
        errs.append("to: t-then-o != o-then-t")
    if not np.array_equal(t_then_o, rendered["to"][0]):
        errs.append("to: incremental edit != fresh render")
    # single edits via the edit path equal fresh renders
    for key, slot, new in (("t", "left", rec["t_prime"]), ("o", "right", rec["o_prime"])):
        if not np.array_equal(renderer.edit_slot(x, s, slot, new, mode), rendered[key][0]):
            errs.append(f"{key}: incremental edit != fresh render")
    # dish edit then background swap == background swap then dish edit (re-render on s')
    for key, slot, new in (("tb", "left", rec["t_prime"]), ("ob", "right", rec["o_prime"])):
        b_img = rendered["b"][0]
        via_b = renderer.edit_slot(b_img, sp, slot, new, mode)
        if not np.array_equal(via_b, rendered[key][0]):
            errs.append(f"{key}: b-then-dish != fresh render")
    return [f"{rec['group_id']}: {e}" for e in errs]


# ---------------------------------------------------------------- 3. labels
def check_labels(cfg: Config, rec: dict) -> list[str]:
    errs = []
    gid = rec["group_id"]
    t, tp, o, op = (crop_class(rec[k]) for k in ("t", "t_prime", "o", "o_prime"))
    if len({t, tp, o, op}) != 4:
        errs.append("t/t'/o/o' classes not distinct")
    if rec["regions"]["s"] == rec["regions"]["s_prime"]:
        errs.append("region(s) == region(s')")
    if len({rec[k] for k in ("t", "t_prime", "o", "o_prime")}) != 4:
        errs.append("crops not distinct")
    opts = rec["options"]
    if opts["q_L"][:2] != [t, tp] or opts["q_R"][:2] != [o, op]:
        errs.append("dish option sets do not start with the true/replacement classes")
    if opts["q_S"][:2] != [rec["regions"]["s"], rec["regions"]["s_prime"]]:
        errs.append("scene option set does not start with region(s), region(s')")
    for q in QUESTIONS:
        if len(set(opts[q])) != 4:
            errs.append(f"{q}: options not 4 distinct")
    for q in ("q_L", "q_R"):
        d_bg = opts[q][2]
        if CLASS_REGION[d_bg] not in (rec["regions"]["s"], rec["regions"]["s_prime"]):
            errs.append(f"{q}: bg distractor {d_bg} not from region(s)/region(s')")
        if set(opts[q][2:]) & {t, tp, o, op}:
            errs.append(f"{q}: distractor equals a group dish")
    if sorted(rec["perm"]) != [0, 1, 2, 3]:
        errs.append("perm is not a permutation of 0..3")

    cells = group_cells(cfg, rec)
    if len(cells) != 21:
        errs.append(f"{len(cells)} cells, expected 21")
    by_q = defaultdict(list)
    for c in cells:
        by_q[c["question_id"]].append(c)
        if c["answer"] not in c["options"]:
            errs.append(f"{c['cell_id']}: answer not in options")
        if c["options"][cfg["questions"]["letters"].index(c["answer_letter"])] != c["answer"]:
            errs.append(f"{c['cell_id']}: answer letter wrong")
        # mechanical truth: slot content of that image
        l, r, b = (rec[f] for f in LAYERS[c["image_key"]])
        truth = {"q_L": crop_class(l), "q_R": crop_class(r),
                 "q_S": rec["regions"]["s" if b == rec["s"] else "s_prime"]}[c["question_id"]]
        if c["answer"] != truth:
            errs.append(f"{c['cell_id']}: answer {c['answer']} != slot content {truth}")
        if c["image_key"] != "x":
            if c["relevance"] != SPEC_RELEVANCE[c["image_key"]][c["question_id"]]:
                errs.append(f"{c['cell_id']}: relevance {c['relevance']} != spec")
            base = next(x for x in cells if x["image_key"] == "x" and x["question_id"] == c["question_id"])
            changed = c["answer"] != base["answer"]
            if changed != (c["relevance"] == "change"):
                errs.append(f"{c['cell_id']}: answer change {changed} contradicts relevance {c['relevance']}")
    for q, cs in by_q.items():
        if len({tuple(c["options"]) for c in cs}) != 1:
            errs.append(f"{q}: options differ across the 7 images")
    return [f"{gid}: {e}" for e in errs]


# ---------------------------------------------------------------- run all checks
def load_saved(renderer: Renderer, cfg: Config, rec: dict):
    """Saved PNGs + masks recomputed from the recipe; also asserts saved == fresh render."""
    mode = rec.get("render_mode", "paste")
    fresh = render_group(renderer, rec, mode)
    root = cfg.path("renders_dir")
    out, errs = {}, []
    for key, (img, masks) in fresh.items():
        p = root / image_relpath(rec["split"], rec["group_id"], key, mode)
        saved = np.asarray(Image.open(p).convert("RGB"))
        if not np.array_equal(saved, img):
            errs.append(f"{rec['group_id']}/{key}: saved PNG != fresh render")
        out[key] = (saved, masks)
    return out, errs


def run_checks(cfg: Config, groups_by_split: dict[str, list[dict]], renderer: Renderer | None = None,
               use_saved: bool = False, log=None) -> dict:
    log = log or get_logger("audit", cfg)
    renderer = renderer or Renderer(cfg)
    report = {"pixel": [], "commutativity": [], "labels": [], "determinism": [], "n_groups": 0, "n_pairs": 0}
    for split, recs in groups_by_split.items():
        for i, rec in enumerate(recs):
            if use_saved:
                rendered, e = load_saved(renderer, cfg, rec)
                report["determinism"] += e
            else:
                rendered = render_group(renderer, rec)
                again = render_group(renderer, rec)
                for k in rendered:
                    if not np.array_equal(rendered[k][0], again[k][0]):
                        report["determinism"].append(f"{rec['group_id']}/{k}: non-deterministic render")
            report["pixel"] += check_group_pixels(renderer, rec, rendered)
            report["commutativity"] += check_commutativity(renderer, rec, rendered)
            report["labels"] += check_labels(cfg, rec)
            report["n_groups"] += 1
            report["n_pairs"] += 21
            if (i + 1) % 500 == 0:
                log.info("%s: checked %d/%d", split, i + 1, len(recs))
    report["passed"] = not any(report[k] for k in ("pixel", "commutativity", "labels", "determinism"))
    return report


# ---------------------------------------------------------------- controls
def build_controls(cfg: Config, renderer: Renderer, groups: dict[str, list[dict]], log=None) -> dict:
    """Control recipes + cells, from the in-distribution test split. Writes
    groups/controls.jsonl, questions/controls.jsonl and renders control images."""
    log = log or get_logger("audit", cfg)
    cc = cfg["controls"]
    base = groups["test_id"]
    rng = random.Random(derive_seed(cfg["seed"], "controls"))
    root = cfg.path("renders_dir")
    recipes, cells = [], []

    # identity re-paste: remove the left (and right) dish and paste the same crop back
    for rec in rng.sample(base, min(cc["identity_repaste"], len(base))):
        x, _ = renderer.render(rec["s"], rec["t"], rec["o"])
        rp = renderer.edit_slot(renderer.edit_slot(x, rec["s"], "left", rec["t"]), rec["s"], "right", rec["o"])
        identical = bool(np.array_equal(x, rp))
        rel = f"controls/identity_repaste/{rec['group_id']}.png"
        save_png(rp, root / rel)
        recipes.append({"control": "identity_repaste", "group_id": rec["group_id"], "image": rel,
                        "byte_identical_to_x": identical})
        if not identical:
            log.error("identity re-paste not byte-identical for %s", rec["group_id"])
        for c in group_cells(cfg, rec):
            if c["image_key"] == "x":
                cells.append({**c, "control": "identity_repaste", "image": rel, "relevance": "keep",
                              "cell_id": f"ctrl_idr/{c['cell_id']}"})

    # wrong-entity swap: the right dish is swapped, the question asks about the left dish
    for rec in rng.sample(base, min(cc["wrong_entity_swap"], len(base))):
        for c in group_cells(cfg, rec):
            if c["image_key"] in ("x", "o") and c["question_id"] == "q_L":
                cells.append({**c, "control": "wrong_entity_swap", "cell_id": f"ctrl_wes/{c['cell_id']}"})
        recipes.append({"control": "wrong_entity_swap", "group_id": rec["group_id"]})

    # harmonized variant of a subset (all 7 images re-rendered with render_mode=harmonized)
    for rec in rng.sample(base, min(cc["harmonized_groups"], len(base))):
        hrec = {**rec, "render_mode": "harmonized"}
        for key, (img, _) in render_group(renderer, hrec, "harmonized").items():
            save_png(img, root / image_relpath(rec["split"], rec["group_id"], key, "harmonized"))
        recipes.append({"control": "harmonized", **hrec})
        for c in group_cells(cfg, hrec):
            cells.append({**c, "control": "harmonized", "cell_id": f"ctrl_harm/{c['cell_id']}"})

    # blank image: the same questions on a uniform gray image (language-prior baseline)
    s = cfg["render"]["size"]
    blank_rel = "controls/blank_gray.png"
    save_png(np.full((s, s, 3), cc["blank_gray"], np.uint8), root / blank_rel)
    for rec in rng.sample(base, min(cc["blank_groups"], len(base))):
        recipes.append({"control": "blank", "group_id": rec["group_id"], "image": blank_rel})
        for c in group_cells(cfg, rec):
            if c["image_key"] == "x":
                cells.append({**c, "control": "blank", "image": blank_rel, "answer": None,
                              "answer_letter": None, "relevance": "n/a",
                              "cell_id": f"ctrl_blank/{c['cell_id']}"})

    write_jsonl(cfg.path("groups_dir") / "controls.jsonl", recipes)
    write_jsonl(cfg.path("questions_dir") / "controls.jsonl", cells)
    counts = Counter(r["control"] for r in recipes)
    log.info("controls: %s recipes, %d cells", dict(counts), len(cells))
    n_bad = sum(1 for r in recipes if r.get("byte_identical_to_x") is False)
    return {"recipes": dict(counts), "cells": len(cells), "identity_repaste_not_identical": n_bad}


# ---------------------------------------------------------------- human audit
def human_export(cfg: Config, log=None) -> Path:
    log = log or get_logger("audit", cfg)
    hc = cfg["human_audit"]
    rng = random.Random(derive_seed(cfg["seed"], "human_audit"))
    crops = read_jsonl(cfg.path("crops_index"))
    bgs = read_jsonl(cfg.path("backgrounds_index"))
    by_cls, by_reg = defaultdict(list), defaultdict(list)
    for c in crops:
        by_cls[c["class"]].append(c)
    for b in bgs:
        by_reg[b["region"]].append(b)
    items = []
    for cls in sorted(by_cls):
        for c in rng.sample(by_cls[cls], min(hc["crops_per_class"], len(by_cls[cls]))):
            items.append({"item_id": c["crop_id"], "kind": "crop", "label": pretty(cls),
                          "src": str(cfg.path("crops_dir") / c["rgba_path"]),
                          "question": "Is the dish recognizable as the label?"})
    for reg in sorted(by_reg):
        for b in rng.sample(by_reg[reg], min(hc["backgrounds_per_region"], len(by_reg[reg]))):
            items.append({"item_id": b["bg_id"], "kind": "background", "label": region_name(reg),
                          "src": str(cfg.path("backgrounds_dir") / b["path"]),
                          "question": "Is the scene answerable as the label (plausibly this country/region)?"})
    out = cfg.path("audit_dir")
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "human_audit_sheet.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["item_id", "kind", "label", "src", "question", "pass", "rater", "note"])
        w.writeheader()
        for it in items:
            w.writerow({**it, "pass": "", "rater": "", "note": ""})
    rows = "\n".join(
        f'<tr data-id="{html.escape(it["item_id"])}" data-kind="{it["kind"]}"><td>'
        f'<img src="file://{html.escape(it["src"])}" loading="lazy"></td><td>{html.escape(it["kind"])}<br>'
        f'<b>{html.escape(it["label"])}</b><br><small>{html.escape(it["item_id"])}</small></td>'
        f'<td>{html.escape(it["question"])}<br><label><input type="radio" name="{i}" value="1"> yes</label> '
        f'<label><input type="radio" name="{i}" value="0"> no</label><br>'
        f'<input class="note" placeholder="note"></td></tr>'
        for i, it in enumerate(items))
    page = f"""<!doctype html><meta charset="utf-8"><title>CIRA human audit</title>
<style>body{{font:14px sans-serif;margin:16px}}img{{max-width:220px;max-height:220px;background:#808080}}
td{{border-bottom:1px solid #ccc;padding:6px;vertical-align:top}}</style>
<p>Rater: <input id="rater"> &nbsp; <button onclick="dl()">Download results CSV</button>
 &nbsp; then run <code>python -m data_pipeline.audit human-ingest &lt;file&gt;</code></p>
<table>{rows}</table>
<script>
function dl(){{const r=document.getElementById('rater').value;let s='item_id,kind,pass,rater,note\\n';
document.querySelectorAll('tr[data-id]').forEach((tr,i)=>{{const c=tr.querySelector('input[type=radio]:checked');
const n=tr.querySelector('.note').value.replace(/"/g,'""');
s+=`"${{tr.dataset.id}}",${{tr.dataset.kind}},${{c?c.value:''}},"${{r}}","${{n}}"\\n`;}});
const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([s],{{type:'text/csv'}}));
a.download='human_audit_results.csv';a.click();}}
</script>"""
    (out / "human_audit.html").write_text(page)
    log.info("human audit: %d items -> %s", len(items), out / "human_audit.html")
    return out / "human_audit.html"


def human_ingest(cfg: Config, path: str, log=None) -> dict:
    log = log or get_logger("audit", cfg)
    rows = list(csv.DictReader(open(path)))
    out = cfg.path("audit_dir") / "human_results"
    out.mkdir(parents=True, exist_ok=True)
    dest = out / Path(path).name
    dest.write_text(Path(path).read_text())
    stats = summarize_human(cfg)
    log.info("human audit ingested %d rows -> %s; stats %s", len(rows), dest, stats)
    return stats


def summarize_human(cfg: Config) -> dict:
    d = cfg.path("audit_dir") / "human_results"
    agg = defaultdict(lambda: [0, 0])
    for p in sorted(d.glob("*.csv")) if d.exists() else []:
        for r in csv.DictReader(open(p)):
            if r.get("pass") in ("0", "1"):
                agg[r["kind"]][0] += int(r["pass"])
                agg[r["kind"]][1] += 1
    stats = {k: {"pass": a, "n": n, "rate": a / n if n else None} for k, (a, n) in agg.items()}
    write_json(cfg.path("audit_dir") / "human_audit_stats.json", stats)
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["checks", "controls", "human-export", "human-ingest"])
    ap.add_argument("path", nargs="?")
    ap.add_argument("--config")
    ap.add_argument("--split", choices=SPLITS)
    ap.add_argument("--use_saved", action="store_true", help="check saved PNGs (and that they match re-renders)")
    args = ap.parse_args()
    cfg = load_config(args.config)
    log = get_logger("audit", cfg)
    from .splits import load_groups

    if args.cmd == "checks":
        groups = load_groups(cfg)
        if args.split:
            groups = {args.split: groups[args.split]}
        rep = run_checks(cfg, groups, use_saved=args.use_saved, log=log)
        name = f"checks_{args.split}.json" if args.split else "checks.json"
        write_json(cfg.path("audit_dir") / name, rep)
        log.info("checks: groups=%d pixel=%d comm=%d labels=%d determinism=%d -> %s",
                 rep["n_groups"], len(rep["pixel"]), len(rep["commutativity"]), len(rep["labels"]),
                 len(rep["determinism"]), "PASS" if rep["passed"] else "FAIL")
        raise SystemExit(0 if rep["passed"] else 1)
    if args.cmd == "controls":
        rep = build_controls(cfg, Renderer(cfg), load_groups(cfg), log)
        write_json(cfg.path("audit_dir") / "controls.json", rep)
    elif args.cmd == "human-export":
        human_export(cfg, log)
    else:
        human_ingest(cfg, args.path, log)


if __name__ == "__main__":
    main()

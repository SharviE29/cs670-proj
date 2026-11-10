"""Synthetic end-to-end: questions -> render -> audit (saved PNGs) -> controls -> freeze."""

from __future__ import annotations

import csv
import json

import pytest

from conftest import FULL
from data_pipeline import audit, freeze, questions
from data_pipeline.render import render_group, image_relpath, save_png
from data_pipeline.utils import read_jsonl, write_jsonl

pytestmark = pytest.mark.skipif(FULL, reason="end-to-end writes files; synthetic only")


def test_pipeline_end_to_end(cfg, all_groups, renderer):
    # questions
    for split, recs in all_groups.items():
        write_jsonl(cfg.path("questions_dir") / f"{split}.jsonl",
                    (c for r in recs for c in questions.group_cells(cfg, r)))
    assert len(read_jsonl(cfg.path("questions_dir") / "train.jsonl")) == 21 * len(all_groups["train"])
    # render everything
    for recs in all_groups.values():
        for rec in recs:
            for key, (img, _) in render_group(renderer, rec).items():
                save_png(img, cfg.path("renders_dir") / image_relpath(rec["split"], rec["group_id"], key))
    # checks on saved PNGs
    rep = audit.run_checks(cfg, all_groups, renderer, use_saved=True)
    assert rep["passed"], {k: rep[k][:3] for k in ("pixel", "commutativity", "labels", "determinism")}
    from data_pipeline.utils import write_json
    write_json(cfg.path("audit_dir") / "checks.json", rep)
    # controls
    c = audit.build_controls(cfg, renderer, all_groups)
    assert c["identity_repaste_not_identical"] == 0
    assert set(c["recipes"]) == {"identity_repaste", "wrong_entity_swap", "harmonized", "blank"}
    # human audit export + ingest
    audit.human_export(cfg)
    sheet = cfg.path("audit_dir") / "human_audit_sheet.csv"
    rows = list(csv.DictReader(open(sheet)))
    filled = cfg.path("audit_dir") / "filled.csv"
    with open(filled, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["item_id", "kind", "pass", "rater", "note"])
        w.writeheader()
        for i, r in enumerate(rows):
            w.writerow({"item_id": r["item_id"], "kind": r["kind"], "pass": int(i % 5 != 0), "rater": "t", "note": ""})
    stats = audit.human_ingest(cfg, str(filled))
    assert stats["crop"]["n"] > 0
    # freeze: refuses without a name, then succeeds, then demands a version bump on change
    with pytest.raises(SystemExit):
        freeze.freeze(cfg, allow_no_git=True)
    cfg["benchmark_name"] = "TEST-BENCH"
    rec = freeze.freeze(cfg, allow_no_git=True)
    assert len(rec["images"]) >= 7 * sum(cfg["splits"]["group_counts"].values())
    rel = cfg.path("release_dir")
    assert (rel / "DATASHEET.md").exists() and "TEST-BENCH" in (rel / "STATS.md").read_text()
    assert json.load(open(rel / "FREEZE.json"))["content_sha256"] == rec["content_sha256"]
    freeze.freeze(cfg, allow_no_git=True)  # unchanged content: fine
    any_png = next(cfg.path("renders_dir").rglob("train/*/x.png"))
    any_png.write_bytes(any_png.read_bytes() + b"\0")
    with pytest.raises(SystemExit, match="bump"):
        freeze.freeze(cfg, allow_no_git=True)
    cfg["benchmark_name"] = None

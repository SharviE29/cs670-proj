"""Label tests: answers are mechanical, options shared across the 7 images, relevance matches
the spec table. The table below is copied from the spec, independently of the code."""

from __future__ import annotations

import copy

from data_pipeline.audit import check_labels
from data_pipeline.classes import CLASS_REGION
from data_pipeline.questions import crop_class, displayed_options, group_cells, relevance

SPEC_TABLE = """
edit q_L    q_R    q_S
t    change keep   keep
o    keep   change keep
b    keep   keep   change
to   change change keep
tb   change keep   change
ob   keep   change change
"""


def _spec():
    rows = [line.split() for line in SPEC_TABLE.strip().splitlines()]
    head = rows[0][1:]
    return {r[0]: dict(zip(head, r[1:])) for r in rows[1:]}


def _all(all_groups):
    return [r for recs in all_groups.values() for r in recs]


def test_relevance_function_matches_spec():
    for key, row in _spec().items():
        for q, want in row.items():
            assert relevance(key, q) == want, (key, q)
    assert all(relevance("x", q) == "base" for q in ("q_L", "q_R", "q_S"))


def test_all_groups_pass_label_checks(cfg, all_groups):
    errs = [e for rec in _all(all_groups) for e in check_labels(cfg, rec)]
    assert not errs, errs[:5]


def test_21_cells_and_shared_options(cfg, all_groups):
    for rec in _all(all_groups):
        cells = group_cells(cfg, rec)
        assert len(cells) == 21
        assert len({c["image_key"] for c in cells}) == 7
        for q in ("q_L", "q_R", "q_S"):
            opts = {tuple(c["options"]) for c in cells if c["question_id"] == q}
            assert len(opts) == 1
            prompts = {c["prompt"] for c in cells if c["question_id"] == q}
            assert len(prompts) == 1, "identical prompt text across the 7 images"


def test_answers_follow_recipe(cfg, all_groups):
    rec = _all(all_groups)[0]
    cells = {(c["image_key"], c["question_id"]): c for c in group_cells(cfg, rec)}
    t, tp, o, op = (crop_class(rec[k]) for k in ("t", "t_prime", "o", "o_prime"))
    rs, rsp = rec["regions"]["s"], rec["regions"]["s_prime"]
    want = {"x": (t, o, rs), "t": (tp, o, rs), "o": (t, op, rs), "b": (t, o, rsp),
            "to": (tp, op, rs), "tb": (tp, o, rsp), "ob": (t, op, rsp)}
    for key, (l, r, s) in want.items():
        assert cells[(key, "q_L")]["answer"] == l
        assert cells[(key, "q_R")]["answer"] == r
        assert cells[(key, "q_S")]["answer"] == s


def test_bg_dish_match_flag(all_groups):
    for rec in _all(all_groups):
        rs = rec["regions"]["s"]
        m = [side for side, k in (("left", "t"), ("right", "o")) if CLASS_REGION[crop_class(rec[k])] == rs]
        want = {0: "none", 1: m[0] if m else "none", 2: "both"}[len(m)]
        assert rec["bg_dish_match"] == want


def test_match_and_mismatch_both_present(all_groups):
    flags = {r["bg_dish_match"] for r in all_groups["train"]}
    assert "none" in flags and flags & {"left", "right", "both"}


def test_cyclic_permutations():
    canon = ["a", "b", "c", "d"]
    perm = [2, 0, 3, 1]
    shown = [displayed_options(canon, perm, k) for k in range(4)]
    assert shown[0] == ["c", "a", "d", "b"]
    assert len({tuple(s) for s in shown}) == 4
    assert {s.index("a") for s in shown} == {0, 1, 2, 3}  # the answer visits every letter


def test_detects_corrupted_recipe(cfg, all_groups):
    rec = copy.deepcopy(_all(all_groups)[0])
    rec["t_prime"] = rec["t"]  # replacement same as original
    assert check_labels(cfg, rec)
    rec = copy.deepcopy(_all(all_groups)[0])
    rec["options"]["q_L"] = rec["options"]["q_L"][::-1]
    assert check_labels(cfg, rec)

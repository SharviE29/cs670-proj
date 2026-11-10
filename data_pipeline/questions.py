"""Step 6: multiple-choice questions.  [FIXED]

Per group: one canonical option set per question and one permutation, shared by all 7
images. The correct answer of each (image, question) cell is read mechanically from the
recipe: the class of the crop in that slot, or the region of that background.

Canonical option order (before the group permutation):
  q_L: [t, t', bg-associated distractor, random distractor]      (classes)
  q_R: [o, o', bg-associated distractor, random distractor]      (classes)
  q_S: [region(s), region(s'), other region, other region]       (regions)

Relevance r(q, edit) is derived from which layers an edit changes; tests/test_labels.py
checks it against the spec table, which is written out independently there.

CLI:  python -m data_pipeline.questions [--show GROUP_ID]
"""

from __future__ import annotations

import argparse
import random

from .classes import CLASS_REGION, pretty, region_name
from .render import LAYERS, image_relpath
from .utils import Config, get_logger, load_config, read_jsonl, write_jsonl

QUESTIONS = ("q_L", "q_R", "q_S")
QUESTION_LAYER = {"q_L": "left", "q_R": "right", "q_S": "background"}
KEY_LAYERS_CHANGED = {  # vs the base image x
    "x": set(), "t": {"left"}, "o": {"right"}, "b": {"background"},
    "to": {"left", "right"}, "tb": {"left", "background"}, "ob": {"right", "background"},
}


class InfeasibleOptions(Exception):
    pass


def crop_class(crop_id: str) -> str:
    return crop_id.split("/", 1)[0]


def make_option_sets(t: str, t_p: str, o: str, o_p: str, reg_s: str, reg_sp: str,
                     class_pool: list[str], region_pool: list[str], rng: random.Random) -> dict:
    """Canonical option sets (class names / region keys). Raises InfeasibleOptions."""
    used = {t, t_p, o, o_p}
    if len(used) != 4:
        raise InfeasibleOptions("t, t', o, o' must be 4 distinct classes")
    if reg_s == reg_sp:
        raise InfeasibleOptions("s and s' must have different regions")
    pool = sorted(class_pool)
    bg_assoc = [c for c in pool if CLASS_REGION[c] in (reg_s, reg_sp) and c not in used]
    if not bg_assoc:
        raise InfeasibleOptions(f"no background-associated distractor for {reg_s}/{reg_sp}")
    sets = {}
    for q, (a, b) in (("q_L", (t, t_p)), ("q_R", (o, o_p))):
        d_bg = rng.choice(bg_assoc)
        rest = [c for c in pool if c not in used and c != d_bg]
        if not rest:
            raise InfeasibleOptions("no random distractor left")
        sets[q] = [a, b, d_bg, rng.choice(rest)]
    others = [r for r in sorted(region_pool) if r not in (reg_s, reg_sp)]
    if len(others) < 2:
        raise InfeasibleOptions("need >= 4 regions")
    sets["q_S"] = [reg_s, reg_sp, *rng.sample(others, 2)]
    return sets


def make_permutation(rng: random.Random, n: int = 4) -> list[int]:
    """perm[i] = canonical index shown at letter i."""
    p = list(range(n))
    rng.shuffle(p)
    return p


def displayed_options(canonical: list[str], perm: list[int], shift: int = 0) -> list[str]:
    """Options in display order. `shift` gives the 4 cyclic variants used at evaluation."""
    shown = [canonical[i] for i in perm]
    k = shift % len(shown)
    return shown[k:] + shown[:k]


def answer_entity(rec: dict, key: str, q: str) -> str:
    left_f, right_f, bg_f = LAYERS[key]
    if q == "q_L":
        return crop_class(rec[left_f])
    if q == "q_R":
        return crop_class(rec[right_f])
    return rec["regions"][bg_f]


def relevance(key: str, q: str) -> str:
    if key == "x":
        return "base"
    return "change" if QUESTION_LAYER[q] in KEY_LAYERS_CHANGED[key] else "keep"


def option_text(q: str, entity: str) -> str:
    return region_name(entity) if q == "q_S" else pretty(entity)


def prompt_text(cfg: Config, q: str, shown: list[str]) -> str:
    qc = cfg["questions"]
    letters = qc["letters"]
    lines = [qc[q]] + [f"{letters[i]}. {option_text(q, e)}" for i, e in enumerate(shown)]
    return "\n".join(lines + [qc["answer_suffix"]])


def group_cells(cfg: Config, rec: dict) -> list[dict]:
    """The 21 (image, question) cells of one group."""
    letters = cfg["questions"]["letters"]
    mode = rec.get("render_mode", "paste")
    cells = []
    for key in LAYERS:
        for q in QUESTIONS:
            canonical = rec["options"][q]
            shown = displayed_options(canonical, rec["perm"])
            ans = answer_entity(rec, key, q)
            cells.append({
                "cell_id": f"{rec['group_id']}/{key}/{q}",
                "group_id": rec["group_id"], "split": rec["split"], "image_key": key,
                "image": image_relpath(rec["split"], rec["group_id"], key, mode),
                "question_id": q, "question": cfg["questions"][q],
                "options": shown, "options_text": [option_text(q, e) for e in shown],
                "answer": ans, "answer_letter": letters[shown.index(ans)],
                "relevance": relevance(key, q), "prompt": prompt_text(cfg, q, shown),
                "perm": rec["perm"],
            })
    return cells


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    ap.add_argument("--show", help="print the 21 cells of this group id and exit")
    args = ap.parse_args()
    cfg = load_config(args.config)
    log = get_logger("questions", cfg)
    gdir, qdir = cfg.path("groups_dir"), cfg.path("questions_dir")
    for split in cfg["splits"]["group_counts"]:
        p = gdir / f"{split}.jsonl"
        if not p.exists():
            continue
        recs = read_jsonl(p)
        if args.show:
            for rec in recs:
                if rec["group_id"] == args.show:
                    print_cells(group_cells(cfg, rec))
                    return
            continue
        n = write_jsonl(qdir / f"{split}.jsonl", (c for r in recs for c in group_cells(cfg, r)))
        log.info("%s: %d groups -> %d cells", split, len(recs), n)


def print_cells(cells: list[dict]) -> None:
    print(f"{'key':3s} {'q':4s} {'rel':6s} {'ans':3s} {'answer':22s} options")
    for c in cells:
        opts = " | ".join(f"{l}. {t}" for l, t in zip("ABCD", c["options_text"]))
        print(f"{c['image_key']:3s} {c['question_id']:4s} {c['relevance']:6s} "
              f"{c['answer_letter']:3s} {option_text(c['question_id'], c['answer']):22s} {opts}")


if __name__ == "__main__":
    main()

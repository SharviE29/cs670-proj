# CIRA: a controlled, pixel-preserving food-scene dataset (CS670)

Vision–language models let the **background leak into object answers**: a taco on a Tokyo
street gets called sushi. This project measures and trains **selective dependence**: an
answer should change only when the entity the question asks about changes.

This repository contains:

1. **`data_pipeline/`**: builds the dataset (the main deliverable). Every image is composited
   from layers: **background + left plate + right plate**, at fixed slots. An edit swaps exactly
   one layer, and every pixel outside that layer's region is byte-identical. No generative model
   ever touches a pixel.
2. **Pilot smoke test** (`make_pairs.py`, `run_eval.py`, `attention_share.py`): the original
   425-pair experiment that checked whether the background effect exists at all. Kept for
   reference; see [Pilot smoke test](#pilot-smoke-test-original-experiment) at the end.

---

## What the dataset looks like

A **group** is one base scene plus six edits, **7 images**:

| key | left dish | right dish | background |
|-----|-----------|------------|------------|
| x   | t  | o  | s  |
| t   | t′ | o  | s  |
| o   | t  | o′ | s  |
| b   | t  | o  | s′ |
| to  | t′ | o′ | s  |
| tb  | t′ | o  | s′ |
| ob  | t  | o′ | s′ |

Each image gets three 4-option multiple-choice questions:

- `q_L`: *What dish is on the left plate?*
- `q_R`: *What dish is on the right plate?*
- `q_S`: *In which country/region was this photo taken?*

That gives 21 labelled cells per group. Within a group, all 7 images share the same options in
the same order. Correct answers come mechanically from the recipe (what is actually in the
slot). Every cell also has a relevance label (`change` / `keep`): whether the answer *should*
change relative to the base image `x`.

| Split | Groups | Contents |
|---|---|---|
| train | 3,000 | 40 seen classes |
| val | 300 | seen classes |
| test_id | 1,000 | seen classes, new photos and backgrounds |
| test_unseen | 400 | only the 10 held-out classes |

No food photo and no background image appears in more than one split. Some (class, region),
dish-swap and region-swap combinations are withheld from train for composition tests.

Sources:

- **Food:** cutouts from the Food-101 **test** split (GroundingDINO → SAM2 → filters).
- **Backgrounds:** about 1,200 Wikimedia Commons scenes across 16 regions, with per-image
  licenses.
- **Rendering:** 448 × 448 PNG.

---

## How to read this project

Start with these, in order:

| File | What it tells you |
|---|---|
| [data_pipeline/config.yaml](data_pipeline/config.yaml) | Every constant: paths, model choices, thresholds, slot positions, split sizes. Items marked `OPEN ITEM` still need a decision. |
| [data_pipeline/classes.py](data_pipeline/classes.py) | The 50 food classes, class → region map, and the 10 held-out classes. Contested mappings are marked `# CHECK:`. |
| [data_pipeline/render.py](data_pipeline/render.py) | The compositor and why it preserves pixels. Read the module docstring. |
| [data_pipeline/groups.py](data_pipeline/groups.py) + [questions.py](data_pipeline/questions.py) | How a group and its 21 question cells are built. |
| [data_pipeline/audit.py](data_pipeline/audit.py) | The checks that must pass before release. |
| [tests/](tests/) | Executable spec: pixel preservation, labels, splits, end to end. |

Pipeline modules, one per step:

```
data_pipeline/
  config.yaml          all constants (config.demo.yaml: small-model laptop variant)
  utils.py             config loading, seeds, JSONL I/O, hashing
  classes.py           Step 1  class list and class -> region mapping
  crops.py             Step 2  Food-101 cutouts: detect, segment, filter, CLIP and Qwen checks
  backgrounds.py       Step 3  Wikimedia download, license filter, CLIP audit
  render.py            Step 4  deterministic layer compositor
  groups.py            Step 5  7-image group recipes
  questions.py         Step 6  multiple-choice questions, 21 cells per group
  splits.py            Step 7  split assignment, near-duplicate check, leakage asserts
  audit.py             Step 8  automated checks, control sets, human-audit sheet
  freeze.py            Step 9  manifests, SHA-256, stats, datasheet
  slurm/build.sbatch   Unity (Slurm) job script
notebooks/
  cira_build_dataset_colab.ipynb   the whole build on Google Colab
tests/                 pytest suite (synthetic data by default, full dataset on demand)
```

### Guarantees and where they are enforced

| Guarantee | How |
|---|---|
| An edit changes only the swapped layer's pixels | Plates never leave their hard mask (the feather ramps *inward*) and the mask never leaves its slot box. `audit.check_group_pixels` checks **every pair** of images in **every** group. |
| A background swap never changes an opaque plate pixel | Same check. |
| Edits commute | `to` reached via t-then-o is byte-identical to o-then-t and to a fresh render (`audit.check_commutativity`). |
| Deterministic output | Same inputs give identical bytes. Saved PNGs are compared with re-renders. |
| Labels are mechanical | Answer = slot content; options identical across the 7 images; relevance matches the spec table (`audit.check_labels`). |
| No split leakage | `splits.verify` asserts disjoint photos and backgrounds, held-out classes only in test_unseen, withheld pairings absent from train, and no near-duplicates (pHash) across splits. |
| A frozen release can't change silently | `freeze.py` refuses changed content without a `version` bump. |

The test suite includes tests that inject a leaked pixel, a cross-split photo or a corrupted
label, and assert the checks catch it. A passing run therefore means the checks were able to
fail and did not.

---

## Quick start (laptop, no GPU)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m data_pipeline.classes      # print and validate the class/region table
pytest -q                            # whole pipeline on a synthetic dataset, about 1 minute
```

`pytest` builds a tiny synthetic dataset (fake plates and backgrounds), then runs splits,
groups, questions, rendering, every audit check, controls, the human-audit export and the
freeze. Run it first after any code change.

---

## Building the real dataset

You need a GPU for Steps 2, 3 (audit) and 8, plus internet for the Food-101 and Wikimedia
downloads. There are three ways to run it.

### Option A: Google Colab (easiest)

Open [notebooks/cira_build_dataset_colab.ipynb](notebooks/cira_build_dataset_colab.ipynb) in
Colab and choose *Runtime → Change runtime type → GPU*, then run the cells top to bottom.

- The notebook clones this repo. For a private repo, add a `GH_TOKEN` Colab secret.
- It installs the pinned dependencies and downloads Food-101.
- It saves a checkpoint to Google Drive after every step.
- Cutout extraction resumes after a disconnect.

### Option B: UMass Unity (Slurm)

```bash
# once, on a login node (internet):
python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
export CIRA_DATA_ROOT=/work/pi_<you>/cira HF_HOME=$CIRA_DATA_ROOT/hf
wget -P $CIRA_DATA_ROOT http://data.vision.ee.ethz.ch/cvl/food-101.tar.gz && tar -xzf $CIRA_DATA_ROOT/food-101.tar.gz -C $CIRA_DATA_ROOT
python -c "from huggingface_hub import snapshot_download as s; [s(m) for m in ['IDEA-Research/grounding-dino-base','facebook/sam2.1-hiera-large','openai/clip-vit-large-patch14','Qwen/Qwen2.5-VL-3B-Instruct']]"
python -m data_pipeline.backgrounds fetch            # Wikimedia download (internet)

# GPU jobs (run offline):
sbatch --export=ALL,STEP=crops   data_pipeline/slurm/build.sbatch
sbatch --export=ALL,STEP=qwen    data_pipeline/slurm/build.sbatch
sbatch --export=ALL,STEP=bgaudit data_pipeline/slurm/build.sbatch
```

Then continue with the CPU steps below, using `sbatch --array=0-15 ... STEP=render` for
rendering. Edit the partition and account names in `build.sbatch` first.

### Option C: any GPU machine, step by step

Set `CIRA_DATA_ROOT` (default `./data`) and put Food-101 at `$CIRA_DATA_ROOT/food-101`.
To use a different config file, set `CIRA_CONFIG=path/to/override.yaml`. An override file
can start with `_base: config.yaml` and list only the keys it changes.

| # | Command | Output | What to look at |
|---|---|---|---|
| 2 | `python -m data_pipeline.crops extract` | `crops/<class>/<id>.png` (RGBA) + `_mask.png`, `rejections.jsonl` | Resumable; `--fresh` restarts |
| 2 | `python -m data_pipeline.crops qwen-check` | `crops/qwen_class_check.json` | Logged selection-bias exception |
| 2 | `python -m data_pipeline.crops finalize` | `crops/index.jsonl`, `class_status.json` | Dropped classes and why |
| 2 | `python -m data_pipeline.crops report` | Rejection table, `crops/contact_sheet.jpg` | **Review before continuing** |
| 3 | `python -m data_pipeline.backgrounds fetch` | `backgrounds/img/`, `licenses.csv` | Needs internet |
| 3 | `python -m data_pipeline.backgrounds audit` | `backgrounds/index.jsonl` | CLIP region and scene check |
| 3 | `python -m data_pipeline.backgrounds report` | `backgrounds/contact_sheet.jpg` | **Review before continuing** |
| 7 | `python -m data_pipeline.splits assign` | `splits/assignment.json`, `heldout_pairings.json` | Split ownership is fixed before groups |
| 5 | `python -m data_pipeline.groups` | `groups/<split>.jsonl` (recipes, IDs only) | |
| 6 | `python -m data_pipeline.questions` | `questions/<split>.jsonl` (21 cells per group) | `--show g000000` prints one group |
| 7 | `python -m data_pipeline.splits verify` | Assertions | Must pass |
| 4 | `python -m data_pipeline.render --demo group.png` | 7 images side by side | Check slot sizes |
| 4 | `python -m data_pipeline.render --split train` (and val, test_id, test_unseen) | `renders/<split>/<group>/<key>.png` | `--shard i --num_shards n` to parallelise |
| 8 | `python -m data_pipeline.audit controls` | Control recipes, cells and images | Identity re-paste, wrong-entity, harmonized, blank |
| 8 | `python -m data_pipeline.audit checks --use_saved` | `audit/checks.json` | **Must PASS** (full dataset) |
| 8 | `CIRA_TEST_FULL=1 pytest -q tests` | | Same tests over every real group |
| 8 | `python -m data_pipeline.audit human-export` | `audit/human_audit.html` | Open, mark, download CSV |
| 8 | `python -m data_pipeline.audit human-ingest results.csv` | `audit/human_audit_stats.json` | |
| 9 | `python -m data_pipeline.freeze` | `release/` (`FREEZE.json`, `STATS.md`, `DATASHEET.md`, manifests) | Needs `benchmark_name`, passing checks, git commit |

Splits are assigned before groups are sampled. That is why Step 7's `assign` runs before
Steps 5–6.

### Using the finished dataset

- **Images:** `renders/<split>/<group_id>/<key>.png`. They can also be re-rendered from
  `groups/<split>.jsonl` with `render.Renderer` + `render.render_group`.
- **Questions:** `questions/<split>.jsonl`, one line per (image, question) cell. Fields include
  `image`, `prompt`, `options`, `answer`, `answer_letter` and `relevance` (`base` / `keep` /
  `change`), plus `perm`. To run the 4 cyclic option orders, use
  `questions.displayed_options(canonical, perm, shift)` for shift = 0..3.
- **Controls:** `questions/controls.jsonl`, with a `control` field set to `identity_repaste`,
  `wrong_entity_swap`, `harmonized` or `blank`.
- **Integrity:** `release/FREEZE.json` lists the SHA-256 of every file. Cite the backgrounds
  using `backgrounds_licenses.csv`.

---

## Open decisions (before freezing)

- Final 50 classes and the class → region mapping (`classes.py`, `# CHECK:` lines).
- Slot positions and footprint (`render.slots`). Proposed: 180 px boxes at the bottom left and
  right, with a 3 px feather.
- Wikimedia categories per region (`backgrounds.categories`).
- Benchmark name (`benchmark_name`; `freeze.py` refuses to run while it is null).

---

## Pilot smoke test (original experiment)

| Script | Question it answers |
|---|---|
| `fetch_backgrounds.py` | Downloads about 30 landmark backgrounds and their credits (`backgrounds/credits.csv`) |
| `make_pairs.py` | Pastes one plate (ellipse cutout) on different backgrounds → `pairs/` |
| `run_eval.py` | Does the background change the dish answer (bad) or the context answer (good)? Uses Qwen2.5-VL, `--dry_run` without a GPU |
| `attention_share.py` | Per-layer attention to the plate for dish vs context questions |

```bash
python fetch_backgrounds.py
python make_pairs.py --from_hf --bg_dir backgrounds --out pairs --n_per_dish 5
python run_eval.py --pairs pairs/meta.jsonl --out results.csv
python attention_share.py --pairs pairs/meta.jsonl --n 30 --out attn
```

The dataset pipeline reuses the pilot's centre-crop resizing and its "same options, same
order across compared images" design. It replaces the ellipse cutout with real segmentation,
the single plate with two independently editable slots, and the per-record options with one
option set per group.

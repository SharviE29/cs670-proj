# Smoke tests: does background context leak into a VLM's answers?

Three small scripts to check, in about an afternoon, whether the project's core
effect exists before you commit to training anything.

| Script | Question it answers | Needs GPU |
|---|---|---|
| `make_pairs.py` | Builds the same food plate on different backgrounds (plus masks) | No |
| `run_eval.py` | Does the background change the **dish** answer (bad) and the **context** answer (good)? | Yes (or `--dry_run`) |
| `attention_share.py` | Does the model look at the plate more for dish questions than for context questions? | Yes |

The pipeline was tested end to end with fake images and `--dry_run`. It has
**not** been run with a real model yet, so expect small fixes on first run
(e.g. transformers version differences).

---

## 0. Setup

```bash
pip install -r requirements.txt
```

GPU memory: Qwen2.5-VL-**3B** needs about 8–10 GB in bf16, **7B** about 17–20 GB.
Start with 3B (the default); switch with `--model Qwen/Qwen2.5-VL-7B-Instruct`.
If you have no GPU yet, everything except the model calls runs on a laptop
(`--dry_run`), and the free Colab T4 (16 GB) can run the 3B model.

## 1. Collect backgrounds

**Automatic (recommended):** download freely licensed landmark photos from
Wikimedia Commons (3 landmarks per country, 2 photos each):

```bash
python fetch_backgrounds.py                                   # all 10 countries
python fetch_backgrounds.py --countries japan italy france mexico india   # just 5
```

Then open `backgrounds/preview.jpg` and **delete anything that isn't a clear,
outdoor, recognisable scene** (maps, interiors, close-ups, crowds of faces, food).
Search results are usually good but never perfect. `backgrounds/credits.csv`
lists the author and license of every image: cite it in your report and poster.

**Manual alternative:** make one folder per country and put 3–5 photos in each:

```
backgrounds/
  japan/     (e.g. Kyoto temple, Tokyo street, Mount Fuji)
  italy/     (Colosseum, Venice canal, Florence street)
  france/    (Eiffel Tower, Paris café street)
  mexico/    (Chichén Itzá, Mexico City square)
  india/     (Taj Mahal, Jaipur palace)
  china/     (Great Wall, Forbidden City)
  korea/     (Gyeongbokgung palace, Seoul street)
  thailand/  (Bangkok temple)
  vietnam/   (Hạ Long Bay, Hoi An lanterns)
  spain/     (Sagrada Família, Seville plaza)
```

Wikimedia Commons is a good source (check each image's license). You only need
a few countries for the smoke test; 4–5 is enough. Prefer photos where the
landmark or setting is clearly recognizable, since that's the "context" signal.

## 2. Build the image pairs

```bash
python make_pairs.py --from_hf --bg_dir backgrounds --out pairs --n_per_dish 5
```

This streams Food-101 from Hugging Face (no full 5 GB download) and keeps
5 photos for each of the 17 dishes in `common.py`. With 5 countries that gives
17 × 5 × 5 = 425 paired records.

Open a few files in `pairs/img/` and check that they look reasonable. The default
cut-out is an ellipse (Food-101 photos are mostly centred on the plate). For a
cleaner cut-out, add `--use_rembg` (needs `pip install rembg`).

`--plate_frac` controls how big the plate is (default 0.45 of image width).
Smaller plates make the background more dominant, so the test is harder.

## 3. Run the main smoke test

```bash
python run_eval.py --pairs pairs/meta.jsonl --out results.csv
```

Each record is asked 5 questions (3 dish, 2 context), so 425 records means
about 2,100 forward passes: roughly 10–20 minutes for the 3B model on one GPU.
Use `--limit 50` for a 2-minute first check.

**How the test is designed (paired):** for each record, the dish question is
asked with the *same* options in the *same* order on the cultural composite, the
neutral composite and the original photo. One wrong option is always a dish from
the **background's** country (the "leak" option). So any change in the answer is
caused by the background alone.

The summary is printed and saved to `results_summary.txt`.

## 4. How to read the results

| Metric | Want for a good model | What it means if it fails |
|---|---|---|
| Accuracy: original vs neutral | about equal | If neutral is much lower, the pasting itself confuses the model. Fix the cut-out (try `--use_rembg`) before going further. |
| Accuracy: neutral vs **mismatched** bg | equal | A drop means the background hurts dish recognition. **This is the effect you're studying.** |
| **Invariance** | close to 100% | % of mismatched cases where the dish answer is unchanged by the background |
| **Leakage**, cultural vs neutral | equal | If it goes up, the model is pulled toward the *background's* cuisine (e.g. "pizza" in front of the Colosseum) |
| **Sensitivity**: bg-only vs bg+food | both high | The model should still recognise the setting when food is added |
| Reverse leakage | equal | Does the food pull the *country* answer toward the food's country? |

Every number has a ± 95% confidence interval. If the interval is wider than the
effect, add more items (`--n_per_dish 10`).

## 5. Decision rules: what to do next

1. **Clear background effect** (accuracy drop or leakage increase of at least ~5
   points, outside the error bars): the project's premise holds. Go to step 6.
2. **Small effect** with Qwen2.5-VL-3B:
   - shrink the plate: `--plate_frac 0.25`
   - try a weaker/older model and a stronger one, to see whether the effect
     shrinks with scale (that trend is itself worth reporting);
   - use CultureMix's own images if they're released (they report a 14% drop).
3. **Neutral is much worse than original**: your images look fake to the model.
   Fix the compositing before interpreting anything else.
4. **Still no effect anywhere**: the honest result is "modern VLMs are robust to
   this". That's a valid finding, but you'd then want to pivot the project's
   emphasis to the mechanistic analysis (why they're robust).

## 6. Attention smoke test (mechanistic first look)

```bash
python attention_share.py --pairs pairs/meta.jsonl --n 30 --out attn
```

Produces `attn.png`: for each LLM layer, the share of the answer token's
attention to the image that lands on the **plate**, for dish questions vs
context questions. The dashed line is the plate's share of the image area.

- Dish line **above** context line in middle–late layers: the model already
  routes attention by question to some degree.
- Lines **on top of each other**: attention doesn't depend on the question, which
  is a good motivation for a question-conditioned method.

Attention is only a rough proxy. Causal tests (attention knockout, activation
patching of background tokens) are the next step, once the effect is confirmed.

## 7. After the smoke test (not included yet)

1. **Baselines:** prompt-only ("focus on the food"); ordinary LoRA fine-tuning on
   mixed backgrounds (the CultureMix-style approach).
2. **Invariance + sensitivity training:** LoRA with an added KL term between the
   dish-answer distributions for the same food on two backgrounds, with every
   batch containing both question types.
3. **Held-out split:** train and test on *different food photos and different
   background photos*, never just new backgrounds for the same food photo.
4. **Causal analysis:** block or patch background tokens and measure how much the
   dish answer moves before vs after training.

## Files

```
fetch_backgrounds.py  download landmark backgrounds + credits from Wikimedia Commons
common.py           dish -> country map, label helpers
make_pairs.py       build composites, masks, metadata (pairs/meta.jsonl)
run_eval.py         paired multiple-choice evaluation + metrics (+ --dry_run)
attention_share.py  per-layer attention to plate vs background
```

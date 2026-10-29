# vm-00046: Text at Twice the Scale, Tokens From the Summaries

**Date:** 2026-10-29
**Status:** Complete
**Source:** `exp/vm-00046` (code: `visual-motor-agent/dev/`)
**Outputs:** `~/data/research/visual-motor-agent/exp_00046/`
**Task plan:** [`../tasks/00054-summary-only-tokens.md`](../tasks/00054-summary-only-tokens.md)

## Goal

The owner's test of the pyramid's premise. A level-1 summary's thumbnail is its block, downscaled. Draw the text world
at twice the scale, letters included, and a B1 thumbnail holds exactly what the normal world's patch holds: one letter.
Then:

1. A model whose tokens are only those thumbnails sees the same information, in the same order, as the normal model.
   It should learn as well (NB2 against NB).
2. Adding the patch tokens back (the usual layout) should cost compute, not quality (NB3a, NB3).

Text world only; the owner asked for no other evaluations.

## Method

- **`model.sum_only = 1`** (task 00054; needs `uni_pyr` ≥ 1 and `sum_thumb = 1`): the patches are not tokens, only
  the input to the thumbnails. A step is [body, the summaries in pyramid order, move]. Checkpoint header slot 53.
  `tests/test_summary_only.py`: with the same parameters, the normal model on a frame and the B1-only model on its 2×
  upscale give the same outputs (C++ and GPU, within 2e-5).
- **The scaled world** (`configs/T_text_B2.json`): 32 × 32 pages, every letter a 4 × 4 block of its grey, the same 64
  letters per page in the same reading order; patch 2, so a 16 × 16 grid and each letter 2 × 2 patches.
- **Arms**, seed 1, 100k ticks, full attention (no window), checkpoints every 10k:

  | arm | world | tokens per step | params | ticks/s |
  |---|---|---|---|---|
  | **NB** | `T_text_B` (normal: a letter is one 2 × 2 patch) | body + 64 patches + move = 66 | 669,570 | 160 |
  | **NB2** | 2× (`T_text_B2`) | body + 64 B1 thumbnails + move = 66 | 669,570 | 159 |
  | **NB3a** | 2×, `sum_only = 0`, `uni_pyr = 1` | body + 256 patches + 64 B1 + move = 322 | 702,338 | 16 (3090) |
  | **NB3** | 2×, `sum_only = 0`, `uni_pyr = 4` | body + 256 patches + 85 summaries (B1–B4) + move = 343 | 705,026 | 24 (4090) |

- **Measure: next-letter cross-entropy** (`dev/letter_ce.py`, nats per letter, per value): on the same 800 val
  windows for every run, the two-hot cross-entropy at the token that predicts each letter's first appearance (its
  top-left patch; in NB2 its B1; in NB the patch itself). The training loss is not comparable across layouts: in the
  2× world three of a letter's four patches, and its thumbnail, are copies of what came before.

## Results

### R1. Training loss, NB against NB2 (per value; the same targets)

| ticks | 0–5k | 5–10k | 10–20k | 20–40k | 40–60k | 60–80k | 80–100k |
|---|---|---|---|---|---|---|---|
| NB | 1.639 | 1.237 | 1.230 | 1.170 | 1.092 | 1.042 | 1.012 |
| NB2 | 1.642 | 1.240 | 1.229 | 1.163 | 1.084 | 1.034 | 1.003 |

Within 1% over the whole life.

### R2. Next-letter cross-entropy on the val text

| run | 20k | 30k | 40k | 100k |
|---|---|---|---|---|
| NB | 1.740 | **1.721** | 1.731 | 1.975 |
| NB2 | 1.733 | **1.720** | 1.741 | 1.971 |
| NB3a | 1.747 | **1.725** | 1.744 | 1.967 |
| NB3 | 1.760 | **1.737** | 1.741 | 1.926 |

NB repeats vm-00030's baseline seed 1 (1.718 at 30k, 1.972 at 100k). Every text lifetime overfits after ~30k: the
corpus is ~1M letters, and a 100k-step memory replays it over and over.

## Analysis

- **The premise holds.** NB2's thumbnails carry exactly what NB's patches do, and a model fed only them learns the
  same curve (R1) and predicts the same (R2: 1.720 against 1.721 at the best point).
- **Patch tokens cost compute, not quality.** NB3a has 5× longer windows, and only 64 of its 320 targets per step are
  new letters (the rest are copies), yet it is level with NB at the best point (1.725). The copy targets dilute the
  loss but not the learning of the real ones.
- **The higher levels (B2–B4) slow learning and overfitting.** NB3 is behind at 20k and 30k (1.737 at best) and well
  ahead at 100k (1.926). Its lead at 100k, first read as better prediction, is slower overfitting.
- One seed per arm: differences of ~0.01 are within seed noise; NB3's 0.016 at the best point is borderline.

## Conclusions

1. A level-1 thumbnail of a world drawn at twice the scale is the normal world's token, and the model treats it so:
   the summary pyramid's premise holds in text.
2. More tokens carrying the same information do not hurt; they cost only speed (16–24 ticks/s against 160).
3. Compare text runs at their best checkpoint, not at 100k, where overfitting decides the order.

## Next Steps

- vm-00047: the 2× world with the full pyramid, split into two halves that share the summaries.
- A second seed for NB3 against NB3a, if the higher levels' effect on overfitting matters.

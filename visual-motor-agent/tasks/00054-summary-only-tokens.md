# Task 00054: Tokens From the Summaries Alone, Text at Twice the Scale (experiment vm-00046)

**Date:** 2026-10-29
**Status:** Complete

## Context

- The owner's test of the pyramid's premise (2026-10-29): a level-1 summary's thumbnail is its block, downscaled. If
  the world is drawn at twice the scale (letters too), a level-1 thumbnail is exactly what the normal world's patch
  holds. A model whose tokens are those thumbnails sees the same information, in the same order, as the normal model,
  and should learn as well.
- Text world only. Two runs, compared by their training loss; no frozen tests or evaluations afterwards.

## Design

- **`model.sum_only = 1`** (needs `uni_pyr` ≥ 1 and `sum_thumb = 1`; not with `uni_sum`, `attn_win` or regions): the
  patches are not tokens. A step is [body, the summaries in the pyramid's order, move]; the patches are read only
  to compute the thumbnails. With `uni_pyr = 1` the summaries are B1 row by row (B1(r, c) follows patch
  (2r + 1, 2c + 1) in the full layout, so the order is row-major on B1's grid).
  - `vm_uni_layout` leaves the patches out; everything else reads the slot table (task 00048), so the thumbnails,
    targets, loss, position codes (block centres) and the GPU need no other change.
  - Checkpoint header slot 53. Python: `agent_lib.uni_layout`, `torch_reference.uni_slot_table`.
- **The scaled text world** (`configs/T_text_B2.json`): `env.size = 32`, `env.cell = 4` (every letter a 4 × 4 block
  of its grey, the same 64 letters per page, the same reading order), model `img = 32`, `patch = 2`, `uni_pyr = 1`,
  `sum_thumb = 1`, `sum_only = 1`. A step is 66 tokens of 4 values each, as in `T_text_B`.

## Verification

- `tests/test_summary_only.py`: the layout ([body, 64 B1 row-major, move] for a 16 × 16 patch grid); the scaled
  world's frame is the normal one upscaled 2× (`np.kron`); with the same parameters, a `T_text_B` model on a frame and
  a `T_text_B2` model on its 2× upscale give the same outputs (C++ and GPU); invalid configs refused; the header keeps
  the flag.

## Experiment vm-00046

- 100k ticks, seed 1, full attention, no window:
  - **NB:** `configs/T_text_B.json` (the normal model).
  - **NB2:** `configs/T_text_B2.json` (the 2× world, tokens = B1 thumbnails).
  - **NB3:** the 2× world with the usual tokens (`sum_only = 0`, `uni_pyr = 4`: 256 patches + 85 summaries + 2 = 343
    per step). **NB3a:** the same with `uni_pyr = 1` (256 patches + 64 B1 + 2 = 322): NB2 plus the patch tokens.
- **Measure:** the training loss in `metrics.jsonl` (`uni_ce_patch`, per value). Expected: the same curve within seed
  noise. No evaluations afterwards (the owner's instruction).
- NB3/NB3a's training loss is not comparable (3 of 4 patches per letter and the thumbnails are copies), so all four
  runs are also scored on one measure: the two-hot cross-entropy per value at the tokens predicting each letter's
  first appearance (its top-left patch; with `sum_only`, its B1), on the same 800 val windows (1,600 ticks, 4 worlds,
  seed 0), from `ckpt_00100000.bin` (a scratch scorer, CPU float32 reference forward).

## Results

- `tests/test_summary_only.py` (9): with the same parameters, the normal model on a frame and the B1-only model on its
  2× upscale give the same values and move probabilities (C++ and GPU, within 2e-5: a thumbnail is 4v / 1020 against
  v / 255).
- vm-00046 (seed 1, 100k ticks, 160 ticks/s each): the training cross-entropy per value of the letter tokens.

  | ticks | NB (normal) | NB2 (2× world, B1 only) | difference |
  |---|---|---|---|
  | 0–5k | 1.639 | 1.642 | +0.003 |
  | 5–10k | 1.237 | 1.240 | +0.003 |
  | 10–20k | 1.230 | 1.229 | −0.001 |
  | 20–40k | 1.170 | 1.163 | −0.007 |
  | 40–60k | 1.092 | 1.084 | −0.008 |
  | 60–80k | 1.042 | 1.034 | −0.008 |
  | 80–100k | 1.012 | 1.003 | −0.009 |

  The same curve within 1% throughout (one seed each; the remaining difference is the init: the slot codes sit at the
  B1 block centres instead of the cells, and the random draws differ). The premise holds: a level-1 thumbnail carries
  exactly what the normal world's token does, and a model fed only those learns the same.

- Next-letter cross-entropy per value on the val text (one seed each):

  | run | tokens per step | ticks/s | next-letter CE |
  |---|---|---|---|
  | NB (normal) | 66 | 160 | 1.9751 |
  | NB2 (2× world, B1 only) | 66 | 160 | 1.9707 |
  | NB3a (2× world, patches + B1) | 322 | 16 | 1.9665 |
  | NB3 (2× world, patches + full pyramid) | 343 | 24 | **1.9258** |

  Adding the patch tokens costs nothing in quality (NB3a within 0.5% of NB and NB2), despite 5× longer windows and
  only 64 of 320 targets per step being new letters. The full pyramid is 2.5% better than the normal model: the B2–B4
  summaries (whole lines, the page) help. NB3 is faster than NB3a (24 against 16 ticks/s) because it ran on the 4090
  and NB3a on the 3090. One seed each: the 0.5% differences are within seed noise; the 2.5% likely is not, but needs a
  second seed to be sure.
- The same measure at 20k, 30k and 40k (the text lifetime overfits after ~30k; one seed each):

  | run | 20k | 30k | 40k | 100k |
  |---|---|---|---|---|
  | NB (normal) | 1.740 | **1.721** | 1.731 | 1.975 |
  | NB2 (2× world, B1 only) | 1.733 | **1.720** | 1.741 | 1.971 |
  | NB3a (2× world, patches + B1) | 1.747 | **1.725** | 1.744 | 1.967 |
  | NB3 (2× world, patches + full pyramid) | 1.760 | **1.737** | 1.741 | 1.926 |

  At the best checkpoint NB, NB2 and NB3a are level (1.720–1.725): the scaled world and the patch tokens cost
  nothing. NB3 learns more slowly (behind at 20k and 30k) and overfits more slowly; its lead at 100k is the slower
  overfitting, not better prediction.

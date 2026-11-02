# Task 00074: Sixty-Four Pieces in One Body (experiment vm-00065)

**Date:** 2026-11-01
**Status:** Complete

## Context

- vm-00064 met the rule set before its results (task 00073): at 16 pieces `gain` makes 57.0 ± 6.4 food against 28.7 ± 5.7 at 4 pieces, share 0.91, and `council` holds its food with 0.62 ± 0.19 poison against 1.30 ± 0.45, share 0.98. So the owner's plan of 2026-11-01 goes on to 64.
- At 64 pieces every patch has its own transformer: the view of the world a piece owns is one patch; all else it knows comes through the 21 summary thumbnails. This is the end of the question "can each patch be independent?" with the move's target still shared.

## Design

No code change. `configs/U_default.json` with `model.band_n=64`: 64 transformers of 1 patch each, 24 slots per step, 59.4M parameters in all (`band_cols` 2 is square blocks, so these are 8 by 8).

| arm | pieces | rule |
|---|---|---|
| **G64** | 64 | `gain` |
| **C64** | 64 | `council`: 4,096 scores per tick |

Seeds 1–8, 60k ticks, with MPS. Alone on the 4090, 2,000 ticks: 6.8 ticks/s for gain, 6.1 for council, about 2.8 GB per lifetime. Ten lifetimes would not fit on one GPU, so 8 per GPU: **seeds 1–4 on the 3090, 5–8 on the 4090** (vm-00063 and vm-00064 split them 3 and 5). 16 lifetimes share about 30 ticks/s, by the throughput of vm-00064, so about 9 hours.

**Control:** vm-00064's G16 and C16, and vm-00063's NA and CO, not rerun: nothing a lifetime runs has changed.

## Measures

As task 00073: food and poison per 1k over the last 20k ticks, mean and 95% interval over seeds; differences seed by seed against the same rule at 16 and at 4 pieces; food share; mean hunger; proposal size; smoothness; win share; patch, body and move CE; ticks/s. `eval_pieces.py --ring` on frozen weights for the size of moves made and their aim at near food, with `--rule random` beside `gain`.

## The reading, set before the results

64 pieces **survive** if, for at least one rule, the interval of the food difference against the same rule at 16 pieces includes zero or is above it, and its food share stays at 0.85 or more. If not, 16 is the best split known and the README says what broke.

## Verification

The existing suite (no code change). `lifetime.py` ran both arms for 2,000 ticks on the GPU.

## Results

The last 20k ticks, per 1k, eight seeds:

| arm | food | poison | food share | mean hunger |
|---|---|---|---|---|
| G64 (`gain`) | 56.5 ± 3.9 | 9.16 ± 1.05 | 0.86 | 0.076 |
| C64 (`council`) | 29.8 ± 8.6 | 0.65 ± 0.69 | 0.98 | 0.052 |
| G16 | 57.0 ± 6.4 | 5.70 ± 0.61 | 0.91 | 0.073 |
| C16 | 34.7 ± 7.0 | 0.62 ± 0.19 | 0.98 | 0.045 |

The reading is met for both rules: G64 − G16 food −0.5 ± 6.8 at share 0.86, C64 − C16 −4.9 ± 8.0 at share 0.98. 64 pieces survive, with no gain over 16 for about 4 times the compute, and 16 stays the split to use. What gain adds on poison over a random piece goes −1.8, 0.0, +2.7 at 4, 16 and 64 pieces. C64's food separates by GPU (38.2 on the 3090, 21.3 on the 4090), which seeds and GPUs being assigned together leaves unexplained; G64 does not. See the [README](../00065-sixty-four-pieces/README.md).

# Task 00073: Sixteen Pieces in One Body (experiment vm-00064)

**Date:** 2026-11-01
**Status:** Complete

## Context

- The owner asked what it would take for each patch to be independent, movement not shared (2026-11-01). The move made must stay an input: the view is egocentric, so a piece that does not know it cannot predict its next frame. The body is one. What can be split is who proposes.
- `model.band_n` already makes one transformer per block of patches, with no code change: 78.7 ticks/s at 4 pieces, 25.0 at 16, 6.8 at 64, alone on the 4090 (1,000 ticks; speed only). A piece sees its own patches and the 21 summary thumbnails.
- The owner chose to start at 16 pieces and go to 64 only if foraging survives. Unsharing the move's training target (a piece learns the move only when it made it) stays a separate experiment.

## Design

No code change. `configs/U_default.json` with `model.band_n=16`: 16 transformers of 4 patches each, 27 slots per step against 39, 14.8M parameters in all against 3.7M.

| arm | pieces | rule |
|---|---|---|
| **G16** | 16 | `gain` |
| **C16** | 16 | `council`: 256 scores per tick; 22.8 ticks/s alone, against gain's 25.1 |

C16 is here because gain favours the piece that misses a danger, and more pieces make that worse: with gain alone, a failure could not be told from the rule's. Seeds 1–8, 60k ticks, 16 lifetimes with MPS, seeds 1–3 on the 3090 and 4–8 on the 4090, about 1.1 GB each at the start.

**Control:** vm-00063's NA (`gain`) and CO (`council`), the same settings and seeds at 4 pieces, not rerun: nothing a lifetime runs has changed since.

## Measures

Food and poison per 1k over the last 20k ticks, mean and 95% interval over seeds; differences seed by seed against the same rule at 4 pieces; food share; proposal size; smoothness; win share per piece; patch, body and move CE; ticks/s. The same pieces' win shares under council are bookkeeping (vm-00063).

## The decision, set before the results

Go to 64 pieces if, for at least one rule, the interval of the food difference against the same rule at 4 pieces includes zero or is above it, and its food share stays at 0.85 or more. Otherwise stop at 16 and report what broke.

## Verification

The existing suite (no code change). `lifetime.py` ran both arms for 1,000 ticks on the GPU, with 16 win shares logged.

## Results

The last 20k ticks, per 1k, eight seeds:

| arm | food | poison | food share | mean hunger |
|---|---|---|---|---|
| G16 (`gain`) | 57.0 ± 6.4 | 5.70 ± 0.61 | 0.91 | 0.073 |
| C16 (`council`) | 34.7 ± 7.0 | 0.62 ± 0.19 | 0.98 | 0.045 |
| NA (`gain`, 4 pieces) | 28.7 ± 5.7 | 3.90 ± 0.47 | 0.88 | 0.089 |
| CO (`council`, 4 pieces) | 32.8 ± 10.4 | 1.30 ± 0.45 | 0.96 | 0.052 |

**The decision:** met for both rules (G16−NA food +28.3 ± 9.4, C16−CO +1.9 ± 12.3; shares 0.91 and 0.98), so 64 pieces follow: task 00074, vm-00065. Full results in `../00064-sixteen-pieces/README.md`.

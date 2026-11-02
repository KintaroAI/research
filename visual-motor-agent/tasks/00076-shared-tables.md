# Task 00076: One Table for All Pieces (experiment vm-00067)

**Date:** 2026-11-01
**Status:** Complete (pending tag)

## Context

- vm-00064: at 16 pieces `gain` doubled its food. A piece drawn at random makes 53 food, so the rise is in the pieces.
- The owner's explanation (2026-11-01): each transformer has its own table of 2,048 rows, so more pieces are more tokens: 128 rows per patch at 4 pieces, 512 at 16, 2,048 at 64. It may be right. Against it so far: council has the same rows and its food did not move, vm-00058's 0.4× to 2.8× parameters changed nothing (but by width, not rows), and patch CE got worse.
- The direct test takes the tables away and keeps all else that 16 pieces change: fewer slots per piece, a larger share of thumbnails, more parameters in the transformers.

## Design

`online.tie_tables` (GPU, `online.pieces`): `""` off; `"in"`: `uni_in_w`, `uni_in_b`; `"inout"`: those and `uni_out_w`, `uni_out_b`. All pieces start from piece 0's tables, and after every update the tied tables become the mean over the pieces (`vmc_tie`, a kernel, no copy to the host). Each piece keeps its own Adam state, so this is the mean step of S optimizers on one table, not one optimizer: close to one shared table, not the same. A shared buffer would have needed a new ownership of parameters in `Model`, for a test.

With one table the body has 2,048 rows for its 64 patches: 32 per patch, fewer than at 4 pieces. So this is a stronger cut than undoing the split.

## Verification

`tests/test_shared_tables.py`: `vmc_tie` copies and averages only the named range; a lifetime's pieces have equal tables at every checkpoint and different other weights; off changes nothing; invalid settings; a tied lifetime on the GPU learns.

## Experiment vm-00067

- **TB:** `configs/U_default.json`, `model.band_n=16`, `gain`, `online.tie_tables=inout`. Seeds 1–8, 60k ticks, on the 4090 with MPS beside vm-00066. Controls: G16 (vm-00064), NA (vm-00063), not rerun.
- **`gain`**, because that is where the food rose. **`"in"` is only run if TB loses food**, to find which table it was: if both tables tied cost nothing, an arm with one tied says nothing new.

## The reading, set before the results

Food over the last 20k ticks, against G16 and against NA, seed by seed:

- **More tokens explain it:** TB − G16 is below zero beyond its interval, and TB − NA includes zero.
- **They do not:** TB − G16 includes zero or is above it.
- **In part:** TB is below G16 and above NA, both beyond their intervals.

## Results

Eight seeds, last 20k ticks, per 1k: TB 49.3 ± 10.8 food, 3.49 ± 0.60 poison; G16 57.0 ± 6.4, 5.70 ± 0.61; NA 28.7 ± 5.7, 3.90 ± 0.47. TB − G16 −7.7 ± 13.4 includes zero, so **by the reading more tokens do not explain the rise**; TB − NA is +20.6 ± 12.7. The intervals still leave room for the tables to account for part of it, up to about three quarters. One table also lowers poison (−2.21 ± 0.83), patch CE (−0.010 ± 0.003) and move CE (−0.232 ± 0.066), and gain's guard against poison returns (frozen, −1.1 ± 0.9 against a random piece). The `in` arm did not run. Full results in `../00067-shared-tables/README.md`.

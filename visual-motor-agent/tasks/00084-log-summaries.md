# Task 00084: Log Summaries in the Visual World (experiment vm-00073)

**Date:** 2026-11-02
**Status:** Complete

## Context

- The owner (2026-11-02): "We need to switch to log summaries by default - lets stop current run and run previous (with default world size) but with log summaries enabled. And I want to compare those 2 runs to understand how log patches affect overall result. We will come back to current run after that".
- Summaries by neighbourhood exist since vm-00050 (`model.band_win = w`, task 00058): a piece keeps a summary only if it is within w cells of the piece at that level, sharp near and coarse far. With pieces of a fixed size this is O(log N) summaries per piece. Every run since has read every summary.
- It was measured once to the end, in the 2x text world: cross-entropy 1.895 against 1.703, one seed, 40k ticks. In the visual world never: vm-00051's QW was stopped at 6k ticks to make room.
- vm-00072, the big world, is paused. It would read 85 summaries per quarter at 3.3 ticks/s, or 39 with `band_win = 1`.

## Design

**One change:** `model.band_win = 1` in vm-00071's experiment: the same 64 x 64 world, K1, K4, K16, six seeds, 60k ticks, seeds 1–3 on the 4090 and 4–6 on the 3090, nine lives on each under MPS. The control is vm-00071 itself, not rerun: each arm against the same k there, paired by seed number.

`configs/U_default.json` now has `band_win = 1`, the owner's decision. `configs/U_big.json` is left as it is until vm-00072 is taken up again. vm-00071 and earlier are repeated from their tags.

**How much it changes here.** Each quarter reads 14 summaries against 21: 9 of the 16 at the finest level, all of the 4 and the 1 above it. A step is 32 tokens against 39, 18% fewer. A 4 x 4 grid of cells has little "far", so this is a mild test: a quarter loses only the far row and column of the finest level. A loss here is a warning for larger worlds, and no loss here does not promise none there.

A life alone on the 4090: 83.6 ticks/s against 76.

## The readings, set before the results

Each is an arm against the same k in vm-00071, seed by seed, 95% interval on six seeds, frozen tests as in vm-00071.

1. **Plumbing:** 32 tokens per step, the ring and the tools work. **Met:** `uni_layout` gives 32; `test_summary_windows`, `test_ring` and `test_age_key` pass (60); a life of 1,500 ticks and `eval_hunger.py chain` ran.
2. **Food and poison hold:** over the last 20k ticks, food is not below by more than its interval and poison not above by more than its interval.
3. **Reach holds (K4):** aim at gaps of 12–24 pixels and the lead over blind and a perfect grab are not below by more than their intervals.
4. **Hunger holds:** mean hunger, frozen, is not above by more than its interval.

Reported beside them: ticks/s and GPU memory.

## Does the cost grow in line with the world?

The owner (2026-11-02): "small world x, large world is 4x - we need to compare the speed of small wold T transformers count vs large world 4T transformers count and whether speed raises linarly for full summaries vs log summaries".

Each doubling of the side takes the area, the things in the world and the transformers up 4 times, so a transformer keeps its 16 patches. `dev/bench_scale.py`, after the lives and the frozen tests, alone on the 4090, without MPS, a short life timed after its memory has warmed up, by its parts.

| side | transformers | foods / poisons / dots | tokens per step, every summary | tokens per step, log summaries |
|---|---|---|---|---|
| 64 | 4 | 1 / 2 / 8 | 39 | 32 |
| 128 | 16 | 4 / 8 / 32 | 103 | 36–48 |
| 256 | 64 | 16 / 32 / 128 | 359 | 40–57 |

**The reading:** the time of the model (acting and updating) per tick, against 4 times the side before it. Linear is 4 times at each step. Every summary should go well past that, since each transformer's tokens grow with the world; log summaries should stay near it, since they grow by a few tokens a level. The 256 row is there for a second point on the slope, and every summary may not fit in memory or time there; that would be a result too.

Transformers run one after the other in a process, so this is total compute, as on one GPU. Pieces on separate GPUs would divide it.

## Implementation

- [x] `configs/U_default.json`: `band_win = 1`
- [x] `00073-log-summaries/`: README, Makefile
- [x] `dev/bench_scale.py`
- [x] Runs, frozen tests, write-up: all readings met, no loss could be seen, in a mild test with wide intervals (`../00073-log-summaries/README.md`)
- [x] Scaling benchmark, on an idle GPU: per quadrupling 5.2 and 5.7 times with log summaries, 10.5 and 26 with every summary

## Verification

Reading 1, on the GPU, before the runs.

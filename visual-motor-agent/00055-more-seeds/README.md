# vm-00055: More Seeds for the Best Arms

**Date:** 2026-10-31
**Status:** Complete
**Source:** `exp/vm-00055` (code: `visual-motor-agent/dev/`)
**Outputs:** `~/data/research/visual-motor-agent/exp_00055/`
**Task plan:** [`../tasks/00063-more-seeds.md`](../tasks/00063-more-seeds.md)

## Goal

Three seeds could not separate QN, QNB and QNA (vm-00054), and misled vm-00043 and vm-00044. With the speed from task 00062, give the
best arms eight seeds each: what is real?

## Method

Five more seeds (4–8) of QN (gain), QNA (gain, age key, ring) and QCA (council, age key, ring), 60k ticks, the no-overlap world, added
to seeds 1–3 of vm-00052 and vm-00054. Fifteen lifetimes under MPS on the 3090, in ~2.5 hours. Intervals: t, 95%, over eight seeds.

## Results

### R1. The last 20k ticks, per 1k

| arm | food | poison | food share | smoothness |
|---|---|---|---|---|
| QN | **41.4 ± 8.0** | 3.3 ± 0.6 | 0.93 | 0.28 |
| QNA | 34.7 ± 10.2 | 3.5 ± 0.6 | 0.91 | 0.29 |
| QCA | 28.8 ± 9.1 | **1.5 ± 0.4** | 0.95 | 0.60 |

### R2. Food by seed (1–8), and how early it shows

| arm | last 20k | at 10k–20k | correlation |
|---|---|---|---|
| QN | 52.2, 31.9, 46.5, 46.0, 47.6, 29.1, 29.4, 48.2 | 30.4, 25.5, 40.0, 44.2, 40.7, 22.8, 27.6, 32.0 | 0.69 |
| QNA | 21.9, 40.5, 59.2, 21.1, 31.8, 35.6, 29.2, 38.5 | 17.3, 33.9, 47.7, 15.8, 27.6, 29.0, 22.5, 40.2 | 0.95 |
| QCA | 18.3, 36.3, 32.9, 41.6, 18.6, 26.4, 14.4, 41.7 | 22.8, 36.0, 28.1, 39.6, 16.7, 28.1, 18.6, 50.7 | 0.92 |

## Analysis

- **Council's poison gain is real:** 1.5 against 3.3, intervals apart. Its food cost is probably real too: 28.8 against 41.4, intervals just touching.
- **The age key and the ring:** 34.7 against 41.4 food, intervals overlapping well. A small cost is possible; poison is the same. Eight seeds
  do not settle it. The frozen test (vm-00054) puts the ring itself near no cost, so if the cost is real it is more likely the age key's.
- **Seeds remain the largest effect:** 14.4 to 59.2 within arms. They settle early: 10k–20k food predicts the last 20k (0.69–0.95). What differs
  in a bad seed is decided in the first 20k ticks, and is not known yet.

## Conclusions

1. Council cuts poison by more than half, at about 30% of the food.
2. QN remains the best forager.
3. A seed's fate shows by 20k ticks, so bad runs can be stopped early, and their cause studied.

## Next Steps

- What differs early between a good seed and a bad one.
- A selection of good starts: run many seeds to 20k, continue the best.

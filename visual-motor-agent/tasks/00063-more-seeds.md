# Task 00063: More Seeds for the Best Arms (experiment vm-00055)

**Date:** 2026-10-31
**Status:** Complete

## Context

- vm-00054 could not separate QN, QNB and QNA: seeds run from 21.9 to 59.2 food, wider than the differences between arms. Three
  seeds also misled the comparisons of vm-00043 and vm-00044.
- Speed now allows more: six lifetimes under MPS run 2.5× faster than this morning.

## Experiment vm-00055

- Five more seeds, 4–8, 60k ticks, the no-overlap world (`configs/U_touch_ego_c2.json`, quadrants, `uni_pyr = 3`,
  `sum_thumb = 1`), all under MPS on the 3090:
  - **QN:** gain (vm-00052).
  - **QNA:** gain, age key, ring (vm-00054).
  - **QCA:** council, age key, ring (vm-00054).
- With seeds 1–3 of vm-00052 and vm-00054: eight seeds per arm. The update's arithmetic is unchanged since those runs, so
  the seeds are comparable; GPU lifetimes are not bit-reproducible in any case.
- **Measure:** food and poison per 1k over 40k–60k, per seed, means with a 95% interval (t) over the eight seeds; share of seeds
  above 30 food; `smooth`.
- **Questions:** Does the age key plus ring cost food (QNA against QN)? How much poison does council remove, at what food?
  Are bad seeds (below 25 food) a stable fraction, and do they share something early (at 10k–20k)?

## Write-up

[`../00055-more-seeds/README.md`](../00055-more-seeds/README.md). QN 41.4 ± 8.0 food, 3.3 ± 0.6 poison; QNA 34.7 ± 10.2, 3.5 ± 0.6; QCA 28.8 ± 9.1, 1.5 ± 0.4.

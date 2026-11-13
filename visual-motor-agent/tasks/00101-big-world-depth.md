# Task 00101: The Value and K4 at Eight Layers in the Big World (experiment vm-00090)

**Date:** 2026-11-12
**Status:** Complete (2026-11-13; results in [`../00090-big-world-depth/README.md`](../00090-big-world-depth/README.md))

## Context

- **The small world is nearly used up as a test.**
  - K4D8T (vm-00089) eats about every 7 ticks. Hunger rises 0.001 per tick and a meal resets it, so the body lives below hunger .01.
  - The next food is about 15 px away after a meal, beyond 24 px only 15% of the time (vm-00072).
  - K4D8T's 137 food per 1k matches a scripted walker that turns to food within 16 px (140). Nothing in this world asks for far reach, a chain of moves, or the value.
- **The big world** (`configs/U_big.json`, vm-00072): 128 × 128, 4 foods, 8 poisons, 32 dots; the same sizes, moves and hunger as the default, so the same density.
  - After a meal the next food is 24.6 px away, beyond 24 px half the time.
  - Scripted walkers: always chasing 170.6, reach 24 px 105.4, reach 16 px 67.7, blind 19.5.
  - vm-00072's K4 (2 layers, tolerance 0.005, 6 seeds): food 42.1, aim past 24 px .077, mean hunger .048.
- **The owner (2026-11-12):** the world is the limit; make it larger first (A), then a world larger than the view (C). Four transformers per model, as now.

## Design

| Arm | Label | Model | World | Seeds |
|---|---|---|---|---|
| **KB** | mean hunger over the next 4 ticks (K4) | 4 quarter transformers × 8 layers, `instinct_tol` 0.002 (K4D8T) | `U_big.json` | 1–6 |
| **VB** | value, γ 0.9, n 4, fresh tail (V90R) | the same (V90RD8T) | `U_big.json` | 1–6 |

- **Only the world changes** from vm-00089's K4D8T and V90RD8T: `--config configs/U_big.json --set online.future_k=4 --set model.n_layer=8 --set online.instinct_tol=0.002`, and for VB `--set online.value_gamma=0.9 --set online.value_refresh=1`.
  - Each quarter transformer now covers 64 × 64 px. It reads 105 tokens per step, against 32 in the small world.
- **References, not rerun:**
  - vm-00072's K4: 2 layers, tolerance 0.005, the same world and seeds 1–6 (seeds 1–3 on the 4090, 4–6 on the 3090);
  - vm-00072's walkers;
  - vm-00089's small-world K4D8T and V90RD8T.
- **Each seed's two lives on one card,** as before.
- **Timing** (1,000-tick lives, 6 per card under MPS, 2026-11-12 23:16–23:30; scratchpad `t90/`):
  - the 4090: KB 2.6 ticks/s, VB 2.3, about 15 ticks/s for the card;
  - the 3090: KB 1.3, VB 1.2, about 7.5 for the card;
  - about 2.2 GB of GPU memory per life (14 GB for 6).
- **Six seeds, weighted to the faster card:**
  - seeds 1–4 on the 4090 (8 lives, about 19 GB), seeds 5–6 on the 3090 (4 lives);
  - both cards should finish in about 9 h, then the frozen tests.
  - Twelve seeds would take about 18 h.
- **Seeds 7–12 follow as a second wave only if** VB − KB's lived food is positive in at least 4 of 6 seeds but within its interval (set now, before the results).
- **No code change.**

## Measures

- **Lived, 40–60k:** food, poison, mean hunger, overrides, move CE, smoothness; VB against KB, paired by seed.
- **Frozen:**
  - `eval_hunger.py chain`: food, aim by the gap (here past 24 px matters most), lead over a blind walk with a perfect grab;
  - `eval_hunger.py value`;
  - `eval_hunger.py reach`.
- **The world's demand:** KB's and VB's mean hunger, and the share of ticks with the nearest food beyond 24 px, against the small world's.

## The readings, set before the results

1. **Plumbing:** all lives run 60k ticks with `img` 128, 8 layers and `instinct_tol` 0.002, without errors.
2. **The world asks more:** KB's lived mean hunger is at least twice K4D8T's in the small world (.015), and its food is below the always-chasing walker's 170.6.
3. **Depth and the tolerance carry over:** KB's lived food is above vm-00072's K4 (42.1), paired by seed over seeds 1–6, by more than its interval.
4. **The value pays when food is far (the main one):** VB's lived food at 40–60k is above KB's, paired by seed, by more than its interval.
5. **Reach:** VB's aim past 24 px and its lead over the grab are above KB's, paired, by more than their intervals.
6. **The value holds:** every VB seed meets the bar (abs bias at most .0141, correlation above .5, median candidate spread at least .0005).
7. **Poison is still seen:** VB's lived poison is not above KB's by more than its interval.

**What follows:**
- **If 4 holds:** the value pays when the world asks for reach. Keep it, and take it into C (a world larger than the view).
- **If 4 fails while 3 holds:** K4 at 8 layers stays the label. The value is put aside, and C is run with KB's settings.
- **If 3 fails:** depth and the tolerance do not carry to a larger view. Look at the model's scale against the world's (16 transformers, task to come) before C.

## Implementation

- [x] Timing lives (scratchpad `t90/`).
- [x] `00090-big-world-depth/` (README, Makefile); the queue.
- [x] Runs (2026-11-12 23:31 to 2026-11-13 09:14), frozen tests (to 09:39), write-up.

**Outcome:** all seven readings met. VB − KB food +22.9 ± 10.6 (6 of 6), so the value is kept and taken into C. The second wave was not triggered.

## Verification

- Every run's `info.json` has `env.size` 128, `model.n_layer` 8, `online.instinct_tol` 0.002.
- `eval_hunger.py show` over vm-00090's files gives KB and VB paired by seed; vm-00072's chain files with them give K4 at 2 layers beside KB.

# Task 00100: The Instinct's Tolerance at the Value's Scale (experiment vm-00089)

**Date:** 2026-11-12
**Status:** Complete (pre-check 15:22–15:57, runs 16:41 to 21:47 on 2026-11-12; written up 2026-11-12)

## Context

- **vm-00088 (task 00099):** over 12 seeds, the fresh value at eight layers (V90RD8) is calibrated like K4D8 (bias −26% of the realized label, against −25%) and aims further (+.061 ± .053 at 12–24 px), but it eats no more: +4.2 ± 12.4 food per 1k, and its lead over a blind walk with a perfect grab is K4D8's.
- **The rule set before vm-00088:** if the value holds but brings no food, the next step is the label's scale, not more seeds.
- **Why the scale:**
  - Most of what one move changes is when the next food is touched, a tick sooner or later.
  - K4's label is the mean hunger of the next 4 ticks, so each of those ticks weighs 1/4. V90's label weighs the next tick (1 − γ) = 0.1.
  - The same move then changes V90's label about 0.4 times as much as K4's.
  - The instinct keeps its first sample unless another candidate is better by more than `online.instinct_tol` = 0.005. That tolerance was set for K4-scale labels, and vm-00085 kept it for every arm.
  - So under the value, many food moves fall within the tolerance and are not taken.
- **Not changed here:** the bins and the two-hot loss. One normalized value range stays (the owner's preference, task 00096).
- **Not needed:** an advantage. Subtracting the mean candidate does not change which candidate is lowest. What matters is the tolerance against the candidates' differences.

## Pre-check (frozen, on vm-00086/88's checkpoints)

Before any new life, on the 24 final checkpoints of K4D8 and V90RD8:
- `eval_hunger.py reach`: on ticks with food one move away, how often the predicted drop clears the tolerance, and how often the food move is lost to it.
- `eval_hunger.py told`: frozen food with the tolerance at 0, against the real one.

**What would support the hypothesis:**
- V90RD8 loses more food moves to the tolerance than K4D8;
- its frozen food gains more than K4D8's at tolerance 0.

**Results** (2026-11-12, 15:22–15:57; scratchpad `p89/`; frozen, so they understate online learners):

Food one move away (`reach`), by the body's real hunger, 12 seeds each:

| | K4D8 | V90RD8 |
|---|---|---|
| Predicted drop, hunger .01–.02 | .0039 | .0015 |
| Predicted drop, hunger .02–.04 | .0070 | .0024 |
| Drop above 0.005, hunger .01–.04 | .22–.40 | .07–.15 |
| The lowest candidate is a food move | .84–.89 | .64–.76 |
| Eaten when in reach | .86–.92 | .62–.79 |
| In reach (share of ticks) | .09–.12 | .12–.21 |

- **The drop is squashed as predicted:** V90RD8's predicted drop is .34–.6 times K4D8's.
- **It clears the tolerance far less often.**
- **It eats less of what it reaches.** It reaches food more often, which evens out the total.

Tolerance 0 against the real 0.005 (`told`), paired by seed:

| | Food | Poison |
|---|---|---|
| K4D8 | +12.7 ± 4.7 | +1.68 ± 0.63 |
| V90RD8 | +18.3 ± 3.7 | +0.36 ± 0.65 |
| V90RD8's gain − K4D8's | +5.6 ± 4.9 (11 of 12) | −1.31 ± 0.88 |
| V90RD8 at 0 − K4D8 at 0.005 | +20.6 ± 14.8 (10 of 12) | +0.49 ± 1.48 |
| V90RD8 at 0 − K4D8 at 0 | +8.0 ± 12.4 | −1.19 ± 1.75 |

- **The tolerance costs the value more than K4,** in 11 of 12 seeds.
- **At tolerance 0, K4D8 eats more but touches more poison.** For K4, the tolerance also filters noise that leads to poison. The value's food gain comes without poison.
- **The hypothesis holds on frozen models.** The lived test is still needed: a lower tolerance also changes what each life learns from. 0.002 sits between the two settings.

## Design

| Arm | Label | `online.instinct_tol` | Layers | Seeds |
|---|---|---|---|---|
| K4D8 | mean hunger over the next 4 ticks | 0.005 | 8 | vm-00086/88, not rerun |
| V90RD8 | value, γ 0.9, n 4, fresh tail | 0.005 | 8 | vm-00088, not rerun |
| **K4D8T** | mean hunger over the next 4 ticks | 0.002 | 8 | 1–12, new |
| **V90RD8T** | value, γ 0.9, n 4, fresh tail | 0.002 | 8 | 1–12, new |

- **The tolerance, 0.002:** 0.005 × (1 − γ) · k = 0.005 × 0.1 × 4. This is the tolerance at the value's scale for the next tick.
- **K4D8T is the control.**
  - A lower tolerance also makes K4 override more.
  - Without K4D8T, a gain could be the tolerance alone, not the value.
- **Everything else is vm-00088's:**
  - four quadrant transformers of 8 layers, gain, log summaries, the ring, the egocentric touch world, `online.future_k = 4`, 60k ticks, under MPS;
  - every seed's four lives on one card: seeds 1–6 on the RTX 4090, 7–12 on the RTX 3090.
- **24 new lives, 12 per card:**
  - about 2 h on the 4090 and 3.5–4 h on the 3090 (vm-00086's 12 lives ran at 8.1 ticks/s on the 4090; vm-00088's 12 at 4–7 on the 3090);
  - then the frozen tests, without MPS, two per card.
- **No code change:** `--set online.instinct_tol=0.002`. The frozen tests read the tolerance from each run's config.

## Measures

- **Lived, 40–60k:** food, poison, mean hunger, overrides, move CE, smoothness. Pairs by seed, 12 seeds:
  - V90RD8T − K4D8T: the value at the scaled tolerance;
  - V90RD8T − V90RD8: the tolerance, with the value;
  - K4D8T − K4D8: the tolerance, with K4.
- **Frozen:**
  - `eval_hunger.py value`: bias, correlation, spread;
  - `eval_hunger.py chain`: food, aim by the gap, lead over the grab;
  - `eval_hunger.py reach`: food moves lost to the tolerance.

## The readings, set before the results

1. **Plumbing:**
   - all 24 lives run 60k ticks;
   - every run's `info.json` carries 0.002;
   - the T arms override more than their 0.005 counterparts, paired.
2. **The value still holds:** every V90RD8T seed meets the bar (abs bias at most .0141, correlation above .5, median spread at least .0005).
3. **The food moves get through:** in `reach`, V90RD8T eats a larger share of the food in reach than V90RD8 (pooled over 12 seeds; the pre-check gives .62–.79).
4. **More food (the main one):** V90RD8T's lived food at 40–60k is above K4D8T's by more than its interval, over 12 seeds.
5. **The tolerance helps the value more than K4:** (V90RD8T − V90RD8) − (K4D8T − K4D8) is above 0, by seed. This is reported with its interval; with 12 seeds it is expected to be wide.
6. **Poison is still seen:** neither T arm's lived poison is above its 0.005 counterpart's by more than its interval.

**What follows:**
- **If 4 holds:** the value pays once the instinct listens at its scale. The next step is the value's reach (a chain of moves, γ 0.97 at the scaled tolerance).
- **If both T arms gain about equally (4 fails, 5 is near 0):** the tolerance was too high for both labels. 0.002 becomes the default, and the value still does not pay.
- **If neither gains:** the scale is not what stops the value. The bootstrapped value is put aside, with K4D8 as the default.

## Implementation

- [x] The pre-check (scratchpad `p89/`): `reach` and `told` on the 24 checkpoints.
- [x] `00089-value-scale/` (README, Makefile with `precheck`); the queue (scratchpad `q89.sh`).
- [x] Runs, frozen tests, write-up.

## Verification

- `eval_hunger.py show` over vm-00086/88's files and vm-00089's gives the four arms, 12 seeds each, paired by seed.
- Each new run's `info.json` has `config.online.instinct_tol = 0.002`.

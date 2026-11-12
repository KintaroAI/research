# Task 00097: Eight Layers in Each Quadrant, With and Without the Value (experiment vm-00086)

**Date:** 2026-11-11
**Status:** In progress (queued 2026-11-11 23:00)

## Context

- **vm-00085 (task 00096):** a bootstrapped value with γ 0.9 (V90) on the default model, which has four quadrant transformers of 2 layers each.
  - Against K4, V90 aims further (aim at 12–24 px +.074 ± .042, 6 of 6 seeds) and eats more (+17.2 ± 22.9 per 1k, 5 of 6 seeds), with no lead over a blind walk with a perfect grab.
  - Its value fails the calibration bar: it predicts 30–55% less hunger than follows, against 14–25% for K4. The bar is an absolute bias of at most twice K4's (.0141), and V90 misses it in 3 of 6 seeds.
- **vm-00084 (task 00095):** in a single transformer, 8 layers forage better than 2 (+8.4 ± 6.1 food per 1k, 6 of 6 seeds). Width does not help. That reading was not set in advance, and depth was never tried in the four-quadrant default.
- **The owner (2026-11-11), after the videos of both:** V90, with each quadrant transformer 8 layers deep.

## Design

- **Depth only:** `--set model.n_layer=8`, as vm-00084's D8. d_model 128, 4 heads, d_ff 512. Nothing else changes: four quadrant transformers, gain, log summaries, the ring, the egocentric touch world, `online.future_k = 4`, six seeds, 60k ticks.
- **A 2 × 2, by label and depth.** vm-00085's 2-layer K4 and V90 are reused, not rerun, as vm-00084 reused vm-00082's C8.

| | 2 layers (vm-00085) | 8 layers (here) |
|---|---|---|
| K4's label | K4 | **K4D8** |
| Value, γ 0.9 (phase 1) | V90 | **V90D8** |

- **The value stays as in vm-00085 (phase 1).** V̂ is the winner's prediction, stored while acting, not refreshed on replay. A result is then a change in depth alone. Phase 2 (the mean of the four as V̂, refreshed on replay) comes next, as vm-00087, on whichever depth this one favours.
- **12 lives** on the RTX 4090 under MPS, once 00021's seed-1 half there has finished. The 3090 carries 00021's seed 2 meanwhile.
  - vm-00085's controls ran on the 3090. GPU lives are not bitwise reproducible on one card either, so arms are compared over seeds, as before.
  - Expected time: about 3–4 h under MPS (vm-00085's 2-layer lives ran at 7.7 ticks/s, 18 to the slower card; depth costs about 3 times the compute), longer without it.

## Measures

As vm-00085:

1. **Lived, 40–60k:** food, poison, mean hunger, the instinct's override share, move cross-entropy. Paired by seed: K4D8 against K4, V90D8 against V90 (across experiments, depth), and V90D8 against K4D8 (the value, at 8 layers).
2. **Frozen `eval_hunger.py chain`:** seen, blind, blind with a perfect grab, the lead over the grab, aim by the gap to the food.
3. **Frozen `eval_hunger.py value`:** the predicted label of the move taken against the realized one; bias, correlation, candidates' spread.

## The readings, set before the results

1. **Plumbing:** every life runs 60k ticks with 8 layers in each of its four transformers.
2. **Depth helps four quadrants:** K4D8's lived food at 40–60k is above K4's by more than its interval.
3. **Depth helps the value hold:** V90D8's frozen abs bias is below V90's, paired by seed, by more than its interval, and V90D8 meets vm-00085's bar (abs bias at most .0141, correlation above .5) in every seed.
4. **The value's reach survives depth:** V90D8's aim at 12–24 px is above K4D8's by more than its interval.
5. **A chain pays:** V90D8's lead over blind with a perfect grab is above K4D8's by more than its interval.
6. **The combination is best:** V90D8's lived food is above both V90's and K4D8's by more than their intervals.
7. **Poison is still seen:** neither 8-layer arm's lived poison is above its 2-layer counterpart's by more than its interval.

## Implementation

- [x] No code: `model.n_layer` already sets the depth of every quadrant transformer.
- [x] Plumbing lives: 600 ticks of K4D8 and V90D8 beside a 2-layer K4, on the 4090 shared with 00021: 2,159,138 parameters per transformer (969,506 at 2 layers), V90D8's label and prediction .33–.39, no NaN. Speed not measurable: 00021 held the card at 100% (1.5–1.9 ticks/s for every arm).
- [x] `00086-value-depth/`: README, Makefile (`runs`, `frozen`, `show`).
- [x] The queue (2026-11-11 23:00): wait for 00021's seed-1 runs to end, then the 12 lives (under MPS if its daemon is up), then the frozen tests.
- [ ] Runs, frozen tests, write-up.

## Verification

- The checkpoint and the log state 8 layers and the parameter count.
- The plumbing lives: no NaN, `value_label` and `value_pred` within 0–1 in V90D8.

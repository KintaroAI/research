# Task 00065: The Delayed-Cue World (experiment vm-00057)

**Date:** 2026-10-31
**Status:** In progress

## Context

- vm-00056: a dream fed back as thought stays alive but collapses to black and white, and holds no linearly readable food place. Food
  cannot tell whether thought is a memory: the world half never requires one.
- The owner chose the delayed-cue world, with the grey fix as an arm (2026-10-31). This world has one answer a window of 8 ticks
  cannot give, so a share above chance is memory, and nothing else.

## Design

### A trial (`env.cue`, `world/cue_world.py`, a CircleWorld)

1. **Flash:** a 16 × 16 block at the top left of the view for `flash` ticks, solid or checker (2 × 2 cells), at random: the sample.
   It fills one B1 cell, so its thumbnail keeps the pattern, and both transformers see it.
2. **Delay:** `delay_min`..`delay_max` ticks with no circles and no block; the dots go on.
3. **Choice:** one food and one poison, at `dist` px on opposite sides of the cursor, in a random direction, not moving. **The food looks
   like the sample:** a solid sample makes the food solid and the poison checker, a checker sample the reverse.
4. A touch of either, or `timeout` ticks, ends the trial and the next flash starts.

The two circles are alike but for the pattern, so without the sample a choice is a coin. Food and poison keep their usual meaning in the
world; only the drawing swaps. `hold`: the block stays on through delay and choice (the control).

- Hunger: `h_rate` 0.001, relief 0.3, poison 0.2: at chance, hunger stays off the cap (a cap hid hazards in vm-00014).
- **`online.thought_decode`:** `pair` (as vm-00056) or `mean`, the grey fix: pair decoding snaps a value to a bin, the mean of the bins
  does not, so a dream should not sharpen the last. `thought_extreme` logs the share of pixels near black or white.

### Measures

Per 1k ticks: `cue_correct` = food / (food + poison) on the choices made; `cue_touch` = the share of trials with a touch (under uncertainty the
instinct may touch nothing); both by sample. Chance is 0.5: a binomial on ~1,000 trials has σ ≈ 0.016, so seeds count, not trials.

## Verification

- `tests/test_cue_world.py`: the phases and their lengths; no circle and no touch before the choice; the block's place, and the pattern
  through thumbnails; food matches the sample in pixels, for both samples; a touch or a timeout ends the trial; `hold`; the sample does not
  show anywhere else in delay; ThoughtWorld over it; the stats. Without `env.cue`, the world is unchanged, bit for bit.

## Experiment vm-00057

Two pieces, gain, age key, ring (as vm-00056's T), [world | thought], 60k ticks, seeds 1–4, on both GPUs with MPS.

| arm | sample | delay | thought | it asks |
|---|---|---|---|---|
| **CH** | held | 20–30 | frozen | can the rule be learned at all? (must be well above 0.5) |
| **CS** | flash | 1–2 | frozen | does the window carry it? |
| **CF** | flash | 20–30 | frozen | the floor: must be 0.5 |
| **CT** | flash | 20–30 | dream, pair | memory in thought, as vm-00056 |
| **CG** | flash | 20–30 | dream, mean | the same, kept grey |

Reading: CT or CG above CF, with CH above them, is memory. If CH fails the world is too hard and says nothing of thought. If CT and CG sit at CF
(as expected: no gradient teaches what to write), the next step is a step of backpropagation through a dream.

## Results So Far

Interim: [`../00057-delayed-cue/README.md`](../00057-delayed-cue/README.md). Seeds 3 and 4 finished, seeds 1 and 2 running. Food share over each run's
last complete 10k ticks: CH 0.540 ± 0.017, CS 0.499, CF 0.487, CT 0.503, CG 0.529. CH has not clearly learned the rule (seed 4 reaches 0.65 on both
samples, seed 2 0.585, the others nothing), so the thought arms say nothing yet. Mean decoding keeps thought grey.

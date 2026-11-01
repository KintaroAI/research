# vm-00057: The Delayed-Cue World

**Date:** 2026-10-31
**Status:** In progress: seeds 3 and 4 finished at 60k ticks; seeds 1 and 2 still running, reported to 40k. Interim write-up, not tagged.
**Source:** to be tagged `exp/vm-00057` when complete (code: `visual-motor-agent/dev/`)
**Outputs:** `~/data/research/visual-motor-agent/exp_00057/`
**Task plan:** [`../tasks/00065-delayed-cue.md`](../tasks/00065-delayed-cue.md)

## Goal

vm-00056 could not tell whether thought is a memory, because foraging never needs one. This world has one answer that an 8-tick window
cannot give, so a choice above chance with the sample gone must come from something that outlives the window.

## Method

- **`CueWorld`** (`world/cue_world.py`, `configs/U_cue.json`): a sample flashes for 2 ticks, a 16 × 16 block at the top left of the view,
  solid or 2 × 2 checker. Then 20–30 ticks with no circles. Then one food and one poison of the same grey at 20 px on opposite sides of the
  cursor. **The food looks like the sample.** A touch or 40 ticks ends the trial. Relief 0.3, poison 0.2, so hunger stays off the cap at chance.
- **Model:** [world | thought] as vm-00056's T: two transformers, gain, age key, ring.
- **`online.thought_decode`:** `pair` as vm-00056, or `mean`, the grey fix: a dream of means should not sharpen itself to black and white.
- **Arms,** seeds 1–4, 60k ticks:

| arm | sample | delay | thought | it asks |
|---|---|---|---|---|
| **CH** | stays on | 20–30 | frozen | can the rule be learned at all? |
| **CS** | flash | 1–2 | frozen | does the window carry it? |
| **CF** | flash | 20–30 | frozen | the floor |
| **CT** | flash | 20–30 | dream, pair | memory in thought |
| **CG** | flash | 20–30 | dream, mean | the same, kept grey |

- **Measure:** food share = food / (food + poison) among trials ended by a touch; chance is 0.5. The ± is binomial on the touches pooled over seeds.

## Results so far

### R1. Food share over the last complete 10k ticks of each run

Seeds 1 and 2 at 30–40k, seeds 3 and 4 at 50–60k.

| arm | food share | touches | s1 | s2 | s3 | s4 |
|---|---|---|---|---|---|---|
| CH | 0.540 ± 0.017 | 858 | 0.453 | 0.585 | 0.473 | **0.648** |
| CS | 0.499 ± 0.013 | 1,376 | 0.505 | 0.504 | 0.492 | 0.497 |
| CF | 0.487 ± 0.018 | 774 | 0.531 | 0.448 | 0.472 | 0.492 |
| CT | 0.503 ± 0.017 | 819 | 0.524 | 0.517 | 0.497 | 0.474 |
| CG | 0.529 ± 0.018 | 783 | 0.519 | 0.544 | 0.523 | 0.526 |

### R2. Food share by age, pooled

Four seeds to 40k, seeds 3 and 4 after it.

| arm | 0–10k | 10–20k | 20–30k | 30–40k | 40–50k | 50–60k |
|---|---|---|---|---|---|---|
| CH | 0.489 | 0.497 | 0.495 | 0.516 | 0.524 | 0.567 |
| CS | 0.481 | 0.500 | 0.491 | 0.507 | 0.529 | 0.495 |
| CF | 0.476 | 0.505 | 0.521 | 0.488 | 0.516 | 0.482 |
| CT | 0.498 | 0.503 | 0.511 | 0.513 | 0.538 | 0.485 |
| CG | 0.497 | 0.522 | 0.516 | 0.499 | 0.464 | 0.525 |

CH seed 4 by 10k: 0.45, 0.53, 0.45, 0.58, 0.56, 0.65; in the last 10k, solid samples 0.65 and checker samples 0.63. CH seed 3: 0.47, 0.52,
0.54, 0.46, 0.49, 0.47.

### R3. Rates over the last 20k logged ticks, mean of the seeds

| arm | trials per 1k | touched | food per 1k | poison per 1k | hunger | capped | patch CE | thought at extremes |
|---|---|---|---|---|---|---|---|---|
| CH | 23.8 | 0.88 | 11.0 | 9.9 | 0.44 | 0.10 | 0.610 | 0.005 |
| CS | 44.6 | 0.77 | 17.7 | 17.2 | 0.27 | 0.04 | 0.615 | 0.005 |
| CF | 22.7 | 0.84 | 9.5 | 9.6 | 0.39 | 0.10 | 0.611 | 0.005 |
| CT | 23.3 | 0.87 | 10.3 | 10.0 | 0.47 | 0.10 | 0.524 | 0.948 |
| CG | 22.6 | 0.86 | 9.9 | 9.5 | 0.50 | 0.09 | 0.714 | 0.052 |

## Analysis so far

- **No memory is shown, and the test is not valid yet.** CT and CG are not separable from CF. But CH, the control that needs no memory, is at
  0.540 pooled, so the rule is at best barely learned, and the plan's reading applies: this says nothing of thought yet.
- **CH may be starting to learn, late.** It climbs with age, and seed 4 ends 4 standard errors above chance on both samples, which is the
  matching rule and not a liking for one pattern. Seed 2 reached 0.585 by 40k. Seeds 1 and 3 show nothing. The windows are small (~200 touches a
  seed) and the seeds differ, so this waits for seeds 1 and 2.
- **CS is at chance with nearly twice the trials,** in every seed. That is unexplained. The sample is inside the window at the choice, as in CH,
  but only at its far end, and only for a few ticks.
- **The likely limit is experience** (a guess, not tested): about 10 food touches per 1k ticks against QG's 43 in vm-00051, and a harder rule,
  since a circle's pattern means nothing without the sample.
- **The grey fix works.** Mean decoding keeps thought grey (5% of pixels at the extremes, 95% with pair decoding, under 1% frozen). Its cost
  is prediction: patch CE 0.714 against 0.524 with pair decoding and 0.611 frozen.
- **Agents reach the circles** (84–88% of long-delay trials end in a touch); they do not choose between them.

## Next

- Finish seeds 1 and 2, then rerun the full suite, read CH again, and complete this report.
- If CH stays near chance, thought cannot be tested in this world: a fixed-rule control (patterns with one meaning) separates too few trials
  from the matching rule itself.

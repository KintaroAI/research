# Task 00112: The Body Label Re-labeled at Every Replay, with the Full Dose (experiment vm-00100)

**Date:** 2026-11-20
**Status:** In progress (the owner, 2026-11-20, after vm-00099: "yes, write task for relabeling with full dose in own channel and run it")

## Context

- **vm-00099 (task 00110):**
  - SD (the stored label in the body's own channel, with the full dose) no longer starves: 0 of 12 lives, against SB's 4.
  - But it still forages blind: 9.8 food per 1k at 40–60k, against IB's 19.9 and MB's 54.2. Its label is right only two or three ticks before a meal: four to seven ticks before, it predicts .18 where the true V is .71.
- **MB (vm-00097) eats five times as much** with the same one-tick instinct, the same world with inertia and the same candidates. MB differs from SD in three ways:
  1. **re-labeled:** memory keeps the facts and every replay recomputes the label from the current model;
  2. **the channel:** the label is the hunger channel's target, not a channel of its own;
  3. **the cost:** the drift (.001 a tick) also costs, and a poison's cost is capped (no full dose).
- **vm-00098's plan** set the split as the next step; vm-00099 confirmed it.
- **The owner's view (2026-11-18):** re-labeling is "a cheat"; a label should be computed once, from what the agent predicted then. This task measures what that choice costs. The owner asked for this run.

## The design

- **One change against SD: re-labeling** (`online.label_refresh = 1`, only with `online.label = meal`).
  - **Memory keeps the facts:**
    - for each step: hunger before and after, whether its episode ended, the hunger the cap cut off (full dose);
    - long-term memory also keeps the one step after each chunk, so that a replayed chunk's last step has a next move.
  - **At every update,** for each replayed window and each transformer k: label of step t = the label SD would store, but with k's **current** prediction for the move made at t + 1, in place of the prediction k made at the time.
    - That prediction comes from a forward pass without gradient over the window that starts one step later (MB's bootstrap window, task 00098).
    - The prediction is read from the label channel.
  - **The label's definition is SD's, exactly:**
    - a meal: V = 1;
    - a poison (hunger rose by more than .01, counting the cut): V = −(the rise) + 0.95 × V̂(t+1);
    - any other step: V = 0.95 × V̂(t+1), with no cost for the drift;
    - clipped to [−1, 1], T = (1 − V)/2.
  - **Each transformer trains on its own labels** (as MB and SD).
- **Everything else is SD's:**
  - `configs/U_view.json`, 4 quarter transformers of 8 layers, tolerance .002, context 8, the ring;
  - `model.body_label = 1` (the label's input blank, 0.5), `online.label = meal`, `label_gamma` 0.95, `online.full_dose = 1`, `env.inertia = 4`.
- **Kept, for comparison:** the stored labels are still written once each tick (as in SD) but not trained on; the monitor `meal_stored` shows what SD's mechanism would have stored in this model.
- **Off by default:** without the option every run is unchanged.

## Arms

| Arm | World | Label | Seeds |
|---|---|---|---|
| SD (vm-00099), SB (vm-00098), MB (vm-00097), IB (vm-00095), linked | inertia from birth | stored, full dose; stored, capped; re-labeled in the hunger channel; hunger value | 1–12 |
| **RD** | inertia from birth | re-labeled at every replay, own channel, full dose | 1–12 |

- **One card:** the RTX 3090, 12 lives under MPS (timed first; the bootstrap passes cost about one forward pass of each transformer per update, as in MB).

## Measures

As vm-00099:
- **Lived** by 5k and at 40–60k: food, poison, mean hunger, overrides, move CE, smoothness; the seeds that starve.
- **The monitors:**
  - `meal`: predicted V against the true V, by ticks to the next meal;
  - `meal_stored`: the labels SD's mechanism would store.
- **Frozen:** `chain`, `value`, `reach` (poison), `move`, and `eval_candidates.py`.

## The readings, set before the results

1. **Plumbing:**
   - the re-labeled label equals SD's stored label for the same next prediction, on a table (meal, poison at and below the cap, drift, an episode's end);
   - the bootstrap window is the training window one step later;
   - the option checks;
   - with the option off, nothing changes (the 7 full-dose and the label tests pass);
   - all 12 lives run 60k ticks.
2. **The front moves back:** RD's predicted V four to seven ticks before a meal at 40–60k is above SD's .18, paired by seed.
3. **Re-labeling forages** (the main reading): RD − SD food at 40–60k, paired by seed.
4. **Against MB** (descriptive): RD − MB food and mean hunger. The remaining differences are the channel, the drift cost and the full dose.
5. **Poison** (descriptive): the rating of poison moves and the instinct's poison picks, against SD and MB.

**What follows:**
- **If 3 holds and RD comes near MB:**
  - storing once is what cost the foraging;
  - the label in its own channel, re-labeled, with the full dose, is the candidate default (the owner's decision);
  - Breakout (task 00111) is built on it.
- **If 3 holds but RD stays well below MB:** the channel or the drift cost matters too; split them next.
- **If 3 fails:** re-labeling alone does not help in the body's channel. Next is the other half of the split: the stored label in the hunger channel.

## Implementation

- [x] `online.label_refresh` (DEFAULT_ONLINE 0):
  - `GatedMemory` keeps one step past each chunk and builds the label's parts and the bootstrap windows as for the meal value (task 00108);
  - `meal_parts(..., drift=False, cut=...)` gives SD's label exactly;
  - `boot_values` reads the label channel when the body has one;
  - `lifetime.py` trains each transformer on its own fresh labels (the body label's target) and checks the options.
- [x] Tests `tests/test_label_refresh.py` (5: the parts give the stored label on a table; the bootstrap windows and labels from memory, newest, replayed and from stm; unchanged without the option; the option check; a short GPU life with its frozen value test); regression: 229 label, value, memory and world tests pass.
- [x] Timing on the RTX 3090: 12 lives at once, 3.0 ticks/s each (SD 4.1: the bootstrap passes), 10 GB; about 5.6 h for 60k ticks.
- [x] `00100-relabeled-label/` (README, Makefile).
- [ ] The runs (started 2026-11-20, queue `q100.sh`: then frozen tests, candidates, videos), the write-up.

## Verification

- Every RD run's `info.json` has `online.label_refresh` 1 and SD's settings otherwise.
- On a table, a re-labeled label equals `stored_labels` with the same next prediction.

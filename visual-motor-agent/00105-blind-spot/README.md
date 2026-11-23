# vm-00105: The Blind Spot — Does the Prediction Restore a Gap the Model Never Sees?

**Date:** 2026-11-23
**Status:** Running
**Source:** `dev/` (pending tag `exp/vm-00105`)
**Outputs:** `~/data/research/visual-motor-agent/exp_00105/`
**Task plan:** [`../tasks/00118-blind-spot.md`](../tasks/00118-blind-spot.md)

## Goal

The owner (2026-11-22): "we will permanently off some area in the visual field from the foraging world mimicing blind spot of the eye but train on the full picture so the predicted picture should have the gap restored - we will examine how good it is in restoring blind spot".

The eye has a blind spot where the optic nerve leaves; the brain fills it from the surround and from what was seen a moment before. Here the default agent sees its world with a permanent hole and learns against the whole frame. How well does its predicted frame restore the hole?

## Method

- **The agent:** the default (`configs/U_agent.json`, vm-00101's TT): the food world with inertia, a 64 × 64 egocentric view, 4 quarter transformers.
- **The gap** (`online.blind = [38, 16, 12, 12]`): 12 × 12 px at x 38–49, y 16–27, set to black in everything the model sees.
  - That covers acting, every update's inputs and the label's bootstrap windows.
  - It's the same frame for all four transformers; the world, memory and targets stay whole.
  - It sits up and to the right of the body, across parts of 6 patches of the top-right quarter, so the top-right transformer predicts those patches. Every transformer's summaries over the gap are blinded in its input and trained against the real pixels.
  - It's big enough to hide a whole food or poison (10–16 px). The view moves with the body, so the scene slides under the fixed gap.
  - The image `videos/vm105/blind_spot_where.png` shows where it is.
- **The arm: BL**, 12 seeds, 60k ticks. Seeds 1–4 on the RTX 3090, 5–12 on the RTX 4090, under MPS.
- **The control: vm-00101's TT** (the same agent without the gap, 12 seeds), not rerun.
- **The frozen test** (`make frozen`): per checkpoint (20k, 40k, 60k) and seed, 2,000 ticks with learning off in a new world (seed 100 + s). Each tick records the model's predicted current frame (`online.pred_record`: each transformer's own quarter, decoded `pair`, every patch predicted from the tokens before it) and the true frame. Four conditions:
  - **BL_gap:** BL as it lived;
  - **BL_open:** BL with the gap removed (pixels it never saw there);
  - **TT_gap:** TT with the gap put in (never trained with it);
  - **TT_open:** TT as it lived (the ceiling).
- **New code:**
  - `online.blind`, `online.pred_record` (`lifetime.py`, `online.blind_frames`);
  - `eval_blind.py` (measures and video);
  - `tests/test_blind.py`.

## Measures

`eval_blind.py`, grey levels 0–255, after the first 20 ticks:

- **gap:** the mean absolute error of the predicted frame on the gap's pixels.
  - Baselines: **bg**, the gap predicted as plain background; **prev**, the true gap of the tick before (a copy the model could make only if it saw the gap).
- **objects:** the same over ticks with an object in the gap (at least 4 px off the background by more than 25 grey levels), and by ticks since it entered (1, 2–3, 4–7, 8–15, 16+).
- **seen or not:** whether the predicted gap shows an object (the same rule): hits, false alarms, d′.
- **the surround:** the error on the visible pixels within 4 px of the gap, and on the other quarters.
- **lived foraging:** food and poison per 1k at 40–60k against TT.
- **Statistics:** paired by seed (BL and TT seed s share the seed number and test world), the mean difference with a 95% interval (t, 11 degrees of freedom), and how many of the 12 seeds point the same way.

## Readings, set before the results

1. **Plumbing:** the tests pass; all 12 lives run 60k ticks.
2. **It fills the gap:** BL_gap's gap error at 60k below the background baseline's, paired by seed.
3. **Filling is learned:** BL_gap's gap error at 60k below TT_gap's, paired by seed.
4. **Objects (descriptive):** BL_gap's error and hit rate on objects in the gap by ticks since entry, against TT_open's.
5. **The cost (descriptive):** BL's lived food and poison at 40–60k against TT's, and BL_gap's error on the surround against TT_open's.

## Results

Running since 2026-11-23.

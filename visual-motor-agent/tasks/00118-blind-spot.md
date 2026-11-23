# Task 00118: The Blind Spot — Filling the Gap (experiments vm-00105 and vm-00106)

**Date:** 2026-11-22
**Status:** Planned (after vm-00104)

The owner (2026-11-22): "next experiment will be about filling the gap - we will permanently off some area in the visual field from the foraging world mimicing blind spot of the eye but train on the full picture so the predicted picture should have the gap restored - we will examine how good it is in restoring blind spot and second experiment - we need to figure out how we can train on a image with blind spot as well but teach model to restore image back nevertheless"

## Context

- **The eye's blind spot:** where the optic nerve leaves the retina there are no receptors. The brain fills the gap from the surround and from what was seen there a moment before. We never see a hole.
- **The world:** the default agent's food world (`configs/U_agent.json`). The view is 64 × 64, egocentric (centred on the body, so the scene slides under the view as the body moves), with inertia. Food and poison are circles about 10–16 px across and do not move.
- **The model:** 4 quarter transformers, each predicting its own 32 × 32 quarter (16 patches of 8 × 8 px) of the next frame. Each patch is predicted by the token before it.
- **Two experiments:**
  1. **vm-00105, the gap in the input only:** the model sees the frame with the gap, and learns against the full frame. How well does its prediction restore the gap?
  2. **vm-00106, the gap in the input and the targets:** like the eye, nothing the model ever learns from has the gap filled. How can it still learn to restore it?

## The gap (both experiments)

- **`online.blind = [x, y, w, h]`** (px in the model's frame; off by default): those pixels set to 0 (black, distinct from the background's 20–30 grey) in what the model sees. That covers acting (the ring), every update's inputs, and the label's bootstrap windows. The world and memory stay clean.
- **Where:** 12 × 12 px at x 38–49, y 16–27. That's off-centre like the eye's (right of and above the body at the centre), inside one quarter transformer's field (the top-right one), and across parts of 6 of its patches.
  - The other three quarters never see the gap: they are a control inside the same model.
  - 12 px can hide a whole food or poison.
- **Same in every seed and the whole life ("permanently").**
- **How it's built:** the input-only path that `online.noise_in` uses (inputs ≠ targets in `update_rollout`), deterministic, with no generator. `noise_in` is refused with the meal label today; the blind spot must work with the default agent's settings.

## vm-00105: the gap in the input, the full frame as the target

- **Arm BL:** the default agent with `online.blind`, 12 seeds, 60k ticks.
- **The control:** vm-00101's TT (the default agent, 12 seeds, checkpoints 10k–60k kept), not rerun.
- **The frozen test** (`eval_blind.py`, new): a frozen life of 2,000 ticks per checkpoint (seed 100 + s). Each tick it records the model's predicted next frame (each transformer's own quarter, decoded `pair`, as the panels were), the true next frame and the input.
  - It needs a recorder of the predicted frame (`online.pred_record`, or the eval tool driving `forward_uni_step` itself), as vm-00102's panels decoded their forecasts.
  - **Test conditions:**
    - BL as it lived (gap in);
    - BL with the gap removed (it never saw those pixels);
    - TT with the gap put in (never trained with it: what filling-in costs without learning);
    - TT sighted (the ceiling: its error on those pixels when it sees them).
- **Measures, on the gap's pixels of the predicted next frame:**
  - **error:** mean absolute grey error against the truth, over all ticks;
  - **objects:** the same over ticks with an object in the gap (any true pixel off the background), and by ticks since the object entered the gap (1, 2–3, 4–7, 8–15, 16+). The gap is fixed while the scene moves, so an object usually slides in from the visible part;
  - **seen or not:** whether the predicted gap shows an object (pixels off the background by a threshold) when one is there and when none is: hits, false alarms, d′;
  - **the edges:** the error on the visible pixels next to the gap, against the same pixels in TT (does the hole disturb its surroundings?);
  - **the cost to foraging:** lived food and poison per 1k against TT, paired by seed;
  - **a video:** the input (with the hole), the predicted frame and the true frame side by side.
- **Baselines for the error:**
  - **background:** the gap predicted empty;
  - **the frame before:** the last frame's gap pixels, which the model never sees (an upper bound for a copy).
- **Readings, to be set before the results:**
  - BL's error in the gap below the background baseline's, paired by seed;
  - BL's error on objects falling with ticks since entry no faster than TT-sighted's (memory, not sight);
  - BL against TT-with-gap (learning to fill against not).

## vm-00106: the gap in the targets too (to be designed; options)

Without a target in the gap, the next-frame loss either teaches the model to predict black there (if the blinded frame is the target), or says nothing about the gap (if the loss is masked there). Options to restore it anyway:

- **A. Masked loss, nothing else:** no target in the gap. Do the shared weights fill it in on their own, as they predict every other patch? This is the null to beat.
  - It needs a per-value loss mask in the CUDA loss. `set_step_mask` masks whole steps; a mask of patch values may need new code.
- **B. Motion as the teacher (recommended):** the gap is fixed in the view while the scene slides under it. The spot the gap hides at tick t is in plain view a few ticks before or after, at a shifted position.
  - The gap's target at t is filled from the nearest frame (later or earlier) where that spot was visible, shifted by the body's movement in between.
  - Every frame used has the gap: nothing the model learns from ever showed those pixels at that moment, only at other moments. This is how filling-in could be learned.
  - The shift: from the world's displacement for a first version (a training signal from the world, never an input). Later from the frames themselves (registering the visible parts).
  - Food and poison stand still, so a shifted frame is exact for them. When the body has not moved enough, the gap stays masked.
- **C. Two eyes:** two views, each with its gap on its own side, as our eyes' blind spots are; one eye's gap is the other's seen part. In this flat world both eyes see the same view, so it reduces to copying from the other eye. It's worth more with parallax.
- **D. A gap that wanders (microsaccades):** the view jitters by a few px each tick, so the gap covers different spots. That's B with more motion. It moves the gap in world terms, not the gap itself.

The design is to be chosen after vm-00105, with A as the null in any case.

## Implementation (vm-00105)

- [ ] `online.blind` in `lifetime.py` (acting, update inputs, the label's bootstrap windows) and `DEFAULT_ONLINE`; allowed with the default agent's settings; tests: the input has the hole, memory and targets do not.
- [ ] The predicted-frame recorder and `eval_blind.py` (measures, baselines, the four test conditions, by seed and checkpoint).
- [ ] `00105-blind-spot/` (README, Makefile); the runs (12 seeds; 12 lives across the two cards), the frozen tests, the video, the write-up.

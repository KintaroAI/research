# Task 00091: Delayed Copies of the View, All Predicting the Present (experiment vm-00080)

**Date:** 2026-11-03
**Status:** In progress

## Context

- The owner (2026-11-03): "what if we construct the field view in a way that helps model to predict future. For example we use 4x field with 16 transformers but we feed first quarter of the field realtime image and other quarters image with some delay (each quarter different, maybe +1 tick, maybe exponential backoff (kind of)) but all of them should predict present image (like all 4 quadrants see 4 copies of present for backprop) and everything else is the same as with one image".
- The owner's decisions:
  - delays 0, 1, 2, 3 and 0, 1, 4, 16;
  - all transformers vote, since they all predict the present;
  - every transformer's body token is the present body.
- Why it may help:
  - **Dead reckoning:** a transformer on a copy d ticks old is trained to output the present. For d up to the context (8 ticks), its context still holds the moves made since its view was taken, so it can learn to shift the old view by them.
  - **Motion:** a present transformer holds the coarse summaries of all four copies, so it sees the scene 1, 4 or 16 ticks back, as frame stacking does.
  - **Reach:** a copy 16 ticks old is past the context.

## Design

**The world:** `env.delays = [0, d1, d2, d3]` (`world/delay_world.py`, a wrapper that `make_world` adds).
- Each observation is a 2 x 2 composite of the view: top-left now, top-right d1 ticks ago, bottom-left d2, bottom-right d3.
- Before tick d the first frame stands in.
- Everything else is the world's own: drives, touches, cursor, food, `gt_*`.
- `env.size` stays 64; the frame is 128 x 128.

**The model:** the default with `img = 128`, `uni_pyr = 4`, `band_n = 16` (square pieces of 4 x 4 patches, 32 x 32 pixels: four per copy, each the size of a default quadrant), `band_win = 1`, K4.

**Targets:** in training, the input is the stored composite (`update_rollout(inputs=...)`), and the target is its top-left quarter, the present, tiled four times (`online.delay_targets`). Each transformer is trained to output the present at its place.

**Memory** stores the composite, in gray: 16 KB a step.

**Acting is unchanged:** all 16 vote by the gain rule.

**Arms:**

| Arm | Delays | Isolates |
|---|---|---|
| D0 | 0, 0, 0, 0 | four copies of the present: 16 transformers' redundancy alone |
| D1 | 0, 1, 2, 3 | linear |
| DX | 0, 1, 4, 16 | exponential; 16 is past the context |

- Reference: vm-00073's K4, one present frame and 4 transformers, not rerun.
- Seeds and placement are decided by a timing.

## Measures

1. **Lived:** food, poison and mean hunger over the last 20k ticks; win shares by copy (the gain rule's `won_*_p*`).
2. **Prediction of the present, by copy (`eval_delay.py error`):** each transformer's prediction of its own patches against the present frame, on things. Beside it, the copy baseline: its own stale input against the present. How far ahead can a transformer predict, and how much does it improve on its input?
3. **Frozen foraging (`eval_delay.py chain`):** as seen, and with the delayed copies blanked to black, to show whether the present transformers use the past.

## The readings, set before the results

1. **Plumbing:**
   - The composite's quarters are the frames 0, d1, d2, d3 ticks back, checked against an undelayed world in lockstep.
   - Everything but the image is the undelayed world's.
   - The targets are the present tiled.
2. **Dead reckoning:** in D1 and DX, a delayed copy's prediction error of the present is below its stale input's error by more than its interval, for every delay up to 4.
3. **The past helps:** DX's lived food is above D0's by more than its interval.
4. **The past is used:** frozen, blanking DX's delayed copies lowers its food by more than its interval.

## Implementation

- [x] `world/delay_world.py`, `env.delays`, `make_world`
- [x] `online.delay_targets`; `lifetime.py`: the targets in training, memory sized by the model's frame, refusals (clog, noise_in, off)
- [x] `eval_hunger.Life(view=...)`; `eval_delay.py`: error, chain with blanking, show
- [x] `tests/test_delay.py`
- [x] `00080-delayed-copies/`: README, Makefile; six seeds, all 18 lives at once (the first minutes are the timing)
- [ ] Runs, frozen tests, write-up

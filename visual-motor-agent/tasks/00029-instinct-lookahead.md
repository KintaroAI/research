# Task 00029: The Instinct Looks 4 Steps Ahead (experiment vm-00023)

**Date:** 2026-10-25
**Status:** In progress

## Context

- The instinct (task 00023) compares candidate moves by the model's prediction of the next hunger, one step ahead.
  Only a click changes hunger in one step, so it acts as a click reflex: click food, don't click poison. It cannot
  steer toward food.
- Reach: 4 steps of at most 4 px per axis is 16–22 px; food discs have radius 12–16 px on a 64 px torus, so food is
  often within 4 steps.

## Design (`online.instinct_h`, default 1 = unchanged)

- **Candidates:** M moves sampled from the model's own move prediction, as now. Each is **held for H steps** (the same
  dx, dy, click): "if I keep doing this, where will my hunger be?". A path that crosses food while clicking shows
  relief, one that crosses poison a rise, one that clicks nothing stays flat.
- **Imagination with the model itself:** the window keeps its last ctx − (H − 1) real steps (the step positions stop at
  ctx), then H − 1 imagined steps. An imagined step's body is the previous move token's predicted next body; its
  patches are decoded from the model's own predictions (expected values) by **Jacobi iteration**: start from the
  current frame, run the forward pass, replace every patch with its prediction given the guess's earlier tokens,
  repeat J times (`online.instinct_jacobi`, 2). With J = the number of patches + 1 this is exactly sequential
  (autoregressive) decoding; few iterations suffice because most patches do not change between steps.
- **Score:** the move token's predicted next hunger after the H-th move; the lowest wins, the first sample is kept
  within `instinct_tol`. Only the first move is made (receding horizon); the next tick plans again.
- Cost: (H − 1) · J + 1 forward passes of the M windows per tick (7 for H = 4, J = 2), against one plus the candidates'
  fast path today.

## Verification

- With H = 1 the score equals the current instinct's reading (the candidates' next hunger).
- Jacobi decoding with J = patches + 1 iterations equals sequential decoding, patch by patch (tiny model).
- The imagined steps' positions and context: a lookahead window equals a forward pass built by hand.
- Imagination quality on a trained model (vm-00021): imagined hunger after 4 real moves against the world's.

## Experiment vm-00023

1. **Frozen:** vm-00021's six seeds at 100k and 200k: the forage test (food and poison clicks per 1,000 steps, hunger,
   time at the cap) with the 1-step instinct against the 4-step lookahead (M = 16), and against the policy alone.
2. **Lifetimes:** the default universal model with H = 4, seeds 1–3, 100k ticks, against vm-00022's U arm (H = 1):
   online food and poison clicks, the forage test at 25k, 50k, 100k.

## Imagination diagnostics (vm-00021 seed 3 at 200k, 16 forage worlds × 150 steps, 16 candidates each)

Each candidate held for H steps, its imagined hunger change against the world's (a copy of the world steps the held
move; scoring only), split by what the true path clicks:

| | no click event | food on the path | poison on the path | correlation |
|---|---|---|---|---|
| H = 1 (the instinct as before) | +0.024 (true +0.001) | −0.055 (−0.265) | +0.051 (+0.257) | 0.11 |
| H = 4, imagined (mean, J = 2) | +0.048 (+0.004) | −0.074 (−0.350) | **−0.020** (+0.402) | 0.10 |
| H = 4, imagined (mean, J = 8) | +0.048 | −0.100 | −0.027 | 0.11 |
| H = 4, imagined (mode, J = 2) | +0.041 | −0.114 | −0.014 | 0.19 |
| H = 4, the real frames (bodies imagined) | +0.055 | −0.138 | +0.099 | 0.23 |

- **The model can use 4 steps:** given the real future frames, the 4-step prediction is twice as informative as the
  1-step one (0.23 against 0.11) and has every sign right.
- **The imagined frames lose the cursor:** the scene (circles, the checkerboard with mode decoding, the dots) survives,
  but the cursor lags, smears (mean) or breaks into fragments (mode). A patch's pixels are predicted independently
  given the context, so when the cursor may be in one of a few places, neither the mean nor the per-pixel mode draws one
  coherent cursor. More Jacobi passes do not help (J = 8); decoding the mode helps food (−0.11 against −0.14 with the
  real frames) but not poison.
- **The imagined hunger drifts up** by about 0.012 per step (true 0.001), with real frames too: the chained body
  predictions are biased. The drift is the same for all candidates, so it does not change their order.
- Added `online.instinct_decode` (`mean` | `mode`, GPU only), `vmc_forward_uni_step(..., mode, ...)`; test against the
  PyTorch reference's argmax.

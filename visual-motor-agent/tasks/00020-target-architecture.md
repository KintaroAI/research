# Task 00020: Build the Target Architecture: Act Token, Signed Command, Parallel Bodies (experiment vm-00016)

**Date:** 2026-10-25
**Status:** Implemented; vm-00016 running

## Context

`../ARCHITECTURE.md` describes the target design. Most of it exists: patch tokens with body (drives), move and
command tokens, block-causal attention, per-patch residual frame heads, the drives head, a detached policy head with a
hidden layer, gated memory, slow hunger, 8×8 patches on the GPU. Three parts are new:

1. **An act token** that the policy reads. In vm-00010 the policy read the move token, which no loss trains once the
   move loss is detached, and it never aimed; letting the move loss train the Transformer eroded the patch tokens
   (vm-00014 P8A). The act token gets its own prediction losses (the next drives and a coarse next frame), so it must
   gather the scene, and the detached policy head reads it.
2. **A signed hindsight command:** the achieved change in hunger over the next k steps, so that moves into hazards are
   marked as well as moves before relief.
3. **Parallel bodies early in life:** N bodies in N worlds driven by the same weights, N shrinking to 1.

## Design

### Act token (`model.act_token=1`; patch models with residual frames only; default 0 leaves every model unchanged)

- One more token per step, after the command token: a learned vector `act_e` [d] plus a fifth type embedding and the
  step embedding. Tokens per step: npch + 4.
- The policy and value heads read the act token's output instead of the move token's (`vm_forward`'s z is the act
  token's output).
- **Auxiliary heads** on [act token output, executed move a_t] (as the patch and drives heads): one hidden ReLU layer
  (`ax1w`, pred_hidden units), then
  - the next drives: sigmoid(`axdw` h + b);
  - the coarse next frame, 8×8×3: the current coarse frame + `axfw` h + b (residual).
  Coarse frames are block averages of the prediction-size frames (32×32 → 8×8: 4×4 blocks; u8, rounded, as
  `downsample_u8`).
- **Loss** `ppo.act_coef` × (the drives loss of the aux head, with the same normalization and unusual-change weight as
  the drives head, + the mean squared error over the 192 coarse values). Its gradient reaches the Transformer (scaled
  by `pred_backbone_scale`, as the other prediction losses).
- New parameters appended at the end of the layout (`act_e`, `ax1w`, `ax1b`, `axdw`, `axdb`, `axfw`, `axfb`), size 0
  when off; `tok_type` has 5 rows when on. Checkpoint header slot 39. New random draws only when on, after all others,
  so default models are byte-identical.
- C++ model, PyTorch reference, CUDA model; stats `act_drive_mse`, `act_frame_mse` appended.

### Signed command (`online.goal_mode="delta"`; default `"relief"` unchanged)

- In training each remembered step's command is `hunger_t − hunger_(t+k)` (k = `goal_k`), from memory (hindsight; at
  the end of memory, the last known hunger).
- When acting, the command is the current hunger: "relieve all of it".

### Parallel bodies (`online.bodies=[N0, ticks, ...]`; default one body, unchanged)

- A schedule: N0 bodies for the first stretch, then halving every `bodies_halve` ticks down to 1.
- Each tick: every active body acts from its own window (one batched forward pass), its world steps, its transition is
  stored in its own stream. One gradient step on the newest window of each active body + replayed windows (`batch`).
- Memory: one stream per body (windows never cross bodies); the replayed windows are drawn across all streams.
- Logged: hits and hazard shares of body 0 (the one that lives on) and of all bodies.
- `online.explore_eps_end` and `online.explore_decay`: ε decays from `explore_eps` to `explore_eps_end` (default:
  no decay).

## Verification

- `tests/test_patch_tokens.py`: with the act token, C++ forward and every gradient against the reference (with and
  without the policy detached, with the aux loss, with `drive_changed_weight`); causality; the move loss detached
  trains the policy head only; checkpoints round-trip.
- `tests/test_cuda_model.py`: an act-token model, every loss setting, against C++.
- Defaults byte-identical: a 300-step patch lifetime and a CNN lifetime before and after.
- `online.py`: the signed command's labels on a hand-made hunger trace; parallel bodies: windows never cross bodies,
  the schedule, one body behaves as before.

## Experiment vm-00016

The target design against its parts, in vm-00015's slow-hunger world (if vm-00015 confirms it), 60k ticks, seeds 1–3:
A (act token), AS (+ signed command), ASB (+ 32 bodies shrinking to 1), and vm-00015's S8D as the control.
Measures: aim and the act token's probes (cursor, circle, contact), the episodic and forage tests, the share of steps
in a hazard, the drives heads inside against outside.

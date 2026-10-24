# Task 00010: Patch Tokens: One Token per Image Patch and per Input, in a World With Moving Dots

**Date:** 2026-10-24
**Status:** Complete

## Context

- **The representation is the bottleneck** (vm-00006 to vm-00008). The CNN squeezes each frame into
  one 128-number vector v. In the basic agent, that vector loses the circle (R² 0.07–0.48, against
  0.94 for the random encoder) and, without inverse dynamics, much of the cursor. A linear policy
  head cannot read "toward the circle" from it.
- **All inputs are fused into one token per step:** v, the drives, the previous move and the command
  are concatenated (134 numbers) and projected. Small inputs can drown: in vm-00006, when the
  features blew up, the hunger and command inputs had an effect of about 1e-7 on the move.
- **Most of the frame is empty background**, so most of a next-frame loss is "nothing changes".

## Goal

Replace the fused token with separate tokens: one per image patch, and one each for the drives, the
previous move and the command. Each token predicts its own part of the next state. Add small moving
dots to the world, so that patches have content and motion that the agent does not cause, and the
agent must work out which moving thing obeys its moves.

## Hypotheses

1. **The cursor and the circle stay readable**, each in its own patch tokens, and the direction to
   the circle becomes readable from the move token's output, better than from vm-00008's CNN vector.
2. **Patch tokens couple where things change.** When the cursor (or a dot) moves from patch A into
   patch B, B's token attends to A's far more than to other patches.
3. **Self versus other.** Changing the executed move changes the predicted next frame around the
   cursor, not around the dots.
4. Open: whether the basic agent then aims.

## Design

### World (`env.distractors=K`, default 0: the world is unchanged)

- K dots of 2×2 px, grey with brightness U[60, 180] (background (20, 20, 30), circle red, cursor
  white with a black outline).
- Each moves in a random direction at U[0.5, 2] px per step (the cursor: up to 4 px per axis) and
  bounces off the walls.
- Drawn under the circle and the cursor. Clicks and hits do not affect them.
- New positions and velocities at every reset, from the episode's random stream, so the test
  episodes stay fixed.
- Their positions go to `info["gt_dots"]`, for scoring only.
- Default for the experiments: K = 8, about one dot per two patches.

### Model (`model.patch=16`; 0 = the current CNN model, unchanged)

- **Tokens per step (19):**
  - 16 patch tokens: a shared linear embedding of the raw 16×16×3 pixels, plus `pos2d` of its cell;
  - one token each for the drives, the previous move and the command, from their own linear
    embeddings;
  - every token also gets a type embedding (patch, drives, move, command) and the time embedding of
    its step (as `wpe` now).
- **`pos2d`:** learned, initialized from a smooth code of the cell centre's (x, y), so geometry is
  there from the first step and can still be adjusted.
- **Attention:** block-causal over a window of `ctx` = 8 steps. A token sees every token of its own
  step and of the earlier steps in its window. The dots need at least two frames: their velocity is
  not visible in one.
- **Each token predicts its own next value:**
  - a patch token: the change of its 8×8×3 cell of the next 32×32 frame, from [its output, the
    executed move], by one MLP shared over patches. Residual prediction and changed-pixel weighting,
    as now;
  - the drives token: the next drives;
  - the previous-move token: the next move (mean dx, dy, click logit; learned std). The agent acts
    on it.
- **No global summary token** (see Later).
- **Losses and agent:** the basic agent of vm-00007/8 (frame, drives and move losses, hindsight
  command, gated memory U). vm-00008 showed that detaching the policy head keeps the features (30k), so
  the move loss is detached (`ppo.pi_detach=1`), with the linear head of vm-00008's arm D.

### Arms (basic agent, 50k steps, 4 threads, seeds 1–2)

| Arm | Model | Dots |
|---|---|---|
| P0 | patch tokens | none |
| PD | patch tokens | 8 |
| CD | CNN | 8 |

All arms are vm-00008's D (linear head, detached, no inverse dynamics, memory U); its runs D-U
(exp_00008/001–002) are the no-dot CNN control. The patch arms use 8 threads (one per training
sequence), the CNN arm 4, as vm-00008. The dots use their own random streams, so every arm is tested on
the same 128 episodes.

## Diagnostics

- **Coupling check.** For steps where the cursor, or a dot, crosses from cell A at step t to cell B
  at t+1: the attention weight from B's token to A's token (same step, and the step before), over
  heads and layers, relative to the mean weight on the other patch tokens. Control: the same ratio for an
  unrelated patch C looking at A (added after 10k: every patch looks a little at the cursor's patch).
  Ground truth is used for scoring only.
- **Action sensitivity (self versus other).** Change the executed move given to the prediction heads
  and measure how much each cell's predicted next patch changes. The share of that change in the
  cursor's cells, against the dots' cells.
- **Probes:** the cursor and the circle from the patch tokens' outputs; the direction from the move
  token's output.
- **The usual:** test success of its own moves, aim cosine, steps at a wall, online hits.

## Implementation details

- **C++:**
  - `VMConfig.patch` (header slot 38). In patch mode the CNN encoder and decoder are not built.
  - New parameters: patch embedding [d, 768]; `pos2d` [16, d]; type [4, d]; drive, move and command
    embeddings; the per-patch prediction MLP; the drives and move heads.
  - The attention mask gets a step index per token.
  - Acting: the basic agent already acts through `vm_forward` on its window, so patch mode does not use
    the KV cache (`vm_act` and `vm_encode` refuse it) and acting equals training by construction.
  - A new call returns attention weights, for the coupling check.
- **PyTorch reference** for patch mode; gradient tests at 4 and 16 threads; acting equals training;
  byte-identical checkpoints with `patch=0`.
- **World:** the dots in `circle_world.py`, with tests: determinism, no effect on hits or drives,
  nothing new in the observation besides the pixels. The JS replica is not updated: its equivalence
  test keeps running with `distractors=0`.
- **Cost estimate** (from operation counts, not measured): about 1.5–2× the current basic agent per
  step.

## Verification

- Gradient tests of patch mode against the reference; the old model byte-identical.
- World tests with and without dots; the dots render as specified (pixel checks).
- A 1,000-step smoke run of each arm: losses fall, no NaNs; the dots' cells' prediction error falls
  once two frames are in the window.

## Later (out of scope here)

- A global summary token that attends to everything and from which the move and drives are read.
  Evidence from the runs (30k, `analyze_patch.py tokens`): with the detached head, no loss trains the move
  token's output, and it holds little (cursor R² 0.17–0.30, direction 0.10–0.20). All patch tokens side by
  side hold nearly everything (cursor 0.91–0.98, circle 0.97–0.98, direction 0.83–0.90; at birth already
  0.81–0.93, 0.98–0.99, 0.72–0.81). A summary token that has to predict the whole next state would have to
  gather it, and the policy would read that token.
- Relative position biases in attention (the offset between two patches built in).
- Inverse dynamics in patch mode.
- The planner agent in patch mode (it needs a latent state to imagine with).
- Dots that look like the cursor: a harder test of telling self from other.
- The dots in the JS replica.

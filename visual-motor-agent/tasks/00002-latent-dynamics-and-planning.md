# Task 00002: Latent Dynamics + Planning: Acting Without Reward (Plan Phase 5)

**Date:** 2026-10-23
**Status:** Complete

## Context

vm-00001 showed that reward-free motor babbling teaches the agent what its actions do:
- how a move shifts the cursor (with the cursor on the background);
- that a click inside the circle removes it;
- that this click relieves hunger.

This knowledge was only *probed*; the agent never used it. The strongest form of the owner's
question ("learn meaningful dynamics without being reinforced into it") is an agent that acts on
what it learned. It should find and click circles using only its self-taught world model and its
hunger, with no reward, no value function and no policy training.

Two obstacles from vm-00001:
1. **Pixel imagination diverges after one step.** Feeding predicted frames back gives cursor
   errors of 0.84 → 8.2 → 27 → 34 px. Multi-step imagination needs a latent dynamics model
   (plan §7 "v1").
2. **Movement over the circle was not learned.** Retrieval is at chance with the cursor over the
   circle. A simulation of the babbling policy shows why:

   | Click probability | Steps with the cursor moving over a live circle | Hunger reliefs per 1k steps |
   |---|---:|---:|
   | 0.5 (vm-00001) | 0.24% | 4.85 |
   | 0.1 | 2.78% | 3.43 |

   Babbling at click probability 0.1 gives 11.6× more coverage for 29% fewer relief events.

## Deliverables

1. **C++ latent dynamics heads** (`model.dyn_hidden > 0`), trained during babbling:
   - The latent is the encoder's visual feature `s_t = vf_t` (128-d, post-ReLU), as used by the
     inverse-dynamics head. Inverse dynamics (`inv_hidden`) keeps the cursor in it.
   - Transition: `s' = s + T2(relu(T1([s, a])))`. Residual; a is the executed action
     (clipped dx, dy, click 0/1).
   - Hunger head: `h' = sigmoid(H2(relu(H1([s, h, a]))))`.
   - Coarse decoder: `x = X2(relu(X1(s)))`, a 16×16×3 frame (`pred_img / 2`). It keeps the circle
     in the latent (plan §7: guard against collapse) and lets us look at imagined states.
2. **Training losses**, summed over chains and scaled per transition like the other losses:
   - Chains start at every token t of a training sequence. They are unrolled k = 1..H
     (`ppo.dyn_horizon`) while the action exists in the sequence and no episode ended before
     step t+k−1. The chain stops after a terminal step.
   - Latent: `dyn_coef · mean_i (ŝ_{t,k} − sg(s_{t+k}))²`, only when token t+k continues the
     same episode.
   - Hunger: `dyn_h_coef · (ĥ_{t,k} − h_{t+k})²`. Input: the imagined latent ŝ_{t,k−1}, the true
     hunger at t+k−1 (teacher forcing) and a_{t+k−1}. Target: `next_drives[t+k−1]`. This
     includes terminal steps, so the relief on a successful click is learned.
   - Decoder: `dyn_x_coef · mse(x(ŝ_{t,k}), next_small[t+k−1]↓2)`, plus the same for real
     latents `x(s_t)` vs the current frame. On terminal steps this is how the transition learns
     that the circle vanishes.
   - ~~Gradients flow into the encoder through s_t~~ **Changed:** chains start from a *detached* s_t.
     With gradients into the encoder, the latent loss collapsed the features (32 of 128 dims active
     after 65k steps; inverse-dynamics R² stuck at 0). Only the decoder on real latents trains the
     encoder now. Gradients never flow through the latent targets.
3. **Babbling coverage:** `train.click_bias` sets the initial click-logit bias. −2.197 gives
   p = 0.1; the policy stays frozen during babbling.
4. **C++ inference API:**
   - `vm_imagine`: batched free-running rollouts from latents and hunger along given action
     sequences. Returns predicted hunger (and optionally latents and decoded frames).
   - `vm_decode`: coarse frames from latents.
5. **Planner (`dev/plan.py`):** model-predictive control with the cross-entropy method (CEM).
   - At every step: encode the frame, sample N = 256 action sequences of length 12
     (Gaussian dx, dy; Bernoulli click), imagine them, cost = Σ_k ĥ_k. Refit to the best 10%
     for 4 iterations, then execute the first action.
   - The plan is warm-started from the previous step.
   - No reward, value function or policy is used anywhere.
6. **Evaluation:**
   - The planner on the fixed test set used for PPO (radius 12–16, 128 episodes): success
     within 200 steps and median steps.
   - Compared with the random policy, the untrained model (control) and PPO (vm-00001: 100%,
     median 5–7).
   - Imagination accuracy over k: decoded frames and cursor position from a probe on imagined
     latents.

## Implementation details

- **Parameter order:** the new tensors are appended after `IV2B` (T1W, T1B, T2W, T2B, H1W, H1B,
  H2W, H2B, X1W, X1B, X2W, X2B). They are zero-sized when `dyn_hidden = 0`.
- **Initialization:** new tensors are initialized after all existing ones, so configs without
  `dyn_hidden` keep bitwise-identical initial weights. T2W starts small (identity-like
  transition).
- **Checkpoint header:** `dyn_hidden` goes in slot 35 (extension fields start at 32).
- **Config fields:**
  - PPOConfig gains floats `dyn_coef`, `dyn_h_coef`, `dyn_x_coef` and the int `dyn_horizon`.
  - The ctypes structs in `agent_lib.py` follow the same order.
- **Decoder targets:** 2×2 box filter with rounding half up, applied to the stored 32×32 small
  frames (C++ and reference identical).
- **Cost:** about 35 chain steps per 8-token sequence. Each costs roughly 0.6 MFLOP forward and
  1.9 MFLOP forward+backward with hidden size 256, comparable to the existing per-sequence cost.

## Verification

- PyTorch reference: the dynamics losses added to `ref.ppo_loss`; a `dyn` model variant and
  loss settings added to `tests/test_model_reference.py`. Gradients must match to about 1e-7
  relative, as for every other op.
- `vm_imagine` must match an unrolled numpy/torch computation using the same parameters.
- Existing C++ and Python tests still pass, including acting == training bitwise. A model without
  `dyn_hidden` gives bitwise-identical training to before.
- Sanity runs:
  - imagined cursor position (probe) tracks the real trajectory for several steps;
  - imagined hunger drops only for imagined clicks inside the circle.

## Success criterion

The planner hits the circle clearly more often and faster than the random policy (58% within 200
steps, median ≈ 160 steps for the agent's own initial policy on the test set), using no reward.
Stretch goal: approach PPO (median 5–7 steps).

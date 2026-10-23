# Task 00001: Circle World + C++ Visual–Motor Agent (Phases 0, 1, 3)

**Date:** 2026-10-23
**Status:** In progress

## Context

New research topic `visual-motor-agent/` (sibling of `thalamus-sorter/` and
`llm-fundamentals/`), following `visual_motor_learning_plan.md` (repo root).
A small randomly initialized agent sees only a 64×64 RGB screenshot plus two
drive scalars. It must learn to move a cursor and click a red circle without
target coordinates, distance shaping, DOM, demos or pretrained features.

Owner's framing: show that the model can learn meaningful sensorimotor
dynamics without being reinforced into it directly, i.e. learn what its
actions do from prediction alone.

Split requested by the owner: **Python models the world, C++ is the model.**
`llm-fundamentals/` uses the same split (Python data prep, C/CUDA training).

**Hardware constraint discovered 2026-10-23:** no GPU is attached to this VM
(QEMU, 128 vCPUs, SSE4.2 only, no AVX/FMA). The model is therefore CPU C++17
with OpenMP and hand-written SSE GEMM kernels, not CUDA. The code layout keeps
a CUDA port possible later.

## Deliverables

1. `dev/world/`: vectorized numpy CircleWorld (renderer, actions, hit test,
   drives, bounded exploration bonuses, auto-reset with terminal frames).
   Also an HTML/JS replica (play + replay), a node-based pixel-equivalence
   check, episode recording and a random-policy baseline.
2. `dev/src/vm_agent.cpp`: the complete model in C++, exposed as `libvmagent.so`
   through a C API:
   - CNN encoder (3× conv k4 s2) → token projection → 2-layer causal
     Transformer (d=128, 4 heads, FF 512, context 32).
   - Heads: Gaussian (dx,dy) + Bernoulli click policy, value, and
     action-conditioned 32×32 next-frame decoder + next-drive prediction.
   - Manual backward for every op, PPO (GAE, clipping, entropy, grad-norm
     clip) and AdamW.
   - Rollout memory, checkpoint save/load, KV-cached acting.
3. `dev/agent_lib.py`: ctypes wrapper. `dev/train.py`: Python orchestrates the
   world and C++ owns all learning. `dev/evaluate.py`: evaluation,
   counterfactual action-conditioned prediction analysis and figures.
4. Tests:
   - C++ (`test_vm_agent`): GEMM, finite-difference op checks, causality,
     acting vs training consistency, checkpoint roundtrip.
   - PyTorch reference (`reference/torch_reference.py`) matching the full
     forward, losses and gradients.
   - pytest env/reward/leak tests.
5. First experiments: random baselines, Run A (sparse reward only), Run C
   (+ prediction), and a reward-free prediction-only "babbling" run that tests
   whether dynamics are learned without reinforcement.

## Design decisions

- **Python world ↔ C++ model via ctypes, in-process.** Python owns the loop
  (env.step, logging). C++ owns weights, acting (with sampling RNG), rollout
  memory, GAE, PPO and the optimizer. Arrays cross as raw pointers (zero
  copy). Pipes or sockets would add latency per step and give no benefit.
- **Block-aligned context.** The Transformer context restarts at episode
  starts and every 32 steps (block boundaries). The rollout length is a
  multiple of 32, so a training sequence is exactly one block. Acting with
  the KV cache and training on whole blocks then compute the same function,
  and the PPO ratio is exactly 1 before the first gradient step (tested).
  The cost is that memory is truncated at block edges, which is acceptable:
  the task is nearly Markov given the frame.
- **Everything is an NN GEMM.** Transposed weight copies are refreshed after
  each optimizer step; activations are transposed or im2row'd as needed.
  Per-element accumulation order is identical for M=1 (acting) and M=T
  (training), so both paths agree to rounding. FMA contraction is disabled.
- **Thread-per-sequence data parallelism.** Each thread runs one 32-step
  sequence forward and backward into a thread-local gradient buffer. The
  threads then reduce and AdamW applies. This needs one parallel region per
  minibatch, and static scheduling keeps runs deterministic.
- **Prediction target.** The 32×32 downsampled next frame, using the
  *terminal* frame at episode end, where the circle vanishes on a successful
  click. This is a learnable visual consequence of clicking. Prediction is a
  supervised auxiliary loss only; its accuracy is never rewarded. A
  `pred_backbone_scale` knob scales the gradient it sends into the shared
  backbone.
- **Diagnostics without leakage.** The changed-pixel MSE and the
  copy-last-frame baseline use only consecutive rendered frames. Ground-truth
  geometry is used only by evaluation scripts.

## Changes made during implementation (2026-10-23)

- The plain pixel-MSE predictor ignored the cursor (about 1% of the pixels). Added model options,
  each mirrored in the PyTorch reference and gradient-checked:
  - `coord` (CoordConv input + spatial-broadcast decoder)
  - `pred_residual` (predict the change)
  - `pred_skip` (U-Net skips of the conv-3/conv-2 maps + tiled action, with the action meeting
    local features before a ReLU)
  - `inv_hidden` with `ppo.inv_coef` (inverse dynamics on consecutive visual features)
  - `ppo.pred_changed_weight` (loss weight on pixels that differ between frames)
- Added `vm_encode` / `vm_inverse` (probes, analysis).
- Added `run_queue.py` (2-slot job queue), `analyze_dynamics.py` / `dynamics_over_training.py` /
  `compare_dynamics.py` (counterfactual tests) and `record_episodes.py`.
- Checkpoint header extension fields live at slots 32+ (an earlier collision with slot 20 was fixed).

## Verification

- `make test` (C++ unit tests) and `make pytest` (env tests + PyTorch
  reference equivalence of loss and gradients) pass. That is 129 Python tests including 60
  reference-equivalence tests across 6 architecture variants, plus the C++ suite.
- Random-policy baselines recorded.
- Training runs logged to `~/data/research/visual-motor-agent/exp_00001/`.
- Counterfactual predictions show action-dependent cursor displacement, and
  the circle vanishing on click, in the reward-free run.

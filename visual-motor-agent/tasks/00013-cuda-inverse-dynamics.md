# Task 00013: Inverse Dynamics on the GPU (DMI's Lifetime There)

**Date:** 2026-10-24
**Status:** Complete

## Context

Tasks 00011 and 00012 run the basic agent's lifetime on the GPU (202 against 24 steps/s), but refuse inverse
dynamics (`model.inv_hidden`, `ppo.inv_coef`). The best basic agent so far, vm-00008's DMI, uses it: a detached
policy head with a hidden layer, plus the inverse-dynamics head.

## The head (as in `vm_agent.cpp`, `seq_inverse` and `seq_bwd_core`)

- Input: [v_t, v_t+1], the visual features of consecutive frames (v_t twice at the sequence's last step).
  Hidden layer ReLU(IV1 · in), output IV2 · h: the executed (dx, dy).
- Loss: inv_coef · ½ |out − clip(a_t)|², only where t + 1 is in the sequence and in the same episode (no episode
  start at t + 1; one-frame attention does not change this). Reported: inv_mse and inv_r2 = 1 − SE / ½|a|².
- The gradient enters v of both frames, so it trains the encoder (it is not scaled by `pred_backbone_scale`).

## Deliverables

- `src/vm_cuda.cu`: the head's forward, loss and backward (inputs, validity per step, the gradient into v_t and
  v_t+1); `inv_mse`, `inv_r2` in the stats (appended to `VMCStats`). `make_dims` accepts `inv_hidden`.
- `cuda_lib.py`: the new stats, reported by `update_rollout`.

## Verification

- `tests/test_cuda_model.py`: models with inverse dynamics (DMI's: hidden-layer head and inverse head; a small one,
  also with one-frame attention), against `vm_debug_grads`: the loss, every gradient tensor, inv_mse and inv_r2,
  one full update, the forward.
- A DMI lifetime on the GPU (vm-00008 DMI-U, seeds 1–2, 50k steps) against the CPU runs: test success, hit rate,
  probes, aim.

## Results (2026-10-25)

- **Implemented** in `src/vm_cuda.cu`: the head's inputs, forward, loss (valid where t + 1 is in the sequence and the
  episode) and backward, with the gradient into v_t and v_t+1; `inv_mse` and `inv_r2` in `VMCStats`.
- **Equivalence:** 91 GPU tests (two models with inverse heads, DMI's and a small one with one-frame attention; two
  inverse loss settings). Loss, gradients, inv_mse and inv_r2 match `vm_debug_grads`; one full update agrees. A build
  that drops the gradient into v_t+1 fails exactly the 6 tests that train an inverse head. All 529 tests pass.
- **DMI lifetimes** (vm-00008 DMI-U, 50k steps; GPU: 6.3 minutes each, two at once; CPU: 146 minutes in vm-00008):

  | | seeds | test success at 50k: mean (range) | mean 10k–50k | online hit rate 40k–50k | aim at 50k |
  |---|---|---|---|---|---|
  | CPU | 1–6 | 0.68 (0.28–0.82) | 0.55 | 0.67 | +0.14 |
  | GPU | 1–8 | 0.67 (0.45–0.86) | 0.52 | 0.65 | +0.14 |

  The two distributions agree. GPU seed 2 learned little (0.48 at 50k), but CPU seed 5 did less (0.28): DMI fails in
  some lifetimes on either device. vm-00008 had only seeds 1–2, which both learned; over six CPU seeds, five reach
  0.68–0.82 and one stays below birth.
- Outputs: `~/data/research/visual-motor-agent/gpu_checks/` (`003`/`004_AHR_DMI_U_gpu_*`, `dmi_gpu_s3..8`,
  `dmi_cpu_s3..6`).

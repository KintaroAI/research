# Task 00015: The Planner Agent on the GPU (Latent Dynamics, Encode, Imagine, Decode)

**Date:** 2026-10-25
**Status:** Complete

## Context

After tasks 00011–00013 the basic agents (D, DMI) run their lifetimes on the GPU. The planner agent (vm-00005's L,
`configs/L_gated.json`) still needs the CPU: its update has the latent-dynamics losses, and its CEM planner (`plan.py`)
calls `encode` and `imagine` with 256 candidate plans × 12 steps, 4 times per step. On the CPU it ran at 3.5 steps/s
under load.

The owner asked to port everything left. Written while implementing, between the vm-00011 launches.

## Deliverables

- `src/vm_cuda.cu`, as `seq_dyn`, `dyn_step`, `dyn_decode`, `vm_encode`, `vm_imagine` and `vm_decode` in
  `vm_agent.cpp`:
  - the chain layout for all sequences of a batch at once (host side: chains start at every frame, stop at the
    horizon, the sequence's end and episode starts); the transition, hunger and coarse decoder heads level by level;
    the latent, hunger and decoder losses; the backward pass from the last level down (v detached);
  - `vmc_encode` (v of n frames), `vmc_imagine` (rollouts of any n, all rows in one batched step), `vmc_decode`;
  - `dyn_mse`, `dyn_hunger_mse`, `dyn_recon_mse` in `VMCStats`.
- `cuda_lib.CudaModel.encode/imagine/decode`; `online.py` with `online.device=cuda` hands the GPU model to the
  CEM planner and uses it for the gain check and the hunger diagnostic.

## Verification

- `tests/test_cuda_model.py`: the planner agent's model (inverse and latent dynamics, one-frame attention) and a
  small one; its loss settings (horizon 7; a short horizon with the hunger head on real latents only): the loss, every
  gradient, the three dynamics MSEs, one full update with the dynamics losses, the forward. `encode`, `imagine`
  (hunger, latents, frames; 70 rollouts, more than the C++ chunk of 64) and `decode` against the C++ model.
- Negative control: dropping the residual path of the chain backward must fail the dynamics tests.
- A planner-agent lifetime on the GPU against vm-00007's CPU runs (online hits, planner test).
- Throughput of the planner agent's online loop, CPU against GPU.

## Results (2026-10-25)

- **Implemented** as planned, plus patch tokens (`run_patch`: token building, block-causal attention with a key range
  per token, the per-cell heads), so every model of `dev/` runs on the GPU except PPO (only `train.py` uses it). The
  Transformer is one shared forward / backward for both token layouts.
- **Equivalence:** 191 GPU tests in all (183 run, 8 skipped where a model has no latent dynamics), with the planner agent's model, a small latent-dynamics model, vm-00010's
  patch model and a small patch model with a hidden-layer head. Negative controls: dropping the chain backward's
  residual path fails exactly the 6 dynamics tests; plain causal attention inside a step fails all 34 patch tests.
- **Speed:** the planner agent L at 42 steps/s per lifetime with 12 lifetimes sharing the GPU under the CUDA
  multi-process service (64 with 6), against 3.5 on the loaded CPU.
- **A planner lifetime on the GPU** (vm-00007 L_U, episodic world, 30k steps) against the CPU runs:

  | | hits 0–10k | 10k–20k | 20k–30k | planner test 10k / 20k / 30k |
  |---|---|---|---|---|
  | CPU seed 1 | 119/152 | 307/332 | 466/479 | 0.70 / 0.80 / 0.92 |
  | GPU seed 1 | 296/309 | 421/428 | 631/634 | 0.81 / 0.85 / 0.97 |
  | CPU seed 2 | 136/152 | 513/518 | 692/695 | 0.93 / 0.84 / 0.95 |
  | GPU seed 2 | 19/61 | 110/142 | 297/311 | 0.23 / 0.48 / 0.97 |

  Both GPU lifetimes reach 0.97 by 30k; seed 2's slow start is within the early spread vm-00005 recorded (69–218
  hits in the first 10k). Outputs: `~/data/research/visual-motor-agent/gpu_checks/L_U_gpu_s*`.

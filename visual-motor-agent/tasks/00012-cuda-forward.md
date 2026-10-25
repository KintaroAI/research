# Task 00012: The Basic Agent's Forward Pass on the GPU: the Whole Lifetime There

**Date:** 2026-10-24
**Status:** Complete

## Context

Task 00011 moved the basic agent's update to the GPU (1.8 ms against 31 ms in C++). The online loop got 2.6×
faster (62 against 24 steps/s), but then 10 of its 16 ms per step were still on the CPU:
- the forward passes, for acting and for the surprise of the gated memory (6.3 ms);
- copying the updated weights back to the C++ agent and re-transposing them (4.3 ms).

Moving the forward pass to the GPU removes both. The weights then stay on the GPU and come back only for
checkpoints.

## Deliverables

- `vmc_forward` in `src/vm_cuda.cu`: the forward pass of S sequences of T ≤ ctx steps, returning what
  `vm_forward` returns (z, value, mu, click p, and with prediction actions the predicted next frames and drives).
  It reuses the update's forward kernels, which take the sequence length of the batch instead of ctx.
- `cuda_lib.CudaModel.forward`, with the outputs of `agent_lib.Agent.forward`.
- `online.device=cuda`: acting and the surprise use the GPU forward. After an update only the log-std (2
  numbers, used to sample moves) is copied back. All the weights come back before each checkpoint.

## Verification

- `tests/test_cuda_model.py`: the GPU forward against `vm_forward` for windows shorter than ctx, episodes that
  start inside a window, with and without prediction actions, one-frame attention.
- The update tests keep passing (the sequence-length change touches the attention).
- An online run with `online.device=cuda` whose checkpoints load in the C++ agent and give the same forward
  outputs as the GPU did.

## Measurement

- Steps per second of the basic agent's online loop: CPU, GPU update only (task 00011), GPU forward and update.

## Results (2026-10-24)

- **Implemented:** `vmc_forward` and `vmc_get_range` (`src/vm_cuda.cu`); the update's forward section takes the
  batch's sequence length (any T ≤ ctx). `CudaModel.forward` and `sync_to`, and `online.device=cuda` now acts, computes
  the surprise and learns on the GPU; checkpoints sync the weights first.
- **Equivalence:** 28 new tests (53 GPU tests in all). The forward matches `vm_forward` (rtol 1e-4) for 4 models,
  windows of 1, 5 and 8 steps not aligned with blocks, with and without prediction actions. After three GPU updates
  the GPU forward equals the C++ forward of the synced weights. All 491 tests pass.
- **Speed** (the basic agent of vm-00008 D, quiet machine):

  | Online loop | steps/s |
  |---|---:|
  | CPU (C++, 16 threads) | 24 |
  | GPU update only (task 00011) | 62 |
  | GPU forward and update | **202** |
  | two lifetimes sharing the GPU | 137 each |

- **A whole lifetime:** vm-00008's D-U (seeds 1, 2, 50k steps) rerun with `online.device=cuda`, two at once:
  6.2 minutes each, against 142 minutes for the CPU runs (on a loaded machine). It learns the same, not
  bitwise (the sums run in another order, and the lifetime diverges from there):

  | | test success 10k–50k (mean) | online hit rate 40k–50k | z at 50k: cursor / circle / direction |
  |---|---|---|---|
  | CPU (vm-00008) | 0.43 0.35 0.37 0.33 0.39 (0.37); 0.27 0.39 0.48 0.54 0.55 (0.45) | 0.54, 0.46 | 0.50 / 0.48 / 0.41; 0.49 / 0.69 / 0.49 |
  | GPU | 0.36 0.41 0.36 0.45 0.45 (0.41); 0.40 0.52 0.47 0.42 0.41 (0.45) | 0.60, 0.57 | 0.63 / 0.44 / 0.54; 0.49 / 0.58 / 0.49 |

  Outputs: `~/data/research/visual-motor-agent/gpu_checks/`.

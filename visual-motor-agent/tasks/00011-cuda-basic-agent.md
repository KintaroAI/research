# Task 00011: The Basic Agent's Update on the GPU, in CUDA

**Date:** 2026-10-24
**Status:** Complete (v1)

## Context

The VM now has an RTX 3090 (24 GB, sm_86, nvcc 12.0). A first check (`dev/README.md`, Performance)
ran the PyTorch reference on it. One online update of vm-00008's DMI model took 32 ms, as long as the
C++ code on 16–64 threads (31 ms), but the GPU was busy for only 5 ms. The rest was Python and the
launches of about 500 kernels. The owner chose a hand-written CUDA version (like the GPT-2 code in
`llm-fundamentals`), starting with a very basic model, to see how it actually goes.

The online loop of the basic agent spends 85% of its time in the update (forward, backward, AdamW of 8
sequences × 8 steps). So the update is what to port first.

## Goal

A CUDA implementation of the basic agent's update that gives the same loss and gradients as the C++
model (to float precision) and is measured against it.

## Scope (v1)

- **The model of vm-00008's arm D** (`configs/AHR_gated.json` with `ppo.pi_detach=1`):
  - the CoordConv encoder (3 convs k4 s2) → linear → v; the token [v, drives, previous move, command];
  - the input projection with the learned time embedding; the pre-LN Transformer (any layers and heads,
    causal within the context block);
  - the policy head, linear or with a hidden layer (`model.pi_hidden`), and its log-std;
  - the prediction MLP on [z, executed move], the drives head and the frame decoder: two transposed convs
    with the action-tiled bottleneck (`pb`), coordinate channels and U-Net skips; residual frames.
- **Losses:** the next frame (changed-pixel weighting), the next drives and the move (`ppo.move_coef`,
  detached or not), with `ppo.pred_backbone_scale`. Then the gradient-norm clip (shared, or
  `ppo.pi_own_clip`) and AdamW, all on the device, with no host synchronization inside an update.
- **Refused for now:** PPO (`use_ppo`), inverse dynamics, latent dynamics, patch tokens, models without
  `coord`, `pred_residual` and `pred_skip`.
- **The same parameter layout as the C++ model**, so weights move both ways with one copy.

## Design

- `dev/src/vm_cuda.cu` (+ `vm_cuda.h`) → `libvmcuda.so` (nvcc -O3 -arch=sm_86, cuBLAS). C API: create from a
  config and the flat parameters; upload a batch of S sequences × K steps; loss and gradients; a full update;
  read parameters, gradients and loss terms back.
- All S·K frames are batched: convolutions as im2col + one GEMM per layer, transposed convolutions as GEMM +
  col2im. One gather kernel serves conv backward and deconv forward, one im2col kernel conv forward and
  deconv backward. Linear layers via cuBLAS (row-major wrappers as in `llm-fundamentals`' GPT-2). Custom
  kernels for LayerNorm, GELU, attention (≤ K keys per query), the token build, the decoder inputs and the
  losses.
- fp32 throughout, TF32 off, so the gradients can match the C++ ones.
- `dev/cuda_lib.py`: the ctypes wrapper, reusing `agent_lib`'s structs.

## Verification

- `tests/test_cuda_model.py` (skipped without a GPU or `libvmcuda.so`): on rollouts filled by the C++ agent,
  the CUDA loss and every gradient tensor match `vm_debug_grads` (relative error below 1e-3), for the
  default and a small model, linear and hidden-layer policy heads, detached or not, own clip or not, and
  episodes that start inside a block. One full update gives the same parameters (to float precision).
- Negative control: a deliberately wrong kernel (for example, the skip channels dropped) must fail it.

## Measurement

- The update time at 8, 32 and 128 sequences against the C++ code (16 and 64 threads) and the PyTorch
  reference on the GPU.

## Results (v1, 2026-10-24)

- **Implemented:** `src/vm_cuda.cu` (`make libvmcuda.so`), `cuda_lib.py`, and `online.device=cuda` in
  `online.py`: the update runs on the GPU and the weights are copied back to the C++ agent, which keeps acting on
  the CPU. The optimizer state stays on the GPU (the final checkpoint is saved without it).
- **Equivalence** (`tests/test_cuda_model.py`, 25 tests): the loss and every gradient tensor match the C++ model to
  about 1e-7 relative error, for the default and a small model, linear and hidden-layer heads, one-frame
  attention, detached or not, and four loss settings. One full update agrees, and so does the rollout adapter.
  Negative controls: dropping the a2 skip gradient fails 20 of 21 tests, and a changed-pixel mask that ignores
  the blue channel fails 12.
- **Update time** (vm-00008 D-U at 50k, 1.35M parameters; quiet machine):

  | Sequences × 8 steps | C++, 16 threads | C++, 64 threads | CUDA | CUDA with the batch upload |
  |---|---:|---:|---:|---:|
  | 8 (the online update) | 31.3 ms | 37.0 ms | 1.83 ms | 2.08 ms |
  | 32 | 73.4 ms | 52.9 ms | 4.15 ms | 4.57 ms |
  | 128 | 263.6 ms | 141.0 ms | 14.4 ms | 16.0 ms |

  At the online batch size the GPU update is 17× faster than the C++ one; the PyTorch reference on the same GPU
  took 32 ms for the slightly larger DMI model (mostly launch overhead).
- **The whole online loop** (the basic agent of vm-00008 D, 16 threads, 3,000 steps): 24 steps/s on the CPU,
  62 with the GPU update (2.6×). The prediction loss falls as on the CPU (0.0050 → 0.0012 by 3k steps). What is
  left per step (16 ms): acting and the surprise forward on the CPU (6.3 ms), refreshing the C++ agent's
  transposed weights after each update (3.6 ms) and copying them (0.7 ms), the GPU update (1.9 ms), the world and
  the memory.

## Later (out of scope)

- The forward pass for acting on the GPU: it removes the CPU forward and the weight copy and refresh (about 10
  of the 16 ms per step).
- CUDA graphs (the update is written without host syncs so it can be captured).
- Inverse dynamics (DMI), PPO, the planner's latent dynamics, patch tokens.

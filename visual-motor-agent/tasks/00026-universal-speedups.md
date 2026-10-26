# Task 00026: Speeding Up the Universal Model

**Date:** 2026-10-25
**Status:** Complete

## Context

A profile of the gray universal model (`configs/U_universal.json`) on the RTX 4090: per tick about 9 ms of GPU work
(the update 5.0 ms, 129 kernels; the policy pass 0.3 ms; the instinct's 16 candidate passes 1.6 ms), and in Python
51% of a tick waiting for the update, 29% for acting, 6% uploading the batch, 5% the world.

## Changes (outputs unchanged except where noted)

1. **The loss kernel** (`k_uni_loss`, 0.89 ms of the 5.0): one warp per (token, value) with a lane per bin (coalesced
   reads, warp-shuffle max and sums), 8 warps per token row, one atomic per row as before. Any number of bins.
2. **The instinct's candidates reuse the policy pass:** only the move token differs between candidates, and every
   earlier token's keys and values are already in the policy pass's per-layer buffers. `vmc_uni_candidates` computes
   the move token alone for M candidates per window (embedding, each layer's attention over the cached keys plus its
   own, the MLP, the final LayerNorm, the head) and returns each candidate's predicted next hunger. The C++ model keeps
   the full pass; `uni_act` uses the fast path when the model has it.
3. **No next frames for the universal model:** its targets come from inside the window and the next drives, so
   `vmc_set_batch` skips uploading and downsampling the next frames (and the current frames' downsample), and the
   Python rollout stops copying them.
4. **Fewer waits for the GPU:** `update_rollout` reads the stats only when asked (`lifetime.py`: every
   `online.stats_every` ticks, default 10; the metrics average those samples) and skips the log-std read for models
   without a policy head.

## Verification

- `tests/test_cuda_model.py`: all universal-model tests against C++ (the new loss kernel); the candidates' fast path
  against the full forward pass with the candidates in place (predicted next hunger, 1e-4); `set_batch` without next
  frames gives the same loss and gradients.
- A lifetime with the changes: metrics as before; ticks per second before and after (alone on the GPU).

## Results

- **The loss kernel:** 0.89 → 0.13 ms; the update 5.00 → 4.24 ms of kernels (Nsight Compute, RTX 4090).
- **The candidates:** the policy pass plus 16 candidates 1.88 → 0.61 ms of kernels. A first version with one thread per
  (candidate, head) took 3.80 ms (64 threads looping over 527 keys); one warp per (candidate, head), lanes splitting the
  keys and merging their online softmaxes, fixed it. Matches the full pass to 1e-5 and the C++ model to 2e-5; a build
  without the candidate's own key fails the test.
- **A lifetime** (`configs/U_universal.json`, 5k ticks, alone on the RTX 4090): **121 → 178 ticks/s (+47%)**, the
  losses the same. "Before" is the previous library with the stats read every tick (the current Python falls back to
  full candidate passes and to next frames with an older library).
- Tests: 571 CPU, 387 GPU pass.


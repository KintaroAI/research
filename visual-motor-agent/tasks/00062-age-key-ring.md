# Task 00062: The Age Key and a Key and Value Ring (experiment vm-00054)

**Date:** 2026-10-31
**Status:** In progress

## Context

- Every tick the window shifts by one and everything is recomputed: 7/8 of the input is the same as on the last tick.
  The owner's direction (2026-10-30): inference on top of old results, with positions that do not move. Slightly off
  results in the ring (old keys from older weights, a token keeping what it saw when new) are accepted.
- Options: `ARCHITECTURE.md` section 9. Chosen: the **age key**: each reader decides how much it cares about age.

## Design

### `model.age_key = 1` (universal)

- The step embedding is no longer added to tokens. The same table `wpe` [ctx, d] holds one key per age, split among the
  heads as K is: `score(q, k_j) = (q · k_j + q · e[age]) / sqrt(hs)`, `age = step(reader) − step(key)`. The parameter count
  does not change. A token's own K and V no longer depend on where it stands in the window.
- All tokens of a step share an age, so per (row, head) only ctx extra dot products are needed: `A = Q Eᵀ` [rows, ctx].
- Backward, from dS (the gradient at the scaled scores), binned by age into D [rows, ctx]: `dQ += D E`, `dE += Dᵀ Q`.
- C++ (`attend_one`, `attention_bwd`), CUDA (always the GEMM path; `k_attn_age`, `k_attn_age_bin`, `k_attn_decode` for
  the candidates), the reference, `letter_ce.py`. Header slot 58.

### `online.ring = 1`: keys and values kept across ticks (GPU, `age_key`, no attention regions)

- A buffer per layer `[N, 2 · ctx · nt, 3d]`, 2 · ctx steps long so the window is always contiguous: the current step
  is written at `wp`, the window is the last ctx steps; at the buffer's end the window is copied to its start.
- `vmc_uni_ring_step(frame, drives, ep_start)`: only the new step's tokens are computed. Each layer: their Q, K, V are
  written to the ring, their scores against the window, ages, masked softmax, the block's rest. The move token's slot
  is zero until the move is known, as in a full pass. Returns the move probabilities.
- `vmc_uni_candidates` scores against the ring when the last pass was a ring step.
- `vmc_uni_ring_move(moves)`: after the move is chosen, its token is computed, and its Q, K, V are written.
- Not the same as a full pass once the window is full: an old token keeps what it read when new, so information
  reaches further back than ctx steps; and with learning the ring holds keys from older weights. Accepted.
- Training is unchanged; each update computes its windows from scratch.

## Verification

- `tests/test_age_key.py`: forward, loss and gradients against the reference (C++ and GPU), candidates against full
  passes, checkpoints, invalid configs.
- `tests/test_ring.py`: within the first ctx ticks the ring equals a full pass exactly (move probabilities and
  candidates); past it, it equals a streaming reference in torch, across the buffer's wrap; episode starts; a lifetime.

## Experiment vm-00054

- The best models so far with the age key and the ring: **QGA** (QG: quadrants, gain, age key, ring), seeds 1-3, 60k
  ticks; arms chosen from vm-00052 and vm-00053's results. The control for the age key alone: **QGB** (age key, no ring).
- **Measure:** foraging against QG; ticks per second; the frozen test with and without the ring.

## After: speed

- Profile a tick (acting, world, memory, update) and optimize what dominates.

## Speed

Profile of QN (four pieces) alone on the 3090 (`online.profile`, `VM_PROF` builds), per piece:

| | before | after |
|---|---|---|
| full window pass | 0.91 ms | 0.80 ms |
| ring step | 0.40 ms | 0.32 ms |
| 16 candidates | 0.33 ms | 0.33 ms |
| the ring's move | 0.27 ms | 0.27 ms |
| update, 8 windows | 4.2 ms | **3.45 ms** |

- An update is ~70% of a tick. **Its time goes with the number of GPU calls (~33 µs each), not with FLOPs:** cutting the
  attention GEMMs at the window's middle saved 25% of the arithmetic and doubled their calls, and took 4.1 ms to 4.7.
- What worked: one batched GEMM across heads and windows (`gemm_heads`, 48 calls to 12: 4.1 ms to 3.45); the age key
  folded into the softmax kernels instead of extra GEMMs and kernels; two-hot writes only of the non-zero bins (0.225
  ms to 0.048).
- What did not: a sparse `W_in` backward (5.7 ms: the dense GEMM takes 0.09); a sparse `W_in` forward in training
  (slower than the GEMM; kept for acting only).
- The ring saves ~2 ms of a ~21 ms tick while learning; it is an acting-only gain, as expected.

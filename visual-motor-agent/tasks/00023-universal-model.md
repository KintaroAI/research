# Task 00023: The Universal Model (experiment vm-00019)

**Date:** 2026-10-25
**Status:** In Progress

## Context

`../ARCHITECTURE.md` (2026-10-25) sets the goals: (1) a universal model, (2) good prediction, (3) basic instincts. The
current patch model treats every kind of token differently: a detached move loss, a head per token type, squared error
for pixels and drives with hand-set weights and boosts, a likelihood for moves, the executed move fed into the heads
only, 32×32 residual targets, a hindsight command, babbling. This task builds the universal model the document
describes, next to the existing models (which stay unchanged).

## Design (`model.universal=1`, with `model.patch`; default 0)

- **Tokens per step, in order:** body (hunger, restlessness), the patches row by row (each `patch²·3` pixel values),
  the move made at this step ((dx+1)/2, (dy+1)/2, click). Every token is a value vector of V = `patch²·3` numbers in
  [0, 1] (body uses 2, move 3; the rest are masked), centred to [−0.5, 0.5] at the input, masked entries 0.
- **Input:** `x = W_in · (v − 0.5) + b_in + slot[s] + wpe[step position]`: one shared `W_in` [d, V], one learned
  embedding per slot (body, each patch, move; patch slots start from the 2D sine-cosine code), the step embedding.
- **Attention:** fully causal over the window's tokens, restarting at context starts (episode starts, block starts),
  as now per step.
- **Output:** one shared head on every token's final output: `W_out` [V·K, d], `b_out`: K = `model.bins` (32) logits
  per value. Bin centres `c_k = k / (K − 1)`.
- **Targets:** every token predicts the next token's values: body_t → patch_1,t; patch_j,t → patch_j+1,t;
  patch_last,t → move_t; move_t → the drives after step t (the rollout's next drives).
- **Loss:** two-hot cross-entropy per used value (a value between two centres splits its target between them), the
  mean over the token's used values, then the mean over all tokens of the batch. One number; no coefficients. The
  `ppo.*_coef` and the move/frame/drive options do not apply.
- **Nothing detached; one gradient clip** (`ppo.max_grad_norm`).
- **Outputs for acting and analysis** (`vm_forward_uni`): every token's expected predicted values `Σ_k p_k c_k`, and
  the full bin probabilities of the move predictions (patch_last positions).
- **Parameters:** `uni_in_w`, `uni_in_b`, `uni_slot`, `uni_out_w`, `uni_out_b` appended at the end of the layout;
  `wpe` and the Transformer as now; every other head size 0 in this mode. New random draws only in this mode.
  Checkpoint header slots 40 (`universal`), 41 (`bins`).
- **Stats:** cross-entropy and the squared error of the expected value, per kind of predicted token (body, patch,
  move), appended to `PPOStats`.

## Acting (`lifetime.py`, the universal model)

- The policy: the move distribution predicted at patch_last of the current step; each of dx, dy, click sampled from
  its bins (dx = 2c − 1; click = c > 0.5).
- The instinct (`online.instinct_m`, 16; 0 = off): sample M candidate moves, put each in the current step's move slot,
  read the predicted next hunger from the move token's output; make the candidate with the lowest, keeping the first
  sample when it is within `online.instinct_tol` (0.005) of the lowest. Two batched forward passes per tick.
- No babbling (`online.explore_eps=0`), no command (`online.goal=0`).

## Phases

1. **C++** (`vm_agent.cpp`): the forward pass, the fused head + loss + head gradient (row by row), the backward through
   the Transformer and the embeddings; `vm_forward_uni`; `vm_debug_grads` / `vm_update` route to it.
2. **PyTorch reference** (`reference/torch_reference.py`): the same model and loss, float64.
3. **Tests** (`tests/test_universal.py`): C++ against the reference (loss, every gradient, at 1 and several threads,
   episode starts inside a window), the forward outputs, causality (a token's outputs depend only on earlier tokens of
   its context), the targets by hand on a tiny window, the two-hot encoding, checkpoints; defaults byte-identical.
4. **CUDA** (`vm_cuda.cu`): the same, with the head as one GEMM over all tokens; tests against C++.
5. **Python** (`agent_lib.py`, `cuda_lib.py`, `lifetime.py`): wrappers, sampling, the instinct; a short lifetime test.
6. **Analysis:** the click test from the move token's prediction of the next body; prediction quality per token kind;
   probes of the patch tokens.

## Experiment vm-00019

In vm-00018's world (additive poison, slow hunger, continual torus): the universal model (8×8 patches, 32 bins) with
and without the instinct, against vm-00018's patch model, seeds 1–3, on both GPUs. Measures: ARCHITECTURE.md §10's
questions: frame error where pixels change, cursor probes from the patch tokens, the click test, food and poison clicks
per 1,000 steps, the click rate and move spread.

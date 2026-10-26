# Task 00028: Surprise-Gated Memory for the Universal Model (experiment vm-00022)

**Date:** 2026-10-25
**Status:** Complete (results in vm-00022)

## Context

- The gated memory (vm-00007, `online.GatedMemory`) can store a chunk by relief (`store_relief`), by surprise
  (`store_surprise`: the top fraction of recent chunks) or by chance (`store_base`). vm-00007: in the planner agent,
  storing what came before relief sped up early learning, and an equal-size random memory did not.
- The universal model has used memory U (store everything) since vm-00019. `lifetime.py` refused surprise: its
  surprise came from the old models' separate heads (frame, drives, move).
- In a 200k lifetime with memory U, the first 100k steps are overwritten by the end. Rare events (poison clicks, about
  2% of steps) are rehearsed only as long as they last.

## Design: one rule for every token

- **A step's surprise** is computed in the update that trains on it as the newest window (lag 0): each token's
  cross-entropy (its prediction of the next token: patch, move, or the drives after the move), divided by that slot's
  mean over the last `surprise_window` steps; the step's surprise is the **largest** ratio. No token types are treated
  differently: a step is surprising when any token is much more surprising than usual for its slot.
- **The chunk's surprise** is its largest step's, as before; chunks in the top fraction `store_surprise` of the last
  `surprise_window` chunks are stored.
- **No extra wait for the GPU:** the loss kernel writes each token's cross-entropy to a buffer; the newest windows' last
  step is read on the next tick, after acting, when the GPU is idle (`vmc_uni_token_ce`, `CudaModel.uni_token_ce`).
  The gate decides a chunk 11 steps after it ends, so the one-tick delay does not matter.
- `GatedMemory(tokens=True)`, `note_tokens(ce, kinds)`; the metrics report stored chunks by the kind of the top token
  (patch, move, body) and each kind's mean cross-entropy. GPU only (the universal model on the CPU is too slow to use).

## Verification

- `tests/test_cuda_model.py`: the per-token cross-entropy against the PyTorch reference's per-token loss; acting in
  between leaves it unchanged; too many sequences refused.
- `tests/test_online_memory.py`: the rule by hand (ratios to each slot's mean, the top kind, an unmeasured step
  counting 0, the metrics); the gate stores the top fraction.
- `tests/test_lifetime.py`: a short GPU lifetime with two bodies stores by surprise; the CPU is refused.
- Alignment with a trained model (vm-00021 seed 4 at 200k, 3,000 ticks): the body token's cross-entropy at the click
  step itself is 1.53 for food clicks and 3.13 for poison clicks, against 0.36 elsewhere and 0.92 one step later.

## Experiment vm-00022

The default universal model (`configs/U_universal.json`), 100k ticks, seeds 1–3, three arms:

| Arm | store_surprise | store_base | |
|---|---|---|---|
| U | 0 | 1.0 | store everything (vm-00021) |
| S | 0.25 | 0 | the most surprising quarter of chunks |
| U25 | 0 | 0.25 | a random quarter: the same memory size as S |

Measures: the memory's composition (share of chunks with a food or a poison click), the click test on real windows
(task 00027) and the forage test with and without the instinct at 25k, 50k, 100k, online food and poison clicks, and
the cross-entropy per kind.

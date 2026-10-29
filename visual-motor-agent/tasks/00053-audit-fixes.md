# Task 00053: Fixes From an External Audit of the Architecture Review

**Date:** 2026-10-29
**Status:** In progress

## Context

- An external audit of `ARCHITECTURE_REVIEW.md` against the code at `ab09c39` (brought by the owner, 2026-10-29). It
  found no numerical error in the forward, backward, targets or loss. It reported:
  1. the review's "any patch reaches any other in two hops" is wrong within a step: attention is causal, so
     information flows only forward (to later tokens and later steps);
  2. the move token's three values are predicted independently (a factorized policy);
  3. 64 of the 66 targets per step are patches;
  4. surprise gating with `relief_k = 1` (gate delay D = 0) decides a chunk before its last step's surprise is noted;
  5. with several bodies, the gate counts in `metrics.jsonl` are body 0's only;
  6. `alloc_batch`'s `cudaMalloc`s and the cuBLAS calls are unchecked;
  7. learning is prediction, not reinforcement (by design, with risks).
- Checked against the code: 1, 2, 3, 5 and 6 are as described. 4 is real: at `decide(tick)` the newest step's
  surprise slot still holds a value from `stm` steps ago (it is noted on the next tick). The default configs
  (`relief_k = 12`, D = 11) are unaffected. 6 matters in practice: in vm-00042–00044's analyses, an out-of-memory test
  went on to an illegal memory access, and one hung for 2 hours holding 3 GB.

## Changes

- **Gate timing** (`GatedMemory`, `online.py`): with token surprise (`tokens=True`), the delay is at least one step,
  `D = max(relief_k, goal_k, 2) - 1`, so every step of a chunk is noted before it is decided. Unchanged for D ≥ 1.
- **Gate metrics across bodies** (`lifetime.py`): the counts (chunks, stored by reason, relief seen, not kept, surprise
  tops) are summed over every body's gate; the shared memory's state (chunks held, relief share, age, label share) is
  read once; the surprise threshold and means stay body 0's (named as before). With one body the metrics are the same.
- **Fail fast on the GPU** (`vm_cuda.cu`): a failed device allocation (`Buf::get`, `alloc_batch`) or cuBLAS call
  prints what failed and aborts the process, instead of continuing into kernels with NULL buffers (illegal memory
  access, hangs). Allocation failure was never recoverable mid-pass.
- **Review corrections** (`ARCHITECTURE_REVIEW.md`): information flow is directional (any patch reaches every later
  token, within a step in two hops through the pyramid, and every token of the next step through completed
  summaries); the factorized move and the target shares as findings; the gate and metric fixes.

## Verification

- `tests/test_info_flow.py` (C++ forward, and the GPU when present): changing the last patch of a step leaves every
  earlier token of that step and of earlier steps unchanged and changes the move's predictor; changing the first patch
  changes the move's predictor (full attention and the pyramid with the window); an episode start cuts the window (a
  change before it leaves every token after it unchanged).
- `tests/test_gate.py`: with `relief_k = 1` and token surprise, a chunk whose last step is the surprising one is
  stored (fails before the fix); with two bodies sharing one store, the merged counts are the sum of the gates'.
- The out-of-memory abort: a subprocess asks for far more device memory than exists and must exit with the message,
  not hang (runs only when the GPU is otherwise free).
- Full suite.

## Not changed (owner's decisions)

- **The factorized move** (2): one token, three independent distributions. The instinct scores whole candidates, so
  what is chosen can be correlated. Measure first: how often the instinct overrides the first sample
  (`instinct_override` in `metrics.jsonl`).
- **Target shares** (3): equal weight per token is the universal rule (ARCHITECTURE.md §5, the owner's priority). A
  weight by target kind would be a per-type special case. Measure first: each kind's share of the gradient norm.
- **Prediction, not reinforcement** (7): the project's question ("learns without being reinforced"). The audit's
  proposed measurements (how well the model ranks candidates by their real outcome; the policy without the instinct)
  are tests to add, not changes.

## Results

- `tests/test_gate.py` (4): the surprising-last-step chunk is dropped by the old code (stored chunks ended at 56, 64,
  72, 88; not 96) and stored after the fix; merged metrics sum both gates.
- `tests/test_info_flow.py` (10, CPU and GPU): forward-only flow, the first patch reaching the move's predictor, the
  episode cut, for full attention and the pyramid with the window (with and without thumbnails). The abort test passes
  on the new GPU build and fails on the old one (it printed the message and ran on).
- Full suite: 1,463 passed, 30 skipped.
- The instinct's override rate (EB, EF, EFT; vm-00036, vm-00042): 24–32% of ticks in the first 10k, 2–5% in the last
  25k. The policy's own first sample ends up chosen 95–98% of the time, so it has absorbed the instinct's choices, and
  the factorized move rarely stands between the policy and the chosen move.

# Task 00055: Two Halves That Share the Summaries (experiment vm-00047)

**Date:** 2026-10-29
**Status:** In progress

## Context

- The owner's design (2026-10-29): split the model into two transformers, one per vertical half of the view. Each
  attends to its own patches and to every summary token; the patches are private to a half, the summaries are what
  the halves exchange. A mask for now (the full QKᵀ is still computed; no compute saving).
- The way toward breaking attention's quadratic cost by locality: if a half needs only its own patches plus the
  summaries, the cost of a half grows with its own size, and the summaries carry what is far away.
- Before: `attn_split` (task 00044) knows only the flat layout [body, patches, move] and refuses `uni_pyr`.
- Text world first (vm-00046's setup: NB3, the 2x world with the full pyramid, val next-letter CE 1.926 at 100k).

## Design

- **Regions from the slot table** (`vm_attn_allow` in `vm_agent.h`, `torch_reference.attn_allow`), with `uni_pyr` ≥ 1
  and `attn_split = S` > 1: the patch grid's columns (`attn_cols = 1`) or rows cut into S bands of g / S.
  - a patch is in its band's region;
  - a summary is in every region its block spans (in the 2x text world, g = 16: B1–B3 in one half, B4 in both);
  - the body and the move in every region.
- **The rule:** a token attends (causally, as always) to every summary, the body and the move; to a patch only if
  the patch's region is one of its own. So a patch reads its half's patches and all summaries; a summary reads the
  patches of the regions it spans and all summaries; B4 reads everything.
- The token that predicts the move is the last summary (B4 at the top of the pyramid), global by the rule: no special
  case. The predictor of a line's first letter right of the middle is the last left patch (or a left summary), which
  sees the right half of earlier lines only through the summaries.
- Valid with `uni_pyr` ≥ 1, no `attn_win`, `attn_overlap`, `attn_checker`, `attn_pred_global`; g % S = 0; S ≤ 32.
  The flat `attn_split` path is unchanged.

## Verification

- `tests/test_pyramid_halves.py`:
  - the table: patch to patch within a half only; every token to every summary, the body and the move; B4 to all
    patches; C++ (`vm_attn_allow` through the forward) and the reference agree; rows and columns;
  - information flow (C++ and GPU): changing right patch (r, g/2) of the last step (r even) leaves the left patches of
    row r + 1 unchanged (they come before any summary that reads it), and changes them without the split;
  - the forward with the split matches the PyTorch reference (CPU and GPU); invalid configs refused; the flat
    `attn_split` and models without a split unchanged (their existing tests).
- Full suite.

## Experiment vm-00047

- **VS:** `configs/T_text_B2.json` with `sum_only = 0`, `uni_pyr = 4`, `attn_split = 2`, `attn_cols = 1`; 100k ticks,
  seed 1 (as NB3).
- **Measure:** next-letter CE per value on the val text (the scratch scorer of vm-00046) at 20k, 30k, 40k and 100k,
  against NB, NB2, NB3, NB3a at the same checkpoints (the text lifetime overfits after ~30k).

## Results

- `tests/test_pyramid_halves.py` (14): the table (rows and columns), the flow (a right patch's change does not reach
  the next row's left patches before a summary reads it; it does without the halves), C++ and GPU losses and gradients
  against the reference, invalid configs. Full suite: 1,488 passed, 30 skipped.
- VS's table: 343 slots per step, 62% of a step's causal pairs allowed; 24 ticks/s on the 4090, as NB3 (a mask).

# Task 00044: One Model, Attention Split Into Regions (experiment vm-00037)

**Date:** 2026-10-27
**Status:** Complete (results in vm-00037)

## Context

- The owner's idea (2026-10-27): split the work by token groups, as if two transformers each handled half the tokens,
  with an overlap between the halves, while the model stays one model with one loss. In text this would cut long-range
  context; in our worlds physics is local (circles drift 1–2 px per step, the cursor moves at most 4 px, a patch is
  8 px), so a region may need only its own patches, a border band, and the global tokens.
- Known parallels: local / windowed attention with shifted windows to cross borders (Swin); the thalamus-sorter's
  columns with overlap (parked, `ARCHITECTURE.md`).
- First step (the owner's choice): a mask in the existing model, halves with one patch row of overlap. No new
  parameters, no saving in compute yet (a masked score is still computed); the question is what the model loses.

## Design

- **Regions over the patch grid** (`model.attn_split` = S, 0 or 1 = off, the default; `model.attn_overlap` = O): the
  grid's patch rows are cut into S bands of equal height (for 64 × 64 frames and 8-pixel patches, 8 rows; S = 2: rows
  0–3 and 4–7); each band reaches O rows beyond its edges (O = 1: the top band rows 0–4, the bottom rows 3–7; rows 3
  and 4 belong to both). The body and move tokens belong to every region (global).
- **The rule:** a token may attend to another (causal and within the context, as now) only if their regions intersect.
  Patches of different bands, outside the overlap, never attend to each other directly; a top-band patch reaches the
  bottom band only through an overlap row or through the global tokens (in a later layer or step). It is a rule over
  slots, the same for every step, defined by the patch grid: every world has one (text pages too), no per-world case.
- **Implementation:** a table `allow[q][k]` over the step's token slots (nt × nt, built from S and O), applied where
  attention weights are formed: the CUDA kernels `k_attn_fwd` (short windows) and `k_attn_softmax` (long windows,
  batched GEMM scores), the C++ `attention_fwd`, the PyTorch reference's mask. A disallowed key gets weight exactly 0,
  so every backward pass is unchanged (a zero weight passes no gradient). The instinct's candidate kernel
  (`k_attn_decode`) attends from move tokens, which are global: unchanged. Checkpoint header slots 44 (S) and 45 (O).
- **Validation:** S needs the universal model; the grid's rows divisible by S; 0 ≤ O < rows per band.

## Verification

- Off (S = 0 or 1): the model is unchanged (the existing tests).
- Reference: with one layer, a bottom-band patch's output does not change when a top-band-only patch's input (earlier
  in the window) changes, and does change for an overlap patch or the body; with S = 2 the allow table is as above.
- C++ and CUDA against the reference with the split: loss and every gradient, short and long attention windows, gray
  and in_bins; a kernel that ignores the table fails (mutation).
- A short lifetime with the split.

## Experiment vm-00037

- **World and setting:** the egocentric touch world (`configs/U_touch_ego.json`), no self-forcing, the 1-step instinct:
  vm-00036's EB, which is the full-attention control (3 seeds, already run).
- **Arm ES:** `model.attn_split = 2`, `model.attn_overlap = 1`, 100k ticks, seeds 1–3 (62 ticks/s, three at once).
- **Measures:** in-life prediction (the online patch / body cross-entropies of `metrics.jsonl`); held-out prediction
  (`eval_predict.py`), the patch cross-entropy also by patch row, to see whether rows near the border suffer; the
  frozen forage tests (`tables.py`); the dream curve. Against EB.
- **Arm ESg** (added after ES): ES with `model.attn_pred_global = 1`. In the universal model the move is predicted by
  the last patch (row 7), so ES's policy sees only the bottom half (rows 3–7) and the global tokens; ESg puts that one
  token in both regions (it attends to every patch and every patch to it). Header slot 46.
- **Arm TS, the text world** (the owner's request): `configs/T_text_B.json` (a page of 8 lines of 8 letters, one letter
  per token) with the same split (lines 0–4 and 3–7), the last letter not shared, 100k ticks, seeds 1–3, against
  vm-00030's B with bin embeddings (the same without the split). Val nats per letter every 10k (`eval_predict.py
  --env-set split=val --env-set tile=true`), also by line; the baseline's checkpoints rescored for the lines. A letter
  in one half reaches the other half's lines only through the overlap line and the global tokens (body, move).
- **Arm TV, the text world split vertically** (the owner's request): as TS, but the bands are of the grid's columns
  (`model.attn_cols = 1`: columns 0–4 and 3–7), while reading stays line by line, so every line crosses the boundary
  and a letter sees only its own columns (plus the overlap) of its line and of the lines before. Header slot 47.
- **Then, depending on the result:** no overlap (O = 0, only the global tokens connect the halves), quarters (S = 4),
  and separate weights per region.

## Results (vm-00037)

- Implemented and tested (`tests/test_attn_regions.py`: the allow table; with one layer a bottom-only patch does not
  see a top-only one, does see the overlap and the body; C++ and GPU against the reference, short and long windows,
  `attn_pred_global` and `attn_cols`; mutants that ignore the table in either CUDA kernel or on the CPU fail; the
  GPU-against-C++ matrix for split models; checkpoints keep the flags, header slots 44–47). `eval_predict.py`:
  cross-entropy by patch row and column. `text_dream.py`: the mask in its forward.
- Image: prediction unchanged or slightly better (held-out patch CE 0.772–0.808 against 0.778–0.815, no row worse);
  behaviour collapsed (ES, pair lookahead 24.5 food / 12.7 poison against EB's 44.4 / 4.1) because the move's
  predictor, the last patch, saw only the bottom half; sharing it (ESg) brings poison back to EB's level (5.1), food
  to 31.7.
- Text: by lines (TS) best 1.724 against 1.733 nats per letter and far less overfitting (1.818 against 1.977 at
  100k), only line 0 worse; by columns (TV) 1.847, the cost per column set by how many of the last letters the
  predictor sees (one: 2.41, two: 2.04, three: 1.77–1.79, more: 1.66–1.68). Generation: TS loses line length, TV words.


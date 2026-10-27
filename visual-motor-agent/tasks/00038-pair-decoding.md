# Task 00038: Sharp Decoding That Respects Two-Hot ("pair")

**Date:** 2026-10-26
**Status:** Complete

## Context

- The universal model predicts each value as bins, trained two-hot: a value between two bin centres is split between
  them (grey 21 with 32 bins: 45% bin 2, 55% bin 3). The expected value (`mean` decoding) recovers it exactly.
- `mode` decoding (task 00029) returns the most probable bin's centre: grey 21 becomes 25, grey 94 becomes 90. In the
  circle world 98% of pixels sit between centres, so a `mode` frame is never the world's own levels: one-step
  predictions of real frames are exactly right for 0.9% of pixels (mean: 94%). Fed back in a dream (and in the 4-step
  lookahead of vm-00023), every step's input is a frame of levels the model never saw, and the dream collapses within
  ~10 steps into a tiled texture (the "dream" videos of vm-00031's S seed 2).

## Design

- A third decoding, **`pair`**: the most probable bin b and its more probable neighbour n (b − 1 or b + 1; at the ends
  the only one), the value (p_b c_b + p_n c_n) / (p_b + p_n), c the bin centres. A two-hot target's value is decoded
  exactly from its own distribution; between two distinct values (background or circle) it still chooses, as `mode`.
- Same for every value of every token (no per-world rule).
- `vmc_forward_uni_step(..., mode)`: 0 the expected value, 1 the most probable bin's centre (unchanged), 2 `pair`
  (only the kernel `k_uni_mode_step` changes; the C signature is the same). `CudaModel.forward_uni_step(mode=0/1/2)`
  (bools as before); `online.uni_lookahead` / `instinct_decode` / `--eval-decode` accept `"pair"`.

## Verification

- The GPU's `pair` values equal the PyTorch reference's (the same formula on its logits), near-ties aside.
- A two-hot distribution (hand-built probabilities) decodes to its value exactly.
- Existing `mode` and `mean` results unchanged (their tests).
- On S seed 2 of vm-00031 at 500k: one-step predictions of 200 real frames with `pair` (pixels exactly right against
  mean's 94% and mode's 0.9%), and the dream video re-rendered with `pair`.

## Results

- Implemented (`k_uni_mode_step` mode 2; `online.DECODE`); tests: a two-hot value decodes exactly, the GPU equals the
  reference formula with and without `in_bins` (a kernel with the wrong neighbour fails it); `test_lookahead.py` passes.
- S seed 2 of vm-00031 at 500k, one-step predictions of 200 real frames: pixels exactly right 98.5% (mean 94%, mode
  0.9%).
- The dream (200 real steps, then 200 dreamed): with `pair` the scene stays clean for all 200 steps (no blur, no
  tiles; circles and poison keep their shapes) but nearly freezes; the distractor dots vanish within a few steps; the
  cursor is right for ~5 steps, vanishes at 209, reappears ~14 px off moving in parallel with the real one, and is gone
  after ~320 (in 79 of 200 frames). What remains is exposure bias (never trained on its own predictions), not decoding.

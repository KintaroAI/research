# Task 00033: The One-Hot Control (experiment vm-00027)

**Date:** 2026-10-26
**Status:** Complete (results in vm-00027)

## Context

- vm-00026: the universal model with one letter per token (B, a letter as one grey level) reached 1.80 nats per letter
  on Tiny Shakespeare, the classical GPT of the same size 1.615. B's output is already a softmax over letters, so the
  suspected cause is the input: `W_in` is linear, a letter enters as code × one vector, and all 65 letters lie on one
  line in the embedding space. GPT's embedding table gives every letter its own vector.
- The control, without changing the model: the world draws a letter as **65 values, one of them bright** (one-hot).
  Then `W_in` · values is one column of `W_in` per letter: an embedding table. If the input map is the cause, the
  one-hot arm closes most of B's gap.

## The world: glyphs (`env.encoding`)

- The text world draws each letter as a glyph, a `cell` × `cell` block from a table of one glyph per letter:
  `level` (default, vm-00026: every pixel at the letter's grey level) or `onehot` (pixel k of the cell, row by row, at
  255 for letter k, the rest 0; needs cell² ≥ the alphabet). Letters as font bitmaps (the glyph idea) would be a third
  table. Reading a frame back: the nearest glyph.
- Arm C: `cell` 9 (81 values, 65 used), 8 × 8 letters per page (72 × 72 px, 64 letters per step as in A and B),
  `patch` 9: one letter per token.

## What else changes, and how it is handled

- **The output:** the model predicts each of the 81 values separately, so a letter is 65 yes/no predictions, not one
  softmax. `bins = 2` (the values are 0 or 1; the default 256 would add 254 bins that never occur).
- **Scoring by glyph** (`eval_predict.py`, any world with a glyph table and one glyph per token): the model's
  factorized probability of each glyph, Π over the token's values of p(value = the glyph's value), normalized over the
  alphabet's glyphs; the true glyph's −log is nats per letter. For one-hot this is the standard one-vs-all
  normalization (each letter's odds over the sum of odds). For B's level glyphs (4 equal values) it multiplies 4
  copies of one prediction (over-confident), so B keeps the per-value cross-entropy of vm-00026.
- **The loss's weight:** a token's loss is the mean over its 81 values, most of them easy zeros, so the letter tokens
  weigh less against the move token than in B. Noted, not corrected (it would be a model change).

## Run

Arm C, 100k ticks, seeds 1–3, the same scoring as vm-00026 (val, 16 readers, 208 windows of 512 letters); against B and
the classical GPT of vm-00026.

## Verification

- `tests/test_text_world.py`: one-hot pages read back as the text; the model's values for a token are the letter's
  one-hot vector; glyph scoring equals a hand computation (factorized glyph probabilities, normalized) on a small model.

## Results (vm-00027)

- One-hot letters: val 1.77–1.79 nats per letter at best, against B's 1.80 and the classical GPT's 1.615. They learn
  much faster early (2.05 at 10k against 2.40) but end in the same place: the input map is a cost, not the cause of the
  gap. Next suspect: the lifetime's page-aligned training windows against GPT's random ones.

# Task 00024: Grayscale, a Textured Poison, and Moving Dots for the Universal Model (experiment vm-00020)

**Date:** 2026-10-25
**Status:** In Progress

## Context

- In the universal model (task 00023) each 8×8 RGB patch is 192 values, and the output head (192 values × 32 bins ×
  d 128 = 0.79M of 1.22M parameters) dominates the update. Colour is what tells food from poison today; a texture can do
  that in one channel, at a third of the cost.
- In grayscale the red food and the blue poison have nearly the same level (94 and 94), so the poison needs a pattern:
  a checkerboard disc (every other pixel missing) against food's solid disc.
- The moving grey dots (`env.distractors`, vm-00010) give the patches content and motion the agent does not cause;
  they have been off since vm-00011.
- The user's request: both on by default for the universal model.

## Changes

- **World:** `env.gray` (the frame's RGB replaced by its luma, `(299 R + 587 G + 114 B + 500) / 1000`, in all three
  channels: food 94, poison 94, background 21, cursor 255 / 0, dots their grey) and `env.hazard_pattern="checker"`
  (hazard pixels only where (row + column) is even, in view coordinates; the rest stays background). Both default off;
  existing worlds unchanged.
- **Model:** `model.gray=1` (universal model only): each patch token carries one channel, `patch²` values (64 for 8×8),
  read from the frame's first channel; body 2, move 3 as before. C++, PyTorch reference, CUDA. Header slot 42.
- **The universal model's defaults:** `configs/U_universal.json` holds the universal model and its world: 8×8 patches,
  32 bins, `model.gray=1`; the continual torus, slow hunger, food, additive poison 0.5 with the checker pattern,
  `env.gray`, 8 distractor dots; no babbling, no command; memory U. Experiments start from it.
- **Docs:** ARCHITECTURE.md: grayscale + texture + dots in the world and the token format; the learned-codebook
  embedding idea (below) under "To do later".

## The embedding idea (documented, not built)

As in a text LLM, where a token id selects a row of a randomly initialized, trainable embedding table: give the model a
table of E learned embeddings and a small trainable network that reads a token's values and chooses among them
(softly: a softmax over the E rows, weighting them; or hard, straight-through). The token's content then indexes an
embedding instead of being projected linearly, decoupling "which kind of thing this is" from the vector that represents
it. A step towards the VQ codebook already on the list.

## Verification

- World tests: gray frames are equal in R, G, B and equal the luma of the colour frame; the checker hazard leaves every
  other pixel background; defaults unchanged.
- `tests/test_universal.py`: the gray model against the reference (loss, gradients, forward), V = patch²; invalid
  without universal.
- `tests/test_cuda_model.py`: a gray universal model against C++.

## Experiment vm-00020

The universal model with the new defaults (gray, checker poison, dots), U0 and UI arms as vm-00019 (60k ticks, seeds
1–3), against vm-00019's RGB arms: speed, the click test (food / poison / nothing), prediction cross-entropy, token
probes (cursor, circle; the dots), the frozen tests.

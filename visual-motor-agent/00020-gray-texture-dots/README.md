# vm-00020: The Universal Model in Grayscale, With a Textured Poison and Moving Dots

**Date:** 2026-10-25
**Status:** In Progress
**Source:** *tagged on completion as `exp/vm-00020`* (code: `visual-motor-agent/dev/`)
**Outputs:** `~/data/research/visual-motor-agent/exp_00020/`
**Task plan:** [`../tasks/00024-gray-texture-dots.md`](../tasks/00024-gray-texture-dots.md)

## Goal

Colour costs the universal model three values per pixel; most of the update is its output head over 192 values per
patch. Here the world is grayscale (one value per pixel: 64 per patch), food and poison differ by texture (a solid
disc against a checkerboard disc), and eight grey dots move on their own. Does the model still predict its clicks'
consequences, and see the cursor among the dots?

## Hypothesis

1. The model separates food (−0.30), poison (+0.50) and nothing (0) by texture as it did by colour (vm-00019).
2. The patch tokens hold the cursor despite the dots.
3. A step costs about a third of the RGB model's.

## Method

- **World and model:** `configs/U_universal.json` (task 00024): `env.gray`, `env.hazard_pattern="checker"`,
  `env.distractors=8`, additive poison 0.5, slow hunger, continual torus; the universal model with `model.gray=1`
  (0.68M parameters against 1.22M), 32 bins, 8×8 patches.
- **Arms** (`lifetime.py`, 60k ticks, seeds 1–3, the RTX 4090): G0 (no instinct), GI (the instinct, 16 candidates).
  Comparison: vm-00019's U0 / UI (RGB, solid blue poison, no dots).
- **Measures:** as vm-00019.

## Log

### 2026-10-25
- **Implemented** (task 00024): the gray world and checker pattern (tests), `model.gray` in C++, the reference and CUDA
  (43 reference tests, 100 GPU tests for the universal model; a model reading the wrong channel fails 8), the default
  config. Smoke run: 679k parameters; the frame's levels: background 21, food and poison 94, dots 103–179, cursor 0 /
  255.
- **Fixed on the way:** `online.click_bias` wrote −2.197 into the parameter after the (empty) policy bias of the
  universal model, the first input-map weight; vm-00019's runs have it. Skipped now for models without a policy head;
  a test checks that a lifetime's newborn weights equal a fresh model's.

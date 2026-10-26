# vm-00020: The Universal Model in Grayscale, With a Textured Poison and Moving Dots

**Date:** 2026-10-25
**Status:** Complete
**Source:** `exp/vm-00020` (code: `visual-motor-agent/dev/`)
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
  a test checks that a lifetime's newborn weights equal a fresh model's.- **Launched** at 16:56 on the RTX 4090; all 6 finished by 17:19 (30–78 ticks/s: about 3× the RGB model's per step).

## Results

### R1. The click test at 60k (world: food −0.30, poison +0.50, nothing 0)

| Arm (seeds 1, 2, 3) | food | poison | nothing |
|---|---|---|---|
| G0 (its own policy) | +0.04, −0.05, −0.10 | +0.02, −0.03, +0.02 | +0.01, −0.02, +0.03 |
| GI (+ instinct) | −0.18, −0.12, **−0.25** | −0.10, +0.02, **+0.18** | −0.09, +0.01, −0.02 |

GI seed 3 had poison at +0.41 by 40k (food then only +0.04).

### R2. Tokens at 60k (R²: cursor / circle / direction)

All patch tokens: 0.95–0.97 / 0.99 / 0.88–0.91, with the eight moving dots on screen. The last patch's output: direction
0.24–0.42.

### R3. Behaviour (frozen tests: success / food hits per 1k / poison clicks per 1k)

| Arm | birth | 60k (the run's instinct) | 60k, instinct off |
|---|---|---|---|
| G0 | 0.74–0.88 / 8.3–8.6 / 31–35 | 0.67, 0.68, 0.59 / 3.3, 4.7, 2.9 / 2.0–2.1 | — |
| GI | 0.41–0.99 | 0.73, 0.98, **1.00** / 3.9, 12.6, **31.8** / 3.6, 12.6, 19.7 | 0.53, 0.91, 0.99 / 5.1, 14.9, 26.8 / 6.5, 17.9, **38.9** |

Online, last 10k: G0's food hits 0.6–1.8 per 1k with hunger at the cap 49–72% of the steps; GI's 3.0–16.1, the cap
5–8%.

## Analysis

- **Grayscale with a texture works as well as colour for seeing** (hypothesis 2, supported): the patch tokens hold the
  cursor (0.95–0.97), the circle (0.99) and the direction (0.88–0.91) with eight moving dots, as the RGB model did without
  them. A step costs about a third (hypothesis 3, supported).
- **Food and poison are told apart by texture where the click is learned** (hypothesis 1, partly): GI seed 3 predicts
  food −0.25 and poison +0.18 (+0.41 at 40k); the others food only, or nothing (G0).
- **The instinct acts on a right prediction:** in GI seed 3, with the instinct the frozen agent makes 31.8 food clicks and
  19.7 poison clicks per 1,000 steps; its policy alone 26.8 and 38.9. The instinct halves the poison and adds food: the
  first time an agent here steers away from poison. 31.8 food hits per 1,000 steps is the best of any agent without a
  planner so far.
- **Without the instinct the gray model's own policy stops clicking** (G0: 0.6–1.8 food hits per 1k online, hunger at
  the cap most of the time): exploration does not survive without babbling here, unlike vm-00019's U0. With few clicks
  it never learns what a click does (click effects ≈ 0).

## Conclusions

1. **Grayscale, a textured poison and moving dots cost the universal model nothing in what it sees**, at a third of the
   compute: the new default world.
2. **When the model predicts a click's consequence, the one-step instinct uses it**: in one seed it halves poison clicks
   and raises food clicks to the best rate yet. In the other seeds the prediction is not there yet at 60k.
3. **The model's own sampling does not keep it exploring in every world**: in grayscale, without the instinct, it stops
   clicking. A general mechanism for exploration is needed (e.g. an entropy floor on every predicted distribution).

## Next Steps

- Longer lifetimes; more seeds (the spread between seeds is large).
- A general exploration mechanism (an entropy floor or temperature on all predicted distributions).
- Value tokens on the input.


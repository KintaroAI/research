# vm-00019: The Universal Model

**Date:** 2026-10-25
**Status:** In Progress
**Source:** *tagged on completion as `exp/vm-00019`* (code: `visual-motor-agent/dev/`)
**Outputs:** `~/data/research/visual-motor-agent/exp_00019/`
**Task plan:** [`../tasks/00023-universal-model.md`](../tasks/00023-universal-model.md)

## Goal

The first test of `../ARCHITECTURE.md`'s universal model: one token format, one input map, fully causal next-token
prediction, one shared head, two-hot cross-entropy over 32 bins for every value, every token weighted the same, nothing
detached, full-resolution frames. Goals in order: the universal model works; it predicts well; a basic instinct
(lower hunger) acts through its predictions.

## Hypothesis

1. **Prediction:** the patch tokens keep the scene (the cursor) without the changed-pixel weight; the move token's
   prediction of the next body tells food (−0.30), poison (+0.50) and nothing (0) apart.
2. **No erasing:** with the move loss attached but uniform, the patch tokens hold the cursor as well as vm-00018's
   detached patch model.
3. **The instinct** (one-step lookahead over 16 candidate moves) eats food and avoids poison clicks, and its choices
   become habit: the policy alone (`--eval-instinct 0`) improves over U0's.
4. **Exploration** survives without babbling.

## Method

- **Model** (task 00023): `model.universal=1`, 8×8 patches (66 tokens per step: body, 64 patches, move), 32 bins,
  d 128, 2 layers, 1.22M parameters (the output head 0.79M). C++, PyTorch reference and CUDA; 26 reference tests, 60
  GPU tests, deliberately broken builds fail them; existing models' checkpoints unchanged.
- **World:** vm-00018's (additive poison 0.5, food relieves, slow hunger, continual torus, 2 hazards, no contact cost).
- **Arms** (`lifetime.py`, 60k ticks, seeds 1–3, no babbling, no command): U0 (the policy alone), UI (+ the instinct,
  16 candidates, tolerance 0.005), UIB (+ 32 bodies for 10k ticks, halving every 5k). Control: vm-00018 Q1.
- **Measures:** cross-entropy per predicted kind (body, patch, move) and the patch values' squared error online; the
  click test (the move token's predicted next hunger); token probes (cursor, circle, direction); the frozen episodic
  and forage tests with the run's instinct and without; online food and poison clicks; the instinct's override share.

## Log

### 2026-10-25
- **Implemented** (task 00023). Smoke run (UIB-like, 8 bodies, 3k ticks): every kind's loss falls (patch
  cross-entropy 1.17 → 0.75, pixel squared error 0.031 → 0.008); at 3k the patch tokens read the cursor at R² 0.86 and
  the circle at 0.99. The instinct overrode the first sample on 38–82% of ticks (early predictions differ by more than
  the tolerance even away from circles).

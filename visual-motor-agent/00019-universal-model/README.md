# vm-00019: The Universal Model

**Date:** 2026-10-25
**Status:** Complete
**Source:** `exp/vm-00019` (code: `visual-motor-agent/dev/`)
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
  the tolerance even away from circles).- **Launched** at 16:03; all 9 finished by 17:34 (4–17 ticks/s at first: the universal update, 16 candidate passes
  per tick for the instinct, and six runs on the RTX 3090). Analysis 17:34–18:40.
- **A defect in these runs:** `online.click_bias` (from `AHR_gated.json`) wrote −2.197 into the first weight of the input
  map at birth (the universal model has no policy bias, so the write landed on the next parameter). One weight of
  24,576; fixed for later runs (task 00024).

## Results

### R1. The click test at 60k: predicted next hunger, click minus stay (world: food −0.30, poison +0.50, nothing 0)

| Arm (seeds 1, 2, 3) | food | poison | nothing |
|---|---|---|---|
| U0 (its own policy) | −0.15, −0.22, **−0.29** | +0.04, −0.05, **+0.34** | +0.03, −0.03, +0.08 |
| UI (+ instinct) | −0.01, 0.00, +0.03 | −0.01, +0.03, +0.04 | −0.01, +0.02, +0.03 |
| UIB (+ instinct, 32 bodies) | −0.19, −0.01, 0.00 | 0.00, −0.01, 0.00 | 0.00, −0.01, 0.00 |

U0 seed 3 separates food and poison from 20k on (−0.37 / +0.09 at 20k, −0.41 / +0.38 at 40k). vm-00018's patch model:
Q1 at 40k food −0.12 to −0.22, poison −0.12 to +0.09; at 100k all three outcomes right.

### R2. What the tokens hold at 60k (ridge R²: cursor / circle / direction to the circle)

All patch tokens side by side: 0.96–0.98 / 0.99–1.00 / 0.90–0.93 in every run. The last patch's output (the token whose
prediction is the move): direction 0.22–0.44. For comparison, with the move loss detached (vm-00016): the move token's
direction 0.09–0.25; with it attached in the old patch model (vm-00014 P8A) the patch tokens lost the cursor (R² < 0).

### R3. Behaviour (frozen tests: episodic success / forage food hits per 1k / forage poison clicks per 1k)

| Arm | birth | 60k (the run's instinct) | 60k, instinct off |
|---|---|---|---|
| U0 | 0.70–0.83 / 7.2–8.2 / 32–36 | 0.95, 0.86, 1.00 / 13.7, 12.3, **20.9** / 14.1, 8.8, 23.3 | — |
| UI | 0.75–1.00 / 9–25 / 5–70 | 0.82, 0.91, 0.98 / 5.6, 5.4, 9.4 / 4.3, 5.2, 10.6 | 0.71, 0.86, 0.84 / 4.5, 4.1, 6.5 / 4.8, 5.2, 8.9 |
| UIB | 0.75–1.00 | 1.00, 0.70, 0.34 / 13.1, 3.9, 2.9 / 13.1, 4.3, 3.6 | 0.91, 0.55, 0.45 / 7.0, 2.9, 2.9 |

At birth the model's move bins are near uniform, so it clicks about half the time and hits the circle by chance (and
poison more often: two discs of radius 5–8). Online, last 10k: food hits U0 6.6–15.4, UI 3.4–5.7, UIB 3.4–8.5 per 1k;
the instinct overrode the policy's first sample on 2–6% of ticks; hunger at the cap 0.03–0.21.

### R4. Prediction quality online (last 10k)

Cross-entropy per value: body 0.24–0.28, patch 0.63, move 0.31–1.42; the patch values' squared error 0.0015–0.0020.

## Analysis

- **The universal model works, and nothing erases the scene** (hypothesis 2, supported). With the move loss attached and
  every token weighted the same, the patch tokens hold the cursor, the circle and even the direction to it (0.90–0.93)
  in every run, and the token the policy reads holds the direction better than the detached designs did. The erasing in
  vm-00007/14 came from the move loss's scale, not from attaching it.
- **The cursor is learned without the changed-pixel weight** (hypothesis 1, first half, supported): cursor R² 0.96–0.98.
- **The click consequences are learned slowly and unevenly** (hypothesis 1, second half, partly): one U0 seed gets food
  and poison right by 40k, the other two only food, and the instinct arms hardly any. The instinct arms click less (food
  hits 3.4–5.7 against 6.6–15.4 per 1k), so they see fewer click outcomes to learn from, and a flat prediction gives the
  instinct nothing to choose by: the two may hold each other down. Not tested directly.
- **The instinct does not help yet** (hypothesis 3, not supported here): with flat predictions it overrides the policy on
  2–6% of ticks and changes little (UI 60k with and without the instinct: 5.4–9.4 against 4.1–6.5 food hits per 1k).
- **Exploration survives without babbling** in this RGB world (hypothesis 4, supported here): U0 keeps clicking (6.6–15.4
  food hits per 1k online). vm-00020 (gray) does not.
- **32 bodies hurt behaviour again** (UIB seeds 2–3: test 0.34–0.70), as in vm-00018: most of each batch is the
  policy's own fresh samples.

## Conclusions

1. **The universal model holds the scene with nothing detached**: patch tokens 0.96–0.98 (cursor), 0.90–0.93 (direction);
   uniform losses removed the need for the detach.
2. **Its prediction of click consequences is slower and less reliable than the old patch model's** at 60k; one seed of
   three gets food and poison right.
3. **The instinct needs right predictions to matter;** where the prediction is flat it does nothing.

## Next Steps

- Longer lifetimes (the patch model needed 100k for all three outcomes).
- Value tokens on the input (no overlap between hunger and pixel 0; non-linear intensity), as discussed.
- Many bodies only with a low share of fresh windows.


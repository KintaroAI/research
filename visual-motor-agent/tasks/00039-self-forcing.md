# Task 00039: Self-Forcing — Training on the Model's Own Predictions (experiment vm-00032)

**Date:** 2026-10-26
**Status:** Complete (results in vm-00032)

## Context

- The universal model is trained only on real windows (teacher forcing): every token predicts the next one given real
  tokens before it. Run on its own output (the dream, `dream_video.py`), it drifts: with `pair` decoding (task 00038)
  the still scene survives 200 steps, but the cursor is right for ~5 steps, vanishes at ~9 and comes back displaced;
  the moving dots vanish within a few steps. This is exposure bias: it never saw its own errors as input, so it never
  learned to correct them.
- The 4-step lookahead now helps a little (frozen, 3 seeds: food 21.5 → 24.3 per 1,000), limited by the same drift.
- Self-forcing (scheduled sampling at the frame level; "self forcing" in recent video models): train some windows
  with their last steps replaced by the model's own predictions, while the targets stay real.

## Design

- **The same loss, other inputs.** The update's targets come from the real window, as now; for chosen windows the
  inputs of their last k steps (frame and body) are the model's own predictions. Moves stay real (they are what the
  agent did; the dream is conditioned on them). Every token is treated alike: the rule is "the model's own predicted
  values as the input of the steps it predicted", the same for any world (text too).
- **GPU:** `vmc_set_uni_inputs(h, S, images, drives)` after `vmc_set_batch`: the input frames and bodies of the batch
  (S sequences of ctx steps), used instead of the batch's own for the tokens' inputs; the targets keep the batch's.
  Cleared by the next `vmc_set_batch`. `k_uni_tokens` reads the inputs from one pair of pointers and the targets from
  the other (the same unless set). `CudaModel.set_uni_inputs`, `update_rollout(..., inputs=(img, drv))`.
- **The reference:** `uni_loss(P, cfg, batch)` with optional `input_images` / `input_drives`: the forward pass on
  those, the targets from `images` / `drives`.
- **Dreaming the inputs** (`online.dream_inputs`): for F windows of the batch, a dream length k_i uniform in
  1..`sf_steps`; steps T − k_i .. T − 1 are predicted, in order: the body from the previous step's move token, the
  frame by J Jacobi iterations (`sf_jacobi`) from the previous frame, each value decoded `pair`; the windows' other
  steps and every move stay real. Only replayed windows are dreamed (the newest window(s) keep real inputs: the
  surprise gate reads their cross-entropies), and none with an episode start after its first step.
- **Config:** `online.sf_windows` (F, 0 = off, the default), `online.sf_steps` (3), `online.sf_jacobi` (2). GPU only
  (`ValueError` on the CPU).
- **Cost:** per update at most sf_steps · (J + 1) forward passes of F windows (F = 8, 3 steps, J = 2: 9 passes).

## Verification

- Reference: with inputs equal to the batch, `uni_loss` is unchanged; with other inputs the targets are still the
  batch's (a hand-built check of one token's target).
- GPU against the reference: loss and every gradient with dreamed (perturbed) inputs, gray and rgb, with and without
  `in_bins`; without `set_uni_inputs` unchanged; a kernel reading targets from the inputs fails the test (mutation).
- `dream_inputs`: real steps and moves untouched; with J = patches + 1 a dreamed frame equals sequential decoding
  (patch by patch); the bodies are the previous move token's `pair`-decoded predictions.
- A lifetime with `sf_windows = 0` is unchanged.

## Measure: the dream curve (`eval_dream.py`)

The frozen agent forages (instinct on) in 16 fresh worlds; from 4 points in each life the model dreams 16 steps with
the **real moves** that followed, so every dreamed frame has a true one: per dream step h = 1..16, the share of
pixels exactly right, the mean absolute error, the cursor (its white pixels' centroid; lost below 10 white pixels: a
scoring rule of this world, not the model's) — lost share and distance — and the hunger error. Decoding `pair`,
exact (Jacobi to convergence).

## Experiment vm-00032

The default universal lifetime (vm-00030's U with bin embeddings, `configs/U_universal.json`), 100k ticks, seeds 1–3,
with self-forcing (F = 8, k ≤ 3, J = 2), against vm-00030's runs (the same without). Measures: the dream curve at
100k; one-step prediction (online patch / body cross-entropy 80–100k); the frozen forage and click tests at 100k;
the 4-step `pair` lookahead in the forage test; a dream video of one seed.

## Results (vm-00032)

- Implemented and tested (GPU against the reference with dreamed inputs; a kernel that takes its targets from the
  inputs fails; full Jacobi equals sequential decoding; a short lifetime; 548 tests pass). 29 ticks/s per lifetime.
- The dream: after 16 dreamed steps the cursor is 3.6–5.2 px from the real one against 7.6–18.6 (every seed); pixels
  exactly right better at every horizon; one-step prediction not worse; the cursor is drawn smaller (its place, not its
  drawing, improves). `eval_dream.py`'s cursor rule corrected (share of its white pixels; lost below 3).
- Behaviour no better (seed 1 worse); the 4-step pair lookahead does not gain more. Not a default.

# Task 00046: Attention Split as a Checkerboard (experiment vm-00039)

**Date:** 2026-10-28
**Status:** Complete (results in vm-00039)

## Context

- vm-00037 (task 00044) split attention into two halves of patch rows. Prediction survived. Behaviour collapsed,
  because the move's predictor, the last patch, saw only the bottom half. Sharing that one token (ESg) brought back
  poison avoidance but not all of the food. In text, a split by lines was free, but a split by columns cost 0.11 nats.
- The owner's idea (2026-10-28): split in two as a checkerboard. One half holds the odd patches of the first row and
  the even patches of the second, and so on; the other half holds the rest. Movement is shared. Each half then covers
  the whole view at half the resolution, instead of half the view at full resolution. Try it in the visual world and
  in the text world.

## Design

- **`model.attn_checker = 1`** with `model.attn_split = S`: patch (row r, column c) belongs to region (r + c) % S
  (S = 2: the checkerboard). Two patches attend to each other only if they share a region. The body and the move are
  in every region, so the move is shared. `attn_pred_global` also puts the last patch in every region. There are no
  bands: `attn_overlap` and `attn_cols` must be 0, and S ≤ the grid's side.
- This is the same allow table as task 00044, built in `vm_attn_allow` (`vm_agent.h`: C++ and both CUDA attention
  kernels) and `torch_reference.attn_allow` (the reference and `text_dream.py`). Checkpoint header slot 48.
- `forage_video.py`: a dot marks the centre of each region-0 patch on a checker model's view.
- **What a token sees.** Image: a patch sees its own parity class over the whole view (its diagonal neighbours), the
  body and the move. The last patch (7, 7) predicts the move; it is even, so without `attn_pred_global` the policy
  reads the even half only. Text (8 × 8 letters per page, read line by line): the letter at column c predicts the next
  one. In its own line it sees columns c, c − 2, c − 4 and so on. In the previous line it sees c ± 1, c ± 3. So it sees
  itself (the letter before the predicted one) but not the letter before itself, except at column 0, whose previous
  letter ends the line above in the same region.

## Verification (`tests/test_attn_regions.py`, `tests/test_cuda_model.py`)

- The allow table for S = 2 (the parity of r + c) and S = 3. The body and move are global, and so is the last patch
  with `attn_pred_global`.
- Invalid configs: the checker without a split, with an overlap, with columns, or with S larger than the grid.
- One layer: a patch's output changes when an earlier patch of its parity changes, and does not when one of the other
  parity does.
- C++ and GPU against the reference: loss and every gradient, short and long windows, gray and in_bins, with and
  without pred_global. Also the instinct's candidates against the full pass. Checkpoints keep the flag.

## Experiment vm-00039

- **Image arms**, in the egocentric touch world (`configs/U_touch_ego.json`), with no self-forcing and the 1-step
  instinct, 100k ticks, seeds 1–3:
  - **EC:** checker, move shared (body and move global).
  - **ECg:** EC with `attn_pred_global = 1` (the move's predictor sees both halves).
  - Controls: vm-00036's EB (full attention) and vm-00037's ES / ESg (halves by rows).
- **Text arm** (`configs/T_text_B.json`), 100k ticks, seeds 1–3:
  - **TC:** checker.
  - Controls: vm-00030's base (full), vm-00037's TS (by lines) and TV (by columns).
- **Measures**, the same as vm-00037. Image: the frozen forage tests (instinct off, 1-step, 4-step mean and pair),
  held-out prediction by row and column (`eval_predict.py`), and the dream curve (`eval_dream.py`). Text: val nats
  per letter every 10k, by line and column, and generation scored by the classical GPT (`text_dream.py`).
- Expected: image prediction close to full attention (diagonal neighbours are within a patch). Behaviour better than
  ES, because the move's predictor sees the whole view at half resolution. Text worse than TV, because the letter two
  back from the predicted one is hidden almost everywhere.

## Results (vm-00039)

- Implemented and tested. `tests/test_attn_regions.py` covers the allow table for S = 2 and 3, invalid configs, what a
  patch can see with one layer, C++ and GPU against the reference (with and without pred_global, short and long
  windows) and the checkpoint flag. `tests/test_cuda_model.py` checks the instinct's candidates. Full suite: 1,243
  passed.
- Image: prediction intact (held-out patch CE 0.765–0.790 against EB's 0.778–0.815; no row or column worse).
  Behaviour between ES and EB: EC 27.4 food / 5.2 poison per 1k with the pair lookahead (EB 44.4 / 4.1, ES 24.5 /
  12.7, ESg 31.7 / 5.1). ECg (move predictor in both halves) adds nothing (29.5 / 5.7).
- Text: TC is the costliest split (best 2.073 nats per letter against 1.733 full, 1.724 by lines, 1.847 by columns).
  The second-to-last letter is hidden at every column except 1 (1.90; column 2, which sees letters 0, 3 and 5 back,
  2.27). Generation: 3.54 nats per letter under GPT, words lost.

# Task 00059: Separate Transformers That Share One Body (experiment vm-00051)

**Date:** 2026-10-30
**Status:** In progress

## Context

- vm-00049 and vm-00050 (tasks 00057, 00058) split the view among separate transformers (`model.band_n`), each a
  lifetime of its own. The text world ignores the moves, so the pieces never had to act together. The window
  (`band_win`) cost a lot in text, where near in reading order is not near in space; the visual world is local in space
  and is the right test for it, but there every piece predicts a move and the body makes one.
- The owner's rule (2026-10-30): every piece keeps its own move token; no predefined split of the directions
  (preferably), the division of labour, if any, is learned. No motors drawn into the frame for now.

## Design

- **`online.pieces = 1`** (lifetime.py, with `model.band_n = S` ≥ 2): one lifetime runs all S band models
  (`band_k` = 0..S-1, the config's `band_k` ignored) on one world and one body. Each has its own weights, optimizer and
  GPU model; they share the experience: one memory (they see the same frames, drives and moves; the gate stores
  everything, `store_base = 1`), and every update trains each piece on the same rows. Piece k starts from seed
  + 7919 k (the same seed would start them the same, but for the slot table).
- **Acting:** each piece makes its move as now (`uni_act`: sampled from its own move prediction, its own instinct
  scoring candidates by its own predicted next hunger). The body executes the **mean** of the pieces' moves (dx, dy;
  the click a majority, unused in the touch world). Every piece's move token then holds the move made, so each learns
  what the body did, not only what it proposed.
- **The body's move, `online.piece_rule`** (added at ~17k ticks of the first runs, when the mean was seen to cancel
  hesitant proposals and to make every piece's instinct plan with a move the body does not make):
  - `mean`: the mean of the proposals (the first runs).
  - `max`: per axis, the proposal of the largest magnitude (ties, frequent with binned moves, at random: the first
    version gave ties to piece 0 and was restarted at 1k ticks). The winner's instinct planned with the move made.
  - `gain`: the whole move of the piece whose instinct predicts the lowest next hunger after it (needs the instinct):
    action selection, not averaging.
  - Logged per piece: the share of ticks it made each axis (`won_x_pK`, `won_y_pK`).
- **Logged:** the usual foraging and prediction metrics (prediction stats averaged over the pieces, and per piece
  `uni_ce_*`); per piece the mean proposed |dx|, |dy| and the agreement with the move made (cosine of proposal and
  mean), to see whether a division of labour appears.
- Checkpoints: `ckpt_NNNNNNNN_pK.bin` per piece.

## Verification

- `tests/test_pieces_lifetime.py` (8): the mean move; short lifetimes on the CPU and the GPU with S = 2 (halves) and
  4 (quadrants): every piece saves checkpoints, learns, starts from its own seed, logs its proposals; invalid settings.
  Passed, with the other lifetime suites (109).
- A run without pieces is unchanged: the same final checkpoint bit for bit as the committed lifetime.py on the CPU
  (on the GPU a lifetime is not deterministic even against itself).
- Speeds, all twelve lifetimes on the 3090 together: F 10.8 ticks/s, H 5.6, Q and QW 2.9 (the cost goes with the
  number of models, not their tokens). A Q lifetime alone: 49 ticks/s.

## Experiment vm-00051

- The egocentric touch world with checker poison in 2 × 2 cells (`configs/U_touch_ego_c2.json`), 64 × 64, patch 8
  (8 × 8 grid), `uni_pyr = 3`, `sum_thumb = 1` (vm-00045's XFT), seeds 1–3, 60k ticks.
- **QM, QG** (`make runs2`): quadrants with `piece_rule` max and gain, seeds 1–3, 60k ticks. QW was stopped at 6k
  ticks to make room (it followed Q seed for seed: the window cost nothing visible).
- **F:** one model (no split): the control. **H:** two pieces, vertical halves (`band_n = 2`, `band_cols = 1`).
  **Q:** four quadrants (`band_n = 4`, `band_cols = 2`, 4 × 4 patches each). **QW:** quadrants with `band_win = 1`.
- **Measure:** food and poison per 1k ticks over the lifetime (last 20k); prediction CE (patch, body, move); per piece
  the proposals (do pieces specialize by direction?).
- Hypotheses: H, Q ≈ F in foraging (each piece sees the whole scene through the thumbnails; moves need coarse
  positions); QW ≈ Q (the scene is local). Division of labour: no prior; the mean rewards agreement.

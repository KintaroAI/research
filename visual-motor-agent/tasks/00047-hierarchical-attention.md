# Task 00047: Local Attention With Summary Tokens (experiment vm-00040)

**Date:** 2026-10-28
**Status:** Complete (results in vm-00040)

## Context

- The owner's goal (2026-10-28): compute must be split one way or another. Aim for high locality, and break the
  squared cost of attention, towards N·log N, at a tolerable loss of quality. MoE does not do this: it splits the
  FFN, and attention stays whole.
- What vm-00037 and vm-00039 showed:
  - **Prediction is local.** With row halves or a checkerboard, next-frame prediction did not degrade.
  - **Behaviour needs integration across regions,** not just sight of the whole view. A whole-view move predictor
    (ESg, the checkerboard) did not restore food-seeking.
  - **Text needs the recent letters contiguous.** The checkerboard, which hides the second-to-last letter, cost 0.34
    nats per letter.
- The design that fits: a hierarchy. Each token sees its neighbourhood in full detail, and the rest of the input only
  through summary tokens. Known parallels: Longformer and BigBird (local plus global tokens), H-Transformer-1D and
  Swin (multi-scale), register tokens in ViTs (Darcet et al. 2023), block-recurrent Transformers.
- This task is two levels (patches, then band summaries), a first step. More levels (summaries of summaries) are what
  would make it N·log N. Like the region masks, it is a mask: it measures quality, not speed. A block-sparse kernel
  that skips masked scores comes after, if the quality holds.

## Design

- **Summary tokens** (`model.uni_sum` = R, 0 = none): one summary token after every R rows of the patch grid, in the
  step's sequence. With g = 8 and R = 2 a step is

  `[body, row 0, row 1, S0, row 2, row 3, S1, row 4, row 5, S2, row 6, row 7, S3, move]`

  that is, 1 + 64 + 4 + 1 = 70 tokens. They are placed in the sequence, not at its end, so that later patches of the
  same step can read the summaries of the bands above them. This matters in text: a letter can read the summary of
  the line before.
- **A summary token has no values.** Its input is W_in of nothing (with in_bins: all bins 0), plus its bias, slot
  embedding and step embedding, like any token with no used values. The one rule stays: every token predicts the
  next token's used values. A token whose next token is a summary has nothing to predict, so it carries no loss. That
  is the last patch of each band. Each summary predicts the next token (the first patch of the next band; the last
  summary predicts the move). The loss is the mean over the tokens that have a target. With `uni_sum = 0` nothing
  changes.
- **Local attention** (`model.attn_win` = W, 0 = off). Two patches attend to each other only if they are within W
  rows and W columns of each other (a (2W + 1)² window, in every step of the context). With summary tokens:
  - A patch attends to every summary. Within its own step these are the bands above it (by causality); in earlier
    steps, all of them. This is its coarse view of everything beyond its window.
  - A summary attends to the patches of its own band and to every summary.
  - The body and the move are in every region, as before.
  - Without `attn_win`, summaries attend to everything and are attended to by everything: register tokens.
- **The allow table is directional.** `allow[q][k]` holds whether q may attend to k. It is built by `vm_attn_allow`
  (`vm_agent.h`) and by `torch_reference.attn_allow`. `attn_win` and `uni_sum` exclude `attn_split`.
- **One layout helper per language** gives slot → (kind, index): `vm_uni_slot` in `vm_agent.h` (C++, CUDA) and
  `agent_lib.uni_layout` (Python: nt, the patch slots, the summary slots, the move slot). It replaces every
  "patches are slots 1..nt − 2" assumption:
  - C++: `uni_used`, `uni_values`, `seq_fwd_uni`, `seq_loss_uni` (the targets, the kind, the mean over targets).
  - CUDA: `k_uni_tokens`, `k_uni_twohot`, `k_uni_loss`, the host dimensions.
  - Python: the reference's tokens and loss, `cuda_lib`'s output shapes, `eval_predict.py` (the kinds, patch targets,
    rows and columns), `lifetime.py` (the surprise kinds), `online.py` (imagined frames from predicted values),
    `text_dream.py` (its own forward and generation).
- Checkpoint header slots 49 (`attn_win`) and 50 (`uni_sum`).

## Verification

- With both off, every existing test passes unchanged.
- The layout: slots of patches and summaries for g = 4 and 8, R = 1, 2 and 4. The allow table is as above: a patch's
  window, patch → every summary, summary → its band, body and move global.
- One layer: a patch's output changes when a patch inside its window changes, and does not change for a patch outside
  it in the same step, unless an earlier step's summary carries it.
- C++ and GPU against the reference: loss and every gradient, short and long windows, gray and in_bins, `uni_sum`
  alone, `attn_win` alone, both. The instinct's candidates against the full pass. Checkpoints keep both flags.
- A short lifetime in each world.

## Experiment vm-00040

- **Image**, in the egocentric touch world (`configs/U_touch_ego.json`), with the 1-step instinct, 100k ticks,
  seeds 1–3:
  - **EW:** `attn_win = 1` (the 3 × 3 neighbourhood; body and move global).
  - **EH:** `attn_win = 1` with `uni_sum = 1` (8 row summaries, 74 tokens per step).
  - **ER:** `uni_sum = 1` with full attention (registers: do extra tokens alone change anything?).
  - Control: vm-00036's EB.
- **Text** (`configs/T_text_B.json`), 100k ticks, seeds 1–3:
  - **TW:** `attn_win = 1`.
  - **TH:** `attn_win = 1` with `uni_sum = 1` (a summary after every line).
  - The same settings in both worlds: no per-world tuning.
  - Controls: vm-00030's full attention, vm-00037's TS and TV.
- **Measures**, as in vm-00037 and vm-00039: the frozen forage tests, held-out prediction by row and column, the
  dream; text val nats per letter, by line and column, and generation scored by the GPT. Also the count of attention
  scores allowed per step, against full attention: the cost the mask would save.

## Results (vm-00040)

- Implemented and tested (`tests/test_hier_attention.py`: the layout, the allow table, what a patch sees, invalid
  configs, C++ and GPU against the reference, checkpoints; `test_cuda_model.py`'s *_hier models; the lookahead and
  self-forcing tests with summaries; `tests/test_eval_dream.py`). Mutants that ignore the window or average the loss
  over every token fail. Full suite 1,318 passed.
- A tooling bug found in the analysis: `eval_dream.py`, `dream_video.py` and `analyze_hazards.py click-real` still
  assumed the old slots. Fixed, and `tests/test_eval_dream.py` added.
- Cost: the window keeps 17% of full attention's scores, the window with summaries 33%.
- Text: TH (window + line summaries) best 1.765 nats per letter against full attention's 1.733 (TW, the window alone,
  1.964); generation 1.775 under GPT (full 1.451, TW 2.574).
- Image: prediction unchanged in every arm (held-out patch CE 0.773–0.809); behaviour lost in every arm (pair
  lookahead food / poison: EB 44.4 / 4.1, EW 17.8 / 7.4, EH 24.6 / 14.2, ER 26.5 / 5.2). Summaries with full
  attention (ER) cost 40% of the food: the layout, with the move predicted by a summary, matters apart from locality.

# Task 00058: Separate Transformers That See Only Nearby Summaries (experiment vm-00050)

**Date:** 2026-10-30
**Status:** In progress (phase 1 done; next: the window in the visual world)

## Context

- vm-00049 (task 00057): two separate transformers, each with its own half's patches and **every** summary. Horizontal
  halves lost nothing (next-letter CE 1.719 at best against NB3's 1.737). But every summary in every transformer is
  ~N/3 tokens per transformer: O(N), so the split does not scale to many pieces.
- The owner's question: organize the summaries by neighbourhood to get O(log N) per transformer. The design (as
  Barnes-Hut / the fast multipole method): sharp near, coarse far. At each pyramid level a transformer reads only the
  summaries within w cells of its own piece at that level. Constant per level, log N levels: X = O(log N), and the
  whole tick (N / C pieces of C patches) about N · C instead of N².
- **Why separate models and not a mask first** (the plan said mask first): a mask inside one model shares each summary
  token among the pieces, and a summary that attends to its block's patches carries the other pieces' patches into
  every piece that reads it. vm-00047 (mask, 1.857) against vm-00049 (separate, 2.093) showed the mask is not a
  faithful stand-in. The band filter of task 00057 already makes separate models; the window is one more filter there.

## Design

- **`model.band_win = w`** (with `band_n`): w ≥ 1 = a band model keeps a summary only if it is within w cells of its
  own piece at the summary's level (rows and columns separately: the piece's first and last patch row and column,
  shifted down by the level, widened by w); 0 = every summary (task 00057). Summaries stay at their places in the step
  (after their block's last patch, whether the block's patches are tokens or not).
- **`band_cols = 2`**: square blocks. `band_n = s²` pieces, each g/s × g/s patches; piece k at block row k / s,
  column k % s. (0 rows, 1 columns, as before.)
- One function per side decides a slot: `vm_uni_has_patch`, new `vm_uni_has_sum` (`vm_agent.h`), mirrored by
  `agent_lib.has_patch` / `has_sum` and `torch_reference.uni_has_patch` / `uni_has_sum`. `vm_uni_nt` counts the
  layout. Thumbnails are computed from a slot's level and position, not its index, so nothing else changes.
- Checkpoint header slot 57. Invalid: `band_win` < 0, `band_win` without `band_n`, `band_cols` outside 0..2,
  `band_cols = 2` with `band_n` not a square or s not dividing the grid.

## Counts (the 2× text world: 16 × 16 patches, B1–B4)

| pieces | patches each | summaries, all | summaries, w = 1 (interior / edge) |
|---|---|---|---|
| 2 horizontal halves (vm-00049) | 128 | 85 | — |
| 4 horizontal quarters | 64 | 85 | 49 / 37 |
| 16 blocks of 4 × 4 | 16 | 85 | 30 / 18 (corner) |

A strip spans the full width, so its window is not logarithmic (strips scale as √N); the blocks are the O(log N)
form. On a 256 × 256 grid a block would read ~9 summaries per level, ~75 in all, against ~21,800.

## Verification

- `tests/test_summary_windows.py`: the layout is the band layout without the summaries outside the window (rows,
  columns, blocks; C++ and reference; hand counts as above); blocks keep exactly their own patches; the other pieces
  reach a piece only through the thumbnails it keeps (the shuffle test of task 00057), and a summary outside the window
  changes nothing; C++ and GPU losses and gradients against the reference; checkpoint round trip; invalid configs.

## Experiment vm-00050

- The 2× text world (`configs/T_text_B2.json`, `sum_only = 0`, `uni_pyr = 4`), seed 1, 40k ticks (text lifetimes peak
  at 30–40k), checkpoints every 10k, all on the 3090.
- **Phase 1** (8 lifetimes): **Q** quarters with every summary (`band_n = 4`, `band_cols = 0`), **QW** quarters with
  `band_win = 1`. Q against H (halves, 1.719): does the cut cost with more pieces; QW against Q: does the window cost.
- **Phase 2** (if the window is cheap): **KW**, 16 blocks of 4 × 4 patches with `band_win = 1`, the O(log N) form. Each
  block cuts every line into four, so it also carries vm-00049's vertical cost (a piece blind to its line's start);
  a K control (every summary) separates the two.
- **Measure:** `letter_ce.py --bands`, by column and line, against NB3 and H.
- Hypotheses: Q ≈ H (each quarter's first line has the line above finished as B1s); QW ≈ Q (character-level text is
  local: the line above at full letter resolution, farther lines blurred, should suffice).

## Results

- `tests/test_summary_windows.py` (32). Full suite: 1,553 passed, 30 skipped.
- Phase 1, eight lifetimes on the 3090 together: Q 151 tokens per step, 680,450 parameters, 8.3 ticks/s each; W at
  most 115 tokens, 674,306 parameters, 9.0 ticks/s.

### Next-letter cross-entropy (val), `letter_ce.py --bands`

| run | 10k | 20k | 30k | 40k |
|---|---|---|---|---|
| NB3 (no split, vm-00046) | 1.903 | 1.760 | 1.737 | 1.741 |
| H (halves, every summary, vm-00049) | 1.917 | 1.760 | 1.719 | 1.721 |
| **Q** (quarters, every summary) | 1.990 | 1.761 | 1.724 | **1.703** |
| **W** (quarters, `band_win = 1`) | 2.740 | 2.239 | 2.049 | **1.895** |

By line at 40k (lines 1–2 quarter 0, 3–4 quarter 1, ...):

| run | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|
| Q | 1.825 | 1.695 | 1.686 | 1.661 | 1.773 | 1.682 | 1.675 | 1.630 |
| W | 2.328 | 1.871 | 1.786 | 1.742 | 1.913 | 1.746 | 1.915 | 1.855 |

- Four pieces with every summary lose nothing (1.703 at 40k, still falling, against H 1.721 and NB3 1.741).
- The window costs: 1.895 against 1.703, worse on every line and column, and still falling fast at 40k (−0.15 from
  30k, Q −0.02). Each step is the next page, so line 1 follows the previous page's line 8: the farthest point in 2-D,
  which quarter 0 sees only as B3/B4 blur (line 1: 2.33 against 1.83). More generally a letter keeps exact context
  only for its own quarter and the line above (~8–24 letters) where Q keeps the whole page. Text is a 1-D sequence
  wrapped into 2-D: near in reading order is not near in space, so a spatial window cuts its context.
- Phase 2 (blocks) not started: the window is not cheap in the text world.

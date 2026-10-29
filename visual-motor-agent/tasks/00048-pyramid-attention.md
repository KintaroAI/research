# Task 00048: A Pyramid of Summary Tokens (experiment vm-00041)

**Date:** 2026-10-28
**Status:** Complete (results in vm-00041)

## Context

- vm-00040 (task 00047) used two levels: a 3 × 3 window of patches and one summary per row, read by every patch. In
  text it came within 0.03 nats per letter of full attention with 33% of the attention scores. In the image world
  prediction stayed local but behaviour was lost. Summaries alone, with full attention, already cost 40% of the
  food.
- The owner's choice (2026-10-28): the pyramid. A patch reads fine detail nearby and coarser summaries further out:
  2 × 2 blocks, then 4 × 4 blocks, up to one token for the whole grid. With N patches there are log N levels and a
  bounded number of keys per level, which makes attention N·log N instead of N².

## Design

- **Levels** (`model.uni_pyr` = L, 0 = none): level 0 is the patches (a g × g grid). Level ℓ = 1..L holds one summary
  token per 2^ℓ × 2^ℓ block, a (g / 2^ℓ) × (g / 2^ℓ) grid. For g = 8 and L = 3 that is 16 + 4 + 1 = 21 summaries, and
  87 tokens per step. g must be divisible by 2^L. `uni_pyr` excludes `uni_sum` and `attn_split`.
- **Placement:** a summary comes right after the last patch of its block, in the patches' row-by-row order, so it is
  complete when it is formed. Finer levels come first when several blocks end at the same patch. For g = 8 and L = 3:
  - after patch (1, 1) comes B1(0, 0);
  - after patch (3, 3) come B1(1, 1) and B2(0, 0);
  - after patch (7, 7) come B1(3, 3), B2(1, 1) and B3, the global summary, which predicts the move.
  - The order is the same in every world, so text keeps its reading order.
- **The one rule stays:** summaries have no values, and the token before a summary has no target (vm-00040).
- **Attention** (with `attn_win` = W > 0; without it, the summaries are registers in full attention). Every token
  has a level and a position on its level's grid (a patch: its row and column; a summary: its block's).
  - **Body and move** are global, as before.
  - **Patch to patch:** within W rows and W columns.
  - **Any token to a summary at the same or a coarser level** (ℓ_k ≥ ℓ_q): allowed if the reader's position,
    projected to the summary's level (divided by 2^(ℓ_k − ℓ_q)), is within W of the summary's position. So each token
    reads at most (2W + 1)² summaries per level: fine nearby, coarse far away.
  - **A summary to patches:** every patch of its block, directly. A level-ℓ summary reads 4^ℓ patches, which costs
    N per level and N·log N in all. It reads them directly rather than through the level below, so that with two
    layers any patch reaches any other in two hops: patch → summary (layer 1) → reader (layer 2).
  - **A summary to finer summaries:** none.
- **Implementation:**
  - The slot layout becomes a table, built once per model: kind and index per slot, plus level and grid position.
    It comes from one function in `vm_agent.h` (`vm_uni_layout`), and C++ and CUDA read the table instead of
    recomputing per token. In CUDA it is a small device array.
  - The allow table is built from the same layout.
  - Python: `agent_lib.uni_layout` and `ref.uni_slots` give the same layout, so the tools need no change (they use
    `layout["patch"]`, `layout["move"]` and nt since task 00047).
  - Checkpoint header slot 51.

## Verification

- The layout for g = 4 and 8 and L = 1..log2 g: every summary comes right after its block's last patch, finer levels
  first, the global summary just before the move. Python and the reference agree.
- The allow table: a patch's window; summaries within W at each level; a summary reads exactly its block's patches;
  no summary reads a finer one; body and move are global.
- The cost: allowed scores per window for g = 8, 16 and 32 grow like N·log N (with the pyramid), against N² (full).
- One layer: a patch's output does not change for a patch outside its window. Two layers: it does through a coarse
  summary (this step's, if complete).
- C++ and GPU against the reference: loss and every gradient (short and long windows, gray and in_bins), the
  instinct's candidates, the lookahead, self-forcing, the dream step. Checkpoints keep the flag.
- With `uni_pyr = 0` all existing tests pass unchanged, including vm-00040's `uni_sum`.

## Experiment vm-00041

- **Text** (`configs/T_text_B.json`), 100k ticks, seeds 1–3:
  - **TP:** `attn_win = 1`, `uni_pyr = 3`.
  - Controls: vm-00040's TH (window + line summaries) and TW (window alone), and vm-00030's full attention.
- **Image** (`configs/U_touch_ego.json`, the 1-step instinct), 100k ticks, seeds 1–3:
  - **EP:** `attn_win = 1`, `uni_pyr = 3`.
  - Controls: vm-00040's EH and ER, and vm-00036's EB.
- **Measures** as in vm-00040, plus the attention-score count.
- **Expected:** in text, possibly worse than TH. The blocks are square, not line-shaped, so the end of the previous
  line reaches a line's first letters only through coarser blocks, and those are complete only every second or fourth
  line. In the image world, prediction should stay unchanged; behaviour is open.

## Results (vm-00041)

- Implemented and tested (`tests/test_pyramid_attention.py`, pyramid models in the candidates, lookahead,
  self-forcing and dream tests; a mutant whose summaries read the whole grid fails; full suite 1,372 passed).
- Cost: the pyramid's share of attention scores is 44% at 8 × 8, 16% at 16 × 16 and 5% at 32 × 32 (N·log N); the row
  summaries use 33% at 8 × 8.
- Text: TP best 1.921 nats per letter (TW 1.964, TH 1.765, full 1.733); generation 2.365 under GPT. Square blocks
  do not carry the end of a line to the next line's start (column 1: 2.37).
- Image: prediction unchanged (held-out patch CE 0.773–0.837, seed 2 worst); behaviour as the row summaries' (pair
  lookahead 24.6 food / 6.8 poison against EB's 44.4 / 4.1).

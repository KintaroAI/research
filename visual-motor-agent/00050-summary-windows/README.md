# vm-00050: Separate Transformers That See Only Nearby Summaries

**Date:** 2026-10-30
**Status:** Complete (pending tag)
**Source:** `exp/vm-00050` (code: `visual-motor-agent/dev/`)
**Outputs:** `~/data/research/visual-motor-agent/exp_00050/`
**Task plan:** [`../tasks/00058-summary-windows.md`](../tasks/00058-summary-windows.md)

## Goal

Separate pieces that read every summary hold ~N/3 tokens each: it does not scale. The owner's question: organize the summaries by
neighbourhood for O(log N) per piece. Sharp near, coarse far: at each level a piece reads the summaries within w cells of itself.
What does that cost?

## Method

- **`model.band_win = w`**: a piece keeps a summary only if it is within w cells of the piece at that level. **`band_cols = 2`**:
  square blocks. Separate models rather than a mask, because a mask's shared summaries carry the other pieces' patches
  (vm-00047's mask 1.857, vm-00049's separate 2.093). Tests: `tests/test_summary_windows.py`, 32.
- **Counts,** 16 × 16 patches, B1–B4, 85 summaries in all: quarters with w = 1 read 49 (37 at an edge); 16 blocks of 4 × 4 read 30 (18
  in a corner).
- **Runs:** the 2× text world, seed 1, 40k ticks: Q (quarters, every summary) and W (quarters, `band_win = 1`). Scored with
  `letter_ce.py --bands`.

## Results

### R1. Next-letter cross-entropy (val)

| run | 10k | 20k | 30k | 40k |
|---|---|---|---|---|
| NB3 (no split) | 1.903 | 1.760 | 1.737 | 1.741 |
| H (halves, every summary) | 1.917 | 1.760 | 1.719 | 1.721 |
| **Q (quarters, every summary)** | 1.990 | 1.761 | 1.724 | **1.703** |
| **W (quarters, window)** | 2.740 | 2.239 | 2.049 | **1.895** |

### R2. By line at 40k (lines 1–2 are quarter 0, and so on)

| run | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|
| Q | 1.825 | 1.695 | 1.686 | 1.661 | 1.773 | 1.682 | 1.675 | 1.630 |
| W | 2.328 | 1.871 | 1.786 | 1.742 | 1.913 | 1.746 | 1.915 | 1.855 |

## Analysis

- **Four pieces with every summary lose nothing:** Q is the best text result of its time, and still falling at 40k.
- **The window costs in text:** 1.895 against 1.703, on every line and column, though it is still falling fast (−0.15 from 30k).
- **Why:** text is a 1-D sequence folded into 2-D, so near in reading order is not near in space. Each step shows the next page, so
  line 1 continues the previous page's line 8, the farthest point on the screen, seen only as blur (line 1: 2.33 against
  1.83). A letter keeps exact context for ~8–24 letters instead of the page.
- **In the visual world** the window ran in vm-00051 as QW, but was stopped at 6k ticks to make room for other rules. To then it
  followed Q seed for seed, too early to conclude from. Phase 2, 16 blocks, the true O(log N) form, was never run.

## Conclusions

1. A window on summaries is not free where the content is not local in space. Text measures long context more than locality.
2. Whether O(log N) summaries work in vision is still open.

## Next Steps

- The window with gain or a later rule in the visual world, and 16 blocks. Both need the move rule, which now exists.

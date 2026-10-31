# vm-00045: Position Codes by the Slot Layout

**Date:** 2026-10-29
**Status:** Complete
**Source:** `exp/vm-00045` (code: `visual-motor-agent/dev/`)
**Outputs:** `~/data/research/visual-motor-agent/exp_00045/`
**Task plan:** [`../tasks/00052-slot-position-codes.md`](../tasks/00052-slot-position-codes.md)

## Goal

At birth each slot gets its cell's 2-D position code, the model's only built-in geometry. Since summaries were interleaved (task
00047), the code went to the wrong slot: 40 of 64 patches carried another cell's code. Every summary arm of vm-00040–00044 started
scrambled, and lost about 40% of the food while predicting well. Was this the cause?

## Method

- **The fix** (task 00052): the code is placed by the slot table, a patch at its cell, a summary at its block's centre
  (`vm_uni_center`). Only the initialization changes. `tests/test_slot_positions.py`: the old build fails all 17 cases with
  summaries. Each slot's projection onto its own code is now 0.93–1.00 (before: −0.07 to 1.05, mean 0.68).
- **Runs**, `configs/U_touch_ego.json`, seeds 1–3, 100k ticks: XF (pyramid, full attention), XFT (with thumbnails), and XB, no
  summaries, which is vm-00036's EB on the current code, added because the controls of vm-00043 and vm-00044 fell so far.
  Text: XTP (`configs/T_text_B.json`, window, pyramid).
- The frozen image tests were not run: by the owner's rule, a test is run only if the lifetime numbers improve, and they did not.

## Results

### R1. Life, the last 20k ticks of 100k, food / poison per 1k

| arm | mean | food by seed | food share | with the bug |
|---|---|---|---|---|
| XB (no summaries) | 28.7 / 7.9 | 26.4, 22.5, 37.2 | 0.78 | EB 36.8 / 8.0 (older code; unaffected by the bug) |
| **XF** | 18.1 / 12.2 | 13.5, 16.4, 24.3 | 0.60 | EF 21.8 / 8.9 |
| **XFT** | 12.8 / 5.9 | 17.8, 10.4, 10.4 | 0.68 | EFT 20.1 / 7.2 |

### R2. By 25k ticks, food / poison

| arm | 0–25k | 25–50k | 50–75k | 75–100k |
|---|---|---|---|---|
| XB | 27.4 / 11.7 | 28.8 / 9.8 | 27.5 / 8.9 | 28.8 / 8.0 |
| XF | 18.5 / 12.6 | 18.3 / 11.4 | 19.3 / 11.9 | 17.8 / 12.7 |
| XFT | 11.3 / 9.4 | 10.8 / 7.4 | 11.4 / 6.5 | 13.1 / 5.8 |

### R3. Text

XTP's best: 1.914 / 1.924 / 1.916 (mean 1.918), against TP's 1.921 with the bug.

## Analysis

- **The bug was real, and it was not the cause.** With the codes right, summaries collect no more: XF and XFT are below EF and
  EFT, and text does not change.
- **The control moved:** XB is 22% under EB on the same world. Seed spread (22.5 to 37.2) can hold much of that; code changes
  between them are not ruled out. So earlier gaps against EB were overstated. Against its own control, the summary cost is about
  a third of the food: 28.7 to 18.1.
- **No arm improves through life:** food is flat from the first quarter. Poison falls, food does not.
- Prediction is level across arms, so the cost is in acting, not seeing. Why was not tested here. vm-00051 later showed that
  how the move is chosen decides foraging far more than these layouts.

## Conclusions

1. Keep the fix: the geometry must be right. It does not explain the summaries' cost.
2. Rerun controls on the code of the arms.

## Next Steps

Done since: vm-00051, where summaries with competition between pieces forage better than any model here.

# 00008 — Catastrophic Forgetting

**Status:** Complete
**Source:** `exp/00008` (`d39e9fb`)

## Goal

Characterize what happens when a cell is trained on task A, then switched to task B,
then tested on A again. Can it at least retrain quickly?

## Method

Two tasks with different cluster centers (4 clusters each, 16D, seed 42 vs 100).
Train sequentially: 5k frames on A, 5k frames on B, then measure recovery speed
when retrained on A.

## Results

### Sequential training

| Phase | Task A NMI | Task B NMI |
|---|---|---|
| After training A (5k frames) | 0.857 | 0.423 |
| After training B (5k frames) | 0.289 | 1.000 |

### Retraining speed on A (after B has overwritten it)

| Retrain frames | Task A NMI |
|---|---|
| 0 | 0.289 |
| 100 | 0.652 |
| 500 | 1.000 |
| 1000 | 1.000 |

### Full cycle A → B → A

| Phase | Task A NMI | Task B NMI |
|---|---|---|
| After A→B→retrain A (2k) | 1.000 | 0.531 |

## Analysis

**Catastrophic forgetting confirmed.** Training on B drops task A from 0.857 to 0.289.
The prototypes move entirely toward B's cluster centers, overwriting A.

**Retraining is 10x faster than initial learning.** Only 500 frames to fully recover A
(vs 5000 for initial training). The prototypes have visited similar regions before, so
competitive learning converges quicker — a form of implicit meta-learning.

**But the cell can only hold one task at a time.** Relearning A partially overwrites B
(drops from 1.000 to 0.531). The cell has 4 prototypes and each task needs 4 — there's
no capacity for both simultaneously.

**Connections:**
- Requirements 9-11 (stability-plasticity balance) — the cell is too plastic
- Task 00004 (silence/noise resilience) — same underlying issue: no mechanism to
  protect learned prototypes from being overwritten
- Possible solutions: freeze high-confidence prototypes, expand capacity by adding
  output units, or use separate cells per task context

**The usage-gated learning rate helps but doesn't prevent overwriting.** Frequent winners
learn slower, but when the entire input distribution shifts, ALL prototypes eventually
get pulled toward the new data. A stronger mechanism is needed — e.g., ART-style
vigilance that locks prototypes when match quality is high and stable.

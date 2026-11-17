# vm-00094: The Flexible Layer, W = S + F

**Date:** 2026-11-16
**Status:** Running
**Source:** `dev/` (not yet tagged)
**Outputs:** `~/data/research/visual-motor-agent/exp_00094/`
**Task plan:** [`../tasks/00105-flexible-layer.md`](../tasks/00105-flexible-layer.md)

## Goal

Three ways to remember food that left the 64 × 64 window have failed in this world: the 8-tick context (vm-00091), a 32-tick context (vm-00092), and a state written and read back across windows (vm-00093), which collapsed to a constant when trained through. Here the agent gets the plan's other memory: fast weights. W = S + F, where S learns as before and F learns only from the newest window, fading by 1% a tick (a half-life of 69 ticks). Does a trace of the last ticks in the weights hold where the food went? The design and the readings set before the results are in the task plan.

## Method

| Arm | Label | Model | F | Seeds |
|---|---|---|---|---|
| C8r = vm-00091's VW, rerun | value, γ 0.9, n 4, fresh tail | 4 quarters × 8 layers, tolerance 0.002, context 8 | none | 1–12 |
| F | the same | the same | λ .99, k 1, from the newest window | 1–12 |

- **The flexible layer** (`online.fast = 0.99`, `online.fast_k = 1`):
  - The parameters hold S + F. S takes the usual AdamW step on the batch.
  - Then F ← .99 F − k lr clamp(g / √v̂, −1, 1): g is the newest window's gradient alone, and v̂ is the slow update's Adam second moment.
  - The parameters change by F's change, so S is untouched by it.
  - **k = 1, changed from 10 before the runs, the owner's choice:** at k = 10, F sat at about a third of S's size within 2,000 ticks and prediction was worse; at k = 1 it was 7% and falling.
- **Memory in life** (`online.mem_score = 1`): a frozen test cannot see F, which only lives while learning.
  - Each life scores, every tick, its move's aim at the nearest unseen food by ticks since food was last in view, without the ticks after a meal.
  - It also scores the same moves in a blind world stepped alongside, using `eval_hunger.py chain`'s code (`MemoryScore`).
  - C8 is rerun with the measure (C8r).
- **Code checks** (`dev/tests/test_fast.py`, 3 tests):
  - the fast step against numpy;
  - F decaying as .99^t with no gradient;
  - S = P − F untouched by fast steps;
  - MemoryScore on a hand-made sequence.
  - `chain`'s rows are unchanged by the move to MemoryScore (a rerun of one vm-00093 run matched in every field but the corrected state spread).

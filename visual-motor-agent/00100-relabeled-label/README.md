# vm-00100: The Body Label Re-labeled at Every Replay, with the Full Dose

**Date:** 2026-11-20
**Status:** Running
**Source:** `dev/` (pending tag `exp/vm-00100`)
**Outputs:** `~/data/research/visual-motor-agent/exp_00100/`
**Task plan:** [`../tasks/00112-relabeled-body-label.md`](../tasks/00112-relabeled-body-label.md)

## Goal

In vm-00099 the stored label with the full dose (SD) no longer starved, but still foraged blind: 9.8 food per 1k against MB's 54.2. Its label was right only two or three ticks before a meal. MB differs from SD in three ways: it is re-labeled at every replay, its label is the hunger channel's target, and its cost includes the drift without the full dose. Here only the first changes: SD's label, in SD's channel, re-labeled at every update from each transformer's current prediction. Does storing once cost the foraging? The design and the readings set before the results are in the task plan.

## Method

| Arm | World | Label | Seeds |
|---|---|---|---|
| SD (vm-00099), SB (vm-00098), MB (vm-00097), IB (vm-00095), linked | inertia from birth | stored, full dose; stored, capped; re-labeled in the hunger channel; hunger value | 1–12 |
| RD | inertia from birth | re-labeled at every update, own channel, full dose (`online.label_refresh = 1`) | 1–12 |

- **The re-labeling:**
  - memory keeps the facts: hunger before and after, the episode's end, the cut, and one step past each chunk;
  - at every update each transformer's label of step t is the stored label's rule, with its current prediction (label channel) for the move made at t + 1, from a forward pass without gradient over the window one step later (MB's bootstrap window).
  - The stored labels are still written, as in SD, but not trained on; the monitor `meal_stored` shows them.
- **Everything else is SD's:** `configs/U_view.json`, 4 quarter transformers of 8 layers, tolerance .002, context 8, the ring, `model.body_label = 1`, `online.label = meal`, `label_gamma` 0.95, `online.full_dose = 1`, `env.inertia = 4`.
- **Card:** the RTX 3090, 12 lives at once under MPS, 3.0 ticks/s each (timed; SD ran at 4.1).

## Results

Running.

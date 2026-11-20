# vm-00101: The Instinct's Candidates at Temperature 2

**Date:** 2026-11-20
**Status:** Queued (after vm-00100)
**Source:** `dev/` (pending tag `exp/vm-00101`)
**Outputs:** `~/data/research/visual-motor-agent/exp_00101/`
**Task plan:** [`../tasks/00113-candidate-temperature.md`](../tasks/00113-candidate-temperature.md)

## Goal

The instinct draws its 16 candidates from the model's own move prediction at temperature 1. Once the model is sure of its move, the draws are one and the same move: all 16 are the same on .30 of ticks in SB and .13 in SD (vm-00098, vm-00099), and the instinct has nothing to choose from. Here the first candidate stays the model's own move, and the other 15 are drawn at temperature 2. Does the instinct then find better moves? The design and the readings set before the results are in the task plan.

## Method

| Arm | World | Candidates | Seeds |
|---|---|---|---|
| the base: RD (vm-00100) or SD (vm-00099), linked | inertia from birth | temperature 1 | 1–12 |
| TT | inertia from birth | the first at temperature 1, the other 15 at temperature 2 (`online.instinct_temp = 2`) | 1–12 |

- **The base, by the rule set before vm-00100's results:** RD if RD − SD food at 40–60k, paired by seed, has a 95% interval above 0, else SD (`make base`).
- **Everything else is the base's:** `configs/U_view.json`, 4 quarter transformers of 8 layers, tolerance .002, context 8, the ring, `model.body_label = 1`, `online.label = meal`, `label_gamma` 0.95, `online.full_dose = 1`, `env.inertia = 4` (and `online.label_refresh = 1` on RD).
- **Card:** the RTX 3090, 12 lives at once under MPS.

## Results

Queued.

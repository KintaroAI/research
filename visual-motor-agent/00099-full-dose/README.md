# vm-00099: A Poison Costs Its Full Dose at Any Hunger

**Date:** 2026-11-20
**Status:** Running
**Source:** `dev/` (pending tag `exp/vm-00099`)
**Outputs:** `~/data/research/visual-motor-agent/exp_00099/`
**Task plan:** [`../tasks/00110-full-dose-poison.md`](../tasks/00110-full-dose-poison.md)

## Goal

In vm-00098 the owner's stored label failed in the world with inertia: SB ate 6.4 food per 1k against IB's 19.9, and a third of its lives starved at the hunger cap. Hunger is clipped to [0, 1], and the stored label's poison cost is the rise of hunger, so near the cap a poison costs little and at the cap nothing. Here a poison costs its full dose at any hunger, while what the model senses stays as it was. Does the label then carry value back in time in the world with inertia? The design and the readings set before the results are in the task plan.

## Method

| Arm | World | Label | Seeds |
|---|---|---|---|
| SB (vm-00098), MB (vm-00097), IB (vm-00095), linked | inertia from birth | stored, capped cost; re-labeled; hunger value | 1–12 |
| SD | inertia from birth | stored, full dose (`online.full_dose = 1`) | 1–12 |

- **The full dose:** the world reports how much hunger the clip at 1 cut off (`info["hunger_cut"]`, 0 below the cap). Memory keeps it, and the stored label counts it in the rise, so a poison always costs 0.5. The monitors and the frozen `value` test count it the same way.
- **Everything else is SB's:** `configs/U_view.json`, 4 quarter transformers of 8 layers, tolerance .002, context 8, the ring, `model.body_label = 1`, `online.label = meal`, `label_gamma` 0.95, `env.inertia = 4`.
- **Card:** since the reboot of 2026-11-20 only the RTX 3090; 12 lives at once under MPS, 3.9 ticks/s each (timed).

## Results

Running since 2026-11-20 08:31.

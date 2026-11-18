# vm-00097: A Meal Label, Learned Back in Time

**Date:** 2026-11-18
**Status:** Running
**Source:** `dev/` (pending tag `exp/vm-00097`)
**Outputs:** `~/data/research/visual-motor-agent/exp_00097/`
**Task plan:** [`../tasks/00108-meal-label.md`](../tasks/00108-meal-label.md)

## Goal

The default agent's value label is roughly how hungry it is × how soon it eats, so its differences between moves grow with hunger. In the world with inertia that made the instinct's fixed tolerance relatively small (vm-00095, presumed). Here the label says only how soon the next meal comes: a meal is worth 1, and every earlier tick is worth 0.95 × its transformer's own prediction for the next one, less the hunger the tick added. The meal's value travels back in time through one-step predictions. Does the agent learn it, and does it forage better with inertia and no worse without? A second batch weighs the label by hunger when choosing. The design and the readings set before the results are in the task plan.

## Method

| Arm | World | Label | Choosing | Seeds |
|---|---|---|---|---|
| IB (vm-00095, linked) | inertia from birth | hunger value (γ .9) | lowest label | 1–12 |
| C8r (vm-00094, linked) | no inertia | hunger value (γ .9) | lowest label | 1–12 |
| MB | inertia from birth | meal label (γ .95) | lowest label | 1–12 |
| M0 | no inertia | meal label (γ .95) | lowest label | 1–12 |
| MBN | inertia from birth | meal label (γ .95) | weighted by hunger | 1–12 |
| M0N | no inertia | meal label (γ .95) | weighted by hunger | 1–12 |

- **The label** (`online.value_target = meal`, `value_gamma` 0.95, `value_n` 1, `value_refresh` 1):
  - a tick whose hunger falls is a meal, worth 1;
  - any other tick is worth minus the hunger it added, plus 0.95 × its transformer's own fresh prediction for the next tick;
  - clipped to [−1, 1] and stored as (1 − V)/2, so that lower is better, as hunger.
- **Weighted by hunger** (`online.need = 1`): the instinct's tolerance is divided by max(hunger, .05).
- **In-life check** (the owner's request): each life logs its predictions against what really happened, by ticks to the next meal (metrics `meal`). This shows how far back in time the label is right, and whether it turns optimistic.
- **Cards:** MB and M0 first (24 lives), then MBN and M0N, each batch on both cards under MPS, started automatically after vm-00096.

## Results

Running.

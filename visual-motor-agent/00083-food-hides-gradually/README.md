# vm-00083: The Food Hides Gradually

**Date:** 2026-11-04
**Status:** Queued after vm-00082's and vm-00081's frozen tests
**Source:** `dev/` (not yet tagged)
**Outputs:** `~/data/research/visual-motor-agent/exp_00083/`
**Task plan:** [`../tasks/00094-food-hides-gradually.md`](../tasks/00094-food-hides-gradually.md)

## Goal

In vm-00081, with the food hidden from birth, the agent never learned to forage, even at 16 frames, where a walker that forgets at once loses nothing. A random-moving agent eats a food it can still see in only 15% of its meals at 16 frames and 6% at 4, so the link between the food's image and eating has little to grow from. Here the food is always seen for the first 20k ticks and then hides after fewer and fewer frames (the owner's suggestion). Does the agent keep foraging as the food hides, and does it learn to remember where it was? The design, arms and readings set before the results are in the task plan.

## How it works

- `online.env_sched = {"food_show": [[tick, value], ...]}`: at the start of each listed tick the life calls `env.reconfigure(food_show=value)`.
  - A world lists the settings it can change mid-life (`CircleWorld.RUNTIME`: only `food_show`, read afresh every frame).
  - Other keys and bad ticks or values are refused at the start. The metrics carry `env_food_show`.
- `tests/test_env_sched.py`: a world switched at step 200 is the always-seen world, frame for frame, before the switch and the world born hidden after it.
- A 300-tick life switched to 8 at tick 101 and to 2 at 201 logs 0, 0, 8, 8, 2, 2 at its six 50-tick reports.

## Arms

| Arm | `food_show` over the life |
|---|---|
| G4 | always seen to 20k; 64, 32, 16, 8 for 4k ticks each; 4 from 36k to 60k |
| G1 | as G4 to 36k; 2 at 40k; 1 from 44k to 60k |
| J4 | always seen to 20k; 4 from 20k |

References, not rerun: vm-00073's K4 (always seen, 70.0 food per 1k at 40–60k); vm-00081's H4 (20.1) and H1 (22.0), hidden from birth.

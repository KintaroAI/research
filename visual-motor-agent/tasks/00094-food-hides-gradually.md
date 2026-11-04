# Task 00094: The Food Hides Gradually (experiment vm-00083)

**Date:** 2026-11-04
**Status:** In progress

## Context

- vm-00081: with the food hidden from birth (shown 16, 4 or 1 frames), the agent never learns to forage. Food stays at 20–22 per 1k ticks, the blind walker's level, in all three arms, against 70 for K4 with the food always seen.
  - At show 16 the "forgetful chaser" walker is as good as one with perfect memory, so this is not a memory limit.
  - An untrained agent moving at random eats a food it can still see in only 15% of its meals at show 16, and 6% at show 4. It seldom gets to see "food on screen, then hunger drops", so the link between the food's image and eating has little to grow from.
- The owner (2026-11-04): "probably food can start to hide progressively - first always visible, then it hides after x amount of ticks and x gradually decreases over time".

## Design

**The mechanism (world-agnostic): `online.env_sched`.** World settings changed at given ticks of the life, as `{"key": [[tick, value], ...]}`.
- At the start of each listed tick, the life calls `env.reconfigure(key=value)`.
- A world lists the settings it can change mid-life (`RUNTIME`). CircleWorld lists only `food_show`, which is read afresh every frame. A key a world does not list is refused at the start, as is a value its config would refuse.
- The food's age counts on whether it is shown or not. When x drops, a food older than the new x vanishes at once.
- The saved `env_config` is the life's starting world, so the frozen tests choose their own `food_show`.
- The metrics carry the current value (`env_food_show`).

**Arms (the default model, K4, log summaries, the default world, six seeds, 60k ticks):**

| Arm | `food_show` over the life |
|---|---|
| G4 | always seen to 20k; then 64, 32, 16, 8 for 4k ticks each; 4 from 36k to 60k |
| G1 | as G4 to 36k; then 2 at 40k; 1 from 44k to 60k |
| J4 | always seen to 20k; 4 from 20k (a jump, not gradual) |

- **References, not rerun:** vm-00073's K4 (always seen) and vm-00081's H4 and H1 (hidden from birth).
- 20k ticks: by then K4 forages at about 42 per 1k and is still rising.

**Placement:** 18 lives, seeds 1–3 on the 4090 and 4–6 on the 3090, after vm-00082's and vm-00081's frozen tests and before the fact experiment's stage 2 (vm-00081's 18 lives of this size took 70 minutes).

## Measures

1. **Lived:** food and poison over ticks 50–60k, when every arm is at its last value; food by 4k block across the schedule.
2. **Frozen (`eval_hidden.py aim`, shows 4 and 1):** aim by time since the food vanished, for G4, G1 and J4. The reference is K4 frozen at the same shows, from vm-00081's frozen tests.

## The readings, set before the results

1. **Plumbing:**
   - The world's `food_show` is the schedule's at every tick.
   - Before 20k, G4's world is the always-seen world, tick for tick.
   - A key the world cannot change is refused.
2. **Learned first, kept after:** G4's lived food at 50–60k is above H4's by more than its interval.
3. **Gradual beats a jump:** G4's lived food at 50–60k is above J4's by more than its interval.
4. **Memory grew:** frozen at show 4, G4's aim 2 and 3–4 ticks after the food vanished is above K4's by more than its interval.

## Implementation

- [x] `CircleWorld.RUNTIME`, `CircleWorld.reconfigure` (wrappers such as delayed copies forward it to the world inside, which draws)
- [x] `online.env_sched` (default none), its checks and the metric in `lifetime.py`; a 300-tick life switched at 101 and 201 logs 0, 0, 8, 8, 2, 2
- [x] `tests/test_env_sched.py` (7 tests)
- [x] `00083-food-hides-gradually/`: README, Makefile; queued after vm-00081's frozen tests, before the fact experiment's stage 2
- [ ] Runs, frozen tests, write-up

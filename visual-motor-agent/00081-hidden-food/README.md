# vm-00081: Hidden Food

**Date:** 2026-11-03
**Status:** Starting after vm-00080's frozen tests
**Source:** `dev/` (not yet tagged)
**Outputs:** `~/data/research/visual-motor-agent/exp_00081/`
**Task plan:** [`../tasks/00092-hidden-food.md`](../tasks/00092-hidden-food.md)

## Goal

How long can the agent remember? The food is drawn only in its first few frames after it appears; then it is invisible until eaten, though touching it still feeds. In the egocentric view the food never leaves the view, so remembering it means keeping track of it through one's own moves. The design, arms and readings set before the results are in the task plan.

## How it works

- `env.food_show = v`: the food is drawn in its first v frames after it appears, then not at all until it is eaten. Poison and dots stay visible. `v = 0` is the old world: identical to the committed world over 5,000 ticks, frames and drives.
- A full step moves 4 pixels on each axis. The next food after a bite is about 15 pixels away, 3 to 4 ticks of travel.

## Walkers, no model (`eval_hidden.py walk`)

32 worlds, 6,000 ticks, food per 1,000 ticks:

| `food_show` | Blind | Forgets at once | Remembers perfectly |
|---|---|---|---|
| 16 | 22.8 | 272.0 | 272.0 |
| 4 | 22.6 | 56.0 | 272.0 |
| 1 | 22.6 | 24.7 | 272.0 |

At 16 forgetting costs nothing: every trip ends before the food vanishes. At 4, memory is worth a factor of 5, and at 1 nearly everything. vm-00073's K4, raised with the food always seen and tested at `food_show = 4` (one seed, 400 ticks), aims at the food 1 tick after it vanished (.25) and not after, and forages below a blind walk (18).

Reading 4 compares shares: the forgetful walker keeps .21 of the perfect one's food at `food_show = 4`, and H4 must keep more than .21 of K4's always-seen food.

## Method

- **Arms:** H16, H4, H1: the default model (4 quadrant transformers, K4, log summaries), `env.food_show` 16, 4, 1, six seeds, 60k ticks; seeds 1–3 on the 4090, 4–6 on the 3090.
- **Control:** vm-00073's K4, always seen, not rerun; also tested frozen with the food hidden.
- **Conditional:** vm-00080's D0 and DX with `food_show = 4` (C0, CX), four seeds, if vm-00080 shows DX using its past.
- **Frozen (`eval_hidden.py aim`):** aim by time since the food vanished, and food per 1,000 ticks.
- **Checks passed before the start:** `tests/test_hidden_food.py` (the food drawn in exactly its first v frames over 600 ticks of chasing, everything but its pixels the always-seen world's, `food_show = 0` unchanged, bad values refused); the world tests and the JavaScript equivalence unchanged.

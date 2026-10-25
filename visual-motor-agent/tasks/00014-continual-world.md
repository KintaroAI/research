# Task 00014: A Continual World: Respawning, Moving and Small Circles, and Hazards (experiment vm-00011)

**Date:** 2026-10-25
**Status:** Complete (results in vm-00011)

## Context

So far every world is episodic: a hit or 200 steps end the episode and the world resets. The owner asked for a
richer world: no reset on a hit (a new circle pops up), moving circles, smaller circles, circles that increase
hunger when they touch the agent, and later an egocentric view. With the GPU (tasks 00011–00013) a 50k lifetime of
the basic agent takes about 6 minutes, so many lifetimes per world are affordable.

The agent is vm-00008's DMI (the basic agent that aims: a detached policy head with a hidden layer, inverse
dynamics, the hindsight hunger command, memory U). Nothing in it refers to episodes except the context restarts,
so it should carry over.

## World options (`world/circle_world.py`, all off by default; tests in `tests/test_world_ext.py`)

- `env.respawn`: a hit circle is replaced at once by a new one, never on the cursor. With `env.end_on_success=false`
  and a large `env.max_steps` the world never resets: one continual lifetime.
- `env.target_speed`: the circle drifts at this speed (px/step) and bounces off the walls.
- `env.hazards`: blue circles that drift and add `env.hazard_cost` to hunger per step while the cursor is inside
  one. They never end anything.
- Smaller circles: `env.radius_min`, `env.radius_max`.
- Separate random streams, so the first circle, the cursor and the drives are the same with or without them.

## Metrics (`online.py`)

- Online: hits per 1,000 steps (`step_hits`), the share of steps inside a hazard, hunger.
- `online.py --eval` in a continual world adds `eval_forage`: 32 worlds × 1,000 steps of the frozen agent, hits per
  1,000 steps and the share of steps in a hazard. The episodic test (first hit within 200 steps) is kept, in the
  same world (motion, hazards and radii kept).

## Arms (DMI, GPU, 60k steps, seeds 1–3)

| Arm | World |
|---|---|
| C | continual: respawn, never reset |
| CM | continual + moving circles (0.5 px/step) |
| CS | continual + small circles (radius 6–8) |
| CH | continual + 2 hazards (radius 5–8, 0.75 px/step, 0.02 hunger per step inside) |

Reference: the episodic world (DMI-U, task 00013: 8 GPU seeds).

## Questions

1. Does the basic agent learn to forage in a world that never resets? Hits per 1,000 steps against its birth.
2. Moving and small circles: harder to aim at; does it still learn?
3. Hazards: does it learn to avoid them? It has no reward and no pain signal, only hunger, and the hindsight command
   labels moves before relief, not moves that avoid a rise. Expectation: no avoidance beyond chance. That would be
   a clear case for a drive-aware objective (the planner minimizes predicted hunger, which hazards raise).

## Later

- The planner agent in these worlds (needs latent dynamics on the GPU, then `plan.py`).
- The egocentric view (a wrap-around world centered on the cursor).

# Task 00092: Hidden Food, How Long Can the Agent Remember? (experiment vm-00081)

**Date:** 2026-11-03
**Status:** In progress

## Context

- The owner (2026-11-03) asked how to test more complex behaviour, such as remembering and recalling, or several steps to get food; then "yes write a task for hidden food and start after 80".
- vm-00057 (the delayed cue) failed even when the cue stayed on: an arbitrary rule linked over a delay was too hard for learning from hunger alone. So this test keeps the agent's own goal, food, and only takes the sight of it away.
- What the model can carry: 8 ticks of context, K4's 4 ticks of hunger ahead, the 1-step instinct, and no recall from memory while acting. vm-00080's delayed copies add sight back to 16 ticks.

## Design

**The world:** `env.food_show = v`: the food is drawn only in its first v frames after it appears, then not at all until it is eaten. Touching it still feeds.
- `v = 0` (the default) is the old world: the draw is the same code path, and the age counter changes nothing else.
- Poison and the dots stay visible.
- `info["gt_food_seen"]` says whether the food is drawn, for scoring only.
- The view is egocentric and covers the whole torus, so the food never leaves the view. Once hidden, its place moves only with the agent's own moves: remembering it means dead reckoning.
- A full step moves 4 pixels on each axis. The gap to the next food right after a bite is about 15 pixels (vm-00072's walkers), 3 to 4 ticks of travel.

**Arms (the default model, K4, log summaries, the default world, six seeds, 60k ticks):**

| Arm | `food_show` | The food is seen |
|---|---|---|
| H16 | 16 | for most of a trip |
| H4 | 4 | for about one trip's length, then from memory |
| H1 | 1 | one frame, the rest from memory |

- **Control:** vm-00073's K4 (always seen), not rerun.
- **Conditional arms:** D0 and DX of vm-00080 with `food_show = 4`, four seeds, only if vm-00080's reading 4 is met (blanking DX's past lowers its food). The decision is made from vm-00080's frozen tests before the start. They test whether seeing 16 ticks back carries the memory.

## Measures

1. **Lived:** food and poison over the last 20k ticks, against the control.
2. **The forgetting curve (`eval_hidden.py aim`):** frozen, the cosine between the move and the direction to the food, by time since the food vanished: seen, 1, 2, 3–4, 5–8, 9–16, 17–32, over 32 ticks. Also food per 1,000 ticks.
3. **Walkers that bracket memory (`eval_hidden.py walk`, no model):**
   - blind;
   - a chaser that forgets at once: it chases while the food is seen and walks blind after;
   - a chaser with perfect memory, always chasing.
   The model's food against these says how much memory it shows.

## The readings, set before the results

1. **Plumbing:**
   - `food_show = 0` is the old world, frame for frame and drive for drive.
   - With `food_show = v`, the food is drawn in exactly the first v frames after each appearance.
   - Everything but the food's pixels is the same as the always-seen world in lockstep.
2. **Memory, a little:** in H1 and H4, frozen aim 1–2 ticks after the food vanished is above aim after 32 ticks, by more than its interval.
3. **Beyond the context it is gone:** in H1, aim 17–32 ticks after the food vanished is within its interval of aim after 32 ticks.
4. **More than forgetting:** H4 keeps a larger share of the always-seen food than a walker that forgets at once. The walker keeps 56.0 of 272.0, .21 (the walkers, before the runs); H4's lived food over K4's (70.0) must be above .21 by more than its interval. A share, because the model forages far below the perfect chaser even when it sees everything, so raw food against the walker would ask much more than memory. (Revised before any run: it first compared raw food.)

## Implementation

- [x] `world/circle_world.py`: `food_show`, `food_age`, `drawn()`, `gt_food_seen`
- [x] `tests/test_hidden_food.py`
- [x] `eval_hidden.py`: aim, walk, show
- [x] `00081-hidden-food/`: README, Makefile; runs queued after vm-00080's frozen tests
- [ ] Runs, frozen tests, write-up

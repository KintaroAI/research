# Task 00108: A Meal Label, Learned Back in Time (experiments vm-00097 on)

**Date:** 2026-11-17
**Status:** In progress (approved 2026-11-17, the owner: "what you wrote in next message I like as it states there"; "run these runs automatically ones current runs are done - I probably be sleeping"). Experiment vm-00097, `00097-meal-label/`, queued to start when vm-00096's frozen tests end.

## Context

- **What the move token learns to predict now** (the "next body's" hunger channel): the bootstrapped value of tasks 00096 and 00098. That is the discounted average of future hunger, (1 − γ) Σ_{i<4} γ^i h_{j+i} + γ⁴ V_{j+4}, with γ .9 and a fresh tail averaged over the four transformers.
- **What that label is, written out over the world's events** (hunger rises 0.001 a tick, a meal resets it to 0, a poison adds 0.5):

  label = current hunger + 0.009 − Σ_meals 0.9^k × (hunger at that meal) + Σ_poisons 0.9^k × 0.5

  So it is roughly *how hungry I am × how soon I eat*, plus a poison penalty. Its differences between moves grow with hunger.
- **What that cost in vm-00095:** with inertia, hunger ran about 5 times higher, so labels and the instinct's candidate spread were 5 times larger, while its tolerance stayed .002. The instinct then overrode 58–70% of moves (C8r 27%); this is the presumed cause there, not tested.
- **The owner's idea (2026-11-17):** a label that says only how soon the next meal comes. A meal is worth 1. Every earlier tick's label is γ × the model's own prediction for the next tick, so the meal's value travels back in time through one-step predictions (temporal-difference learning). The scale is the same in every world and at every hunger.

## The label

For the move made at tick j, with the hunger h before it and h′ after it:

- **The event at j + 1,** read from the hunger alone, as the present label is (the agent knows only its body):
  - a meal if hunger fell (h′ < h): r = 1;
  - otherwise r = −(h′ − h), the hunger it added. A poison costs its 0.5 (less at the cap of 1); the drift of 0.001 a tick is a tiny cost per tick.
  - The owner's choice 1: poison is a penalty.
- **The target:** V_j = r + γ × V̂_{j+1}, except after a meal: V_j = 1 (the value is "the path to the next meal"; the next meal starts anew).
  - **V̂_{j+1}** is this transformer's own prediction of the label of the move actually made at j + 1, made fresh before each update (owner's choice 4: each transformer keeps its own version of the label; vm-00087: a fresh tail removed the optimism of the stored one).
  - **γ = 0.95** (owner's choice 5): 0.95^k is .60 at 10 ticks, .36 at 20, .13 at 40.
  - **One step** (owner's choice 3: the simple approach first). The meal's value moves back one tick each time a stretch is replayed. The hindsight version, exact values from the stored future, is in ARCHITECTURE.md section 9 for later.
- **On the model's scale:**
  - V is clipped to [−1, 1] and stored as T = (1 − V) / 2 in the body's hunger channel, so that lower is better as now: T = 0 is a meal on the next tick, .5 is nothing in sight, 1 is deep in poison.
  - The instinct is unchanged: it makes the candidate with the lowest predicted T.
- **What the tolerance then means:** a candidate that brings the meal one tick closer, k ticks away, lowers T by .025 × 0.95^k. The tolerance .002 is that difference out to about 49 ticks, in every world and at any hunger.

## Need (the owner's choice 2)

- **The problem:** the label alone makes a meal worth the same full or starving, so the agent would eat all the time.
- **The owner's answer** (confirmed 2026-11-17: "label weighted with hunger"): weigh the label by hunger when choosing the move, not when learning it. The predicted differences between candidates are multiplied by the current hunger, which is the same as dividing the tolerance by it: tolerance / max(hunger, .05).
  - Hunger is the same for every candidate, so with one label the weighting does not change which candidate is best. It changes only whether the instinct overrides the model's own move.
  - With two labels (meal and poison apart, for later), hunger × meal − poison would also change which candidate wins: food against safety.
  - Sated, the instinct rarely overrides the model's own move; hungry, it steers.
- **A known cost:** one label holds both meal and poison, so weighing it by hunger also weakens poison avoidance when sated. Two labels, meal and poison apart, would keep avoidance at full strength (for later, ARCHITECTURE.md section 9).
- **For later (the owner):** moving should cost energy. Hunger should rise with movement, so that standing still becomes an option and eating all the time has a price. With this label such a cost needs no change to the label: it is a rise in hunger, so it is a penalty.

## Arms (proposed)

| Arm | World | Label | Choosing | Seeds |
|---|---|---|---|---|
| IB (vm-00095, linked) | inertia from birth | hunger value (γ .9) | lowest label | 1–12 |
| C8r (vm-00094, linked) | no inertia | hunger value (γ .9) | lowest label | 1–12 |
| **MB** | inertia from birth | meal label (γ .95) | lowest label | 1–12 |
| **M0** | no inertia | meal label (γ .95) | lowest label | 1–12 |
| **MBN**, **M0N** (second batch) | as above | meal label | need-weighted | 1–12 |

- **Everything else is C8's** (the ring, H = 1, tolerance .002).
- **First batch:** MB and M0, 24 lives, about 4 h.
- **Second batch:** the need-weighted arms, if the label learns.
- **The comparisons are by mean hunger, the agent's own objective, as well as food and poison:** need weighting is meant to eat less when sated.

## The readings, set before the results

1. **Plumbing:** the label's tests pass (a hand-built stretch with meals, poisons and drift; each transformer's own tail; T's mapping), and all lives run 60k ticks without errors.
2. **The label is learned:** frozen, the predicted T falls as food comes closer, and on ticks k before a real meal the predicted V is near 0.95^k (k up to 20).
3. **It helps with inertia:** MB's lived mean hunger at 40–60k is lower than IB's, paired by seed, and its instinct overrides less.
4. **No harm without inertia:** M0's lived mean hunger at 40–60k is not higher than C8r's by more than its interval.
5. **One scale** (descriptive): the candidate spread against the tolerance is alike in both worlds for M, unlike the hunger value (vm-00095).

**What follows:**
- **If 3 and 4 hold:** the meal label becomes the candidate default (the owner's decision). Next are need weighting (the second batch), then the movement cost.
- **If 3 fails and 4 holds:** the scale was not the inertia world's problem; next is action repeat (ARCHITECTURE.md section 9).
- **If 4 fails:** the hunger value carried something this label lacks. Next is hindsight (exact values from the stored future) or need weighting.

## Added at the owner's request (2026-11-17, before the runs)

- **"A check that the label is actually propagating back in time over time for more and more steps, and actually accurately predicts what happened next":**
  - **A test** (`tests/test_meal_label.py`): a table in place of the model, trained round after round on its own labels. After r rounds exactly the r ticks nearest a meal are right, and farther ones are not yet.
  - **A monitor in life** (`online.MealMonitor`, metrics `meal`): each tick's predicted V of the move taken waits for the body's next meal. Then its real value is known exactly, and predicted and real V add up by ticks to that meal (0, 1, 2–3, 4–7, 8–15, 16–31, 32–63). Over the life this shows how far back the label is right.
- **"There is no penalty for wrong prediction of label — if you predict 1 all the time you'll fill labels with 0.9 … I hope it will be self-healing":**
  - The loss pulls each prediction to its target, and with γ < 1 the targets heal an optimistic start: the excess shrinks by γ per tick it travels. A test starts every prediction at "a meal next tick" in an endless run with no meal, and the excess falls by exactly 0.95 per round.
  - **What could still make it optimistic:** the instinct makes the move whose prediction looks best, so a transformer's own prediction for the chosen move is biased low (vm-00085's winner's curse, removed in vm-00087 by the mean of fresh tails). With each transformer's own tail some of it may return. The monitor's predicted against real V shows it in life, and the frozen `value` test shows it at 60k.
- **Both batches queued** (`q97.sh` in the job's tmp): batch 2 (need) follows batch 1 directly while the owner sleeps. Its readings stand whatever batch 1 shows.

## Implementation

- [x] `online.value_target` = `hunger` (default, unchanged) or `meal`:
  - `meal_parts` gives the label's parts from the hunger before and after each step; `meal_target` maps it onto the model's scale.
  - The memory hands out the parts (`GatedMemory._parts`). With `value_refresh` each transformer's own fresh tail (`boot_values(mean=False)`) makes its own label, written into the rollout's hunger target just before its update (`set_batch` copies at once).
  - The checks need `value_refresh`, `value_n` 1 and no `model.state`.
- [x] `online.need = 1`: the instinct's tolerance divided by max(hunger, `NEED_FLOOR` .05), in life, in the frozen tests (`eval_hunger.Life`) and in the videos (`forage_video`).
- [x] `online.MealMonitor`, in life (metrics `meal`); `eval_hunger.py value` for meal runs: the realized meal label (backwards from the real hunger), and predicted against real V by ticks to the next real meal, 0–30 (row field `meal`, reading 2).
- [x] Tests: `tests/test_meal_label.py`, 12 pass:
  - the parts, the scale and the clip;
  - with exact tails the label is the exact recursion;
  - the memory's windows;
  - propagation, round by round, and healing from all-optimistic, with and without a meal;
  - the monitor;
  - the per-body tolerance;
  - short lives on the GPU with and without need, writing `meal` rows;
  - the refusal without the settings.
  The 150 tests of the touched paths pass (value labels, lifetimes, pieces, state, fast, lookahead, inertia, gate). 200-tick lives with the exact MB and MBN settings ran without errors.
- [ ] The runs (queued), the frozen tests, the write-up and videos.

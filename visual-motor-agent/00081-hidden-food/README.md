# vm-00081: Hidden Food

**Date:** 2026-11-03
**Status:** Lives done; frozen tests to rerun (queued)
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
- **Run notes:**
  - The 18 lives ran 00:49–01:59 on 2026-11-04.
  - The first frozen tests failed:
    - the aim test of the H arms ran out of GPU memory, because vm-00082's 30 lives started at the same moment (my queueing: vm-00082 waited for these lives, not for the frozen tests);
    - K4's aim tested only seed 1: the run list, joined by line breaks, was passed into `sh -c`, which ran each further line as a command.
  - The Makefile now joins the list with spaces. The frozen tests rerun after vm-00082's lives; the failed outputs are kept as `*.failed.*`.
- **The conditional arms were not run:** vm-00080's reading 4 is met only as worded (blanking D0's present copies costs more than blanking DX's past), and, by the lived results below, hidden food from birth would test learning, not memory.

## Results (interim: lived only)

### R1. Lived, ticks 40–60k, six seeds, 95% interval (t, df 5), paired by seed

| Arm | Food per 1k | Poison per 1k | Mean hunger | Food against K4 |
|---|---|---|---|---|
| K4 (vm-00073, always seen) | 70.0 | 3.68 | .058 | |
| H16 | 22.1 | 4.88 | .140 | −47.9 ± 12.7 |
| H4 | 20.1 | 3.83 | .126 | −49.9 ± 9.6 |
| H1 | 22.0 | 3.81 | .146 | −48.0 ± 11.9 |

Food by 10k ticks of life, mean of six seeds:

| Arm | 0–10k | 10–20k | 20–30k | 30–40k | 40–50k | 50–60k |
|---|---|---|---|---|---|---|
| K4 | 23.3 | 41.8 | 54.0 | 62.1 | 67.9 | 72.1 |
| H16 | 16.9 | 18.8 | 20.6 | 21.2 | 21.4 | 22.8 |
| H4 | 15.5 | 17.6 | 18.3 | 20.1 | 19.8 | 20.5 |
| H1 | 16.5 | 18.3 | 19.5 | 20.4 | 22.0 | 22.0 |

### R2. Meals eaten while the food is still drawn, moving at random (no model, 16 worlds, 5,000 ticks)

| `food_show` | Food per 1k | Eaten while still drawn |
|---|---|---|
| 16 | 4.6 | 14.8% |
| 4 | 4.6 | 6.2% |
| 1 | 4.6 | 2.4% |

<!-- figures -->
## Figures

![Frames of the world](images/world.png)

*The world of this experiment as its model sees it, every 12 ticks. Moves are random: this shows the world, not the trained agent.*

![Food touched per 1k ticks; Poison touched per 1k ticks](images/curves.png)

*From each run's `metrics.jsonl`, the mean of its seeds by arm, smoothed.*
<!-- /figures -->

## Analysis (interim)

- **With the food hidden from birth the agent does not learn to forage, whatever the show.** All three arms end near the blind walker (22.6) and gain about 5 food per 1k over the whole life, against K4's 49. H16 is no better than H1, though at 16 frames a walker that forgets at once loses nothing against perfect memory. So the failure is not a limit of memory.
- **It is a failure to start.** An agent moving at random eats a food it can still see in only 15% of its meals at 16 frames, 6% at 4. It almost never sees "food on screen, then hunger falls", so nothing ties the food's image to eating, and the image never becomes worth approaching.
- **Poison is still learned.** Poison stays drawn, and every arm keeps it low (3.8 to 4.9, against 27 to 28 for the blind walker).
- **What this means for the question:** memory can only be tested in an agent that already forages. That is vm-00083 (task 00094): the food always seen for 20k ticks, then hiding after fewer and fewer frames.

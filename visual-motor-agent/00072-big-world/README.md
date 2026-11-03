# vm-00072: The Big World

**Date:** 2026-11-02
**Status:** Running: seeds 1–3 finished, seeds 4–6 at 42k of 60k on 2026-11-02. Interim results below; the readings wait for all six and the frozen tests
**Source:** `dev/` (not yet tagged)
**Outputs:** `~/data/research/visual-motor-agent/exp_00072/`
**Task plan:** [`../tasks/00083-big-world.md`](../tasks/00083-big-world.md)

## Goal

vm-00071 at twice the side: a 128 x 128 world with food and poison of the same size in pixels and four times as many of each, with log summaries, K1 and K4. Does K4's wider reach hold? Design, arms and the readings set before the results are in the task plan.

## Interim, from the training logs

Not the readings: those are frozen tests, on all six seeds. Food and poison per 1,000 ticks.

| | K1 | K4 | K4 − K1 |
|---|---|---|---|
| Big world, 16–36k, six seeds | 20.0 | 30.2 | +10.2 ± 7.5 |
| Big world, last 20k, seeds 1–3 | 22.0 | 39.2 | +17.2 ± 18.3 |
| Small world (vm-00073), 16–36k, six seeds | 33.7 | 54.6 | +20.9 ± 8.7 |
| Small world (vm-00073), last 20k, six seeds | 35.8 | 70.0 | |

K4 is ahead in all six seeds at 16–36k, seed 6 only by .4. Over the last 20k the interval is too wide to read with three seeds. K4 takes no more poison: −1.5 ± 3.1 at 16–36k, −.65 ± 3.05 over the last 20k, seeds 1–3.

## Why both arms eat less here: the world asks for more walking

The density is the same, but both arms eat about 60% of what they eat in the small world. The owner's explanation: in the big world food can respawn far from the model, so it is harder to reach. `eval_food_gap.py` tests that with no model: scripted walkers that go straight, and turn to the nearest food once it is within a reach, 32 bodies, 5,500 ticks, ignoring poison.

| Walker | Food, 64 x 64 | Food, 128 x 128 | Big against small |
|---|---|---|---|
| Blind | 22.7 | 19.5 | .86 |
| Reach 16 pixels | 139.8 | 67.7 | .48 |
| Reach 24 pixels | 243.4 | 105.4 | .43 |
| Always chases | 272.0 | 170.6 | .63 |
| K1, last 20k (three seeds in the big world) | 35.8 | 22.0 | .61 |
| K4, last 20k (three seeds in the big world) | 70.0 | 39.2 | .56 |

| Right after a bite, chasing | 64 x 64 | 128 x 128 |
|---|---|---|
| Gap to the nearest food | 15.3 px | 24.6 px |
| Share of bites with the next food beyond 24 px | .15 | .50 |

- **The explanation holds.** Even a perfect chaser loses more than a third of its food in the big world, and walkers with the reach this model has lose more than half. The model's drop is no larger than that, so nothing about the model is needed to explain it.
- **Why the same density gives longer gaps:** a torus of 64 pixels cannot put anything more than about 45 pixels away. In the big world the next food is the nearest of a few scattered circles, and the one just eaten was the near one.
- **Not tested:** the second half of the explanation, that food concentrates in places. Foods are placed independently, so any clumps are chance; this measures the gap, not the clumps.
- **Beside blind:** K1 in the big world is barely above a blind walk (22.0 against 19.5), K4 is at twice it. In the small world they are at about one and a half and three times.

<!-- figures -->
## Figures

![Frames of the world](images/world.png)

*The world of this experiment as its model sees it, every 12 ticks. Moves are random: this shows the world, not the trained agent.*

<!-- /figures -->

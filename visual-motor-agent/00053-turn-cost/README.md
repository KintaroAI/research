# vm-00053: Turning Costs Hunger

**Date:** 2026-10-31
**Status:** Complete (pending tag)
**Source:** `exp/vm-00053` (code: `visual-motor-agent/dev/`)
**Outputs:** `~/data/research/visual-motor-agent/exp_00053/`
**Task plan:** [`../tasks/00061-turn-cost.md`](../tasks/00061-turn-cost.md)

## Goal

QG forages well on a chaotic path (the owner, from the video): successive moves have a cosine of 0.43 frozen, and no rule
for choosing changed that much (at best 0.50). The owner's idea: a little hunger for changing direction. The cost
lives in the world, the model learns it by prediction, and the instinct avoids needless turns.

## Method

- **`env.turn_cost = c`:** `hunger += c · |move − previous move|² / 8`: a steady move is free, a full reversal on both axes
  costs c. Tests: `tests/test_turn_cost.py`.
- **Runs:** QG (quadrants, gain, the old world) with `online.instinct_tol = 0.001` (at 0.005 the instinct would ignore
  a 90° turn at c = 0.02), seeds 1–3, 60k ticks: QT2 (c = 0.02), QT4 (c = 0.04). Control: vm-00051's QG, tolerance 0.005.

## Results

### R1. The last 20k ticks

| arm | food per 1k | seeds | poison per 1k | seeds | smoothness | seeds | proposal size | instinct changes the move |
|---|---|---|---|---|---|---|---|---|
| QG | 42.8 | 50.1, 40.8, 37.5 | 4.8 | 5.8, 4.8, 3.8 | 0.43 (frozen) | | 0.68 | 5% of ticks |
| QT2 | 31.7 | 23.8, 26.1, 45.1 | 6.5 | 7.0, 6.3, 6.2 | 0.71 | 0.61, 0.69, 0.84 | 0.43 | 40% |
| QT4 | 38.9 | 34.2, 52.0, 30.4 | 10.3 | 6.2, 14.9, 10.0 | **0.91** | 0.89, 0.93, 0.91 | 0.59 | 25% |

### R2. By 10k ticks, food / poison

| arm | 0–10k | 10–20k | 20–30k | 30–40k | 40–50k | 50–60k |
|---|---|---|---|---|---|---|
| QG | 25.2 / 19.7 | 35.9 / 12.0 | 38.3 / 7.0 | 42.4 / 5.4 | 41.2 / 5.6 | 44.4 / 3.9 |
| QT2 | 18.3 / 19.1 | 27.5 / 15.5 | 27.0 / 10.2 | 27.1 / 8.6 | 29.7 / 6.2 | 33.6 / 6.8 |
| QT4 | 24.5 / 22.1 | 33.9 / 18.8 | 37.9 / 14.6 | 41.2 / 14.4 | 40.6 / 10.5 | 37.2 / 10.2 |

## Analysis

- **The world did what no selection rule did:** smoothness 0.91 at c = 0.04. Learned through hunger alone, with no change
  to the model. Agreement between the pieces rises to 0.92: a shared momentum leaves them less to disagree about.
- **QT4 keeps most of the food** (38.9 against 42.8, one seed at 52.0). A straight body still finds food.
- **Poison is the price:** twice QG's at c = 0.04. A body that resists turning steers around poison worse, and QT4's
  poison falls more slowly throughout. Poison is the one thing that needs a sharp turn at short notice.
- **QT2 is worse than both on food:** smaller moves (0.43), no strong straightening, and low tolerance, so the
  instinct acts on 40% of ticks, mostly to save small costs. A small cost seems to nag more than it steers. This one
  is a hypothesis; the tolerance changed with the cost in these arms, so its effect is not separated.
- Mean hunger rises (0.09 to 0.12–0.14), so the cost is felt, but less than poison (0.5 per touch).

## Conclusions

1. A cost in the world is the way to change how the body moves: 0.91 against 0.43, from one number.
2. It trades against dodging: QT4 keeps 91% of the food at twice the poison.

## Next Steps

- A stronger poison signal or a cost on large changes only, so a sharp dodge stays cheap (a cost linear in the change).
- With no overlap and a rule for caution, whose paths are already smooth (council: 0.70–0.81 at no cost in hunger).

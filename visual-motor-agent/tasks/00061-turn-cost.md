# Task 00061: Turning Costs Hunger (experiment vm-00053)

**Date:** 2026-10-30
**Status:** In progress

## Context

- vm-00051's QG forages well, but its path is chaotic (the owner, from the video): successive moves have a cosine of
  ~0.43, a turn of ~60° per tick. Frozen rules barely change that: keeping the winner on ties 0.42, the most likely
  move 0.50, the most confident piece 0.50. The model decides anew every tick, with nothing that carries a plan.
- The owner's idea: add a little hunger on a change of direction. The cost lives in the world, not the model: the model
  learns it by predicting hunger, as everything else, and the instinct then avoids needless turns. No special handling.

## Design

- **`env.turn_cost = c`** (default 0): `hunger += c · |move − previous move|² / 8`, the moves in [-1, 1] per axis, so a full
  reversal on both axes costs c. On the change of the move, not its angle: defined at zero moves, and it acts as
  momentum (a steady move is free, any change costs, starting and stopping too). The previous move restarts at zero
  on a reset, and is in `get_state`.
- Learnable in one step: the previous move is in the window, the candidate in the move token.
- `online.instinct_tol` lowered from 0.005 to 0.001 in these arms: a 90° turn at full speed costs c / 4, and at
  c = 0.02 that would be at the old tolerance, where the instinct keeps its first sample.
- `lifetime.py` logs `smooth`, the cosine of successive moves made.

## Verification

- `tests/test_turn_cost.py`: the formula, a straight run free, a reversal costing c, the reset, the saved state, the
  world unchanged at 0, invalid values.

## Experiment vm-00053

- As vm-00051's QG (quadrants, gain, the old world), seeds 1–3, 60k ticks, sharing the 3090 with vm-00052.
- **QT2:** `turn_cost = 0.02`. **QT4:** `turn_cost = 0.04`. Control: QG (`instinct_tol = 0.005`, not logging `smooth`:
  0.43 frozen).
- **Measure:** food and poison per 1k ticks over 40k–60k, smoothness, mean hunger, and frozen aim.
- Hypotheses: paths get smoother; food holds if the body keeps aiming; the risk is sailing past food.

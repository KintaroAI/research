# vm-00085: A Value That Reaches Past Four Ticks

**Date:** 2026-11-04
**Status:** Running (started 2026-11-04 22:25, after vm-00084's frozen tests)
**Source:** `dev/` (not yet tagged)
**Outputs:** `~/data/research/visual-motor-agent/exp_00085/`
**Task plan:** [`../tasks/00096-value-bootstrap.md`](../tasks/00096-value-bootstrap.md)

## Goal

K4 scores a move by the mean of the real hunger over the next 4 ticks, labelled in hindsight. Nothing past those 4 ticks reaches the label, which fits what vm-00071, vm-00081–00083 and vm-00082 found: aim fades past about 16 px, hidden food is not pursued, and no chain of moves forms. Game-playing RL agents get their reach from a bootstrapped value: the value of a moment is trained toward what happens next plus the model's own estimate of the value after it, so credit spreads back one step per round of learning, with no fixed window. Here K4's label gets such a tail. The design and the readings set before the results are in the task plan.

## Method

- **The label** (`online.value_gamma = γ`, `online.value_n = n`, default `future_k`):

  V_j = (1 − γ) Σ_{i<n} γ^i h_{j+i} + γ^n V̂_{j+n}

  - h is the hunger after a step.
  - V̂_{j+n} is the model's own predicted label of the move it took at tick j + n, stored while acting: the prediction of the transformer whose move the body made.
  - The label is the exponentially weighted mean of all future hunger, so it stays in 0–1.
  - It sits in K4's slot: the move token predicts it and the instinct reads it. No new token, no CUDA change.

| Arm | γ | n | Reach, about 1 / (1 − γ) ticks |
|---|---|---|---|
| K4 (control, rerun) | off | — | 4 |
| V90 | 0.9 | 4 | 10 |
| V97 | 0.97 | 4 | 33 |

- **Everything else** is the default: four quadrant transformers, gain, log summaries, the ring, the egocentric touch world, `online.future_k = 4`, six seeds, 60k ticks. All 18 lives run at once on the RTX 3090, since the 4090 is out.
- **Frozen tests:**
  - `eval_hunger.py chain`: seen, blind, and blind with a perfect grab; aim by the gap to the food.
  - `eval_hunger.py value` (new): over 2,000 ticks on 16 bodies, the predicted label of each move taken against the realized one (bias, mean absolute error, correlation) and the spread of the instinct's candidates.
- **Checks passed before the start:**
  - `tests/test_value_labels.py`:
    - the labels equal the exact discounted mean to 1e-6;
    - an episode end cuts the tail;
    - the `future` mode is unchanged;
    - a gated memory hands training (the newest window and every stored chunk) the same labels as a direct sum on the stream.
  - The existing memory and gate tests pass.
  - 600-tick lives of K4, V90 and V97, beside vm-00084's 12 lives: the label and the prediction stay within 0–1 (both about .4 at the start), at the same speed as K4 (11.5 ticks/s).
  - `eval_hunger.py value` on vm-00073's K4 s1 (400 ticks, 4 bodies): bias −.010 against a realized mean of .025, correlation .84, median candidate spread .0016.

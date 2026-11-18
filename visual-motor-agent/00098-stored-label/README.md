# vm-00098: The Label as a Body Value, Stored Once

**Date:** 2026-11-18
**Status:** Running
**Source:** `dev/` (pending tag `exp/vm-00098`)
**Outputs:** `~/data/research/visual-motor-agent/exp_00098/`
**Task plan:** [`../tasks/00109-stored-label-body.md`](../tasks/00109-stored-label-body.md)

## Goal

In vm-00097 the meal label made the default agent forage far better. But memory kept only the raw facts, and every replay re-labeled old memories with the current model's prediction: in the owner's words, a cheat. Here the label is the owner's original idea: computed once, when the experience happens, from what the agent predicted then, and stored in memory as the truth. It is also a property of the body, like hunger, with its own learned parameters. Does the agent still learn it, and does the value still travel back in time? The design and the readings set before the results are in the task plan.

## Method

| Arm | World | Label | Seeds |
|---|---|---|---|
| C8r (vm-00094), IB (vm-00095), linked | none, inertia | the hunger value | 1–12 |
| M0, MB (vm-00097), linked | none, inertia | meal label in the hunger channel, re-labeled at each replay | 1–12 |
| S0 | no inertia | stored meal label, its own body channel | 1–12 |
| SB | inertia from birth | stored meal label, its own body channel | 1–12 |

- **The body's label** (`model.body_label = 1`): a third body value with its own channel (input embedding, output rows and bias). Its input is blank (0.5), since a move's label is known only after the next prediction; the move token predicts it, and the instinct reads it. Hunger is plain hunger again.
- **The stored label** (`online.label = meal`, `label_gamma` 0.95):
  - when acting, each transformer scores the move actually made;
  - one tick later step t's label for each transformer is written into memory, once: a meal 1, otherwise 0.95 × that transformer's prediction for the next step's move, less the hunger a poison added (no cost for the drift);
  - it is never re-labeled, and each transformer trains on its own stored labels.
- **Two monitors in life:** the predictions, and the labels stored in memory, against what really happened, by ticks to the next meal (metrics `meal`, `meal_stored`).
- **Cards:** 24 lives, seeds 1–8 on the RTX 4090 and 9–12 on the RTX 3090, under MPS.

## Results

Running.

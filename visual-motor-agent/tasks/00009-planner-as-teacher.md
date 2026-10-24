# Task 00009: Planner as Teacher: a Habit Learned From the Planner's Own Moves

**Date:** 2026-10-24
**Status:** Complete

## Context

From vm-00002 to vm-00007, the decision is made outside the network. The CEM planner (`plan.py`)
searches imagined futures for lower hunger; the network only predicts. The basic agent, which acts
from its own policy head, never learned to aim (vm-00006, vm-00007).

The topic README's first open question: let the planner teach. The policy head of the planner agent
(vm-00005 L) also learns to predict the planner's executed moves, with the move loss of the basic
agent. Then the head's own moves are tested without the planner. If this habit works, search is
needed only while learning.

The two open decisions:
- **Babbled moves (5%) enter the move loss:** yes. All executed moves do, as in the basic agent
  (whose 20% babbled moves are in its loss). This keeps the head's std from collapsing.
- **May the move loss train the shared features?** Both are tested: through the features, and
  detached (vm-00008's `ppo.pi_detach`).

## Hypotheses

1. **The habit aims.** L's features hold the direction to the circle (R² 0.84 from z at 100k), and
   the planner's moves point at it. So the head's own moves beat the random initial model and every
   basic agent so far (best 0.41–0.42 test success).
2. **Clicking is harder than moving for a linear head.** "Inside the circle" is not linear in the
   cursor and circle positions. The head with a hidden layer clicks more selectively and succeeds more
   often.
3. **Through the features, the move loss may erode them** as in the basic agent, although L's
   inverse-dynamics head may protect them. If it does, the planner gets worse than with the detached
   head.

## Deliverables

- Experiment vm-00009 (README + Makefile); runs in `~/data/research/visual-motor-agent/exp_00009/`.
- No new code: `ppo.move_coef`, `ppo.pi_detach` and `model.pi_hidden` exist, and `online.py --eval`
  scores any run's policy head.

## Method

The planner agent of vm-00007 (`configs/L_gated.json`, memory U: store everything), 30k steps,
4 threads, seeds 1–2, plus `ppo.move_coef=1`:

| Arm | Policy head | Move loss reaches the features |
|---|---|---|
| T | linear | yes |
| TD | linear | no (`ppo.pi_detach=1`) |
| TDM | hidden layer of 128 (`model.pi_hidden=128`) | no |

**Controls:** vm-00007's L_U runs (seeds 1–2: exp_00007/009–010; seeds 3–4: 025–026), the same agent
without the move loss. Their untrained head (checkpoint 0) gives the random initial model.

**Evaluation:**
- the head's own moves on the 128 test episodes at 0, 10k, 20k and 30k (`online.py --eval`): success,
  median steps, clicks per episode;
- the planner's success at 10k, 20k and 30k (`plan.py`) and its online hits: does the move loss hurt
  it?
- feature probes and the head's aim cosine at 30k (`analyze_self_actor.py probe`, `aim`).

## Verification

- The configs echo `move_coef=1`, `pi_detach` and `pi_hidden` as intended (`info.json`).
- The move NLL falls during training (the head learns the planner's moves).
- With `pi_detach=1`, the world model's losses track the L_U controls'.

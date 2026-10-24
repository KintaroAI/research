# Task 00006: The Basic Agent: Predict the Next Full State, Make That Move, Learn From What Happened

**Date:** 2026-10-23
**Status:** Complete

## Context

The owner wants the simplest possible agent, without planning:
- **Input:** a window of the last N real states. A state is the frame, the internal state (hunger,
  restlessness) and the move made.
- **One prediction:** the next full state, i.e. the next frame, the next internal state and the next
  move.
- **Act:** make the predicted move.
- **Learn on every step:** compare the predictions with what actually happened. One world, one
  agent, no reward, no planning.

The model already has this shape:
- the Transformer reads the window of real states;
- the policy head outputs the next move (a distribution over dx, dy and click);
- the prediction head outputs the next frame and drives, given that move.
What is missing is a loss that trains the move prediction on the move actually made, and an acting
loop that executes the predicted move.

## Hypothesis

1. **Pure version (A).** The agent learns the world's dynamics (next frame, drives), as the babblers
   did. Its moves stay babbling: training a model on its own sampled moves has zero expected
   gradient, so the move distribution only drifts. No preference for circles appears.
2. **Hunger-steered version (A+H).** The token gets one extra input: the hunger relieved within the
   next k = 12 steps. In training it is the true value from memory (hindsight); when acting it is
   the current hunger, i.e. "relieve it". The agent then learns which moves preceded hunger relief,
   and makes them when hungry. Online hits rise above babbling (about 50% of episodes). This
   uses no planning and no reward: one prediction per step, trained on what actually happened.

## Deliverables

1. **C++ `ppo.move_coef`:** the move-prediction loss −log π(a_t | window) for the move actually
   made. It is added to the policy-head gradient and works with `use_ppo=0`. Also
   `PPOStats.move_nll`, the PyTorch reference and a gradient test.
2. **`online.py` with `online.actor=self`.** Every step:
   1. Encode the last N steps (frames, drives, previous moves, plus the hunger input for A+H) and
      the current one as one sequence.
   2. Sample the move from the predicted distribution and make it.
   3. Store the transition.
   4. Take one gradient step on the newest window of N steps. For A+H, the window ends k steps
      back, once its hindsight label is known.
   - The extra input uses the existing scalar token slot (`use_prev_reward=1`; it carries the
     hindsight hunger relief, never reward).
   - Initial moves: a click probability of 0.1 (`click_bias`), as the babbler.
3. **Configs:** `A_basic.json` (window N = 8, next frame + drives + move; no inverse dynamics, no
   latent model, no replay) and the same with `online.goal=1`.
4. **Arms**, each an independent agent in its own world, run in parallel: A seeds 1–2 and A+H
   seeds 1–2, for 100k steps.
5. **Evaluation:** online hits per 1,000 steps and episode lengths; the agent's own moves on
   vm-00002's 128 test episodes at checkpoints, with the hunger command for A+H.

### Changes after the smoke runs

- Learning only from its own sampled moves collapses exploration within about 1,000 steps: the
  click probability falls 0.10 → 0.01 and hits stop.
- Added ε-babbling for the self actor (`online.explore_eps`), as in vm-00005.
- Arms (100k steps): A (ε 0.05), A+H (ε 0.05 and 0.2), A+H+R (ε 0.2, replay with batch 8 and a
  memory of 100k steps), 2 seeds each.

## Implementation details

- **Acting window.** The window is recomputed every step (no KV cache): positions 0..N−1 from the
  oldest to the newest step, the same as in training windows. A new circle inside the window
  starts a new segment, as in training.
- **Hindsight label** g_j = hunger_j if a hit happens at one of steps j..j+k−1, else 0. When acting,
  g = current hunger.
- **Learning rate:** 3e-4, Adam, one window of 8 steps per update. The loss weights are 1 for frame,
  drives and move.

## Verification

- Gradient test for `move_coef` against the reference.
- Smoke run of 3,000 steps per arm: losses fall, no NaNs, speed measured.

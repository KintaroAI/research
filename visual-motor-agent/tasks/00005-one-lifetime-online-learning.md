# Task 00005: One Lifetime: a Single World, Learning on Every Step

**Date:** 2026-10-23
**Status:** Complete

## Context

So far, learning and acting have been split:
- **Training data:** 128 parallel worlds act with fixed weights for 128 steps, then the model
  learns from those 16,384 transitions. Every transition is used 4 times.
- **Phases:** babbling came first (vm-00002, 2.1M steps). The planner then used frozen checkpoints
  (vm-00002) or kept learning from its own plans (vm-00004).

The owner's goal is closer to an animal's life:
- **one world:** a single agent in a single run, no parallel copies;
- **continuous learning:** it operates in the world and learns at the same time, with a weight
  update after every step;
- **one frame:** each prediction uses only the current frame. The model predicts the next frame
  (and its drives), compares with the real one, and learns from the error.
- **no reward:** hunger and restlessness are internal states, inputs and prediction targets.

## Hypothesis

1. A single agent that learns on every step from its own stream learns the same action
   consequences as the batched babbler:
   - cursor movement (inverse dynamics R², cursor readout);
   - the click contingency and hunger relief (edge test).
2. Each step it acts with the planner when its imagination finds a way to lower hunger, and
   babbles otherwise. Its behavior then turns from babbling into goal-directed clicking within one
   lifetime, with no phase switch and no reward. Test success of the planner rises from the
   untrained level (0.16) toward vm-00002/4 (0.89–1.00).
3. Replaying its own recent past (hippocampus-like) is needed. Learning only from the latest steps
   forgets rare events: hits are about 1 in 10–100 steps. Arm P tests this.

## Deliverables

1. **`model.one_frame`** (C++, checkpoint header slot 36):
   - Each token attends only to itself, so every prediction uses only the current frame (plus the
     drives and the previous action, which are part of the token).
   - Training sequences stay `ctx` steps long, so inverse dynamics and the latent chains still get
     the real next frames as targets.
   - Acting clears the KV cache every step. Training and acting then compute identical outputs.
2. **`vm_rollout_store(rollout, t, images, drives, prev_action, prev_reward, ep_start, action)`**
   fills a training batch from stored experience. `vm_rollout_outcome` already exists.
   `agent_lib.Rollout.store`.
3. **`online.py`: the lifetime loop**, with a single `CircleWorld` (auto-reset, drives on, no
   bonuses). Every step:
   1. **Act.** The CEM planner (vm-00002 settings) plans with the current weights. Let *gain* be
      the imagined per-step hunger reduction of its plan vs doing nothing. If gain < θ (0.02), or
      with probability ε (0.05), the agent babbles instead: dx, dy ~ N(0, 0.607), click with
      probability 0.1, as the vm-00002 babbler. Early in life nothing lowers imagined hunger, so it
      babbles. Later the planner takes over near circles.
   2. **Step the world** and store the transition in a replay memory (default 100k steps, about
      2.5 GB).
   3. **Learn.** One gradient step on B sequences of `ctx` consecutive steps: always the newest
      sequence (ending at the current step), plus B−1 drawn uniformly from the memory. The losses
      are as `P_dyn`: next frame, next drives, inverse dynamics, latent chains, all reward-free.
      `use_ppo=0`, epochs 1, minibatches 1.
   4. Every `save_every` steps: a checkpoint. Online metrics go to `metrics.jsonl` every 1,000
      steps: hits, episode lengths, share of planner steps, losses, steps/s.
4. **Arms** (seed 1, about 200k steps each, limited by speed):
   - **L:** one frame, replay, B = 8 sequences of 8 steps (64 transitions per update).
   - **P (pure online):** one frame, B = 1, only the newest 8 steps. No memory beyond the last
     8 steps.
5. **Evaluation** (offline, on frozen checkpoints, measurement only):
   - `plan.py` on the 128 test episodes;
   - `analyze_imagination.py` (edge and trap tests, cursor retrieval);
   - online success over the lifetime.
   The reference is the split pipeline: vm-00002's batched babbler at the same number of
   *environment steps*, and vm-00004.

### Changes after the smoke runs

- **The exploration gate rarely fires.**
  - In the first 3,000 steps, the planner's plan beat doing nothing by an imagined 0.2–0.4 hunger per
    step: the CEM optimizes against even an untrained hunger head. The agent planned on 92–96% of
    steps from step 0.
  - So the planner, chasing its own model's errors, *is* the exploration, and learning corrects
    those errors (the vm-00004 loop). The 5% random babbling (ε) remains.
  - The gate stays in the code (`explore_gain`) and is reported. The control arm B always babbles.
- **Speed.** One update on 8 sequences of 8 steps takes about 56 ms (no gain beyond 8 threads). The
  planner for one world takes about 41 ms. Hence about 10 steps/s with planning and 17 without.
- **Arms** (each its own model in its own world, run in parallel; 200k steps):
  - **L**, seeds 1–4: plans, plus learning from the newest 8 steps and 7 replayed sequences (memory
    of 100k steps);
  - **P**, seeds 1–3: pure online, learning from the newest 8 steps only (memory of 8 steps);
  - **B**, seeds 1–2: the same online learner (with memory), but it always babbles.

## Implementation details

- **Speed.** One update on 8 sequences of 8 tokens runs one sequence per thread, about 40 ms. The
  planner for one world takes about 20 ms. Expect about 10–20 steps/s, i.e. 40–70k steps per
  hour. Measure before launching.
- **Learning rate:** 3e-4. Updates use 64 transitions instead of 512.
- **Replay sampling.** Sequences never cross the write head of the ring buffer. Episode starts
  inside a sequence are marked (`ep_start`), as in rollouts.
- **Reproducibility.** Seeded world, planner and replay RNG. The planner is not bitwise
  reproducible across thread counts, so the thread count is fixed per run.

## Verification

- C++ and Python tests, plus new ones:
  - with `one_frame`, acting equals training bitwise;
  - gradients match the reference with `one_frame` (each token is its own segment);
  - filling a rollout with `vm_rollout_store` reproduces the rollout that acting wrote.
- A smoke run of 2,000 steps: losses fall, no NaNs, steps/s measured.

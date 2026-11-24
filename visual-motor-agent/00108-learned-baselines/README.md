# vm-00108: Learned Baselines — Is the Default Agent Fast or Slow for Its Ticks?

**Date:** 2026-11-23
**Status:** Running (since 2026-11-23 15:01: the Dreamer pilots, then PP and PL, then DR breakout and food)
**Source:** `exp/vm-00108` (code: `visual-motor-agent/dev/`)
**Outputs:** `~/data/research/visual-motor-agent/exp_00108/`
**Task plan:** [`../tasks/00120-learned-baselines.md`](../tasks/00120-learned-baselines.md)

## Goal

Calibrate. Since vm-00001 our results are scored against scripted walkers and our own earlier arms, never against a standard learner given the same experience. The owner, 2026-11-23: "yes write the task and queue it after 107".

- **The question:** does the default agent's architecture learn faster per tick than standard RL given the same signal? That architecture is one next-token predictor for pixels, body and value; one life; one update a tick; and the instinct choosing among sampled moves by the predicted meal label.
- **Not reward-free:** the default agent learns the meal label, a bootstrapped value of meals and poison. The baselines get that label's immediate part as their reward.

## Method

- **Worlds** (unchanged, `world.make_world`, one body, one life of no episodes):
  - **breakout:** stage 1 (`configs/U_breakout.json`, a paddle hit a meal, a miss a poison). Ours: vm-00103's BD.
  - **food:** the default agent's world (`configs/U_agent.json`: 128 world, 64 view, egocentric, inertia 4). Ours: vm-00101's TT.
- **The same input** (`dev/baselines/vm_env.py`): the 64 × 64 grey frame, the drives and the previous move; no `gt_*` keys.
- **The same signal:** reward +1 on a tick whose hunger fell, minus the full dose a poison added, else 0. That's the meal label's immediate part, tested equal to `online.stored_labels`.
- **The action:** the move (dx, dy) in [−1, 1]², continuous.
- **The arms**, seeds 1–6 (the same world seeds as ours):
  - **DR:** DreamerV3 (the PyTorch port `NM512/dreamerv3-torch`, commit 6ef8646; JAX can't run on this CPU). Its defaults for 64 × 64 pixels, train ratio 512, one body, 60k ticks (a 2,500-tick random prefill included).
  - **PP:** PPO (`dev/baselines/run_ppo.py`, CleanRL-style, the Nature CNN), one body, 60k ticks.
  - **PL:** the same PPO, 16 bodies, 2M samples: how many ticks it needs to reach our level.
- **Cards:** seeds 1–3 on the RTX 3090, 4–6 on the RTX 4090, under MPS. Dreamer is timed alone on each card first (`make pilot`).

## Measures

- **breakout:** paddle hit share by 10k lived, against BD (paired by seed, seeds 1–6) and the random paddle (.235).
- **food:** food and poison per 1k by 10k lived, against TT (paired by seed, seeds 1–6).
- **PL:** samples to reach our 50–60k level, by seed.
- **Compute:** wall-clock per life.

## The readings, set before the results

1. **Plumbing:** the wrapper's tests pass; every life runs its ticks.
2. **Against PPO at our budget:** ours at 40–60k against PP's, paired by seed (expected: ours far ahead).
3. **Against Dreamer at our budget:** ours at 40–60k against DR's, paired by seed. Either direction is the answer.
4. **PPO's sample cost:** PL's samples to our level (descriptive; "not reached in 2M" is a result).

## Results

(pending)

- **Dreamer's speed alone** (`make pilot`, 5,000 ticks of breakout seed 1, 2,500 of them learning at train ratio 512): 596 s on the RTX 3090 and 563 s on the 4090, about 4.4 learning ticks a second; a 60k life alone about 3.6 hours.

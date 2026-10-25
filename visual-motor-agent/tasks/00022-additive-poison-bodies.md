# Task 00022: Additive Poison, and One Model in Many Worlds (experiment vm-00018)

**Date:** 2026-10-25
**Status:** In Progress

## Context

- vm-00017: the 8×8 patch model predicts food clicks (most of the drop; the eaten circle vanishes in its predicted
  frame), poison only partly (+0.07 to +0.21 of +0.70) and still rising at 60k, and partly confuses a click on nothing
  with food. Poison that sets hunger to 1 left hunger at the cap on 23–57% of steps, where a poison click changes
  nothing.
- vm-00016: 32 bodies early in life (`lifetime.py`) made the best basic agent, but with both more data (500k
  body-steps against 60k) and bigger batches (39 windows against 8). Which one matters is open.
- Poison clicks are rare for one body (a hazard covers about 6% of the screen; the agent clicks on about a tenth of
  its steps): many bodies see many more of them early on.

## Changes

- **`env.hazard_click_add`** (default 0: off): a click inside a hazard adds this to hunger (after any relief, clipped
  to [0, 1]). With 0.5 a poison click rarely reaches the cap. `env.hazard_click` (set) is unchanged.
- **Poison clicks counted** (scoring only): `poison` (body 0) in `online.py` and `lifetime.py` metrics, `all_poison_per_1k`
  over the active bodies in `lifetime.py`.
- `analyze_hazards.py click` accepts either kind of poison (the world's outcome is measured, so the truth is +0.5 here).

## Experiment vm-00018

World: vm-00017's (continual torus, 2 hazards, no contact cost, hunger 0.001 per step) with `env.hazard_click_add=0.5`
instead of `hazard_click`. Model: vm-00017's PP (8×8 patch tokens, move loss detached, hidden policy layer, memory U).
`lifetime.py`, 100k ticks, seeds 1–3, on both GPUs.

| Arm | Bodies | Batch (windows per update) |
|---|---|---|
| Q1 | 1 | 8 (1 newest + 7 replayed) |
| Q32 | 32 for 10k ticks, halving every 5k (one from tick 30k) | 39 → 8 |
| Q1B | 1 | 39 (1 newest + 38 replayed) |

Q1 against Q1B: batch size alone. Q1B against Q32: more fresh experience at the same early batch size (Q1B keeps 39
throughout).

## Measures

- `analyze_hazards.py click` at 0, 10k, 20k, 40k, 60k, 100k: the drives head's next hunger for food / poison /
  background × click / stay (world: −0.301, +0.5, 0 click minus stay), the eaten circle's predicted redness.
- Online: food hits and poison clicks per 1,000 steps (body 0 and all bodies), hunger at the cap.
- The episodic and forage tests at the same checkpoints.

## Verification

- `tests/test_world_ext.py`: an additive poison click adds and clips; the set form unchanged; off by default.

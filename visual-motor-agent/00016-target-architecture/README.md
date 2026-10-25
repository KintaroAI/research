# vm-00016: The Target Architecture Against Its Parts

**Date:** 2026-10-25
**Status:** In Progress
**Source:** *tagged on completion as `exp/vm-00016`* (code: `visual-motor-agent/dev/`)
**Outputs:** `~/data/research/visual-motor-agent/exp_00016/`
**Task plan:** [`../tasks/00020-target-architecture.md`](../tasks/00020-target-architecture.md)

## Goal

Build the three new parts of `../ARCHITECTURE.md` and measure what each adds to the 8×8 patch-token agent of
vm-00014/15:

- **the act token:** a learned query per step that the policy reads, trained by its own predictions of the next drives
  and the coarse (8×8) next frame. In vm-00010/14/15 the policy read the move token, which nothing trains when the move
  loss is detached;
- **the signed hindsight command:** the change of hunger over the next 12 steps, so that moves into hazards are marked;
- **parallel bodies early in life:** 32 bodies driven by the same weights for 10k ticks, halving every 5k down to one.

## Hypothesis

1. The act token's output holds the scene (cursor, circle, direction) better than the move token did (vm-00010:
   R² of the direction 0.08–0.13), and the agent aims.
2. With the signed command, the share of steps in a hazard falls below chance (about 0.063).
3. Parallel bodies speed up early learning (the episodic test at 10k–20k) without hurting the single lifetime after.

## Method

- **Implementation** (task 00020):
  - `model.act_token=1` in the C++ model, the PyTorch reference and the CUDA model; `ppo.act_coef`. Gradients
    verified against the reference (24 tests) and the GPU against C++ (2 models × 11 loss settings); deliberately broken
    builds fail them. Defaults byte-identical (300-step patch and CNN lifetimes).
  - `online.goal_mode=delta` (`delta_labels`), and `lifetime.py`: N bodies with their own short-term buffers and gates
    writing into one shared long-term memory (`ChunkStore`), one batched forward pass per tick, one update on the newest
    window of every body plus 7 replayed. With one body it reproduces `online.py` bit for bit (patch and CNN).
  - `tests/test_lifetime.py`; 804 Python tests and the C++ tests pass.
- **World:** vm-00015's slow hunger (continual torus, 2 hazards at 0.02 per step, hunger 0.001 per step).
- **Arms** (8×8 patches, policy head detached with a hidden layer, memory U, ε 0.2 constant, 60k ticks, seeds 1–3):
  A (+ act token, `act_coef` 0.5), AS (+ signed command), ASB (+ 32 bodies shrinking to 1 by 35k ticks).
  Control: vm-00015 S8D.
- **Measures:** body 0's online hits and hazard share, all bodies' hits; the episodic and forage tests at 0, 10k, 20k,
  40k, 60k; `analyze_hazards.py contact`; `analyze_patch.py tokens` (with the act token) and `analyze_self_actor.py aim`.
- **Speed:** 24 ticks/s alone with 32 bodies (760 body-steps per second).

## Log

### 2026-10-25
- **Implemented** the act token, the signed command and parallel bodies (task 00020).
- **Launched** the 9 lifetimes at 10:55: 18 ticks/s for A and AS, 10 for ASB while it has 32 bodies (sharing the GPU).


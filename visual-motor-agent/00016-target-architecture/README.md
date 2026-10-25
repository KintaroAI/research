# vm-00016: The Target Architecture Against Its Parts

**Date:** 2026-10-25
**Status:** Complete (pending tag)
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
  A (+ act token, `act_coef` 0.5), AS (+ signed command), ASB (+ 32 bodies for 10k ticks, halving every 5k: one body from tick 30k; 500k body-steps in all, against 60k).
  Control: vm-00015 S8D.
- **Measures:** body 0's online hits and hazard share, all bodies' hits; the episodic and forage tests at 0, 10k, 20k,
  40k, 60k; `analyze_hazards.py contact`; `analyze_patch.py tokens` (with the act token) and `analyze_self_actor.py aim`.
- **Speed:** 24 ticks/s alone with 32 bodies (760 body-steps per second).

## Log

### 2026-10-25
- **Implemented** the act token, the signed command and parallel bodies (task 00020).
- **Launched** the 9 lifetimes at 10:55: 18 ticks/s for A and AS, 10 for ASB while it has 32 bodies (sharing the GPU).
- **All 9 finished at about 11:40** (46 minutes). The aim analysis was rerun with the fixed `analyze_self_actor.py`
  (vm-00014's log). The control's token probes and aim were added to vm-00015.

## Results

### R1. Learning: online, the episodic test and the forage test (seeds 1, 2, 3)

| Arm | episodic test 10k | 20k | 60k | forage at 60k (hits per 1k) | body 0 online, last 10k (hits per 1k) |
|---|---|---|---|---|---|
| S8D (control, vm-00015) | — | 0.77, 0.84, 0.88 | 0.77, 0.80, 0.89 | 6.8, 9.6, 8.8 | 6.0, 8.0, 8.3 |
| A (+ act token) | 0.88, 0.66, 0.76 | 0.96, 0.78, 0.81 | 0.89, 0.75, 0.80 | 13.5, 8.7, 8.6 | 12.1, 7.1, 8.0 |
| AS (+ signed command) | 0.89, 0.85, 0.84 | 0.90, 0.76, 0.78 | 0.97, 0.74, 0.80 | 11.5, 7.0, 9.3 | 9.0, 6.0, 10.4 |
| ASB (+ 32 bodies) | **0.91, 0.98, 0.85** | **0.93, 0.94, 0.94** | **0.91, 0.98, 0.96** | **10.8, 17.6, 16.9** | **9.9, 12.3, 11.8** |

At birth 0.59–0.64 (episodic), 4.3–5.2 (forage). In the first 10k ticks the 32 bodies together hit 5.3, 9.3, 10.5
times per 1,000 body-steps.

### R2. Hazards

| Arm | in a hazard, last 10k (chance about 0.063) | drives head inside − outside at 60k (world +0.020) | R² of contact from z (the policy's token) | from the drives token |
|---|---|---|---|---|
| S8D | 0.074, 0.061, 0.110 | +0.003, +0.001, +0.002 | 0.10, 0.36, 0.13 (move token) | 0.12, 0.29, 0.26 |
| A | 0.081, 0.064, 0.086 | +0.002, +0.007, +0.002 | 0.37, 0.49, 0.44 | 0.38, 0.25, 0.52 |
| AS | 0.071, 0.062, 0.093 | +0.005, +0.007, +0.002 | 0.46, 0.34, 0.46 | 0.56, 0.38, 0.45 |
| ASB | 0.093, 0.060, 0.074 | −0.001, +0.002, +0.001 | 0.55, 0.69, 0.63 | 0.45, 0.67, 0.63 |

The token under the cursor: 0.69–0.76 in every arm.

### R3. What the policy reads, and aim (60k)

| Arm | the policy's token: R² of cursor / circle / direction | the move token | aim cos(μ, to the circle) |
|---|---|---|---|
| S8D | (the move token) | 0.06–0.11 / 0.70–0.76 / 0.13–0.16 | +0.070, +0.181, +0.120 |
| A | act: 0.03–0.19 / 0.09–0.53 / 0.04–0.26 | 0.04–0.20 / 0.60–0.65 / 0.09–0.25 | +0.173, +0.066, +0.152 |
| AS | act: 0.02–0.12 / 0.21–0.41 / 0.07–0.16 | 0.04–0.12 / 0.45–0.51 / 0.09–0.18 | +0.145, +0.073, +0.146 |
| ASB | act: 0.06–0.15 / 0.21–0.38 / 0.09–0.21 | 0.12–0.15 / 0.41–0.59 / 0.14–0.23 | +0.093, +0.224, +0.131 |

All patch tokens side by side: cursor 0.90–0.97, circle 0.97–0.98, direction 0.76–0.88 (ASB highest: 0.97 / 0.98 /
0.86–0.88).

## Analysis

### Parallel bodies make the best basic agent so far (hypothesis 3, supported)

ASB passes 0.85–0.98 of the test episodes at 10k ticks (A and AS 0.66–0.89) and keeps it after it is down to one
body (0.91–0.98 at 60k), while its forage rate at 60k, 10.8–17.6 hits per 1,000 steps (mean 15.1), is 1.8 times the
control's (mean 8.4). Its single surviving body hits 9.9–12.3 times per 1,000 steps online in the last 10k, against
6.0–12.1 in the other arms. Nothing is lost when the bodies retire: it is one model, and what it learned in childhood stays.
The cost is experience, not updates: ASB made the same 60k updates on bigger batches (39 windows instead of 8 while it
had 32 bodies) from 500k body-steps instead of 60k. Which of the two matters (more fresh data, or bigger batches) this
experiment does not separate.

### The act token holds contact, not the scene (hypothesis 1, not supported)

The act token's own predictions (the next drives, the coarse next frame) make it hold hazard contact far better than
the move token did (R² 0.34–0.69 against 0.10–0.36), but not the cursor or the direction to the circle (0.04–0.26),
no better than the move token beside it. Aim is about the control's (+0.07 to +0.22). The coarse 8×8 frame does not
need the cursor, which spans about one coarse pixel, and the next drives mostly follow the current drives; neither loss
asks the token to know where the cursor is relative to the circle.

### The signed command does not produce avoidance (hypothesis 2, not supported)

AS spends 0.062–0.093 of its steps in hazards, as A (0.064–0.086) and chance. Its command marks moves into hazards
(labels of −0.02 per step inside), but a hazard visit's label (about −0.1 to −0.2) is small beside relief's (+0.3 to
+0.5), and the policy reads a token that holds the cursor poorly, so it cannot tell which move leads into which hazard.
The drives head, which reads the same patch tokens, still does not predict the hazard's cost (+0.001 to +0.007 of
+0.020) in any arm.

### Caveats

- Three seeds; seed spread is large in the single-body arms (episodic 0.66–0.96 at 10k–20k).
- ε stays at 0.2: a fifth of the moves are random, which bounds both aim and avoidance.
- The control S8D has no 10k checkpoint test.

## Conclusions

1. **Many bodies early in life, one brain, is the strongest change so far for the basic agent**: 32 bodies for the
   first 10k ticks, shrinking to one, give 0.85–0.98 of the test episodes at 10k and 0.91–0.98 at 60k, and 1.8 times
   the control's forage rate, with the same number of updates.
2. **The act token holds hazard contact but not the scene**: its prediction losses do not require the cursor's
   position relative to the circle, so the policy's input still lacks the direction.
3. **No arm avoids hazards**, with or without the signed command, and no drives head predicts their cost.

## Next Steps

- Separate data from batch size: one body with 39-window batches, and 32 bodies with 8-window batches.
- Give the act token a target that needs the direction, e.g. predicting the next move's effect on the cursor (its
  next position in the coarse frame at full resolution around the cursor), or the hindsight command itself.
- Decay ε, now that the agent aims (`online.explore_eps_end`).
- Hazard cost large enough to matter in squared error (or a separate hazard-change head), keeping the cap rare.


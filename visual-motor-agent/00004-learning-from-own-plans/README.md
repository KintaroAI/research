# vm-00004: Learning From Its Own Plans

**Date:** 2026-10-23
**Status:** Complete (pending tag)
**Source:** *tagged on completion as `exp/vm-00004`* (code: `visual-motor-agent/dev/`)
**Outputs:** `~/data/research/visual-motor-agent/exp_00004/`
**Task plan:** [`../tasks/00004-learning-from-own-plans.md`](../tasks/00004-learning-from-own-plans.md)

## Goal

In vm-00002, a planner using the world model of a reward-free babbler hit 89% of the test episodes.
Its failures all had one cause. Its hunger head imagines relief for clicks just outside the circle's
edge, so the planner stops 0–5 px short and clicks there up to 192 times. Each such click is an
observed prediction error: the hunger does not drop. Yet the model never trains on what the planner
does.

This experiment lets the agent keep learning from its own goal-directed experience. The losses are
the same reward-free prediction losses as babbling; there is no reward, value function or policy
training.

## Hypothesis

1. Training the world model on the planner's own experience removes the false relief near the edge.
   The planner then approaches 100% success.
2. The gain comes from the planner's data, not from more training. Babbling for the same number of
   steps does not improve the planner: 005 went from 0.85 at 1.05M to 0.89 at 2.1M.

## Method

- **Start:** `exp_00002/005_P_dyn_s1/ckpt_02097152.bin` (click 0.1 babbler, 2.1M steps). Loaded with
  `--init` (fresh optimizer) and the `P_dyn` config: constant lr 1e-3, ctx 8, 32 minibatches.
- **S (self-experience):** `train.actor=planner`.
  - At every step, the CEM planner of vm-00002, using the current weights, picks the actions of all
    128 environments. Default settings: 256 samples, horizon 12, 4 iterations.
  - Environments restart after a hit or after 200 steps, as in training.
  - The transitions train the model exactly as babbling does: next frame, next drives, inverse
    dynamics and latent chains (`ppo.use_ppo=0`).
  - The stored action is the planner's (`vm_rollout_set_action`).
- **B (control):** the random babbling policy continues for the same number of steps.
- **R:** babbling continued with `ppo.dyn_h_horizon=1`, so the hunger head trains on real latents
  only (added during the experiment, see the log).
- **SR:** S with `ppo.dyn_h_horizon=1`.
- **S006 / B006:** S and B from the second vm-00002 babbler (006, click probability 0.5, 2.1M steps,
  config `P_dyn_p50`).
- **Length:** 262,144 steps each (16 updates), with milestones every 32,768 steps.
- **Evaluation at every milestone:**
  - `plan.py` on the 128 test episodes;
  - `analyze_imagination.py`: the edge test, approach test and cursor retrieval;
  - `analyze_planner.py`: failure traces.

## Log

### 2026-10-23
- Implemented `vm_rollout_set_action` (C API, `Rollout.set_action`) and `train.actor=planner`.
  A new test checks two things:
  - replacing the stored actions with the ones `act` returned leaves the losses and statistics
    bitwise identical;
  - replacing them with other actions changes the inverse-dynamics loss.
  All 155 Python tests and the C++ tests pass.
- Smoke run: 2 updates with 32 threads under load, 80 env-steps/s. The planner dominates the time:
  about 1.3 s per step for 128 environments. Launched S (64 threads, about 140 env-steps/s) and B
  (32 threads, about 600 env-steps/s).
- First checkpoint of S (32k steps): success .891 (from .875), clicks per episode 13.6 (from 33).
  But the hunger head's edge got *blurrier*: imagined relief at +1 / +2 / +3 px outside went from
  .36 / .20 / .09 to .59 / .44 / .30. B at 32k sharpened instead: .25 / .13 / .05.
- **Is the edge information in the latent?** A ridge probe on the encoder's features classifies
  inside vs outside with the cursor at a signed distance g from the edge (4,096 frames, half held
  out):

  | g (px) | −3 | −2 | −1 | −0.5 | +0.5 | +1 | +2 | +3 |
  |---|---:|---:|---:|---:|---:|---:|---:|---:|
  | probe accuracy (2.1M babbler) | .99 | .93 | .71 | .61 | .70 | .81 | .94 | .99 |
  | hunger head relief (2.1M babbler) | .88 | .77 | .64 | .58 | .43 | .33 | .21 | .09 |
  | hunger head relief (S at 32k) | .91 | .84 | .77 | .72 | .62 | .55 | .46 | .26 |

  The latent carries the edge to about ±2 px; the hunger head smears it over about ±3–4 px.
  Hypothesis: the hunger head is trained mostly on *imagined* latents. With chains of up to 7
  steps over 8-token sequences, 27 of 35 chain steps (77%) are imagined, and imagined latents have
  1.5–4 px of cursor error. The MSE-optimal prediction from them is a smeared relief. The
  planner's near-edge data (mostly successful clicks just inside) pushes the smear outward.
- Added `ppo.dyn_h_horizon`: only chain steps k ≤ dyn_h_horizon train the hunger head. The PyTorch
  reference and gradient test are updated; 162 tests pass. Arm R: babbling continued with
  `dyn_h_horizon=1`, i.e. the hunger head is trained on real latents only.

- S finished (262k steps, about 35 minutes) and B finished. A diagnostic of where the planner
  clicks, and the multi-step imagination after a click just outside the edge, explain the result
  (R2).
- `analyze_imagination.py` gained two measurements:
  - `edge_cost`: the summed 12-step imagined hunger after one click at a signed gap g;
  - the trap test: whether "one move in, then click" beats "click here" at +2 and +5 px.
  The one-step edge relief misses what matters to the planner.
- Started SR (S with `dyn_h_horizon=1`) and S006 (S from the second babbler, click 0.5) as
  replications, then B006 (the control for S006).
- All six arms finished and were evaluated at every milestone. In SR's last checkpoint the
  planner fails in a new way: it moves away from the circle (R3).

## Results

All planner numbers are on vm-00002's 128 test episodes (radius 12–16, 200-step limit). "Median
steps" counts a miss as 200 steps.

### R1. The planner while its model learns from its own plans (S) vs more babbling (B)

| Steps after the babbler | S: success | median steps | clicks / episode | B: success | median steps | clicks / episode |
|---|---:|---:|---:|---:|---:|---:|
| 0 (005 at 2.1M) | 0.875 | 11 | 33.1 | 0.875 | 11 | 33.1 |
| 32k | 0.891 | 13 | 13.6 | 0.906 | 21 | 27.5 |
| 65k | 0.797 | 38.5 | 62.6 | 0.461 | 200 | 65.6 |
| 98k | 0.969 | 9.5 | 16.5 | 0.945 | 9 | 23.2 |
| 131k | 0.930 | 9 | 17.2 | 0.891 | 41 | 30.9 |
| 164k | 0.969 | 6.5 | 13.5 | 0.867 | 57.5 | 31.9 |
| 197k | **1.000** | **5.5** | 6.2 | 0.938 | 29.5 | 17.6 |
| 229k | **1.000** | **5** | 4.8 | 0.922 | 9 | 18.5 |
| 262k | **1.000** | **6** | 4.5 | 0.781 | 19.5 | 20.6 |

- Over the last three checkpoints, S hits 384 of 384 test episodes and B hits 338 of 384.
- Median steps over hits: S 5.5 / 5 / 6, B 23 / 8.5 / 10.
- PPO with reward (vm-00001) reaches 1.00 and 5–7 steps.
- `analyze_planner.py` on S at 262k: no failures at any starting distance, including the two
  starts more than 48 px from the edge (0/2 at the start).
- Planner success with a babbled model swings between neighboring checkpoints: B dropped to 0.46
  at 65k. Single-checkpoint numbers in vm-00002 carry the same uncertainty.

### R2. What changed in S (262k vs its starting model)

**Where the planner clicks** (all clicks on the 128 test episodes, signed gap to the edge where the
click lands):

| | Clicks | Inside the circle | 0–4 px outside | 4–16 px outside | > 16 px outside |
|---|---:|---:|---:|---:|---:|
| start | 4,231 | 112 (3%) | 525 | 2,686 | 908 |
| S at 262k | 580 | 128 (22%) | 111 | 234 | 107 |

**Outside the planner's own trajectories, the world model got worse.** It became a model of the
planner's own experience:

| | start | S at 262k | B at 262k |
|---|---:|---:|---:|
| straight paths: imagined / true displacement after 12 moves (full / half speed) | .84 / .65 | .70 / .51 | .76 / .54 |
| hunger error at imagined step 12 (random actions) | .06 | .15 | .05 |
| one imagined move over the circle, cursor error (px) | 2.56 | 4.54 | 2.47 |
| one-step imagined relief at +2 px outside (truth 0) | .20 | .52 | .11 |

**But it learned what follows a click just outside.** Imagined hunger over 12 steps after a single
click (then no moves), starting at hunger 0.4. With no relief, the true values are 0.405 → 0.46,
summing to about 5.2.

| Cursor | start model | S at 262k |
|---|---|---|
| 6 px inside | 0.00 → 0.14 (sum 1.01) | 0.03 → 0.05 (sum 0.47) |
| 2 px outside | **0.32 → 0.40 (sum 4.30)** | **0.18 → 0.59 (sum 5.02)** |
| 4 px outside | 0.39 → 0.45 (sum 5.08) | 0.30 → 0.68 (sum 6.27) |
| 8 px outside | 0.40 → 0.47 (sum 5.28) | 0.40 → 0.56 (sum 5.65) |

The start model imagines a lasting partial relief after a click just outside: it stays at 0.32–0.40
for 12 steps. S's model imagines a brief dip followed by a rebound, so a click at the edge now costs
more than no click at all. Trap test: "one full move toward the center, then click" is preferred
over "click here" at +5 px in 64% of states for the start model. For S it is 80–90% (164–262k); for
B, 54–77%.

### R2b. Replication from the second babbler (006, click 0.5)

| Steps after the babbler | S006: success | median steps | B006: success | median steps |
|---|---:|---:|---:|---:|
| 0 (006 at 2.1M) | 0.875 | 8 | 0.875 | 8 |
| 32k | **1.000** | 6 | 0.930 | 8 |
| 65k | 1.000 | 6 | 0.883 | 9 |
| 98k | 1.000 | 5 | 0.898 | 8 |
| 131k | 0.977 | 6 | 0.836 | 8 |
| 164k | 1.000 | 5 | 0.961 | 8 |
| 197k | 1.000 | 5 | 0.523 | 170 |
| 229k | 0.961 | 5 | 0.984 | 7 |
| 262k | 1.000 | 5 | 0.898 | 8.5 |

| Arm | Hits, last 3 checkpoints | Hits, all 8 checkpoints after the start |
|---|---:|---:|
| S (from 005) | 384 / 384 | 0.944 |
| B (from 005) | 338 / 384 | 0.839 |
| S006 | 379 / 384 | 0.992 |
| B006 | 308 / 384 | 0.864 |

- From the second babbler, learning from its own plans works faster: 100% from the first checkpoint
  (32k steps) on, with a median of 5–6 steps.
- Its babbling control fluctuates between 0.52 and 0.98.
- At matched checkpoints, S beats B in 14 of 16 pairs (7 of 8 per babbler). Checkpoints of one run
  are not independent, so this is a count, not a test.

### R3. A hunger head trained on real latents only (R, SR)

| Steps | R: success | median steps | SR: success | median steps |
|---|---:|---:|---:|---:|
| 32k | 0.898 | 9 | 0.906 | 17 |
| 65k | 0.961 | 20.5 | 0.586 | 24 |
| 98k | 0.539 | 135 | 0.922 | 12 |
| 131k | 0.578 | 40 | 0.953 | 12.5 |
| 164k | 0.891 | 14 | 0.953 | 10 |
| 197k | 0.570 | 145 | 0.945 | 8 |
| 229k | 0.898 | 42 | 1.000 | 9 |
| 262k | 0.945 | 57.5 | 0.742 | 17 |

- Hits over the last three checkpoints: R 309 of 384, SR 344 of 384. For comparison, B has 338 and
  S has 384.
- R's one-step edge is no sharper than B's (relief at +2 px: .10–.15 vs .10–.19). Its planner is the
  least stable of all arms.
- SR's failures at 262k are of a new kind (`analyze_planner.py`, 33 failures):
  - the cursor moves *away* from the circle: final gap 15–39 px, larger than at the start;
  - the planner rarely clicks;
  - it expected relief of 0.6–0.8 along those paths.
- A hunger head that never sees imagined latents is exploited wherever imagination drifts off the
  manifold of real latents. Training it on imagined chain steps blurs the edge, but keeps the cost
  robust to the model's own errors.

## Analysis

1. **Learning from its own plans brings the planner to PPO level, still without reward.**
   - From two babblers, the planner's own experience makes it hit (nearly) every test episode in a
     median of 5–6 steps.
   - It takes about 200k steps from babbler 005 and 32k from babbler 006.
   - The losses are unchanged: next frame, inverse dynamics, latent chains, next hunger.
   - Babbling for as long does not do this: B and B006 stay at 0.84–0.86 on average, with drops to
     0.46–0.52.
2. **Why it works: failed clicks are in the training chains, successful ones end them.**
   - The planner's typical mistake, a click just outside the edge, is followed by more steps of the
     same episode. Those steps show the circle still there and the hunger still rising.
   - The latent chains therefore learn that this "relief" does not last.
   - A successful click ends the episode, and the chain stops there.
   - The one-step prediction is not what improved; it got worse (.52 relief at +2 px). The
     consequence over the planning horizon did: the imagined relief now fades instead of
     persisting, so the planner stops exploiting it.
   - This is the relevant quantity, because the planner scores whole imagined trajectories.
3. **The model specializes to its own behavior.** Against B:
   - hunger error along random action sequences tripled (.15 vs .05);
   - movement over the circle got worse (4.5 vs 2.5 px);
   - the one-step relief at the edge blurred (.52 vs .11).
   Straight-path shrinkage worsened in both arms, so it is not specific to S. The planner's
   experience is narrow (directed moves toward circles), and the model tracks what the planner
   does. The cost is general accuracy. For control, accuracy along the planner's own trajectories
   is what counts.
4. **No reward was needed.** The agent improves its world model from the errors of its own
   predictions, gathered while pursuing its drive. In vm-00001, prediction of the agent's own
   actions taught it what its actions do. Here, prediction of the agent's own *goal-directed*
   actions teaches it which of its expectations are wrong.

## Conclusions

- A reward-free agent can reach the performance of PPO trained with reward: 100% hits in a median
  of 5–6 steps. It plans with a world model learned by babbling, and keeps learning that model from
  its own goal-directed experience. This holds from two babblers.
- The gain comes from the planner's own experience, not from more training: babbling for as long
  averages 0.84–0.86. The mechanism is the multi-step imagined consequence of near-miss clicks,
  which this experience corrects.
- Training the hunger head only on real latents (R, SR) makes planning less stable, not more: the
  head must also learn on the imagination it will be queried on.

## Next Steps

- Seeds: the training seed was 1 in every arm.
- Learn from the planner's experience from the start: babbling 0.5–1M steps, then S. How much
  babbling does the loop need?
- Smaller targets (radius 4–8), where PPO from scratch plateaus for 0.5M steps (vm-00003).

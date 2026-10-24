# Task 00004: Learning From Its Own Plans (Reward-Free Self-Correction)

**Date:** 2026-10-23
**Status:** Complete

## Context

vm-00002: a CEM planner on the latent world model of a reward-free babbler hits 89% of the test
episodes. Random actions hit 54%, and the same planner with random costs 23%. PPO (vm-00001) hits
100%. The failures share one pattern (005_P_dyn_s1 at 2.1M, 16 failed episodes of 128):
- the cursor gets within 0–5 px of the circle edge but never enters the circle;
- the planner clicks there 116–192 times per episode;
- its best plan imagines the hunger falling to about 0 (imagined drop > 1.0; successful episodes:
  median 0.37).

The planner exploits an error of the hunger head: relief imagined for clicks just outside the edge.
Longer horizons and other sampling noise do not help (sweep: 0.81–0.89). The agent sees each such
mistake (it clicks and the hunger does not drop), but the model never trains on what the planner
does.

## Hypothesis

1. If the world model keeps training on the planner's own experience, the false relief near the
   edge goes away and the planner approaches 100% success. The losses stay reward-free: the same
   prediction losses as babbling (next frame, inverse dynamics, latent chains, next hunger).
2. The gain comes from the planner's data, not from more training. Continuing to babble for the
   same number of steps does not improve the planner. 005 went from 0.85 at 1.05M to 0.89 at 2.1M.

## Deliverables

1. **C API `vm_rollout_set_action(rollout, t, action[N,3])`** overwrites the action stored for
   step t. The token inputs (previous executed action) already come from Python. Only valid with
   `ppo.use_ppo=0`: logp and value of the replaced action are not recomputed.
2. **`agent_lib.Rollout.set_action(t, a)`.**
3. **`train.py`: `train.actor = "planner"`** (default `"policy"`) with `train.planner` = CEMPlanner
   settings. At every step:
   - the planner picks the actions with the current weights;
   - `agent.act` still runs, to store the tokens and keep the KV cache;
   - the stored action is replaced by the planner's;
   - the world steps on the planner's action;
   - envs that restarted get their warm-start plan reset.
   Requires `ppo.use_ppo=0` and `env.drives=true`.
4. **Experiment vm-00004.** Both arms start from `exp_00002/005_P_dyn_s1/ckpt_02097152.bin` with
   `--init` (optimizer reset) and the `P_dyn` config:
   - **S:** `train.actor=planner`, 262,144 steps (16 iterations).
   - **B (control):** babbling continued for the same 262,144 steps.
   - Milestones every 32,768 steps.
5. **Evaluation per milestone:**
   - `plan.py` success and median steps on the 128 test episodes;
   - `analyze_imagination.py`: false relief rate, approach-test relief, cursor retrieval error;
   - `analyze_planner.py`: in failed episodes, clicks outside and steps inside.

### Added during the experiment: arm R (hunger head on real latents only)

The first S checkpoint blurred the hunger head's edge further instead of sharpening it: imagined
relief at +2 px went from .20 to .44. A linear probe on the encoder's latent separates inside from
outside at 93–95% at ±2 px and 99% at ±3 px, so the information is there. The likely cause is
training on imagined latents: 27 of 35 (77%) of the hunger head's training inputs per sequence are
imagined chain steps (k = 2..7). Those carry 1.5–4 px of cursor error, so the MSE-optimal prediction smears the
relief over that error.

6. **`ppo.dyn_h_horizon`** (C++, ctypes mirror, PyTorch reference, gradient test): only chain steps
   k ≤ dyn_h_horizon train the hunger head. 0 means all steps; 1 means real latents only.
7. **Arm R:** babbling continued from the same checkpoint for 262,144 steps with
   `ppo.dyn_h_horizon=1`. Compare with B. If R sharpens the edge, the next run is S with
   `dyn_h_horizon=1`.

## Implementation details

- The planner runs every step on all 128 envs (auto-reset on): about 0.5–1 s per step, so about
  2 minutes per 16k-step iteration. The update is unchanged (about 5 s).
- Training uses the planner's executed action in three places:
  - the prediction heads;
  - inverse dynamics, which sees larger, directed moves;
  - the dynamics chains.
  The encoder keeps training with the prediction and inverse losses, as in babbling.
- Training-episode success in `metrics.jsonl` is now the planner's own online success.

### Outcome (added during the experiment)

- S reached 100% success at 197–262k steps (median 5–6). B ended at 0.78–0.94.
- R and SR did not help. A hunger head trained only on real latents is exploited where imagination
  drifts (SR at 262k: the cursor moves away from the circle while the planner expects relief).
- Replication from the second babbler: S006 (planner) and B006 (control).

## Verification

- `make test` and `make pytest` pass.
- A new test: replacing the stored actions with the ones `act` returned leaves the update
  bitwise identical. Replacing them with different actions changes the inverse-dynamics loss.
- A smoke run of 2 iterations with `train.actor=planner`.

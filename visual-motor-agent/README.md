# Visual–Motor Agent

> A small, randomly initialized agent learns to operate a mouse from pixels alone, with no labels,
> coordinates, DOM, demonstrations or distance shaping. What does it learn about its own actions
> without being reinforced for it?

Plan: [`../visual_motor_learning_plan.md`](../visual_motor_learning_plan.md) · Literature notes:
[`RESEARCH.md`](RESEARCH.md) · Code: [`dev/`](dev/README.md)

## Objective

Build a pixels-to-mouse agent: a CNN encoder, a causal Transformer, and policy, value and
action-conditioned prediction heads. Use it to study whether self-supervised prediction and
internal drives give it sensorimotor knowledge (what its moves and clicks do) before, or instead
of, reward. Then test whether that knowledge helps it learn the task.

## Hypothesis

1. A forward model trained only to predict the next frame from (frame, action) learns the meaning
   of its motor API without any reward:
   - dx/dy move the cursor;
   - a click while the cursor is over the circle makes the circle disappear;
   - that same click relieves hunger.
2. Sharing that representation with the policy (auxiliary loss), or pre-training it by babbling,
   makes the sparse-reward task learnable faster.

## Success criteria

- Counterfactual action-conditioned predictions beat chance and trivial baselines (copy last
  frame, action-agnostic predictor) on held-out states. Metric: `analyze_dynamics.py`.
- The click contingency is specific: the circle vanishes when the click is inside it, not outside.
- PPO arms compared on steps-to-success and sample efficiency, over seeds, against the
  random-policy baseline (73.5% success at radius 12–16, 39% at radius 4–8).

## Setup

- **Hardware:** QEMU VM, 128 vCPUs (SSE4.2 only), 251 GB RAM; since 2026-10-24 an RTX 3090 (24 GB). The basic
  agent's lifetime can run on it (`online.device=cuda`, tasks 00011–00013: 202 against 24 steps/s).
- **Model:** C++17 + OpenMP + hand-written SSE GEMM, verified against a float64 PyTorch reference
  (gradients match to about 1e-7 relative error).
- **World:** Python/numpy, with an HTML/JS replica that renders pixel-identical frames.
- **Throughput:** about 2k env-steps/s per run; two runs in parallel.

## Progress

| # | Experiment | Status | Key result |
|---|---|---|---|
| [00001](00001-pixel-agent-and-babbling/README.md) | World + C++ agent; sparse-reward PPO; reward-free motor babbling; transfer | Complete | Reward-free babbling (forward + inverse prediction, random policy, no reward) learns: <br>• movement semantics on the background (retrieval 0.91 vs 0.11 chance; over the circle, not yet) <br>• the click contingency (circle vanishes on inside clicks 0.97, outside 0.04) <br>• hunger relief <br>Replicated over 2 seeds. PPO initialized from the babbler reaches 95% success 3.3–3.5× faster (2 seeds). |
| [00002](00002-acting-without-reward/README.md) | Acting without reward: latent world model from babbling + CEM planner on predicted hunger | Complete | The planner hits 89% of test episodes (median 11 steps) using only the babbler's model and its hunger. The same planner with random costs hits 23%; random actions 46–54%; PPO with reward 100%. Failures: the planner exploits the blurred inside/outside edge of the hunger head and clicks just outside the circle. |
| [00003](00003-small-targets-drives-coordconv/README.md) | Small targets (radius 4–8) transfer; arms B, E, F, D; plain architecture seeds | Complete | • Radius 4–8: babbler-initialized PPO reaches 95% success 3.9–4.1× faster (2 seeds); from scratch, PPO sits near the random level for about 0.5M steps. <br>• The hunger cost and prediction speed up PPO (D: 262k vs A 508–590k); bonuses alone do not (one seed per arm). <br>• CoordConv slows pure PPO 2–3× (plain 180–262k vs 508–590k, 3 seeds each). |
| [00004](00004-learning-from-own-plans/README.md) | Learning from its own plans: the world model keeps training on the planner's experience, with the same reward-free losses | Complete | From two babblers, the planner's own experience brings it to 100% hits in a median of 5–6 steps, PPO's level, with no reward. It hits 384/384 and 379/384 over the last three checkpoints. Babbling for as long: 338/384 and 308/384. Mechanism: failed near-edge clicks enter the training chains, so the imagined relief after them fades. A hunger head trained only on real latents is less stable. |
| [00005](00005-one-lifetime-online-learning/README.md) | One lifetime: a single world, one agent, a gradient step after every step, one-frame model, no reward | Complete | With memory (L), online it misses only 3–5 of 14–15k episodes after 50k steps, hitting in 9–12 steps, with no reward. Frozen checkpoints score only 0.82–0.99: from the same weights, learning during play hits every episode, while frozen it misses far starts. Without memory (P): below babbling online, test 0.02–0.56. |
| [00006](00006-basic-next-state-predictor/README.md) | The basic agent: predict the next state (frame, drives, move) from the last 8 real ones, make that move, learn every step; no planning, no reward | Complete | On average, no arm beats the random initial model (0.44–0.48); the best arm averages 0.27–0.33 over its checkpoints and ends at 0.34–0.38. Without replay, the encoder features blow up (norm 10²–10⁵), which hides the hunger input. |
| [00007](00007-salience-gated-memory/README.md) | Salience-gated memory: 8-step chunks enter long-term memory when they lead to hunger relief or were surprising (error in predicting the whole next state); basic and planner agents | Complete | Planner agent: storing relief speeds up early learning (test success 0.90–0.95 at 10k, against 0.70–0.93), with no optimism. Follow-up with 4 seeds: a random memory of the same size learns more slowly than storing everything, so the gain comes from what is stored. Basic agent: no gate helps, with or without inverse dynamics; its features lose the cursor during training, so it cannot aim. |
| [00008](00008-detached-policy-head/README.md) | Detached policy head: the basic agent's move loss trains only the policy head, so its features learn from prediction alone; with and without inverse dynamics, and with a hidden layer in the head | Complete | The move loss eroded the features: detached, the cursor and the direction hold (with inverse dynamics, cursor R² 0.79–0.91 vs 0.42–0.52). A linear head still barely aims. With one hidden layer the hindsight command steers it, and **the basic agent aims for the first time**: test success 0.63–0.82 at 50k vs 0.45 at birth, no reward, no planner. |
| [00009](00009-planner-as-teacher/README.md) | Planner as teacher: the planner agent's policy head also learns the executed moves (a habit), through the features or detached, linear or with a hidden layer; then it acts alone, without the planner | Complete | **A habit learned from the planner's own moves acts alone**: a detached linear head hits in 0.80–0.96 of the test episodes at 30k (birth 0.52–0.56), aiming at the circle, without reward or planner; slower than the planner (26–44 vs 8–14 steps). With its own clip (`ppo.pi_own_clip`) it leaves the world model bitwise unchanged. Through the features the move loss erodes them (cursor R² 0.25 vs 0.9) and both planner and habit suffer. A hidden layer does not help. |
| [00010](00010-patch-tokens-and-distractors/README.md) | Patch tokens: one token per 16×16 patch and per input (drives, move, command), each predicting its own part of the next state; moving grey dots the agent does not cause | Complete | **Patch tokens hold the scene but the policy's token does not**: all patch tokens together give cursor R² 0.94–0.98 (0.81–0.93 already at birth), the move token, which no loss trains when the head is detached, 0.21–0.33; no patch agent aims, even with a hidden layer. Patches couple only weakly and only without dots. Both models learn where their own move acts; the CNN separates self from the dots more cleanly. Moving dots delay vm-00008's DMI by 20k–30k steps (0.87–0.90 at 40k–50k in one seed). |
| [00011](00011-continual-world/README.md) | A continual world: no resets (a new circle after a hit), moving circles, hazards that raise hunger, a wrap-around screen, an egocentric view; vm-00008's DMI on the GPU | Complete | **Episode resets did the basic agent's exploring**: with walls, a continual world starves it (its drift sticks to a wall; 0–2 hits per 1,000 steps). On a torus it forages (episodic test 0.89–0.95 at 60k, its best so far), and **with an egocentric view it hits every test episode by 40k**. It never avoids hazards: its hindsight command marks only moves before relief. |
| [00012](00012-planner-continual-world/README.md) | The planner agent (L) in vm-00011's continual torus worlds, on the GPU; hazards at two costs | Complete | **The planner forages 3–7× better than the basic agent** (28–33 hits per 1,000 steps on the torus; **135–142 with the egocentric view**; moving circles cost it nothing). **Neither agent avoids hazards**, even at 0.1 hunger per step: nothing in the network represents hazard contact (probe R² ≤ 0.12; the drives head and the imagined hunger see no difference), because a hazard step is a tiny error beside a hit. |
| [00013](00013-drive-changed-weight/README.md) | Weighting steps where hunger changes unusually (`ppo.drive_changed_weight`) in the drives and latent hunger losses, so hazard contact gets learned | Complete | **Weighting (10×, 100×) does not make hazards learned for the planner**: its imagination starts from the encoder's features, which learn from frame prediction only and never hold contact (R² ≤ 0.08), so its hunger model stays blind and it does not avoid them. The planner can only plan around what its latent state holds. |
| [00014](00014-unified-patch-transformer/README.md) | One Transformer for everything (`../visual_motor_learning_transformer.md`): 8×8 RGB patch tokens, nothing detached, the move loss through the Transformer or not; 16×16 patches and the whole frame as one token; the hazard world | Complete | **Letting the move loss train the Transformer erases the cursor from the patch tokens** (R² 0.87–0.93 detached → below 0) and the agent stops aiming; detached, 8×8 patches keep the scene and hazard contact under the cursor (R² ≈ 0.7). No arm learns that contact raises hunger, but hunger sat at its cap of 1 on about half of the steps, where a hazard leaves no trace (found here). The whole frame as one token never locates the cursor. |
| [00015](00015-slow-hunger/README.md) | Slow hunger (0.001 per step, hazards 0.02), so that hunger rarely reaches its cap of 1 and hazard contact shows in the drives; 8×8 patch agents, DMI, the planner | Complete | **With the trace present, the CNN agent's drives head learns hazard contact by 60k** (+0.012 to +0.019 of the world's +0.020), the first model to do so. The 8×8 patch models hold contact in the token under the cursor but do not predict its cost; the planner's imagination stays blind. No arm avoids hazards. |
| [00016](00016-target-architecture/README.md) | The target architecture (`ARCHITECTURE.md`) against its parts: the act token, the signed hindsight command, 32 parallel bodies early in life (`lifetime.py`), 8×8 patch tokens, slow hunger | Complete | **32 bodies early in life, shrinking to one, make the best basic agent so far**: test episodes 0.85–0.98 at 10k and 0.91–0.98 at 60k, forage rate 1.8× the control's, with the same number of updates (500k body-steps instead of 60k). The act token holds hazard contact (R² 0.34–0.69) but not the direction to the circle; the signed command does not produce avoidance. |

## Open Questions

- **Partly answered (vm-00001).**
  - A reward-free predictor needs about 0.5M (movement) to 1M (click/hunger contingencies) random-
    policy steps, about 4–8k updates.
  - It needs U-Net skips, change prediction, nonlinear action conditioning and a high update rate.
  - Inverse dynamics speeds it up 2–4×.
- **Partly answered.** Babbling pre-training transfers: PPO is 3.3–3.5× faster (2 seeds). Open: how much of the gain comes from the encoder (cursor localization) vs the
  decoder/latent, and does it hold for small targets?
- **Answered: yes (vm-00002).** The learned model can *act* without reward, by planning to minimize
  predicted hunger. A latent dynamics model trained during babbling supports a CEM planner: 89% hits
  against 23% for the control. Open: why the planner falls short of 100% (it exploits model errors at
  the circle's edge), and whether learning from its own failed clicks closes the gap. **It does**
  (vm-00004, two babblers): 100% hits in a median of 5–6 steps.
- At radius 12–16 the task is close to non-sparse for random clicking. At radius 4–8, PPO stays at
  the random level for about 0.6M steps before learning (vm-00001 R5). Which conditions make
  exploration genuinely hard (smaller targets, short episodes, click cost)?
- Drives as inputs/costs vs exploration bonuses (vm-00003, one seed per arm): the hunger cost speeds
  up PPO, and bonuses do not. More seeds are needed.
- **Partly answered: moving the decision into the network.** In vm-00002 to vm-00005, the network only
  predicts. The goal (lower predicted hunger, summed over 12 imagined steps) and the search (CEM)
  are hand-written code outside it (`dev/plan.py`). A pure predictor has no preference: in vm-00006
  A, its own moves just drift. Ideas to try:
  1. **Planner as teacher (habit).** Run vm-00005's L with `ppo.move_coef=1`, so the policy head
     also learns to predict the planner's executed moves. Then evaluate the head's own moves without
     the planner (`online.py --eval`, as in vm-00006). Search would then be needed only while
     learning. **Done in vm-00009:** a detached linear head acts alone at 0.80–0.96 test success at 30k
     (its planners 0.88–0.95), in 26–44 steps against the planner's 8–14. The babbled moves enter the move
     loss. The move loss must not train the shared encoder: through it, it erodes the features. Open:
     does the habit hold once the planner is switched off?
  2. **Hindsight hunger input, no search** (vm-00006 A+H). Hits are rare (1–4 per 1,000 steps), so
     only a few percent of the training steps carry the signal. **Tried in vm-00007:** memory gated
     by relief raises the labelled share to 9–14%, but the agent still does not aim, because its
     features lose the cursor during training. Next: stop the move loss's gradient at the policy
     head, or start from a world model learned by babbling or by L. **Done in vm-00008:** detached, the
     features hold; with one hidden layer in the head, the command steers it, and the basic agent aims
     (0.63–0.82 at 50k against 0.45 at birth), without reward or planner. Moving dots delay it by
     20k–30k steps (vm-00010).

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

- **Hardware:** QEMU VM, 128 vCPUs (SSE4.2 only), 251 GB RAM, **no GPU**.
- **Model:** C++17 + OpenMP + hand-written SSE GEMM, verified against a float64 PyTorch reference
  (gradients match to about 1e-7 relative error).
- **World:** Python/numpy, with an HTML/JS replica that renders pixel-identical frames.
- **Throughput:** about 2k env-steps/s per run; two runs in parallel.

## Progress

| # | Experiment | Status | Key result |
|---|---|---|---|
| [00001](00001-pixel-agent-and-babbling/README.md) | World + C++ agent; sparse-reward PPO; reward-free motor babbling; transfer | Complete | Reward-free babbling (forward + inverse prediction, random policy, no reward) learns: <br>• movement semantics on the background (retrieval 0.91 vs 0.11 chance; over the circle, not yet) <br>• the click contingency (circle vanishes on inside clicks 0.97, outside 0.04) <br>• hunger relief <br>Replicated over 2 seeds. PPO initialized from the babbler reaches 95% success 3.3–3.5× faster (2 seeds). |
| [00002](00002-acting-without-reward/README.md) | Acting without reward: latent world model from babbling + CEM planner on predicted hunger | Complete | The planner hits 89% of test episodes (median 11 steps) using only the babbler's model and its hunger. The same planner with random costs hits 23%; random actions 46–54%; PPO with reward 100%. Failures: the planner exploits the blurred inside/outside edge of the hunger head and clicks just outside the circle. |
| [00003](00003-small-targets-drives-coordconv/README.md) | Small targets (radius 4–8) transfer; arms B, E, F, D; plain architecture seeds | Complete (pending tag) | • Radius 4–8: babbler-initialized PPO reaches 95% success 3.9–4.1× faster (2 seeds); from scratch, PPO sits near the random level for about 0.5M steps. <br>• The hunger cost and prediction speed up PPO (D: 262k vs A 508–590k); bonuses alone do not (one seed per arm). <br>• CoordConv slows pure PPO 2–3× (plain 180–262k vs 508–590k, 3 seeds each). |
| [00004](00004-learning-from-own-plans/README.md) | Learning from its own plans: the world model keeps training on the planner's experience, with the same reward-free losses | Complete (pending tag) | From two babblers, the planner's own experience brings it to 100% hits in a median of 5–6 steps, PPO's level, with no reward. It hits 384/384 and 379/384 over the last three checkpoints. Babbling for as long: 338/384 and 308/384. Mechanism: failed near-edge clicks enter the training chains, so the imagined relief after them fades. A hunger head trained only on real latents is less stable. |

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

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

## Open Questions

- **Partly answered (vm-00001).**
  - A reward-free predictor needs about 0.5M (movement) to 1M (click/hunger contingencies) random-
    policy steps, about 4–8k updates.
  - It needs U-Net skips, change prediction, nonlinear action conditioning and a high update rate.
  - Inverse dynamics speeds it up 2–4×.
- **Partly answered.** Babbling pre-training transfers: PPO is 3.3–3.5× faster (2 seeds). Open: how much of the gain comes from the encoder (cursor localization) vs the
  decoder/latent, and does it hold for small targets?
- Can the learned forward + drive model *act* without reward, by planning to minimize predicted
  hunger? One-step predictions are accurate (0.8 px), but pixel-space rollouts diverge after one
  step, so latent dynamics are needed.
- At radius 12–16 the task is close to non-sparse for random clicking. At radius 4–8, PPO stays at
  the random level for about 0.6M steps before learning (vm-00001 R5). Which conditions make
  exploration genuinely hard (smaller targets, short episodes, click cost)?
- Drives as inputs/costs vs exploration bonuses: confounded unless controlled (plan §9). Runs B, D,
  E and F are configured but not yet run on the final architecture.

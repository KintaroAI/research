# Task 00008: Detached Policy Head: Keep the Move Loss Out of the Shared Features

**Date:** 2026-10-24
**Status:** Complete

## Context

The basic agent (vm-00006, vm-00007) predicts its next move, frame and drives, makes that move, and
learns every step. It never aims, and its features lose the cursor during training:
- the cursor's R² from the visual features v falls from 0.62 at 10k steps to 0.46 at 50k (U seed 1);
- the circle's R² falls from 0.94 (random encoder) to 0.28–0.39;
- the direction to the circle is readable from the Transformer output z with R² 0.21–0.26.

The planner agent (vm-00005 L) keeps them: cursor 0.96 from v, direction 0.84 from z at 100k. Its
encoder is trained only by frame, drive and inverse-dynamics prediction. The basic agent's encoder
and Transformer are also trained by the move loss, -log π(a_t) of its own sampled moves. Per sample,
that gradient grows about as 1/σ as the moves narrow (σ falls from 0.6 to 0.13–0.20).

The memory is not the bottleneck: the moves of the 12 steps before a hit point at the circle
(cosine +0.35 to +0.39), but the agent's features cannot express "toward the circle".

## Hypotheses

1. **The move loss erodes the features.** With its gradient stopped at the policy head's input z
   (`ppo.pi_detach=1`):
   - with inverse dynamics, v keeps the cursor (R² ≥ 0.9, as in L) and z keeps the direction;
   - without inverse dynamics, nothing teaches v the cursor (it stays near the random encoder's
     0.06), but the circle is no longer erased (R² stays near 0.94).
2. **With features that hold the direction, the agent aims.** Its aim cosine rises clearly above
   vm-00007's −0.05 to +0.04, and its test success beats the random initial model (0.44–0.48).
3. **A linear head cannot use the command.** A linear readout of z can aim, but it cannot make
   aiming depend on the hindsight command unless z holds their product. A head with a hidden layer
   (`model.pi_hidden`, stage 2) can. Then asking for relief rather than none changes the aim
   cosine by much more than vm-00007's 0.017–0.028.

## Deliverables

- `ppo.pi_detach` (C++ `PPOConfig`): the policy head learns, its gradient stops at z.
- `model.pi_hidden` (stage 2): a policy head with one ReLU hidden layer, π = W2 relu(W1 z + b1) + b2;
  0 = the linear head (unchanged). Checkpoint header slot 37.
- Reference gradients for both (PyTorch float64), a test that the detached move loss trains only
  the head, and byte-identical checkpoints with the defaults.
- Experiment vm-00008 (README + Makefile), runs in `~/data/research/visual-motor-agent/exp_00008/`.

## Method

All arms are vm-00007's basic agent (`configs/AHR_gated.json`: window of 8 real states, hindsight
hunger input, 20% babbled moves, batch 8), 50k steps, 4 threads, seeds 1–2, under memory U (store
everything) and R (relief chunks plus 10% of the rest).

| Arm | Policy head | Inverse dynamics | Controls (vm-00007, same code by default) |
|---|---|---|---|
| D | linear, detached | no | U, R (runs 001–004) |
| DI | linear, detached | yes | I+U, I+R (runs 017–020) |
| DMI (stage 2) | hidden layer 128, detached | yes | DI |

**Evaluation** (as in vm-00007):
- own moves on the 128 test episodes every 10k steps (`online.py --eval`);
- feature probes (cursor, circle, direction from v and z) at 10k, 30k and 50k
  (`analyze_self_actor.py probe`);
- aim cosine, steps at a wall and the command effect at 30k and 50k (`analyze_self_actor.py aim`);
- online hits and memory composition from the logs.

## Implementation details

- `seq_bwd_core` passes a null input gradient to the policy head's `linear_bwd` when `pi_detach` is
  set. Both update paths (one thread per sequence, frame-parallel) go through it.
- The PyTorch reference computes π from `z.detach()` when `pi_detach` is set.
- `pi_detach` is appended to `PPOConfig` (C++ and ctypes), default 0.
- Stage 2: new parameters `pi_h_w`, `pi_h_b` (size 0 when `pi_hidden=0`, so offsets and old
  checkpoints are unchanged). The hidden layer is computed wherever π is: the sequence forward
  (training, `vm_forward`) and the acting path.

## Verification

- Gradient tests against the reference for the detached move loss alone and with PPO, at 4 and 16
  threads. Negative control: the old library fails exactly these cases.
- With only the move loss and `pi_detach=1`, every gradient outside `pi_w`, `pi_b` and `logstd` is
  exactly zero.
- 300-step runs with the defaults give byte-identical checkpoints before and after the change
  (with and without inverse dynamics).
- `make test_all` passes.

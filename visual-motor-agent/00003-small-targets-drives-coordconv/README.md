# vm-00003: Small-Target Transfer, Drives and Bonuses, CoordConv Seeds

**Date:** 2026-10-23
**Status:** Complete (pending tag)
**Source:** *tagged on completion as `exp/vm-00003`* (code: `visual-motor-agent/dev/`)
**Outputs:** `~/data/research/visual-motor-agent/exp_00003/`

## Goal

Three follow-up questions from vm-00001, each answered with existing code and configurations:

1. **Does babbling pre-training remove the plateau on small targets?** At radius 4–8, PPO from
   scratch stays at random-level success for about 0.6M steps (vm-00001 R5).
2. **Do the plan's exploration bonuses and hunger drive help sparse-reward PPO?** These are arms
   B, D, E and F of the plan's experiment matrix (§9), run on the final architecture.
3. **Do the CoordConv input channels slow RL?** In vm-00001 R3, the plain architecture (one seed)
   reached 95% success at 180k steps, against 508–590k for three skip-architecture seeds.

## Hypothesis

1. PPO initialized from the 1M-step babbler of vm-00001 (run 012) skips most of the small-target
   plateau. The babbler's encoder already localizes the cursor and its forward model knows that
   clicking on red is consequential. The babbler saw only radius 12–16 circles, so this also tests
   generalization to smaller ones.
2. The bounded bonuses and hunger cost change little at radius 12–16, where random clicking already
   succeeds 50–70% of the time. They add at most 0.37 per episode against +10 for a hit.
3. Open; two more seeds of the plain architecture decide it.

## Method

All runs: radius 12–16 unless noted, sparse reward +10 per hit and −0.001 per step, evaluation on the
fixed 256-episode test set with bonuses off.

| Run | Config | Architecture | Steps | Compare with |
|---|---|---|---:|---|
| `A_small_ctx8inv_s1` | A_small (radius 4–8) | skip, ctx 8, inverse head | 2M | the run below |
| `T_small_from_babble_s1` | A_small, initialized from vm-00001 run 012 at 1M | same | 2M | the run above |
| `A_small_ctx8inv_s2`, `T_small_from_babble_s2` | the same pair, seed 2; T from vm-00001 run 015 (babbler seed 2) at 1M | same | 2M | each other |
| `B_arch_s1` | B_exploration: bonuses | skip | 1M | vm-00001 A skip seeds 1–3 |
| `E_arch_s1` | E_drives_only: drives observed, hunger cost 0.01 | skip | 1M | same |
| `F_arch_s1` | F_full_minus_pred: bonuses + drives | skip | 1M | same |
| `D_arch_s1` | D_combined_w10: bonuses + drives + prediction | skip | 1M | vm-00001 C skip seeds 1–3 |
| `A_plain_s2`, `A_plain_s3` | A_baseline | plain (no CoordConv) | 1M | vm-00001 run 002 (plain, seed 1) |

"Skip" means `coord=1, pred_residual=1, pred_skip=1`, as in vm-00001.

## Log

### 2026-10-23
- Queue started at 15:50 (`dev/queue_exp3.txt`, 2 slots × 64 threads).
- At 16:17 the queue was cut to 1 slot (`queue_exp3b.txt`) to make room for vm-00002's babbling runs.
  The first B and E jobs were stopped after one iteration and deleted. The small-target seed-2 pair
  was added to the queue.
- Seed 1 and seed 2 of the small-target pair finished, and so did B_arch_s1.
- Queue c finished at 19:54: E, F, D and plain seeds 2–3, each 1M steps.
- 18:15: runner b had been idle behind vm-00004's runs (one slot, and other training jobs count
  against it). It was stopped. The jobs it had not started (E, F, D, plain seeds 2 and 3) went to
  `queue_exp3c.txt` with 2 slots.

## Results

### R1. Small targets (radius 4–8): PPO from scratch vs from the babbler

Evaluation every 5 updates on 256 fixed test episodes (stochastic / deterministic policy):

| Run | Steps to success ≥ .95 | Steps to det ≥ .9 | succ / det @262k | @508k | @1M | Final median steps |
|---|---:|---:|---:|---:|---:|---:|
| A scratch, seed 1 | 1.41M | 1.49M | .32 / .09 | .36 / .10 | .71 / .21 | 13 |
| **T from babbler, seed 1** | **344k** | **344k** | .94 / .79 | 1.00 / 1.00 | 1.00 / 1.00 | 8 |
| A scratch, seed 2 | 1.65M | 1.90M | .26 / .07 | .29 / .09 | .61 / .14 | 20 |
| **T from babbler, seed 2** | **426k** | **426k** | .78 / .54 | .97 / .95 | 1.00 / 1.00 | 8 |

- Babbler initialization reaches 95% success **4.1× and 3.9× faster**, and a reliable
  deterministic policy 4.3× and 4.5× faster.
- The from-scratch agents sit near the random level (.26–.36) for about 0.5M steps. The
  babbler-initialized agents start learning at once.
- Both T runs end with a shorter median, 8 steps against 13 and 20.
- The babbler only ever saw radius 12–16 circles, so the benefit generalizes to targets a third the
  size.

### R2. The plan's arms B, E, F, D on the skip architecture (radius 12–16, seed 1)

Compared with vm-00001's A and C on the same architecture (3 seeds each). Same evaluation as R1.

| Run | Bonuses | Drives + hunger cost 0.01 | Prediction | Success ≥ .95 | det ≥ .9 | succ / det @344k |
|---|:-:|:-:|:-:|---:|---:|---:|
| A, seeds 1–3 (vm-00001) | | | | 508k / 590k / 590k | 508k / 672k / 590k | .95 / .40, .76 / .27, .80 / .27 |
| C, seeds 1–3 (vm-00001) | | | ✓ | 344k / 508k / 426k | 426k / 508k / 508k | 1.00 / .90, .83 / .33, .79 / .30 |
| B | ✓ | | | 508k | 590k | .79 / .28 |
| E | | ✓ | | 426k | 426k | .95 / .60 |
| F | ✓ | ✓ | | 344k | 426k | .97 / .66 |
| **D** | ✓ | ✓ | ✓ | **262k** | **344k** | 1.00 / .94 |

All arms end at success 1.00 / det 1.00, with a median of 5–6 steps.

### R3. The plain architecture (no CoordConv), seeds 2 and 3

| Run | Success ≥ .95 | det ≥ .9 | succ / det @180k | @344k |
|---|---:|---:|---:|---:|
| plain seed 1 (vm-00001 run 002) | 180k | 344k | .96 / .64 | 1.00 / 1.00 |
| plain seed 2 | 262k | 262k | .82 / .38 | 1.00 / 1.00 |
| plain seed 3 | 262k | 262k | .72 / .28 | 1.00 / 1.00 |
| skip (CoordConv) seeds 1–3, vm-00001 | 508k / 590k / 590k | 508k / 672k / 590k | .73 / .30, .51 / .21, .73 / .24 | .95 / .40, .76 / .27, .80 / .27 |

Without a prediction loss, the only difference that reaches the policy is the two CoordConv input
channels. The prediction-head options (`pred_residual`, `pred_skip`) are unused when
`ppo.pred_coef=0`.

## Analysis

1. **Babbling pre-training removes the small-target plateau (H1 confirmed, 2 seeds).**
   - From scratch, PPO at radius 4–8 stays near the random level for about 0.5M steps and needs
     1.4–1.7M steps to reach 95%.
   - From the 1M-step babbler it learns at once and gets there 3.9–4.1× sooner.
   - The babbler never saw a circle this small. What transfers is its knowledge of the cursor and of
     clicking on red, not of the target's size.
2. **Drives help sparse-reward PPO, bonuses do not (H2 partly refuted, one seed per arm).**
   - With bonuses alone, B is exactly A's speed: 508k, against 508–590k for A.
   - The hunger cost speeds learning: E reaches 95% at 426k, F at 344k, both outside A's range. The
     cost charges 0.01 × hunger per step, so episodes that stall cost more.
   - Adding prediction on top (D) is fastest of all: 262k. That is below every A and C seed.
   - These are single seeds. E and F are just outside A's range, and C's range already reaches
     down to 344k.
3. **CoordConv slows RL by about 2–3× (H3 answered, 3 seeds each).**
   - The plain architecture reaches 95% at 180–262k; with the two coordinate channels it takes
     508–590k. The ranges do not overlap.
   - CoordConv is what makes reward-free prediction learn the cursor's dynamics (vm-00001), so the
     two goals pull in different directions. For PPO alone, the plain encoder is better.

## Conclusions

- Babbler initialization is worth 4× on small targets, where exploration is genuinely hard (2 seeds).
- The hunger drive used as a cost, and the prediction loss, each speed up PPO. Movement and
  empty-click bonuses do not (one seed per arm).
- The CoordConv input channels, needed for reward-free learning of action consequences, slow pure
  PPO 2–3× (3 seeds each).

## Next Steps

- Seeds 2–3 for E, F and D. Their gains over A are single-seed and partly within the range of C.
- A plain encoder for PPO with a CoordConv prediction head, if the coordinate channels can be
  confined to the decoder.

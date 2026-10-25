# vm-00013: Weighting Unusual Drive Changes, So That Hazard Contact Is Learned

**Date:** 2026-10-25
**Status:** Complete (pending tag)
**Source:** *tagged on completion as `exp/vm-00013`* (code: `visual-motor-agent/dev/`)
**Outputs:** `~/data/research/visual-motor-agent/exp_00013/`
**Task plan:** [`../tasks/00017-drive-changed-weight.md`](../tasks/00017-drive-changed-weight.md)

## Goal

In vm-00012 neither agent avoided hazards, even at 0.1 hunger per step inside one, and nothing in the network
represented hazard contact. A hazard step changes hunger by 0.1, a hit by up to 1.0, so in squared error a hazard
step weighs about a hundredth of a hit. vm-00001 solved the same problem for the 1%-of-pixels cursor by weighting the
pixels that change. Here, steps where hunger changes by more than its steady rise get extra weight in the drives
loss and in the latent hunger loss.

## Hypothesis

1. With the weight, contact is represented: a probe reads "inside a hazard" from the features, the drives head
   predicts the extra hunger, and the planner's hunger model imagines it.
2. The planner agent then avoids hazards: fewer steps inside one than chance (about 6.5%).
3. The basic agent does not, even if contact is represented: its hindsight command still marks only relief.

## Method

- `ppo.drive_changed_weight=10`, `ppo.drive_change_eps=0.01`: a step counts 11 times in the drives loss and in the
  latent hunger loss when hunger changes by more than 0.01 over it (the steady rise is 0.005; a hazard step adds
  0.02 or 0.1; a hit removes up to 1.0). C++, PyTorch reference and CUDA; gradients verified (tests), and the defaults
  give byte-identical 300-step checkpoints for the planner and the basic agent.
- World: vm-00012's continual torus with 2 hazards. Arms (seeds 1–3): W10P (planner L, hazards 0.1/step, 40k steps),
  W10D (DMI, 0.1, 60k), W02P (L, 0.02, 40k), W02D (DMI, 0.02, 60k). Controls without the weight: vm-00012's LTH10,
  DTH10, LTH and vm-00011's TH.
- Measures: online hits and the share of steps in a hazard; `analyze_hazards.py contact` (probes, drives head) and
  `hazard` (the planner's imagined hunger inside against outside).

## Log

### 2026-10-25
- **Implemented and verified** `ppo.drive_changed_weight` (task 00017): 667 tests pass with it (two new reference
  gradient settings, one new GPU setting).
- **Launched** the 12 lifetimes (about 01:28).
- **All 12 finished at 01:45.** Results (seeds 1, 2, 3):

  | Arm | hits per 1,000 steps, last 10k | in a hazard, 0–5k → last 10k | control without the weight (vm-00012 / vm-00011) |
  |---|---|---|---|
  | W10P (planner, 0.1) | 28.6, 32.0, 24.3 | 0.079 → 0.062, 0.059 → 0.052, 0.081 → 0.077 | LTH10: 23.1, 23.0, 22.6; 0.058, 0.055, 0.067 |
  | W10D (DMI, 0.1) | 9.0, 8.5, 7.6 | 0.072 → 0.088, 0.064 → 0.059, 0.095 → 0.104 | DTH10: 6.7, 8.2, 7.8; 0.067, 0.055, 0.071 |
  | W02P (planner, 0.02) | 29.1, 33.2, 31.2 | 0.065 → 0.071, 0.070 → 0.061, 0.080 → 0.077 | LTH: 25.1, 24.6, 27.8; 0.070, 0.058, 0.068 |
  | W02D (DMI, 0.02) | 10.6, 9.4, 7.5 | 0.078 → 0.071, 0.053 → 0.053, 0.076 → 0.086 | TH (at 60k): 9.5, 8.0, 6.8 |

- **Contact at 40k** (`analyze_hazards.py contact`; R² of "inside a hazard" from v and z; the drives head's next hunger
  inside vs outside, the world's +0.100 or +0.020):

  | Arm | R² from v | R² from z | drives head, inside − outside |
  |---|---|---|---|
  | W10P | 0.08, −0.02, −0.02 | 0.12, 0.02, 0.01 | **+0.114**, −0.025, −0.001 |
  | W10D | **0.28**, −0.01, 0.06 | **0.22**, 0.00, 0.07 | +0.033, −0.006, +0.006 |
  | W02P | 0.01, 0.06, 0.02 | 0.06, 0.06, 0.04 | +0.045, +0.006, +0.011 |
  | W02D | 0.13, 0.00, 0.05 | 0.15, 0.01, 0.05 | +0.008, −0.002, +0.002 |
  | controls LTH10, DTH10 | −0.05 to 0.10 | −0.03 to 0.12 | −0.017 to +0.013 |

- **The planner's imagined hunger** (`analyze_hazards.py hazard`, extra rise over 12 still steps inside against
  outside): W10P +0.059, −0.088, +0.018 (world +0.630); W02P −0.048, +0.009, +0.026 (world +0.183).
- **Reading:** the weight is at most enough in one or two lifetimes for the frame-reading parts (W10P seed 1's drives
  head predicts the hazard's +0.1 almost exactly; W10D seed 1's features partly hold contact). It is not enough for
  the planner's latent hunger head, which reads the detached v. Neither agent avoids hazards. The planner forages a
  little better with the weight (24–33 against 23–28 hits per 1,000 steps), within the seed spread.
- **Launched a follow-up** with weight 100 (planner, hazards 0.1; runs 013–015): if it still fails, the limit is that v
  cannot learn from the drives at all (it is detached from the latent heads, and the drives head reaches it only
  through the Transformer).
- **Weight 100 does not help either** (W100P, runs 013–015, at 40k): R² of contact from v 0.01, −0.00, −0.01 and from z
  0.03, 0.03, 0.02; drives head inside − outside +0.036, +0.054, +0.004; the planner's imagined extra rise −0.021,
  +0.028, +0.014 (world +0.630); in a hazard over the last 10k 0.077, 0.058, 0.076; hits 32.8, 30.4, 32.4 per 1,000 steps.
  The heavy weight also spoils the hunger model's calibration: it now imagines a rise of 0.51–0.56 over 12 still steps
  outside a hazard, where the world's is 0.07.
- **Written up; committed with vm-00011 and vm-00012** (the GPU ports of the latent dynamics, the patch tokens and the
  drive weight, and the world options, are in the same commit). PPO is the one part of the model not on the GPU; only
  `train.py` uses it.

## Results

| Planner agent, hazards 0.1/step, 40k | weight 0 (vm-00012) | 10 | 100 |
|---|---|---|---|
| R² of contact from v | −0.05 to 0.07 | −0.02 to 0.08 | −0.01 to 0.01 |
| imagined extra hunger inside (world +0.63) | −0.009 to +0.013 | −0.088 to +0.059 | −0.021 to +0.028 |
| share of steps in a hazard, last 10k | 0.055–0.067 | 0.052–0.077 | 0.058–0.077 |
| hits per 1,000 steps, last 10k | 22.6–23.1 | 24.3–32.0 | 30.4–32.8 |

For DMI (weight 10): contact R² up to 0.28 in one lifetime, no avoidance (0.059–0.104).

## Analysis

- **The weight does not make contact learned where it matters** (hypothesis 1, not supported). The drives head, which
  reads the Transformer output, picks it up in a few lifetimes (+0.11 in W10P seed 1, +0.05 in W100P seed 2). The
  encoder's features v never hold it. The planner's hunger model reads v, and v is detached from the latent heads by
  design (vm-00002: their gradients collapse the features), so no weight on the hunger loss can teach v about contact.
- **No avoidance** (hypotheses 2 and 3): the planner has nothing to avoid in its imagination, and the basic agent has
  no signal for a rise in hunger.
- **The bottleneck is the planner's latent state, not the loss scale.** Its imagination starts from v, which encodes
  what frame prediction needs; frame prediction never needs "the cursor overlaps a blue circle", because the decoder's
  skips copy the hazard's pixels.

## Conclusions

1. **Weighting unusual drive changes, the remedy that made the cursor learnable in the frame (vm-00001), does not make
   hazard contact learnable for the planner** at 10× or 100×.
2. **The planner can only plan around what its latent state holds.** v is trained by frame prediction alone, so
   anything that matters only to the drives is invisible to its imagination.

## Next Steps

- **Let the hunger loss reach the encoder**, scaled down (a `dyn_h_backbone_scale`): contact enters v, at the risk
  vm-00002 found for the latent loss (test the hunger loss alone).
- **Imagine from the Transformer's token** (z, which also sees the drives) instead of v.
- **A two-sided hindsight command** for the basic agent (the achieved hunger change).

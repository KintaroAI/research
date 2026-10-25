# vm-00015: Slow Hunger, So That Hazard Contact Shows in the Drives

**Date:** 2026-10-25
**Status:** In Progress
**Source:** *tagged on completion as `exp/vm-00015`* (code: `visual-motor-agent/dev/`)
**Outputs:** `~/data/research/visual-motor-agent/exp_00015/`
**Task plan:** [`../tasks/00019-slow-hunger.md`](../tasks/00019-slow-hunger.md)

## Goal

Hunger is clipped to [0, 1], and at the cap a hazard step changes nothing. vm-00014 found that hunger sat at the cap
on about half of the steps, so only about a third of hazard steps showed in the drives at all, in every hazard
experiment so far (vm-00011–14). Here hunger rises slowly (0.001 per step instead of 0.005) and hazards cost 0.02 per
step, so the cap is rarely reached and most hazard steps leave a trace.

## Hypothesis

1. With the trace present, hazard contact is learned: the drives head predicts the extra hunger inside a hazard, and
   the planner's hunger model imagines it.
2. The planner then avoids hazards (fewer steps in one than about 0.063, chance).
3. The basic agents still do not avoid them (their hindsight command marks only relief).

## Method

- **World:** the continual torus with 2 hazards, `env.h_rate=0.001`, `env.hazard_cost=0.02`. Random agents (64 worlds ×
  3,000 steps): hunger at the cap on 14% of steps (0.59 at the old rates with hazards 0.1), 82% of hazard steps raise
  hunger (28%), hunger at a hit 0.38 (0.74). Choice in task 00019.
- **Arms (seeds 1–3, GPU):** S8D (vm-00014's P8D: 8×8 patch tokens, move loss detached, 60k), S8DW (S8D +
  `ppo.drive_changed_weight=10`), SD (DMI, the CNN model, 60k), SL (the planner agent L, 40k).
- **Controls:** vm-00014 P8D, vm-00011 TH, vm-00012 LTH (h_rate 0.005).
- **Measures:** online hits, share of steps in a hazard, `hazard_felt`, `hunger_capped`; `analyze_hazards.py contact`
  and (SL) `hazard`; the frozen tests.

## Log

### 2026-10-25
- **Chose the rates** from random agents (task 00019).
- **Launched** the 12 lifetimes at 09:54, while vm-00014's 8×8 lifetimes were finishing.
- **First 1–3k steps:** most hazard steps now show in the drives (felt / in a hazard: S8D 0.057/0.095, 0.041/0.067,
  0.095/0.099; S8DW 0.061/0.063, 0.033/0.048, 0.054/0.054; SD 0.100/0.101, 0.031/0.073, 0.094/0.102; SL 0.050/0.050,
  0.066/0.069, 0.098/0.098). Basic-agent lifetimes that hit rarely still reach the cap (up to 0.68 of steps); the planner
  never does.
- **All 12 finished at about 10:30; analysis at 10:42.** With slow hunger the cap is rarely reached (online, last 10k
  steps: capped 0.00–0.19 of steps; felt / in a hazard 0.049/0.050 to 0.097/0.110).
- **Online, last 10k steps:** hits per 1,000 steps S8D 6.0, 8.0, 8.3; S8DW 4.9, 5.2, 7.1; SD 6.0, 8.3, 7.0; SL 17.0,
  17.8, 16.9 (vm-00012 LTH at the old rates: 25–28). In a hazard: S8D 0.074, 0.061, 0.110; S8DW 0.074, 0.059, 0.095;
  SD 0.081, 0.068, 0.079; SL 0.060, 0.050, 0.062 (0–5k: 0.054–0.099; chance about 0.063).
- **Frozen episodic test at 60k:** S8D 0.77, 0.81, 0.89; S8DW 0.63, 0.77, 0.71; SD 0.77, 0.88, 0.81 (birth 0.59–0.64).
- **Contact, the drives head inside − outside** (the world +0.020):

  | Arm | 20k | 40k | 60k |
  |---|---|---|---|
  | S8D | +0.005, 0.000, −0.001 | −0.004, −0.002, −0.001 | +0.004, +0.001, +0.002 |
  | S8DW | −0.005, −0.002, +0.002 | +0.007, −0.021, 0.000 | +0.005, +0.007, 0.000 |
  | SD (DMI, CNN) | −0.008, +0.001, +0.004 | 0.000, +0.006, +0.008 | **+0.019, +0.012, +0.016** |
  | SL (planner) | +0.041, +0.028, +0.033 | −0.005, +0.013, +0.022 | — |

  R² of contact: S8D/S8DW cursor's patch 0.59–0.72, drives token up to 0.54 (S8DW seed 2); SD from v 0.12–0.17 at 60k
  (0.01–0.06 at 20k–40k); SL from z 0.05–0.15.
- **The planner's imagined hunger** (12 still steps, inside − outside; world +0.183): 20k +0.020, +0.013, −0.009; 40k
  −0.012, −0.002, −0.010.
- **Reading:** with the trace present, the CNN agent's drives head learns contact by 60k (most of the world's +0.020),
  and the planner's Transformer drives head sees it early (20k) but not stably. The 8×8 patch models do not learn it
  at all, even though the cursor's patch token holds contact (R² ≈ 0.7) and the weight on unusual changes pushes the
  drives token's R² to 0.38–0.54. The planner's imagination (from v) stays blind. No arm avoids hazards.


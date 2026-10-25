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

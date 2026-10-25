# Task 00019: Slow Hunger, So That Hazard Contact Shows in the Drives (experiment vm-00015)

**Date:** 2026-10-25
**Status:** In Progress

## Context

- Hunger is clipped to [0, 1]. At the cap a hazard step changes nothing, so the drives carry no trace of it and no loss
  can learn it. Found in vm-00014: at birth (forage test) 5.9% of steps are in a hazard, but only 1.7% raised hunger,
  and hunger is at the cap on 55% of steps. This holds for every hazard experiment so far (vm-00011–14); least for the
  planner, which hits most often.
- Lowering the hunger rate alone does not fix it: the hazards themselves drive hunger to the cap. Random agents in the
  continual torus with 2 hazards (64 worlds × 3,000 steps):

  | h_rate | hazard cost | steps at the cap | hazard steps that raise hunger | hunger at a hit |
  |---|---|---|---|---|
  | 0.005 (so far) | 0.1 (vm-00013/14) | 0.59 | 28% | 0.74 |
  | 0.005 (so far) | 0.02 (vm-00011/12) | 0.45 | 50% | 0.67 |
  | 0.0005 | 0.1 | 0.43 | 43% | 0.57 |
  | 0.001 | 0.03 | 0.20 | 73% | 0.44 |
  | **0.001** | **0.02** | **0.14** | **82%** | **0.38** |
  | 0.001 | 0.01 | 0.07 | 91% | 0.31 |

- Chosen: `env.h_rate=0.001`, `env.hazard_cost=0.02`. A hazard step adds 21 times the steady rise, and a hit still
  relieves 0.38 on average.

## Experiment vm-00015

World: the continual torus with 2 hazards, `env.h_rate=0.001`, `env.hazard_cost=0.02`. Seeds 1–3, on the GPU.

| Arm | Agent | Steps |
|---|---|---|
| S8D | patch tokens 8×8, move loss detached (vm-00014 P8D) | 60k |
| S8DW | S8D + `ppo.drive_changed_weight=10` | 60k |
| SD | DMI, the CNN model (vm-00011 TH) | 60k |
| SL | the planner agent L (vm-00012 LTH) | 40k |

Controls at the old rates: vm-00014 P8D (0.005, 0.1), vm-00011 TH and vm-00012 LTH (0.005, 0.02).

## Questions

1. With the cap rarely reached, is contact learned: does the drives head predict the extra hunger inside a hazard
   (+0.020 against +0.001), and does the planner's hunger model imagine it?
2. Does the planner then avoid hazards (share of steps in a hazard below chance, about 0.063)?
3. Does the weight on unusual drive changes matter once the changes are visible?

## Measures

- Online: hits per 1,000 steps, the share of steps in a hazard, `hazard_felt`, `hunger_capped`.
- `analyze_hazards.py contact` (all arms) and `hazard` (SL) at 20k and the end; the frozen tests (basic agents).

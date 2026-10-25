# Task 00016: The Planner Agent in the Continual Torus World, With Hazards (experiment vm-00012)

**Date:** 2026-10-25
**Status:** Complete (results in vm-00012)

## Context

vm-00011 found that the basic agent starves in a continual world with walls (its drift sticks to a wall) and forages
on a torus. It has no reason to avoid hazards: its hindsight command marks the moves before relief, and nothing marks
the moves before a rise in hunger. The planner agent (vm-00005's L) picks moves by imagining 12 steps of hunger with its
learned latent dynamics, so if its hunger head learns that hazards raise hunger, it should steer around them. Task
00015 moved its lifetime to the GPU.

## Arms (L: `configs/L_gated.json`, memory U, `online.device=cuda`, 40k steps, seeds 1–3)

| Arm | World |
|---|---|
| LT | continual torus |
| LTH | + 2 hazards (as vm-00011 TH) |
| LTM | + moving circles (0.5 px/step) |
| LTE | + egocentric view |

## Questions

1. Does the planner agent forage in a continual world, and better than the basic agent (vm-00011 T, TM, TE, TH)?
2. Does it avoid hazards? The share of steps inside one, against random agents and the basic agent in the same world.
   Its imagined hunger should rise inside a hazard once the hunger head has seen it happen.
3. Moving circles and the egocentric view: do the latent dynamics keep up?

## Measures

- Online: hits per 1,000 steps, the share of steps in a hazard, the planner's share of steps and its imagined gain.
- `online.py --eval` does not apply (it scores the policy head, which the planner agent does not train). `plan.py`
  tests the frozen planner in episodic worlds only. Extend the forage test to the planner if time allows.

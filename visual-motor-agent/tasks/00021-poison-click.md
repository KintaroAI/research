# Task 00021: Poison Circles: Does the Prediction Get Both Kinds of Click Right? (experiment vm-00017)

**Date:** 2026-10-25
**Status:** Complete (results in vm-00017)

## Context

Hazard contact (vm-00011–16) costs 0.02 hunger per step: an error of about 4·10⁻⁴ in squared terms, beside a hit's
0.1–0.2. Only the CNN agent's drives head learned it, and only once slow hunger kept the cap out of the way (vm-00015).
The aim here is narrower than behavior: **check that the models predict what a click does**, with an effect as large
as a hit's.

## Change to the world (`env.hazard_click`, default −1 = off: worlds unchanged)

- A click with the cursor inside a hazard sets hunger to `env.hazard_click` (here 1.0): poison. A click inside the red
  circle relieves hunger as now (food). A click elsewhere does nothing.
- The hazard stays where it is (it drifts as before); only the red circle is eaten and respawns.
- Contact without a click costs nothing in this experiment (`env.hazard_cost=0`).
- If the cursor is inside both, the poison wins (hunger is set after the relief).
- `info["gt_hazard_click"]` (scoring only).

## Experiment vm-00017

World: the continual torus with 2 hazards, slow hunger (`env.h_rate=0.001`, so that hunger is mostly far from 1 and a
poison click is a large jump). Seeds 1–3, on the GPU:

| Arm | Model | Steps |
|---|---|---|
| PD | DMI, the CNN basic agent (vm-00015 SD) | 60k |
| PP | 8×8 patch tokens, move loss detached (vm-00015 S8D) | 60k |
| PL | the planner agent L (vm-00015 SL): its latent hunger model too | 40k |

## The prediction test (`analyze_hazards.py click`)

For placed states (true positions, scoring only) with hunger 0.3: the cursor in the red circle, in a hazard, or on the
background (away from both). From each state, the model's prediction of the next state for the move "stay and click"
and "stay, no click", against the world's (stepping copies of the world):

- the drives head's next hunger: food click about 0, poison click 1.0, others 0.301;
- the predicted frame: whether the red circle vanishes on a food click (its pixels' predicted redness against no
  click), and stays on a poison click;
- the planner's latent hunger head (one imagined step) for the same moves (PL).

At 0, 10k, 20k, 40k (and 60k for PD, PP). Plus online: food and poison clicks per 1,000 steps.

## Verification

- `tests/test_world_ext.py`: a poison click sets hunger to the value, a food click still relieves, the poison wins
  inside both, contact alone costs `hazard_cost`, default worlds unchanged (a fixed-seed trace).

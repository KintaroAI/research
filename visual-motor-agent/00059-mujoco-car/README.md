# vm-00059: A Small Car in MuJoCo

**Date:** 2026-10-31
**Status:** Complete
**Source:** `exp/vm-00059` (code: `visual-motor-agent/dev/`)
**Outputs:** `~/data/research/visual-motor-agent/exp_00059/`
**Task plan:** [`../tasks/00067-mujoco-car.md`](../tasks/00067-mujoco-car.md)

## Goal

A quick test: does the universal model live in a 3-D physics world, unchanged, and does its instinct work there?

## Method

- **`CarWorld`** (`world/car_world.py`, `configs/U_car.json`): a differential-drive car in MuJoCo 3.17.0, with wheels, contact and friction. Forward
  ~1 m/s, turning ~2.6 rad/s, 20 ticks/s of world time. One camera on its nose, 64 × 64, grey. Three solid food boxes and three checker poison boxes on
  a 5 × 5 m torus with no walls or seams, touched within 0.25 m and then replaced. Hunger as the default world: +0.001 a tick, food −1, poison +0.5.
- **The model:** `configs/U_default.json`, unchanged: four quadrant transformers, gain, age key, ring. Move value 0 turns, value 1 drives, the click does nothing.
- **Runs:** seeds 1–3, 20k ticks, on the 3090 with MPS: a quick test, a third of a usual life. **Baseline** (`car_chance.py`): blind drivers, 20k ticks.
- **Videos** (`car_video.py`): frozen final weights, 1,000 ticks, the camera and a view from above.

## Results

### R1. Food / poison per 1k ticks, by 5k ticks

| run | 0–5k | 5–10k | 10–15k | 15–20k | hunger at the end |
|---|---|---|---|---|---|
| s1 | 0.8 / 0.6 | 1.2 / 1.0 | 0.0 / 0.6 | 1.6 / 1.0 | 0.71 |
| s2 | 0.2 / 0.6 | 0.0 / 0.2 | 0.0 / 0.0 | 0.0 / 0.0 | 1.00 |
| s3 | 0.8 / 0.0 | 0.4 / 0.6 | 0.6 / 0.4 | 0.4 / 1.2 | 0.82 |

### R2. Blind drivers

| driver | food per 1k | poison per 1k |
|---|---|---|
| a new random move each tick | 0.2 | 0.2 |
| a random move held for 20 ticks | 1.5 | 1.2 |
| straight ahead | 0.1 | 0.1 |

### R3. The last 5k ticks

| run | patch CE | body CE | move CE | mean proposal, turn and drive |
|---|---|---|---|---|
| s1 | 0.420 | 0.114 | 0.212 | 0.65, 0.67 |
| s2 | 0.116 | 0.017 | 0.027 | 1.00, 1.00 |
| s3 | 0.523 | 0.115 | 0.347 | 0.58, 0.64 |

Speed: 29 ticks/s each, three together. The world alone runs ~2,000 ticks/s. Frozen, 1,000 ticks: 0, 1 and 2 food, no poison.

## Analysis

- **The plumbing works.** The model runs in MuJoCo with no change to it: frames, moves, hunger, memory, four pieces. Prediction falls, and the world costs little.
- **It does not forage.** Seeds 1 and 3 touch ~1 food per 1k ticks, the blind held driver's level, with no food : poison gain. This is not evidence that
  the instinct fails in 3-D, because of the next point.
- **Too little experience, likely.** Blind, a touch comes about once per 370 ticks at best (2.7 per 1k); a random walk in the 2-D touch world makes 16.5
  per 1k (vm-00035), about 6× as many, and QNA ends at 40 food per 1k. Twenty food touches in 20k ticks are little to learn what food does, the same limit as vm-00057. Not tested yet.
- **Seed 2 got stuck.** Every piece proposes full turn and full drive, hunger sits at the cap, and its low CE (0.116) probably means its view changes in
  a regular way (not measured). At the cap a poison leaves no trace (vm-00014) and food is rare, so nothing pulls it out.
- **Moves saturate:** proposals of 0.58–1.00 against the ~0.5 of a uniform draw. Whether this is the same feedback as max's in vm-00051 is not measured.

## Conclusions

1. The universal model runs unchanged in a MuJoCo car, at about the speed of the 2-D world.
2. It does not forage yet, at 20k ticks in this world. The world is probably too sparse to teach it: more objects and a larger touch distance come first.

# Task 00068: A Dense Start for the Car (experiment vm-00060)

**Date:** 2026-10-31
**Status:** Complete

## Context

- vm-00059: the unchanged model runs as a MuJoCo car but does not forage in 20k ticks: ~1 food per 1k, a blind held driver's level. Blind drivers
  touch at most 2.7 boxes per 1k, against 16.5 for a random walk in the 2-D touch world. The likely limit is experience.
- The owner's plan (2026-10-31): a really dense world first, thinned over time to today's density, 60k ticks. The touch distance stays as it is.

## Design

- **`env.dense = D`** (`CarConfig`, default 1: no schedule): a life starts with D × `foods` food and D × `poisons` poison boxes. Up to tick `dense_hold` the
  count stays. It then falls geometrically, the same share per tick, and reaches `foods` and `poisons` at tick `dense_end`, staying there.
- Boxes are removed by index, the highest first, and never return. A removed box is out of sight below the floor and cannot be touched. A touched
  box is still replaced at once. The food : poison balance stays one to one.
- `env.stats()` logs the counts (`car_foods`, `car_poisons`). The torus's side, the touch distance, the camera and the hunger do not change.
- `car_video.py` starts at the life's last tick, so a frozen car lives at the final density; `--world-tick` picks another. `car_chance.py` takes `--foods`
  and `--poisons`, for the blind level at any density.

## Experiment vm-00060

- **D:** `configs/U_car_dense.json`: `dense = 8` (48 boxes), `dense_hold = 10000`, `dense_end = 40000`: 48 boxes for 10k ticks, three halvings over 30k, 6 boxes
  for the last 20k. Seeds 1–3, 60k ticks.
- **C:** `configs/U_car.json`, 6 boxes all the way, seeds 1–3, 60k ticks: the control, added so that the effect of the dense start is not mixed with
  the effect of three times the ticks.
- D on one card, C on the other, with MPS.
- **Measure:** food and poison per 1k over the last 20k ticks, all at 6 boxes, against C and the blind drivers (1.5 food, 1.2 poison held); share of food; by 10k ticks; a video.
- **Reading:** D above C and above blind, with food above poison, means the dense start teaches something that lasts. D falling back to blind as the boxes
  thin means it learned only while food was everywhere.

## Verification

`tests/test_car_world.py`: the counts at given ticks; a removed box is neither drawn nor touched; a touched box is replaced while the count stands; no
schedule by default, with the same frames as before; invalid settings; a short lifetime with the schedule.

## A Bug Found on the Way

The first start of these runs was stopped at 36k and 20k ticks. Under moves that change sharply, as a model's do, the car of task 00067 flipped onto its back
and stayed there: 58–66% of ticks upside down in a blind test, the camera below the floor. Only smooth drivers stayed up. The tall chassis had its wheels at the
back and strong motors. The car now has a low ballast, its wheels under its centre with a skid at each end, and torque limits: at most 0.2° of tilt under the same
drivers, 0.9 m/s, 2.6 rad/s. `car_max_tilt` is logged and `test_harsh_driving_never_tips_it` keeps it so. Outputs of the stopped start: `exp_00060_flipped/`. The
blind drivers are rerun. vm-00059's results are from the flipping car.

## Results

Write-up: [`../00060-dense-car/README.md`](../00060-dense-car/README.md). At 6 boxes D makes 0.7 food and 0.47 poison per 1k against a blind held driver's 1.2 and 1.2; two of C's three seeds touched
nothing. The dense start gave experience, not skill. Next: the clock (task 00070).

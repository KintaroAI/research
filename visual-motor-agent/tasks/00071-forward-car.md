# Task 00071: A Car That Only Goes Forward (experiment vm-00062)

**Date:** 2026-10-31
**Status:** Complete

## Context

- The owner saw in the videos that the car more often than not moves backward, and so collects things it does not see. The car should only stop or go forward, at different speeds.
- This fits vm-00061's early numbers: 80–90 touches per 1k in the dense phase, well above a blind driver's 60, but with no food : poison preference. A box hit in reverse was never seen.

## Design

- **`env.reverse`** (`CarConfig`, default false): the drive value −1..1 is a speed from 0 to full: −1 stops, 0 is half speed, +1 is full speed. The whole range means something, so each value has its
  own effect, which a clamp at zero would not give. Turning is unchanged. `reverse = true` is the car of vm-00059..61. The model does not change.
- `car_chance.py --reverse`, `car_video.py --reverse` for the old car.

## Experiment vm-00062

- **FD:** `configs/U_car_dense.json`, **FC:** `configs/U_car.json`, both at `frame_skip = 40`, forward only. Seeds 1–3, 60k ticks; FD on the 4090, FC on the 3090, with MPS.
- **Control:** vm-00061's SD and SC: the same but for reverse. **Baseline:** `car_chance.py --frame-skip 40`.
- **Measure:** food and poison per 1k over the last 20k ticks at 6 boxes; share of food; ticks at capped hunger; a video.
- **Reading:** a share of food above 0.5 that the reversing car lacks means the camera now sees what it touches, and the instinct can choose.

## Verification

`tests/test_car_world.py`: −1 stops, a middle value is half speed, no drive value goes backward, and `reverse` keeps the old car.

## Results

Write-up: [`../00062-forward-car/README.md`](../00062-forward-car/README.md). Forward only stops the starving: FC 6.6 food, 5.5 poison per 1k against SC's 0.9, 0.7, without a dense start, which no longer helps. Food share 0.530 ± 0.014, faint and not proven; a blind forward driver makes 6.2 and 6.2.

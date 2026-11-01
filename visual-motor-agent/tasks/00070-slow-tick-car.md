# Task 00070: A Slower Tick for the Car (experiment vm-00061)

**Date:** 2026-10-31
**Status:** Complete

## Context

- vm-00060: with the fixed car, a dense start did not teach foraging: D stays near a blind driver's level while boxes are dense, and at 6 boxes it is below it.
- A new guess: the clock. The instinct looks one tick ahead, and at 20 ticks/s the car goes ~4.5 cm a tick, so a move pays only in the last tick before a touch, though a box
  can be seen from metres away. At 5 ticks/s one tick is ~18 cm, about the touch distance. The owner chose this test (2026-10-31), before a longer instinct horizon.

## Design

- `env.frame_skip = 40` instead of 10: 200 ms a tick. No other change, and no code change: the touch distance, boxes, camera, hunger per tick and the model stay the same.
- A touch is tested at the end of a tick. A tick covers less than the touch distance, so a box in the path is not skipped.

## Experiment vm-00061

- **SD:** `configs/U_car_dense.json` with `frame_skip = 40`: the same schedule by tick. **SC:** `configs/U_car.json` with it, 6 boxes all the way. Seeds 1–3, 60k ticks. SD on the
  4090, SC on the 3090, with MPS.
- **Baseline:** `car_chance.py --frame-skip 40` at 48, 24, 12 and 6 boxes.
- **Measure:** food and poison per 1k over the last 20k ticks, at 6 boxes, against vm-00060's D and C and the new blind drivers; share of food; ticks at capped hunger; a video.
- **Reading:** food above blind and above poison means the guess holds. No change sends the question to the instinct's horizon, or to what one forward camera can show.

## Verification

`tests/test_car_world.py`: at 5 ticks/s a full-drive tick is over 15 cm and under the touch distance, and harsh driving stays upright.

## Results

Write-up: [`../00061-slow-tick-car/README.md`](../00061-slow-tick-car/README.md). At 5 ticks/s the car seeks boxes (nearly 3x blind at 48 boxes) but shows no preference for food; at 6 boxes SD makes 4.3 food, 4.2 poison, against blind 3.9, 3.5.

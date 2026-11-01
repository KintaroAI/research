# Task 00067: A Small Car in MuJoCo (experiment vm-00059)

**Date:** 2026-10-31
**Status:** Complete

## Context

- The owner wants the model in a 3-D physics world, starting with a quick test. MuJoCo 3.17.0 runs here (a no-AVX source build, see
  `../CLAUDE.md`): 123k physics steps/s on one core, 4,260 frames/s at 64 × 64 through EGL on the 3090.
- The car comes first because it needs no model change: two controls fit the 3-value move token, and an onboard camera is the egocentric view
  of the default world.

## Design

- **`world/car_world.py`** (`env.kind = car`, `CarConfig`, `CarWorld`), CircleWorld's interface, one `MjData` per body:
  - A differential-drive car with real wheels, contact and friction: two driven wheels and a low-friction caster. Velocity actuators on two
    tendons, forward and turn. Move value 0 is turn, value 1 is forward; the click is ignored. 10 physics steps of 5 ms per tick (20 Hz).
  - The camera sits on the car's nose, looking forward and slightly down, 64 × 64, converted to grey. A dim floor grid gives optic flow.
  - Food: solid light boxes. Poison: black and white checker boxes. They are visibly different, even far away: this test asks if the model works
    in 3-D, not if it can tell hard patterns apart.
  - Objects live on a torus around the car: each is drawn at its nearest image, so the world has no walls or seams. Boxes do not collide; the car
    touches one within `touch` metres of it (as `env.touch`), and it is replaced at least `spawn_min` from the car and other objects.
  - Hunger as in the default world: `h_rate` 0.001 per tick, food −1.0, poison +0.5, clipped to 0..1.
- **The model:** unchanged, `configs/U_car.json` is `U_default.json` with this world.
- **`car_chance.py`:** the same world driven blind, for what chance gives: random moves and straight ahead.
- **`car_video.py`:** a frozen run's body: what the camera sees, and a view from above.

## Verification

`tests/test_car_world.py`: frame shape and grey; seeded worlds repeat; forward moves the car forward, turn turns it, and no move stays put;
touching food lowers hunger and poison raises it, and each is replaced; the torus keeps objects within reach; the model runs a short lifetime
through `lifetime.py`; invalid configs. Without `env.kind = car`, nothing else changes, and mujoco is imported only by this world.

## Experiment vm-00059

- `configs/U_car.json`, seeds 1–3, 20k ticks, on the RTX 3090 with MPS: a quick test, not a full lifetime.
- **Measure:** food and poison per 1k over the last 5k ticks against `car_chance.py`; patch, body and move CE; ticks/s; a video.
- **Reading:** food above chance with poison at or below it means the instinct works in 3-D, from one camera, with no change to the model.

## Results

Write-up: [`../00059-mujoco-car/README.md`](../00059-mujoco-car/README.md). The model runs unchanged, at 29 ticks/s each (three together; the world alone ~2,000). It does not
forage at 20k ticks: ~1 food per 1k, a blind held driver's level (1.5 food, 1.2 poison), and one seed stuck at full moves with hunger capped. Too little experience is the likely cause, not tested.

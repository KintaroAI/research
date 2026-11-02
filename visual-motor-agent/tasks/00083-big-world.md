# Task 00083: The Big World, the Same Experiment at Twice the Side (experiment vm-00072)

**Date:** 2026-11-02
**Status:** In progress (restarted 2026-11-02 after task 00084, with log summaries, K1 and K4, the owner's decision)

## Context

- The owner (2026-11-02): "lets make world larger 2x the width and the height but keep food and poison same size in pixels - since it will occupy less space - add 4x more of each and re-run same experimet as last one but on this larger world".
- vm-00071 is the experiment to repeat: `online.future_k` at 1, 4 and 16, six seeds, 60k ticks. K4 won in the 64 x 64 world: food 39.0 to 67.8, no more poison.
- The question it adds: does the wider reach of K4 hold when there is four times as much to look at, and what does a larger view cost?

## Design

**The world.** `env.size = 128`. Food, poison, dots, the cursor, the move and hunger's rate keep their sizes in pixels and per tick. Four times as many of each thing, so the density is the same:

| | 64 x 64 | 128 x 128 |
|---|---|---|
| Food circles (radius 8–11) | 1 | 4 |
| Poison circles (radius 5–8) | 2 | 8 |
| Dots | 8 | 32 |

The owner named food and poison. Dots are scaled too (my decision), since otherwise the big world would be emptier of noise than the small one, and that would be a second change.

**Several foods.** The world has held one food since the start. `env.foods = F` adds F - 1 still circles: `fx_pos`, `fx_rad`, with their own generator, so that `foods = 1` is the world as it was, draw for draw. Slot 0, `target`, is always the food nearest the cursor by gap, so that the step's hit test, relief and respawn, `gt_target` and the frozen tools all keep working on "the food". A swap into slot 0 before the hit test and after a respawn is all it takes; circles do not overlap, so the cursor is inside at most one. It needs wrap, touch-style respawn and still food.

**The model: the same four quarters, each reading every summary.** Patch 8, so 16 x 16 patches, `uni_pyr = 4` so that the pyramid still ends in one summary, 85 summaries in all. A quarter holds 64 patches, 85 summaries, the body and the move: 151 tokens per step against 39. Same `d_model`, layers, heads, `ctx`, bins, instinct, tolerance, and about the same parameters: 984,738 per piece against 970,402.

Timed alone on the 4090, 300 ticks, without MPS:

| shape | ticks/s | peak GPU memory | tokens per step |
|---|---|---|---|
| **Quarters, every summary (chosen)** | 18.2 | 1.8 GB | 151 |
| Quarters, `band_win = 1` | 27.5 | 1.3 GB | 105 |
| 16 pieces, `band_win = 1` | 17.3 | 1.5 GB | 48 |

The others are cheaper or no faster, but each is a second change against vm-00071, and the question here is the world. The default world runs at 76 ticks/s alone, so about 4 times slower.

**The restart (2026-11-02).** After vm-00073 the owner chose log summaries, the new default, and K1 and K4 only, since K16 failed on poison twice. The first start, with every summary and K16, was stopped at about 3,000 and 1,000 ticks and nothing was read from it; its lives are in `exp_00072/stopped_all_summaries/`. So a quarter holds 64 patches, 39 of the 85 summaries, the body and the move: 105 tokens per step, and the comparison in the world is K4 against K1, with vm-00073 beside it for the small world. The table above stands for what was timed then.

**Same readings as vm-00071,** K4 against K1 of this world. Aim uses the same gaps in pixels, and the density is the same, so a move's reach means the same thing.

## The readings, set before the results

Each is an arm against K1, seed by seed, 95% interval.

1. **Plumbing:** with `foods = 1` the world is the old one, draw for draw; with `foods = 4`, every food is drawn, touching any of them feeds, and no circles overlap. **Met:** the default world matches the code before this task in every frame and drive over 5,000 ticks of 4 worlds, and `tests/test_many_foods.py` passes. A touched food is replaced alone.
2. **Reach holds:** aim at gaps of 12–24 pixels, weighted by ticks, is above K1's by more than its interval.
3. **A chain pays:** the lead over blind and a perfect grab is above K1's by more than its interval.
4. **Food:** over the last 20k ticks food is above K1's by more than its interval, with poison not above K1's by more than its interval.

Beside them, mean hunger (vm-00071's section on what poison costs): the score the agent is minimizing.

## Implementation

- [x] `world/circle_world.py`: `foods`, `fx_pos`, `fx_rad`; placement, overlaps, respawn, drawing, state
- [x] `tests/test_many_foods.py`
- [x] `eval_hunger.py`: `would_touch` sees every food
- [x] `configs/U_big.json`
- [x] Benchmark, the model decided
- [x] `00072-big-world/`: README, Makefile
- [ ] Runs, frozen tests, write-up

## Verification

Tests of the world, the old draw-for-draw check, and a short life on each GPU.

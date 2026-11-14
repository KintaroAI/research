# Task 00102: A World Larger Than the View (experiment vm-00091)

**Date:** 2026-11-13
**Status:** Complete (2026-11-13; write-up: `../00091-world-larger-than-view/README.md`)

## Context

- **vm-00090 (task 00101):** in the 128 × 128 world the value pays for the first time. VB eats 79.2 food per 1k against KB's 56.4 (+22.9 ± 10.6, 6 of 6 seeds), by aiming at food more than 24 px away. But there the agent always sees the whole world: the view is the world, centered on the cursor.
- **The owner (2026-11-12, 2026-11-13):** after the bigger world (A), a world larger than the view (C), with a 64 px view.
- **What a 64 px window hides** (scripted walkers, scratchpad `c91/`; a walker turns to the nearest food it can see, and otherwise goes straight on a heading it redraws about every 32 ticks; 32 bodies, 6,000 ticks for 4 foods, 3,500 ticks for 1 and 2):

| World 128 × 128 | View | Ticks with no food in view | Seeing walker, food per 1k |
|---|---|---|---|
| 4 foods (vm-00090's) | 128 | 0 | 170.6 |
| 4 foods | 64 | .12 | 157.7 |
| 2 foods | 128 | 0 | 136.8 |
| **2 foods** | **64** | **.37** | **99.9** |
| 1 food | 128 | 0 | 112.8 |
| 1 food | 64 | .62 | 55.9 |

  - With vm-00090's 4 foods, a 64 px window rarely hides all of them.
  - **With the tool** (`eval_food_gap.py`, `U_view.json`, the pixel test of `foods_in_view`; `exp_00091/walkers_view*.txt`):
    - at view 64: blind 9.9, reach 16 px 34.6, reach 24 px 54.4, chasing all it sees 96.2, with no food in view on .39 of its ticks;
    - with the whole world in view, chasing always: 136.7.
  - **The owner chose 2 foods** (2026-11-13): food is out of sight 37% of ticks, and a walker that only chases what it sees loses 27%.
- **The model's memory is short.** A transformer's context is 8 ticks (`model.ctx`). A food seen more than 8 ticks ago is no longer in its input. To head for it after that, the model would have to carry where it is in its weights or in its moves. vm-00081/83 (hidden food) found no sign of memory beyond the model's own baseline aim.

## Design

| Arm | Label | Model | World | Seeds |
|---|---|---|---|---|
| **KW** | K4 | 4 quarter transformers × 8 layers, `instinct_tol` 0.002 (vm-00090's KB) | 128 × 128, 2 foods, 8 poisons, 32 dots, view 64 | 1–12 |
| **VW** | value, γ 0.9, n 4, fresh tail | the same (vm-00090's VB) | the same | 1–12 |

- **The world** (`configs/U_view.json`): `U_big.json` with `env.foods = 2` and `env.view = 64`.
  - The frame is the 64 × 64 center of the full egocentric frame, so the model is the small world's: `img` 64, `uni_pyr` 3. Each quarter transformer covers 32 × 32 px, as in vm-00089.
- **The labels and the model are vm-00090's:** `--set online.future_k=4 --set model.n_layer=8 --set online.instinct_tol=0.002`, plus `--set online.value_gamma=0.9 --set online.value_refresh=1` for VW.
- **No full-view control at 2 foods.**
  - The question is whether the agent heads for food it cannot see. Its aim then is compared with the same agent's moves scored against another world's food (the blind baseline), which needs no second arm.
  - The walkers give the ceilings. vm-00090 is the reference for the value at full view in the 4-food world.
- **Each seed's two lives on one card,** under MPS: seeds 1–7 on the RTX 4090, 8–12 on the RTX 3090.
  - **Timing** (1,000-tick lives, 12 per card, 2026-11-13 11:47–11:52; scratchpad `t91/`): on the 4090 KW 5.7 ticks/s and VW 4.8, about 63 for the card; on the 3090 KW 3.9 and VW 3.3, about 43.
  - With 7 and 5 seeds, both cards should take about 4 h. With 6 and 6, the 3090 would take 5 h.

## Code

1. **`env.view`** (`world/circle_world.py`): the side in px of a window centered on the cursor; 0, the default, is the whole world.
   - Needs `egocentric`. It must be even, at most `size`, and `(size − view) / 2` a multiple of `2 · hazard_checker_cell`, so the poison's checker keeps its phase.
   - The frame is the center crop of the full egocentric frame. Noise, if any, is drawn on the window.
   - Food, poison, touches and hunger are unchanged: only the frame changes.
   - `CircleWorld.foods_in_view()`: (N, F) bool, which foods have a pixel inside the window. For scoring only.
2. **`eval_hunger.py chain`,** with `env.view` set:
   - the share of ticks with no food in view;
   - aim at the nearest food on those ticks, by ticks since any food was last in view (1–8, within the context; 9–32; 33+), seen and blind. The blind world is scored on its own out-of-view ticks.
   - `show` prints them, and `versus` pairs them.
3. **`eval_food_gap.py --view V`:** the walkers chase only food in the window.
4. **`configs/U_view.json`.**
5. **Tests** (`tests/test_view.py`):
   - with `view` the frame equals the center of the full frame, over random states, with and without noise off;
   - `view` 0 gives the old frame, bit for bit;
   - the validation errors;
   - `foods_in_view` agrees with the food's pixels in the frame.
6. **Video tools** (`forage_video.py`, `compare_video.py`): the frame's own size for the view, the trail and rings at the window's scale. Done before the videos, not before the runs.

## Measures

- **Lived, 40–60k:** food, poison, mean hunger, overrides, move CE, smoothness; VW against KW, paired by seed.
- **Frozen `chain`** (1,500 ticks):
  - food; the share of ticks with no food in view;
  - aim at unseen food by ticks since the last sight, seen against blind;
  - aim by the gap, as before.
- **Frozen `value` and `reach`,** as before.
- **Walkers in `U_view.json`:** blind, and the seeing walker, `eval_food_gap.py --view 64`.

## The readings, set before the results

1. **Plumbing:** the view tests pass. All 24 lives run 60k ticks with 64 × 64 frames, `env.view` 64, `env.size` 128, 2 foods, without errors.
2. **The window hides food:** KW's share of frozen ticks with no food in view is at least .25 (the walker's .37).
3. **Short memory (the main one):** 1–8 ticks after the last sight of food, VW's aim at the unseen nearest food is above the blind baseline's, paired by seed over 12 seeds, by more than its interval.
4. **Memory past the context:** the same at 9–32 ticks. Expected to fail: the context is 8 ticks.
5. **The value in the window:** VW's lived food at 40–60k is above KW's, paired by seed, by more than its interval.
6. **The value holds:** every VW seed meets the bar (abs bias at most .0141, correlation above .5, median candidate spread at least .0005).
7. **Poison is still seen:** VW's lived poison is not above KW's by more than its interval.

**What follows:**
- **If 3 holds and 4 fails:** the agent remembers what it saw while it is in its context, and no longer. Next: a memory that outlasts the context (a longer context, or a summary carried across windows), tested in this world.
- **If 3 and 4 both hold:** memory beyond the context, through the weights or the gait. Look at how, before building anything.
- **If 3 fails:** the agent only reacts to what it sees. Check that the out-of-view ticks are not dominated by ticks right after a meal, when no food is near. Then the next step is the same as when 3 holds and 4 fails, but with the need shown more plainly.
- **5 decides the label for what comes after:** if VW eats no more than KW here, the value's gain in vm-00090 was about seeing far, not about planning, and K4 stays the cheaper label.

## Results (2026-11-13)

- Lived food 40–60k: KW 21.2, VW 47.6; VW − KW +26.4 ± 5.9 (12 of 12), poison −2.82 ± 0.66 (VW lower in 12 of 12). VW still rising (50.5 at 50–60k).
- Frozen: no food in view KW .525, VW .423; VW − KW food +33.0 ± 5.0, aim 12–24 px +.267 ± .041, lead over grab +26.2 ± 4.0.
- Aim at unseen food, seen − blind:
  - all out-of-view ticks, 1–8: VW +.138 ± .022 (12 of 12), KW +.058 ± .016 (12 of 12); 9–32: VW −.036 (1 of 12), KW −.017 (3 of 12).
  - without ticks after a meal (VW: 54% of the 1–8 ticks follow a meal; KW: 25%): 1–8 VW +.062 ± .017 (12 of 12), KW +.035 ± .013 (11 of 12); 9–32 VW −.022 (1 of 12), KW −.020 (3 of 12).
- Value bar: corr .76–.88, spread .0010–.0047; bias over .0141 in seeds 1, 4, 6, 8 (worst −.024; about 30% pessimistic).
- Readings: 1, 2, 3 (also without meals, at half the size), 5, 7 met; 4 fails as expected; 6 fails on bias.

## Implementation

- [x] `env.view`, `foods_in_view`, validation; `tests/test_view.py` (14 tests); the world's old tests (143 passed).
- [x] `eval_hunger.py chain` and `show` with the view's measures (checked on a 1,000-tick life); `eval_food_gap.py --view`.
- [x] `configs/U_view.json`; walkers in it.
- [x] Timing; `00091-world-larger-than-view/` (README, Makefile); the runs (2026-11-13 11:56–15:56; frozen tests to 16:23).
- [x] After the results, the check named under "If 3 fails", run whatever the outcome: `chain` also scores the out-of-view ticks with no food eaten since food was last in view (`cos_out_nomeal`), since a tick right after a meal has no food in view only because it was eaten. Rerun as `chain2_*.jsonl`.
- [x] Frozen tests, write-up; the video tools (the whole world beside the view, the trail clipped to the window) and the videos (`videos/vm91_s5`, `vm91_s12`).

## Verification

- `tests/test_view.py` and the full `make pytest` pass.
- Every run's `info.json` has `env.view` 64, `env.size` 128, `env.foods` 2, `model.img` 64, `n_layer` 8, `instinct_tol` 0.002.
- The first frames of a run, saved as PNG, show a 64 × 64 window with the cursor at its center.

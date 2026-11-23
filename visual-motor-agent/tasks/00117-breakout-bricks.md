# Task 00117: Breakout Stage 2 — a Faster Ball and Bricks Worth Different Amounts (experiment vm-00104)

**Date:** 2026-11-22
**Status:** Complete (vm-00104, 2026-11-23: both arms return slightly more balls than a random paddle, BK 28% and BS 33% at 50–60k against 23%; BS +.036 over BK; no preference for better bricks in either arm, lived or frozen). The owner, 2026-11-22, after vm-00103: "lets have a faster ball first, keep paddle as is plus bricks and make bricks visibly distinct and they should source food in differenct amouns depending on how brick look so we can estimate if it can target better blocks first. if it completely ate everything all blocks appear back but do not place same type of blocks in the same slots - shuffle it so we know it's not just memory.")

## Context

- **vm-00103:** the default agent, unchanged, learned stage 1, a court with no bricks where a paddle hit was a meal. Its hit share went from a random paddle's .24 to .68 lived (.64 frozen) in 60k ticks, all 12 seeds.
- **Stage 2, the owner's choices:**
  - the ball 1.5 times as fast (2.25 px a tick), the paddle as it is;
  - bricks, visibly distinct by type, each type feeding a different amount;
  - when all the bricks are gone they come back, with no slot keeping its type, so a preference must come from how a brick looks, not from where it was;
  - **a paddle hit is no longer food; a miss is still a poison.** All food comes from bricks.
- **The catch:** the default meal label scores any fall in hunger as the same meal, V = 1 (`online.meal_parts`). The unchanged agent cannot value a big brick over a small one through its label.
- **So two arms (the owner's choice):**
  - **BK:** the default agent unchanged. It tests whether a preference appears anyway (expected: no).
  - **BS:** one new learning option, `online.meal_size = M`: a meal is worth its size, V = min(1, relief / M), M the largest brick's food. Off by default (0), so every earlier run and config is unchanged.

## The world (`world/breakout_world.py`, new settings, defaults = stage 1)

- **`ball_speed` 2.25** in the stage-2 config (stage 1's 1.5 stays the default). The tracker must still catch every ball (tested).
- **`paddle_relief`** (default 1.0, stage 1): the food a paddle hit gives. Stage 2 sets 0: a paddle hit is no meal.
- **Bricks** (`bricks = 1`; 0 = stage 1):
  - 4 rows of 6 bricks, each 9 × 3 px with 1 px gaps, the top row at y = 10 (a gap above them, so a ball can get behind the wall);
  - **3 types, 8 bricks each**, with a different look in grey:
    - **solid:** flat grey 120;
    - **checker:** 1 px checkerboard of 230 and 60;
    - **hollow:** a 230 frame around the background;
  - **the food by type** (`brick_food`, .1, .25, .5 in hunger);
  - **which look feeds how much rotates with the seed** (seed mod 3): a preference for a look is then separable from a preference for food over the 12 seeds;
  - **the layout:** random at the start; when the last brick is gone, a new full layout where no slot keeps its type (a random derangement of the types, 8 of each again);
  - **a ball meeting a brick** removes it and bounces: off its top or bottom if it was inside the brick's columns before the move, else off its side. If it meets two bricks on one tick it removes both and bounces once.
- **The serve** (after a miss) comes from just below the lowest brick row when there are bricks, falling within 45° of vertical as before.
- **The body:** a brick hit relieves its food; the floor at 0 may cut it. **`hunger_cut` becomes signed:** the clip's correction, unclipped minus clipped hunger, positive at the cap (as before) and negative at the floor. The earlier label code adds it only on steps whose hunger did not fall, where it is never negative, so the default agent's labels are unchanged.
- **`success`** is a tick that fed (a brick hit in stage 2, a paddle hit in stage 1), so `step_hits` stays meals per 1k.
- **Stats** (cumulative, into metrics): `bo_hits` (paddle hits), `bo_misses`, `bo_bricks`, `bo_brick_t0..2` (hits by food rank: small, middle, big), `bo_food` (food given, before the floor), `bo_clears`, and the preference sums `bo_pref_hit` and `bo_pref_stand` (below).
- **Ground truth for scoring:** `gt_bricks` (each slot's type, −1 gone), `gt_brick_hit` (the food rank hit this tick, −1 none).

## The sized meal (`online.meal_size = M`, BS only)

- Only with `label = meal`. A step whose full change in hunger, rise + cut, is negative is a meal, worth V = min(1, (−rise − cut) / M); with no tail, as now. Any other step is as before.
- In `meal_parts` (re-labeled every replay, the default agent's path), `stored_labels` (stored once, still monitored) and `MealMonitor`.
- Off (0): exactly today's code path.
- **BS sets M = .5**, the big brick's food: a big brick is worth 1, a middle one .5, a small one .2.
- `tests/test_meal_size.py`: off equals today's labels exactly; sized meals by relief, with the floor's cut counted.

## The run: vm-00104

- **Configs:** `configs/U_breakout2.json` (`U_breakout.json` with the stage-2 world); BS adds `--set online.meal_size=0.5`.
- **Arms:** BK and BS, 12 seeds each, 60k ticks. Seeds 1–6 of both arms on the RTX 3090, 7–12 on the RTX 4090: 12 lives per card under MPS (timed on a short pilot first).
- **Walkers** (`breakout_walkers.py`, the stage-2 world): random, still, tracker.

## Measures

- **Play:** paddle hit share (paddle hits over paddle hits and misses), lived by 10k and frozen at 10k to 60k.
- **Food:** bricks and food per 1k ticks, mean hunger, clears.
- **The preference for better bricks:** at each brick hit, the food rank of the brick hit (0, 1, 2) minus the mean rank of the bricks standing at that moment; its mean over hits (`bo_pref_hit` − `bo_pref_stand`, over `bo_bricks`).
  - Standing bricks are not all reachable: the lower rows are hit first. The layout is random and reshuffled, so position balances out over types, and the tracker (which aims at nothing) gives the chance level in the same courts.
  - Also the share of hits by rank (small, middle, big) against the tracker's.

## The readings, set before the results

1. **Plumbing:** the world's tests pass (bricks, bounces, the derangement on refill, signed cut, the tracker never misses); the meal-size tests pass; all 24 lives run 60k ticks.
2. **It still plays:** BK's and BS's paddle hit share at 40–60k above the random walker's, paired by seed.
3. **It targets better bricks:** BS's preference at 40–60k (lived) and at 60k (frozen) above the tracker's, paired by seed.
4. **The unchanged agent (descriptive, expected near the tracker):** BK's preference against the tracker's.
5. **The sized meal's effect:** BS − BK, preference and food per 1k, paired by seed.

## Implementation

- [x] `BreakoutConfig`: `paddle_relief`, `bricks`, the brick layout and looks, `brick_food`; bricks in `_move_ball`, `_frames`, refill with a derangement, the serve below the bricks, signed `hunger_cut`, the new stats and gt keys. Stage 1's defaults unchanged.
- [x] `tests/test_breakout_world.py`: stage 1 still exact (its 10 tests); new tests for stage 2.
- [x] `online.meal_size` in `meal_parts`, `stored_labels`, `MealMonitor`, `GatedMemory` and `lifetime.py`; `DEFAULT_ONLINE`; refused without `label = meal`; `tests/test_meal_size.py`.
- [x] `configs/U_breakout2.json`; `breakout_walkers.py` reports bricks, food and the preference.
- [x] A pilot (timing, 24 lives); `00104-breakout-bricks/` (README, Makefile); the runs, frozen tests, video, write-up.
- [x] `readme_images.py`: stage-2 panels (paddle hit share, food per 1k).

## Verification

- **Information hygiene:** the model gets the frame, the body and its previous move only; `gt_*` and the stats are for scoring.
- **Stage 1 reproduces:** with the defaults, frames and events are identical to vm-00103's world (the old tests, unchanged).
- **The default agent unchanged in BK:** its resolved config equals `U_agent.json`'s `model`, `online` and `ppo` (tested as before); BS differs only in `meal_size`.

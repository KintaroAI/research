# Task 00119: Breakout Stage 2b — Food for the Paddle Hit Again (experiment vm-00107)

**Date:** 2026-11-23
**Status:** Planned (queued after vm-00106). The owner, 2026-11-23, after watching vm-00104's video: "it feels like we need to feed it on hittin paddle - movement feels random"

## Context

- **vm-00104 (stage 2):** a ball at 2.25 px a tick, and food only from 24 bricks of 3 looks (.1, .25, .5), with the paddle hit no meal.
  - Both arms barely play: they return 28% (BK) and 33% (BS) of balls at 50–60k, against a random paddle's 23%.
  - Neither prefers the better bricks.
  - The likely cause: food comes 15 or more ticks after the return, past the meal label's horizon of about 10 ticks.
- **vm-00103 (stage 1):** with a paddle hit as a meal, the same agent learned to return .68 of balls.
- **The fix (the owner):** feed the paddle hit again, so play is learned as in stage 1. The bricks then become a choice on top of it.

## The change

- **`env.paddle_relief = .25`:** a paddle hit relieves .25 of hunger, as much as a middle brick. Everything else is vm-00104's court (`configs/U_breakout2.json`), and the setting already exists (no new code; `configs/U_breakout2p.json`).
- **Why .25:**
  - BK's label scores any meal V = 1, so for BK the size of the paddle food changes nothing in the label.
  - In BS (`meal_size = .5`) a paddle hit is worth .5, a big brick 1, a small brick .2. Play is clearly paid, and the big brick is still worth twice a return.
- **The arms**, 12 seeds each, 60k ticks:
  - **PK:** vm-00104's BK with the paddle food;
  - **PS:** vm-00104's BS with the paddle food.
  - Cards and seeds as in vm-00104.
- **Walkers:** rerun in the new court (their food per 1k changes; their hit share and preference do not).

## The readings, set before the results

1. **It plays again:** PK's and PS's paddle hit share at 40–60k above the random walker's, paired by seed; and above vm-00104's BK and BS (descriptive: different courts' food, the same ball).
2. **It targets better bricks:** PS's preference (the hit's food rank minus the standing bricks' mean rank) at 40–60k lived and at 60k frozen, above the aimer's, paired by seed.
3. **The unchanged agent:** PK's preference against the aimer's (expected near chance: its label cannot tell sizes apart).
4. **The sized meal's effect:** PS − PK, preference, paddle share and food per 1k, paired by seed.

## Implementation

- [x] `00107-breakout-paddle-food/` (README, Makefile: vm-00104's with `configs/U_breakout2p.json`, the stage-2 court with `env.paddle_relief = .25`); walkers run.
- [ ] The runs (queued after vm-00106), walkers, frozen tests, video, write-up (analysis: vm-00104's script with the new arm names).

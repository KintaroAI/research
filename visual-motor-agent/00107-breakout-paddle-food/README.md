# vm-00107: Breakout with Bricks and Food for the Paddle Hit

**Date:** 2026-11-23
**Status:** Planned (queued after vm-00106; starts automatically when its frozen tests and video are done)
**Source:** `exp/vm-00107` (code: `visual-motor-agent/dev/`)
**Outputs:** `~/data/research/visual-motor-agent/exp_00107/`
**Task plan:** [`../tasks/00119-breakout-paddle-food.md`](../tasks/00119-breakout-paddle-food.md)

## Goal

The owner, 2026-11-23, after watching vm-00104's video: "it feels like we need to feed it on hittin paddle - movement feels random".

- **vm-00104:** with food only from bricks, the default agent barely played. It returned 28% of balls at 50–60k (BK; 33% with a meal worth its size, BS), against a random paddle's 23%, and preferred no brick.
  - The likely cause: food came 15 or more ticks after the return, past the meal label's horizon of about 10 ticks.
- **vm-00103 (stage 1):** a paddle hit was a meal, and the same agent learned to return .68 of balls.
- **Here:** vm-00104's court with a paddle hit feeding again, .25 of hunger (as much as a middle brick). Play should be learned as in stage 1, and the bricks become a choice on top of it.

## Method

- **The court:** `configs/U_breakout2p.json`, vm-00104's `U_breakout2.json` with `env.paddle_relief = .25`.
  - A miss is still a poison.
  - The bricks feed .1, .25 and .5 by type. Which look feeds how much rotates with the seed (`env.brick_shift` = seed mod 3).
  - The wall comes back with no slot keeping its type.
- **The arms**, 12 seeds each, 60k ticks:
  - **PK:** the default agent unchanged. Its label scores any meal as 1, so a paddle hit counts as a full meal.
  - **PS:** the same with `online.meal_size = .5`: a paddle hit is worth .5, and a small, middle or big brick .2, .5 or 1.
- **The cards and tests:** as in vm-00104.
  - Seeds 1–4 run on the RTX 3090 and 5–12 on the RTX 4090, under MPS.
  - Frozen tests at 10k–60k (2,000 ticks, seed 100 + s).
  - The walkers (random, still, tracker, aimer) run in this court.
  - The video shows PK and PS, seed 1, at 60k.
- **No new code:** `paddle_relief` exists since vm-00104.

### The walkers in this court (20k ticks, 12 seeds)

| Walker | Paddle hit share | Bricks per 1k | Food per 1k | Preference |
|---|---|---|---|---|
| random | .229 | 11.1 | 4.82 | +.005 |
| still | .265 | 12.5 | 5.45 | +.021 |
| tracker | 1.000 | 0.0 | 5.03 | −.070 |
| aimer | 1.000 | 24.6 | 12.06 | −.014 |

The tracker returns every ball straight up one emptied column, so its preference over its few bricks means little. The aimer is the chance level of the preference.

## The readings, set before the results

1. **It plays again:** PK's and PS's paddle hit share at 40–60k is above the random walker's, paired by seed. It's also compared with vm-00104's BK and BS (descriptive).
2. **It targets better bricks:** PS's preference at 40–60k lived and at 60k frozen is above the aimer's, paired by seed.
3. **The unchanged agent:** PK's preference is compared with the aimer's. It's expected near chance, since its label can't tell sizes apart.
4. **The sized meal's effect:** PS − PK in preference, paddle hit share and food per 1k, paired by seed.

## Results

(pending)

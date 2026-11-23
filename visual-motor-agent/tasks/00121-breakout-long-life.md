# Task 00121: Breakout Stage 2b — One Long Life of 1M Ticks (experiment vm-00109)

**Date:** 2026-11-23
**Status:** Planned (starts after vm-00108's Dreamer pilots; runs beside vm-00108). The owner, 2026-11-23, after vm-00107: "does paddle improves over time? what if we train to 1m ticks (PS? and just one seed)"

## Context

- **vm-00107:** in the stage-2 court with food for the paddle hit (`configs/U_breakout2p.json`), both arms were still rising at 60k:
  - PS (a meal worth its size) returned .232 of balls at 0–10k and .466 at 50–60k; by 5k from 30k: .348, .381, .406, .455, .455, .479.
  - 50–60k minus 40–50k: PK +.068 ± .017, PS +.037 ± .015, 12 of 12 seeds each.
  - Stage 1 (vm-00103, slower ball, no bricks) reached .68; the aimer walker returns every ball.
  - No preference for better bricks in either arm, lived or frozen.
- **The questions:** where does the return rate level off, and does a taste for the better bricks appear given far more time? The meal label's horizon (about 10 ticks) argues no for the taste; a 60k life can't tell.

## The run

- **One life:** PS (`online.meal_size = .5`), seed 1 (`env.brick_shift` 1, as vm-00107's PS seed 1), 1M ticks; nothing else changed.
- **Checkpoints every 50k** (20 checkpoints, about 0.7 GB; vm-00107 saved every 10k).
- **Memory:** unchanged (`online.replay` 100k): past 100k ticks the long-term memory holds the newest chunks only, as in any life longer than it. Learning rate is constant (no schedule depends on `online.steps`).
- **Card:** the RTX 4090, under MPS, beside vm-00108's runs. The speed alone is unknown (vm-00107: 4.4 ticks/s with 24 lives on the two cards); it's in the metrics rows (`sps`).
- **Frozen tests:** 2,000 ticks with learning off at 100k, 250k, 500k, 750k and 1M, in 6 new courts (seeds 101–106, the same look shift 1), so each checkpoint has more than one court.
- **Video:** 900 ticks at 1M from vm-00107's video court (seed 201), beside vm-00107's PS seed 1 at 60k.

## Measures

- **Lived:** paddle hit share, bricks, food, misses, hunger and preference by 50k (from the cumulative `bo_*` counters).
- **Frozen:** the same per checkpoint, the 6 courts' mean, against the random paddle (.229) and the aimer (preference −.014) in this court.

## The readings, set before the results

1. **Does the return keep improving?** The lived hit share by 50k: where it levels off, against vm-00107's PS seed 1 at 50–60k (.47) and stage 1's .68. Descriptive (one seed).
2. **Does a taste appear?** The lived preference by 100k (about 1,500–3,000 bricks per window), against the aimer's chance level; and frozen at 500k and 1M. Expected: none (the label's horizon).
3. **Does a long life break anything?** Hunger, poison, the meal monitor's predicted against real label over the life.

## Implementation

- [ ] `00109-breakout-long-life/` (README, Makefile: vm-00107's `SET`, PS's setting, `--set online.steps=1000000 --set online.save_every=50000`).
- [ ] The run, frozen tests, video, write-up.

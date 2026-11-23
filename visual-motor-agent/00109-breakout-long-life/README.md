# vm-00109: Breakout Stage 2b — One Long Life of 1M Ticks

**Date:** 2026-11-23
**Status:** Planned (starts after vm-00108's Dreamer timing pilots; runs beside vm-00108)
**Source:** `exp/vm-00109` (code: `visual-motor-agent/dev/`)
**Outputs:** `~/data/research/visual-motor-agent/exp_00109/`
**Task plan:** [`../tasks/00121-breakout-long-life.md`](../tasks/00121-breakout-long-life.md)

## Goal

The owner, 2026-11-23, after vm-00107: "does paddle improves over time? what if we train to 1m ticks (PS? and just one seed)".

- **vm-00107:** with food for the paddle hit, PS returned .23 of balls at 0–10k and .47 at 50–60k, still rising in all 12 seeds (50–60k minus 40–50k: +.037 ± .015). Stage 1 (vm-00103, a slower ball) reached .68.
- **Here:** the same life run 1M ticks, to see where the return levels off and whether a taste for the better bricks appears given far more time.

## Method

- **One life:** vm-00107's PS (`configs/U_breakout2p.json`, `online.meal_size = .5`), seed 1 (`env.brick_shift` 1), 1M ticks, checkpoints every 50k. Nothing else changed: memory 100k ticks (past it, the newest chunks), a constant learning rate.
- **Card:** the RTX 4090, under MPS, beside vm-00108's runs.
- **Frozen tests:** 2,000 ticks with learning off at 100k, 250k, 500k, 750k and 1M, in 6 new courts each (seeds 101–106, look shift 1).
- **Video:** 900 ticks at 1M beside vm-00107's PS seed 1 at 60k, from the same court (seed 201).

## The readings, set before the results

1. **Does the return keep improving?** The lived hit share by 50k: where it levels off, against vm-00107's .47 at 50–60k and stage 1's .68 (descriptive, one seed).
2. **Does a taste appear?** The lived preference by 100k against the aimer's chance level (−.014), and frozen at 500k and 1M. Expected: none (the label's horizon is about 10 ticks; a brick is 15 or more ticks from the return).
3. **Does a long life break anything?** Hunger, poison and the meal monitor's predicted against real label over the life.

## Results

(pending)

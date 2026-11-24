# vm-00110: Dreamer's Training Budget — the Default Agent at 7 Times the Replay

**Date:** 2026-11-23
**Status:** Running (since 2026-11-24 00:58, after vm-00108)
**Source:** `exp/vm-00110` (code: `visual-motor-agent/dev/`)
**Outputs:** `~/data/research/visual-motor-agent/exp_00110/`
**Task plan:** [`../tasks/00122-replay-ratio.md`](../tasks/00122-replay-ratio.md)

## Goal

The owner, 2026-11-23: "yes write the task and queue it after dreamer".

- **vm-00108:** in Breakout stage 1, DreamerV3 returns nearly every ball by 40–50k ticks; our default agent (vm-00103's BD) .61–.70 at 50–60k.
- **But Dreamer trains on about 8 times as much per tick:** 29.5M replayed steps in its 60k ticks (one update every 2 ticks on 1,024 steps) against our 3.84M (one update a tick on 64 steps: 8 windows of 8, the newest among them).
- **Here:** the same agent with Dreamer's replayed steps per tick, on the same 60k ticks of experience. Does the training budget explain the gap?

## Method

- **RB:** the default agent in Breakout stage 1 (`configs/U_breakout.json`) with `online.batch = 56`: each update 56 windows of 8 steps, the newest and 55 replayed, 448 steps (BD: 8 windows, 64 steps). 26.9M steps in 60k ticks (Dreamer 29.5M). Nothing else changed: no imagination, the same model, label and instinct.
- **Seeds 1–12, 60k ticks**, 6 per card under MPS; frozen at 30k and 60k (2,000 ticks, seed 100 + s, as vm-00103).
- **Pilot:** 2,000 ticks beside three Dreamer lives: batch 56 at 2.4 ticks/s against batch 8's 3.8.

## The readings, set before the results

1. **Does more training per tick speed learning?** RB against BD at 40–60k lived, paired by seed (12 seeds).
2. **Does it close the gap to Dreamer?** RB against DR at 40–60k lived, paired by seed (seeds 1–6).
3. **Cost:** wall-clock per life against BD's.

## Results

(pending)

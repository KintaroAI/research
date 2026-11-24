# Task 00122: Dreamer's Training Budget — the Default Agent at 7 Times the Replay (experiment vm-00110)

**Date:** 2026-11-23
**Status:** Planned (queued after vm-00108's Dreamer runs). The owner, 2026-11-23, on the count below: "yes write the task and queue it after dreamer"

## Context

- **vm-00108 (running):** in Breakout stage 1 (`configs/U_breakout.json`), DreamerV3 returns nearly every ball by 40–50k ticks (.83–.99 by seed); our default agent (vm-00103's BD) returns .61–.70 at 50–60k. Both are near random until about 30k.
- **The training each gets per tick is not the same:**
  - **Dreamer** (train ratio 512, batch 16 × 64): one update every 2 ticks on 1,024 replayed steps, after a 2,500-tick prefill and 100 updates at the start: about 28,850 updates and **29.5M replayed steps** in 60k ticks. Its actor and critic also train on about 15,000 imagined steps per update.
  - **Ours** (`online.batch` 8, context 8): one update every tick on 8 windows of 8 steps (the newest and 7 replayed; `online.batch` counts the newest), 64 steps: 60,000 updates and **3.84M steps**, 7.7 times fewer.
- **The question:** is Dreamer ahead because it trains more on the same experience? Give our agent Dreamer's replayed steps per tick, with the same 60k ticks of experience.

## The arm

- **RB:** the default agent unchanged in Breakout stage 1 except `online.batch = 56`: 56 windows of 8 steps (the newest and 55 replayed), 448 steps per update, 26.9M in 60k ticks (Dreamer 29.5M, so .91 of it). One update a tick, as before; no imagination added.
- **Seeds 1–12, 60k ticks**, 6 per card under MPS: paired with vm-00103's BD (seeds 1–12, batch 8) and vm-00108's DR (seeds 1–6).
- **The pilot** (2026-11-23, 2,000 ticks, seed 1, on the 3090 beside three Dreamer lives): batch 8 at 3.8 ticks/s and batch 56 at 2.4. The update takes about 4 times as long and the batch's assembly 2 times.

## Measures

- Paddle hit share by 10k lived (and misses per 1k), against BD and DR, paired by seed.
- Frozen at 30k and 60k (2,000 ticks, seed 100 + s, as vm-00103).
- The ticks to cross .5 and .9 lived (by 5k), per seed, for RB, BD and DR.

## The readings, set before the results

1. **Does more training per tick speed learning?** RB against BD at 40–60k lived, paired by seed (12 seeds).
2. **Does it close the gap to Dreamer?** RB against DR at 40–60k lived, paired by seed (seeds 1–6). Closing it says the gap was the training budget; not closing it points to what else differs (the imagined actor-critic, the model's size, the acting rule).
3. **Cost:** wall-clock per life against BD's.

## Implementation

- [ ] `00110-replay-ratio/` (README, Makefile: vm-00103's with `--set online.batch=56`, arm RB, frozen at 30k and 60k); queued after vm-00108's `make dreamer`.
- [ ] The runs, analysis, write-up.

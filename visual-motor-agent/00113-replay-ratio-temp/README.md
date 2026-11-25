# vm-00113: Dreamer's Training Budget with Varied Candidates

**Date:** 2026-11-25
**Status:** Running (seeds 1–6 on the 3090 since 2026-11-25 00:24, 7–12 on the 4090 since 01:18)
**Source:** `exp/vm-00113` (code: `visual-motor-agent/dev/`)
**Outputs:** `~/data/research/visual-motor-agent/exp_00113/`
**Task plan:** [`../tasks/00126-replay-ratio-temp.md`](../tasks/00126-replay-ratio-temp.md)

## Goal

The owner, 2026-11-25, on queuing it on the free 3090: "yes - queue it".

- **vm-00110:** with Dreamer's replayed steps per tick (`online.batch` 56) the default agent learns Breakout twice as fast but levels off at .73 (Dreamer .99). The move token fits the instinct's picks and the 16 candidates narrow to 8 distinct moves (BD 14).
- **Is the plateau the narrowed candidates?** Draw them at a higher temperature and see.
- **The check before the run** (`eval_candidates.py`, RB seed 1 at 60k, frozen): `instinct_temp` 2 gives 7.9 distinct moves, 4 gives 13.5 (BD 13.6), 8 gives 15.5. Temperature 4 restores BD's variety.

## Method

- **RT:** vm-00110's RB (`configs/U_breakout.json`, `online.batch = 56`) with `online.instinct_temp = 4` (default 2). Nothing else changed.
- **Seeds 1–12, 60k ticks**, under MPS, 1–6 on the RTX 3090 and 7–12 on the RTX 4090, as RB's; frozen at 30k and 60k (seed 100 + s).
- Paired by seed with RB (vm-00110), BD (vm-00103) and DR (vm-00108, Breakout).
- The candidates at 60k (`eval_candidates.py`), the move token's cross-entropy and the override share.

## The readings, set before the results (12 seeds)

1. **Does widening lift the plateau?** RT − RB at 40–60k above zero, and RT still climbing at 50–60k.
2. **Is the early speed kept?** RT − BD at 10–30k (RB: +.22).
3. **Toward Dreamer:** RT − DR at 40–60k (RB: −.23).

## Results

(pending)

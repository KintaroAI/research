# Task 00126: Dreamer's Training Budget with Varied Candidates (experiment vm-00113)

**Date:** 2026-11-25
**Status:** Complete (negative). The owner, 2026-11-25, on queuing it on the free RTX 3090: "yes - queue it"

## Context

- **vm-00110:** with Dreamer's replayed steps per tick (`online.batch` 56) the default agent learns Breakout twice as fast (.5 at 17–27k against BD's 32–55k) but levels off at .73 (DR .99). The move token fits the instinct's picks (cross-entropy .08 against .55) and the instinct's 16 candidates, samples of that prediction, narrow to 8 distinct moves (BD 14).
- **The check before the run** (`eval_candidates.py`, RB seed 1 at 60k frozen, its config's `instinct_temp` changed): temperature 2 (the default) 7.9 distinct moves, 4 gives 13.5 (BD 13.6), 8 gives 15.5. Temperature 4 gives back BD's variety under batch 56.
- **The question:** is the plateau the narrowed candidates? If so, the acting rule's imitation can be kept and its choice widened; if not, the ceiling lies elsewhere (task 00123).

## The arm

- **RT:** vm-00110's RB (`configs/U_breakout.json`, `online.batch = 56`) with `online.instinct_temp = 4` (the default 2). Nothing else changed.
- **Seeds 1–12, 60k ticks** under MPS, 1–6 on the RTX 3090 and 7–12 on the RTX 4090 (as RB's; 7–12 added when the 4090 came free), then frozen at 30k and 60k (seed 100 + s).
- Paired by seed with RB (vm-00110), BD (vm-00103) and DR (vm-00108, Breakout).

## Measures

- Paddle hit share by 10k lived; ticks to .5, .7, .9; frozen at 30k and 60k.
- The candidates (`eval_candidates.py`, frozen at 60k): distinct moves, all-same share, label spread.
- Move-token cross-entropy and the instinct's override share.

## The readings, set before the results (12 seeds)

1. **Does widening lift the plateau?** RT − RB at 40–60k above zero, and RT's 50–60k above its 40–50k (still climbing).
2. **Is the early speed kept?** RT − BD at 10–30k (RB: +.22).
3. **Toward Dreamer:** RT − DR at 40–60k (RB: −.23).

## Implementation

- [x] `00113-replay-ratio-temp/` (README, Makefile: vm-00110's with `--set online.instinct_temp=4`, arm RT, seeds 1–6 on GPU 0, 7–12 on GPU 1).
- [x] Queued after vm-00111's seeds 4–6 on the 3090 (with their frozen tests).
- [x] The runs, the candidates check, analysis, write-up.

## Outcome

- **Negative.** The candidates' variety came back (14.2 distinct moves against RB's 8.1 and BD's 13.6, 12 of 12) but the play did not change: RT − RB −.009 ± .037 at 40–60k (5 of 12), RT flat from 40–50k to 50–60k (−.002); frozen at 60k −.006. The early speed is kept (+.21 over BD at 10–30k, 12 of 12); Dreamer still leads by .25 at 40–60k.
- The plateau at about .72 is not the narrowed candidates; the ceiling is elsewhere in the acting rule (task 00123; vm-00115).

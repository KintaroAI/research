# Task 00090: Outage With Four Transformers (experiment vm-00079)

**Date:** 2026-11-03
**Status:** In progress

## Context

- vm-00078 (task 00089): with 16 transformers, a quarter off at random cost nothing. Lived food was level with all on, and frozen food fell about 2% even with half off. Each transformer alone foraged at about 75% of the team, because each holds the coarse summaries of the whole view.
- The owner (2026-11-03): "test 4 transformers with 0.25 and 0.5".
- With 4 quarters each transformer holds a larger share of the detail: 16 patches against 4, and a quarter of the view. So an outage should cost more here. It is the case where redundancy is smallest.

## Design

- **Arms:** Q25 and Q50: the default model (4 quadrant transformers, K4, log summaries), `online.off` .25 and .5, `online.off_span = 32`, six seeds, 60k ticks.
- **Control:** vm-00073's K4, seeds 1–6, the same model all on, not rerun.
- **At .5,** all four are off together with probability 1/16 per tick. Then the lowest, the top-left quarter, stays on, so that piece is on a little more than the others (about 53% of ticks against 50%).
- **Placement:** seeds 1–3 on the 4090, 4–6 on the 3090, all 12 at once.
- **Frozen:** `eval_outage.py chain` at test off shares 0, .25 and .5 for Q25, Q50 and K4; `restore`; `alone` (each transformer foraging by itself).

No code changes: the outage of task 00089 works for any number of transformers.

## The readings, set before the results

1. **Graceful:** Q25's lived food over the last 20k ticks is at least 75% of K4's; Q50's at least 50%.
2. **It costs here:** frozen at test off .5, K4's food is below its own at off 0 by more than its interval. Unlike 16 transformers, 4 do not carry each other.
3. **Practice pays:** frozen at test off .5, Q50's food is above K4's by more than its interval.
4. **Alone:** each quadrant transformer by itself gets less than 75% of the team's food: the reason 4 should be less redundant than 16.

## Implementation

- [ ] `eval_outage.py alone`, also run for vm-00078
- [ ] `00079-outage-four/`: README, Makefile
- [ ] Runs, frozen tests, write-up

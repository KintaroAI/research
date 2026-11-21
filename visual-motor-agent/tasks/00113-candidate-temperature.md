# Task 00113: The Instinct's Candidates Drawn at a Higher Temperature (experiment vm-00101)

**Date:** 2026-11-20
**Status:** Complete (vm-00101; the owner, 2026-11-20, on "do we use temperature in our models?": "write task for candidate temperature and run it after vm-00100")

## Context

- **How a move is chosen now** (`online.uni_act`):
  - the move token predicts a distribution over 32 bins for dx, dy and the click;
  - the instinct draws 16 candidates from it at temperature 1 and scores each by its predicted label;
  - it makes the best, unless the first sample is within the tolerance (.002) of the best;
  - with 4 transformers, `piece_rule = gain` makes the move of the one that predicts the best label.
- **The candidates collapse** (vm-00098 R8, vm-00099 R7):
  - once the model is sure of its move, almost all the probability sits in one bin, so the 16 draws are one and the same move;
  - all 16 candidates are the same move on .30 of ticks in SB and .13 in SD, against .02 in IB;
  - then the instinct has nothing to choose from, however good the label is.
- **The owner's question** led to this: the move's sampling has no temperature, and the candidates could have one.

## The design

- **`online.instinct_temp = T`** (default 1: unchanged):
  - the first candidate is drawn as now, at temperature 1. It is the model's own move: what the body does when nothing is at stake;
  - the other 15 are drawn from the same distributions at temperature T: p^(1/T), renormalized per value (dx, dy, click);
  - the scoring and the choice are unchanged.
- **T = 2.** A bin with 90% of the probability, the rest spread evenly over 31 bins, falls to about 35%, so about two thirds of the draws are other moves. The model's own move is still the most likely candidate.
- **The same draws at T = 1:** the 16 candidates are drawn as now, and with T ≠ 1 the 15 are drawn again from extra random numbers. So T = 1 repeats every earlier run exactly.
- **Used wherever the run's instinct acts:**
  - in life (`lifetime.py`);
  - frozen in `eval_hunger.py` (chain, value, reach, move, and so `eval_candidates.py`);
  - in the videos (`forage_video.py`, `compare_video.py`).
  - The older tools of other worlds (`eval_sense`, `car_video` and so on) keep T = 1.

## Arms

- **The base, a rule set before vm-00100's results:**
  - **RD** (re-labeled, task 00112) if RD − SD food at 40–60k, paired by seed, has a 95% interval above 0;
  - **SD** (stored once, vm-00099) otherwise.
  - The queue reads this from vm-00100's metrics and starts the arm.

| Arm | Base | Candidates | Seeds |
|---|---|---|---|
| the base (RD or SD), linked | as run | temperature 1 | 1–12 |
| **TT** | the base's settings | 15 of 16 at temperature 2 (`online.instinct_temp = 2`) | 1–12 |

- **One change against the base:** the candidates' temperature.
- **One card:** the RTX 3090, 12 lives under MPS. Drawing at a temperature costs nothing measurable, so the base's speed applies: RD 3.0 ticks/s (5.6 h), SD 4.1 (4.1 h).

## Measures

As vm-00099:
- **Lived** by 5k and at 40–60k: food, poison, mean hunger, overrides, move CE, smoothness; the seeds that starve.
- **The monitors** (`meal`, `meal_stored`).
- **Frozen:** `chain`, `value`, `reach` (poison), `move`, `eval_candidates.py`.

## The readings, set before the results

1. **Plumbing:**
   - with T = 1 the moves are exactly those of the code before (a test, same random numbers);
   - with T = 2 the first candidate is drawn as at T = 1, and the others are more varied on a sharp distribution (a test);
   - all 12 lives run 60k ticks.
2. **The candidates differ:** frozen, TT has more distinct moves among the 16 and fewer ticks with all 16 the same than the base, paired by seed.
3. **It forages** (the main reading): TT − base food at 40–60k, paired by seed.
4. **Descriptive:**
   - overrides;
   - poison, the instinct's poison picks;
   - smoothness: more varied candidates could make the path less straight, and more overrides could make it jittery.

**What follows:**
- **If 3 holds:** the candidates' temperature joins the instinct (the owner's decision), and Breakout (task 00111, renumbered vm-00102) is built with it.
- **If 2 holds and 3 fails:** the candidates were not the limit; the label is (its horizon, or the one-tick choice).
- **If 2 fails:** T = 2 is not enough; try a higher T or candidates from fixed directions.

## Implementation

- [x] `uni_act(..., temp=1.0)`: the 15 candidates after the first redrawn at temperature temp when it is not 1; `online.instinct_temp` in DEFAULT_ONLINE; passed by `lifetime.py`, `eval_hunger.Life`, `forage_video.py`.
- [x] Tests `tests/test_instinct_temp.py` (5: T = 1 the old draw exactly; T = 2 more varied with the first kept; the instinct finds a better move more often; off by default; a short GPU life and `eval_candidates.py`); regression: 336 pass, 4 skipped.
- [x] `00101-candidate-temperature/` (README, Makefile with the base as a variable, `make base`).
- [x] The queue (`q101.sh`, started 2026-11-20): wait for vm-00100, check its 12 lives reached 60k, choose the base, run, frozen tests, candidates, videos; the write-up.

## Verification

- Every TT run's `info.json` has `online.instinct_temp` 2 and the base's settings otherwise.
- The base chosen and the numbers it was chosen by are printed to the queue's output and given in the README.

## Results (vm-00101, [README](../00101-candidate-temperature/README.md))

- **The base:** RD (RD − SD food at 40–60k +45.76 ± 8.38, all 12 seeds).
- **Reading 2 met:** TT has 13.3 distinct moves among the 16 against RD's 4.4 (+8.9 ± 0.9, all 12 seeds), and no tick with all 16 the same (RD .10).
- **Reading 3 met:** TT − RD food at 40–60k +16.1 ± 7.4 (11 of 12): 71.6 per 1k against 55.5. Poison −1.5 ± 0.7, mean hunger −.032 ± .029.
- **Faster and alike:** TT eats 41 per 1k at 15–20k (RD reaches that at 30–35k); its seeds end at 69–76 (RD 41–76).
- **The label is right further out:** frozen, to 10 ticks before a meal (RD 6); still flat and optimistic beyond 15.
- **What follows (the plan):** the candidates' temperature joins the instinct if the owner decides so, and Breakout (task 00111, vm-00102) is built with it.

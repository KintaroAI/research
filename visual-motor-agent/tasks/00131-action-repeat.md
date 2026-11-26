# Task 00131: Action Repeat — a Move Decided Every 4 Ticks and Held (experiment vm-00118)

**Date:** 2026-11-25
**Status:** Running (built and queued 2026-11-25 on the RTX 4090 behind vm-00117's chain; the lives start at about 04:30). The owner, 2026-11-25, to the recommendation after vm-00116 (action repeat with k = 4, the agent holding its move while the model still sees every tick): "yes, lets commit and do the next arm when time is right". The idea is the owner's, 2026-11-17 (`ARCHITECTURE.md` section 9).

## Context

- **vm-00116's autopsy** of the default agent in Breakout stage 1 (BD, 60k): its label knows the court as a state (.085 with the paddle under the ball against .407 with it 16–24 px away, the last 4 ticks) but orders two candidate moves correctly only .60–.62 of the time in the last 8 ticks; an even grid of dx is no better. Its misses are a paddle that travels enough (about 31 px in its last 12 ticks) but not to the ball: 59% of them; 39% went mostly the wrong way; 2% moved too little.
- **Not the cause:** the ring, the horizon, the candidates' variety, the choice among the transformers (vm-00113, vm-00116).
- **A reason for the weak order** (vm-00116's analysis, not tested there): one move changes the next state's value very little. Between 5 and 16 ticks before a landing the label changes by about .005 per px, so two moves ending 4 px apart differ by about .02, the size of the label's whole spread over its 16 candidates.
- **Action repeat answers both findings:**
  - a move held for 4 ticks shifts the paddle up to 18 px, so what a decision does to the label is four times larger to learn;
  - a held move cannot dither.
- **Earlier, related:** vm-00096 held each candidate for 4 imagined ticks in the world with inertia and ate 47% more, at 2.4 times the acting cost. vm-00114's lookahead in the dream hurt with learning off. Action repeat is the form with real commitments and no imagination.

## The change: `online.act_every = k`

- **A move is decided on ticks 1, 1 + k, 1 + 2k, ...** as now: each transformer samples its 16 candidates, the instinct picks by the label, the choice among the transformers is made.
- **On the k − 1 ticks between, the move made is the kept one,** whatever the instinct picks there (`lifetime.hold_move`).
- **Everything else runs every tick, unchanged:** the model sees each frame, the ring takes each move, each transformer scores the move made, memory stores it, an update follows. The move token's target is the move made, so on held ticks the model learns to predict the repeat.
- **What the label then means:** the label of a deciding move is learned from lives in which that move was always held, so it carries what the hold leads to. Nothing tells the model which ticks decide; it can read the phase from its own last moves in the context (8 ticks, two holds).
- **Refused** below 1, and above 1 without the meal label, with `instinct_h > 1`, `actor_pg` or `off`.
- **No compute is saved here:** the instinct still runs on held ticks (its pick unused), to keep the change small. Skipping it is a later step.
- **One rule for every world;** nothing reads the world's state.

## Code

- **`dev/lifetime.py`:** `hold_move(a, won, held, decide)`; the option's check; two lines in the acting loop after the choice among the transformers; the metric `held` (the share of body-ticks whose move was a held one; logged only above 1).
- **`dev/online.py`:** `act_every = 1` in `DEFAULT_ONLINE`.
- **`dev/eval_hunger.py`:** `Life.act` holds as the life did (`Life.decided` says whether the tick decided), so the frozen tools act as the run lived.
- **`dev/tests/test_act_every.py`** (9 tests): `hold_move` by hand; a 40-tick life in each world logs `held` = .75 at k = 4; the default life logs no `held`; the refusals.
- **Checked before queueing:**
  - with the option off the changed code plays exactly as the old: BD seed 1's 60k weights with learning off, 600 ticks, the same hits, misses and losses row by row (only `grad_norm` differs, in its ninth decimal, as between any two runs);
  - 49 tests around the label and the acting pass against the change (`test_policy_gradient`, `test_instinct_temp`, `test_label_refresh`, `test_stored_label`, `test_full_dose`, `test_meal_size`, `test_misses`);
  - the change was built and tested in a copy of `dev/` and merged file by file, `online.py` first, because vm-00117's chains start new `lifetime.py` processes through the night.

## The arm

- **BA:** `configs/U_breakout.json` with `online.act_every = 4`. Nothing else changes.
- **Seeds 1–12, 60k ticks,** against BD (vm-00103), paired by seed. All 12 on the RTX 4090, 6 at a time (BD's seeds 1–6 ran on the 3090).
- **Then, learning off** at 60k: 6,000 ticks in new courts (seed 100 + s), acting as lived, against BD's in vm-00116 (the ring arm, the same courts and ticks).
- **Then vm-00116's autopsy** on BA's 60k weights.

## Readings, set before the results (12 seeds, paired by seed)

1. **Lived:** BA − BD hit share at 40–60k, and at 20–40k.
   - above 0 with its 95% interval at 40–60k: action repeat helps the play in Breakout. Next: the food world (vm-00112: test any acting rule in both worlds), and k = 2.
   - the interval includes 0, or below: it does not, at k = 4.
2. **The start:** the first 5k window at .5 (BD: 32–55k, median 40.5k).
3. **Learning off:** BA − BD hit share over 6,000 ticks in the same courts. No tick of BA's test has been seen before.
4. **The autopsy, the reason given for the arm:** at deciding ticks in the last 8 ticks before a landing, the label's order of the candidates (BD: .60–.62) and the share of moves toward the ball (BD: .58–.60); the misses by kind (BD: 59% enough travel but not to the ball).
   - the order rises with the play: the reason holds;
   - the play rises without the order: action repeat helps another way (less dithering alone);
   - neither rises: the label's order is not mended by a larger effect per decision. The arms left are those that teach the label what a move does (an advantage-style target; labels for moves not made, task 00114 part A).
   - **Added 2026-11-26, before the autopsy was run** (the lives of seeds 1–6 had ended; no recording existed): the order is scored twice. vm-00116 scores a pair of candidates by where one tick of each leaves the paddle. A deciding move of BA is made 4 ticks running, and a full move held for 4 ticks (18 px) can pass a ball that one tick of it (4.5 px) would only approach, so a label that is right about held moves could be counted wrong. The second scoring takes the paddle after the hold (4 ticks, or up to the landing if sooner). BD decides every tick, so for BD the two are the same number.
5. **What it may cost,** said beforehand: a decision only every 4 ticks cannot correct late, so balls that need a last-tick move may be lost; a full held move overshoots the catching zone (18 px against 14).
6. **Cost:** ticks per second against BD's.

## Not here

- The food world, and other k.
- Skipping the instinct on held ticks (the compute saving).
- ~~The autopsy's tables restricted to deciding ticks~~ (done with the change: `analyze_misses.py read` takes what belongs to a choice on the deciding ticks).

## Implementation

- [x] `hold_move`, the option, `Life.act`; `tests/test_act_every.py`
- [x] The checks above; merged into `dev/`
- [x] `00118-action-repeat/` Makefile and README; the lives queued
- [ ] The lives, the frozen tests, the autopsy (with deciding ticks apart)
- [ ] The tables and the write-up

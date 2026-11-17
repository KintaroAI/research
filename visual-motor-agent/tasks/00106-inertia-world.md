# Task 00106: A World With Inertia (experiment vm-00095)

**Date:** 2026-11-17
**Status:** Complete (2026-11-17; experiment vm-00095, `00095-inertia-world/`; asked for by the owner, 2026-11-17: "create intertia world where you need to spend effort to change direction of movement (if you strated movememnt it will continue unless new signal arrive and it will change it with inertia) - i want to see our default models to be able to adapt to changed environment without changing model itself - create a tast and run experiments")

## Context

- **The agent so far** has moved in a world where its move is its displacement: what it asks for, it gets, at once.
- **The owner's question:** can our default model adapt to a world whose dynamics change, without any change to the model?
- **The new dynamics:** in a world with inertia, a moving body keeps moving, and changing direction takes effort over several ticks.
- **The model here is the line of vm-00091–00094,** unchanged: the value agent C8 (`configs/U_view.json`, value γ .9 with n 4 and a fresh tail, 4 quarter transformers of 8 layers, tolerance .002, context 8). Its control is C8r, from vm-00094: the same agent in the same world without inertia, 12 seeds, the same code.

## The world (`env.inertia = k`, `world/circle_world.py`)

- **The move is a push,** not a displacement: vel ← clip(vel × (1 − `inertia_drag`) + move × max_step_px / k, ±max_step_px per axis), and the cursor moves by vel.
  - With no push the body keeps moving; with drag 0 it never slows by itself.
  - A full push takes k ticks from rest to full speed, and 2k to reverse.
  - Top speed is the old world's (4 px a tick per axis).
- **`inertia = 0` is the old world,** byte for byte (the world's tests, the JS replica's equivalence and the window's tests all pass).
- **Switching mid-life:** `inertia` and `inertia_drag` are in CircleWorld.RUNTIME, so `online.env_sched` can switch them mid-life. The body keeps moving as it last moved: the velocity is the last displacement.
- **The frozen tests and videos** test a run in the inertia it ended with (`lifetime.env_at_end`, only the inertia keys; other schedules as before).
- **No energy price is added:** the effort is the pushing, k ticks to start and 2k to turn round. A hunger price on pushing could follow (`env.turn_cost` prices changes of move already).
- **Tests:** `tests/test_inertia.py`, 9 tests: ramp and coast, reversal in 2k ticks, the per-axis cap, drag, a wall stopping the blocked axis, the switch mid-life, state round trips, inertia 0 as before, invalid settings.

## Arms

| Arm | World | Seeds |
|---|---|---|
| C8r (vm-00094, linked, not rerun) | no inertia | 1–12 |
| **IB** | inertia from birth, k = 4 | 1–12 |
| **IS** | no inertia until tick 30k, then k = 4 (`online.env_sched = {"inertia": [[30000, 4]]}`) | 1–12 |

- **Everything else is C8's:** `--config configs/U_view.json --set online.future_k=4 --set model.n_layer=8 --set online.instinct_tol=0.002 --set online.value_gamma=0.9 --set online.value_refresh=1`, plus `--set env.inertia=4` (IB) or the schedule (IS).
- **k = 4:** 4 ticks to full speed, 8 to reverse. Food is 6–10 px across, so a body at full speed overshoots food unless it slows down early.
- **Cards:** 24 lives, seeds 1–8 of both arms on the 4090 and 9–12 on the 3090 (as vm-00094, about 4 h).

## Measures

- **Lived, by 5k ticks:** food, poison, hunger, instinct overrides, move CE, patch CE, body CE, smoothness; each arm against C8r and IS against IB, paired by seed.
- **The switch:** IS's food in each 1k tick before and after 30k; recovery, the ticks until its food (smoothed over 3k) is back at its level of 25–30k, per seed.
- **Savings:** IS's first 10k ticks with inertia (30–40k) against IB's first 10k (0–10k), and IS against IB at the same ticks.
- **Frozen tests** (`chain`, `value`, `reach`) of IB and IS in the inertia world, for continuity.
- **Added 2026-11-17 during the runs (descriptive; the readings below are unchanged):**
  - **C8r frozen in the inertia world** (`chain --env inertia=4`): a model raised without inertia that does not learn. This is what the change costs without adaptation, against IS's lived food after the switch.
  - **How the body moves** (`eval_hunger.py move`, frozen): speed, the share of ticks at top speed, and braking (the move against the velocity kept). By the gap to the nearest food and poison: the speed, the move's aim and the motion's aim at the food, and the motion away from the poison. Run on IB, IS, C8r in its own world, and C8r in the inertia world.
  - `chain`'s aim scores the move. Under inertia the move is a push, not the motion, so `move` scores both.
- **Added during the analysis** (descriptive; `make frozen2`, after frozen C8r in the inertia world out-ate both learners):
  - IS's checkpoint at tick 30k (the model at the switch, before any learning with inertia) frozen in the inertia world (`chain`, `move`, `--step 30000`). This is the no-learning control for what IS's 30k ticks with inertia did.
  - Every arm's mean hunger, the agent's own objective (`eval_hunger.py poison`).

## The readings, set before the results

1. **Plumbing:** the world's tests pass; all 24 lives run 60k ticks without errors; every IS life switches at tick 30k (its metrics carry `env_inertia`).
2. **The default model learns the inertia world:** IB's lived food at 40–60k is at least half of C8r's.
3. **It adapts after a change:** IS's lived food at 50–60k is not below IB's at 50–60k by more than its interval, paired by seed. Having learned the old dynamics first leaves no lasting cost.
4. **The change is felt:** IS's food at 30–35k is below its food at 25–30k, paired by seed, by more than its interval.
5. **Poison** (descriptive): IB's lived poison at 40–60k against C8r's.
6. **Savings** (descriptive): IS at 30–40k against IB at 0–10k. Does a model that already forages learn the new dynamics faster than one learning them from birth?

**What follows:**
- **If 2 and 3 hold:** the model adapts to new dynamics as it is. Next is how fast: stronger inertia (k = 8), and switches back and forth.
- **If 2 fails:** the model does not learn to forage with inertia. Its instinct looks one move ahead and its hunger label 4 ticks, while a push pays only over several ticks. Next is the instinct's horizon (`online.instinct_h`) or the value's.
- **If 3 fails:** the old dynamics leave a lasting cost. Next is where: the policy's moves, the instinct's choices, or the value.

## Implementation

- [x] `env.inertia`, `env.inertia_drag` (`world/circle_world.py`): velocity state, the push, `gt_vel` (scoring), RUNTIME, state; `tests/test_inertia.py` (9 pass), the world's tests (213 pass).
- [x] `lifetime.env_at_end`: the frozen tests (`eval_hunger.Life`) and videos (`forage_video`) use the inertia a life ended with.
- [x] `CircleWorld.push_of(moves)`: where moves would take the body from the current state; `step()` moves by it. `eval_hunger.would_touch` and `in_reach` (the `reach` test, `chain`'s perfect-grab baseline) and `eval_pieces.touches_poison` use it, so they follow inertia; before, they assumed the move was the displacement. Inertia 0 is unchanged (the world's tests and the JS replica's equivalence pass).
- [x] `eval_hunger.py --env KEY=VALUE` (rows `ARM-inertia4_sN`), `--step N` (an earlier checkpoint, rows `ARM-stepN_sN`) and `eval_hunger.py move`; `tests/test_inertia.py` 11 tests, adding the scoring helpers against stepping copies (landing points, touches, in reach).
- [x] `00095-inertia-world/` (README, Makefile). The runs ran 2026-11-17 07:25 to 11:16 (4.4 ticks/s per life), the frozen tests until 12:00, and `frozen2` until 12:39. Analysis, write-up and videos (`videos/vm95_s5`, `vm95_s12`) are done.
- [x] Tests: `tests/test_inertia.py` 11 pass; the world's tests 212 pass; the GPU tests touching the changed files (lifetime, pieces, shared tables, fast, car, sense and thought worlds) 122 pass, with 4 skips by design (CPU variants of GPU-only piece rules).

## Results (summary; full report in the experiment README)

- **Readings:**
  1. Plumbing: **met.**
  2. IB learns the inertia world: **fails.** IB's lived food at 40–60k is .41 of C8r's (19.9 against 48.4).
  3. IS adapts: **met.** IS − IB at 50–60k is +4.5 ± 2.0 food, 12 of 12.
  4. The change is felt: **met.** Food −4.8 ± 2.8 and poison +13.4 ± 1.5 at 30–35k against 25–30k.
  5. IB's poison is 3.7 times C8r's.
  6. Savings: yes.
- **What the learning did** (frozen, IS at 60k against IS's own checkpoint at 30k, both in the inertia world): poison 21.3 to 10.7, food 23.0 to 25.9, and mean hunger .355 to .156, in 12 of 12 seeds. The model learned to brake, slow near poison and move away from it; it never learned to slow for food.
- **Frozen C8r in the inertia world** eats the most food (33.3) but touches 16.5 poison. By hunger it loses to the adapted IS (−.062 ± .030, 10 of 12).
- **Presumed, not tested:** the instinct overrides 58–70% of moves (C8r 27%). The candidates' spread is 5 times C8r's, as are the labels (hunger is higher), so the fixed tolerance (.002) is relatively 5 times smaller.
- **What follows (set above):** reading 2 failed, so next is the instinct's horizon (`online.instinct_h`) or the value's. Also proposed: IB with the tolerance scaled to its labels (.01).

## Verification

- [x] Every run's `info.json` has `env.inertia` 4 (IB), or 0 with `online.env_sched` `{"inertia": [[30000, 4]]}` (IS), and C8's settings otherwise (checked for all 24).
- [x] IS's metrics rows carry `env_inertia` 0 up to 30k and 4 after (all 12; the row at tick 30,000 already reads 4, as the switch comes before that tick's move).
- [x] Before the switch IS is C8r again: at 20–30k, IS − C8r food −0.2 ± 5.4, poison +0.1 ± 1.0, patch CE +.0001 ± .0014.

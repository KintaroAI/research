# Task 00110: A Poison Costs Its Full Dose at Any Hunger (experiment vm-00099)

**Date:** 2026-11-20
**Status:** Complete (2026-11-20; experiment vm-00099, `00099-full-dose/`; the owner, 2026-11-20: "we might want to remove hunger cap yes - start implementig it. Maybe we need firts to check if backprop in time works if we remove cap on inertia world"; design choice: "Full dose, sense capped")

## Context

- **vm-00098 (task 00109):** the owner's stored label works without inertia (S0: 1.4 times C8r's food), but fails with it. SB eats 6.4 food per 1k against IB's 19.9, and 4 of 12 lives starve at the hunger cap.
- **The trap (vm-00098's R8):**
  - hunger is clipped to [0, 1], and the stored label's poison cost is the rise of hunger;
  - near the cap a poison (+0.5) adds little, and at the cap nothing, so SB's label hardly sees poison (a poison move rated .004 worse in T, against MB's .040) and its instinct picks poison on .53 of such ticks;
  - in the starving lives the label is flat, and the instinct's 16 candidates (samples of the model's own move prediction) are mostly one and the same move.
- **The owner's question:** does the label still carry value back in time in the world with inertia once the cap no longer hides poison?

## The design (the owner's choice: "Full dose, sense capped")

- **The world** (`CircleWorld`) reports per body and tick how much hunger the clip at 1 cut off: `info["hunger_cut"]`.
  - It is 0 below the cap; nothing the world does changes.
  - At the cap a poison's cut is its whole 0.5, and the drift's .001.
- **`online.full_dose = 1`** (only with `online.label = meal`):
  - memory keeps each step's cut (field `cut`);
  - the stored label's cost is the hunger the step added counting the cut: a poison always costs its full 0.5, at any hunger;
  - the in-life monitors and the frozen `value` test count it the same way.
- **What the model senses is unchanged:** hunger in [0, 1] as before (the owner's one normalized range), and a meal still brings hunger to 0.
- **Off by default:** without it every run is unchanged.

## Arms

| Arm | World | Label | Seeds |
|---|---|---|---|
| SB (vm-00098), IB (vm-00095), MB (vm-00097), linked | inertia from birth | stored, capped cost; hunger value; re-labeled | 1–12 |
| **SD** | inertia from birth | stored, full dose | 1–12 |

- **C8's settings otherwise**, as SB: `configs/U_view.json`, 8 layers, tolerance .002, context 8, the ring, `model.body_label = 1`, `online.label = meal`, `label_gamma` 0.95, `env.inertia = 4`.
- **One card:** since the reboot of 2026-11-20 the VM has only the RTX 3090. The 12 lives run on it under MPS (timed first).
- **One change against SB:** the full dose. Different card than most of SB's lives (GPU lives are not bitwise reproducible anyway; arms are compared over seeds).

## Measures

- **Lived** by 5k and at 40–60k: food, poison, mean hunger, overrides, move CE, smoothness; the seeds that starve (mean hunger above .95 at 40–60k).
- **The monitors** (`meal`, `meal_stored`): the front by ticks to the next meal, and the bias.
- **Frozen:** `chain`, `value`, `reach` (poison in reach), `move`, and `eval_candidates.py` (how different the instinct's candidates are).

## The readings, set before the results

1. **Plumbing:**
   - `hunger_cut` is 0 below the cap and the poison's dose at it (tests);
   - with `full_dose` off, labels are as in task 00109 (tests);
   - all 12 lives run 60k ticks.
2. **Poison is seen:** frozen, SD's label rates a move into poison worse than SB's (.004 in T), and its instinct picks a poison move less often than SB's .53, paired by seed.
3. **No starvation trap:** fewer SD lives starve than SB's 4 of 12.
4. **The front moves back:** SD's predicted V four to seven ticks before a meal at 40–60k is above SB's .20.
5. **It forages:** SD − SB food at 40–60k, paired by seed; and against IB's 19.9 (descriptive).

**What follows:**
- **If 2 and 3 hold but SD stays near or below IB:** the cap was one trap, and the one-tick choice under inertia is the remaining one. Next is a choice that lasts (the instinct's horizon or the owner's action repeat) and real alternatives for the instinct; the Breakout world (task 00111) needs both anyway.
- **If SD passes IB:** the full dose becomes part of the stored label, and Breakout is built on it.
- **If 2 fails:** the cap was not what hid poison; look at the label's channel (the plan of task 00109: stored labels in the hunger channel).

## Implementation

- [x] `CircleWorld`: `hunger_cut` (both clips of hunger at 1), in `info`, always.
- [x] `online.full_dose`: `stored_labels(..., cut)`, `MealMonitor.add(..., cut)`, the memory field `cut` (`Replay(cut=True)`), `GatedMemory.write_label` passing it; `lifetime.py` writes it from `info`, checks the options (only with `label = meal`; refused if the world has no `hunger_cut`).
- [x] `eval_hunger.py value`: the realized stored label with the full dose for such runs.
- [x] Tests `tests/test_full_dose.py` (7: the world's cut, below the cap unchanged, the label, the monitor, memory with and without the option, the check, a short GPU life with its frozen value test); regression: 242 world, label, value and memory tests pass.
- [x] Timing on the RTX 3090: 12 lives at once, 3.9 ticks/s each, 10 GB; about 4.3 h for 60k ticks.
- [x] `00099-full-dose/` (README, Makefile).
- [x] The runs (2026-11-20 08:31 to 12:32, all 60k ticks), frozen tests, candidates, videos (`videos/vm99_s5`, `vm99_s3`), the write-up.

## Results (vm-00099)

- **Readings:** 1 met; 2 met, partly (poison rated .012 worse in T against SB's .004; picked on .42 of poison pairs against .53); 3 met (no SD life starves, SB 4 of 12); 4 fails (four to seven ticks before a meal SD predicts .18, SB .20); 5 not resolved against SB (food +3.3 ± 4.3), far below IB (9.8 against 19.9 per 1k) and MB (54.2).
- **The plan's branch** ("2 and 3 hold but SD stays below IB": the one-tick choice is the remaining trap) is not supported: MB has the same one-tick choice and eats 54.2. The stored label stalls two or three ticks before a meal; what separates MB is the label (re-labeled at every replay, in the hunger channel).
- **Next** (the owner's decision): split the two changes, re-labeled in the body channel against stored, with the full dose.

## Verification

- Every SD run's `info.json` has `online.full_dose` 1 and SB's settings otherwise.
- At the cap, a poison step's stored label is the label it would have below the cap (tested on a table).

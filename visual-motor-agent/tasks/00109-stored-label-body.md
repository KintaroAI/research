# Task 00109: The Label as a Body Value, Stored Once (experiment vm-00098)

**Date:** 2026-11-18
**Status:** Complete (2026-11-18; experiment vm-00098, `00098-stored-label/`; the owner, 2026-11-18: "yes, each transformer its own - write the task and run it ... I think we need to have lable same as hunger - property of the body and learinig parameter in embedding (same as hunger)"; choices: the label's input blank, built in one step; write-up: "lets commit and push and generate videos for stored label")

## Context

- **vm-00097 (task 00108):** the meal label made the default agent forage 1.8 times better without inertia and 2.7 times with it.
- **How that label was made:** memory kept only the raw facts (hunger before and after each step), and every replay completed the label with the current model's fresh prediction (`value_refresh`). Old memories were re-labeled by what the model knows now.
- **The owner, 2026-11-18: that is a cheat.** A label should be computed once, when the experience happens, from what the agent predicted then, and stored in memory as the truth:
  - a step that eats food: label 1;
  - any other step: 0.95 × the model's prediction for the next step's move, made at that next step;
  - a poison step: minus the hunger it added, plus the same.

  The value front then moves back one step each time the model has learned the newer labels, carried only by new experience: …, 0.9025, 0.95, 1.
- **The owner, 2026-11-18: the label should be a property of the body, like hunger,** with its own learned parameters. In vm-00097 it took over the hunger channel's target.

## The design

- **A third body value, the label** (`model.body_label = 1`):
  - The body token carries hunger, restlessness and the label, each with its own channel: its own input embedding rows (`model.uni_chan`, with `in_bins`), its own output rows and bias.
  - The move token predicts all three for the next body.
  - **The label's input is blank** (the owner's choice): a constant 0.5, because a move's label is only known after the model's next prediction.
  - Hunger goes back to predicting real hunger: no `future_k` or value label on it.
  - **The instinct reads the label channel:** lowest predicted T, as now.
- **The stored label** (`online.label = meal`, `online.label_gamma = 0.95`):
  - **When acting,** after the body's move is chosen, each transformer scores the move actually made (the one-token candidate pass that council uses), and memory stores each transformer's own predicted label for it (the owner's choice 1).
  - **One tick later,** step t's label for transformer k is written into memory, once:
    - a meal (hunger fell): V = 1;
    - a poison (hunger rose by more than .01): V = −(the hunger it added) + 0.95 × P_k(t+1), the owner's choice 2;
    - any other step: V = 0.95 × P_k(t+1), with no cost for the drift, as the owner described.
  - V is clipped to [−1, 1] and stored as T = (1 − V)/2 (lower is better).
  - **Never re-labeled:** windows replay the labels as stored, in short-term and long-term memory.
  - **Each transformer trains its label channel on its own stored labels.**
- **Optimism** (the owner, choice 3): addressed later. The in-life monitor (`online.MealMonitor`) shows it, as in vm-00097.

## Arms

| Arm | World | Label | Seeds |
|---|---|---|---|
| M0, MB (vm-00097, linked) | none, inertia | meal label in the hunger channel, re-labeled at each replay | 1–12 |
| **S0** | no inertia | stored meal label, its own body channel | 1–12 |
| **SB** | inertia from birth | stored meal label, its own body channel | 1–12 |

- **C8's settings otherwise:** `configs/U_view.json`, 8 layers, tolerance .002, context 8, the ring. `future_k` 0 and `value_gamma` 0 (hunger is plain hunger).
- **Cards:** 24 lives, seeds 1–8 on the 4090 and 9–12 on the 3090, under MPS.
- **Not one change at a time:** two things change against M0 and MB (stored against re-labeled, and the label's own channel against the hunger channel). That is the owner's choice, one step, and the readings say so.

## Measures

- **Lived** by 5k and at 40–60k: food, poison, mean hunger, overrides, move CE, patch CE, body CE (now hunger's own prediction), smoothness.
- **The in-life monitor:** predicted against true V by ticks to the next meal, and the bias over all ticks, by life stage. It shows how the stored labels spread back in time, and how optimistic they turn.
- **What memory holds:** the stored labels by ticks to the next real meal, sampled from long-term memory at the end of each life: the front the owner described.
- **Frozen** (`chain`, `value`, `reach`, `move`) in each run's world.

## The readings, set before the results

1. **Plumbing:**
   - the body label's gradients match the C++ model and the reference (tests);
   - the stored labels are written once and never change;
   - all 24 lives run 60k ticks without errors.
2. **The front moves back:** in life, the predicted V near meals rises over the life toward the true V (0, 1, 2–3 ticks first), as in vm-00097.
3. **It still forages:** S0's lived food at 40–60k is above C8r's (48.4), and SB's above IB's (19.9), paired by seed.
4. **Stored against re-labeled** (descriptive): S0 − M0 and SB − MB at 40–60k, food and hunger, paired by seed. Two changes are confounded here (see Arms).
5. **Optimism** (descriptive): the monitor's bias over all ticks at 40–60k, against vm-00097's +.03.

**What follows:**
- **If 2 and 3 hold:** the stored label in the body is the honest version. It becomes the base for need done fairly and the two labels.
- **If 2 fails** (the front does not move back) **or 3 fails:** re-labeling did work that stored labels cannot do in a life of 60k ticks. Next is splitting the two changes: stored labels in the hunger channel, and re-labeled ones in the body channel.

## Results (summary; full report in the experiment README)

- **Readings:**
  1. Plumbing: **met,** after the reruns. Four lives died when the disk filled, not from the code: three were rerun from tick 0, one final was restored.
  2. The front moves back: **met without inertia, partly with it.**
     - S0's predicted V one tick before a meal rises from .10 to .94 over the life, four to seven ticks before from .06 to .64.
     - SB's moves back for 10k ticks, then stalls (.74 and .20).
  3. It still forages: **fails.**
     - S0 − C8r food +21.5 ± 7.9 (higher in 11 of 12): met for S0.
     - SB − IB −13.5 ± 3.6 (lower in 12): SB eats 6.4 per 1k against IB's 19.9.
  4. Stored against re-labeled (descriptive): S0 − M0 food −17.2 ± 14.2, SB − MB −47.8 ± 7.2.
  5. Optimism: **none.** S0's predictions are −.008 ± .007 over all ticks at 40–60k (M0 +.034).
- **Without inertia the stored label works as the owner described it.** The front moves back over the life, the predictions end unbiased, and S0 eats 1.4 times C8r's food (0.80 of M0's), still climbing at 60k.
- **With inertia it fails, through a trap.**
  - A third of the lives starve at the hunger cap with a silent instinct, three of them locked into one straight push. Frozen, the candidates' labels in those lives are practically all equal (spread .0001 or less).
  - Under inertia one push moves the next position by at most 1 px, SB's label reaches only one to three ticks before a meal, and at the cap a poison costs nothing.
  - The owner, watching the videos: "in case of inertia it moves in straight line fool speed and periodically changes direction - looks like there is not much change in the possible future moves to choose from".
- **Next:**
  - set in the plan: split the two changes in the world with inertia;
  - also proposed: choices that last (the instinct's horizon, or the owner's action repeat), a poison cost that survives the hunger cap, and ties broken at random.

## Implementation

- [x] **`model.body_label`** in C++ (`vm_agent`), CUDA (`vm_cuda`), `agent_lib`, `cuda_lib` and the reference:
  - the body's used values 2 + 1; the move's channel after the label's; `VC` one larger;
  - the label's input 0.5;
  - the move token's third target from the frame's prev_reward slot, which the universal model never read (the GPU now uploads it when the body has a label);
  - `next_body` 3 wide; `uni_candidates` returns the label;
  - header slot 62;
  - refused without `uni_chan` or with `state`.
- [x] `online.uni_lookahead` and `uni_act` read the label when the model has it; the lookahead's imagined bodies keep only hunger and restlessness.
- [x] **`online.label = meal`** (`label_gamma`):
  - `stored_labels` (a meal 1; a rise above `POISON_RISE` .01 a cost; no drift cost);
  - memory fields `vhs` (each transformer's prediction for the move made) and `lab` (the stored label), with `GatedMemory.write_label` writing the step before the newest, once;
  - windows hand the labels out as written;
  - the newest window ends a step back.
- [x] **`lifetime.py`:**
  - the checks (`online.label = meal` goes with `model.body_label` and only with it; no hunger value, need, state, outage, clog, noise or surprise gating);
  - each transformer scoring the move made after babbling and before the ring takes it;
  - writing the labels;
  - each transformer's own labels as its update's label target;
  - the two monitors (`meal`, `meal_stored`).
- [x] `eval_hunger.py value`: the realized label as stored (no drift cost, `label_gamma`). `eval_hunger.Life` and `forage_video` need nothing: the candidates return the label.
- [x] **Tests:**
  - `tests/test_body_label.py` (29): the tables; C++ and GPU against the reference, forward, loss and gradients; the label's target reaching the loss through the label's rows only; the candidates reading the label; the ring; checkpoints; refusals.
  - `tests/test_stored_label.py` (7): the arithmetic; the owner's mechanism on a table (only the meal's 1 first, then 0.95, then 0.9025: the front moving back a step per run); memory writing each label once and handing it out unchanged; the options' checks; a short GPU life.
  - Regression: 1,140 model tests and 118 tests of lives, memory and labels pass.
- [x] `00098-stored-label/` (README, Makefile), the runs, the frozen tests, the write-up and videos. No separate timing: the 24 lives took 2 h 50 min at once. The disk filled as they ended; three lives were rerun from tick 0 and one final was restored (the README's run notes).

## Verification

- Every run's `info.json` has `model.body_label` 1, `online.label` meal, `label_gamma` 0.95, `future_k` 0, `value_gamma` 0, and C8's settings otherwise. **Checked: 24 of 24, the reruns included.**
- The stored labels in memory never change after they are written (checked in a test, and by sampling at the end of a life). **Checked in the test only:** the sample at the end of a life was not taken. The `meal_stored` monitor records each label as it is written, which is what memory holds.

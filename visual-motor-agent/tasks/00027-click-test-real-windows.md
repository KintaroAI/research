# Task 00027: The Click Test on Real Windows

**Date:** 2026-10-25
**Status:** Complete

## Context

- The click test (`analyze_hazards.py click`, vm-00017) places the cursor on food, on poison or on the background,
  sets hunger to 0.3, and gives the model a **one-step window** (`es=1`) with the move "stay, click" or "stay, no
  click". The agent never acts like that: in its lifetime it has 7 earlier steps in context, its own non-zero moves
  and its own hunger.
- vm-00021: the click test said that five seeds of six predict no poison effect at 200k, yet in the forage test the
  instinct, which reads the same prediction inside real windows, cut poison clicks in five seeds of six (seed 4:
  47.3 → 18.8 per 1,000 steps). The placed test under-measures the in-context prediction.

## The test (`analyze_hazards.py click-real`)

1. **The frozen agent lives as in the forage test** (`online.eval_forage`): 64 continual worlds (the forage test's
   seed), 1,000 steps, its own moves from the last `ctx` steps, with the run's instinct (`--instinct` overrides).
2. **At every step, the agent's chosen move is tried with and without a click.** The predicted effect of the click is
   the move token's predicted next hunger with click 1 minus with click 0, same (dx, dy), as the instinct reads it.
3. **The truth from copies of the world** (`get_state` / `set_state`, scoring only): the step with the other click is
   run on a copy and the world restored; the real step is the actual one. The click branch's info says where the
   cursor landed: food, poison, both (dropped), nothing. The true effect is the difference of the two branches' next
   hunger (so clipped at 0 and 1 as the world clips it).
4. **History lengths:** the same predictions from the window cut to its last 1, 2, 4 and `ctx` steps (the context
   restarting at the cut). The 1-step case is the placed test's window, at real states and moves.

## Measures (per checkpoint, `eval/ckpt_NNNNNNNN_click_real.json`)

- Per kind (food / poison / nothing) and history length: cases, mean predicted and true effect, and over the cases
  whose true effect is felt (|true| > 0.05) the share ranked right beyond the instinct's tolerance (food: click
  predicted lower by more than 0.005; poison: higher). For "nothing": the share predicted flat (|effect| < 0.05).
- In context: food and poison cases split by whether the window's earlier steps hold a click on the same kind.
- The next-hunger error of the move actually made, and the prediction error of the current step's patches (mean
  squared error of the expected values, all patches and the patches that changed since the previous step), per
  history length: how much the model uses its history.
- Behaviour: food and poison clicks per 1,000 steps (as the forage test).

## Verification (`tests/test_click_real.py`)

- The branching does not disturb the real run: the world's state after the test equals a run of the same moves without
  branches.
- The truth: on placed states, food clicks relieve hunger, poison clicks raise it, clicks on nothing do not (beyond
  h_rate), as the world's rules say.
- The predictions: a history length of k gives the same predicted hunger as a separate forward pass of the last k
  steps with the context starting there; the GPU's fast path matches the C++ model's full passes.

## Run

vm-00021's six seeds at 25k, 50k, 100k, 150k, 200k, with the instinct, against the placed test (R1 of vm-00021).

## Results (2026-10-25, vm-00021's six seeds; 64 worlds × 1,000 steps per checkpoint, the instinct on)

Cases: every step where the agent's chosen move lands on food or poison counts, clicked or not (the click is tried
both ways). Felt cases (|true effect| > 0.05) per checkpoint: 570–7,200 per kind.

**Share of felt poison clicks predicted to raise hunger (beyond 0.005), the window's full 8 steps:**

| Seed | 25k | 50k | 100k | 150k | 200k | placed test at 200k (poison click − stay) |
|---|---|---|---|---|---|---|
| 1 | 0.66 | 0.70 | 0.72 | 0.86 | 0.66 | −0.00 |
| 2 | 0.49 | 0.54 | 0.65 | 0.68 | 0.78 | +0.02 |
| 3 | 0.84 | 0.87 | 0.84 | 0.86 | 0.79 | +0.16 |
| 4 | 0.03 | 0.21 | 0.45 | 0.53 | 0.46 | +0.02 |
| 5 | 0.02 | 0.09 | 0.02 | 0.13 | 0.18 | +0.02 |
| 6 | 0.76 | 0.49 | 0.72 | 0.44 | 0.72 | −0.03 |

Food (share predicted to relieve) at 200k: 0.60, 0.45, 0.81, 0.82, 0.92, 0.60. Clicks on nothing predicted flat
(|effect| < 0.05): 0.95, 0.72, 0.83, 0.95, 0.40, 0.50.

**Magnitudes are far too small:** at 200k the mean predicted poison effect is +0.02 to +0.14 against a true +0.20 to
+0.48 (lower where hunger is near the cap), food −0.01 to −0.13 against −0.11 to −0.41.

**History (200k, mean squared error of the current step's patches):** 1 step 0.0036–0.0039, 2 steps 0.0017–0.0018,
4 and 8 steps the same as 2 (changed patches: 0.0090–0.0096 → 0.0055–0.0063). The poison share rises from 1 to 8
steps in five seeds of six (seed 1: 0.54 → 0.66, seed 2: 0.63 → 0.78, seed 6: 0.52 → 0.72; seed 4 falls 0.64 → 0.46).

**In context (200k, 8 steps):** poison clicks after a poison click earlier in the window are predicted right more
often in three seeds (seed 1: 0.83 against 0.65, seed 4: 0.62 against 0.38, seed 5: 0.30 against 0.11), the same in
two, less in one (seed 2: 0.45 against 0.78, 51 cases).

**Early in life, seeds 4 and 5 predict relief for every click:** at 25k food −0.57/−0.52, poison −0.28/−0.35, nothing
−0.70/−0.73. They unlearn it slowly (seed 5 still predicts relief on 60% of clicks on nothing at 200k).

## Conclusions

1. **The placed test missed the poison prediction.** With real windows, four seeds of six predict that a poison click
   raises hunger in 66–79% of the cases by 200k (seeds 1, 2, 3, 6); for three of them the placed test showed −0.03 to
   +0.02. It has the right sign but 5–30% of the true size.
2. **Two steps of history are all the frames use** (one step doubles the patch error, more than two steps add
   nothing). A longer context (ctx 16) would not improve frame prediction in this world. The click outcome uses a bit
   more history in five seeds of six.
3. **There is some in-context learning of poison** (three seeds of six), but not a reliable one.

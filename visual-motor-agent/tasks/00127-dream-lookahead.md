# Task 00127: The Instinct Looks Ahead in Its Dream (experiment vm-00114; option (a) of task 00123)

**Date:** 2026-11-25
**Status:** Complete, negative (vm-00114). The owner, 2026-11-25, on trying option (a) in the food world and Breakout stage 1: "sure, lets do this"; on the frozen result: "close it as negative" (no lives).

## Outcome

- Frozen, 12 seeds: H = 4 hurts in both worlds (food world −11.4 ± 1.0 food per 1k, 0 of 12 up; poison +3.8 ± 1.2; Breakout hit share −.096 ± .044, 9 of 12 down). H = 2 is neutral in the food world (+0.5 ± 1.3) and −.064 ± .035 in Breakout.
- The dream probe (`analyze_lookahead.py record/read`, 3 seeds per world): with no real meal in the next 4 ticks the best candidate is still credited .19 of a meal (food world, hunger median .009), and the 4th move's dreamed label alone varies .32 across the candidates. Both parts of the score are noisy; a better hunger reading would not rescue it.
- Next for task 00123: option (b), a critic trained on the model's dreams.

## Context

- **The acting rule** (task 00123): our instinct picks among 16 candidate moves by each one's predicted label one move ahead. Dreamer trains an actor on 15-step dreams scored by a critic. vm-00111 (the long label lags) and vm-00112 (Dreamer's world model knows poison, its acting does not) both point at the acting rule.
- **The lookahead exists** (`online.instinct_h`, task 00029; vm-00096): each candidate is held for H ticks, and the model imagines the ticks after the first (Jacobi decoding; a quarter piece imagines only its own block). In the inertia world H = 4 ate 47% more than H = 1 (12 of 12 seeds), at 2.4 times the compute. It was never tried with the meal label (task 00108), in Breakout or against poison.
- **The meal label breaks its score.** vm-00096 scored predicted hunger after the H-th move, which a meal inside the dream lowers. With the meal label (`model.body_label`) `uni_lookahead` reads the label of the H-th move, which counts only what follows it: a candidate that eats, or touches poison, at imagined steps 1 to H − 1 gets no credit or blame.

## The change: an H-step return from the dream

- Each imagined step k < H: its meal and poison are read from the dream's own predicted hunger (before and after the move k), by the stored-label rule (`stored_labels`: a fall is a meal, a rise above `POISON_RISE` a poison costing the rise; `online.meal_size` as there). The predicted hunger is an expectation, so it is read softly: a fall is a meal with probability fall / hunger before (in both worlds a meal empties hunger: `h_relief` 1, a Breakout hit), a rise is its expected cost.
- The return, backward from the H-th move's own predicted label V_H (V = 1 − 2T): v_k = q_k · m_k + (1 − q_k)(−p_k + γ v_{k+1}), with q the meal chance, m its worth (1, or the size rule), p the poison cost, γ `online.label_gamma`. The candidate's score is v_1 on the label's scale.
- H = 1 is unchanged (v_1 = V_1). With the meal label and H > 1 the return is the only score: `uni_lookahead` refuses H > 1 with `body_label` without γ.
- One rule for every world; nothing reads the world's state.

## Step 1: frozen check (no training)

The same 60k checkpoints acting three ways for 2,000 ticks in new worlds (seed 100 + s): as they lived (ring, H = 1), full windows with H = 1 (`online.ring = 0`), full windows with H = 4.
- Food world: TT (vm-00101, `configs/U_agent.json`), food and poison per 1k.
- Breakout stage 1: BD (vm-00103, `configs/U_breakout.json`), hit share.
- If H = 4 helps frozen, the lives follow; if it hurts, look at the dream (`eval_dream.py`) before any life.

## Step 2: lives (arm LA: `online.instinct_h = 4`, `online.ring = 0`)

- Food world, seeds 1–12, against TT; Breakout stage 1, seeds 1–12, against BD (paired by seed). 60k ticks, then frozen at 30k and 60k. Seeds 1–6 first, then 7–12; the first 6 are not a result.
- Cost: about 2.4 times a default life (vm-00096); measured in the first hour.

## Readings, set before the results (12 seeds, 40–60k)

1. **Food world:** LA − TT poison per 1k below zero, and food not below TT's (TT 72.6 food, 8.5 poison).
2. **Breakout:** LA − BD hit share above zero; ticks to .5.
3. **Frozen** at 60k agrees in sign with 1 and 2.

Known costs counted in the package, not controlled: full windows instead of the ring (−1.7 food per 1k in vm-00096).

## Not here

- The bricks court: a brick is 20+ ticks from the aim, beyond 4 imagined ticks (option (b), a critic trained on dreams).
- H > 4: the quarter pieces hold the rest of the frame at the current one; the ball crosses quarters.

## Implementation

- [x] `online.dream_return`; `uni_lookahead(..., gamma)` and `uni_act(..., gamma)`, refused without γ when the body has the label and H > 1; `lifetime.py` and `eval_hunger.Life` pass `label_gamma`. `lifetime.py` refuses H > 1 with `meal_size` (a brick's meal does not empty hunger, so the soft reading does not hold there).
- [x] `tests/test_lookahead.py`: the return by hand (a meal at step 2, a poison at step 2, nothing, half a meal's fall, the clip); H = 1 unchanged; the refusal; with γ 0 the score is move 1's outcome read from its predicted hunger. 84 tests pass (lookahead, temperature, body label, stored label).
- [x] **A fault found on the way** (it predates this task: H > 1 had never run with the label in the body). `lifetime.py` scored the move made with `uni_candidates`, which reuses the last forward pass. After a lookahead that pass is the 16 candidate windows', so the C side wrote 16 rows into a 1-row buffer: a heap overrun, and a crash by tick 4. Fix: with H > 1 the move made is scored by a full `forward_uni` pass. `CudaModel.uni_candidates` now refuses a batch that differs from the last `forward_uni`, `forward_uni_step` or `uni_ring_step`.
- [x] Smoke test: both worlds run at H = 4, 7.3 ticks/s alone on the 4090 (full windows at H = 1: 26; 3.6 times, as vm-00096 found with J = 2).
- [x] Step 1 on the RTX 4090 (`make check`, 2026-11-25), H = 2 added after H = 4 hurt; the dream probe (`make probe`, `dev/analyze_lookahead.py`).
- [ ] Step 2 (lives): not run, closed negative frozen (the owner).

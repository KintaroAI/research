# vm-00114: The Instinct Looks Ahead in Its Dream

**Status:** Complete (negative, frozen; the lives were not run, the owner, 2026-11-25)
**Task:** `tasks/00127-dream-lookahead.md` (option (a) of task 00123, the acting rule)
**Source:** `dev/` (pending tag `exp/vm-00114`)

## Question

The default agent picks among 16 candidate moves by each one's predicted label one move ahead. Does holding each candidate for 4 ticks in the model's own dream, and scoring it by the dream's 4-step return, help where the one-step choice falls short: poison in the food world (where Dreamer fails) and the plateau in Breakout?

## Method

- **The lookahead** (`online.instinct_h = H`, task 00029; vm-00096): each candidate is held for H ticks. The model imagines ticks 2–H (Jacobi decoding, J = 2), and each quarter piece imagines only its own block. It needs full windows (`online.ring = 0`).
- **The score** (new, `online.dream_return`): each imagined step's meal or poison is read from the dream's predicted hunger by the stored label's rule. A fall is a meal, with chance fall ÷ hunger before. A rise above .01 is a poison costing the rise. The H-th move's own predicted label closes the return, with γ = `label_gamma` (.95). Before this, a lookahead with the meal label scored only the H-th move's label, which misses a meal or poison inside the dream.
- **A fault fixed on the way:** with the label in the body and H > 1, `lifetime.py` scored the move made by `uni_candidates`. That call reuses the last forward pass, which was the lookahead's (16 windows) and not the policy pass (1), so the C side wrote 16 rows into a 1-row buffer (a heap overrun; the life crashed by tick 4). With H > 1 the move is now scored by a full pass, and `uni_candidates` refuses a batch that does not match the last forward pass.
- **Frozen check** (`make check`): TT's (vm-00101, food world) and BD's (vm-00103, Breakout stage 1) 60k checkpoints, seeds 1–12, 2,000 ticks with learning off in new worlds (seed 100 + s). Each acts as it lived (ring, H = 1; BD's is vm-00103's frozen 60k), on full windows at H = 1, at H = 2 (added after H = 4 hurt) and at H = 4. Pairs are by seed, against full windows at H = 1.
- **The dream probe** (`make probe`): seeds 1–3 of each world at H = 4, 600 ticks frozen, saving every input of `dream_return`: the 16 candidates' predicted hunger over the 4 ticks and their 4th labels.
- **Lives** (step 2 of the task, arm LA, 12 seeds per world) were planned only if H = 4 helped or was neutral frozen. They were not run.

## Readings (set before the results)

1. Food world: LA − TT poison per 1k at 40–60k below zero, with food not below TT's. *Not run.*
2. Breakout: LA − BD hit share at 40–60k above zero; ticks to .5. *Not run.*
3. Frozen at 60k agrees in sign with 1 and 2. *Read as the gate: H = 4 frozen hurts in both worlds.*

## Results

### R1. Frozen, H = 4 hurts in both worlds; H = 2 is neutral in the food world

12 seeds, 2,000 ticks each; differences against full windows at H = 1, paired by seed (mean ± standard error; seeds up and down).

| Measure | As lived | Full, H = 1 | Full, H = 2 | Full, H = 4 | H = 2 − full | H = 4 − full |
|---|---|---|---|---|---|---|
| Food world: food per 1k | 69.8 | 70.0 | 70.5 | 58.6 | +0.5 ± 1.3 (7 up, 5 down) | −11.4 ± 1.0 (0 up, 12 down) |
| Food world: poison per 1k | 8.2 | 7.9 | 7.8 | 11.7 | −0.2 ± 1.1 (6, 5) | +3.8 ± 1.2 (9 up, 3 down) |
| Breakout: hit share | .644 | .765 | .701 | .669 | −.064 ± .035 (4 up, 7 down) | −.096 ± .044 (3 up, 9 down) |

- H = 4 eats 16% less and takes 47% more poison. In Breakout it catches fewer balls in 9 of 12 seeds.
- H = 2 changes nothing in the food world and costs a little in Breakout (not significant).
- The instinct overrides the model's first sample more often with each step of horizon (food world .73 → .78 → .83; Breakout .47 → .71 → .77), and the score's gain over the first sample grows 3 to 5 times (food world .018 → .056; Breakout .006 → .030). The lookahead changes the choices a lot, and for the worse.
- Side reading: in Breakout full windows beat the ring frozen (.765 against .644, as lived); in the food world they are equal (70.0 against 69.8).

### R2. The dream's score is mostly noise

Seeds 1–3 of each world at H = 4, 600 ticks frozen; one row per tick and quarter piece (7,140 in each world).

| | Food world | Breakout |
|---|---|---|
| Real hunger (median) | .009 | .068 |
| Ticks with a real meal in the next 4 | .21 | .033 |
| Best candidate's dreamed meal chance, no real meal in the next 4 ticks | .19 (above .1 at 52% of ticks) | .10 (above .1 at 17%) |
| Best candidate's dreamed meal chance, a real meal in the next 4 ticks | .44 | .52 |
| Spread of the return across the 16 candidates | .31 | .17 |
| Spread of the 4th move's label alone | .32 | .12 |

- **The meal reading is noisy.** The food-world agent eats so often that its hunger stays near zero (median .009). There a wobble of a few thousandths in the dreamed hunger reads as a large part of a meal: with no meal coming, the best candidate is still credited .19 of one. A real meal reads higher (.44), so there is signal, but the noise is half its size.
- **The dreamed label is noisy too.** The 4th move's label alone varies across the 16 candidates by .32 (on the label's −1 to 1 scale) in the food world, as much as the whole return. A better hunger reading would not rescue the score: the model's predictions on its own frames 3 ticks out are not reliable enough to rank moves, and holding one candidate for 4 ticks amplifies their errors.

## Analysis

1. **A dream read by a model trained only on real frames does not rank moves.** At H = 2 the one imagined tick is good enough not to hurt (food world); at H = 4 the noise in the dreamed hunger and the dreamed label outweighs the one-step label's real differences between candidates (gain .018 at H = 1), and the instinct follows the noise: food −16%, poison +47%.
2. **This is unlike vm-00096**, where H = 4 ate 47% more in the inertia world. There the score was the predicted hunger after the 4th move (before the meal label), and the lookahead ran through the whole life. A frozen check cannot show whether a life lived with the lookahead learns a more reliable dream; it was not tested here (the owner closed it as a negative).
3. **For task 00123:** Dreamer trains its critic on dreamed states, so the critic learns to value what the dream shows, errors included, instead of reading outcomes from the dream with rules fit for real frames. That is option (b), the next candidate.

## Not done

- Lives with H = 4 (step 2 of the task): about 3.6 times the compute of a default life; closed negative frozen.
- Other readings of the dream (a meal only above a fixed fall, per-step dreamed labels).

## Files

- `dev/online.py`: `dream_return`; `uni_lookahead(..., gamma)`, `uni_act(..., gamma)`.
- `dev/lifetime.py`: passes `label_gamma`; the move made scored by a full pass when H > 1; H > 1 with `meal_size` refused.
- `dev/cuda_lib.py`: `uni_candidates` refuses a batch that differs from the last forward pass.
- `dev/analyze_lookahead.py`: `table` (R1), `record` and `read` (R2).
- `dev/tests/test_lookahead.py`: the return by hand, the refusal, the first imagined hunger.

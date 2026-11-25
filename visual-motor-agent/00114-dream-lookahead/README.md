# vm-00114: The Instinct Looks Ahead in Its Dream

**Status:** Running (step 1, the frozen check)
**Task:** `tasks/00127-dream-lookahead.md` (option (a) of task 00123, the acting rule)
**Source:** `dev/`

## Question

The default agent picks among 16 candidate moves by each one's predicted label one move ahead. Does holding each candidate for 4 ticks in the model's own dream, and scoring it by the dream's 4-step return, help where the one-step choice falls short: poison in the food world (where Dreamer fails) and the plateau in Breakout?

## Method

- **The lookahead** (`online.instinct_h = 4`, task 00029; vm-00096): each candidate is held for 4 ticks. The model imagines ticks 2–4 (Jacobi decoding, J = 2), and each quarter piece imagines only its own block. It needs full windows (`online.ring = 0`).
- **The score** (new, `online.dream_return`): each imagined step's meal or poison is read from the dream's predicted hunger by the stored label's rule. A fall is a meal, with chance fall ÷ hunger before. A rise above .01 is a poison costing the rise. The 4th move's own predicted label closes the return, with γ = `label_gamma`. Before this, a lookahead with the meal label scored only the 4th move's label, which misses a meal or poison inside the dream.
- **A fault fixed on the way:** with the label in the body and H > 1, `lifetime.py` scored the move made by `uni_candidates`. That call reuses the last forward pass, which was the lookahead's (16 windows) and not the policy pass (1), so the C side wrote 16 rows into a 1-row buffer (a heap overrun; the life crashed by tick 4). With H > 1 the move is now scored by a full pass, and `uni_candidates` refuses a batch that does not match the last forward pass.
- **Step 1, frozen:** TT's (vm-00101, food world) and BD's (vm-00103, Breakout stage 1) 60k checkpoints, 12 seeds, 2,000 ticks with learning off in new worlds (seed 100 + s). Each acts three ways: as it lived (ring, H = 1), full windows at H = 1, and full windows at H = 4.
- **Step 2, lives:** arm LA (H = 4, full windows), 12 seeds per world against TT and BD, paired by seed.

## Readings (set before the results)

1. Food world: LA − TT poison per 1k at 40–60k below zero, with food not below TT's.
2. Breakout: LA − BD hit share at 40–60k above zero; ticks to .5.
3. Frozen at 60k agrees in sign with 1 and 2.

## Results

Pending.

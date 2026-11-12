# Task 00099: The Fresh Value at Eight Layers, Twelve Seeds (experiment vm-00088)

**Date:** 2026-11-12
**Status:** Complete (runs 2026-11-12 06:51 to 11:47; written up 2026-11-12)

## Context

- **vm-00086 (task 00097):** eight layers in each quadrant help K4 in every lived measure:
  - poison is lower in 6 of 6 seeds and move CE in 6 of 6;
  - food is +24 ± 34 at 40–60k, not past its interval.
  - With vm-00085's value (the winner's stored prediction), V90D8 passes the bias bar on paper, but it is still about 43% optimistic. Its candidates' spread collapses (0 in two seeds), and it adds no food over K4D8.
- **vm-00087 (task 00098):** a tail recomputed at every update as the mean of the four transformers (V90R) removes the optimism:
  - abs bias is .0043, against V90's .0127, lower in 6 of 6 seeds, and about −20% of the realized label, like K4;
  - the spread does not collapse;
  - at 2 layers it adds no food beyond its interval (+14 ± 30 against K4).
- **Next step, as planned (task 00098):** if depth helped, an 8-layer V90R follows. The owner agreed on 2026-11-12.
- **Six seeds are too few:** every food interval so far is ±20–35 per 1k. Here the 8-layer pair gets 12 seeds.

## Design

| Arm | Label | Tail | Layers | Seeds |
|---|---|---|---|---|
| K4D8 | mean hunger over the next 4 ticks | none | 8 | 1–6 from vm-00086 (RTX 4090), 7–12 new (RTX 3090) |
| **V90RD8** | value, γ 0.9, n 4 | the four's mean prediction of the move made, fresh at every update | 8 | 1–6 on the RTX 4090, 7–12 on the RTX 3090 |

- **The same as vm-00086 and vm-00087 otherwise:** four quadrant transformers of 8 layers, gain, log summaries, the ring, the egocentric touch world, `online.future_k = 4`, 60k ticks, under MPS.
- **Every seed pair runs on one card:**
  - seeds 1–6 on the 4090 (where vm-00086's K4D8 ran);
  - seeds 7–12 on the 3090.
  - GPU lives are not bitwise reproducible, so pairing by seed within a card keeps the card out of the difference.
- **18 new lives:**
  - 6 on the 4090, about 1.5 h;
  - 12 on the 3090, about 3–3.5 h;
  - then the frozen tests, without MPS, two per card.
- **No code change:** `online.value_refresh = 1` with `model.n_layer = 8`. `eval_hunger.py`'s interval table is extended to 12 seeds.

## Measures

- **Lived, 40–60k:** food, poison, mean hunger, overrides, move CE, smoothness. Paired by seed, V90RD8 against K4D8 (12 pairs).
- **Across experiments, 6 seeds** (not readings): V90RD8 against V90R (depth with the fresh value) and against V90D8 (the fresh tail at 8 layers).
- **Frozen:** `eval_hunger.py value` (bias, correlation, candidates' spread) and `eval_hunger.py chain` (food, aim by the gap, lead over a blind walk with a perfect grab).

## The readings, set before the results

1. **Plumbing:** all 18 lives run 60k ticks; the value is logged within 0–1.
2. **The fresh value holds at depth:** every V90RD8 seed meets vm-00085's bar (abs bias at most .0141, correlation above .5). Its median candidate spread is at least .0005 in every seed, so there is no collapse as in V90D8.
3. **Reach:** V90RD8's aim at 12–24 px is above K4D8's, paired over 12 seeds, by more than its interval.
4. **A chain pays:** V90RD8's lead over blind with a perfect grab is above K4D8's by more than its interval.
5. **More food (the main one):** V90RD8's lived food at 40–60k is above K4D8's by more than its interval, over 12 seeds.
6. **Poison is still seen:** V90RD8's lived poison is not above K4D8's by more than its interval.

If reading 5 fails while 2 holds, the value is right but does not pay at this horizon. The next step then is the label's scale (the squashing seen with γ 0.97), not more seeds.

## Implementation

- [x] `eval_hunger.py`: the t table up to 12 seeds.
- [x] `00088-value-refresh-depth/` (README, Makefile); the queue (scratchpad `q88.sh`).
- [x] Runs, frozen tests, write-up.

## Verification

- `eval_hunger.py show` over vm-00086's and vm-00088's files gives K4D8 12 seeds, paired with V90RD8 by seed.

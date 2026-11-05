# Task 00095: One Transformer, Bigger Instead of Longer (experiment vm-00084)

**Date:** 2026-11-04
**Status:** Complete

## Context

- vm-00082 gave one transformer over the whole small world a context of 8 to 32 ticks.
  - The longer context did not raise food (16 to 25 per 1k against K4's 70), and the longest ate more poison.
  - No arm moved in a chain toward the food: each did no better than its own moves replayed blind, with a perfect grab when a food was one move away.
- The owner (2026-11-04): "lets plan experiment where instead of increasing context window we increase model parameters (size)".
- vm-00058 (2026-10-31) changed the width of the 4-quadrant model (QNA, an older build, 3 seeds).
  - From 0.4 to 2.8 times the parameters, foraging did not visibly change, but prediction improved with size, above all of the move.
  - Depth was never varied: every model so far has 2 layers.
- A single transformer that forages poorly is where size has the most room to show. It is also the setting where vm-00082 measured the context, so the two are directly comparable.

## Design

**The model:** vm-00082's: one transformer over the whole view (`model.band_n = 0`, `online.pieces = 0`), ctx 8, K4, the instinct, the ring, the default 64 x 64 world, six seeds, 60k ticks.

**Arms (heads stay 32 wide; d_ff is 4 d):**

| Arm | d_model | Layers | Heads | d_ff | Parameters | Ticks/s, 400 ticks beside the fact runs |
|---|---|---|---|---|---|---|
| C8 (vm-00082, not rerun) | 128 | 2 | 4 | 512 | 976,546 | 10.4 |
| D4 | 128 | 4 | 4 | 512 | 1,373,090 | 6.6 |
| D8 | 128 | 8 | 4 | 512 | 2,166,178 | 3.4 |
| W2 | 256 | 2 | 8 | 1024 | 2,737,314 | 9.9 |
| W4 | 512 | 2 | 16 | 2048 | 8,618,146 | 8.6 |

- D8 and W2 are close in size (2.2M and 2.7M) and differ in shape: depth against width.
- W4 is the largest, at 8.8 times C8.
- The input and output tables grow with d, not with d², so the parameter counts include them; they are the counts the life prints.

**Measures:**
1. **Lived:** food, poison and mean hunger over 40–60k, against C8, paired by seed; food by 10k ticks.
2. **Prediction:** patch, body and move cross-entropy over 40–60k, from the metrics.
3. **Frozen:** `eval_hunger.py chain` (seen, blind, blind with a perfect grab; aim by the gap to the food), and `eval_hidden.py aim` at shows 4 and 1, as in vm-00082; C8's rows are vm-00082's.

**Placement:** 24 lives before the fact experiment 00021 (the owner), seeds 1–3 on the 4090 and 4–6 on the 3090, all at once. D8 is the slowest, about a third of C8's speed.

## The readings, set before the results

1. **Plumbing:** a 400-tick life runs at every size, with the parameter counts above.
2. **Bigger helps:** W4's lived food at 40–60k is above C8's by more than its interval.
3. **Depth beats width at equal size:** D8's lived food is above W2's by more than its interval.
4. **Prediction follows size:** W4's patch cross-entropy at 40–60k is below C8's in every seed.
5. **A chain of moves appears:** in W4 or D8, seen food minus blind food with a grab is above zero by more than its interval (in vm-00082 no arm's was).

## Implementation

- [x] No code change: `model.d_model`, `model.n_layer`, `model.n_head`, `model.d_ff` exist; 400-tick lives at every size
- [x] `00084-model-size/`: README, Makefile
- [x] Runs, frozen tests, write-up (seeds 4–6 rerun from tick 0 on the 3090 after a reboot)

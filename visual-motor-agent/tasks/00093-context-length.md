# Task 00093: One Transformer, Context 8 to 32 Ticks (experiment vm-00082)

**Date:** 2026-11-03
**Status:** Complete (vm-00082)

## Context

- The owner (2026-11-03): "maybe 1 transformer model on small world but with lets say 32 ticks context window, maybe with 8, 12, 16, 24 and 32 and compare results - it should be a quick run and I wonder will it help".
- Every model so far has had `ctx = 8`. The age key allows up to 32 (its softmax keeps one lane per age).
- vm-00081 (hidden food) asks how long the agent can remember. A longer context is the most direct way to remember longer, so each final model here is also tested with hidden food.

## Design

**The model:** one transformer over the whole view (`model.band_n = 0`, `online.pieces = 0`): 64 patches, 21 summaries, body and move, 87 tokens per step, 976,546 parameters at ctx 8. Otherwise the default: K4, the instinct, the ring, the default 64 x 64 world.

**Arms:** C8, C12, C16, C24, C32 (`model.ctx`), six seeds, 60k ticks.
- Reference: vm-00073's K4 (4 transformers, ctx 8), not rerun.
- **Note:** the replayed training windows are ctx steps long (8 windows per update), so a longer context also brings more tokens per update: 4 times at 32. This is kept, so each arm is the default life with a longer context. It means a gain cannot be credited to the context alone.

**Speed:** at 300 ticks with the GPU shared, C8 ran at 36.7 ticks/s and C32 at 12.5.

**Placement:** seeds 1–3 on the 4090 and 4–6 on the 3090, all 30 at once, after vm-00081's lives: a C32 life needs about 2 GB of the card for attention, so they do not fit beside other runs.

## Measures

1. **Lived:** food, poison and mean hunger over the last 20k ticks, by context.
2. **Frozen:** `eval_hunger.py chain`: food, poison and aim at 12–24 px.
3. **Memory without having needed it:** `eval_hidden.py aim` with the food hidden after 4 frames and after 1, for every final model. Aim by time since the food vanished should reach further with a longer context.

## The readings, set before the results

1. **Plumbing:** the ring's window and the training windows are ctx steps; a 300-tick life runs at ctx 8 and 32.
2. **Longer helps:** C32's lived food is above C8's by more than its interval.
3. **Memory reaches further:** at hidden food 1, C32's aim 9–16 ticks after the food vanished is above C8's by more than its interval.

## Implementation

- [x] No code change; a short life at ctx 8 and 32
- [x] `00082-context-length/`: README, Makefile; runs queued after vm-00081's lives (GPU memory: the 30 lives need about 15 GB a card)
- [x] Runs, frozen tests, write-up

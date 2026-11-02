# Task 00082: Hunger Ahead, a Chain of Moves (experiment vm-00071)

**Date:** 2026-11-02
**Status:** Complete

## Context

- vm-00069's section "After the write-up": aim at food fades past 2–3 moves though the whole world is in view, and on about 64% of ticks the food is hardly aimed at. A blind path with a perfect one-step grab gets about 60% of the model's food. Nothing in training links a move to a food several moves later.
- The owner (2026-11-02): "write a task for the multi-step hunger prediction and start implementing".
- vm-00023 tried a chain by imagining 4 frames, and failed: the imagined frames could not draw the cursor. Its conclusion 3 names the other road, taken here: predict the hunger several moves ahead from the real window, trained on real futures, with no imagined frames.

## Design

**One change, in the target.** The move token predicts the next body, and the instinct reads that prediction's hunger. With `online.future_k = k`, the target for that hunger is the mean hunger over the next k ticks. Nothing else changes: no new token, table, value or head, the same two-hot loss, and the same read by the instinct, by `gain` and by the other rules. The frozen tools need no change.

- **It is a value of a move under the model's own habits:** what the mean hunger will be if I make this move and go on as I do. As moves get better, the target reflects it, so reach can grow on itself.
- **The mean, and not the hunger at tick t + k.** Eating sooner should score better. Hunger at t + k scores a late meal better, since less time passes after it; the mean does not.
- **Trained on real futures.** The labels are `online.py`'s hindsight machinery, already there for the old command (`goal_k`, `goal_labels`, mode `future`): labels stored with the chunks, the newest training window ending k - 1 ticks back, and chunks decided once their future is known.
- **`future_k = 1` is the default model,** the same rows and the same training, so it is the control.

**What it gives up.** The model no longer predicts next-tick hunger. So with k above 1 body CE is not comparable with k = 1, and the body of a dream would be wrong; self-forcing is refused with it.

**A body-only target, for now.** The general form is every token predicting itself k ticks ahead, in pixels too, which needs a second output table and frames from the future in the batch. That is larger, and it is not needed to answer whether a chain forms. If this works it is the next step, for challenge 3 of §13 in `../ARCHITECTURE.md`.

**A known risk: the signal is small.** Food's relief is only the hunger of the moment, and a single move changes the odds of a meal a little, against a tolerance of .005 and a spread of predictions of .01–.05 (vm-00069). If the arms do not move, the next try is `env.h_rate` at 5 times, so that hunger is 5 times larger, with its own control. Tolerance stays at .005: without it vm-00069's LT lost food.

| arm | `online.future_k` | spans |
|---|---|---|
| **K1** | 1 | the default, the control |
| **K4** | 4 | about the reach the model has now |
| **K16** | 16 | about one meal (food every 17–24 ticks), past the context of 8 |

Six seeds each, 60k ticks, `configs/U_default.json` (4 pieces, gain, channels). Seeds 1–3 on the 4090, 4–6 on the 3090 in every arm, so that arm and GPU are not confounded. Nine runs on each GPU under MPS.

## Measures

Online: food, poison and their share over the last 20k ticks. Frozen, final weights, `eval_hunger.py chain`, 16 worlds, 1,500 ticks: aim by the gap to the food, the blind path, the blind path with a perfect grab. `eval_hunger.py reach`: how often the instinct overrides, and the spread of its predictions.

## The readings, set before the results

Each is an arm against K1, seed by seed, 95% interval on six seeds.

1. **Plumbing:** training gets the right numbers. As first set this asked for the default's weights bit for bit, which cannot be read: two default runs of the same seed on the GPU differ from each other as much as from `future_k = 1` (rms .0166 and .0163, weights about .05), so GPU runs do not repeat exactly. Changed before any main result: inside a real life, for `future_k = 1` every batch's labels equal the next hunger, and for any k every stored chunk's labels equal the mean of the next k hungers of the stream. **Met,** 2,500 ticks: k = 1 exact, in all 2,493 batches; k = 4 and 16 within 6e-8.
2. **Reach grows (the main one):** aim at gaps of 12–24 pixels, weighted by ticks, is above K1's by more than its interval.
3. **A chain pays:** the model's lead over blind and a perfect grab is above K1's by more than its interval.
4. **Food:** over the last 20k ticks food is above K1's by more than its interval, with poison not above K1's by more than its interval.

## Implementation

- [x] `online.py`: `future_k` in `DEFAULT_ONLINE`; `future_labels` and mode `future` in `goal_labels`
- [x] `lifetime.py`: memory and lag from `future_k`; the rows' next hunger from the labels; refuse without the universal model or with self-forcing
- [x] `tests/test_online_memory.py`: the labels
- [x] Reading 1 on the GPU
- [x] `00071-hunger-ahead/`: README, Makefile
- [x] Runs, frozen tests, write-up: K4 meets all four readings; K16 meets reach and chain, and fails on poison (`../00071-hunger-ahead/README.md`)

## Verification

Unit tests of the labels, including k = 1, a truncated episode and a sooner meal. Reading 1 inside real lives, with the lag and the stored labels of k = 16.

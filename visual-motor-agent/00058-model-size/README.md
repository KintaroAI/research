# vm-00058: Bigger and Smaller Transformers

**Date:** 2026-10-31
**Status:** Complete (pending tag)
**Source:** `exp/vm-00058` (code: `visual-motor-agent/dev/`)
**Outputs:** `~/data/research/visual-motor-agent/exp_00058/`
**Task plan:** [`../tasks/00066-model-size.md`](../tasks/00066-model-size.md)

## Goal

The default model's size (d = 128, 2 layers, 4 heads, d_ff = 512) was never tested in the piece design. What do more parameters and heads give,
and what does less cost?

## Method

`configs/U_default.json` (vm-00054's QNA) at half and twice the width. Heads scale with width, so a head stays 32 wide; layers stay 2. Seeds 1–3, 60k
ticks, all six together on the RTX 4090 with MPS. No code changed. The default arm is QNA's seeds 1–3 from vm-00054: the same settings, an older
build, not rerun.

| arm | d_model | heads | d_ff | parameters | ticks/s, six together |
|---|---|---|---|---|---|
| **S** | 64 | 2 | 256 | 367,362 | 31 |
| default | 128 | 4 | 512 | 929,282 | |
| **B** | 256 | 8 | 1024 | 2,642,946 | 25 |

The input and output tables grow with d, not d², so twice the width gives 2.84× the parameters.

## Results

### R1. The last 20k ticks, per 1k

| arm | food | seeds | poison | patch CE | body CE | move CE |
|---|---|---|---|---|---|---|
| S | 37.5 | 48.2, 35.5, 28.9 | 3.2 | 0.805 | 0.249 | 0.889 |
| default | 40.5 | 21.9, 40.5, 59.3 | 4.0 | 0.760 | 0.233 | 0.251 |
| B | 33.4 | 52.0, 12.7, 35.5 | 3.7 | 0.737 | 0.225 | 0.115 |

### R2. By 10k ticks, food / poison, mean of the seeds

| arm | 0–10k | 10–20k | 20–30k | 30–40k | 40–50k | 50–60k |
|---|---|---|---|---|---|---|
| S | 15.9 / 17.3 | 24.0 / 11.3 | 31.1 / 6.9 | 34.5 / 5.4 | 33.8 / 3.7 | 41.1 / 2.7 |
| B | 18.9 / 18.6 | 28.0 / 12.1 | 29.6 / 7.6 | 32.5 / 4.7 | 32.7 / 4.2 | 34.1 / 3.2 |

## Analysis

- **Size barely changes foraging.** The three means lie inside a seed spread of 13–59, and vm-00055 puts the default's own eight seeds at 34.7 ± 10.2.
  Three seeds exclude only a large effect, in either direction.
- **Prediction does follow size,** in every seed: patch CE falls from 0.805 to 0.737. The move's CE falls most, 0.889 to 0.115, a factor of 8.
- **The instinct does not need the better prediction.** S predicts the move far worse and forages the same. It may be that what foraging needs from
  the model is the next hunger, whose CE moves little (0.249 to 0.225).
- **Speed:** B is only ~20% slower than S with 7.2× the parameters, with all six sharing the card. Alone they may differ more.

## Conclusions

1. From 0.4× to 2.8× the parameters, foraging does not visibly change. The default is not a bottleneck, and half its width would do.
2. Width buys prediction, above all of the move. If an arm needs a good dream or text, size may matter there; this test is foraging only.

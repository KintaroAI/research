# vm-00028: Is the Gap the Data?

**Date:** 2026-10-26
**Status:** Complete
**Source:** `exp/vm-00028` (code: `visual-motor-agent/dev/`)
**Outputs:** `~/data/research/visual-motor-agent/exp_00028/`
**Task plan:** [`../tasks/00034-text-data-test.md`](../tasks/00034-text-data-test.md)

## Goal

The universal model in a text lifetime reaches 1.77–1.80 nats per letter (vm-00026/27), the classical GPT of the same
size 1.615; the input map is not the cause (vm-00027). The two were also trained on different data: GPT on random
windows, the lifetime on its stream's newest window and page-aligned chunks from memory. Swap the data both ways.

## Hypothesis

If the data causes the gap, GPT on a lifetime's batches falls to about 1.8, and the universal model on random windows
reaches about 1.62.

## Method

- **GPT on a lifetime's data** (`reference/char_gpt.py --data lifetime`): the same model, its batches built as a
  lifetime builds them (a stream advancing one page per update, the newest window and 7 page-aligned chunks from
  memory, decided 11 pages after they end).
- **The universal model on GPT's data** (`train_text_offline.py`): arm B's model and GPU update, 8 windows of 8 pages
  per update from random letter offsets, moves and drives 0.
- 100k updates, seeds 1–3; the same 208 val windows as vm-00026.

## Log

### 2026-10-26
- **Implemented** `--data lifetime` and `train_text_offline.py` (task 00034); ran 16:27–16:46, scored until 16:51.

## Results

Val nats per letter (the rows marked vm-00026 for comparison):

| Model | Data | 10k | 20k | 30k | 50k | 100k | best |
|---|---|---|---|---|---|---|---|
| GPT (vm-00026) | random windows | 1.67–1.69 | 1.615–1.635 | 1.62–1.63 | 1.66–1.70 | 1.80–1.92 | **1.615** (20–30k) |
| GPT | a lifetime's | 2.04–2.23 | 1.76–1.79 | 1.70–1.72 | 1.77–1.81 | 1.94–1.99 | **1.696–1.716** (30–35k) |
| universal B, offline | random windows | 2.40–2.45 | 2.04–2.08 | 1.92–1.94 | 1.83–1.85 | 1.78–1.79 | **1.780–1.788** (80–100k) |
| universal B (vm-00026) | a lifetime's | 2.39–2.40 | 2.03–2.10 | 1.94–1.98 | 1.82–1.89 | 1.80–1.82 | **1.796–1.804** |

The offline universal model's train loss at 90–100k: 1.55–1.57 (the lifetime's: 1.56–1.59).

## Analysis

- **The data explains a part of the gap for GPT, none of it for the universal model.** On a lifetime's batches GPT is
  0.08–0.10 nats per letter worse at its best (1.70 against 1.615) and overfits sooner and harder (1.94–1.99 at
  100k). The universal model on random windows is only 0.01–0.02 better than in its lifetime (1.78 against 1.80):
  the same curve, step for step.
- **At equal data the universal model is behind GPT by 0.10 (a lifetime's data) to 0.17 (random windows)** and learns
  much more slowly: at 20k updates 2.04–2.08 against 1.615–1.635. It has not overfit by 100k; GPT passes its best
  within 20–35k.
- So the remaining gap is in the model itself, in what is left after the input map (vm-00027) and the data (here):
  the 256-bin two-hot output against 65 classes; the factorised positions (66 slots + 8 steps against 512 positions);
  the initialisation (`W_in` and the output layer); the body and move tokens; the loss over tokens (each token's mean
  over its values). None is text-specific.

## Conclusions

1. **The lifetime's data is not why the universal model is behind:** trained on GPT's random windows it ends in the
   same place. (For GPT the lifetime's page-aligned data is worse than random windows, by 0.08–0.10 nats per letter.)
2. **The gap, 0.17 nats per letter at equal data, is in the model,** and it shows as slow learning rather than
   overfitting.

## Next Steps

- An ablation ladder in the PyTorch reference, offline on random windows: start from the universal model (it should
  reproduce 1.78) and change one thing at a time toward the classical GPT (the output: 65 classes instead of 256 bins;
  absolute positions; GPT's initialisation; no body and move tokens), to find which costs the 0.17. Whatever it is, the
  fix goes into the model for every world.

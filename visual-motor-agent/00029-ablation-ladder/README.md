# vm-00029: The Ablation Ladder — the Input Map Is the Gap

**Date:** 2026-10-26
**Status:** Complete
**Source:** `exp/vm-00029` (code: `visual-motor-agent/dev/`)
**Outputs:** `~/data/research/visual-motor-agent/exp_00029/`
**Task plan:** [`../tasks/00035-ablation-ladder.md`](../tasks/00035-ablation-ladder.md)

## Goal

At equal data the universal model is 0.17 nats per letter behind a classical GPT of the same size on Tiny Shakespeare
(vm-00028). Find which design choice costs it, one switch at a time.

## Method

`reference/text_ladder.py`: one PyTorch model trained offline on random windows (8 × 512 letters per update, AdamW
3e-4, gradient clip 0.5, the same Transformer), with five switches from the universal model (arm B of vm-00026,
initialised from the C++ model's own weights) to the classical GPT: `tokens` (with or without the body and move
tokens), `input` (the linear map `W_in` or an embedding table per letter), `pos` (slot + step or one per position),
`output` (256 bins per value or one 65-way softmax), `init` (the C++ initialisation, content at unit scale and
positions at 0.02, or GPT-2's, both at 0.02). Both ends checked: R0 at birth 5.5628 nats per letter as the C++ model,
at 3k updates 2.569 (C++ offline 2.572); R6 at 3k 1.997 (char_gpt.py 2.00). A sixth input, `bins`: every token's
embedding is the sum over its values of a learned embedding per (value slot, bin), weighted two-hot, the output's bins
used on the input. 30k updates, the same 208 val windows (every letter but each window's first).

## Log

### 2026-10-26
- **Ran** R0–R6 17:04–17:13, R7, R8 and a second seed of R0 and R5 17:15–17:20.

## Results

Val nats per letter:

| Rung | Switch from R0 | 5k | 10k | 15k | 20k | 30k |
|---|---|---|---|---|---|---|
| R0 (seeds 1, 2) | none: the universal model | 2.51, 2.55 | 2.39, 2.37 | 2.19, 2.17 | 2.05, 2.09 | 1.93, 2.00 |
| R1 | GPT's initialisation | 2.57 | 2.33 | 2.12 | 2.02 | 1.89 |
| R2 | output: 65 classes | 2.51 | 2.42 | 2.20 | 2.15 | 2.01 |
| R3 | letter tokens only | 2.53 | 2.46 | 2.25 | 2.05 | 1.95 |
| R4 | absolute positions | 2.56 | 2.50 | 2.31 | 2.04 | 1.91 |
| **R5** (seeds 1, 2) | **input: a letter table** | 1.92, 1.94 | 1.74, 1.73 | 1.66, 1.69 | 1.65, 1.66 | **1.62, 1.64** |
| R6 | all five: the classical GPT | 1.80 | 1.69 | 1.63 | 1.63 | 1.63 |
| **R7** | **input: bin embeddings** | 1.93 | 1.75 | 1.68 | 1.65 | **1.63** |
| **R8** | **input: bin embeddings, GPT's initialisation** | 1.75 | 1.62 | **1.57** | 1.58 | 1.61 |

## Analysis

- **The input map is the whole gap.** An embedding table for letters alone takes the universal model from 1.93–2.00
  to 1.62–1.64 at 30k updates, level with the classical GPT (1.63). No other switch comes close: GPT's initialisation
  gains 0.04, the body and move tokens and the factorized positions cost nothing measurable, and a 65-way softmax
  output is *worse* than the bins (2.01).
- **Why the linear map fails on letters:** a letter's grey level enters as level × one vector, so all 65 letters lie
  on one line in the embedding space; the Transformer must pull them apart before it can use them, and learns about
  6× more slowly (R0 needs 30k updates for what R5 reaches by 5k: 1.92).
- **vm-00027's one-hot control said otherwise because its output changed too** (very likely): a one-hot letter's input
  is a table, but its output was 81 yes/no values and its loss the mean over them, so each letter's error counted
  about 1/81 of what it does here. Its early speed (2.05 against 2.40 at 10k) was the table's gain; its plateau the
  output's cost.
- **The universal fix, bin embeddings, works as well as a table** (R7: 1.63), and with GPT-2's small initialisation it
  **beats the classical GPT** (R8: 1.57 at 15k, against 1.62; it then overfits). It is the same rule for every token:
  each value is split two-hot over the bins, as its prediction is, and each (value slot, bin) has its own learned
  embedding. For values on bin centres (letters) it is a lookup table; for continuous values (pixels, hunger) a learned
  piecewise-linear function; today's `W_in` is the special case of embeddings that grow linearly with the bin.

## Conclusions

1. **The universal model's text gap was its input map,** not its output, positions, extra tokens or data. My first
   hypothesis was right; the one-hot control (vm-00027) was confounded by its output.
2. **Bin embeddings on the input** (the output's bins, used on the input), with small initialisation, make the
   universal model as good as or better than a classical GPT of the same size on text, with no text-specific part.

## Next Steps

- Bin-embedding input in the C++ and CUDA universal model (a config switch, tested against the PyTorch reference),
  then: the text lifetime (B) and the circle world (vm-00021's default lifetime) with it, to check that vision gains
  or at least does not lose. If both hold, it becomes the default for every world.

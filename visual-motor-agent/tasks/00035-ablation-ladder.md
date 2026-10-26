# Task 00035: The Ablation Ladder, Universal Model to GPT (experiment vm-00029)

**Date:** 2026-10-26
**Status:** Complete (results in vm-00029)

## Context

- vm-00028: at equal data (random windows, offline) the universal model is 0.17 nats per letter behind a classical GPT
  of the same size on Tiny Shakespeare (1.78 against 1.615) and learns much more slowly. Not the input map alone
  (vm-00027), not the data (vm-00028).
- What still differs, none of it text-specific: the body and move tokens, the input map, the positions (slot + step
  against one per position), the output (256 bins per value, a token's loss the mean over its values, against one
  softmax over 65 letters), the initialisation (content at unit scale, positions at 0.02, against both at 0.02).

## The ladder (`reference/text_ladder.py`)

One PyTorch model, offline on random windows (as vm-00028), with five switches: `tokens` (universal | letters),
`input` (linear | table), `pos` (factorized | absolute), `output` (bins | classes), `init` (cpp | gpt). All first: the
universal model, initialised from the C++ model's own weights; all second: the classical GPT.

Checked at both ends: R0 (all universal) at birth 5.5628 nats per letter, as the C++ model; at 3,000 updates 2.569
(the C++ model offline: 2.572). R6 (all GPT) at 3,000 updates 1.997 (char_gpt.py: 2.00).

## Runs

30k updates each (GPT is at its best by then), val every 5k, seed 1:

| Rung | Switch from R0 |
|---|---|
| R0 | none (the universal model) |
| R1 | init gpt |
| R2 | output classes |
| R3 | tokens letters |
| R4 | pos absolute |
| R5 | input table |
| R6 | all (the classical GPT) |

Then combinations of whatever helps, and the same from R6 backwards if no single switch explains the gap.

## Results (vm-00029)

- A letter table alone closes the gap (R5: 1.62–1.64 at 30k against R0's 1.93–2.00 and GPT's 1.63); GPT's
  initialisation gains 0.04; tokens and positions nothing; a 65-way output is worse (2.01).
- The universal version, bin embeddings on the input (a learned embedding per value slot and bin, two-hot), matches
  the table (R7: 1.63) and with GPT-2's initialisation beats the classical GPT (R8: 1.57 at 15k).
- Next: bin-embedding input in the C++ / CUDA model, then the text and circle lifetimes.

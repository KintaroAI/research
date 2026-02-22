# Experiment: Sequential Modular Arithmetic Evaluation

**Date:** 2026-02-21
**Status:** Complete
**W&B:** [seq-eval group](https://wandb.ai/kintaroai-dot-com/gpt2-cuda) (group: seq-eval)

## Goal

Establish the sequential modular arithmetic evaluation protocol: train on 4
algorithmic tasks in sequence (same checkpoint carried forward), evaluating
retention of all previous tasks after each phase. This provides a baseline
measurement of catastrophic forgetting in the dense grokking model.

## Hypothesis

A small transformer trained sequentially on related modular arithmetic tasks
will exhibit significant catastrophic forgetting — learning each new task but
losing performance on all prior tasks. The dense 2-layer model has no mechanism
to protect learned representations from being overwritten.

## Method

### Tasks (in training order)

| # | Operation | Formula |
|---|-----------|---------|
| 1 | Addition | `(a + b) mod 97` |
| 2 | Subtraction | `(a - b) mod 97` |
| 3 | Multiplication | `(a * b) mod 97` |
| 4 | Squared sum | `(a^2 + b^2) mod 97` |

All tasks: p=97, vocab_size=101, format `BOS a OP b EQ c EOS EOS`,
50/50 train/val split, seed 42 (4704 train / 4705 val equations each).

### Setup
- Hardware: RTX 4090
- Model: 2-layer, 4-head, 128-dim transformer (~414K params, grokking preset)
- Loss masking: `-p 4` (answer position only)
- Full-batch training: B=4703, T=8
- 50,000 steps per phase (50K epochs over full dataset)
- lr=0.001, wd=1.0, beta2=0.98, seed=1337

### Protocol

For each phase: train on the new task's training set starting from the
previous phase's checkpoint, save a new checkpoint, then evaluate it on
all tasks' val sets using eval-only mode (`-n 0 -v 1`).

See `dev/EVAL.md` for the full executable protocol.

## Results

### Evaluation Matrix (val loss, answer position only)

| Phase | Trained on | val_add | val_sub | val_mul | val_sq_sum |
|-------|------------|---------|---------|---------|------------|
| 1 | add | **0.044** | - | - | - |
| 2 | sub | 9.456 | **0.128** | - | - |
| 3 | mul | 11.242 | 11.176 | **0.050** | - |
| 4 | sq_sum | 9.881 | 9.834 | 9.633 | **0.061** |

Random-guess baseline for 97-class answer: `ln(97) = 4.575`.

### Key observations

- **Diagonal (learning):** The model learns each task to low loss (0.04-0.13).
  Addition and multiplication converge faster than subtraction and squared sum.
- **Off-diagonal (retention):** Complete catastrophic forgetting. After training
  on each new task, all prior tasks jump to loss ~9-11, worse than random
  guessing (4.58). The model isn't just forgetting — it's actively anti-learning
  prior tasks.
- **Worse-than-random:** Val losses of 9-11 vs random baseline of 4.58 suggest
  the model is confidently predicting wrong answers for prior tasks, not
  outputting uniform distributions.
- **Training dynamics:** Val loss during training is noisy with intermittent
  spikes (e.g., subtraction showed spikes to 1.4), typical of grokking dynamics
  with weight decay.

## Analysis

The dense 2-layer model exhibits total catastrophic forgetting on sequential
modular arithmetic. This is expected — with only ~414K parameters and full-batch
training that completely overwrites the loss landscape, there is no mechanism
for protecting prior task representations.

This establishes a clear baseline for the sequential eval protocol. Any
architectural modification that shows improved retention (lower off-diagonal
values) has found a way to maintain more stable internal representations.

The fact that forgetting is worse-than-random (not just returning to chance)
is interesting. The model's representations for one operation actively interfere
with others — the OP token embedding and associated pathways are being
repurposed rather than extended.

## Conclusions

- Sequential eval protocol works end-to-end: data generation, 4-phase training,
  cross-task eval, all logged to W&B
- Dense baseline shows complete catastrophic forgetting (loss ~10 vs random 4.58)
- The protocol is ready for comparing architectural variants (banded sparsity,
  sort layer, etc.) — any retention above this baseline is a positive signal
- Subtraction is harder to learn than addition/multiplication (0.128 vs 0.044-0.050)

## Next Steps

- [ ] Run sequential eval with banded sparsity variants
- [ ] Run sequential eval with sort layer
- [ ] Investigate whether interleaved training (mixing tasks) prevents forgetting
- [ ] Try elastic weight consolidation (EWC) or similar continual learning techniques

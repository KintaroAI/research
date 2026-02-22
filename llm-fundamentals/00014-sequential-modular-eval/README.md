# Experiment: Sequential Modular Arithmetic Evaluation

**Date:** 2026-02-21
**Status:** Complete
**W&B:** [seq-eval group](https://wandb.ai/kintaroai-dot-com/gpt2-cuda) (groups: seq-eval, seq-eval-nop, seq-eval-nop-10k)

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

### Ablation: All-position loss (no `-p 4`)

To isolate the effect of position masking, the same protocol was re-run without
`-p 4` (loss computed over all 8 sequence positions, not just the answer). Two
step counts were tested: 1k and 10k steps per phase.

#### 1k steps, all positions

| Phase | Trained on | val_add | val_sub | val_mul | val_sq_sum |
|-------|------------|---------|---------|---------|------------|
| 1 | add | **1.632** | - | - | - |
| 2 | sub | 2.624 | **1.827** | - | - |
| 3 | mul | 2.566 | 2.561 | **1.639** | - |
| 4 | sq_sum | 2.619 | 2.619 | 2.594 | **1.613** |

#### 10k steps, all positions

| Phase | Trained on | val_add | val_sub | val_mul | val_sq_sum |
|-------|------------|---------|---------|---------|------------|
| 1 | add | **1.518** | - | - | - |
| 2 | sub | 3.059 | **1.524** | - | - |
| 3 | mul | 2.757 | 2.748 | **1.473** | - |
| 4 | sq_sum | 2.369 | 2.371 | 2.363 | **1.585** |

#### Cross-condition comparison

| | 50k, `-p 4` | 1k, all pos | 10k, all pos |
|---|---|---|---|
| Diagonal (learning) | 0.04–0.13 | 1.6–1.8 | 1.47–1.58 |
| Off-diagonal (forgetting) | 9.5–11.2 | 2.5–2.6 | 2.4–3.1 |
| Forgetting gap (off - diag) | ~9 nats | ~1.0 nats | ~1.0–1.5 nats |
| Worse than random? | Yes (2x) | No | No |

### Key observations (all-position ablation)

- **Position masking amplifies forgetting.** With `-p 4`, all model capacity is
  focused on the answer token — the only thing that differs between tasks. This
  makes every learned representation task-specific and fully overwritable.
  Without it, 7/8 positions share identical structure (`BOS a OP b EQ _ EOS EOS`)
  which acts as an anchor.
- **More training = more forgetting.** At 10k steps the diagonal improves
  slightly (1.5 vs 1.6) but the off-diagonal gets worse (2.4–3.1 vs 2.5–2.6).
  Longer training increases specialization and overwriting of prior task features.
- **Never worse-than-random.** Without position masking, off-diagonal losses
  stay below the all-position random baseline (~4.6). The shared sequence
  format provides a floor that prevents the catastrophic anti-learning seen
  with `-p 4`.

## Analysis

The dense 2-layer model exhibits total catastrophic forgetting on sequential
modular arithmetic when using position-masked loss (`-p 4`). With only ~414K
parameters and full-batch training that completely overwrites the loss landscape,
there is no mechanism for protecting prior task representations.

The all-position ablation reveals that the extreme forgetting (worse-than-random)
in the original run is largely an artifact of position masking. When the model
must predict all tokens (not just the answer), it learns shared sequence structure
that partially transfers across tasks. The forgetting gap drops from ~9 nats to
~1.0–1.5 nats.

This has implications for the evaluation protocol: `-p 4` produces a cleaner
signal for measuring architectural differences (the answer token is the only
task-discriminative position), but the worse-than-random effect means the
baseline is unusually harsh. Architectural comparisons should use `-p 4` for
consistency with the grokking literature, but the all-position results confirm
the forgetting is real and not just a measurement artifact.

The fact that forgetting is worse-than-random with `-p 4` (not just returning
to chance) is interesting. The model's representations for one operation actively
interfere with others — the OP token embedding and associated pathways are being
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

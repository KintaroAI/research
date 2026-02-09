# Experiment 10: Delayed Generalization on Modular Arithmetic

## Overview

Reproduce the "delayed generalization" phenomenon from [Generalization Beyond Overfitting on Small Algorithmic Datasets](https://mathai-iclr.github.io/papers/papers/MATHAI_29_paper.pdf) (Power et al., ICLR 2021 MathAI Workshop).

**Goal**: Train a small transformer on modular addition (mod 97) using our C/CUDA training infrastructure and observe generalization emerging long after the model has memorized the training set.

---

## Background

The phenomenon:
1. First **memorizes** the training data (training loss → 0 within ~10³ steps)
2. Then, much later, **generalizes** to unseen data (validation loss drops after ~10⁴–10⁵ steps)

The gap between memorization and generalization can be 100–1000× in optimization steps. Weight decay is the key enabler.

---

## Task: Modular Addition (mod 97)

We learn the binary operation:

```
a + b ≡ c  (mod 97)
```

- **Vocabulary**: 97 residues (0–96) + 4 special tokens (OP=97, EQ=98, BOS=99, EOS=100) → **101 tokens** (padded to 128 for llm.c)
- **Equation format**: 8-token sequences: `BOS a OP b EQ c EOS EOS`
- **Total equations**: 97 × 97 = **9,409**
- **Training split**: 50% of equations → ~4,705 train, ~4,704 val
- **Data format**: Equations are concatenated into a flat token stream in llm.c shard format
- With T=8, each batch row contains exactly one equation
- With B=4688, each step processes 4688/4704 = 99.7% of training equations (full-batch)

---

## Model Architecture (from the paper)

| Parameter | Value |
|-----------|-------|
| Type | Decoder-only Transformer (GPT-2 style) |
| Layers (n_layer) | 2 |
| Embedding dim (n_embd) | 128 |
| Attention heads (n_head) | 4 |
| Head dimension | 32 |
| FFN hidden dim | 512 (4 × n_embd) |
| Max sequence length (block_size) | 8 |
| Vocabulary size | 101 (padded to 128) |
| Positional encoding | Learned |
| Total parameters | ~400K |

---

## Training Configuration

| Parameter | Value |
|-----------|-------|
| Optimizer | AdamW |
| Learning rate | 1e-3 |
| β₁, β₂ | 0.9, 0.98 |
| Weight decay | 1.0 |
| Batch size B | 4688 (full-batch) |
| Sequence length T | 8 |
| Max steps | 100,000 |
| Validation every | 100 steps |
| Training code | C/CUDA (adapted from dev/) |

**Why full-batch?** The paper uses full-batch gradient descent (all training
equations in every step). With mini-batches (e.g. B=512), the task-relevant
gradient signal is too weak relative to weight decay (wd=1.0), preventing
generalization. B=4688 is the largest multiple of 16 that satisfies
B×T+1 ≤ train_tokens (the matmul kernel requires B×T to be a multiple of 128).

---

## Experiment Plan

### Phase 1: Data & Model Setup
- [ ] Generate 9,409 modular addition equations
- [ ] Write train/val splits as llm.c shard files (.bin)
- [ ] Create model.bin with tiny architecture (2L, 128d, 4H)

### Phase 2: Training (C/CUDA)
- [ ] Build training binary from adapted dev/ C code
- [ ] Train with AdamW (wd=1.0, lr=1e-3) for 100K steps
- [ ] Log train loss + val loss every 100 steps

### Phase 3: Analysis
- [ ] Plot training/validation loss curves (expect delayed second descent in val loss)
- [ ] Measure: steps to train convergence vs steps to val convergence
- [ ] Compare loss curves with wd=0 (expect NO generalization)

### Phase 4 (stretch): Ablations
- [ ] Vary training fraction: 30%, 40%, 50%, 60%, 70%
- [ ] Vary weight decay: 0.0, 0.1, 0.5, 1.0
- [ ] Try other operations: modular subtraction, multiplication

---

## Expected Results

Based on the paper:
- **Training loss**: Reaches ~0 within ~1,000 steps (memorization)
- **Validation loss**: Stays high until ~10,000–50,000 steps, then drops sharply (generalization)
- **Weight decay is critical**: Without it, generalization may not occur within the budget
- The "double descent" in validation loss should be visible

---

## How It Works with llm.c

The equations are concatenated as a flat token stream:
```
BOS a1 OP b1 EQ c1 EOS EOS BOS a2 OP b2 EQ c2 EOS EOS ...
```

With T=8, each row in a batch sees exactly one equation:
- Position 0 (BOS): predicts `a` (random, ~log(97) loss)
- Position 1 (a): predicts `OP` (trivial)
- Position 2 (OP): predicts `b` (random)
- Position 3 (b): predicts `EQ` (trivial)
- Position 4 (EQ): predicts `c = (a+b) mod 97` ← **the hard task**
- Position 5 (c): predicts `EOS` (trivial)
- Position 6 (EOS): predicts `EOS` (trivial)

The model's generalization on position 4 is what creates the delayed generalization signature.

**Why T=8?** Padding from 5 to 8 ensures T is divisible by 4, which is required by the CUDA `float4` vectorized loads in the softmax kernels. The BOS/EOS tokens add minimal overhead — the model learns to predict them trivially.

---

## Key Modifications to dev/ Code

The C training code is the **unmodified dev/ version** with experiment-specific flags at invocation:
1. **Full-batch training** (`-b 4688`): processes (nearly) all training equations per step, matching the paper's full-batch gradient descent. B=4688 is the largest multiple of 16 that satisfies the dataloader's B×T+1 ≤ num_tokens constraint.
2. **Weight decay** (`-w 1.0`): strong regularization per the paper (dev default is 0.0)
3. **β₂** (`-a 0.98`): Adam beta2 per the paper (dev default is 0.999)
4. **No text generation** (`-s 0`): sampling disabled (tokens are abstract symbols, not text)

---

## Files

```
00010-delayed-generalization/
├── README.md                # This file — experiment plan and results
├── generate_dataset.py      # Generate modular arithmetic data in llm.c format
├── create_model.py          # Create model.bin with tiny architecture
├── Makefile                 # Build and run targets
└── src/
    ├── train_gpt2_fp32.cu   # Training code (identical copy from dev/)
    └── llmc/
        ├── dataloader.h     # Data loading (from dev/)
        ├── tokenizer.h      # Tokenizer (from dev/, unused with -s 0)
        ├── utils.h          # File I/O utilities (from dev/)
        └── rand.h           # RNG (from dev/)
```

## Dependencies

- CUDA toolkit (nvcc + cuBLAS)
- Python 3.8+ with a virtual environment activated (e.g., `source venv/bin/activate`) and the following packages installed:
    - `numpy` (`pip install numpy`)
    - `torch` (`pip install torch`)

## Quick Start

```bash
cd 00010-delayed-generalization

# 1. Generate dataset (ensure venv is active)
python3 generate_dataset.py

# 2. Create model checkpoint (ensure venv is active)
python3 create_model.py

# 3. Build training binary
make train

# 4. Train (expect ~100K steps, watch for val loss dropping)
#    B=4688 = full-batch training (paper requirement)
./train -e model.bin -i data/train.bin -j data/val.bin -t 8 -b 4688 -n 100000 -l 0.001 -w 1.0 -a 0.98 -s 0 -o log.txt
```

---

## Experiment Log

### Attempt 1: Mini-batch training (B=512)

**Config**: `./train -e model.bin -i data/train.bin -j data/val.bin -t 8 -b 512 -n 1000000 -l 0.001 -w 1.0 -a 0.98 -s 0`

**Result**: ❌ No generalization even after 1M steps. Training loss drops (memorization), but validation loss never decreases.

**Analysis**: The paper uses **full-batch gradient descent** (all ~4704 training equations per step). With B=512, each step sees only 10.9% of training data. Combined with strong weight decay (1.0), the task-relevant gradient (only position 4 out of 8 per equation) is too weak relative to regularization. Weight decay wins the tug-of-war, preventing the model from developing generalized representations.

### Attempt 2: Full-batch training (B=4688)

**Config**: `./train -e model.bin -i data/train.bin -j data/val.bin -t 8 -b 4688 -n 10000000 -l 0.001 -w 1.0 -a 0.98 -s 0`

**Result**: ❌ No generalization. Same behavior as B=512.

**Analysis**: Full-batch alone is not sufficient. Remaining differences from the paper:

1. **Loss dilution from padding tokens**: The paper's equation format is `a ○ b = c` (5 tokens). Our format `BOS a OP b EQ c EOS EOS` (8 tokens, padded for T%4==0) means the actual task signal (predicting `c` at position 4) is only **1/8 = 12.5%** of the total gradient. Worse, positions 0 and 2 have **irreducible loss** (~log(97) each) that generates large, permanent noise gradients. In the paper's 5-token format, position 1 (`○` → `b`) is the only irreducible position, so the task signal is ~1/4 = 25% — **2× stronger**.

2. **Weight tying**: The llm.c code hardcodes `wte` as the output projection (weight tying). The paper likely uses separate input embeddings and output projection. Weight tying constrains the embedding to serve double duty, potentially hindering the modular arithmetic representations.

3. **Loss on all positions vs answer-only**: Many grokking reproductions compute loss **only on the answer token** (`c`). Our llm.c training code computes loss uniformly on all 8 positions. The irreducible noise from positions 0 and 2 dominates the gradient, especially with the 1/(B×T) averaging that makes each position's contribution very small.

### Remaining ideas (not yet tried)

- **Mask loss to answer-only**: Modify `fused_classifier_kernel3` (or the `dlosses` input) to zero out loss at all positions except position 4 (the answer). This would match how most grokking reproductions work.
- **Reduce sequence to T=4**: Use format `a b c PAD` (drop OP/EQ/BOS/EOS) so the task signal is 1/4 of the total, and only 2/4 positions have irreducible loss.
- **Lower weight decay**: Try wd=0.1–0.5 to compensate for the diluted task gradient.
- **Remove weight tying**: Add a separate `lm_head` parameter to the checkpoint format and CUDA code.
- **Port to PyTorch**: Bypass llm.c constraints entirely — implement in PyTorch with loss masking, no weight tying, and the paper's exact 5-token format.

---

## Status

🔴 **Blocked** — grokking not reproduced. The llm.c training infrastructure imposes constraints (loss on all positions, weight tying, T%4 padding) that dilute the task signal. See Experiment Log above for details and next steps.
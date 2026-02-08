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

- **Vocabulary**: 97 residues (0–96) + 2 special tokens (OP=97, EQ=98) → **99 tokens** (padded to 128 for llm.c)
- **Equation format**: 5-token sequences: `a OP b EQ c`
- **Total equations**: 97 × 97 = **9,409**
- **Training split**: 50% of equations → ~4,705 train, ~4,704 val
- **Data format**: Equations are concatenated into a flat token stream in llm.c shard format
- With T=5, each batch row contains exactly one equation

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
| Max sequence length (block_size) | 5 |
| Vocabulary size | 99 (padded to 128) |
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
| Batch size B | 512 |
| Sequence length T | 5 |
| Max steps | 100,000 |
| Validation every | 100 steps |
| Training code | C/CUDA (adapted from dev/) |

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
a1 OP b1 EQ c1 a2 OP b2 EQ c2 a3 OP b3 EQ c3 ...
```

With T=5, each row in a batch sees exactly one equation:
- Position 0: sees `a` → predicts `OP` (trivial)
- Position 1: sees `a OP` → predicts `b` (random)
- Position 2: sees `a OP b` → predicts `EQ` (trivial)
- Position 3: sees `a OP b EQ` → predicts `c = (a+b) mod 97` ← **the hard task**
- Position 4: sees `a OP b EQ c` → predicts next `a'` (random)

The model's generalization on position 3 is what creates the delayed generalization signature.

---

## Key Modifications to dev/ Code

The C training code is the **unmodified dev/ version** with experiment-specific flags at invocation:
1. **Weight decay** (`-w 1.0`): strong regularization per the paper (dev default is 0.0)
2. **β₂** (`-a 0.98`): Adam beta2 per the paper (dev default is 0.999)
3. **No text generation** (`-s 0`): sampling disabled (tokens are abstract symbols, not text)

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
#    (ensure venv is active when running train)
./train -e model.bin -i data/train.bin -j data/val.bin -t 5 -b 512 -n 100000 -l 0.001 -w 1.0 -a 0.98 -s 0 -o log.txt
```

---

## Status

🔬 **Setting up** — data generation & C code adaptation in progress.
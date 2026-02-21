# Experiment 12: Conditions for Grokking on Modular Arithmetic

## Hypothesis

Classic grokking (delayed generalization) on modular addition requires specific conditions. We systematically vary four factors — batch size, weight decay, model depth, and loss masking — to identify which are necessary and sufficient.

## Background

Experiment 00010 reproduced grokking with full-batch training, answer-only loss masking (`-p 4`), and weight decay. But it left open which factors are truly necessary. Power et al. (2022) computed loss on all positions and emphasized weight decay as the key enabler. Our `-p 4` masking is a deviation from their setup. This experiment tests whether grokking occurs under conditions closer to the original paper and maps the interaction between these factors.

## Setup

### Data

Modular addition mod 97, using `gen_modular_data.py` (new in dev/):
- 9,409 total equations: a + b = c (mod 97)
- 50/50 train/val split: 4,704 train, 4,705 val
- Token encoding: 101 tokens (97 residues + OP + EQ + BOS + EOS), padded to 128
- Sequence format: `[BOS, a, OP, b, EQ, c, EOS, EOS]` (8 tokens)
- Seed: 42

For 50/50 2-way split (needed for full-batch), used 00010's `generate_dataset.py`.
For 3-way splits, used the new `gen_modular_data.py`.

### Model

Created via `create_model.py --preset grokking --prime 97` (new CLI flags added):
- 2-layer default: 2L, 128d, 4H, 410K params
- 1-layer variant: `--n-layer 1`, 212K params
- Same random init copied for each wd=0/wd=1 pair within a run

### Training

All runs: lr=0.001, beta2=0.98, 50K steps, eval every 500 steps, seed 1337.

### Theoretical loss bounds (no -p4, all-position loss)

With loss averaged over 7 prediction positions:
- Positions 1, 3 (random operands a, b): irreducible loss = ln(97) = 4.575
- Positions 2, 4, 6, 7 (deterministic OP, EQ, EOS, EOS): learnable to 0
- Position 5 (answer c): 0 if learned, ~4.575 if random

| Scenario | Average loss |
|----------|-------------|
| Perfect generalization (answer learned) | (4.575 + 4.575) / 7 = **1.307** |
| Random on answer | (4.575 + 4.575 + 4.575) / 7 = **1.961** |
| Memorized (operands + answer on train) | **< 1.307** (predicts train-specific patterns at pos 1,3) |

---

## Results

### Phase 1: With -p 4 (answer-only loss masking)

All runs with `-p 4` generalize immediately. No grokking observed.

#### Mini-batch (B=512), 3-way split

| wd  | Val @ s500 | Final val (s50K) | Final test | Pattern |
|-----|-----------|-----------------|------------|---------|
| 0.0 | 0.0001    | 0.0001          | 0.0001     | Instant generalization |
| 1.0 | 0.116     | 0.116           | 0.116      | Oscillating, unstable |

Mini-batch B=512 with wd=0 generalizes by step 500. Weight decay causes instability with periodic val spikes up to ~3.5.

#### Full-batch (B=4703), 2-layer, 50/50 split

| wd  | Val @ s500 | Final val (s50K) | Pattern |
|-----|-----------|-----------------|---------|
| 0.0 | 0.0015    | 0.0000          | Instant generalization |
| 1.0 | 0.0305    | 0.0441          | Oscillating ~0.04 with spikes to ~4.8 |

Same story at full-batch. wd=0 reaches near-zero val by step 500. wd=1 oscillates but never shows delayed onset.

#### Full-batch (B=4703), 1-layer, 50/50 split

| wd  | Val @ s500 | Final val (s50K) | Pattern |
|-----|-----------|-----------------|---------|
| 0.0 | 0.0011    | 0.0001          | Instant generalization |
| 1.0 | 0.0181    | 0.0119          | Stable ~0.01, no memorization phase |

Reducing to 1 layer doesn't change the picture. With -p 4, the model finds the general solution immediately regardless of depth or weight decay.

### Phase 2: Without -p 4 (loss on all positions)

Removing -p 4 dramatically changes the dynamics.

#### Full-batch (B=4703), 1-layer, 50/50 split, no -p4

| wd  | Train (s50K) | Val (s50K) | Gap   | Pattern |
|-----|-------------|-----------|-------|---------|
| 0.0 | 1.057       | 2.511     | +1.45 | **Pure memorization** |
| 1.0 | 1.091       | 1.384     | +0.29 | **Immediate (partial) generalization** |

**wd=0**: Train converges to 1.057 (below theoretical 1.307, confirming memorization of input patterns). Val climbs monotonically from 1.64 to 2.51 over 50K steps — textbook overfitting. The model memorizes training equations but gets progressively worse on unseen ones.

**wd=1**: Train converges to ~1.09, val stabilizes at ~1.38 (close to the 1.307 theoretical minimum for full generalization). This means the model HAS learned the addition function — val loss near the irreducible minimum confirms generalization at the answer position. The 0.08 excess over 1.307 comes from slight memorization of training-specific input patterns at positions 1 and 3. But there is no delayed onset: val reaches ~1.34 by step 1000 and stays there.

---

## Analysis

### Why no grokking?

We tested 8 conditions spanning batch size, weight decay, model depth, and loss masking. None exhibited classic grokking (memorization phase followed by delayed generalization). The dynamics fall into three regimes:

1. **Instant generalization** (all -p 4 runs; no-p4 wd=1): The model finds the general addition algorithm quickly. With -p 4, the task signal is 100% of the gradient and the function-learning problem is straightforward for even a 1-layer transformer. Without -p 4 but with wd=1, weight decay provides enough regularization pressure that the model generalizes at the answer position while the aggregate loss is dominated by irreducible components.

2. **Pure memorization** (no-p4, wd=0): Without regularization and with diluted gradients, the model memorizes training data and never discovers the general algorithm. Val loss keeps climbing — the model is moving further from generalization over time.

3. **Unstable oscillation** (-p 4, wd=1, larger models): Weight decay creates a dynamic equilibrium at sharp minima. The model oscillates between good generalization and catastrophic forgetting with periodic loss spikes.

### The role of -p 4

Task-position masking is the dominant factor. With -p 4, the optimization landscape is clean: 100% of the gradient is task-relevant, and the model finds the general solution in <500 steps regardless of other settings. Without -p 4, the task signal is diluted to 1/7 of the gradient and large irreducible-loss gradients at positions 1 and 3 dominate.

### Comparison to 00010

Experiment 00010 reported "grokking achieved" with -p 4, B=4688, wd=1.0 after 100K steps (train=0.03, val=0.03). Our results are consistent — both train and val reach near-zero — but we show this isn't delayed generalization: it happens within the first 500 steps. The remaining 99.5K steps are just the model sitting at the solution with occasional instability spikes from wd=1.0.

### Comparison to Power et al. (2022)

The original paper computed loss on all positions and found weight decay critical for grokking. Our no-p4 results partially support this: wd=0 memorizes while wd=1 generalizes. However, we don't see the *delayed* onset — wd=1 generalizes early (val=1.34 by step 1000). The gap between memorization and generalization that defines grokking may require:
- Longer training (Power et al. sometimes needed >10^5 steps)
- Different learning rate / optimizer dynamics
- Specific initialization conditions
- The model being in a memorization basin first, then escaping

### Why our setup resists grokking

The most likely explanation: our small models (1-2 layers, 128d) combined with Adam's adaptive learning rates are efficient enough to find the general solution directly, without passing through a memorization phase. Grokking may require the model to be large enough relative to the task that memorization is the "easy" solution, with generalization requiring a qualitative phase transition in the learned representations (e.g., discovering circular/Fourier features for modular arithmetic). Our models may be small enough that the general solution IS the easy solution.

---

## Runs Summary

All runs: p=97, lr=0.001, beta2=0.98, 50K steps, seed=1337.

| # | Layers | B    | wd  | -p 4 | Split    | Train (final) | Val (final) | Grokking? |
|---|--------|------|-----|------|----------|--------------|------------|-----------|
| 1 | 2      | 512  | 0.0 | yes  | 3-way    | ~0           | 0.0001     | No — instant generalization |
| 2 | 2      | 512  | 1.0 | yes  | 3-way    | ~0.17        | 0.116      | No — unstable oscillation |
| 3 | 2      | 4703 | 0.0 | yes  | 50/50    | ~0           | 0.0000     | No — instant generalization |
| 4 | 2      | 4703 | 1.0 | yes  | 50/50    | ~0.25        | 0.0441     | No — oscillating with spikes |
| 5 | 1      | 4703 | 0.0 | yes  | 50/50    | ~0           | 0.0001     | No — instant generalization |
| 6 | 1      | 4703 | 1.0 | yes  | 50/50    | ~0.03        | 0.0119     | No — stable, no memorization |
| 7 | 1      | 4703 | 0.0 | no   | 50/50    | 1.057        | 2.511      | No — pure memorization |
| 8 | 1      | 4703 | 1.0 | no   | 50/50    | 1.091        | 1.384      | No — immediate generalization |

---

## Code Changes

This experiment used the new tooling added to dev/ in the prior commit:
- **`gen_modular_data.py`**: Modular arithmetic data generator with 3-way splits, multiple operations (add/mul), configurable prime/seed/fractions
- **`create_model.py`**: CLI flags (`--preset grokking --prime P --n-layer L`, etc.) so one script creates both TinyStories and grokking models

No changes to CUDA training code. All experiments used existing `train` binary from dev/.

## Reproduction

```bash
cd llm-fundamentals/dev
source venv/bin/activate

# Generate 50/50 data (for full-batch runs)
python ../00010-delayed-generalization/generate_dataset.py \
    --prime 97 --train-frac 0.5 --seed 42 --output-dir data/modular

# Create model (copy for paired wd=0/wd=1 runs)
python create_model.py --preset grokking --prime 97 --n-layer 1 -o model_grok.bin
cp model_grok.bin model_grok_wd0.bin
cp model_grok.bin model_grok_wd1.bin

make train

# Run 7: no -p4, wd=0 (pure memorization)
./train -e model_grok_wd0.bin \
    -i data/modular/train.bin -j data/modular/val.bin \
    -t 8 -b 4703 -n 50000 -v 500 \
    -l 0.001 -w 0.0 -a 0.98 -s 0 -q 1337

# Run 8: no -p4, wd=1 (immediate generalization)
./train -e model_grok_wd1.bin \
    -i data/modular/train.bin -j data/modular/val.bin \
    -t 8 -b 4703 -n 50000 -v 500 \
    -l 0.001 -w 1.0 -a 0.98 -s 0 -q 1337
```

## Status

Complete. Grokking (delayed generalization) was NOT observed under any tested condition. The -p 4 masking makes the task too easy (instant generalization), and without it the model either memorizes (wd=0) or generalizes immediately (wd=1) without the delayed transition phase.

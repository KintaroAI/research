# Experiment: Formal Language Tasks (Eval Roadmap 2b)

**Date:** 2026-02-21
**Status:** Complete

## Goal

Implement and validate formal language tasks for the sequential evaluation
protocol (Eval Roadmap Phase 2b). Four tasks — parity, copy, reverse, majority
— test structurally different algorithmic primitives (counting, identity mapping,
position reversal, thresholding) using a binary alphabet with exhaustive
enumeration. This complements Phase 2a (modular arithmetic) by testing whether
catastrophic forgetting patterns hold across qualitatively different task types.

## Hypothesis

Copy, reverse, and majority should be learnable by a 2-layer transformer in the
grokking regime (full-batch, exhaustive dataset, weight decay). Parity (XOR over
8 bits) may be harder due to its inherent non-linearity. The multi-position loss
flag (`-P`) should enable training on copy/reverse where 8 output positions all
need gradient.

## Method

### Infrastructure changes

1. **`gen_formal_data.py`** — new data generator for all 4 tasks
2. **`-P` flag in `train_gpt2_fp32.cu`** — multi-position loss range
   `[task_position, task_position_end]` inclusive, backward-compatible
3. **`formal` preset in `create_model.py`** — block_size=24, vocab_size=101,
   2-layer/4-head/128-dim
4. **`EVAL.md`** — Phase 2b protocol with exact commands

### Task designs

| Task | Sequence layout | T | `-p`/`-P` | Examples |
|------|-----------------|---|-----------|----------|
| Parity-8 | `BOS b1..b8 EQ ans EOS` + 8 EOS pad | 20 | `-p 9` | 256 |
| Copy-8 | `BOS x1..x8 SEP x1..x8 EOS EOS` | 20 | `-p 9 -P 16` | 256 |
| Reverse-8 | `BOS x1..x8 SEP x8..x1 EOS EOS` | 20 | `-p 9 -P 16` | 256 |
| Majority-9 | `BOS t1..t9 EQ ans EOS` + 6 EOS pad | 20 | `-p 10` | 512 |

All tasks: binary alphabet (0/1), vocab_size=101, 50/50 train/val split, seed 42.
Special tokens: BOS=99, EOS=100, EQ=98, SEP=97. T=20 uniform across all tasks
(required for sequential eval since B,T are locked on first forward pass).

### Training setup

- Model: 2-layer, 4-head, 128-dim transformer (~413K params, formal preset)
- Full-batch: B=127 (parity/copy/reverse), B=255 (majority)
- 50,000 steps per task, lr=0.001, wd=1.0, beta2=0.98, seed=1337
- Val eval every 1000 steps

## Results

### Standalone learnability (each task trained from fresh model)

| Task | Val loss (typical) | Learned? | Notes |
|------|-------------------|----------|-------|
| Parity-8 | 0.693 | No | Converges to ln(2), coin-flip baseline |
| Copy-8 | 0.03-0.07 | Yes | Near-perfect by step 1000 |
| Reverse-8 | 0.03-0.06 | Yes | Slightly noisier than copy, occasional spikes |
| Majority-9 | ~0.0001 | Yes | Essentially solved by step 1000 |

### Training dynamics

**Copy-8:** Val loss drops to 0.006 by step 1000. Steady state fluctuates around
0.03-0.07 with occasional spikes (up to ~2.0). Attention heads can implement
identity mapping across the separator directly.

**Reverse-8:** Similar trajectory to copy but noisier — more frequent spikes to
1.5-2.0. Reverse requires cross-position attention patterns rather than simple
identity, making the learned solution less stable during training.

**Majority-9:** The easiest task. Val loss reaches 0.000006 by step 1000 and
stays near zero throughout. Thresholding (is sum > 4?) is a simple computation
for attention + MLP.

**Parity-8:** The model briefly begins memorizing around step 5-6k (val loss
dips to ~0.05) but then collapses back to the coin-flip baseline (0.693). XOR
over 8 bits is a notoriously hard function for gradient-based learning — it
requires representing a high-order interaction that doesn't decompose into
simpler features.

### Multi-position loss (`-P` flag) validation

The `-P` flag works correctly for copy and reverse:
- Config table shows `task_position_end = 16`
- Log confirms: `loss masking enabled: positions 9..16 contribute to loss/gradient`
- Gradient weight: `1/(B * 8)` per position (8 output positions)
- Backward-compatible: single-position tasks (`-p 9` only) show
  `task_position_end = 9` (normalized from -1 to match `-p`)

## Analysis

3 of 4 formal language tasks are learnable by the 2-layer model in the grokking
regime. The difficulty ordering is: majority (easiest) > copy > reverse > parity
(not learned). This matches theoretical expectations:

- **Majority** is a threshold function — a single attention head aggregating
  counts into MLP can solve it.
- **Copy** requires identity attention patterns (output position i attends to
  input position i), which is a natural induction head.
- **Reverse** requires position-crossing attention (output i attends to input
  N+1-i), which needs the model to learn position arithmetic.
- **Parity** requires computing XOR — a high-order boolean function with no
  useful lower-order approximation. The model can memorize the training set
  briefly but cannot find a generalizing circuit within 50k steps.

The infrastructure is validated: `gen_formal_data.py`, the `-P` flag, and the
`formal` model preset all work end-to-end. The sequential eval protocol
(Phase 2b in EVAL.md) is ready to run with copy, reverse, and majority. Parity
may need to be excluded or given significantly more training steps.

## Conclusions

- Copy, reverse, and majority are confirmed learnable — ready for sequential eval
- Parity-8 does not learn in 50k steps at these hyperparameters
- Multi-position loss (`-P` flag) works correctly for copy/reverse
- The formal language task suite provides structurally diverse tasks that
  complement the modular arithmetic suite (Phase 2a)

## Next Steps

- [ ] Run full 4-phase sequential eval (parity -> copy -> reverse -> majority)
  - Consider excluding parity or giving it 200k+ steps
- [ ] Compare forgetting patterns between formal tasks and modular arithmetic
- [ ] Test banded sparsity on formal language tasks
- [ ] Investigate whether parity-8 can be learned with different hyperparameters
  (lower weight decay, higher lr, more steps, larger model)

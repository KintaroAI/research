# 00001 — Soft-WTA Cell Implementation

**Date:** 2026-03-15
**Status:** In Progress

## Context

PLANNING.md defines a soft-WTA prototype with 5 components: prototypes, competition,
winner update, usage-counter anti-collapse, and match threshold. No code exists yet —
column.py and main.py are both TODO.

## Deliverables

1. `dev/column.py` — SoftWTACell class
   - Prototype vectors (m × n), initialized on unit sphere
   - Forward: dot-product similarity → softmax(temperature) → probabilities
   - Update: Hebbian pull on winner, usage-gated learning rate
   - Anti-collapse: usage counters (EMA of win frequency) gate plasticity
   - Match threshold: poor matches recruit least-used unit instead
   - State save/load

2. `dev/main.py` — CLI entry point
   - Synthetic clustered data generation (Gaussian clusters on unit sphere)
   - Online training loop with logging
   - Metrics: unique winners, category purity, match quality
   - Save results.json + cell.pt

3. Tests in `dev/Makefile`
   - Forward produces valid probabilities (sum to 1, non-negative)
   - Anti-collapse: white noise → no single dominant winner
   - Cluster learning: purity improves over training

4. Experiment `00001-soft-wta-cell/` with README and Makefile

## Implementation Details

- Pure PyTorch tensors, no nn.Module (no backprop needed)
- Normalize both inputs and prototypes to unit sphere for stable dot-product similarity
- Usage counter: EMA with configurable decay (default 0.99)
- Usage-gated LR: `effective_lr = lr / (1 + n_outputs * usage[winner])`
- Match threshold recruits dormant (least-used) unit at full LR

## Verification

```bash
cd dev && make test
```

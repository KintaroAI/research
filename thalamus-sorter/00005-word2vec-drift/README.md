# Experiment: Word2vec-Style Drift

**Date:** 2026-03-10
**Status:** In Progress
**Source:** *tagged on completion as `exp/ts-00005`*

## Goal

Replace centroid-averaging update with word2vec-style skip-gram + negative sampling. Key properties borrowed from word2vec:
1. **Pairwise updates** (not centroid averaging)
2. **Sigmoid self-regulation** (focus learning on "mistakes")
3. **Explicit negative sampling** (push dissimilar pairs apart)
4. **Per-dimension coefficients** (each dim updated differently based on peer)

## Motivation

The centroid approach (exp 00002-00003) averages K neighbors' positions and moves toward the centroid. This:
- Updates all dimensions by the same scalar
- Has no explicit repulsion (relies on LayerNorm to prevent collapse)
- Collapses to lines in 2D with small K

Word2vec (Skip-gram) learns embeddings via pairwise pull/push with sigmoid scaling. The sigmoid provides natural curriculum: strong gradients for "wrong" pairs, saturation for "correct" pairs. Explicit negative sampling provides repulsion without relying on normalization.

## Method

### Update Rule

Each tick, for every neuron:

**Positive (1 random neighbor from top-K):**
```python
delta = pos_j - pos_i                           # toward peer
dist² = sum(delta²)
σ_pos = sigmoid(dist²)                          # large when far → needs pull
pos_i += lr * σ_pos * delta                     # pull toward positive peer
```

**Negative (k_neg random neurons):**
```python
delta = pos_neg - pos_i                          # toward negative peer
dist² = sum(delta²)
σ_neg = sigmoid(-dist²)                         # large when close → needs push
pos_i -= lr * σ_neg * delta                     # push away from negative peer
```

### Key Differences from Word2vec

Original word2vec uses **dot product** and updates in the **direction of** the peer vector (`pos_j`). This optimizes angular alignment, not spatial proximity. Our adaptation:
- Uses **Euclidean distance** in the sigmoid instead of dot product
- Updates **toward/away from** peer (`pos_j - pos_i`) instead of **in the direction of** peer (`pos_j`)
- This preserves the spatial update semantics needed for topographic maps

First attempt with dot product + direction-of-peer collapsed into binary clusters (all embeddings aligned/anti-aligned). Euclidean + toward/away works correctly.

### Parameters
- `k`: positive neighbors (precomputed top-K, same as exp 00002-00003)
- `k_neg`: negative samples per positive per tick (default 5)
- `lr`: learning rate
- `dims`: embedding dimensionality

## Log

### Test 1: Dot product version (original word2vec math), D2, K=11, 30k ticks

Collapsed into two clusters (black/white binary split). Dot product + direction-of-peer update optimizes angular alignment, not spatial proximity. Combined with LayerNorm, creates a degenerate solution.

### Test 2: Euclidean version, D2, K=11, k_neg=5, 30k ticks

Some spatial structure forming — dark/light clustering with coherence. No line collapse (unlike centroid drift at D2 K11). The explicit negative sampling prevents the degenerate solutions that centroid averaging falls into.

### Test 3: Euclidean version, D16, K=25, k_neg=5, 30k ticks

Clear K reconstruction. Comparable to centroid drift (exp 00003) at similar tick count.

## Findings So Far

1. **Dot product word2vec collapses**: original skip-gram math (dot product + direction-of-peer) creates binary clusters, not topographic maps. Euclidean distance + toward/away delta is the correct adaptation.
2. **No line collapse at D2 K11**: the explicit negative push prevents the degenerate solutions that centroid averaging produces. This is a direct benefit of negative sampling.
3. **Vectorized negatives**: all k_neg negatives computed in one batch (no Python loop) — much faster than iterating.

## Next Steps

- [ ] Compare convergence speed vs centroid drift at matched parameters
- [ ] Grid search: k_neg ratio, lr, dims
- [ ] Test at 160px and 1024px scale
- [ ] Analyze per-dimension coefficients — do dims specialize more than with centroid updates?
- [ ] Try combining with temporal correlation (random pairs + sigmoid updates)

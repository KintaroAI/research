# Sorter: Design Decisions & Evolution

## What this system does

Given N neurons with temporal activity patterns, discover the spatial arrangement where correlated neurons are neighbors. No labels, no known topology — just temporal signals in, spatial map out.

## Algorithm evolution

### Phase 1: Discrete grid swaps (ts-00001)

**Approach:** Greedy Hebbian drift — pick a neuron, compute centroid of its K nearest neighbors, swap toward it.

**What worked:** GPU acceleration (10-175x), matrix-free top-K (O(nK) memory). K=20-30 is optimal; below K=10, convergence fails.

**Problem:** Discrete swaps create conflicts (two neurons want the same cell), slow convergence, hard to scale.

### Phase 2: Continuous embeddings (ts-00002, ts-00003)

**Approach:** Replace grid positions with float vectors. Each neuron has a learned D-dimensional embedding. Attract toward neighbor centroids.

**What worked:** 5-10x faster convergence than discrete, no swap conflicts, differentiable.

**Problem:** Pure centroid attraction causes dimensional collapse in 2D (converges to 1D lines by ~5k ticks).

**Fix:** High-dimensional embeddings (D>=10) prevent collapse entirely. D=16 is stable to 10M+ ticks. Extra dims are harmless — PCA/UMAP project to 2D for display. LayerNorm (mean=0, std=1 per dimension) keeps embeddings stable indefinitely.

### Phase 3: Skip-gram learning (ts-00005, ts-00007)

**Approach:** Replace centroid averaging with word2vec skip-gram + negative sampling. Dual vectors (W positions, C contexts). Sigmoid objective with explicit negative sampling.

**Why:** Centroid drift encodes one relationship ("move toward neighbors"). Skip-gram encodes richer structure — positive pairs attract, negative samples repel, and the sliding window captures transitive relationships.

**Key finding:** Sliding window over neighbor sentences matters enormously. Given sentence [A, B, C, D], the window generates pairs like (B,C) and (B,D) — not just (A,B). These transitive pairs encode "B's neighbors are also near each other" which is implicit spatial inference. Without it, learning is ~10x slower.

**Dot product vs Euclidean:** Dot product + LayerNorm collapses to binary clusters. Dot product without LayerNorm self-regulates but produces non-spatial structure. Euclidean sigmoid works for spatial sorting. Final choice: dot product skip-gram (encodes similarity in high-D space, project with UMAP for visualization).

### Phase 4: Correlation-based neighbor discovery (ts-00009)

**Approach:** Replace precomputed spatial neighbors with online discovery. Each tick: pick random anchors, sample random candidates, measure temporal correlation, keep only correlated pairs as sentences for the skip-gram learner.

**Why:** Precomputed top-K from known grid is cheating — we already know the layout. Real system (thalamus) has no spatial labels. Only signal: which neurons fire together.

**What worked:** sigma=8, threshold=0.5, k_sample=200, T=100-200 on 80x80 → disparity 0.02-0.04. System discovers full 2D spatial structure purely from stochastic correlation probes.

**Scaling rule:** sigma ≈ grid_size / 10. At 160x160, sigma=16 works (disparity 0.17 at 5k ticks, still converging).

**Key insight:** Signal quality matters more than learner quality. The bottleneck is whether correlation sampling finds real neighbors, not whether the learner can use them.

## Current architecture

```
Signal buffer (N, T)     Gaussian-smoothed random fields
        |                 sigma controls correlation radius
        v
Random sampling           256 anchors/tick, 200 candidates each
        |
        v
Pearson correlation       |corr| > threshold keeps ~10-15 neighbors
        |
        v
Sliding window            [anchor, nb1, nb2, ...] → all (center, ctx) pairs
        |
        v
Skip-gram + neg sampling  Attract positive pairs, repel random negatives
        |
        v
D-dimensional embeddings  Updated via SGD, periodic normalization
        |
        v
UMAP projection           Warm-start, Procrustes alignment for display
```

### Key parameters

| Parameter | Value | Role |
|-----------|-------|------|
| dims | 8 | Embedding dimensionality |
| k_sample | 200 | Random candidates per anchor |
| threshold | 0.5 | Min correlation to accept neighbor |
| signal_T | 100-200 | Signal buffer length |
| signal_sigma | grid/10 | Spatial correlation radius |
| k_neg | 5 | Negative samples per positive |
| lr | 0.001 | Learning rate (reduce for >10k ticks) |
| normalize_every | 100 | L2 normalization frequency |
| window | 5 | Sliding window half-width |

### Infrastructure

- **Async display** (ts-00008): Double-buffered shared memory, pull-based render worker. Training never blocks on visualization. 14x speedup at 1k ticks.
- **UMAP warm start** (ts-00007): Reuse previous projection, 50 epochs instead of default. 27x speedup per frame.
- **Procrustes alignment** (ts-00007): Align each frame to ground truth grid. Disparity metric drops monotonically from ~1.0 (random) to ~0.02 (converged).
- **info.json**: Each run saves command, git hash, all args, and final results for reproducibility.

## Design decisions and why

| Decision | Alternatives considered | Why this one |
|----------|------------------------|--------------|
| Float embeddings, not grid | Discrete grid swaps | 5-10x faster, no conflicts, differentiable |
| D=8 embeddings | D=2, D=16, D=32 | D=2 collapses; D>=8 stable; D=8 is sufficient for 2D maps |
| Skip-gram, not centroid | Centroid attraction | Richer learning signal, handles transitive relationships |
| Sliding window, not anchor-only | Only verified (anchor, neighbor) pairs | 4x better disparity at equal pair count; transitive inference helps |
| Correlation discovery, not precomputed K | Top-K from known grid | Biologically plausible; no topology labels needed |
| Pearson correlation | Cosine similarity, mutual info | Natural for continuous firing rates; robust to scale |
| threshold=0.5 | 0.15, 0.3 | Precision beats recall — fewer clean neighbors > many noisy ones |
| UMAP, not PCA | PCA, t-SNE | High-D structure is nonlinear; UMAP recovers it, PCA fails |
| Async pull-based render | Sync, push queue | Training never blocks; natural best-effort; no backpressure |
| Normalized firing rates | Binary spikes, raw sensor | Good for Pearson correlation; biologically what downstream areas see |

## What's next

See [MODALITIES.md](MODALITIES.md) for multimodal signal design — vision, audio, touch with within-modal spatial smoothing and cross-modal temporal coincidence.

# ts-00016: KNN Hierarchy — Clustering and Multi-Layer Processing

**Date:** 2026-03-15
**Status:** In progress
**Source:** `exp/ts-00016`
**Design:** `dev/KNN_HIERARCHY.md`

## Goal

Implement hierarchical KNN merging: cluster n neurons into m groups via k-means
on learned embeddings, then derive cluster-level KNN lists via frequency-based
selection. This is the compression step needed for feedback loops — without it,
each cycle through KNN → categorizer → output multiplies the working set by c.

Deliverables:
1. k-means clustering on DriftSolver embeddings (post-training, then streaming)
2. Frequency-based cluster KNN selection (knn2)
3. Cluster adjacency graph
4. Visualization of clusters on the grid
5. Streaming cluster maintenance (incremental updates per tick)

## Motivation

ts-00015 showed that KNN lists stabilize (spatial accuracy 0.993–1.000) at
320×320 with sufficient anchor coverage. The embeddings and KNN lists are now
reliable enough to build higher-level structure on top of them.

The immediate need: a compression step for the feedback loop. n=102,400 neurons
each with k=10 neighbors is too many individual relationships to reason about.
Grouping into m clusters (e.g., m=1000 for ~100:1 compression) with cluster-level
KNN lists preserves the essential topology while making the graph tractable.

## Approach

All code in `dev/cluster_experiments.py` — standalone script, loads saved models.
Start with 80×80 grayscale (n=6400). Test three regimes: n>m, n=m, n<m.

### Phase 1: Post-hoc baseline (offline k-means)

Run k-means on saved embeddings from a converged 80×80 model. This is the
"ideal" clustering — full converged embeddings, batch k-means with multiple
restarts. Serves as the quality baseline for streaming approaches.

**Experiments:**
- Load model.npy from a converged gray_80x80_saccades 50k run
- k-means with m ∈ {25, 100, 400, 1600, 6400, 25600}
  - m=25:    256 neurons/cluster (high compression, n>>m)
  - m=100:   64 neurons/cluster (moderate compression)
  - m=400:   16 neurons/cluster (low compression)
  - m=1600:  4 neurons/cluster (near 1:1)
  - m=6400:  1 neuron/cluster (n=m, degenerate — should reproduce original KNN)
  - m=25600: ~0.25 neurons/cluster (n<m, over-specified — what happens?)

**Metrics:**
- Cluster spatial contiguity: mean/max diameter of clusters on the grid
- Cluster size distribution: min, max, mean, std
- knn2 spatial accuracy: do cluster-level KNN lists point to adjacent clusters?
- knn2 vs original KNN agreement: how much structure is preserved?

### Phase 2: Streaming from converged state

Start from the same converged model but initialize clusters randomly (random
centroids or random assignment). Update clusters incrementally as if processing
anchors — simulate the per-tick update loop from KNN_HIERARCHY.md:

1. Sample a batch of "anchors" (neurons whose embeddings just changed)
2. Check each anchor against its current centroid
3. Reassign if drifted past threshold, nudge centroids
4. Recompute knn2 for affected clusters

Run for N iterations and measure convergence to the Phase 1 baseline.
Compare: how many iterations to match offline k-means quality?

### Phase 3: Streaming from scratch

Start with random embeddings AND random clusters (simulating the state at
tick 0 of training). Feed real embedding snapshots from training history
(or interpolate random → converged). Do clusters converge to sensible
structure even when both embeddings and clusters start random?

This tests whether the streaming approach works in the real scenario where
clustering runs simultaneously with training from the beginning.

### Phase 4: Integration test

If phases 1–3 succeed, integrate streaming clusters into an actual training
run (hook into DriftSolver's tick loop). Compare final cluster quality vs
post-hoc clustering on the same model.

## Implementation Plan

`cluster_experiments.py` — standalone script with subcommands:

```
python cluster_experiments.py offline --model model.npy -W 80 -H 80 --m 100
python cluster_experiments.py streaming --model model.npy -W 80 -H 80 --m 100
python cluster_experiments.py from-scratch --model model.npy -W 80 -H 80 --m 100
```

Core functions needed:
1. `kmeans_cluster(embeddings, m)` — batch k-means, returns cluster_ids, centroids
2. `frequency_knn(knn_lists, cluster_ids, m, k2)` — cluster-level KNN selection
3. `cluster_adjacency(knn2, cluster_ids)` — cluster-to-cluster graph
4. `streaming_update(anchors, embeddings, centroids, cluster_ids, threshold)`
   — single-step incremental reassignment + centroid nudge
5. `eval_clusters(cluster_ids, centroids, knn2, width, height)`
   — spatial contiguity, size distribution, knn2 quality metrics
6. `visualize_clusters(cluster_ids, width, height)` — color-coded grid image

## Results

### Run 001: Baseline model (gray saccades 80×80, 50k ticks)

Standard gray_80x80_saccades preset, 50k ticks. Produces converged embeddings
(n=6400, dims=8) and KNN lists (k=10) as input for clustering experiments.

| Metric | Value |
|--------|-------|
| Elapsed | 146s |
| PCA disparity | 0.601 |
| K10 <3px | 96.9% |
| K10 <5px | 100% |
| KNN spatial | 0.997 |
| KNN overlap | 0.710 |

### Run 002: Offline k-means (Phase 1)

Batch k-means on converged embeddings with k-means++ init (5 restarts).
Frequency-based knn2 selection with k2=10.

| m | n/m | empty | size_std | diam_mean | contiguity | knn2_agr |
|---|-----|-------|----------|-----------|------------|----------|
| 25 | 256 | 0 | 28.7 | 25.2 | 1.000 | 1.000 |
| 100 | 64 | 0 | 10.5 | 12.4 | 1.000 | 1.000 |
| 400 | 16 | 0 | 4.0 | 5.8 | 1.000 | 1.000 |
| 1600 | 4 | 0 | 1.8 | 2.2 | 0.997 | 1.000 |
| 6400 | 1 | 0 | 0.0 | 0.0 | 1.000 | 1.000 |

**Findings:**

1. **100% knn2 agreement** across all m — frequency selection perfectly captures
   the original KNN structure at every granularity.
2. **100% contiguity** (0.997 at m=1600) — clusters are spatially contiguous blobs,
   confirming embeddings encode spatial proximity faithfully.
3. **Diameters scale as sqrt(n/m)** — m=25 has ~25px diameter, m=400 has ~6px.
4. **No empty clusters** at any level.
5. **n=m degenerate case** works: each neuron is its own cluster, knn2 reproduces
   original KNN exactly.

### Run 003: Streaming from converged (Phase 2)

Random centroid init on converged embeddings, then iterative streaming updates
(batch_size=256 anchors/iter, threshold=0.5, lr=0.1, 200 iterations).

| m | n/m | Converged by | Reassign rate (early) | Final contiguity | knn2_agr | Baseline agree |
|---|-----|-------------|----------------------|------------------|----------|----------------|
| 25 | 256 | ~160 iters | 5–25/iter | 1.000 | 1.000 | 0.961 |
| 100 | 64 | ~160 iters | 10–25/iter | 1.000 | 1.000 | 0.989 |
| 400 | 16 | ~50 iters | 0–2/iter | 0.995 | 1.000 | 0.997 |
| 1600 | 4 | instant | 0 | 0.991 | 1.000 | 1.000 |

**m=100 trajectory (representative):**

| iter | reassigned | empty | size_std | diam | contiguity | knn2_agr | agree |
|------|-----------|-------|----------|------|------------|----------|-------|
| 0 | 0 | 0 | 39.8 | 14.1 | 0.991 | 1.000 | 0.986 |
| 50 | 18 | 0 | 32.0 | 14.1 | 0.996 | 1.000 | 0.988 |
| 100 | 8 | 0 | 27.5 | 13.3 | 0.999 | 1.000 | 0.989 |
| 160 | 3 | 0 | 24.8 | 12.8 | 1.000 | 1.000 | 0.989 |
| 199 | 9 | 0 | 23.5 | 12.6 | 1.000 | 1.000 | 0.989 |

**Findings:**

1. **Streaming matches offline quality.** All m values reach contiguity ≥0.991 and
   knn2_agreement=1.000 — identical to offline baseline.
2. **Larger clusters need more iterations.** m=1600 (4/cluster) converges instantly
   with 0 reassignments — random init already places nearby embeddings in the same
   Voronoi cell. m=25 (256/cluster) still has reassignments at iter 199.
3. **200 iterations × 256 batch = 51,200 anchor touches (~8× coverage) is sufficient**
   for all m values tested.
4. **Size balance is worse than offline** (std=23.5 vs 10.5 at m=100) — streaming
   doesn't have Lloyd's balancing pressure. Structure quality (contiguity, knn2)
   is unaffected.
5. **Baseline agreement is lower for small m** (0.961 at m=25 vs 1.000 at m=1600) —
   more valid ways to partition into few large clusters than many small ones.

### Run 004: From scratch — random embeddings → converged (Phase 3)

Embeddings interpolate linearly from random to converged over 20 phases (α=0.05
→ 1.0). Clusters start random and stream-update alongside the evolving embeddings.
50 iterations per phase × 256 batch = 256,000 total anchor touches.

| m | Final empty | Final contiguity | knn2_agr | Baseline agree | Notes |
|---|------------|------------------|----------|----------------|-------|
| 25 | 0 | 1.000 | 1.000 | 0.967 | Works perfectly |
| 100 | 3 | 1.000 | 1.000 | 0.990 | 3 clusters died |
| 400 | 142 | 0.964 | 1.000 | 0.996 | 35% clusters died |
| 1600 | **1209** | 0.588 | 1.000 | 0.997 | **75% clusters died** |

**m=100 trajectory:**

| Phase | α | reassign/iter | empty | diam | contiguity | agree |
|-------|------|---------------|-------|------|------------|-------|
| 0 | 0.05 | 25–38 | 0 | 103 | 0.044 | 0.975 |
| 5 | 0.30 | 4–33 | 0 | 110 | 0.058 | 0.980 |
| 10 | 0.55 | 13–57 | 0 | 106 | 0.248 | 0.981 |
| 13 | 0.70 | 18–53 | 0 | 52 | 0.636 | 0.982 |
| 15 | 0.80 | 22–55 | 2 | 28 | 0.908 | 0.985 |
| 17 | 0.90 | 9–42 | 3 | 17 | 0.992 | 0.989 |
| 19 | 1.00 | 6–35 | 3 | 13 | 1.000 | 0.990 |
| Final | 1.00 | — | 3 | 12.9 | 1.000 | 0.990 |

**Findings:**

1. **Streaming clusters successfully track embeddings from random to converged.**
   Contiguity transitions sharply around α=0.75–0.90 — once embeddings have enough
   spatial structure, clusters snap into place within ~100 iterations.

2. **Cluster death is the main failure mode for large m.** At m=400 (16 neurons/cluster),
   35% of clusters die during the chaotic early phases. Small clusters lose all members
   as embeddings shift, and without rebalancing they never recover. At m=25 (256/cluster),
   clusters are large enough to survive the turbulence — zero deaths.

3. **m=100 is the sweet spot for 80×80.** Only 3 clusters die (97% survival), and the
   final quality matches the offline baseline (contiguity=1.000, knn2_agr=1.000).

4. **Cluster death scales with m.** Death rate: m=25→0%, m=100→3%, m=400→35%,
   m=1600→75%. Clusters with fewer members are more fragile during the chaotic
   random→converged transition. With 4 neurons/cluster (m=1600), a single
   embedding shift can empty a cluster permanently.

5. **Balance enforcement is required for high m.** The min_size / split-merge
   logic from KNN_HIERARCHY.md would prevent this — block moves that would empty a
   cluster, and periodically split oversized clusters to backfill dead ones.

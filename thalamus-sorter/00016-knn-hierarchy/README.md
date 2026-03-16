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

TBD

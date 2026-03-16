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

### Phase 1: Post-hoc clustering (offline)

Run k-means on saved embeddings from a converged model. Validate that:
- Clusters are spatially contiguous (members should be nearby on the grid)
- Cluster KNN via frequency selection matches the spatial neighbor structure
- Different m values produce sensible granularity levels

Test with 80×80 grayscale (n=6400) first, then 320×320 (n=102400).

### Phase 2: Streaming maintenance (online)

Integrate clustering into the training loop:
- Initialize clusters after N warmup ticks (e.g., 5000)
- Per-tick: check updated anchors against their centroid, reassign if drifted
- Nudge centroids, patch cluster KNN for affected clusters
- Track cluster stability metrics (reassignment rate, cluster size distribution)

### Phase 3: Derived hierarchy (Approach A)

Use cluster centroids as layer 2 embeddings, frequency-selected knn2 as layer 2
KNN. No additional training — pure derivation from layer 1 output. Validate that
the derived structure captures the same topology at lower resolution.

## Implementation Plan

TBD — discuss before coding.

## Results

TBD

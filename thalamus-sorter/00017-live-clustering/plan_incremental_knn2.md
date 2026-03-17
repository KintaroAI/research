# Plan: Decouple Clustering from Neuron-Level KNN — Incremental knn2

## Context

Clustering currently requires `--knn-track` because `knn2` (cluster-level neighbor graph) is built by pooling neuron-level KNN lists. We want to maintain knn2 incrementally from the same skip-gram pairs that drive training — no bulk init, no KNN dependency, same O(pairs) cost already being paid.

## Core Idea

Each tick, skip-gram produces (anchor, neighbor) pairs that pass correlation threshold. For each pair where anchor and neighbor are in different clusters, check if the neighbor's cluster should be in knn2[anchor's cluster] based on centroid distance. This is exactly how the DriftSolver maintains neuron-level KNN — piggyback on existing work.

## Design

### knn2 format change
- **Before:** `knn2[c]` = k2 neuron indices, mapped to cluster IDs at query time
- **After:** `knn2[c]` = k2 cluster indices + distances. Fixed length, padded with -1

### Init
- `knn2` starts filled with random cluster indices (uniform random, excluding self)
- Distances initialized to infinity
- No bulk computation needed — fills in naturally over first few hundred ticks

### Per-tick knn2 update (inside `_ClusterManager.tick`)

Input: set of (anchor, neighbor) pairs from skip-gram (exposed via `dsolver._last_pairs`)

```
# Deduplicate: for each anchor cluster, collect unique neighbor clusters seen this tick
pair_clusters = {}  # c_a -> set of c_n
for (a, n) in pairs:
    c_a, c_n = cluster_ids[a], cluster_ids[n]
    if c_a != c_n:
        pair_clusters.setdefault(c_a, set()).add(c_n)

# Update knn2 for each cluster that saw cross-cluster pairs
for c_a, neighbor_set in pair_clusters.items():
    for c_n in neighbor_set:
        dist = ||centroid[c_a] - centroid[c_n]||²
        worst_idx = argmax(knn2_dists[c_a])  # farthest current neighbor
        if dist < knn2_dists[c_a][worst_idx]:
            knn2[c_a][worst_idx] = c_n
            knn2_dists[c_a][worst_idx] = dist
```

Key points:
- **Dedup per tick**: each (c_a, c_n) pair checked at most once per tick
- **Fixed k2 length**: always k2 entries, initialized with random clusters
- **Distance-based replacement**: closer centroid replaces farthest current entry

### Exposing pairs from DriftSolver

In `tick_correlation` (drift_torch.py), after computing `mask` and packing pairs:
- Already have `center_flat` and `ctx_flat` (neuron index tensors of correlated pairs)
- Store: `self._last_pairs = (center_flat.cpu(), ctx_flat.cpu())` alongside existing `self._last_anchors`
- Lightweight — these tensors already exist, just saving a reference

### Data structures

```python
# In _ClusterManager:
self.knn2 = np.random.randint(0, m, size=(m, k2))  # cluster indices
self.knn2_dists = np.full((m, k2), np.inf)          # centroid distances
```

### streaming_update candidate selection (simplified)

```python
# Before (neuron indices → cluster mapping):
neighbors = knn2[cur]
valid_neighbors = neighbors[neighbors >= 0]
neighbor_clusters = np.unique(cluster_ids[valid_neighbors])
candidates = np.unique(np.concatenate([[cur], neighbor_clusters]))

# After (direct cluster indices):
neighbors = knn2[cur]
candidates = np.unique(np.concatenate([[cur], neighbors[neighbors >= 0]]))
```

## Changes by file

### `solvers/drift_torch.py`
- In `tick_correlation`: save `self._last_pairs = (center_flat.cpu(), ctx_flat.cpu())` after the pair packing loop
- One new line, minimal overhead (tensors already exist on GPU)

### `cluster_experiments.py`
- `streaming_update_v3_gpu`: change candidate selection to use cluster-index knn2 (2 lines replace 4)
- No other changes needed to this file

### `main.py` — `_ClusterManager`
- `__init__`: remove `frequency_knn` import, add `knn2_dists` array, init knn2 with random clusters
- `init_clusters(embeddings_t)`: remove `knn_lists_np` param, remove `frequency_knn` call, init knn2 random + compute initial distances
- `tick(embeddings_t, anchors_np, pairs, global_tick)`: add `pairs` param. Dedup (c_a, c_n) pairs, update knn2 by centroid distance. Remove old knn_lists-based patching (lines 213-231)
- After splits: recompute knn2_dists for affected rows (centroids changed), no full rebuild needed
- `report()`: pass `knn_lists=None` to eval_clusters, remove `self.knn_lists`
- Remove `self.knn_lists` attribute entirely

### `main.py` — training loop
- Remove `dsolver.knn_k > 0` gate (lines 315-320)
- Pass pairs to cluster_mgr: `pairs = dsolver._last_pairs if hasattr(dsolver, '_last_pairs') else None`
- Remove periodic `cluster_mgr.knn_lists = dsolver.get_knn_lists()` refresh (line 326)

### `main.py` — argparse
- Remove "Requires --knn-track" from `--cluster-m` help
- Change `cluster_k2` default from `dsolver.knn_k` to fixed value (16)

## Complexity

| Operation | Before | After |
|-----------|--------|-------|
| Init knn2 | O(n×K) via frequency_knn | O(m×k2) random fill |
| Per-tick knn2 update | O(size_c × K) per affected cluster | O(unique_cross_pairs × k2) |
| Post-split | O(n×K) full rebuild | O(k2) recompute distances for split clusters |
| Candidate selection | O(k2) + cluster_ids lookup | O(k2) direct |

Everything is O(pairs) or O(m×k2) — no O(n) scans for knn2 maintenance.

## Verification

1. Run 10k test WITHOUT `--knn-track`: `python main.py word2vec --preset gray_80x80_saccades -f 10000 --cluster-m 100 --cluster-hysteresis 0.3 --eval`
2. Verify clusters converge (contiguity > 0.99 by tick 10k)
3. Compare with Run 001 baseline (contiguity 1.000, K10 <3px 96.9%)
4. Run m=640 50k test to verify higher-m convergence

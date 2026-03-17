# Plan: Multi-Cluster Neurons via Ring Buffer

## Context

Boundary neurons between clusters oscillate (A→B→A→B) as embeddings drift.
Each jump is locally correct but the net effect is pointless churn. Instead of
a separate LRU cache, we replace the single cluster assignment with a ring
buffer of recent cluster memberships — a neuron belongs to ALL clusters in its
ring simultaneously.

## Data Structures

```python
# Current: cluster_ids shape (n,) — single int per neuron
# New:
cluster_ids:  (n, max_k) int64, initialized to -1
pointers:     (n,) int64, initialized to 0
```

- `max_k`: CLI param `--cluster-max-k` (default 1 = current behavior, 2-4 for multi)
- `pointers[i]`: index of the current (most recent) cluster in `cluster_ids[i]`
- Primary cluster: `cluster_ids[i, pointers[i]]`
- All clusters: `cluster_ids[i]` where entry >= 0
- On init: `cluster_ids[i, 0] = initial_cluster`, rest -1, `pointers[i] = 0`

## Ring Buffer Jump Mechanics

On jump from `cur` to `best`:

```python
# Advance pointer (wraps around)
pointers[anchor] = (pointers[anchor] + 1) % max_k

# What's being evicted at this slot?
evicted = cluster_ids[anchor, pointers[anchor]]  # may be -1 (empty)

# Write new primary
cluster_ids[anchor, pointers[anchor]] = best
```

O(1) per jump — no shifting, no array copy. The evicted cluster ID tells us
which cluster the neuron is leaving (if not -1), giving a clean hook for
centroid/size updates.

## Changes

### `cluster_experiments.py` — `streaming_update_v3_gpu`

1. **Signature**: add `pointers=None` (when None, cluster_ids is classic single-int)

2. **Reading current cluster** (line 157):
```python
if pointers is not None:
    cur = cluster_ids[anchor, pointers[anchor]]
else:
    cur = cluster_ids[anchor]
```

3. **Candidate selection** — gather neighbors from ALL of neuron's clusters:
```python
if pointers is not None:
    my_clusters = cluster_ids[anchor]
    my_clusters = my_clusters[my_clusters >= 0]
    # Gather knn2 neighbors from all my clusters
    all_neighbors = []
    for mc in my_clusters:
        neighbors = knn2[mc]
        all_neighbors.append(neighbors[neighbors >= 0])
    neighbor_clusters = np.unique(np.concatenate(all_neighbors)) if all_neighbors else np.array([], dtype=np.int64)
    # Filter out clusters neuron is already in (can't "jump" to where you are)
    neighbor_clusters = neighbor_clusters[~np.isin(neighbor_clusters, my_clusters)]
    candidates = np.unique(np.concatenate([[cur], neighbor_clusters]))
else:
    # existing code unchanged
```

4. **Reassignment** (line 192):
```python
if pointers is not None:
    pointers[anchor] = (pointers[anchor] + 1) % cluster_ids.shape[1]
    evicted = cluster_ids[anchor, pointers[anchor]]
    cluster_ids[anchor, pointers[anchor]] = best
    # Update sizes
    if evicted >= 0:
        sizes[evicted] -= 1
        affected.add(evicted)
    sizes[best] += 1
    affected.add(best)
    # Note: cur stays in the ring — neuron still "in" cur
else:
    cluster_ids[anchor] = best
    sizes[cur] -= 1
    sizes[best] += 1
    affected.add(cur)
    affected.add(best)
```

Key difference: `cur` is NOT removed from sizes — only the evicted (oldest)
cluster loses the neuron. This means the neuron remains a member of `cur`
until it gets pushed out of the ring.

5. **Centroid update (nudge)** — members includes all neurons with cluster in ring:
```python
if pointers is not None:
    # members = neurons that have c anywhere in their ring
    members = np.where(np.any(cluster_ids == c, axis=1))[0]
else:
    members = np.where(cluster_ids == c)[0]
```

Optional: weight by position relative to pointer (primary=1.0, older=less):
```python
# Weighted variant (future refinement):
weights = []
for neuron in members:
    age = (pointers[neuron] - np.where(cluster_ids[neuron] == c)[0][0]) % max_k
    weights.append(1.0 / (age + 1))
```

### `main.py` — `_ClusterManager`

1. **`__init__`**: add `max_k=1` param, store it

2. **`init_clusters`**: replace cluster_ids allocation:
```python
if self.max_k > 1:
    self.cluster_ids = np.full((self.n, self.max_k), -1, dtype=np.int64)
    self.cluster_ids[:, 0] = ids_np
    self.pointers = np.zeros(self.n, dtype=np.int64)
else:
    self.cluster_ids = ids_np  # classic (n,) shape
    self.pointers = None
```

3. **`tick`**: pass `pointers=self.pointers` to `_stream_update` calls

4. **`report`**: stability uses primary cluster:
```python
if self.pointers is not None:
    primary = self.cluster_ids[np.arange(self.n), self.pointers]
else:
    primary = self.cluster_ids
```

5. **sizes computation**: count multi-membership:
```python
if self.pointers is not None:
    sizes = np.zeros(self.m, dtype=int)
    for col in range(self.max_k):
        valid = self.cluster_ids[:, col] >= 0
        np.add.at(sizes, self.cluster_ids[valid, col], 1)
else:
    sizes = np.bincount(self.cluster_ids, minlength=self.m)
```

6. **Eval/visualization**: always use primary cluster for these

### `main.py` — argparse + construction

1. **argparse**: `--cluster-max-k` int, default=1
2. **Construction**: pass `max_k` to `_ClusterManager`
3. **Print**: include `max_k={max_k}` in startup message

### `split_largest_cluster_gpu`

When splitting, use primary clusters only. Neurons assigned to the new `dead`
cluster get it written at their next pointer position (same ring buffer
mechanics). Old cluster entries age out naturally.

## max_k behavior

- **max_k=1**: exactly current behavior. Ring has 1 slot, pointer always 0,
  evicted is always -1 on first jump then always the previous cluster.
  Degenerates to single assignment.
- **max_k=2**: neuron belongs to current + previous cluster. On jump A→B,
  neuron is in {A, B}. On jump B→C, A is evicted, neuron is in {B, C}.
- **max_k=3**: neuron belongs to last 3 clusters visited.

## Edge Cases

- **`-1` padding**: ring starts mostly empty (-1). Evicting -1 is a no-op
  (don't decrement sizes). Filter -1 from member queries.
- **Same cluster in multiple ring slots**: possible if neuron visits A, then
  B, then A again. A appears twice in ring. sizes counts it twice — that's
  correct (stronger membership). Centroid sees the neuron once (dedup in
  np.where).
- **max_k=1 compatibility**: all `if pointers is not None` guards ensure
  zero overhead when disabled. Classic `(n,)` cluster_ids preserved.

## Files to modify

- `thalamus-sorter/dev/cluster_experiments.py` — streaming_update_v3_gpu
- `thalamus-sorter/dev/main.py` — _ClusterManager, argparse, construction

## Verification

1. Run 20k with `--cluster-max-k 1` — must match current behavior exactly
2. Run 20k with `--cluster-max-k 2` lr=1.0 h=0.3 — expect fewer jumps, more stable
3. Run 20k with `--cluster-max-k 3` — check contiguity doesn't degrade
4. Compare with Run 035 baseline (lr=1.0, h=0.3, 85% stability)

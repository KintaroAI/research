# Plan: Per-Neuron LRU Cache for Cluster Ping-Pong Suppression

## Context

Boundary neurons between clusters oscillate (A→B→A→B) as embeddings drift.
Each jump is locally correct but the net effect is pointless churn that hurts
stability metrics. We want an LRU cache per neuron tracking recently visited
clusters, so returns to a just-left cluster are suppressed.

LRU size is small (0-4), so simple array iteration — no fancy data structures.

## Design

- `lru_cache`: numpy array `(n, lru_size)` initialized to `-1`
- LRU=0: no array allocated, zero overhead (current behavior)
- LRU=1: stores previous cluster only — prevents immediate A→B→A return
- LRU=2+: deeper history
- On jump from `cur` to `best`: shift LRU right, insert `cur` at position 0
- During candidate evaluation: filter out clusters found in neuron's LRU
- Always keep `cur` in candidate set (neuron can stay put, never forced to jump)

## Changes

### `cluster_experiments.py` — `streaming_update_v3_gpu`

1. **Signature**: add `lru_cache=None` parameter

2. **After candidate list built** (after line 164), filter LRU clusters:
```python
if lru_cache is not None:
    lru_row = lru_cache[anchor]
    valid_lru = lru_row[lru_row >= 0]
    if len(valid_lru) > 0:
        candidates = candidates[~np.isin(candidates, valid_lru)]
        if cur not in candidates:
            candidates = np.concatenate([[cur], candidates])
        if len(candidates) <= 1:
            continue
```

3. **After reassignment** (after `cluster_ids[anchor] = best`, line 192), update LRU:
```python
if lru_cache is not None:
    lru_cache[anchor, 1:] = lru_cache[anchor, :-1]
    lru_cache[anchor, 0] = cur
```

### `main.py` — `_ClusterManager`

1. **`__init__`**: add `lru_size=0` param, store `self.lru_size = lru_size`

2. **`init_clusters`**: allocate after cluster init:
```python
if self.lru_size > 0:
    self.lru_cache = np.full((self.n, self.lru_size), -1, dtype=np.int64)
else:
    self.lru_cache = None
```
Update init print to include `lru={self.lru_size}`

3. **`tick`**: pass `lru_cache=self.lru_cache` to both `_stream_update` call sites (knn mode line 243, incremental mode line 272)

4. **Print in `run_word2vec`**: include `lru={lru_size}` in startup message

### `main.py` — argparse + construction

1. **argparse** (after cluster-centroid-mode): add `--cluster-lru` int, default=0
2. **Construction site**: read `lru_size = getattr(args, 'cluster_lru', 0)`, pass to `_ClusterManager`

### `split_largest_cluster_gpu` — no changes needed

Split reassignments don't go through the LRU — they're structural recovery,
not boundary oscillation. Stale LRU entries from split clusters are harmless
and age out naturally.

## Edge Cases

- **`-1` sentinel**: `lru_row[lru_row >= 0]` filters empty slots; cluster IDs are 0..m-1
- **`cur` in LRU**: possible if neuron visited cur previously (A→B→A, LRU=[B,A]). Guard re-inserts cur after filtering so neuron can stay put
- **All candidates filtered**: with cur re-inserted, at minimum candidates=[cur], triggers existing `len(candidates) <= 1` skip

## Files to modify

- `/home/user/research/thalamus-sorter/dev/cluster_experiments.py` — streaming_update_v3_gpu (lines 124-212)
- `/home/user/research/thalamus-sorter/dev/main.py` — _ClusterManager class (lines 156-427), argparse (~line 1144), construction (~line 816)

## Verification

1. Run 20k with `--cluster-lru 0` — should match current behavior exactly
2. Run 20k with `--cluster-lru 2` — expect higher stability, fewer jumps
3. Run 20k with `--cluster-lru 4` — even more stable but check contiguity doesn't stall
4. Compare all three with lr=1.0 h=0 baseline (Run 032)

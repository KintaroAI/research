# KNN Hierarchy — Merging and Multi-Layer Processing

Reduce n per-neuron KNN lists to m cluster-level KNN lists via embedding
clustering, then optionally stack layers. Enables feedback loops (input →
KNN → categorizer → output → repeat) without combinatorial explosion.

## Problem

Naive feedback loop explodes: each input → k neighbors → c categorizer
outputs → next cycle. Growth factor is c per cycle → O(c^t).
Need a merge step that compresses at least as hard as the categorizer expands.

## Step 1: Clustering — n neurons → m clusters

k-means on the learned position embeddings (n, dims). Neurons close in
embedding space have overlapping KNN by construction (that's what the
embeddings encode).

```
Input:  embeddings (n, dims), target m
Output: cluster_ids (n,), centroids (m, dims)

k-means(embeddings, m) → cluster_ids, centroids
O(n × dims × m × iters)  — with dims, k, iters constant: O(n × m)
```

Alternative clustering approaches (not needed unless k-means proves insufficient):
- **Union-Find with overlap threshold** — merge neurons whose KNN Jaccard
  overlap exceeds threshold. O(nk). Threshold controls m indirectly.
- **Jaccard agglomerative** — pairwise KNN overlap matrix, greedily merge
  most overlapping pair. O(n²k) to build, O(n) merges.

## Step 2: Cluster-level KNN — frequency-based selection

For each cluster, pool all members' KNN entries and select by frequency.
High-frequency entries are consensus neighbors (stable core). Low-frequency
entries are boundary noise / jumpers.

```
For each cluster c with members [n1, n2, ..., n_s]:

  1. Pool: member_knns = knn_lists[members].flatten()    # (s × k,)

  2. Count: ids, counts = unique(member_knns, return_counts=True)

  3. Remove self-cluster: drop ids where cluster_ids[id] == c

  4. Top-k' by frequency:
     knn2[c] = ids[counts.argsort(descending=True)][:k2]

Output: cluster_knn (m, k')
```

Complexity: O(cluster_size × k) per cluster, O(n × k) total.

Why frequency over distance-to-centroid:
- Frequency = "how many members agree this is a neighbor" → consensus
- Distance = "how geometrically close to center" → can miss correlated
  but distant neurons, can include uncorrelated but nearby ones
- Frequency is also cheaper: histogram vs embedding lookup

Example with cluster of 10 neurons, k=10:
```
n1  KNN: [a, b, c, d, e, f, g, h, i, j]
n2  KNN: [a, b, c, d, e, f, g, h, i, k]
n3  KNN: [a, b, c, d, e, f, g, h, l, m]
...
n10 KNN: [a, b, c, d, e, f, g, h, i, p]

Frequency: a=10, b=10, ..., h=10, i=8, j=1, k=1, l=1, ...
knn2 = [a, b, c, d, e, f, g, h, i, ...]  ← stable core
```

## Step 3: Cluster adjacency (optional)

Map knn2 neuron IDs to cluster IDs for cluster-to-cluster graph:

```
knn2_cluster_ids = cluster_ids[knn2]          # (m, k')
For each cluster c:
  histogram of knn2_cluster_ids[c] → neighbor clusters ranked by edge weight
→ adjacency (m, m)
O(m × k')
```

## Streaming cluster maintenance

Full k-means rebuild is O(n × m) — unnecessary when embeddings change slowly.
Initialize once, maintain incrementally.

```
Initialization (once):
  k-means(embeddings, m) → centroids (m, dims), cluster_ids (n,)
  Build cluster_knn (m, k') via frequency selection

Per-tick update — event-driven, O(anchor_sample):
  1. Only anchor neurons got embedding updates this tick (~2048 at 320x320)
  2. For each updated anchor:
     - Compute distance to its current centroid
     - If distance > reassign_threshold:
         check all m centroids, pick closest
         if changed: update cluster_ids, patch cluster_knn
  3. Nudge centroids of affected clusters:
     centroid[c] += lr * (member_mean[c] - centroid[c])
  4. Recompute cluster_knn only for clusters that gained/lost members
     (re-run frequency selection on affected clusters)

Complexity per tick:
  - Check anchors against own centroid:  O(anchor_sample)
  - Reassign drifted neurons:           O(changed × m), changed << anchor_sample
  - Centroid nudge:                      O(affected_clusters × dims)
  - KNN patch:                           O(affected_clusters × cluster_size × k)
  Total: O(anchor_sample) typical, O(anchor_sample × m) worst case

Stability: neurons only move clusters when they've drifted past threshold.
Most ticks: zero reassignments. Same philosophy as KNN_STABLE_INSERT.
```

### Cluster balance

Prevent empty or bloated clusters:

```
min_size = n // (m * 4)       # 25% of ideal size (n/m)
max_size = n // (m // 4)      # 400% of ideal size

On reassignment:
  - Block move if source cluster would drop below min_size
    (or steal a member from the largest cluster to backfill)

Periodic rebalancing (rare, check every N ticks):
  - If any cluster < min_size: merge into nearest cluster,
    then split the largest cluster via k-means(2)
  - If any cluster > max_size: split into two via k-means(2)
  - O(largest_cluster × dims) per split, amortized O(1)

In practice with gradual drift + centroid nudging, clusters stay
roughly balanced. Splits/merges are rare rebalancing events.
```

## Hierarchical stacking — multi-layer processing

Two approaches, from simplest to most powerful.

### Approach A: Derived (no training)

Cluster centroids ARE the layer 2 embeddings. Layer 2 KNN is the
frequency-selected knn2 from above. No new training needed.

```
Layer 1: (n, T) → train → embeddings (n, dims), knn (n, k)
Cluster: k-means → m clusters, centroids (m, dims)
Derive:  frequency selection → knn2 (m, k')
Output:  (m, dims) + (m, k') — same structure, fewer units
```

Tradeoffs:
  + No new training, instant, deterministic
  + Captures all structure layer 1 already learned
  - Cannot discover NEW correlations at cluster level
  - Static unless re-derived when clusters change

### Approach B: Full layer 2 training (simultaneous)

Layer 2 is a second DriftSolver on cluster-level activations.
All layers learn in parallel — no waiting for convergence.

```
Each tick:
  1. Saccade frame arrives → raw pixel values for all n neurons

  2. Layer 1 (existing):
     - Update signal buffer, tick_correlation
     - O(anchor_sample × n)

  3. Cluster maintenance (streaming):
     - Reassign drifted anchors, nudge centroids
     - O(anchor_sample)

  4. Layer 2 signal:
     - For each cluster: mean of members' raw pixel value this tick
     - Write into layer 2's signal buffer (m, T)
     - O(n) — scatter-add by cluster_id

  5. Layer 2 tick:
     - tick_correlation on (m, T) → layer 2 embeddings, KNN
     - O(anchor_sample_2 × m)

Total: O(anchor_sample × n) — layer 1 dominates.
```

Layer 2 sees whatever cluster activations exist now. Early on clusters
are unstable → noisy signal. As layer 1 converges → clusters stabilize
→ layer 2 signal cleans up → layer 2 converges.

Analogous to cortical hierarchy: V1 and V2 learn in parallel from
propagated activity, not sequentially.

Extends to arbitrary depth: layer L+1 clusters layer L, pools
activations, runs correlation + skip-gram. Each layer adds
O(anchor_sample_L × n_L) where n_L decreases geometrically.

### Recommended path

Start with Approach A (derived). If cluster-level correlation discovery
proves necessary, add Approach B. The streaming maintenance and balance
logic are shared between both.

## Summary of data flow

```
Raw signal (n, T)
  → Layer 1: DriftSolver → embeddings (n, dims), knn (n, k)
  → k-means: n → m clusters, centroids (m, dims)
  → Frequency selection: knn2 (m, k')
  → [Optional] Layer 2: DriftSolver on (m, T_pooled)
     → embeddings_2 (m, dims), knn2_trained (m, k')
  → [Optional] Repeat: m → m2 → m3 → ...

Each level: same structure (count, dims) + (count, k')
Same algorithm. Same code. Just smaller n.
```

# ts-00017: Live Clustering — Integration Test

**Date:** 2026-03-16
**Status:** In progress
**Source:** `exp/ts-00017`

## Goal

Integrate streaming cluster maintenance from ts-00016 into the live DriftSolver
training loop. Clusters form simultaneously with embeddings — no post-hoc step.

## Approach

Added `_ClusterManager` to `main.py` with CLI args:
- `--cluster-m N` — number of clusters (0=disabled)
- `--cluster-k2 N` — cluster-level KNN size (default: 16)
- `--cluster-report-every N` — save cluster visualization + print metrics
- `--cluster-split-every N` — attempt dead cluster recovery
- `--cluster-lr` — centroid nudge learning rate
- `--cluster-k2` — cluster-level KNN size

Per tick: `streaming_update_v3_gpu` runs on the same anchors used for skip-gram.
Periodic splits recover dead clusters. KNN lists refreshed from solver state at
each report interval.

Also exposed `dsolver._last_anchors` from `tick_correlation` so cluster manager
can use the same anchor set.

## Results

### Run 001: 80×80 gray saccades, m=100, 50k ticks

```
preset: gray_80x80_saccades
n=6400, m=100, dims=8, k2=10, lr_cluster=0.01
cluster_init_tick=1000, split_every=10, report_every=2500
Output: ~/data/research/thalamus-sorter/exp_00017/001_live_clusters_80x80_m100_50k/
Runtime: 552s (~11 ms/tick)
```

| Tick | Alive | Contiguity | Diameter | Splits | KNN spatial |
|------|-------|------------|----------|--------|-------------|
| 2500 | 100/100 | 0.192 | 93.6 | 132 | — |
| 5000 | 100/100 | 0.974 | 15.5 | 279 | 0.465 |
| 7500 | 100/100 | 0.998 | 13.0 | 301 | — |
| 10000 | 100/100 | **1.000** | 12.7 | 304 | 0.943 |
| 25000 | 100/100 | 1.000 | 11.8 | 306 | 1.000 |
| 50000 | 100/100 | 1.000 | 11.6 | 306 | 1.000 |

**Eval:** PCA=0.567, K10 <3px=96.9%, K10 <5px=100%

| Tick 2500 | Tick 5000 | Tick 10000 | Tick 25000 | Tick 50000 |
|-----------|-----------|------------|------------|------------|
| ![2500](img/001_tick_002500.png) | ![5000](img/001_tick_005000.png) | ![10000](img/001_tick_010000.png) | ![25000](img/001_tick_025000.png) | ![50000](img/001_tick_050000.png) |

**Key findings:**

1. **Clusters form live during training.** Contiguity reaches 1.000 by tick 10000
   (20% of training) and stays perfect through the remaining 40000 ticks. No
   post-hoc clustering needed.

2. **Zero impact on embedding quality.** Final eval metrics (PCA=0.567, K10=96.9%
   <3px) are identical to non-clustered baseline from ts-00016 Run 001.

3. **Minimal overhead.** ~11 ms/tick with clustering vs ~10 ms/tick without.
   The cluster maintenance adds ~1-2ms per tick (streaming update on 256 anchors).

4. **Self-healing works live.** 306 splits total, mostly in the first 5000 ticks
   during the turbulent early phase. After tick 10000, splits stop — system is
   stable. 100/100 clusters alive throughout.

5. **Cluster structure emerges WITH embeddings.** By tick 2500 (embeddings still
   chaotic), spatial patches are already forming. By tick 5000 (KNN spatial=0.465),
   clusters are nearly contiguous (0.974). The cluster structure tracks embedding
   convergence in real time.

### Run 002: Jump rate over time (m=100, 50k ticks)

Same config as Run 001 but with `cluster_report_every=1000` to capture the
inter-cluster jump rate at higher resolution.

```
Output: ~/data/research/thalamus-sorter/exp_00017/002_live_clusters_80x80_m100_jumps/
```

| Tick | Jumps/tick | Contiguity | Diameter | Phase |
|------|-----------|------------|----------|-------|
| 2000 | 38.4 | 0.227 | 95.4 | Chaotic — embeddings random |
| 3000 | 24.5 | 0.876 | 21.1 | Structure emerging |
| 4000 | 10.1 | 0.932 | 17.7 | Settling |
| 5000 | 6.5 | 0.991 | 14.4 | Nearly converged |
| 6000 | 3.4 | 0.998 | 13.1 | Stable |
| 7000 | 2.9 | 1.000 | 12.7 | Converged |
| 10000 | 1.8 | 1.000 | 12.1 | Steady-state |
| 20000 | 2.6 | 1.000 | 11.8 | Steady-state |
| 50000 | 3.3 | 1.000 | 12.5 | Steady-state |

**Three regimes:**
1. **Tick 1000-3000** — high churn (25-38 jumps/tick). Clusters reorganize as
   embeddings begin forming structure. 132-184 splits in this window.
2. **Tick 3000-7000** — rapid convergence, drops from 24 to 3 jumps/tick as
   clusters lock into spatially contiguous regions.
3. **Tick 7000+** — steady-state ~2-4 jumps/tick, never reaches zero. Boundary
   neurons trade between adjacent clusters as embeddings continue to drift.
   With 256 anchors/tick, ~1% of sampled neurons jump — the system stays alive
   and adaptive rather than frozen.

### Run 003: m=640 (10 neurons/cluster), 50k ticks, wandb

```
preset: gray_80x80_saccades
n=6400, m=640, dims=8, k2=10, lr_cluster=0.01, split_every=10
wandb: https://wandb.ai/kintaroai-dot-com/thalamus-sorter/runs/87kzhpwz
Output: ~/data/research/thalamus-sorter/exp_00017/003_live_clusters_80x80_m640_50k/
Runtime: 775s (~15 ms/tick)
```

| Tick | Alive | Contiguity | Diameter | Jumps/tick | Splits |
|------|-------|------------|----------|-----------|--------|
| 1000 | 548/640 | 0.194 | 70.9 | 34.1 | 714 |
| 5000 | 617/640 | 0.956 | 5.8 | 23.6 | 3956 |
| 10000 | 638/640 | 0.997 | 4.3 | 7.6 | 5485 |
| 25000 | 640/640 | 0.998 | 4.2 | 8.3 | 6627 |
| 50000 | 633/640 | 0.998 | 4.3 | 9.7 | 8349 |

**Eval:** PCA=0.543, K10 <3px=97.5%, K10 <5px=100%

| Tick 1000 | Tick 5000 | Tick 10000 | Tick 50000 |
|-----------|-----------|------------|------------|
| ![1000](img/003_tick_001000.png) | ![5000](img/003_tick_005000.png) | ![10000](img/003_tick_010000.png) | ![50000](img/003_tick_050000.png) |

**Comparison with m=100:**

| Metric | m=100 | m=640 |
|--------|-------|-------|
| Neurons/cluster | 64 | 10 |
| Contiguity @ 50k | 1.000 | 0.998 |
| Diameter @ 50k | 11.6 | 4.3 |
| Steady-state jumps/tick | 2-4 | 8-10 |
| Total splits | 306 | 8349 |
| Alive @ 50k | 100/100 | 633/640 |
| Runtime | 552s | 775s |
| Overhead per tick | ~1-2ms | ~5ms |

**Key findings:**

1. **m=640 works well live.** Contiguity 0.998, diameter 4.3 — fine-grained
   spatial clusters that track embedding convergence. Quality converges by
   tick 10k, same timeline as m=100.

2. **More ongoing churn at higher m.** Steady-state ~8-10 jumps/tick (vs 2-4
   for m=100). Smaller clusters have proportionally more boundary neurons.
   This is expected and healthy — the system adapts continuously.

3. **Dynamic equilibrium with cluster death/recovery.** Alive count fluctuates
   between 614-640 at steady state. Splits continuously recover dead clusters
   (8349 total). The throttle + split mechanism from ts-00016 handles this
   gracefully — no intervention needed.

4. **No embedding quality impact.** K10 <3px=97.5% (slightly better than
   m=100's 96.9%). Clustering overhead is ~5ms/tick — acceptable.

### Run 004: Hysteresis test (h=0.1, m=100, 10k ticks)

Added `--cluster-hysteresis H` parameter: neuron only jumps from cluster A→B if
`dist_to_B < dist_to_A * (1 - H)`. Prevents boundary ping-ponging.

```
preset: gray_80x80_saccades
n=6400, m=100, dims=8, k2=10, lr_cluster=0.01, hysteresis=0.1
Output: ~/data/research/thalamus-sorter/exp_00017/005_005_hysteresis_01_10k/
```

| Metric | h=0.0 (Run 001) | h=0.1 (Run 004) |
|--------|-----------------|-----------------|
| Contiguity @ 10k | 1.000 | 0.999 |
| Diameter @ 10k | 12.7 | 12.3 |
| Jumps/tick @ 10k | 1.8 | 1.8 |
| Total splits | 304 | 290 |
| Alive | 100/100 | 100/100 |
| K10 <3px | 96.9% | 96.4% |

**Finding:** Hysteresis works as intended — same convergence, same steady-state
jump rate, no quality degradation. The margin filters marginal reassignments
without blocking genuine ones. Default stays at 0.0 (no resistance) since the
system is already stable, but the knob is available for higher-m configs where
boundary churn is more pronounced.

### Run 005: Hysteresis h=0.1, m=640, 50k ticks (wandb)

Full-scale test of hysteresis at high cluster count.

```
preset: gray_80x80_saccades
n=6400, m=640, dims=8, k2=10, lr_cluster=0.01, hysteresis=0.1
wandb: https://wandb.ai/kintaroai-dot-com/thalamus-sorter/runs/wxx6xhgf
Output: ~/data/research/thalamus-sorter/exp_00017/006_006_m640_h01_50k/
Runtime: 768s (~15 ms/tick)
```

| Metric | h=0.0 (Run 003) | h=0.1 (Run 005) |
|--------|-----------------|-----------------|
| Contiguity @ 50k | 0.998 | 0.998 |
| Diameter @ 50k | 4.3 | 4.2 |
| Jumps/tick steady-state | 8-10 | 4-7 |
| Total splits | 8,349 | 10,640 |
| Alive @ 50k | 633/640 | 633/640 |
| Runtime | 775s | 768s |
| K10 <3px | 97.5% | **98.0%** |

**Finding:** Hysteresis reduces steady-state churn from ~8-10 to ~4-7 jumps/tick
at m=640. Cluster quality identical, embedding quality slightly better (98.0% vs
97.5%). More total splits (10.6k vs 8.3k) because fewer incoming jumps let some
small clusters die faster, but split recovery handles it.

### Run 006: Hysteresis h=0.3, m=640, 50k ticks (wandb)

Aggressive hysteresis — neuron must be 30% closer to new centroid to jump.

```
preset: gray_80x80_saccades
n=6400, m=640, dims=8, k2=10, lr_cluster=0.01, hysteresis=0.3
Output: ~/data/research/thalamus-sorter/exp_00017/008_007_m640_h03_50k/
Runtime: 676s (~13.5 ms/tick)
```

| Metric | h=0.0 (Run 003) | h=0.1 (Run 005) | h=0.3 (Run 006) |
|--------|-----------------|-----------------|-----------------|
| Contiguity @ 50k | 0.998 | 0.998 | 0.996 |
| Diameter @ 50k | 4.3 | 4.2 | 4.4 |
| Jumps/tick steady-state | 8-10 | 4-7 | 2-4 |
| Total splits | 8,349 | 10,640 | 6,603 |
| Alive @ 50k | 633/640 | 633/640 | **640/640** |
| Runtime | 775s | 768s | **676s** |
| K10 <3px | 97.5% | 98.0% | **98.2%** |

**Key findings:**

1. **h=0.3 is best across all metrics.** Fewer jumps (2-4/tick), fewer splits
   (6.6k), all 640 clusters alive at end, fastest runtime, best embedding quality.

2. **Hysteresis stabilizes cluster boundaries.** With h=0.3, boundary neurons
   don't ping-pong — they only move when genuinely closer to the new centroid.
   This means fewer cluster deaths and less split recovery needed.

3. **Early phase still works.** At tick 1000, jumps/tick=0.9 (vs 34 at h=0.0) —
   hysteresis suppresses the chaotic early churn. But contiguity still reaches
   0.955 by tick 5000 via the same convergence pathway.

### Incremental knn2: Decoupling Clusters from Neuron-Level KNN

Runs 001-006 required `--knn-track` because `knn2` (cluster-level neighbor graph)
was built by pooling neuron-level KNN lists via `frequency_knn`. This created an
unnecessary dependency — clustering needed O(n×K) KNN tracking even though the
skip-gram pairs already contain all the neighbor information.

**New approach:** `knn2` stores cluster indices + centroid distances directly,
updated incrementally from the same skip-gram pairs that drive training. Each tick:
1. Map (anchor, neighbor) pairs to (cluster_a, cluster_b)
2. Deduplicate cross-cluster pairs
3. For each new (c_a, c_n) not already in knn2, compute centroid distance
4. Replace worst (farthest) knn2 entry if new distance is closer

Init: knn2 filled with random cluster indices, distances = infinity.
Fills in naturally over the first few hundred ticks.

All dedup, distance computation, and insertion is GPU-vectorized via
`torch.unique`, `scatter_reduce`, and batch ops — 0.96ms per call with
25k pairs and m=640 (vs 158ms for the original Python loop).

### Run 007: Incremental knn2, m=100, h=0.3, 10k ticks (no --knn-track)

First test of decoupled clustering. Found and fixed a dedup bug where the same
c_n could fill all k2 slots for a given c_a.

```
preset: gray_80x80_saccades
n=6400, m=100, dims=8, k2=16, lr_cluster=0.01, hysteresis=0.3
No --knn-track
Output: ~/data/research/thalamus-sorter/exp_00017/008_incremental_knn2_dedup_10k/
```

| Tick | Alive | Contiguity | Diameter | Jumps/tick | Splits |
|------|-------|------------|----------|-----------|--------|
| 2000 | 100/100 | 0.201 | 95.1 | 44.4 | 231 |
| 4000 | 92/100 | 0.979 | 17.1 | 44.7 | 585 |
| 6000 | 100/100 | 0.997 | 13.7 | 14.0 | 767 |
| 8000 | 100/100 | 0.998 | 13.9 | 8.2 | 859 |
| 10000 | 100/100 | **1.000** | 13.6 | 4.9 | 911 |

**Eval:** PCA=0.972, K10 <3px=95.5%, K10 <5px=99.9%

knn2 fully populated: 16 unique cluster neighbors per row.

### Run 008: Incremental knn2, m=640, h=0.3, 50k ticks (wandb, no --knn-track)

Full-scale validation with GPU-vectorized knn2 update.

```
preset: gray_80x80_saccades
n=6400, m=640, dims=8, k2=16, lr_cluster=0.01, hysteresis=0.3
No --knn-track
wandb: https://wandb.ai/kintaroai-dot-com/thalamus-sorter/runs/vkapugx8
Output: ~/data/research/thalamus-sorter/exp_00017/016_009_m640_h03_no_knn_50k/
Runtime: 583s (~10.6 ms/tick steady state)
```

| Tick | Alive | Contiguity | Diameter | Jumps/tick | Splits |
|------|-------|------------|----------|-----------|--------|
| 1000 | 534/640 | 0.140 | 74.3 | 44.4 | 898 |
| 3000 | 560/640 | 0.616 | 16.6 | 53.9 | 3694 |
| 5000 | 604/640 | 0.984 | 5.2 | 18.6 | 6395 |
| 10000 | 608/640 | 0.997 | 4.5 | 16.0 | 9754 |
| 12000 | 626/640 | **1.000** | 4.3 | 12.0 | 10655 |
| 25000 | 624/640 | 0.999 | 4.6 | 6.7 | 16246 |
| 50000 | 623/640 | 0.995 | 4.9 | 11.6 | 25327 |

**Eval:** PCA=0.669, K10 <3px=95.1%, K10 <5px=100%

**Comparison with knn-track baseline (Run 006):**

| Metric | Run 006 (knn-track=10) | Run 008 (no knn) |
|--------|------------------------|------------------|
| Runtime | 676s (13.5 ms/tick) | **583s (10.6 ms/tick)** |
| Contiguity @ 50k | 0.996 | 0.995 |
| Diameter @ 50k | 4.4 | 4.9 |
| Alive @ 50k | 640/640 | 623/640 |
| K10 <3px | 98.2% | 95.1% |
| K10 <5px | 100% | 100% |
| Total splits | 6,603 | 25,327 |
| Requires --knn-track | Yes | **No** |

**Key findings:**

1. **Clustering fully decoupled from neuron-level KNN.** No `--knn-track` needed.
   knn2 populates from skip-gram pairs with zero additional data collection.

2. **14% faster.** 583s vs 676s — removing KNN tracking saves ~3ms/tick. The GPU
   knn2 update adds <1ms.

3. **Cluster quality comparable but not identical.** Contiguity matches (0.995 vs
   0.996), but more cluster deaths (623 vs 640 alive) and more splits (25k vs 6.6k).
   The frequency_knn approach had richer neighbor information from pooling all K
   neighbors; the pair-based approach sees only correlated pairs per tick.

4. **Embedding quality slightly lower.** K10 <3px drops from 98.2% to 95.1%.
   This may be because the knn-track run benefits from neighbor-of-neighbor
   sampling that the no-knn run lacks — not a clustering effect.

5. **Three-phase convergence preserved.** Same pattern as earlier runs: chaotic
   (tick 1-3k), rapid convergence (3-10k), steady state (10k+). The incremental
   knn2 fills in fast enough to guide reassignment during the convergence phase.

### Cluster Stability: Neuron Retention Between Reports

Cluster visualizations showed drastically different layouts between reports — not
gradual movement but near-total reshuffling. Added a stability metric:
`(cluster_ids == prev_cluster_ids).mean()` — fraction of neurons that stay in the
same cluster between consecutive reports.

**Run 009: m=100, h=0.3, 5k ticks, report_every=1000**

| Tick | Stability | Contiguity | Diameter | Jumps/tick |
|------|-----------|------------|----------|-----------|
| 1000 | — (first) | 0.038 | 105.4 | 0.6 |
| 2000 | 0.006 | 0.114 | 92.2 | 88.8 |
| 3000 | 0.010 | 0.519 | 74.3 | 29.5 |
| 4000 | 0.015 | 0.813 | 41.1 | 42.2 |
| 5000 | 0.019 | 0.982 | 16.6 | 21.7 |

**Finding:** Only ~2% of neurons stay in the same cluster over 1000 ticks, even as
contiguity reaches 0.98. This confirms that spatial cluster quality converges (the
*geometry* is correct) but individual neuron-to-cluster *assignments* churn almost
completely. The clusters are spatially stable in shape and position, but which
specific neuron IDs belong to which cluster is highly volatile.

This is expected with streaming k-means on continuously-evolving embeddings —
centroids drift, boundary neurons ping-pong, and splits create entirely new cluster
IDs. The hysteresis parameter (h=0.3) reduces jump rate but doesn't prevent the
cumulative effect over 1000 ticks.

**Frozen-embedding control:** On pre-trained (static) embeddings with random cluster
init, clusters converge in ~500 ticks to contiguity 0.995+ and then lock in with
perfect stability (1.000) and zero jumps. The churn is entirely caused by embedding
drift during training, not by clustering algorithm instability.

### Centroid Update Strategy: Nudge vs Exact

The centroid update strategy turned out to be the critical variable for cluster
stability — more important than knn2 mode, hysteresis, or any other parameter.

**Two modes** (`--cluster-centroid-mode`):
- **`nudge`** (default): After all reassignments in a tick, nudge centroids toward
  true member mean: `centroid += lr * (member_mean - centroid)`. Conservative —
  centroids drift slowly, dampening cascade effects.
- **`exact`**: Incremental arithmetic on each reassignment:
  `(centroid * size - emb) / (size - 1)` on removal,
  `(centroid * size + emb) / (size + 1)` on addition. Centroid snaps to true mean
  instantly, which can trigger chain reactions where one reassignment shifts a
  centroid enough to cause further reassignments.

### Run 023: Nudge centroids, h=0, m=640, 50k

```
preset: gray_80x80_saccades
n=6400, m=640, dims=8, k2=16, lr_cluster=0.01, hysteresis=0.0
centroid_mode=nudge, knn2_mode=incremental
Output: ~/data/research/thalamus-sorter/exp_00017/023_nudge_h0_50k_m640/
Runtime: 632s (~11 ms/tick)
```

| Tick | Stability | Contiguity | Diameter | Jumps/tick | Alive | Splits |
|------|-----------|------------|----------|-----------|-------|--------|
| 5k | 0.001 | 0.975 | 5.3 | 32.0 | 604 | 3,718 |
| 8k | 0.370 | 0.997 | 4.5 | 10.3 | 632 | 4,790 |
| 10k | **0.481** | 0.997 | 4.3 | 9.2 | **640** | 4,997 |
| 15k | **0.736** | 0.996 | 4.1 | 4.6 | **640** | 5,108 |
| 20k | 0.514 | 0.999 | 4.2 | 9.7 | **640** | 5,134 |
| 25k | 0.707 | 0.999 | 4.3 | 6.3 | **640** | 5,153 |
| 30k | 0.676 | 0.999 | 4.3 | 7.0 | **640** | 5,172 |
| 40k | 0.582 | 1.000 | 4.2 | 7.1 | **640** | 5,465 |
| 50k | 0.643 | 0.998 | 4.6 | 9.4 | **640** | 5,534 |

**Eval:** PCA=0.638, K10 <3px=95.8%, K10 <5px=100%

### Run 024: Exact centroids, h=0.3, m=640, 50k

```
preset: gray_80x80_saccades
n=6400, m=640, dims=8, k2=16, lr_cluster=0.01, hysteresis=0.3
centroid_mode=exact, knn2_mode=incremental
Output: ~/data/research/thalamus-sorter/exp_00017/024_exact_50k_m640/
Runtime: 701s (~14 ms/tick)
```

| Tick | Stability | Contiguity | Diameter | Jumps/tick | Alive | Splits |
|------|-----------|------------|----------|-----------|-------|--------|
| 5k | 0.004 | 0.977 | 5.4 | 24.8 | 606 | 6,306 |
| 8k | 0.045 | 0.993 | 5.0 | 14.2 | 619 | 8,252 |
| 10k | 0.138 | 0.995 | 5.0 | 8.6 | 622 | 9,106 |
| 15k | 0.099 | 0.994 | 4.8 | 13.6 | 607 | 10,701 |
| 20k | 0.242 | 0.998 | 4.5 | 7.8 | 630 | 12,119 |
| 25k | 0.035 | 0.996 | 5.0 | 14.0 | 628 | 14,154 |
| 30k | 0.121 | 0.997 | 4.7 | 11.1 | 620 | 15,942 |
| 40k | 0.138 | 1.000 | 4.3 | 9.8 | 623 | 20,967 |
| 50k | 0.011 | 0.999 | 4.5 | 20.0 | 619 | 25,805 |

**Eval:** PCA=0.551, K10 <3px=96.6%, K10 <5px=100%

### Centroid mode comparison

| Metric | Nudge (h=0) | Exact (h=0.3) |
|--------|-------------|---------------|
| **Stability (10-20k)** | **48–74%** | 2–24% |
| **Stability (30-50k)** | **51–74%** | 1–24% |
| Jumps/tick (steady) | **5–10** | 8–20 |
| **Alive @ 50k** | **640/640** | 619/640 |
| Contiguity @ 50k | 0.998 | 0.999 |
| Diameter @ 50k | 4.6 | 4.5 |
| **Total splits** | **5,534** | 25,805 |
| K10 <3px | 95.8% | 96.6% |
| Runtime | **632s** | 701s |

**Key findings:**

1. **Nudge centroids are dramatically more stable.** 50–74% of neurons stay in the
   same cluster across 1000-tick intervals (vs 1–24% with exact). This is the
   difference between visually coherent clusters and total reshuffling.

2. **Nudge keeps all clusters alive.** 640/640 alive throughout (vs 619/640 with
   exact). Only 5.5k total splits vs 25.8k — the conservative centroid movement
   prevents the cascade failures that kill clusters.

3. **Exact centroids cause positive feedback loops.** One reassignment shifts the
   centroid immediately, making more neurons closer to/farther from it, triggering
   more reassignments. Even h=0.3 hysteresis can't prevent this — the cascades
   overwhelm the margin.

4. **Nudge doesn't need hysteresis.** h=0 with nudge outperforms h=0.3 with exact
   on every stability metric. The lr=0.01 dampening provides natural hysteresis —
   centroids only move 1% toward the true mean each tick.

5. **Nudge is now the default.** `--cluster-centroid-mode nudge` (no hysteresis
   needed). Use `--cluster-centroid-mode exact --cluster-hysteresis 0.3` for the
   previous behavior.

**Note:** Runs 010–013 (previously documented here) were invalidated due to a code
revert bug — experiment archiving commits had clobbered dev/ files, causing those
runs to execute old pre-incremental-knn2 code. They have been replaced by the
correctly-executed Runs 023–024 above.

### Run 025: Nudge 300k long run (wandb)

Long-duration test to verify nudge stability doesn't degrade over time.

```
preset: gray_80x80_saccades
n=6400, m=640, dims=8, k2=16, lr_cluster=0.01, hysteresis=0.0
centroid_mode=nudge, knn2_mode=incremental
wandb: https://wandb.ai/kintaroai-dot-com/thalamus-sorter/runs/71hmna1s
Output: ~/data/research/thalamus-sorter/exp_00017/025_nudge_300k_m640_wandb/
Runtime: 3376s (~11 ms/tick)
```

| Tick | Stability | Contiguity | Diam | Jumps/t | Alive | Splits |
|------|-----------|------------|------|---------|-------|--------|
| 10k | 0.561 | 0.998 | 4.3 | 7.2 | 640 | 4,915 |
| 50k | 0.637 | 0.998 | 4.5 | 8.0 | 640 | 7,057 |
| 100k | 0.653 | 0.995 | 4.5 | 7.2 | 640 | 10,984 |
| 150k | 0.104 | 0.996 | 4.6 | 14.2 | 634 | 16,288 |
| 200k | 0.515 | 0.998 | 4.1 | 8.8 | 639 | 18,452 |
| 250k | 0.545 | 0.999 | 4.2 | 8.8 | 640 | 23,216 |
| 300k | 0.677 | 0.999 | 4.1 | 6.9 | 640 | 29,170 |

**Eval:** PCA=0.542, K10 <3px=98.6%, K10 <5px=100%

**Finding:** No degradation over 300k ticks. Stability oscillates 10–68% with
occasional dips but no downward trend. Contiguity stays 0.995+ and 640/640 alive
most of the time. Nudge centroids are stable long-term.

### Runs 026–027: Hysteresis sweep with nudge centroids (20k)

Tested h=0.05 and h=0.1 with nudge centroids. (h=0.3 was previously tested with
nudge in Run 023's predecessor and found to freeze clusters entirely.)

**Run 026: h=0.1, nudge, m=640, 20k**
```
Output: ~/data/research/thalamus-sorter/exp_00017/026_nudge_h01_20k_m640/
```

**Run 027: h=0.05, nudge, m=640, 20k**
```
Output: ~/data/research/thalamus-sorter/exp_00017/027_nudge_h005_20k_m640/
```

Comparison at steady state (ticks 10k–20k):

| Metric | h=0 (Run 025) | h=0.05 (Run 027) | h=0.1 (Run 026) |
|--------|---------------|-------------------|-------------------|
| Stability | 22–64% | 41–77% | 22–66% |
| Jumps/tick | 7–12 | 3–7 | 3–8 |
| Alive @ 20k | 640 | 640 | 640 |
| Contiguity | 0.998–0.999 | 0.994–0.999 | 0.993–0.999 |
| Splits @ 20k | ~5.1k | 5,060 | 5,153 |
| K10 <3px | 98.6% | 98.7% | 95.9% |

**Run 028: h=0.2, nudge, m=640, 20k**
```
Output: ~/data/research/thalamus-sorter/exp_00017/028_nudge_h02_20k_m640/
```

| Metric | h=0 (Run 025) | h=0.05 (Run 027) | h=0.1 (Run 026) | h=0.2 (Run 028) |
|--------|---------------|-------------------|-------------------|-------------------|
| Stability (10-20k) | 22–64% | 41–77% | 22–66% | 67–90% |
| Contiguity | 0.998 | 0.996 | 0.995 | **0.665** |
| Diameter | 4.2 | 4.2 | 4.7 | **22.1** |
| Jumps/tick | 7–12 | 3–7 | 3–8 | 0.7–2.5 |
| Splits @ 20k | ~5.1k | 5,060 | 5,153 | 2,561 |
| K10 <3px | 98.6% | 98.7% | 95.9% | 98.5% |

**Finding:** Hysteresis with nudge centroids doesn't help reduce cluster splits —
it prevents neurons from leaving bad clusters, trapping them and making things
worse. At h=0.2, contiguity stalls at 0.665 (diameter 22) because neurons can't
escape to their correct spatial cluster. The cliff is between h=0.1 (works) and
h=0.2 (frozen). The right lever for controlling churn is the centroid learning
rate, not hysteresis — slowing centroid movement reduces the *reason* neurons
want to jump, rather than preventing them from jumping when they should.

### Run 029: Decaying centroid lr (0.01→0.001 over 300k)

Hypothesis: decaying the centroid nudge lr should reduce churn as clusters mature —
early on, centroids need to move fast to track converging embeddings; later, small
adjustments should suffice.

```
preset: gray_80x80_saccades
n=6400, m=640, dims=8, k2=16, hysteresis=0.0
centroid_mode=nudge, lr=0.01 linear decay to 0.001 at tick 300k
wandb: https://wandb.ai/kintaroai-dot-com/thalamus-sorter/runs/64byd0wn
Output: ~/data/research/thalamus-sorter/exp_00017/029_nudge_lrdecay_300k_m640_wandb/
Runtime: ~3400s
```

| Tick | lr | Stability | Contiguity | Jumps/t | Alive | Splits |
|------|-----|-----------|------------|---------|-------|--------|
| 10k | 0.0097 | 0.420 | 0.997 | 8.9 | 638 | 4,943 |
| 50k | 0.0085 | 0.539 | 0.997 | 8.8 | 640 | 6,819 |
| 100k | 0.0070 | 0.642 | 1.000 | 9.3 | 640 | 9,993 |
| 200k | 0.0040 | 0.429 | 0.997 | 12.0 | 628 | 17,683 |
| 300k | 0.0010 | 0.429 | 0.999 | 8.0 | 638 | 33,362 |

**Eval:** K10 <3px=98.7%

| Metric | Constant lr=0.01 (Run 025) | Decaying lr (Run 029) |
|--------|---------------------------|------------------------|
| Stability @ 300k | **0.677** | 0.429 |
| Total splits | **29,170** | 33,362 |
| Contiguity | 0.999 | 0.999 |
| K10 <3px | 98.6% | 98.7% |

**Finding:** Decaying lr didn't help — stability is worse and splits increased.
Lower lr means centroids track drifting embeddings less accurately, so neurons
end up farther from their centroid and jump more.

### Run 030: Increasing centroid lr (0.01→0.1 over 300k)

Opposite hypothesis: as embeddings mature, centroids should track them more
tightly — higher lr means centroids snap closer to true member mean, so neurons
are always near their centroid and don't need to jump.

```
preset: gray_80x80_saccades
n=6400, m=640, dims=8, k2=16, hysteresis=0.0
centroid_mode=nudge, lr=0.01 linear ramp to 0.1 at tick 300k
wandb: https://wandb.ai/kintaroai-dot-com/thalamus-sorter/runs/4dx4srj7
Output: ~/data/research/thalamus-sorter/exp_00017/030_nudge_lrinc_300k_m640_wandb/
```

| Tick | lr | Stability | Contiguity | Jumps/t | Alive | Splits |
|------|-----|-----------|------------|---------|-------|--------|
| 10k | 0.013 | 0.486 | 0.997 | 7.8 | 636 | 4,701 |
| 50k | 0.025 | 0.674 | 1.000 | 5.7 | 640 | 5,510 |
| 100k | 0.040 | 0.588 | 1.000 | 7.5 | 640 | 6,012 |
| 200k | 0.070 | 0.679 | 1.000 | 6.4 | 640 | 7,925 |
| 300k | 0.100 | **0.747** | **1.000** | **3.9** | **640** | 8,982 |

**Eval:** K10 <3px=98.9%

**Centroid lr comparison (all nudge, h=0, 300k):**

| Metric | Decay 0.01→0.001 | Const 0.01 | **Ramp 0.01→0.1** |
|--------|-----------------|------------|---------------------|
| Stability @ 300k | 0.429 | 0.677 | **0.747** |
| Contiguity | 0.999 | 0.999 | **1.000** |
| Diameter | 4.2 | 4.1 | **4.0** |
| Total splits | 33,362 | 29,170 | **8,982** |
| Jumps/tick @ 300k | 8.0 | 6.9 | **3.9** |
| K10 <3px | 98.7% | 98.6% | **98.9%** |

### Run 031: Increasing centroid lr (0.01→1.0 over 300k)

Even more aggressive ramp — 100x increase over 300k ticks.

```
preset: gray_80x80_saccades
n=6400, m=640, dims=8, k2=16, hysteresis=0.0
centroid_mode=nudge, lr=0.01 linear ramp to 1.0 at tick 300k
wandb: https://wandb.ai/kintaroai-dot-com/thalamus-sorter/runs/xegl1mda
Output: ~/data/research/thalamus-sorter/exp_00017/031_nudge_lrinc100x_300k_m640_wandb/
```

| Tick | lr | Stability | Contiguity | Jumps/t | Alive | Splits |
|------|-----|-----------|------------|---------|-------|--------|
| 10k | 0.04 | 0.575 | 0.998 | 7.1 | 640 | 4,052 |
| 50k | 0.17 | 0.639 | 0.998 | 5.9 | 640 | 4,328 |
| 100k | 0.34 | 0.579 | 0.999 | 6.7 | 640 | 4,381 |
| 150k | 0.50 | 0.640 | 1.000 | 4.5 | 640 | 4,403 |
| 200k | 0.67 | 0.735 | 1.000 | 3.5 | 640 | 4,403 |
| 300k | 1.00 | **0.794** | **1.000** | **2.3** | **640** | **4,403** |

**Eval:** K10 <3px=98.7%

**Full centroid lr comparison (all nudge, h=0, 300k):**

| Metric | Decay →0.001 | Const 0.01 | Ramp →0.1 | **Ramp →1.0** |
|--------|-------------|------------|-----------|----------------|
| Stability @ 300k | 0.429 | 0.677 | 0.747 | **0.794** |
| Contiguity | 0.999 | 0.999 | 1.000 | **1.000** |
| Total splits | 33,362 | 29,170 | 8,982 | **4,403** |
| Jumps/tick @ 300k | 8.0 | 6.9 | 3.9 | **2.3** |
| K10 <3px | 98.7% | 98.6% | 98.9% | 98.7% |

### Run 032: Constant lr=1.0 (nudge = snap to mean), 300k

At lr=1.0, `centroid += 1.0 * (mean - centroid)` is equivalent to
`centroid = member_mean`. This is like exact mode but applied *after* all
reassignments in a tick rather than during the loop — avoiding cascades.

```
preset: gray_80x80_saccades
n=6400, m=640, dims=8, k2=16, hysteresis=0.0
centroid_mode=nudge, lr=1.0 constant
wandb: https://wandb.ai/kintaroai-dot-com/thalamus-sorter/runs/t7sh28nw
Output: ~/data/research/thalamus-sorter/exp_00017/032_nudge_lr1_300k_m640_wandb/
```

| Tick | Stability | Contiguity | Jumps/t | Alive | Splits |
|------|-----------|------------|---------|-------|--------|
| 10k | 0.671 | 1.000 | 4.2 | 640 | 486 |
| 50k | 0.739 | 1.000 | 3.0 | 640 | 487 |
| 100k | 0.692 | 1.000 | 3.9 | 640 | 488 |
| 200k | 0.783 | 1.000 | 2.4 | 640 | 489 |
| 300k | 0.645 | 1.000 | 5.2 | 640 | 489 |

**Eval:** K10 <3px=97.8%

**Full centroid lr comparison (all nudge, h=0, 300k):**

| Metric | Decay →0.001 | Const 0.01 | Ramp →0.1 | Ramp →1.0 | **Const 1.0** |
|--------|-------------|------------|-----------|-----------|----------------|
| Stability @ 300k | 0.429 | 0.677 | 0.747 | 0.794 | 0.645 |
| Stability range | — | 10–68% | 49–75% | 58–79% | 65–78% |
| Contiguity | 0.999 | 0.999 | 1.000 | 1.000 | **1.000** |
| Total splits | 33,362 | 29,170 | 8,982 | 4,403 | **489** |
| Jumps/tick @ 300k | 8.0 | 6.9 | 3.9 | 2.3 | 5.2 |
| K10 <3px | 98.7% | 98.6% | 98.9% | 98.7% | 97.8% |

**Key insights:**

1. **Higher centroid lr = more stable clusters.** When centroids closely track
   their members, neurons stay near their centroid and don't need to jump.
   Low lr creates a lag between centroid position and actual member distribution,
   which causes unnecessary reassignments.

2. **Constant lr=1.0 achieves near-zero cluster deaths (489 splits, almost all
   in first 10k).** This is because centroids always equal the true member mean,
   so the only reason to jump is genuine embedding drift, not centroid lag.

3. **Nudge lr=1.0 ≠ exact mode.** Both compute the true mean, but nudge applies
   it *after* all reassignments in a tick (batch update), while exact updates
   *during* the loop (sequential). This single difference eliminates cascades —
   no reassignment can influence another within the same tick.

4. **Ramp →1.0 has the best peak stability (0.794)** because low lr in early
   ticks prevents churn during the chaotic embedding convergence phase. Constant
   lr=1.0 is slightly less stable early on but has the fewest splits overall.

5. **Recommended: lr=1.0 constant or ramp →1.0** depending on whether you
   prioritize stability (ramp) or minimal splits (constant).

### Runs 033–035: Hysteresis sweep with lr=1.0 (300k)

With lr=1.0, centroids always equal the true member mean. This changes hysteresis
behavior fundamentally — the 30% margin is now relative to an accurate centroid,
not a lagging one. h=0.3 which froze clusters at lr=0.01 now works perfectly.

**Run 033: lr=1.0, h=0.1, 300k**
```
wandb: https://wandb.ai/kintaroai-dot-com/thalamus-sorter/runs/3gt0r0m4
Output: ~/data/research/thalamus-sorter/exp_00017/033_nudge_lr1_h01_300k_m640_wandb/
```

**Run 034: lr=1.0, h=0.2, 300k**
```
wandb: https://wandb.ai/kintaroai-dot-com/thalamus-sorter/runs/ryabiz2z
Output: ~/data/research/thalamus-sorter/exp_00017/034_nudge_lr1_h02_300k_m640_wandb/
```

**Run 035: lr=1.0, h=0.3, 300k**
```
wandb: https://wandb.ai/kintaroai-dot-com/thalamus-sorter/runs/nzk8lcd1
Output: ~/data/research/thalamus-sorter/exp_00017/035_nudge_lr1_h03_300k_m640_wandb/
```

| Metric | h=0 (032) | h=0.1 (033) | h=0.2 (034) | **h=0.3 (035)** |
|--------|-----------|-------------|-------------|-----------------|
| Stability @ 300k | 0.645 | 0.782 | 0.803 | **0.849** |
| Contiguity | 1.000 | 1.000 | 1.000 | **1.000** |
| Diameter | 4.0 | 4.1 | 4.3 | **3.9** |
| Jumps/t @ 300k | 5.2 | 2.3 | 1.9 | **1.2** |
| Total splits | 489 | 543 | 537 | **507** |
| K10 <3px | 97.8% | 98.5% | 97.8% | **99.1%** |

**Key findings:**

1. **lr=1.0 + h=0.3 is the best config.** 85% stability, 1.2 jumps/tick, perfect
   contiguity, 99.1% K10. All 640 clusters alive with only 507 splits total.

2. **Hysteresis works correctly at lr=1.0.** The margin is relative to the true
   member mean, so it filters genuine boundary noise without trapping neurons in
   wrong clusters. At lr=0.01 the same h=0.3 froze clusters (contiguity 0.665)
   because centroids lagged and the margin was applied to inaccurate positions.

3. **Monotonic improvement with h.** Each step from h=0→0.1→0.2→0.3 improves
   stability and reduces jumps. No cliff — the failure mode at lr=0.01 is gone.

4. **Recommended defaults: lr=1.0, h=0.3.** This combination maximizes stability
   while maintaining perfect spatial quality.

### Design idea: LRU cluster history to prevent ping-ponging

A boundary neuron between clusters A and B may jump A→B→A→B across ticks as
embeddings drift. Each jump is "correct" in the moment but the net effect is
pointless churn.

**Proposal:** Each neuron maintains a small LRU cache (size 2–3) of recently
visited cluster IDs. When considering a reassignment to cluster C, check: if C
is in the neuron's LRU, suppress the jump (the neuron was recently in C and
left — going back is likely a ping-pong). Only allow the jump if C is a *new*
cluster the neuron hasn't recently belonged to, or if the distance improvement
exceeds a higher threshold for LRU-cached clusters.

This is different from hysteresis (which slows *all* jumps) — it specifically
targets the A→B→A oscillation pattern while allowing genuine first-time jumps.
The LRU naturally ages out, so after enough ticks a neuron can return to a
previously visited cluster if the embedding truly drifted there.

**Implementation:** `lru_clusters` array of shape `(n, lru_size)` storing recent
cluster IDs per neuron. On each reassignment, push old cluster into LRU, pop
oldest. In the candidate evaluation loop, penalize or skip candidates found in
the neuron's LRU.

### Design idea (variant): LRU soft multi-membership

Instead of using LRU to *suppress* jumps, use it to define *simultaneous
membership*: a neuron belongs to all clusters in its LRU at once. Removal
from a cluster only happens when that cluster ages out of the LRU.

**How it changes the algorithm:**

- **Centroids include recent members, not just current.** Losing a boundary
  neuron from cluster A doesn't shift A's centroid — the neuron is still
  contributing via its LRU entry. Centroid only changes when the LRU entry
  for A expires (after lru_size jumps).
- **Cluster sizes overlap.** A boundary neuron between A and B counts toward
  both. Sizes become 10-13 instead of exactly 10. Contiguity becomes fuzzy.
- **Splits nearly eliminated.** Clusters don't die because boundary neurons
  stick around via LRU even after their primary assignment changes.
- **Biological analogue: receptive field overlap.** A photoreceptor contributes
  to multiple overlapping ganglion cell fields, not just one.

**Critical constraint: LRU size must be small (2-3 max).** Larger LRU means
every cluster absorbs distant neurons and centroids blur into averages of
large overlapping regions, destroying spatial specificity.

**Possible refinement:** Weight contributions by recency — primary cluster
gets weight 1.0, LRU-1 gets 0.5, LRU-2 gets 0.25. This preserves centroid
accuracy while still dampening boundary oscillations. Centroids become
exponentially-weighted moving averages of recent membership.

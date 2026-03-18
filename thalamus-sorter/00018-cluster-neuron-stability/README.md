# ts-00018: Cluster Neuron Stability

**Date:** 2026-03-17
**Status:** In progress
**Source:** `exp/ts-00018`

## Goal

Measure per-neuron cluster stability: how many neurons stay in the same cluster
over time, which neurons are chronic oscillators, and whether multi-cluster ring
buffers (max_k=2,3) genuinely reduce boundary churn vs single-membership.

ts-00017 showed aggregate stability (fraction of neurons unchanged between
reports). This experiment digs deeper: per-neuron lifetime, oscillation patterns,
and the spatial distribution of unstable neurons.

## Method

### Warm-start baseline

Training 50k ticks from scratch takes ~4 minutes but is wasted work when the
experiment is about cluster dynamics, not embedding convergence. Instead:

1. **Pre-train a 50k baseline model** (no clustering) and save it
2. **Warm-start from the saved model** with `--warm-start model.npy`
3. Continue training with clustering enabled — clusters fit within ~5k ticks
   on already-converged embeddings
4. Experiment on the stabilized clusters

This lets each experimental run be ~5-10k ticks instead of 50k+.

### Baseline pre-train

```
preset: gray_80x80_saccades
n=6400, dims=8, 50k ticks, no clustering
Output: ~/data/research/thalamus-sorter/exp_00018/001_pretrain_50k/
```

The saved `model.npy` becomes the warm-start checkpoint for all subsequent runs.

### Planned experiments

Once the baseline model is ready:

1. **Warm-start + max_k=1,2,3** (5-10k ticks each): compare per-neuron
   stability with ring buffer depth. Track which specific neurons oscillate
   and whether max_k=2 eliminates their oscillations or just masks them.

2. **Neuron stability histogram**: at each report interval, compute per-neuron
   "time since last cluster change". Plot distribution — are most neurons
   stable with a long tail of oscillators, or is instability widespread?

3. **Spatial map of instability**: render a heatmap where pixel brightness =
   number of cluster changes. Are unstable neurons at cluster boundaries
   (expected) or scattered (would indicate centroid drift)?

4. **Oscillation detection**: track per-neuron cluster history over a window.
   Flag neurons with A->B->A patterns. Compare oscillation rate across max_k.

## Results

### Run 001: Baseline pre-train (50k, no clustering)

```
preset: gray_80x80_saccades, n=6400, dims=8, 50k ticks, no clustering
Runtime: 86s (1.7 ms/tick)
Output: ~/data/research/thalamus-sorter/exp_00018/001_pretrain_50k/
Model: ~/data/research/thalamus-sorter/exp_00018/001_pretrain_50k/model.npy
```

**Eval:** PCA=0.5292, K10 mean=1.89, <3px=96.8%, <5px=100.0%

Embeddings converged (std=1.2089). This model is the warm-start checkpoint for
all subsequent runs in this experiment.

### Runs 002–004: Suppress mode (warm-start, 10k, max_k=1,2,3)

When best candidate is already in ring, **skip entirely** — neuron doesn't
move. Ring suppresses oscillation by blocking return to registered clusters.

```
warm-start from Run 001, 10k ticks, m=640, lr=1.0, h=0.0, report_every=500
```

| Metric | max_k=1 | max_k=2 | max_k=3 |
|---|---|---|---|
| Alive | 640/640 | 602/640 | 390/640 |
| Contiguity | 1.000 | 0.995 | 0.970 |
| Total jumps | 27,029 | 21,344 | 15,150 |
| Stability | 0.863 | 0.932 | 0.938 |
| Oscillators (A->B->A) | 1,678 (26.2%) | 340 (5.3%) | 12 (0.2%) |
| Never changed | 14.5% | 9.0% | 18.5% |
| >3 changes | 28.1% | 17.5% | 2.8% |
| K10 <3px | 99.2% | 99.4% | 99.3% |

Oscillation suppression works — max_k=3 drops A->B->A to 0.2%. But **clusters
die**: max_k=3 loses 250 clusters (390/640 alive). The ring blocks neurons from
leaving, so oversized clusters can't shed members and undersized ones starve.

### Runs 005–007: Primary-switch mode (warm-start, 10k, max_k=1,2,3)

Changed behavior: when best candidate is already in ring, **switch primary to
it** (just move pointer, no ring advance). Neuron genuinely moves between
registered clusters. Oscillation within the ring is free (no eviction cost),
but sizes and centroids track the primary.

| Metric | max_k=1 | max_k=2 | max_k=3 |
|---|---|---|---|
| Alive | **640/640** | **640/640** | **640/640** |
| Contiguity | **1.000** | **1.000** | **1.000** |
| Total jumps | 31,900 | 28,477 | 25,364 |
| Stability | 0.871 | 0.883 | 0.839 |
| Oscillators (A->B->A) | 1,978 (30.9%) | 1,727 (27.0%) | 1,439 (22.5%) |
| Never changed | 9.9% | 14.2% | 16.5% |
| >3 changes | 38.1% | 30.7% | 25.9% |
| K10 <3px | 98.7% | 99.0% | 98.5% |

**Key finding: no dead clusters.** All 640 alive with contiguity=1.000 for every
max_k. The tradeoff: oscillation rates are higher (27% vs 5% for max_k=2)
because neurons can ping-pong between registered clusters. But cluster health
is perfect — no starvation, no dead clusters, no splits needed.

**Suppress vs primary-switch:**

- **Suppress** optimizes for stability: fewer oscillations, but kills clusters
- **Primary-switch** optimizes for cluster health: all alive, perfect contiguity,
  but allows oscillation within the ring

Primary-switch is the better default — dead clusters are a structural problem,
while oscillation at boundaries is natural and doesn't hurt eval metrics.

### Runs 008–015: LRU eviction + jump counting fix (max_k=1,2,3,4,10)

Two improvements over runs 005–007:

1. **LRU eviction** instead of round-robin: `last_used[n, max_k]` tracks when
   each slot was last accessed. New cluster evicts the least-recently-used slot
   instead of blindly advancing the pointer. Keeps frequently-visited clusters
   in the ring longer.
2. **Jump counting fix**: `n_reassigned` only counts genuine new-cluster entries
   (ring writes), not in-ring primary switches. Previously all primary changes
   counted as jumps, inflating the metric.

#### Full run results (new-cluster jumps only)

| Metric | mk=1 | mk=2 | mk=3 | mk=4 | mk=10 |
|---|---|---|---|---|---|
| New-cluster jumps | 25,800 | **18,153** | 16,251 | 17,002 | 17,022 |
| Jumps/tick (final) | 1.9 | 1.5 | 1.4 | 1.1 | 1.1 |
| Alive | 640 | 640 | 640 | 640 | 640 |
| Contiguity | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| K10 <3px | 99.0% | 98.8% | 98.5% | 98.0% | 98.3% |
| Oscillators (full) | 24% | 28% | 26% | 30% | 31% |

New-cluster jumps drop sharply mk=1→2 (26k→18k), then plateau at mk=3+.
Diminishing returns — boundary neurons oscillate between 2 clusters, not more.

#### Settled-phase analysis (skip first 5k ticks)

The initial turbulence (k-means init + settling) inflates the difference. After
tick 5000, when clusters have fully converged:

| Metric | mk=1 | mk=2 | mk=3 | mk=4 | mk=10 |
|---|---|---|---|---|---|
| Oscillators | 14.8% | 14.8% | 16.0% | 16.2% | 19.8% |
| Mean changes | 1.4 | 1.4 | 1.6 | 1.6 | 1.7 |
| Never changed | 32.9% | 30.2% | 25.5% | 25.0% | 23.5% |
| >3 changes | 7.0% | 6.3% | 8.4% | 9.3% | 12.0% |

**Key finding:** In steady state, mk=1 and mk=2 are identical (14.8%
oscillators both). The ring doesn't reduce steady-state oscillations — it
absorbs initial turbulence (faster settling). Higher mk actually *increases*
instability (mk=10: 19.8%) because neurons accumulate registered clusters,
causing more primary switches.

**Conclusion:** mk=2 is the practical sweet spot. Same steady-state behavior as
mk=1, but faster settling and 30% fewer new-cluster jumps overall. mk>2 adds
overhead without benefit.

### Runs 016–017: 20k warm-start with wandb (mk=1 vs mk=2)

Longer runs with wandb logging to see the jump/switch rate curves over time.
Also split the jump counter: `total_jumps` counts new-cluster ring writes only,
`total_switches` counts in-ring primary changes separately.

```
warm-start from Run 001, 20k ticks, m=640, lr=1.0, h=0.0, report_every=500
wandb: mk1_20k (5rn8xc5i), mk2_20k (eusp47pw)
```

| Metric | mk=1 | mk=2 |
|---|---|---|
| New-cluster jumps | 56,583 | **30,507** |
| In-ring switches | 0 | 27,555 |
| Jumps/tick (final) | 2.9 | **0.6** |
| Switches/tick (final) | 0.0 | 0.9 |
| Stability | 0.832 | **0.908** |
| K10 <3px | 98.4% | **99.5%** |

mk=2 cuts new-cluster jumps by 46% (57k→31k). The ~27k "saved" jumps become
in-ring switches — the neuron bounces between its two registered clusters
without entering new territory. By tick 20k, mk=2 has 0.6 jumps/tick vs 2.9
for mk=1, while switches settle at ~0.9/tick (boundary neurons oscillating
within their ring, which is cheap and expected).

Total primary changes (jumps+switches) are similar: mk=1 has 57k, mk=2 has 58k.
The ring doesn't reduce total movement — it reclassifies half of it as harmless
in-ring switching.

### Runs 018–019: 300k warm-start (mk=1 vs mk=2)

Long runs to see if the jump rate settles or keeps growing. Wandb logging for
continuous graphs.

```
warm-start from Run 001, 300k ticks, m=640, lr=1.0, h=0.0, report_every=5000
wandb: mk1_300k (hjgnt0gh), mk2_300k (j6hjm05i)
```

| Metric | mk=1 | mk=2 |
|---|---|---|
| New-cluster jumps | 929,877 | **369,082** |
| In-ring switches | 0 | 536,360 |
| Jumps/tick (steady) | ~3.1 | **~1.2** |
| Switches/tick (steady) | 0 | ~1.9 |
| Total primary changes | 929,877 | 905,442 |
| Splits | 1 | **0** |
| Stability (5k window) | ~0.50 | ~0.51 |
| K10 <3px | 98.2% | 98.0% |

**Key findings:**

1. **mk=2 cuts new-cluster jumps by 60% at 300k** (930k→369k). The benefit
   grows over time: 46% at 20k → 60% at 300k. The ring accumulates more
   registrations, absorbing more would-be jumps as switches.

2. **Jump rate never settles for mk=1** — steady ~3.1 jumps/tick from tick 10k
   through 300k. Boundary neurons keep discovering new clusters. mk=2 holds at
   ~1.2 jumps/tick — still nonzero, but 60% lower.

3. **Total primary changes are equal** (930k vs 905k). The ring doesn't reduce
   boundary movement — it reclassifies 60% of jumps as in-ring switches. The
   neuron still oscillates, but without entering new territory each time.

4. **Zero splits for mk=2.** No cluster ever died in 300k ticks. mk=1 had 1
   split (likely during initial settling). The primary-switch design keeps all
   clusters healthy indefinitely.

5. **Stability metric plateaus at ~0.50 for both.** This measures primary
   changes (jumps + switches) between 5k-tick windows. Since total primary
   changes are equal, stability is equal — the ring's benefit is invisible to
   this metric. Need to track new-cluster jumps separately for meaningful
   stability comparison.

### Per-neuron tenure analysis (Runs 016–017, 20k)

How long does a neuron stay in the same cluster before switching primary?
Analyzed from history snapshots (every 500 ticks), skipping first 1000 ticks.

| Tenure | mk=1 | mk=2 |
|---|---|---|
| 1 snapshot (500 ticks) | 20.2% | 21.8% |
| 2 snapshots (1000 ticks) | 17.8% | 17.2% |
| 3-4 snapshots (1.5-2k ticks) | 21.0% | 21.1% |
| 5-9 snapshots (2.5-4.5k ticks) | 23.5% | 23.3% |
| 10-19 snapshots (5-9.5k ticks) | 13.2% | 12.2% |
| 20+ snapshots (10k+ ticks) | 4.3% | 4.4% |

Mean tenure ~2800 ticks for both. Distributions are nearly identical because
history tracks primary changes (including in-ring switches for mk=2).

### Runs 020–023: Jump-only tenure analysis (mk=1,2,3,4, 20k)

Added per-neuron `jump_counts` array that increments only on new-cluster ring
writes (not in-ring switches). Saved in `history_jumps.npy` — cumulative jump
count per neuron at each snapshot. This separates genuine cluster exploration
from harmless boundary oscillation.

```
warm-start from Run 001, 20k ticks, m=640, lr=1.0, h=0.0, report_every=500
Analysis window: tick 2500-20000 (skip initial settling)
```

#### New-cluster jumps per neuron

| Metric | mk=1 | mk=2 | mk=3 | mk=4 |
|---|---|---|---|---|
| Mean jumps/neuron | 6.9 | **3.8** | 2.9 | 2.6 |
| Median jumps/neuron | 7 | 3 | 3 | 3 |
| Jump tenure (mean ticks) | 4,174 | **6,716** | 7,801 | 8,394 |
| Never jumped | 3.7% | 3.5% | 4.2% | 4.3% |
| 6+ jumps (heavy oscillators) | 61.7% | 20.2% | 5.1% | **1.4%** |
| 11+ jumps | 17.6% | 0.7% | 0.0% | 0.0% |

#### Jump distribution (new-cluster only)

| Jumps | mk=1 | mk=2 | mk=3 | mk=4 |
|---|---|---|---|---|
| 0 | 3.7% | 3.5% | 4.2% | 4.3% |
| 1-2 | 9.6% | 28.4% | 37.4% | 43.7% |
| 3-5 | 25.0% | 48.0% | 53.2% | 50.7% |
| 6-10 | 44.1% | 19.5% | 5.1% | 1.4% |
| 11+ | 17.6% | 0.7% | 0.0% | 0.0% |

**Key finding:** The ring buffer doubles cluster tenure. Neurons stay in their
cluster 4.2k ticks (mk=1) vs 6.7k (mk=2) vs 7.8k (mk=3) before entering a
genuinely new cluster. Heavy oscillators (6+ jumps) drop from 62% → 20% → 5%.
Primary tenure is similar (~4.5k ticks for all) because in-ring switches fill
the gap — total boundary movement is unchanged, but the ring keeps neurons in
familiar territory.

mk=2 captures most of the benefit: jump tenure +61% (4.2k→6.7k), heavy
oscillators -68% (62%→20%). mk=3/4 add diminishing returns.

## Known bugs / edge cases

1. **Last neuron can leave via in-ring switch.** When `sizes[primary] == 1` and
   best is already in the ring, the min_size=0 guard allows the switch.
   `sizes[primary]` drops to 0 — cluster dies. The neuron didn't enter a new
   cluster, just flipped primary within its ring. Rare in practice (zero splits
   in 300k warm-start runs), but structurally possible.

2. **Stale ring entries after split reuses dead cluster ID.** When a cluster
   dies and split reuses its ID (`dead = empty[0]`), the new cluster gets a
   fresh centroid and members. But neurons that previously visited the old
   cluster may still have its ID in their ring slots. If they later do an
   in-ring switch to that ID, they switch primary to what is now a completely
   different cluster (different centroid, different location). The LRU
   timestamp on the stale slot predates the cluster's death, which could be
   used to detect and invalidate stale entries.

# ts-00021: Closing the Loop — Column Output as Neuron Input

**Date:** 2026-03-19
**Status:** In progress
**Source:** `exp/ts-00021`

## Goal

Close the perception-action loop by feeding column outputs back into the
signal buffer as additional input. Currently the signal flows one way:

```
saccade crop → neurons → clusters → columns → (dead end)
```

The column outputs have nowhere to go. This experiment extends the signal
buffer so that a portion of each tick's signal comes from the external world
(saccade images, as before) and another portion comes from the previous
tick's column outputs — creating a recurrent loop:

```
        ┌──────────────────────────────────────┐
        │                                      ▼
saccade crop → neurons[0..N-1] → clusters → columns
                  ▲                              │
                  │    neurons[N..N+K-1]          │
                  └──── (column output t-1) ──────┘
```

## Architecture

The signal buffer `T` currently has shape `(n, signal_T)` where every neuron
gets its value from the saccade crop (external input). We extend this:

- **External neurons** `[0, N)`: signal comes from saccade crop pixels, as
  before. These are the "sensory" neurons.
- **Feedback neurons** `[N, N+K)`: signal comes from the previous tick's
  column outputs. These are the "internal" neurons that carry top-down
  information back into the map.

With M clusters and `n_outputs` per column, the column output is
`(M, n_outputs)` = `M * n_outputs` scalar values. Each becomes the signal
for one feedback neuron. So `K = M * n_outputs`.

The feedback neurons participate in the same embedding space, get sorted by
the same topographic map, join clusters, and wire to columns — just like
sensory neurons. The only difference is where their signal comes from.

This means columns now receive a mix of:
- Raw pixel values from sensory neurons in their cluster
- Column output values from feedback neurons in their cluster

The feedback creates temporal depth — column outputs at tick `t` influence
cluster signals at tick `t+1`, which influence column outputs at tick `t+1`,
and so on. The system can develop internal representations that persist and
evolve beyond what the instantaneous saccade crop provides.

## Key Questions

1. Do feedback neurons self-organize into meaningful spatial positions in the
   topographic map, or scatter randomly?
2. Does the recurrent signal improve per-column differentiation (the weakness
   from ts-00020)?
3. Does the system develop stable attractor states — persistent internal
   patterns that survive across saccade positions?
4. How much feedback (K) relative to sensory input (N) is needed?

## Column Learning Dynamics

### Entropy-Scaled Learning Rate

Columns with uniform outputs (all 4 outputs ≈ 25%) learn at full rate to
differentiate quickly. Columns that have already differentiated learn slowly,
maintaining stability while still allowing gradual re-learning.

```
lr_col = lr_base * (entropy / max_entropy)
```

Controlled by `ENTROPY_SCALED_LR = True` in `column_manager.py`.

### Lower Temperature (0.5 → 0.2)

Default softmax temperature reduced from 0.5 to 0.2 for peakier winner-take-all
dynamics. Higher temperature keeps outputs near-uniform even when prototypes
diverge; lower temperature amplifies small differences into clear winners.

`--column-temperature 0.2` (was 0.5).

### Embedding Visualization

`--render-mode embed` saves `embed_NNNNNN.png` scatter plots at each
`cluster_report_every` interval alongside normal cluster maps. Shows all neurons
projected to 2D via PCA: sensory as small gray dots, feedback as larger colored
dots (color = column hue). Useful for tracking feedback neuron organization.

## Early Observations (10k ticks, 80×80, 10pp)

- Feedback neurons form a distinct cloud, completely separated from sensory
  neurons in embedding space. Zero mixed clusters — 589 pure-sensory, 474
  pure-feedback.
- Within-cluster input spread ≈ within-column output spread (0.35 vs 0.35),
  but outputs are NOT near their driving inputs (cosine 0.22, distance 2.1).
- Same-column feedback neurons cluster tighter (0.35) than random feedback
  pairs (0.54) — column identity is captured.
- Column prototypes are angularly well-separated (mean intra-column cosine ≈ 0)
  but softmax outputs were near-uniform for 46% of columns at temperature=0.5.

## Results

### Run 008: 80×80, 10pp, 10k ticks (temp=0.5, no entropy lr)

Config: `--cluster-neurons-per 10 --column-outputs 4 --column-feedback --lr 0.01`
M=1066, K=4264, n_total=10664. 29ms/tick, 336s total.

Clustering: 589/1066 alive, contiguity=0.387, diameter=26.3, stability=0.436.

**Embedding separation:** Complete segregation — zero mixed clusters. 589
pure-sensory, 474 pure-feedback. Feedback neurons form their own cloud in a
separate region of embedding space (centroid distance 1.67).

**Embedding statistics:**
- Sensory intra-distances: mean=1.53, per-dim std≈0.40
- Feedback intra-distances: mean=0.54, per-dim std≈0.13 (3× tighter)
- Cross distances: mean=2.02 (larger than either intra)
- Same-column output spread: 0.35 (tighter than random feedback pairs 0.54)
- Output-to-input-centroid cosine: 0.22 (weak alignment)

**Cluster composition:** Sensory clusters avg 10.9 neurons, feedback clusters
avg 9.0 neurons. Both evenly distributed despite complete type segregation.

**Column differentiation (temp=0.5):** Prototype directions well-separated
(mean intra-column cosine ≈ 0), but softmax outputs near-uniform for 46% of
columns. Max output probability: mean=0.387, 22% have clear winner (>0.50).
Winner distribution skewed toward output 0: [619/163/145/136].

### Run 009: 80×80, 10pp, 100k ticks (temp=0.5, no entropy lr)

Config: same as 008 but 100k ticks.

Clustering: 567/1066 alive, contiguity=0.496, diameter=19.4, stability=0.571.
Winner dist [442/212/222/190] — still skewed toward output 0.

Embedding separation persists at 100k — two distinct clouds on opposite
corners of PCA projection. No mixing emerged with longer training.

### Key Finding: Feedback Loop Does Not Close

The derivative-correlation metric used for neighbor discovery produces
fundamentally different signal profiles for sensory neurons (pixel crops in
[0,1]) vs feedback neurons (softmax probabilities in [0,1]). The temporal
derivatives have different variance structure, so the two populations never
correlate with each other. Result: they form completely isolated embedding
regions and never share clusters.

The feedback neurons do self-organize — same-column outputs cluster together,
column identity is captured in embedding directions — but this is an isolated
system that doesn't interact with the sensory representation.

### Run 011: 16×16, garden, 10pp, 5k ticks (temp=0.2, entropy-scaled lr)

Config: `--preset gray_16x16_garden -f 5000`
M=42, K=168, n_total=424. 47ms/tick, 279s total.

Clustering: 22/42 alive, contiguity=0.970, diameter=4.7, stability=0.84.
Eval: PCA=0.51, K10: mean=1.93, **94.5% within 3px, 100% within 5px**.
Winner dist [11/12/12/7] — well balanced columns.

Feedback neurons split into 2-3 clearly separated sub-clusters, each with
distinct column colors. Sensory neurons form a structured loop/manifold.

**Embedding at tick 500 (early):**

![embed_16_500](embed_16_500.png)

**Embedding at tick 5000 (converged):**

![embed_16_5000](embed_16_5000.png)

**Clusters at tick 500 (early):**

![clusters_16_500](clusters_16_500.png)

**Clusters at tick 5000 (converged):**

![clusters_16_5000](clusters_16_5000.png)

### Run 010: 80×80, 10pp, 100k ticks (temp=0.2, entropy-scaled lr)

*(running — lr=0.001 matching presets)*

### XOR Synthetic Benchmark (runs 012-015)

**Setup:** 16×16 grid, `--signal-source xor`. Four quadrants with binary
features A, B, XOR=A^B, AND=A&B. Each tick draws random bits, held for 5
ticks. Tests whether columns can detect non-linear (XOR) features.

**Results across configs:**

| Run | lr    | batches | A    | B    | XOR  | AND  |
|-----|-------|---------|------|------|------|------|
| 013 | 0.001 | 1       | 0.11 | 0.26 | 0.20 | 0.20 |
| 014 | 0.001 | 2       | 0.11 | 0.26 | 0.17 | 0.20 |
| 015 | 0.01  | 2       | 0.17 | 0.29 | 0.17 | 0.22 |

Max |r| between any column output and each feature (500-tick sample after
10k training). All correlations at noise floor (~0.1-0.3), invariant to lr
and anchor count.

**Root cause:** Columns are per-cluster, and correlation-based clustering
separates A, B, XOR, AND into different spatial clusters. No column ever
sees neurons from multiple regions simultaneously, so no column can compute
cross-region functions like XOR.

For XOR detection, a column would need inputs from both region A and region B,
which requires them to share a cluster. But A and B are uncorrelated signals →
they cluster separately. The feedback loop doesn't help because feedback
neurons also form isolated clusters (see Key Finding above).

**Architectural implication:** The current per-cluster column architecture can
only detect features that are **local to one cluster** (i.e., variations among
neurons that already correlate enough to cluster together). Cross-cluster
non-linear features require either: (1) a mechanism to route information
between clusters (lateral connections, attention), or (2) a hierarchical layer
where cluster-level representations are combined.

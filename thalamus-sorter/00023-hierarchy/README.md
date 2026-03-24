# ts-00023: Visual Hierarchy Formation

**Date:** 2026-03-22
**Status:** In progress
**Source:** `exp/ts-00023`
**Depends on:** ts-00021 (feedback loop), ts-00022 (lateral connections)

## Goal

Observe that visual processing hierarchy forms naturally:
- **V1** — clusters that directly process sensory pixel input
- **V2** — clusters that process V1's output (feedback neurons from V1 columns)
- **V3+** — deeper processing layers

The feedback loop already creates this structure: sensory neurons form
V1 clusters with columns, column outputs become feedback neurons that
form V2 clusters with their own columns. The question: does V2 capture
higher-level features than V1?

## Architecture

```
Pixels → V1 clusters → V1 columns → feedback neurons →
    V2 clusters → V2 columns → deeper feedback → V3 ...
```

With M clusters, K = M × n_outputs feedback neurons:
- Pure sensory clusters = **V1** (directly see pixels)
- Feedback clusters whose source columns sit on V1 = **V2**
- Feedback clusters whose source columns sit on V2 = **V3**

The hierarchy depth depends on how many layers of feedback→cluster→column
the system develops. With enough clusters and feedback neurons, multiple
layers should emerge naturally.

## Metrics

### Layer identification

For each cluster, trace its neurons:
- If it contains sensory neurons → V1
- If it contains feedback neurons from V1 columns → V2
- If it contains feedback neurons from V2 columns → V3
- Mixed clusters span multiple layers

### Feature complexity per layer

V1 should detect local features (pixel correlations in small patches).
V2 should detect combinations of V1 features (larger spatial patterns).

Measure: **receptive field size** per layer.
- V1 neuron: correlates with a small pixel patch
- V2 neuron: correlates with a larger region (union of V1 receptive fields)

### Information flow

Track which V1 columns' outputs end up in which V2 clusters.
The lateral connections between V2 columns should span broader
spatial regions than V1 lateral connections.

## Benchmark: LAYERS

Synthetic signal with explicit hierarchy:
- **L1** (8 groups): each group gets an independent random value
- **L2** (4 groups): each is the XOR/AND/OR of two L1 groups
- **L3** (2 groups): each is a function of two L2 groups

V1 columns should track L1 features. V2 columns should track L2.
The system must form at least 2 processing levels.

## Verification

1. Run with feedback enabled on natural image (garden)
2. Classify each cluster as V1/V2/V3 based on neuron type
3. Measure receptive field size per layer
4. Check if V2 columns track broader features than V1

## LAYERS benchmark

16×16 grid (256 neurons). 8 neurons per signal group:
- L1: 8 groups (64 neurons) — independent binary signals
- L2: 4 groups (32 neurons) — XOR of L1 pairs (L2_0 = L1_0^L1_1, etc.)
- L3: 2 groups (16 neurons) — XOR of L2 pairs (L3_0 = L2_0^L2_1, etc.)
- 144 zero neurons (unused)

No spatial/visual field — purely abstract grouped signals. Each level
requires combining information from the previous level. L1 can be
detected locally. L2 requires cross-cluster combination (like XOR
benchmark). L3 requires combining L2 outputs.

Analysis classifies each cluster as V1 (contains sensory neurons),
V2 (contains feedback from V1 columns), or V3 (feedback from V2
columns). Then measures which layer detects which feature level.

## Results

### Run 001: 10k ticks, T=1000, lateral K=2, mix 10%

Config: `--signal-source layers --layers-hold 50 --column-lateral
--predictive-shift 1 --predictive-mix 0.1 --lr 0.01 -f 10000`

**Hierarchy formed:** V1=12 clusters, V2=30 clusters, V3=0 clusters.

| Level | Features | Mean r | Detected by |
|-------|----------|--------|-------------|
| L1 | 8 binary | 0.32 | V1 and V2 |
| L2 | 4 XOR | 0.35 | **V2 column detects L2_1 at r=0.45** |
| L3 | 2 XOR² | 0.29 | V1 only (V3 not formed) |

**Key finding:** V2 column 25 detects L2_1 (XOR of L1_2 and L1_3)
at r=0.45 — a feedback column detecting a composite feature that
no single V1 column can compute. This IS the visual hierarchy
working: V1 processes raw signals → V2 processes V1 output →
V2 detects combinations.

L3 features detected by V2 in second run (r=0.30) — V2 columns
detecting L3 composite features through lateral connections.

### Cluster analysis: poor cohesion

Each group of 8 identical neurons scatters across 5-8 clusters.
Same-signal neurons don't cluster together — they get mixed with
neurons from other groups and especially with 144 zero neurons
(56% of grid is dead weight, diluting clusters).

Example: cluster 4 contains L1_1, L1_3, L1_6, L2_0, L2_1, L2_3,
L3_1 AND zero neurons — a mixed soup, not a clean feature detector.

**Root cause:** 144 zero neurons dominate clustering. 8 neurons per
group can't form coherent clusters when competing with zeros. Need
either smaller grid (no zeros) or more neurons per group.

## EDGES benchmark — spatial visual hierarchy

Synthetic image (128×128) with checkerboard, lines, rectangles, circles.
16×16 saccade crops as signal. Tests whether V1→V2→V3 forms from
real spatial edge correlations.

### Hierarchy depth scales with M

| M | V1 | V2 | V3 | V1 diam | V2 RF |
|---|----|----|----|---------| ------|
| 42 | 24 | 15 | 3 | 3.9 | 10.6 |
| 80 | 35 | 35 | 10 | 4.6 | 8.7 |
| 120 | 43 | 61 | 16 | 3.9 | 8.3 |
| 200 | 69 | 110 | 27 | 10.4 | 10.9 |

V3 grows with M: 3→10→16→27. No V4 formed — V3 feedback neurons
get absorbed back into existing V1/V2 layers.

### V1 output distribution (M=200)

Each V1 column produces 4 outputs (SoftWTA). These 4 feedback neurons
scatter into DIFFERENT V2 clusters:
- **0/67** V1 clusters have all 4 outputs in the same V2 cluster
- **49/67** (73%) have all 4 in different V2 clusters
- Each output category carries different information → embeds differently

### V2 combines multiple V1 sources

- **62/106** (58%) V2 clusters receive feedback from 2-5 different V1 clusters
- Spatial spread of V1 sources: 3-15 pixels (mean ~8-9, half the grid)
- V2 therefore integrates information across broad spatial regions

This IS visual hierarchy:
1. V1 detects local edges (diameter ~4 pixels)
2. V1 columns differentiate into 4 categories per cluster
3. Each category finds a different V2 cluster
4. V2 clusters collect categories from multiple V1 regions
5. V2 has 2-3× broader receptive fields than V1

## SHAPES benchmark — visual + categorical association

160×160 image with 8 distinct shapes (square, circle, cross, lines,
checkerboard, diagonal). 16×16 saccade crop (256 pixels) + 64 label
neurons (8 per shape, one-hot when crop overlaps that shape).

Grid: 20×16 = 320 neurons total.

### Results

| Config | V1 | V2 | V3 | Shape detection | Label cohesion |
|--------|----|----|----|----|---|
| M=80, 10k | 45 | 31 | 4 | shape6 r=0.77 | scattered |
| M=200, 100k | 47 | 50 | **59** | shape5 r=0.66 | **perfect (8/8)** |

**V3 is the largest layer at M=200** — 59 clusters processing V2 output.
Three-layer hierarchy confirmed. All 8 shape labels achieve perfect
cohesion (one V1 cluster each).

Shape detection is through V1 columns — the label neurons cluster with
the pixels that co-activate when that shape is visible. V2 and V3
process these associations further but the analysis currently only
measures shape identity correlation, not hierarchical feature complexity.

### Deep hierarchy: 6 layers at M=200

Extended layer classification beyond V3 (tracing feedback→cluster→column
chains iteratively) reveals **6 processing levels**:

| Layer | Clusters | Role |
|-------|----------|------|
| V1 | 47 | Sensory (pixels + labels) |
| V2 | 50 | Feedback from V1 columns |
| V3 | 59 | Feedback from V2 columns (largest) |
| V4 | 33 | Feedback from V3 columns |
| V5 | 9 | Feedback from V4 columns |
| V6 | 1 | Feedback from V5 columns |

Total: 199/200 clusters classified. V3 remains the largest layer — the
system allocates the most processing capacity at 3 levels of indirection
from raw input.

### Shape information trace through hierarchy

Each shape's 8 label neurons achieve perfect cohesion in V1 (one cluster
per shape). Tracing their column outputs through the feedback chain:

1. **V1**: Each shape occupies one dedicated cluster. Column outputs (4 per
   cluster) fan out to different V2 clusters.
2. **V2**: Shape information spreads — each shape's V1 outputs reach 2-4
   different V2 clusters. Multiple shapes' outputs converge in shared V2
   clusters.
3. **V3**: Further fan-out and convergence. V3 clusters receive from
   multiple V2 sources, mixing shape representations.
4. **V4**: Strongest convergence — shapes that were separate at V1 now
   share processing clusters.

### Convergence at higher layers

Shapes that are spatially distinct at V1 converge at higher processing
levels. Example: shape 1 (hollow square) and shape 5 (cross) share V2,
V3, and V4 clusters — the system discovers that different visual patterns
feed into shared higher-order representations.

This convergence pattern mirrors biological visual hierarchy: V1 neurons
are selective (one shape per cluster), while higher areas become
increasingly invariant (multiple shapes sharing clusters). The system
self-organizes this structure without any supervision — purely from
temporal correlations in saccade-driven input.

## XOR blind evaluation

Previous XOR benchmark was circular: XOR value was in the signal during
both training AND evaluation. A column sitting on XOR-quadrant neurons
trivially reads the answer.

**Fixed evaluation:** during analysis, XOR and AND quadrants are zeroed
out. Only A and B are provided. "Blind" columns = those with NO
XOR/AND sensory neurons.

### M-sweep results (10k ticks each)

| M | Blind cols | XOR blind r | AND blind r | A r | B r |
|---|---|---|---|---|---|
| 10 | 2/10 | 0.15 | 0.13 | 0.16 | 0.24 |
| 42 | 18/42 | 0.19 | 0.36 | 0.24 | 0.30 |
| 80 | 44/80 | 0.29 | 0.45 | 0.32 | 0.37 |
| 150 | 108/150 | 0.39 | 0.42 | 0.34 | 0.40 |
| 200 | 156/200 | 0.39 | 0.40 | 0.36 | 0.39 |
| 300 | 243/300 | 0.36 | 0.46 | 0.35 | 0.39 |
| 500 | 418/500 | 0.36 | 0.46 | 0.37 | 0.39 |

XOR blind detection scales with M up to ~150, then **plateaus at r≈0.39**.
AND (linear combination) reaches 0.46 and also plateaus. More clusters
beyond M=150 doesn't help — the bottleneck is the learning rule, not
capacity. Power iteration finds principal components of covariance
(linear). XOR requires multiplicative interaction that covariance-based
Hebbian learning can't directly compute. The ~0.39 likely comes from
partial correlation via lateral connections between A and B clusters.

### K-means column mode

Replaced power iteration (linear) with k-means centroids (non-linear
partitioning). Similarity = negative squared distance to centroid.
Learning = nudge winner centroid toward mean input pattern.

| M | Variance XOR | **K-means XOR** | Variance AND | **K-means AND** |
|---|---|---|---|---|
| 10 | 0.15 | 0.15 | 0.13 | 0.16 |
| 42 | 0.19 | **0.48** | 0.36 | 0.40 |
| 80 | 0.29 | **0.47** | 0.45 | **0.56** |
| 150 | 0.39 | **0.53** | 0.42 | **0.58** |
| 200 | 0.39 | **0.58** | 0.40 | **0.59** |

K-means blind XOR: **0.58 vs 0.39** at M=200. Still climbing (no plateau).
K-means can distinguish [A=1,B=0] from [A=1,B=1] because they're different
points in input space — different distances to centroids. Power iteration
can't because both project identically onto a linear direction.

A and B detection also jumps to r>0.9 with k-means — exact pattern matching
vs variance direction gives much sharper linear feature detection too.

### Full benchmark suite with k-means columns

M=100, 10k ticks, k-means column mode, lateral connections, feedback loop.

| Benchmark | Key metric | r | Notes |
|-----------|-----------|---|-------|
| echo | voice/echo detection | 0.84 | Delayed signal tracking |
| edges | V1/V2/V3 hierarchy | 41/36/21 | V2 RF=7.5 vs V1 diam=3.3 (2.3×) |
| layers | L1/L2/L3 features | 0.89/0.84/0.71 | All detected by V1 columns |
| majority | MAJ(A,B,C) | 0.65 | Non-linear majority vote |
| match | pattern EQ | 0.47 | Spatial pattern comparison |
| mirror | self-tracking | **1.00** | Perfect action-consequence |
| oddball | novelty detection | 0.50 | Odd-one-out identification |
| sequence | temporal order | **0.71** | Detects A-then-B vs B-then-A |
| xor | blind XOR | 0.58 | Non-linear (M=200) |
| shapes | V2+ shape detect | 0.55 | Visual hierarchy + labels |
| forage | collections | 88/100k | hunger r=0.96, direction r=0.83 |

K-means mode is now default. Mirror is perfect (r=1.0) — k-means
excels at tracking output patterns. Sequence (r=0.71) shows temporal
order detection. Layers detects all 3 feature levels with L1 at r=0.89.
Edges forms 3-layer hierarchy with V2 receptive fields 2.3× V1 diameter.

### Column mode experiments

**Derivative inputs** (k-means on frame-to-frame diffs instead of means):
Worse across the board. With hold=50, most frames within the window are
identical → derivatives mostly zero → k-means matches noise. Mean state
is the right representation for these benchmarks.

**Window=1** (instantaneous, no averaging): Worse — hunger drops 0.96→0.67,
collections 88→60. The 10-frame mean acts as a low-pass filter that
helps k-means centroids converge.

**Contrastive push** (pull winner + push losers away from input):
XOR improves +15% (0.48→0.55) but echo -8%, mirror -1%. Push forces
better differentiation but is unbounded — loser prototypes can diverge
exponentially since `proto -= lr*(input - proto)` has a `(1+lr)` growth
factor. Better winner distribution (33/21/22/24 vs 51/22/17/10) but
less stable tracking. Reverted — needs bounded formulation before
production use.

**Output tiredness** (rate=0.001, ~1k ticks to fatigue):
Winners accumulate tiredness, losers recover. Tired winners get
suppressed → eventually yield to rested runners-up. Forces category
exploration and creates temporal dynamics visible to embeddings.

| Benchmark | Key metric | No tiredness | **With tiredness** |
|-----------|-----------|-------------|-------------------|
| xor | blind XOR r | 0.48 | **0.66** (+38%) |
| echo | voice r | 0.84 | **0.85** |
| layers | L1 r | 0.89 | 0.88 |
| majority | MAJ r | 0.65 | **0.71** (+9%) |
| match | EQ r | 0.47 | **0.49** |
| mirror | stimulus r | 0.83 | 0.82 |
| oddball | odd_val r | 0.50 | **0.66** (+32%) |
| sequence | SEQ r | 0.71 | **0.83** (+17%) |

Tiredness is a clear win across the board. Big improvements on
non-linear and temporal benchmarks (XOR, sequence, oddball) while
linear tracking (mirror, echo, layers) holds steady. The winner
rotation creates richer temporal dynamics for the embedding system.

### Tiredness rate sweep (XOR blind, 10k ticks)

| Rate | ~ticks to tire | XOR blind r | AND blind r |
|------|---------------|-------------|-------------|
| 0 (off) | ∞ | 0.58 | 0.59 |
| 0.0001 | 10k | 0.51 | 0.56 |
| 0.0005 | 2k | 0.51 | 0.62 |
| **0.001** | **1k** | **0.66** | **0.84** |
| 0.005 | 200 | 0.58 | 0.69 |
| 0.01 | 100 | 0.49 | 0.60 |
| 0.05 | 20 | 0.47 | 0.78 |

Clear peak at 0.001. Too slow (0.0001) = no effect, too fast (0.01+) =
too disruptive, centroids can't stabilize. 1k ticks matches the signal
buffer length (T=1000), so each output gets ~one full buffer of
dominance before yielding.

## Eligibility traces

Deferred learning: columns accumulate traces every tick (decay + winner
direction). When reward arrives, traces are applied to prototypes.
Normal unsupervised pull continues every tick regardless.

`set_reward(value)` called by benchmark → applied on next tick → reset.

### Forage results with eligibility (M=100, trace_decay=0.99)

| Config | Collections | Dense rate | Sparse rate | dir r |
|--------|------------|------------|-------------|-------|
| No eligibility (baseline) | 88/100k | 0.0116 | 0.0003 | 0.78-0.83 |
| Collection reward only | 56/100k | 0.0038 | 0.0004 | 0.64-0.71 |
| Collection + distance reward | 57/100k | 0.0072 | 0.0002 | 0.71-0.74 |
| Distance reward 1M | 88/1M | 0.0028 | 0.0001 | 0.76-0.77 |

Eligibility traces hurt overall throughput vs baseline at 100k. Distance-
based reward (small positive when getting closer to nearest POI) helps
vs collection-only reward.

### Extended training with spasm floor

Added spasm_floor=0.01 so agent always has baseline random walk.

| Training | Collections | Dense rate | dir r | hunger r |
|----------|------------|------------|-------|----------|
| 100k (no elig) | 88 | 0.0116 | 0.78-0.83 | 0.96 |
| 100k (elig+dist) | 57 | 0.0072 | 0.71-0.74 | 0.73 |
| 1M (elig+dist) | 88 | 0.0028 | 0.76-0.77 | 0.69 |
| 2M (elig+dist+floor) | **128** | 0.0031 | **0.80** | **0.90** |

System continues improving with more training — no plateau at 2M.
Spasm floor ensures continued exploration in sparse phase (97/128
collections from sparse phase). Hunger tracking recovers to r=0.90
with extended training.

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

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

## Results

*(pending)*

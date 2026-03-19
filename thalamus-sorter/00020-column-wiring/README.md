# ts-00020: Column Wiring — Thalamus-to-Cortex Connection

**Date:** 2026-03-18
**Status:** In progress
**Source:** `exp/ts-00020`

## Goal

Wire the thalamus-sorter's cluster output to the column (cortical minicolumn)
model as input. The thalamus produces a topographic map where spatially
contiguous clusters track visual features via mean signal. The column takes a
low-dimensional input vector and categorizes it via soft winner-take-all
competition with online Hebbian learning.

This creates a two-stage visual processing pipeline:

```
saccade crop (160×160 pixels)
    → thalamus: 25,600 neurons, 2,560 clusters
        → cluster signals: 2,560-dim vector (mean intensity per cluster)
            → column: N_outputs soft-WTA categories
                → winner = "what am I looking at"
```

## Architecture

### Thalamus (existing)

- 160×160 grayscale saccade input → 25,600 neurons
- 2,560 clusters (mk=2, LRU, primary-switch)
- Each cluster averages ~10 pixels → mean signal = local brightness
- Contiguity=1.000 → clusters are spatially contiguous patches
- Output: 2,560-dim vector of per-cluster mean signal values

### Column (existing)

- `SoftWTACell(n_inputs, n_outputs, temperature, lr, ...)`
- Takes input vector, computes dot-product similarity against prototypes
- Softmax competition → winner's prototype moves toward input (Hebbian)
- Usage counters prevent collapse
- Temporal modes: instantaneous, correlation, streaming variance

### Wiring

The thalamus cluster signal vector (2,560 float values) becomes the column's
input. Each saccade crop position produces a different input vector — the
column should learn to categorize different visual scenes.

Key questions:
1. How many column outputs (categories)? Start with 8-16.
2. Does the column learn meaningful visual categories from cluster signals?
3. Does temporal mode (streaming variance) help — saccade-driven temporal
   patterns might carry information about scene structure.
4. How to evaluate? Visual inspection of which scenes map to which category.

## Implementation plan

1. **Adapter script** that runs both systems in a loop:
   - Thalamus processes saccade crop → cluster signals
   - Cluster signals → column input
   - Column forward + learn
   - Log column outputs, prototypes, usage

2. **Start simple**: use a pre-trained thalamus model (warm-start), feed
   cluster signals to an instantaneous-mode column.

3. **Iterate**: try temporal mode, different n_outputs, different temperatures.

## Results

*(pending)*

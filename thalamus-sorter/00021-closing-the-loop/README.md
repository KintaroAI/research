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

*(pending — running 100k with entropy-scaled lr and temperature=0.2)*

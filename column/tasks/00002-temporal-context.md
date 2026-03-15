# 00002 — Temporal Context Support

**Date:** 2026-03-15
**Status:** Complete

## Context

Requirement 1 specifies optional temporal context: input can be an `(n, T)` matrix where
each input carries a trace of `T` recent values. This enables richer similarity — distinguishing
"both silent" from "both co-varying" — which instantaneous vectors cannot do.

The thalamus-sorter project already has derivative correlation buffers that solve this same
problem. The key question is what similarity measure to use over temporal traces.

## Problem

Current SoftWTACell takes `(n,)` vectors. Prototypes are `(m, n)`. Similarity is dot product
on normalized vectors. This works for instantaneous snapshots but loses temporal structure.

## Design Options

### Option A: Flatten
- Input `(n, T)` → flatten to `(n*T,)`, prototypes become `(m, n*T)`
- Simple but destroys temporal structure, prototypes grow large
- Probably not what we want

### Option B: Temporal similarity reduction
- Prototypes stay `(m, n)` as representative "state vectors"
- Similarity computed by reducing the `(n, T)` trace to a single similarity score per prototype
- Candidates:
  - **Correlation:** per-input-channel correlation between trace and prototype's expected trace → mean across channels
  - **MSE over trace:** how well does the prototype match the recent trajectory
  - **Derivative correlation:** correlate rate-of-change patterns (from thalamus-sorter)

### Option C: Temporal prototypes
- Prototypes are `(m, n, T)` — each prototype stores a full trace template
- Similarity = mean correlation across channels between input trace and prototype trace
- Richer but more memory, slower learning

## Recommended: Option B first, then C if needed

Option B keeps prototypes small and compatible with the instantaneous case (T=1 degrades
to current behavior). The similarity function becomes pluggable:

```python
def temporal_similarity(x_trace, prototype, method='correlation'):
    """x_trace: (n, T), prototype: (n,) or (n, T)"""
```

## Deliverables

1. Extend `SoftWTACell.__init__` with `temporal_mode` parameter (None, 'correlation', 'mse')
2. `forward()` accepts `(n,)` or `(n, T)` input transparently
3. Similarity function dispatches based on mode
4. Update prototypes: for Option B, winner prototype still pulled toward the mean of the trace
5. Tests: verify T=1 matches current behavior, T>1 separates correlated vs uncorrelated signals

## Open Questions

- Should prototypes store temporal templates (Option C) or just state vectors (Option B)?
- What T values are practical? Likely 5–20 based on thalamus-sorter experience
- Should the cell maintain its own trace buffer, or does the caller manage that?

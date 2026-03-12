# ts-00011: Variance Weighting for MSE-Based Neighbor Scoring

**Date:** 2026-03-12
**Status:** In Progress
**Source:** *tagged on completion as `exp/ts-00011`*

## Goal

Add a variance component to MSE-based neighbor scoring that distinguishes "both dead" from "both active and co-varying" without compressing the 15x near/far MSE discrimination ratio.

## Motivation

From ts-00010: pure MSE thresholding discovers spatial structure from raw firing rates with no global operations. But MSE has a blind spot:

```
A = [0, 0, 0, 0]    B = [0, 0, 0, 0]    → MSE = 0  (both dead)
A = [0.8, 0.2, 0.5]  B = [0.8, 0.2, 0.5]  → MSE = 0  (genuinely co-varying)
```

Both get MSE=0 and are treated as perfect neighbors. In a cross-modal system where some neurons may be inactive for long stretches, dead pairs would pollute the neighbor graph.

### Why the previous attempt failed

The combined score `sqrt(var_A × var_B) × (1 - MSE)` compressed the discriminating MSE signal. With natural images:
- `(1 - MSE)` ranges 0.95-1.00 (only 5% spread)
- variance multiplier (≈0.05 for all pixels) squashes everything into 0.02-0.04
- The 15x near/far ratio in raw MSE is destroyed

## Approach candidates

### 1. Two-stage: MSE gates, variance weights

Keep `MSE < threshold` as the binary neighbor decision (preserves full 15x discrimination). Then weight the *learning update* by `min(var_A, var_B)`.

```
neighbor = MSE(A, B) < threshold
weight   = min(var_A, var_B)
gradient *= weight
```

Dead pairs pass the gate but produce near-zero gradient — they don't learn. Active pairs get full learning signal. The selection criterion is never compressed.

**Pros:** Clean separation of concerns. MSE ratio untouched.
**Cons:** Dead pairs still appear in the neighbor count, potentially affecting hit-ratio diagnostics.

### 2. Additive penalty

```
score = MSE + λ / (var_A + var_B + ε)
neighbor = score < threshold
```

Low-variance pairs get a penalty that pushes their score above threshold. This *shifts* MSE rather than compressing it. The 15x ratio between near (0.003) and far (0.05) is preserved — dead pairs just get bumped up past threshold.

**Pros:** Single score, clean threshold semantics.
**Cons:** Introduces hyperparameter λ. Need to calibrate so the penalty exceeds threshold for dead pairs but is negligible for active ones.

### 3. Hard gate on joint variance

```
neighbor = MSE(A, B) < threshold  AND  (var_A + var_B) > min_var
```

Simple binary filter. With natural images it's a no-op (all pixels have similar variance, which is fine). In cross-modal settings with dead neurons, it catches them.

**Pros:** Simplest. No new hyperparameters if min_var is set conservatively.
**Cons:** Binary — no soft transition. Borderline-variance neurons are either fully in or fully out.

## Results

*(to be filled)*

## Files

- `main.py` — variance weighting implementation
- `solvers/drift_torch.py` — `tick_correlation()` modifications

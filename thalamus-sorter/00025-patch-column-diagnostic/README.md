# ts-00025: Patch Column Diagnostic

**Date:** 2026-03-26
**Status:** Complete
**Source:** `exp/ts-00025`
**Depends on:** ts-00024 (modularized building blocks), ts-00023 (hierarchy)

## Goal

Understand how the model's clustering organizes column outputs by hardcoding the first layer. Two questions:

1. Do the 4 outputs from the same patch land in the same model cluster? (bad — means model groups anti-correlated signals)
2. Are the 4 column outputs actually different? (measurable via entropy and correlation)

## Architecture

```
Source image (saccades_gray.npy)
  ↓ saccade walk (step=35, 56×56 crops for 7×7 patches)
  ↓
First layer (HARDCODED, inside benchmark):
  56×56 crop → 64 non-overlapping 7×7 patches (8×8 grid, uses 56×56)
  Each patch → 1 ColumnManager column (4 outputs, k-means mode)
  Total: 64 × 4 = 256 outputs
  ↓
Model (NORMAL pipeline):
  256 "neurons" on 16×16 grid → DriftSolver → ClusterManager (16 clusters)
  ↓
Analysis:
  For each patch, check which model clusters its 4 outputs landed in
```

## Benchmark

Created `benchmarks/patch_column.py` with 4 metrics:

1. **Spread distribution** — for each patch, count unique model clusters among its 4 outputs. Histogram of {all-4-same, 2 clusters, 3 clusters, all-different}.
2. **Output entropy** — per-patch Shannon entropy of the 4 softmax outputs. High = differentiated, low = collapsed.
3. **Inter-output correlation** — per-patch mean pairwise Pearson correlation of the 4 output time series. High = column not differentiating.
4. **Winner distribution** — per-output global win rate and per-patch dominant winner fraction. Detects winner-take-all collapse.

Preset: `presets/patch_column_baseline.json` (16×16 grid, k_sample=50, cluster_m=16, no column_outputs/column_feedback).

```bash
python main.py word2vec --preset patch_column_baseline -f 10000
```

## Results

### Patch size comparison (10k ticks)

| Metric | 3×3 (9 in) | 5×5 (25 in) | 7×7 (49 in) | 9×9 (81 in) |
|---|---|---|---|---|
| Entropy (normalized) | 0.93 | 0.69 | 0.09 | 0.003 |
| Mean spread | — | 3.05 | 2.89 | 2.45 |
| Inter-output corr | — | -0.07 | -0.02 | -0.05 |
| Dominant winner % | — | — | 99.2% | — |

### Spread distribution at 7×7 (10k ticks)

```
all-4-same:    0/64
2 clusters:   15/64
3 clusters:   47/64
all-different:  2/64
Mean spread: 2.80 (1.0=collapsed, 4.0=differentiated)
```

### Winner distribution at 7×7 (10k ticks, 500-tick post-training sample)

```
Global:  28.1% / 20.3% / 34.3% / 17.3%
Per-patch dominant winner: mean=0.992
Output 0 dominates: 18/64 patches
Output 1 dominates: 13/64 patches
Output 2 dominates: 22/64 patches
Output 3 dominates: 11/64 patches
```

## Findings

### 1. Columns collapse to a single winner

After training, each column's dominant output wins ~99% of ticks. The other 3 outputs are permanently near-zero. This is winner-take-all collapse driven by softmax competitive dynamics with Hebbian learning.

### 2. More inputs → sharper collapse

With 9 inputs (3×3 patches), columns can't differentiate — entropy stays near-random at 0.93. With 49+ inputs, columns fully collapse (entropy <0.1). The sweet spot around 25 inputs (5×5) gives moderate differentiation but still trends toward collapse with more training.

### 3. The embedding algorithm groups silent outputs

The skip-gram DriftSolver measures temporal correlation. Three silent outputs (near-zero simultaneously) have trivially high correlation — they "fire together" by being quiet together. The model embeds them as neighbors and clusters them together. This is why spread is 2.8 not 4.0.

### 4. Sharper columns → worse model separation

Paradoxically, better-differentiated columns lead to worse model spread:
- 5×5 (moderate sharpness): spread 3.05
- 9×9 (fully collapsed): spread 2.45

Because when one output dominates, the 3 losers are even more uniformly silent, making them even more indistinguishable to the embedding algorithm.

### 5. Global winner imbalance

Outputs 0 and 2 win disproportionately (28-34%) vs outputs 1 and 3 (17-21%). This likely stems from the random initialization of k-means centroids — some centroids start closer to common input patterns.

## Root Causes

1. **Column collapse**: softmax + Hebbian learning creates positive feedback loop. Winner gets better → wins more → gets even better. Entropy-scaled lr is on but insufficient.

2. **Embedding blindness to silence**: DriftSolver correlation treats "both near-zero" as "both similar." It can't distinguish shared silence from shared signal.

## Next Steps

ts-00026: tune column parameters to prevent collapse:
- **tiredness_rate**: penalize consecutive wins, force output rotation
- **temperature scheduling**: start hot, cool down
- **usage_decay**: slower decay keeps dormant outputs alive longer
- **Explore derivative-correlation for column outputs**: may help embedding distinguish silent vs active

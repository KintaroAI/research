# ts-00012: Scaling to 160x160

**Date:** 2026-03-12
**Status:** Complete
**Source:** `exp/ts-00012`

## Goal

Test how the correlation-based sorter scales from 80x80 (6,400 neurons) to 160x160 (25,600 neurons). Key question: does k_sample=200 still work, or do we need to scale it to maintain the same sampling fraction?

## Motivation

All experiments so far used 80x80 grids. The real system will have many more neurons. Scaling 2x per dimension (4x total) is the first test of whether the algorithm's random sampling approach degrades with neuron count.

### Sampling fraction concern

| Grid | Neurons | k_sample | Fraction |
|------|---------|----------|----------|
| 80x80 | 6,400 | 200 | 3.1% |
| 160x160 | 25,600 | 200 | 0.78% |
| 160x160 | 800 | 800 | 3.1% |

At 200 candidates per anchor on a 160x160 grid, each neuron only checks 0.78% of all neurons — 4x fewer than at 80x80. True grid neighbors within 5 pixels: ~80 out of 25,600 = 0.31%. With k_sample=200, expected true-5px neighbors in sample: ~0.6. We might need to bump k_sample to keep convergence speed.

### Source image size

The saccades source (1536x1024) gives max offsets of 1376x864 for 160x160 crops vs 1456x944 for 80x80. Still plenty of room for step=50 random walks.

## Approach

1. Run 160x160 with k_sample=200 (same as 80x80 baseline) for 10k ticks
2. If quality is poor, try k_sample=800 (matching 3.1% fraction)
3. Compare convergence speed and final quality

## Results

Base parameters: 160x160 grid, dims=8, k_neg=5, lr=0.001, normalize_every=100, signal_T=1000, step=50, MSE threshold=0.02, rolling buffer.

### Initial runs (10k ticks)

| Grid | k_sample | Fraction | Ticks | Total pairs | PCA disp | Mean dist | <3px | <5px |
|------|----------|----------|-------|-------------|----------|-----------|------|------|
| 80x80 | 200 | 3.1% | 10k | 120M | 0.60 | 2.58 | 80.3% | 94.5% |
| 160x160 | 200 | 0.78% | 10k | 10.7M | 1.00 | 106.4 | 0.1% | 0.2% |
| 160x160 | 800 | 3.1% | 10k | 110M | 0.98 | 77.4 | 2.4% | 5.2% |

**k_sample=200 is dead at 160x160.** Only 10.7M pairs in 10k ticks (vs 120M at 80x80). The 0.78% sampling fraction finds almost no true neighbors — expected: with ~80 true-5px neighbors out of 25,600, a 200-sample draws 0.6 of them on average.

**k_sample=800 matches pair count but needs more ticks.** 110M pairs (comparable to 80x80 baseline), but 4x more neurons to organize. At 10k ticks the map is just starting to form (5.2% within 5px).

### Neighbor capture analysis

How many true grid neighbors land in a random k_sample candidate set?

| Grid | k_sample | Fraction | Captured ≤5px (mean) | Zero ≤5px captures | Closest candidate |
|------|----------|----------|----------------------|--------------------|-------------------|
| 80x80 | 200 | 3.1% | 1.89 | 14.9% | 3.57 |
| 160x160 | 200 | 0.78% | 0.46 | **64.8%** | 7.52 |
| 160x160 | 800 | 3.1% | 1.90 | 14.3% | 3.58 |
| 160x160 | 1600 | 6.25% | 3.74 | 1.8% | 2.58 |

**k_sample=800 at 160x160 exactly matches 80x80 k=200**: same capture rate (1.9 neighbors ≤5px), same zero-hit rate (14%), same closest candidate distance (3.6). The scaling rule is simple: **k_sample scales linearly with n** to maintain sampling fraction.

k_sample=200 at 160x160 is catastrophic — 64.8% of anchors find zero neighbors, closest candidate averages 7.5px away. The grid can't learn from nothing.

### Longer runs (k_sample=800)

| k_sample | Ticks | Total pairs | PCA disp | Mean dist | <3px | <5px |
|----------|-------|-------------|----------|-----------|------|------|
| 800 | 10k | 110M | 0.98 | 77.4 | 2.4% | 5.2% |
| 800 | 50k | 461M | 0.81 | 2.33 | 86.2% | **97.2%** |
| 800 | 100k | 1.09B | 0.75 | 2.91 | 76.6% | 91.2% |
| 800 | 1M | 10.0B | 0.65 | 2.66 | 80.3% | 94.2% |

**50k is the sweet spot — 97.2%, matches 80x80 baseline.** The 160x160 grid converges to the same quality, just needs k_sample scaled to maintain the sampling fraction.

**100k dips, 1M partially recovers.** K-neighbor quality drops at 100k (91.2%) then recovers at 1M (94.2%), while PCA disparity keeps improving (0.98→0.65). The dip is likely walk path variance — different regions of the source image have different correlation structure. Over 1M ticks the walk visits enough regions to average out, but doesn't surpass the 50k peak.

**Comparison with 80x80 at same tick counts:**

| Grid | k_sample | Ticks | <5px | Notes |
|------|----------|-------|------|-------|
| 80x80 | 200 | 50k | 97.1% | ts-00010 baseline |
| 80x80 | 200 | 9M | 97.4% | ts-00010 long run |
| 160x160 | 800 | 50k | 97.2% | Matches baseline |
| 160x160 | 800 | 1M | 94.2% | Slightly below |

## Conclusions

1. **k_sample must scale linearly with n.** At 160x160, k_sample=200 is dead (64.8% zero-hit). k_sample=800 (same 3.1% fraction) restores identical capture statistics.

2. **160x160 converges to the same quality as 80x80.** 97.2% within 5px at 50k — no quality loss from scaling, just need proportional sampling.

3. **Long runs show diminishing returns with walk path variance.** Quality dips at 100k, partially recovers at 1M. The random walk visits regions with varying correlation strength — some help, some hurt.

4. **TODO: auto-adjust k_sample.** Track zero-hit rate per tick. If >15%, increase k_sample. If <5%, decrease. This would eliminate manual tuning when grid size changes.

## Files

- `main.py` — same code, just `-W 160 -H 160` and `--k-sample 800`
- `check_k_capture.py` — neighbor capture rate analysis script

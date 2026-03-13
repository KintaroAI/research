# ts-00012: Scaling to 160x160

**Date:** 2026-03-12
**Status:** In Progress
**Source:** *tagged on completion as `exp/ts-00012`*

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

### Longer runs (k_sample=800)

| k_sample | Ticks | Total pairs | PCA disp | Mean dist | <3px | <5px |
|----------|-------|-------------|----------|-----------|------|------|
| 800 | 10k | 110M | 0.98 | 77.4 | 2.4% | 5.2% |
| 800 | 50k | 461M | 0.81 | 2.33 | 86.2% | **97.2%** |
| 800 | 100k | 1.09B | 0.75 | 2.91 | 76.6% | 91.2% |
| 800 | 1M | | | | | |

**50k hits 97.2% — matches 80x80 baseline.** The 160x160 grid converges to the same quality, just needs k_sample scaled to maintain the sampling fraction.

**100k degrades.** PCA disparity improves (0.81→0.75) but K-neighbor quality drops (97.2%→91.2%). Same pattern seen in ts-00010 with T=200 MSE — possible walk path variance or overtraining where later walk regions have weaker correlation structure, pushing embeddings away from early good positions.

## Files

- `main.py` — same code, just `-W 160 -H 160`

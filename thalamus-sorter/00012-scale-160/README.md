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

*(to be filled)*

## Files

- `main.py` — same code, just `-W 160 -H 160`

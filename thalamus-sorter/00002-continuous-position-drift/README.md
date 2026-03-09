# Experiment: Continuous Position Drift

**Date:** 2026-03-08
**Status:** In Progress
**Source:** *tagged on completion as `exp/ts-00002`*

## Goal

Replace the discrete grid swap mechanism with learned continuous position vectors. Each neuron gets a float coordinate that drifts freely under the same Hebbian attraction rule — no grid, no swaps, no conflict resolution.

## Hypothesis

1. Continuous positions will converge faster than discrete grid swaps because there are no conflict rejections and neurons can move fractional distances per step (no rounding to ±1).
2. The algorithm becomes trivially parallel — every neuron updates independently, no swap conflicts to resolve.
3. Extending position vectors beyond 2D will allow the system to capture affinity structure that a 2D grid cannot represent.
4. The approach is fully differentiable, opening the door to gradient-based optimization of the affinity function itself.

## Method

### Approach

Each neuron `i` has:
- A **position vector** `p[i]` — unconstrained float, initialized randomly. Starts as 2D but the dimensionality is a hyperparameter.
- A **top-K neighbor list** — same as experiment 00001, precomputed from ideal affinity.

Each tick, for every neuron:
1. Look up current positions of its K neighbors: `p[top_k[i]]`
2. Compute centroid: `c[i] = mean(p[top_k[i]])`
3. Update position: `p[i] += lr * (c[i] - p[i])`

That's it. No grid, no swaps. Pure vectorized GPU operation — a single matrix gather + mean + lerp per tick.

### Rendering

To produce an image from continuous positions, quantize to a 2D grid at display time:
- For 2D positions: round each neuron's position to nearest grid cell, resolve collisions (e.g., closest neuron wins).
- For N-D positions: project to 2D (PCA or first two dimensions), then quantize.

The continuous state is the ground truth; the grid image is just a visualization.

### Key differences from experiment 00001

| | Discrete (00001) | Continuous (00002) |
|---|---|---|
| State | Integer grid positions | Float position vectors |
| Movement | Swap with neighbor, ±1 step | Lerp toward centroid, any distance |
| Conflicts | Must resolve (priority scheme) | None — all updates independent |
| Parallelism | Partial (conflict rejection) | Full — every neuron every tick |
| Differentiable | No | Yes |
| Dimensionality | Fixed 2D grid | Arbitrary (2D, 3D, ...) |
| Rendering | Direct (grid = output) | Requires quantization step |

### Parameters to explore

- **Learning rate**: step size toward centroid. Too high = oscillation, too low = slow convergence.
- **K**: number of neighbors (same as 00001, expect similar sweet spot ~20-30).
- **Dimensionality**: 2D, 3D, higher. Does higher-D produce better separation before projection?
- **Initialization**: random uniform, random normal, grid-aligned. Does it matter?
- **Momentum/decay**: add momentum to position updates? Weight decay to prevent drift?

### Setup
- Hardware: NVIDIA GPU with CuPy
- Dependencies: numpy, scipy, cupy, opencv-python-headless
- Dataset: K_80_g.png, K_160_g.png (same as 00001)

## Log

*To be filled as experiment progresses.*

## Next Steps

- [ ] Implement ContinuousDrift solver in `solvers/continuous_drift.py`
- [ ] Add `continuous` subcommand to main.py
- [ ] Run on K_80_g.png, compare convergence speed to discrete greedy at same K
- [ ] Try 3D+ position vectors, visualize with PCA projection
- [ ] Explore learning rate schedules

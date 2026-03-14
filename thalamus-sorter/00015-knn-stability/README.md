# ts-00015: KNN Stability Over Training

**Date:** 2026-03-14
**Status:** In progress
**Source:** `exp/ts-00015`

## Goal

Measure whether K-nearest neighbor lists in embedding space stabilize as training progresses. Track KNN overlap between consecutive snapshots at regular intervals across long runs to determine:

1. Do KNN lists converge (overlap → 1.0) or plateau at an equilibrium?
2. How does convergence speed depend on grid size, signal type, and hyperparameters?
3. Can KNN stability serve as a reliable stopping criterion?

## Motivation

ts-00014 found that KNN overlap plateaus at ~0.58 with default settings (norm=100) and ~0.77 without normalization. But those measurements used coarse intervals (every 2k–5k ticks). This experiment provides a focused, systematic study with finer granularity and multiple configurations to understand the stability dynamics.

Key questions from ts-00014:
- The no-norm equilibrium at 0.77 overlap: is this a hard ceiling or does it keep climbing slowly?
- How does KNN stability correlate with the multi-metric eval quality (deriv_corr, color similarity)?
- Does RGB multi-channel sorting show different stability dynamics than grayscale?

## Approach

Use `--knn-track K --knn-report-every N` to monitor KNN overlap during training. Compare:

1. **Grayscale saccades 80x80** — known to converge well (97%+ <3px at 10k ticks)
2. **RGB saccades 80x80** — strong signal, 3 channels (19,200 neurons)
3. **RGB garden 80x80** — weak signal, slow convergence

Track with K=10, report every 100–500 ticks depending on run length. Measure overlap, spatial quality, and deriv_corr quality at each snapshot.

## Results


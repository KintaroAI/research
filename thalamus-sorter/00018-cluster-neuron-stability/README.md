# ts-00018: Cluster Neuron Stability

**Date:** 2026-03-17
**Status:** In progress
**Source:** `exp/ts-00018`

## Goal

Measure per-neuron cluster stability: how many neurons stay in the same cluster
over time, which neurons are chronic oscillators, and whether multi-cluster ring
buffers (max_k=2,3) genuinely reduce boundary churn vs single-membership.

ts-00017 showed aggregate stability (fraction of neurons unchanged between
reports). This experiment digs deeper: per-neuron lifetime, oscillation patterns,
and the spatial distribution of unstable neurons.

## Method

### Warm-start baseline

Training 50k ticks from scratch takes ~4 minutes but is wasted work when the
experiment is about cluster dynamics, not embedding convergence. Instead:

1. **Pre-train a 50k baseline model** (no clustering) and save it
2. **Warm-start from the saved model** with `--warm-start model.npy`
3. Continue training with clustering enabled — clusters fit within ~5k ticks
   on already-converged embeddings
4. Experiment on the stabilized clusters

This lets each experimental run be ~5-10k ticks instead of 50k+.

### Baseline pre-train

```
preset: gray_80x80_saccades
n=6400, dims=8, 50k ticks, no clustering
Output: ~/data/research/thalamus-sorter/exp_00018/001_pretrain_50k/
```

The saved `model.npy` becomes the warm-start checkpoint for all subsequent runs.

### Planned experiments

Once the baseline model is ready:

1. **Warm-start + max_k=1,2,3** (5-10k ticks each): compare per-neuron
   stability with ring buffer depth. Track which specific neurons oscillate
   and whether max_k=2 eliminates their oscillations or just masks them.

2. **Neuron stability histogram**: at each report interval, compute per-neuron
   "time since last cluster change". Plot distribution — are most neurons
   stable with a long tail of oscillators, or is instability widespread?

3. **Spatial map of instability**: render a heatmap where pixel brightness =
   number of cluster changes. Are unstable neurons at cluster boundaries
   (expected) or scattered (would indicate centroid drift)?

4. **Oscillation detection**: track per-neuron cluster history over a window.
   Flag neurons with A->B->A patterns. Compare oscillation rate across max_k.

## Results

### Run 001: Baseline pre-train (50k, no clustering)

```
preset: gray_80x80_saccades, n=6400, dims=8, 50k ticks, no clustering
Runtime: 86s (1.7 ms/tick)
Output: ~/data/research/thalamus-sorter/exp_00018/001_pretrain_50k/
Model: ~/data/research/thalamus-sorter/exp_00018/001_pretrain_50k/model.npy
```

**Eval:** PCA=0.5292, K10 mean=1.89, <3px=96.8%, <5px=100.0%

Embeddings converged (std=1.2089). This model is the warm-start checkpoint for
all subsequent runs in this experiment.

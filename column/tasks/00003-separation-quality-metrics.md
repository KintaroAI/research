# 00003 — Separation Quality Metrics

**Date:** 2026-03-15
**Status:** Complete

## Context

Current evaluation relies on category purity (requires ground-truth labels) and unique winner
count. These are coarse — we need richer metrics that answer: "is the cell actually learning
meaningful, stable, well-separated categories?"

## Goals

Metrics should cover three axes:
1. **Selectivity** — does each unit respond to a distinct input pattern?
2. **Consistency** — does the same input pattern reliably get the same winner?
3. **Coverage** — are all output units being used, or are some dead?

Some metrics require ground-truth labels (supervised evaluation on synthetic data), others
should work label-free (for real data later).

## Proposed Metrics

### With labels (synthetic data)

- **Purity** (already have): per winner unit, fraction of inputs from dominant cluster
- **Mutual information (NMI):** normalized mutual information between cluster labels and
  winner assignments. Unlike purity, penalizes many-to-one mappings
- **Confusion matrix:** full cluster × winner co-occurrence table. Visual diagnostic —
  ideal is a permuted identity matrix

### Without labels (unsupervised)

- **Winner entropy:** H(winners) over a window. Uniform usage → high entropy → good
  coverage. Collapse → low entropy. Compare to max entropy log2(m)
- **Winner consistency:** present the same input twice (or very similar inputs), measure
  how often the winner is the same. High consistency = stable mapping
- **Prototype spread:** mean pairwise cosine distance between prototypes. If prototypes
  cluster together → poor separation. Should be near the expected distance for random
  unit vectors as a baseline
- **Confidence gap:** mean difference between top-1 and top-2 output probabilities. High
  gap = sharp decisions. Low gap = ambiguous/overlapping categories
- **Usage Gini coefficient:** how unequal is the usage distribution? Gini=0 is perfectly
  uniform, Gini→1 is total collapse. More nuanced than just counting unique winners

### Temporal / stability metrics

- **Lock-in curve:** after N frames of stationary input, measure winner consistency over
  a held-out window. Should increase and plateau
- **Adaptation speed:** after a distribution shift, how many frames until purity recovers?
  Measures plasticity

## Implementation Plan

1. Add a `Metrics` class or `compute_metrics()` function in `dev/metrics.py`
2. Accepts: winners array, probabilities array, labels (optional), prototypes
3. Returns dict of all applicable metrics
4. `main.py` calls it at end of training and at periodic checkpoints
5. Add to results.json output

## Deliverables

- `dev/metrics.py` with all metric functions
- Integration into `main.py` training loop (periodic + final)
- Tests verifying: perfect separation → NMI=1, purity=1; random assignment → low scores;
  collapse → low entropy, high Gini
- Update experiment 00001 README with metric results from a baseline run

## Priority Order

Start with the most diagnostic ones:
1. Winner entropy + Gini (coverage)
2. Confidence gap (selectivity)
3. NMI (separation quality, replaces purity as primary metric)
4. Prototype spread (geometric health check)
5. Lock-in curve (stability, ties into temporal work)

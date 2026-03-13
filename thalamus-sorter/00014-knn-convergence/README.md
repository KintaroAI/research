# ts-00014: Online KNN Convergence Tracking

**Date:** 2026-03-13
**Status:** In progress
**Source:** `exp/ts-00014`

## Goal

Track per-neuron K-nearest neighbor lists in embedding (W-vector) space as a convergence monitor. If the embeddings have stabilized, each neuron's nearest neighbors should stop changing. The KNN overlap metric (fraction of neighbors shared between consecutive snapshots) quantifies this.

## Motivation

Current evaluation metrics (spatial <3px, <5px, PCA disparity) measure output quality but not training dynamics. We have no way to know if the embeddings have converged or are still moving. A stable KNN list means the embedding landscape has settled — useful for:
- Deciding when to stop training
- Triggering learning rate decay
- Comparing convergence speed across configurations

## Approach

1. Maintain a per-neuron KNN list (size K=10) in W-vector cosine similarity space
2. Initialize randomly; refine only with correlation-passing candidates (same candidates that feed skip-gram)
3. At report intervals, refresh all distances with current W vectors, measure overlap with previous snapshot
4. Track overlap over time as a convergence curve

### Implementation

- `--knn-track K`: enable tracking with K neighbors per neuron
- `--knn-report-every N`: snapshot and measure stability every N ticks
- KNN update happens inside `tick_correlation`, after max_hit_ratio filter, before skip-gram training
- Cosine similarity in W-vector space (not C-vectors) — W encodes "what I am"
- Saves `knn_lists.npy` and stability history to `info.json`

## Results

### Run 001: 5k ticks (sanity check)

Quick test confirming the implementation works. Overlap ramps from 0.001 to ~0.65.

### Run 002: 50k ticks

| Tick | Overlap | Neurons changed |
|------|---------|-----------------|
| 2k   | 0.001   | 6400/6400       |
| 10k  | 0.570   | 6080/6400       |
| 30k  | 0.634   | 5465/6400       |
| 50k  | 0.582   | 5867/6400       |

Spatial quality at 50k: 77.5% <3px, 92.8% <5px.

### Run 003: 200k ticks

| Window (ticks) | Avg overlap |
|----------------|-------------|
| 5k–40k         | 0.484       |
| 45k–80k        | 0.606       |
| 85k–120k       | 0.568       |
| 125k–160k      | 0.585       |
| 165k–200k      | 0.600       |

Spatial quality at 200k: 90.7% <3px, 98.7% <5px.

**Key observation**: Overlap plateaus at ~0.58 by 20k ticks and stays flat through 200k. Spatial quality continues improving (77% → 91% <3px) but the KNN lists never stabilize. ~90% of neurons have at least one neighbor change every 5000-tick window.

**Important**: The 0.58 overlap at 5000-tick intervals does NOT mean 42% of neighbors change in a single event. It's the compound effect of many small changes — each 100-tick window only changes ~10% of neighbors (see Run 004 below). The KNN lists churn continuously at a steady rate; the longer you wait between snapshots, the more cumulative drift you measure.

### Run 004: 5k ticks, 100-tick report interval (normalization probe)

Tested whether L2 normalization (every 100 ticks) causes periodic disruption to KNN lists.

| Tick range | Overlap range | Pattern |
|------------|---------------|---------|
| 100–1000   | 0.09 → 0.93   | Ramp-up from random init |
| 1000–2000  | 0.90–0.93     | Stable, small oscillation |
| 2000–2800  | 0.85–0.96     | Wider oscillation wave |
| 2800–5000  | 0.86–0.93     | Steady with occasional peaks |

**Finding: Normalization does NOT disrupt KNN.** No periodic drops aligned with `normalize_every=100`. The oscillation waves (dips to 0.85, peaks to 0.96) don't correlate with normalization timing — likely driven by signal buffer content or training dynamics. Cosine similarity is scale-invariant, so normalization only affects KNN indirectly through gradient dynamics on subsequent ticks.

## Analysis

The embeddings jitter around their optimal positions indefinitely with constant lr=0.001, even when spatial structure is already excellent. The churn is continuous and uniform — not caused by normalization or any periodic event.

**How the churn works**: Each 100-tick window changes ~10% of neighbors (overlap ~0.90). Over 5000 ticks (50 windows), these small changes compound to 42% total churn (overlap ~0.58). This is accumulated drift from constant-rate learning, not a sudden disruption.

Contributing factors:
- **Constant learning rate**: lr=0.001 never decays, so every tick pushes vectors by the same amount regardless of how well-sorted the map is. This is the primary cause.
- **Stale KNN distances**: Between refresh intervals, stored cosine similarities become stale as embeddings move. This may cause some artificial churn when distances are refreshed and rankings shift.
- **Sparse candidate coverage**: Each neuron is an anchor ~4% of ticks (batch=256, n=6400). Over 5000 ticks, a neuron sees ~6000 random candidates — enough to find good neighbors, but not exhaustive.

**What normalization is NOT doing**: L2 normalization every 100 ticks does not visibly disrupt the KNN landscape. Since cosine similarity is scale-invariant, resetting magnitudes to 1.0 doesn't change neighbor rankings. The gradient dynamics do change (all dot products become cosine similarities right after normalization), but this doesn't produce measurable KNN instability.

## Next Steps

- **Learning rate decay**: If overlap triggers lr reduction (e.g., halve lr when overlap > 0.7 for N consecutive reports), embeddings should converge toward full stability. The KNN overlap metric could serve as the decay trigger itself.
- **Early stopping**: Stop training when overlap exceeds a threshold (e.g., 0.95) for several consecutive snapshots — the embeddings have converged and further training adds only jitter.
- **Sweep KNN K**: Test K=20, K=50 to see if larger neighborhoods are more/less stable.
- **Compare with lr decay**: Run identical configuration but with lr decay schedule, measure how overlap responds.

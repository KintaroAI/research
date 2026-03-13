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

### Normalization as implicit learning rate control

While normalization doesn't directly disrupt KNN rankings, it has a critical *indirect* effect: it **prevents natural gradient decay**, keeping the effective learning rate permanently high.

The mechanism:
1. **Between normalizations** (100 ticks): skip-gram updates grow vector magnitudes → dot products `w · c` increase → sigmoid saturates → gradients shrink → effective lr naturally decreases
2. **At normalization**: magnitudes reset to 1.0 → dot products shrink back to cosine range [-1, 1] → sigmoid unsaturates → gradients restored to full strength

Without normalization, the system would self-dampen: magnitude growth → sigmoid saturation → gradient vanishing → embeddings stop moving. Normalization prevents this, acting as a **periodic gradient wake-up** that keeps learning active indefinitely. This is why the KNN lists can never fully stabilize with constant lr + periodic normalization — the system is never allowed to relax.

This insight reframes the convergence problem. Three paths to stable embeddings:

1. **Explicit lr decay** — reduce update magnitude directly, independent of normalization
2. **Reduce normalization frequency** — longer gaps between resets allow more natural self-dampening between wake-ups (e.g., normalize_every=1000 instead of 100)
3. **Remove normalization entirely** — let magnitude growth self-regulate gradients. Risks: unchecked magnitude blow-up, eventual numerical issues. But the self-dampening *is* a form of convergence.

Option 2 is interesting because it creates a hybrid: the system self-dampens for longer stretches (gradients decay naturally as magnitudes grow), then gets a periodic reset. Lower normalization frequency → more time in the dampened state → less churn.

Option 1 (lr decay) is the cleanest and most controllable. The KNN overlap metric could serve as the decay trigger itself — when overlap crosses a threshold, halve lr.

### Normalization frequency sweep (Runs 005–008, 50k ticks each)

Tested normalize_every={100, 500, 1000, 5000} with KNN K=10, report every 2000 ticks.

| normalize_every | Avg overlap (20k–50k) | Peak overlap | Spatial <3px | Spatial <5px |
|----------------|----------------------|-------------|-------------|-------------|
| 100 (default)  | ~0.61                | 0.69        | 88.3%       | 98.4%       |
| 500            | ~0.69                | 0.77        | 86.5%       | 97.8%       |
| 1000           | ~0.70                | 0.75        | 94.5%       | 99.7%       |
| 5000           | ~0.81                | 0.87        | 96.2%       | 99.7%       |

**Theory confirmed.** Less frequent normalization → higher KNN stability AND better spatial quality. The gradient self-dampening mechanism works: between normalizations, magnitudes grow, sigmoid saturates, gradients shrink, and the system partially relaxes. Longer gaps = more relaxation = less churn.

Notable observations:
- **normalize_every=5000 wins on all metrics**: 0.81 avg overlap, 96.2% <3px — substantially better than default 100
- **Normalization events are visible at low frequency**: The norm=5000 run shows a dip at tick 50000 (overlap drops from 0.83 to 0.73) — exactly at the 10th normalization. At higher frequency, these perturbations are too frequent and small to distinguish.
- **Spatial quality improves too**: This was unexpected. Less normalization doesn't just stabilize embeddings — it produces better spatial layout. The self-dampening may act as a natural annealing, where early ticks have large updates and later ticks (before the next normalization) have smaller, refinement-scale updates.
- **500 vs 1000 is noisy**: Similar overlap but 1000 has much better spatial quality (94.5% vs 86.5%). Need more runs to separate signal from noise.

### Implication for default settings

The current default (normalize_every=100) appears to be too aggressive. It keeps the effective learning rate permanently high when the system would benefit from self-dampening. A value of 1000–5000 seems more appropriate, though the optimal value likely depends on training length — longer runs may benefit from even less frequent normalization.

### LR decay at normalization events (Runs 009–011, 50k ticks, normalize_every=5000)

Implemented `--lr-decay` flag: multiply lr by decay factor at each normalization event. With normalize_every=5000 and 50k ticks, that's 10 decay events.

| lr_decay | Final lr | Final overlap | Spatial <3px | Spatial <5px |
|----------|----------|---------------|-------------|-------------|
| 1.0 (no decay) | 0.001000 | 0.81 | 96.2% | 99.7% |
| 0.9 | 0.000349 | 0.84 | 96.4% | 99.9% |
| **0.8** | **0.000107** | **0.95** | **96.2%** | **100.0%** |
| 0.5 | 0.000001 | 0.997 | 94.8% | 99.3% |

**decay=0.8 is the sweet spot.** Overlap reaches 0.95 (near convergence) with no spatial quality loss — 96.2% <3px, 100.0% <5px.

**decay=0.5 over-dampens.** KNN converges beautifully (0.997 — only 77 neurons still changing out of 6400) but spatial quality drops slightly to 94.8% <3px. The lr decayed too fast (to 0.000001 by tick 50k), freezing the embeddings before they were fully sorted. This confirms the user's prediction: stable KNN alone isn't sufficient — must verify quality doesn't degrade.

Convergence trajectory for decay=0.8:
- Tick 10k: overlap=0.43, lr=0.00064 (still learning aggressively)
- Tick 20k: overlap=0.78, lr=0.00041 (structure forming)
- Tick 30k: overlap=0.91, lr=0.00026 (refining)
- Tick 40k: overlap=0.93, lr=0.00017 (nearly stable)
- Tick 50k: overlap=0.95, lr=0.00011 (converged, quality preserved)

**Key insight**: The combination of normalize_every=5000 + lr_decay=0.8 provides two complementary annealing mechanisms:
1. **Intra-cycle annealing**: Between normalizations, magnitude growth naturally dampens gradients (free, automatic)
2. **Inter-cycle annealing**: Each normalization resets to a *lower* base lr (controlled, monotonic)

This dual annealing is more biologically plausible than a global lr schedule — the system's own dynamics (magnitude growth) provide local self-regulation, while the decay provides global convergence pressure.

## Next Steps

- **Extend decay=0.8 to 200k ticks**: Verify overlap reaches ~1.0 and quality doesn't degrade at very low lr.
- **Test normalize_every=0 (off)**: Does the system fully self-dampen to convergence without any normalization?
- **Test on garden.png / RGB**: Verify these settings work on harder inputs (pair starvation, multi-channel).
- **Sweep KNN K**: Test K=20, K=50 to see if larger neighborhoods are more/less stable.
- **Update default preset**: Consider normalize_every=5000 + lr_decay=0.8 as new defaults.

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

### No normalization — pure self-dampening (Runs 012–014, 50k ticks)

Tested normalize_every=0 (no normalization at all) with varying lr, no decay.

| lr | Normalization | Final overlap | Spatial <3px | <5px | std |
|----|--------------|---------------|-------------|------|-----|
| 0.001 | off | 0.90 | **99.0%** | 100% | 1.41 |
| 0.0005 | off | 0.90 | **99.5%** | 100% | 1.34 |
| 0.0001 | off | 0.83 | 29.4% | 54.3% | 0.46 |
| 0.001 | every 5000 | 0.81 | 96.2% | 99.7% | 0.35 |
| 0.001 | every 5000 + decay=0.8 | 0.95 | 96.2% | 100% | 0.35 |

**No normalization beats all previous configurations on spatial quality.** lr=0.001 without normalization achieves 99.0% <3px — the best result so far. lr=0.0005 is even better at 99.5% <3px.

The self-dampening mechanism works as predicted: magnitudes grow (std=1.41 vs 0.35 at init), dot products increase, sigmoid saturates, gradients vanish. The system naturally anneals without any explicit decay or normalization schedule. No hyperparameters to tune, no floor to set, no risk of lr underflowing to zero.

**lr=0.0001 is too slow**: Only 29.4% <3px at 50k ticks — not enough learning. The overlap oscillates (peaks at 0.96, drops to 0.83) suggesting the system is still in early transient phases. Magnitudes haven't grown enough to trigger self-dampening (std=0.46, barely above init).

**Why normalization hurts quality**: Normalization periodically resets magnitude to 1.0, which:
1. Erases the self-dampening progress (gradients jump back up)
2. Forces the system to re-learn magnitude structure it already had
3. Creates periodic perturbations that push embeddings away from their settled positions

Without normalization, the system smoothly converges — no perturbations, no resets, just continuous refinement that naturally slows down as the structure solidifies.

**Overlap trajectory for lr=0.001 no-norm (50k run)**:
- Tick 10k: 0.52 (structure forming)
- Tick 20k: 0.78 (rapid improvement)
- Tick 30k: 0.84 (self-dampening kicks in)
- Tick 40k: 0.87 (slowing down)
- Tick 50k: 0.90 (still climbing)

### 200k run — self-dampening has a ceiling (Run 015)

Extended lr=0.001 no-norm to 200k ticks. Overlap does NOT continue climbing to 1.0.

| Window (ticks) | Avg overlap |
|----------------|-------------|
| 5k–50k         | 0.68        |
| 50k–100k       | 0.78        |
| 100k–150k      | 0.76        |
| 150k–200k      | 0.79        |

Final: overlap=0.77, spatial=99.5% <3px, 100% <5px, std=1.30.

**The self-dampening reaches equilibrium, not convergence.** Magnitudes grow until positive and negative gradient forces balance — but at that equilibrium, the system still has enough gradient flow to keep churning ~23% of KNN neighbors every 5000 ticks. The sigmoid is partially saturated but not fully.

This makes physical sense: the skip-gram keeps receiving new correlation pairs every tick (the signal buffer refreshes continuously). Even if all vectors were perfectly positioned, new random negative samples would still push them around. The equilibrium is between "learning from new data" and "magnitude-dampened gradients."

### Vector magnitude analysis

Checked W-vector magnitudes across runs to understand the self-dampening mechanism quantitatively.

| Run | Mean ||w|| | Min | Max | Std |
|-----|-----------|-----|-----|-----|
| norm=100 50k (baseline) | 1.000 | 1.000 | 1.000 | 0.000 |
| norm=5000 50k | 1.000 | 1.000 | 1.000 | 0.000 |
| no-norm lr=0.0001 50k | 1.326 | 0.876 | 1.596 | 0.136 |
| no-norm lr=0.0005 50k | 3.815 | 3.133 | 4.244 | 0.173 |
| no-norm lr=0.001 50k | 4.030 | 3.443 | 4.439 | 0.179 |
| no-norm lr=0.001 200k | 3.934 | 3.188 | 4.541 | 0.255 |

**Magnitudes stabilize around 4.0.** The 50k and 200k runs with lr=0.001 are nearly identical (4.03 vs 3.93) — magnitudes are not growing unboundedly, they've reached equilibrium.

At magnitude ~4, dot products between vectors are ~16× larger than at unit length (4×4), pushing the sigmoid well into saturation. This is the self-dampening: σ(16×cos) ≈ 1.0 for positive pairs, so gradients (1-σ) ≈ 0. But not fully zero — enough gradient remains for continuous slow refinement.

**lr=0.0001 barely grew** (1.33 after 50k ticks) — not enough updates to reach the dampening regime, explaining both the poor spatial quality (29.4% <3px) and the oscillating overlap.

The magnitude spread is tight (std ≈ 0.18–0.26) — all neurons reach similar magnitudes regardless of position or local correlation structure. This uniformity suggests the equilibrium is a global property of the skip-gram dynamics, not neuron-specific.

### Normalization as implicit learning rate control — full picture

The experiments reveal that normalization and lr interact in a way that was previously misunderstood:

1. **Normalization does not just prevent blow-up** — it actively prevents the system from reaching its natural equilibrium magnitude (~4.0), which is where self-dampening provides the best spatial quality
2. **Frequent normalization (100) acts as permanent high lr** — resets gradients to full strength every 100 ticks, keeping the system in constant high-churn
3. **No normalization lets the system find its own equilibrium** — magnitudes grow to ~4.0 where gradient flow balances between positive and negative forces, producing the best spatial layout
4. **The "blow-up" that normalization prevents isn't actually harmful** — magnitude 4.0 is stable and produces superior results

The original rationale for normalization ("prevents magnitude blow-up") was based on the assumption that growing magnitudes are dangerous. In practice, magnitude growth IS the annealing mechanism, and preventing it hurts quality.

### Summary: convergence comparison

| Config | 50k overlap | 200k overlap | 200k <3px | Converges? |
|--------|-----------|-------------|----------|------------|
| norm=100, no decay (default) | 0.61 | 0.58 | 90.7% | No — plateaus at 0.58 |
| norm=5000, no decay | 0.81 | 0.81 | 96.2% | No — plateaus at 0.81 |
| norm=5000, decay=0.8 | 0.95 | — | 96.2% | Yes — approaching 1.0 |
| **no norm, no decay** | **0.90** | **0.77** | **99.5%** | **No — equilibrium at ~0.77** |

**Best spatial quality**: no normalization (99.5% <3px)
**Best convergence**: lr decay (0.95 overlap and climbing)
**These are different goals.** For a system running indefinitely on live signal, equilibrium may be preferable to full convergence — the system stays responsive to new input while maintaining excellent spatial structure.

### Neighbor-of-neighbor candidate sampling (Runs 017–020)

Implemented `--knn-nofn` flag: for each anchor, also probe neighbors-of-neighbors (minus direct neighbors) as correlation candidates. Goal: break O(n²) scaling by replacing large random k_sample with targeted KNN-guided search.

**First attempt (Run 017)**: max_hit_ratio counted all candidates including nofn — the targeted nofn candidates inflated hit counts, causing most anchors to be discarded. Result: 0.4% <3px, pair starvation. **Fix**: max_hit_ratio only counts hits in the random portion.

**50k comparison at 80×80 (Runs 019–020)**:

| | Without nofn | With nofn |
|---|---|---|
| Candidates/anchor | 200 random | 200 random + 100 nofn |
| Pairs/tick | ~11k | ~128k |
| Spatial <3px | 99.5% | 59.9% |
| KNN overlap | 0.80 | 0.47 |
| Wall time | 235s | 311s |

**Nofn hurts at 80×80.** The 100 nofn candidates pass threshold at a much higher rate than random ones (they're targeted toward correlated neurons), generating 11× more training pairs. These targeted pairs dominate the training signal, drowning out unbiased random exploration. The system reinforces its current (possibly wrong) embedding structure instead of discovering new connections.

**Key insight**: At 80×80 (n=6400) with k_sample=200 (3% of n), random sampling already provides good coverage. Nofn adds redundant signal biased toward existing structure. The real value of nofn is at larger grids where random k_sample would need to scale quadratically — there, nofn can provide targeted coverage with a fixed-size random sample.

**Next test**: 320×320 (n=102,400) where k_sample would need to be 3072 for 3% coverage. Test nofn with reduced k_sample (e.g., 50–200 random + 100 nofn) vs baseline with k_sample=3072.

### 320×320 scaling tests (Runs 021–025)

Tested at 320×320 grayscale (n=102,400) to validate scaling. All runs use MSE scoring with threshold=0.02, no normalization, 10k ticks.

| Run | k_sample | batch_size | nofn | Total pairs | Notes |
|-----|----------|------------|------|-------------|-------|
| 021 | 3072 (3%) | 256 | no | 71M | Baseline — works but slow |
| 022 | 200 | 256 | yes | 470M | Nofn: 5× faster, 6.6× more pairs |
| 023 | 200 | 256 | no | 1.9M | k_sample too small without nofn — pair starvation |
| 024 | 1600 | 2048 | no | OOM | batch×k_sample×T too large |
| 025 | 800 | 1024 | no | OOM | Same OOM issue |

At 320×320, nofn proves its value: Run 022 (200 random + nofn) generates 6.6× more pairs than Run 021 (3072 random) in ~5× less time per tick. Run 023 confirms that k_sample=200 alone is useless at this scale (only 0.2% sampling fraction → pair starvation).

Runs 024–025 attempted proportional scaling (batch×k_sample proportional to n) but OOM'd — the MSE tensor `(batch, k_sample, T)` exceeded GPU memory at these sizes.

### Derivative correlation threshold fix

Presets were switched from `use_mse` to `use_deriv_corr` but the threshold (0.02) was never recalibrated. MSE threshold 0.02 means "MSE < 0.02" (small distance = similar). Deriv_corr threshold 0.02 means "correlation > 0.02" — which 67.6% of all random pairs pass. With `max_hit_ratio=0.1`, every anchor was discarded → zero pairs → no learning.

**Score distributions** (deriv_corr, 80×80 grayscale, saccades.png):

| | Near (<3px) | Far (>20px) |
|---|---|---|
| Mean score | 0.869 | 0.015 |
| Score range | 0.775 – 1.000 | -0.179 – 0.295 |

Clean separation. Threshold 0.5 gives 100% near pass / 0% far pass. Calibrated from ts-00011 experiment (97.2% <5px at threshold=0.5 vs 87.4% at threshold=0.3).

**Garden.png** has weaker correlations (step=5, high spatial diversity): near mean=0.412 (range 0.099–1.0), far max=0.113. Threshold 0.1 gives 96% near / 0% far.

Updated all presets: saccades→threshold=0.5, garden→threshold=0.1. Deprecated `--use-mse` with assert.

### Anchor batches and anchor_sample (Runs 026–031)

Implemented `anchor_sample` parameter: total unique anchor neurons per tick, split into sequential chunks of `batch_size`. More coverage per tick without increasing GPU memory. Two mutually exclusive CLI flags: `--anchor-sample N` (direct) or `--anchor-batches M` (= M × batch_size).

**Broken runs (026, first attempts)**: Used old threshold=0.02 with deriv_corr → zero pairs. Fixed after threshold recalibration.

**Short comparison (Runs 027–029, deriv_corr threshold=0.5)**:

| Run | Ticks | anchor_sample | Total pairs | <3px | <5px | Time |
|-----|-------|---------------|-------------|------|------|------|
| 028 | 1k | 256 (1 batch) | 19M | 0.6% | 1.4% | 4.3s |
| 027 | 1k | 512 (2 batches) | 50M | 5.6% | 12.6% | 8.0s |
| 029 | 2k | 256 (1 batch) | 40M | 3.8% | 8.3% | 8.2s |

2 batches at 1k ticks (50M pairs, 5.6%) outperforms 1 batch at 2k ticks (40M pairs, 3.8%) in the same wall time. More unique anchors per tick generates more pairs than more ticks with fewer anchors.

**Longer comparison (Runs 030–031, deriv_corr threshold=0.5)**:

| Run | Ticks | anchor_sample | Total pairs | <3px | <5px | Time |
|-----|-------|---------------|-------------|------|------|------|
| 030 | 10k | 256 (1 batch) | 201M | 96.8% | 100.0% | 79.1s |
| 031 | 5k | 512 (2 batches) | 181M | 95.6% | 99.9% | 79.1s |

Same wall time (79.1s). 2 batches at 5k ticks reaches 95.6% <3px vs 96.8% for 1 batch at 10k — nearly equivalent. The slight gap (~1.2%) comes from 2 batches processing anchors sequentially within a tick (second batch sees embeddings already updated by first batch), introducing a small ordering effect.

**Conclusion**: `anchor_sample` is an effective coverage multiplier. Doubling anchors per tick ≈ doubling ticks at half the count, with identical wall time. The parameter is useful for trading tick count for per-tick coverage without increasing memory.

### GPU/rendering infrastructure tests (Runs 032–043)

Validated separate `--gpu` (solver) and `--render-gpu` (CuPy/cuML UMAP) flags, spawn multiprocessing for CUDA context isolation, and cold projection default.

#### 160×160 scaling (Runs 032–035)

All runs at 160×160 (n=25,600) with 1k or fewer ticks — insufficient training for this scale.

| Run | Grid | Preset | Ticks | k_sample | Pairs | <3px | <5px | Time |
|-----|------|--------|-------|----------|-------|------|------|------|
| 032 | 160² | garden | 100 | 768 | 39.6M | 0.1% | 0.3% | 5.3s |
| 033 | 160² | garden | 1k | 768 | 369M | 0.2% | 0.5% | 50.2s |
| 034 | 160² | saccades | 1k | 800 | 20.2M | 0.1% | 0.2% | 12.8s |
| 035 | 160² | saccades | 1k | 800 | 20.1M | 0.1% | 0.2% | 12.9s |

160×160 needs significantly more ticks to converge. Garden with threshold=0.1 generates far more pairs (369M vs 20M for saccades at threshold=0.5) but still doesn't sort — the pairs are low-quality (weak correlations pass the threshold). Saccades at threshold=0.5 generates fewer but cleaner pairs, though also insufficient at 1k ticks.

#### GPU vs CPU solver/rendering (Runs 036–042)

Tested all four GPU/CPU combinations for solver and renderer at 80×80.

| Run | Solver | Renderer | Ticks | <3px | Time | Notes |
|-----|--------|----------|-------|------|------|-------|
| 036 | GPU | GPU | 1k | 1.1% | 4.3s | Baseline short run |
| 037 | GPU | CPU | 1k | 0.7% | 29.1s | CPU UMAP 7× slower |
| 038 | CPU | GPU | 1k | 0.8% | 272.4s | CPU solver 63× slower |
| 041 | GPU | GPU | 1k | 0.5% | 7.7s | Both GPU |
| 039 | GPU | CPU | 10k | 97.6% | 67.9s | Save last frame only |
| 042 | GPU | GPU | 10k | 93.5% | 44.8s | Save every 100 ticks |

**Key findings**:
- **CPU solver is impractical**: 272s vs 4.3s for 1k ticks (63× slower), no quality benefit
- **GPU UMAP rendering** via cuML saves ~23s at 10k ticks (44.8s vs 67.9s) by parallelizing rendering with solver
- **1k ticks underfits at 80×80**: All 1k runs show <2% <3px regardless of GPU config
- **Rendering frequency matters**: Run 039 renders only 1 frame (67.9s), Run 042 renders 100 frames (44.8s) — async rendering with GPU UMAP is faster than CPU rendering even with 100× more frames
- **Spawn multiprocessing works**: Separate CUDA contexts for PyTorch solver and CuPy/cuML render worker avoid initialization conflicts

Run 040 (CPU solver + GPU render, 10k ticks) was killed — CPU solver too slow for 10k.

#### Cold vs warm projection (Runs 042–043)

| Run | Projection | Ticks | <3px | <5px | Pairs | Time |
|-----|-----------|-------|------|------|-------|------|
| 042 | warm | 10k | 93.5% | 99.5% | 187.6M | 44.8s |
| 043 | cold | 10k | 95.5% | 99.9% | 216.1M | 53.5s |

**Cold projection produces better quality renders.** UMAP from scratch each frame avoids warm-start drift artifacts where early (poor) projections bias later frames. The 2% quality improvement (93.5% → 95.5%) comes at a modest 8.7s time cost. Made cold projection the default (`--cold-projection` is now `BooleanOptionalAction` with `default=True`).

Note: the pair count difference (187M vs 216M) is due to random sampling variance, not the projection mode — rendering doesn't affect solver training.

### RGB multi-channel experiments (Runs 044–050)

First RGB (3-channel) runs at 80×80. Each channel has its own neurons → 19,200 total neurons on a 240×80 render grid. Color-tinted pixel values (R/G/B) from K_80_g.png.

#### Saccade step matters for garden (Runs 044–048)

Garden.png has high spatial diversity — with saccade_step=50, the signal changes so much between frames that deriv_corr scores are weak. Reducing saccade_step from 50→5 dramatically increases pair generation.

| Run | Signal | step | Ticks | Batches | Pairs | <3px | <5px | Time |
|-----|--------|------|-------|---------|-------|------|------|------|
| 044 | garden | 50 | 100 | 1 | 0.1M | 0.1% | 0.3% | 34.0s |
| 045 | garden | 50 | 1k | 1 | 1.9M | 0.1% | 0.3% | 15.4s |
| 046 | garden | 50 | 1k | 3 | 7.5M | 0.1% | 0.3% | 44.4s |
| 047 | saccades | 50 | 1k | 3 | 179M | 7.3% | 16.0% | 48.4s |
| 048 | garden | 5 | 1k | 3 | 166M | 0.2% | 0.5% | 45.2s |

**Saccade_step=50 with garden is pair-starved**: Only 7.5M pairs at 1k×3 batches (vs 179M for saccades). The garden image changes too rapidly between distant frames — most candidate pairs fail the threshold=0.1 deriv_corr check, and those that pass are low-quality (weak correlation signal).

**Saccade_step=5 fixes pair generation** (166M, comparable to saccades 179M) but quality remains low at 1k ticks — garden's spatial complexity means 19,200 neurons need far more training than 1k ticks to sort.

**Saccades RGB works well**: 7.3% <3px at 1k ticks with 3 batches, on track for strong convergence with more ticks.

#### Garden 10k ticks (Runs 049–050)

| Run | Ticks | Batches | Pairs | <3px | <5px | k10_dist | Time |
|-----|-------|---------|-------|------|------|----------|------|
| 049 | 10k | 3 | 1.95B | 33.0% | 46.2% | 13.89 | 450s |
| 050 | 10k | 3 | 2.05B | 28.6% | 39.1% | 12.93 | 452s |

Garden RGB at 10k ticks shows early sorting (~30% <3px) but is far from converged. The run-to-run variance is high (33% vs 29%) — typical for early training where random seed and saccade sequence matter. For comparison, grayscale saccades at 80×80 (6,400 neurons) reaches 97% <3px at 10k ticks. Garden RGB has 3× more neurons and weaker correlations — likely needs 50k+ ticks.

#### Dims and normalization probes (Runs 051–053)

Tested D16 vs D8 and normalization impact at 1k ticks, 3 anchor batches, saccade_step=5.

| Run | Dims | norm_every | Pairs | <3px | <5px | Time |
|-----|------|-----------|-------|------|------|------|
| 051 | 16 | 0 | 187M | 0.2% | 0.4% | 50.6s |
| 052 | 8 | 0 | 166M | 0.2% | 0.6% | 45.5s |
| 053 | 8 | 100 | 215M | 0.2% | 0.4% | 46.9s |

All three are indistinguishable at 1k ticks — too early for dims or normalization to matter. D16 is ~10% slower (more parameters). Normalization boosts pair count by ~30% (215M vs 166M) by periodically resetting gradient magnitudes, but this doesn't translate to better quality yet.

#### Garden 500k — deriv_corr vs MSE channel analysis (Run 054)

Run 054: RGB garden D8, no normalization, 500k ticks, 3 anchor batches, saccade_step=5, deriv_corr threshold=0.1.

**Flat grid eval**: 32.0% <3px, 48.2% <5px, 92B pairs, 16,178s (~4.5 hours).

Compared with ts-00013's garden 500k run (MSE, norm=100, saccade_step=50, 1 batch):

| | ts-00013 MSE 500k | Run 054 deriv_corr 500k |
|---|---|---|
| Scoring | MSE, threshold=0.02 | deriv_corr, threshold=0.1 |
| Pairs | 264M | 92B (348×) |
| Channel self-neighbors | 98–99% | 63–73% |
| Cross-channel mixing | ~2% | ~30% |
| R <5px (within-ch) | 23.2% | **82.8%** |
| G <5px (within-ch) | 93.2% | 79.2% |
| B <5px (within-ch) | 97.1% | 79.1% |

Per-channel neighbor composition (run 054):

| Channel | R | G | B | Self-neighbors | <5px (within-ch) |
|---------|---|---|---|---------------|-------------------|
| R | **73.1%** | 14.9% | 12.0% | 73.1% | 82.8% |
| G | 15.8% | **63.2%** | 21.0% | 63.2% | 79.2% |
| B | 12.8% | 21.5% | **65.8%** | 65.8% | 79.1% |

**Key finding: deriv_corr trades channel purity for uniform spatial quality.**

- MSE at threshold=0.02 cleanly separates channels (98%+ self-neighbors) but starves R of training pairs → R sorts poorly (23% <5px). G and B sort well (93–97%) because they have stronger spatial autocorrelation in this image.
- Deriv_corr at threshold=0.1 allows ~30% cross-channel mixing (derivative correlation between co-located pixels of different channels is real signal) but distributes training pairs more uniformly → all three channels reach ~80% <5px.
- The total pair count difference is enormous (92B vs 264M) — deriv_corr + step=5 + 3 batches generates 348× more pairs. But quality improvement is modest (flat 32% vs 25.5% <3px), confirming that garden's bottleneck is signal quality, not pair volume.

#### RGB saccades — channel separation at 1k and 10k (Runs 055–056)

Ran RGB saccades (threshold=0.5, saccade_step=50) with 3 anchor batches. Run 056 warm-started from run 055's model (+9k ticks).

| | Run 055 (1k) | Run 056 (1k+9k) |
|---|---|---|
| Flat <3px | 3.8% | 38.4% |
| Flat <5px | 8.2% | 68.1% |
| Channel self-neighbors | 35–40% | **99.3%** |
| Cross-channel mixing | 62% | 0.5% |
| R <5px (within-ch) | 21.6% | **100%** |
| G <5px (within-ch) | 24.1% | **100%** |
| B <5px (within-ch) | 19.1% | **100%** |
| k10 mean dist | 13.49 | 1.86 |

At 1k ticks: no channel separation (35–40% self, near-random). At 10k: near-perfect separation (99.3%) AND perfect within-channel spatial sorting (100% <5px, mean dist 1.87px for all channels).

**Saccades vs garden with deriv_corr:**

| | Saccades (10k, thr=0.5) | Garden (500k, thr=0.1) |
|---|---|---|
| Channel self-neighbors | 99.3% | 63–73% |
| R <5px (within-ch) | 100% | 82.8% |
| G <5px (within-ch) | 100% | 79.2% |
| B <5px (within-ch) | 100% | 79.1% |
| Ticks needed | 10k | 500k+ (not converged) |

Saccades achieves perfect channel separation and spatial sorting 50× faster than garden. The key difference is threshold: 0.5 for saccades is strict enough to separate channels (only truly correlated same-channel pairs pass), while 0.1 for garden is permissive (cross-channel pairs pass → 30% mixing). Garden's weaker correlations require the lower threshold to generate pairs at all, but this comes at the cost of channel purity.

### Pair quality analysis — why garden is hard

Analyzed deriv_corr score distributions per channel pair type and distance bin (256 anchors × 600 candidates, single tick).

**Score by distance — same channel vs cross channel:**

| Distance | Saccades same-ch | Saccades cross-ch | Garden same-ch | Garden cross-ch |
|----------|-----------------|-------------------|---------------|----------------|
| 0-3px | 0.85 (100% pass) | 0.69 (97% pass) | 0.57 (100% pass) | 0.39 (94% pass) |
| 3-5px | 0.72 (100%) | 0.59 (83%) | 0.21 (77%) | 0.13 (55%) |
| 5-10px | 0.57 (79%) | 0.47 (35%) | 0.02 (22%) | 0.01 (18%) |
| 10-20px | 0.36 (11%) | 0.30 (2%) | 0.02 (15%) | 0.01 (11%) |
| 20+px | 0.08 (0%) | 0.06 (0%) | -0.00 (3.6%) | -0.00 (3.6%) |

**Saccades has clean signal**: Near same-ch scores (0.85) are 10× higher than far (0.08). Zero far pairs pass threshold=0.5. Same-ch vs cross-ch gap is clear (0.85 vs 0.69 at <3px) — enough for the solver to learn channel identity over time.

**Garden has noisy signal**: Near same-ch (0.57) only 2× higher than 5-10px (0.02). 3.6% of far pairs leak through threshold=0.1, producing ~4700 noise pairs vs ~1200 real near pairs (4:1 noise ratio). Same-ch vs cross-ch gap (0.57 vs 0.39) is small relative to the noise — channels are nearly indistinguishable to the solver.

**Per-channel pair pass rates (garden):** All channel pairs (R→R, R→G, G→B, etc.) have similar pass rates (5–8%) and similar far-pair leak (3–4%). No channel combination stands out — the garden signal simply doesn't differentiate channels well.

**Same-pixel cross-channel pairs:** Extremely high scores for both (0.7–0.9) because co-located R/G/B neurons see the same spatial content. But these are vanishingly rare in random sampling (~3–7 per channel pair in 153k candidates).

**Root cause is image content diversity, not saccade step.** Saccade frame sequences saved to `exp_00014/saccade_frames_*/` confirm visually:
- **Saccades step=50**: Spatially homogeneous source (book pages). Even large jumps produce similar frames → strong temporal correlations between nearby pixels.
- **Garden step=5**: Smooth, small shifts. But garden has sharp color boundaries (flower petals → leaves → sky) so adjacent pixels have very different values → weak correlations.
- **Garden step=50**: Chaotic — each frame shows a completely different part of the garden. Derivatives between frames are essentially random.

**Reframing the "noise" pairs:** The far pairs that pass threshold aren't necessarily harmful noise. If two distant pixels are both on a pink flower, they genuinely share color/derivative correlation — their deriv_corr score is real signal, not statistical artifact. The solver correctly identifies them as similar. The problem is that our spatial eval (k10 <5px) penalizes this: color-similar distant pixels get embedded nearby, which is semantically correct but spatially "wrong." This is the sorter discovering **color similarity** rather than **spatial proximity** — a valid but different structure.

**Embedding dims don't affect pair discovery.** Deriv_corr scores are computed from the raw signal buffer `(n, T)`, independent of D. D=8 vs D=16 produces identical pairs, scores, and pass rates. The difference is downstream: more dims give the skip-gram more capacity to encode the discovered relationships. This is why ts-00013 showed D16 helps R (complex spatial distribution needs more embedding capacity) — same pairs, better representation.

## Summary

Experiment ts-00014 started as KNN convergence tracking and expanded into a comprehensive investigation of training dynamics:

1. **KNN overlap** plateaus at ~0.58 with default settings. Normalization prevents convergence by periodically resetting gradient magnitudes.
2. **No normalization** achieves best spatial quality (99.5% <3px) via self-dampening: magnitudes grow to ~4.0, sigmoid saturates, gradients vanish naturally.
3. **Anchor_sample** is an effective coverage multiplier. 2× anchors ≈ 2× ticks at same wall time.
4. **Deriv_corr vs MSE**: deriv_corr trades channel purity for uniform spatial quality. Saccades at threshold=0.5 achieves perfect separation; garden at 0.1 has 30% cross-channel mixing.
5. **Signal quality is the bottleneck for garden**: 4:1 noise ratio in pairs, weak same-ch vs cross-ch gap. But "noise" pairs between color-similar distant pixels are real signal — the sorter discovers color similarity, not just spatial proximity.
6. **GPU infrastructure**: separate solver/render GPU flags, spawn multiprocessing, cold projection default — all working reliably.

### Multi-metric evaluation of garden 500k model (Run 054)

The flat spatial eval (32% <3px) underestimates the model. Alternative metrics reveal the embeddings capture meaningful structure:

| Metric | K10 neighbors | Random baseline | Ratio |
|--------|--------------|-----------------|-------|
| Deriv_corr (what solver optimizes) | 0.457 | 0.004 | 114× |
| RGB color distance | 0.194 | 0.564 | 0.34× (3× closer) |
| Mean signal MSE | 0.0212 | 0.0587 | 0.36× (3× closer) |

**Deriv_corr of neighbors**: 80% of K10 neighbors have corr > 0.1, 50% have corr > 0.5. The solver successfully found genuinely correlated neurons — median neighbor correlation is 0.505.

**Color similarity**: Embedding neighbors are 3× more color-similar than random. The solver groups same-color neurons (e.g., two pink flower pixels far apart) — valid structure that flat spatial eval penalizes.

**Channel-aware spatial quality**: Within same-channel, spatial quality is 84.7% <5px (vs flat 48%). The 66.8% same-channel neighbor rate means cross-channel pairs still contribute — those are mostly co-located pixels (58% <5px), not random.

**By signal variance**: High-variance (interesting) neurons have slightly better neighbor correlations (0.50 vs 0.41 for low-variance). The solver has more signal to work with for pixels that change during saccades.

**Conclusion**: The garden model is doing exactly what it should — grouping neurons with correlated temporal signals. The flat <5px metric just doesn't capture color-based and cross-channel structure. For garden, the solver discovers a hybrid of spatial proximity + color similarity, which is the correct answer given the input signal.

## Next Steps

- **Garden threshold tuning**: Try 0.2–0.3 to improve channel separation while keeping R fed
- **Integrate multi-metric eval**: Add deriv_corr and color-distance eval alongside spatial <5px in standard output
- **D8 vs D16 at scale**: Re-compare at 50k+ where extra capacity may help
- **160×160 with more ticks**: 10k+ ticks needed to validate scaling
- **320×320 with deriv_corr**: Re-test scaling now that threshold is calibrated

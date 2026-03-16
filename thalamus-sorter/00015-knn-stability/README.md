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

### Run 002: Gray garden, step=5, 10k ticks

`gray_80x80_garden` preset with `--knn-track 10 --knn-report-every 100`. Threshold=0.1, deriv_corr, no normalization.

| Metric | Value |
|--------|-------|
| Total pairs | 141M |
| Elapsed | 51.4s |
| Spatial <3px | 1.2% |
| Spatial <5px | 2.7% |
| PCA disparity | 0.857 |
| K10 mean dist | 43.62 |

**KNN overlap trajectory:**
- tick 100: 0.04 (initial random)
- tick 200: 0.53 (rapid initial learning)
- tick 600: 0.84 (first peak)
- tick 1000: 0.67 (dip — restructuring)
- tick 2000: 0.94 (recovery)
- tick 3600: 0.99 (peak)
- tick 5000: 0.98 (plateau with gentle oscillation)
- tick 6600: 0.99 (second peak)
- tick 9200: 0.98 (late dip)
- tick 10000: 0.99 (final)

**Pattern:** Fast convergence to 0.99 by tick 3k, then stable oscillation in 0.97–0.99 range. Two restructuring episodes (dip at tick 900–1100 and minor dip at tick 9000–9300). Despite near-perfect KNN stability, spatial quality is only 1.2% <3px — the model converges to color similarity structure (same as ts-00014 findings), which is internally consistent but spatially diffuse.

### Run 003: Gray saccades, step=50, 10k ticks

`gray_80x80_saccades` preset with `--knn-track 10 --knn-report-every 100`. Threshold=0.5, deriv_corr, no normalization.

| Metric | Value |
|--------|-------|
| Total pairs | 201M |
| Elapsed | 54.4s |
| Spatial <3px | 96.0% |
| Spatial <5px | 100.0% |
| PCA disparity | 0.673 |
| K10 mean dist | 1.92 |

**KNN overlap trajectory:**
- tick 100: 0.03 (initial random)
- tick 600: 0.81 (first peak)
- tick 900: 0.53 (major dip — restructuring)
- tick 2000: 0.95 (recovery)
- tick 3300: 0.79 (second major dip)
- tick 5600: 0.96 (second recovery)
- tick 8000: 0.97 (plateau)
- tick 10000: 0.97 (final — still ~200 neurons changing per tick)

**Pattern:** Two large restructuring cycles (tick 700–900, tick 2100–3300) where KNN overlap drops 0.15–0.27. After tick 5k, gradual climb from 0.94 → 0.97. Unlike garden, saccades oscillations correspond to the model finding *better* spatial structure — each recovery reaches higher quality. Final plateau at 0.97 (not 0.99 like garden) because the model is still making small refinements. 96% spatial <3px confirms it found the correct topology.

### Run 004: Gray garden, step=50, 10k ticks

`gray_80x80_garden` preset with `--saccade-step 50 --knn-track 10 --knn-report-every 100`. Threshold=0.1, deriv_corr, no normalization.

| Metric | Value |
|--------|-------|
| Total pairs | 3.75M |
| Elapsed | 49.2s |
| Spatial <3px | 0.4% |
| Spatial <5px | 1.0% |
| PCA disparity | 1.00 |
| K10 mean dist | 53.28 |

**KNN overlap trajectory:**
- tick 100: 0.995 (already near-frozen)
- tick 300: 1.000
- tick 1200: 0.992 (minor dip)
- tick 3600: 1.000
- tick 5000: 0.970 (largest dip)
- tick 5800: 0.999 (recovery)
- tick 8200: 0.985 (second dip)
- tick 10000: 0.999 (final)

**Pattern:** Pair starvation. Only 3.75M pairs vs 141M for step=5 and 201M for saccades — a 37× reduction. With so few pairs entering the model, embeddings barely move from random initialization. KNN overlap is ~1.0 not because of convergence but because nothing is being learned. The two small dips (tick 4600–5200 and 7600–8200) show occasional batches of pairs briefly perturbing the frozen embeddings before they relax back. PCA disparity of 1.0 (perfect randomness) and 0.4% spatial quality confirm no structure was learned.

### KNN Spatial Accuracy — New Metric

Runs 002–004 showed KNN overlap alone is misleading. Added `knn_spatial_accuracy(width, radius=3, channels)` to the solver: for each neuron, what fraction of its K=10 embedding-space neighbors are within 3 pixels on the original grid (same channel). This directly measures whether KNN lists represent close pixels.

Random baseline: radius=3 circle covers ~28 pixels out of 6400 ≈ 0.4%.

### Run 005: Gray garden, step=5, 10k ticks (with spatial accuracy)

Repeat of run 002 with the new spatial accuracy metric.

| Metric | Value |
|--------|-------|
| Total pairs | 177M |
| Elapsed | 41.1s |
| Spatial <3px | 3.0% |
| Spatial <5px | 5.6% |

**KNN overlap vs spatial accuracy trajectory:**

| Tick | Overlap | Spatial | Notes |
|------|---------|---------|-------|
| 100 | 0.046 | 0.103 | Random init (spatial ~10% = weak color grouping already) |
| 500 | 0.826 | 0.104 | Overlap climbing fast, spatial flat |
| 1000 | 0.903 | 0.104 | High overlap, zero spatial progress |
| 2000 | 0.893 | 0.099 | Overlap steady, spatial *decreasing* |
| 3000 | 0.980 | 0.081 | Near-perfect overlap, spatial getting *worse* |
| 5000 | 0.977 | 0.068 | Minimum spatial — actively anti-sorting |
| 7000 | 0.964 | 0.082 | Slight recovery |
| 8000 | 0.974 | 0.120 | Spatial finally climbing |
| 9000 | 0.980 | 0.151 | Still climbing |
| 10000 | 0.969 | 0.188 | Final: overlap=0.97, spatial=0.19 |

**Pattern:** Overlap converges to 0.97 by tick 2k while spatial accuracy *decreases* from 0.10 to 0.07. The model is finding stable structure, but it's color similarity (distant same-color pixels), not spatial proximity. Spatial only begins recovering after tick 5k, reaching 0.19 by 10k — still far from useful. The KNN lists are "stable" in the wrong topology.

### Run 006: Gray saccades, step=50, 10k ticks (with spatial accuracy)

Repeat of run 003 with the new spatial accuracy metric.

| Metric | Value |
|--------|-------|
| Total pairs | 213M |
| Elapsed | 41.3s |
| Spatial <3px | 96.2% |
| Spatial <5px | 100.0% |

**KNN overlap vs spatial accuracy trajectory:**

| Tick | Overlap | Spatial | Notes |
|------|---------|---------|-------|
| 100 | 0.033 | 0.073 | Random init |
| 500 | 0.807 | 0.072 | Overlap climbing, spatial flat |
| 900 | 0.473 | 0.066 | Major restructuring dip |
| 1000 | 0.484 | 0.068 | Still restructuring |
| 2000 | 0.951 | 0.090 | Overlap recovered, spatial starting |
| 3000 | 0.792 | 0.288 | Second dip in overlap, but spatial 4× higher |
| 4000 | 0.891 | 0.710 | Overlap recovering, spatial surging |
| 5000 | 0.923 | 0.847 | Both climbing |
| 6000 | 0.948 | 0.919 | |
| 7000 | 0.955 | 0.958 | Spatial > overlap |
| 8000 | 0.957 | 0.973 | |
| 9000 | 0.970 | 0.987 | |
| 10000 | 0.969 | 0.992 | Final: overlap=0.97, spatial=0.99 |

**Pattern:** Both reach 0.97 overlap at 10k, but spatial tells the real story. Saccades' spatial accuracy lags overlap until tick 7k, then *surpasses* it. The restructuring dips (tick 900, tick 3000) correspond to the model abandoning wrong local structure — spatial doesn't improve during these dips but accelerates immediately after. By tick 7k, 95.8% of KNN neighbors are within 3px.

### Run 007: Gray garden, step=5, 50k ticks (with spatial accuracy)

| Metric | Value |
|--------|-------|
| Total pairs | 706M |
| Elapsed | 203.8s |
| Spatial <3px | 78.4% |
| Spatial <5px | 93.1% |

**KNN overlap vs spatial accuracy trajectory:**

| Tick | Overlap | Spatial | Notes |
|------|---------|---------|-------|
| 5000 | 0.897 | 0.088 | Still in color-similarity phase |
| 10000 | 0.907 | 0.219 | Spatial starting to climb |
| 15000 | 0.918 | 0.266 | Slow |
| 20000 | 0.758 | 0.360 | Major restructuring dip |
| 25000 | 0.685 | 0.660 | Overlap collapses, spatial surges |
| 30000 | 0.840 | 0.898 | Recovery |
| 34500 | 0.903 | 0.959 | Peak spatial |
| 40000 | 0.874 | 0.921 | Another restructuring cycle |
| 47000 | 0.919 | 0.978 | Second peak |
| 50000 | 0.960 | 0.953 | Final |

**Pattern:** Garden *does* eventually sort spatially — it just needs 5× more ticks than saccades. Three phases: (1) ticks 0–10k: color similarity dominates, spatial stays flat at ~0.1; (2) ticks 10k–30k: transition with large restructuring dips (overlap drops to 0.69), spatial climbs from 0.22 to 0.90; (3) ticks 30k–50k: oscillating equilibrium at spatial ~0.95, overlap ~0.90. Overlap never stabilizes — still oscillating 0.85–0.99 with ~2500 neurons changing per tick.

### Run 008: Gray garden, step=5, 100k ticks (warm start from run 007)

Continued from run 007's model.npy for another 50k ticks (100k total).

| Metric | Value |
|--------|-------|
| Total pairs | 766M (this segment) |
| Elapsed | 203.0s |
| Spatial <3px | 68.8% |
| Spatial <5px | 86.3% |

**KNN overlap vs spatial accuracy trajectory (total ticks):**

| Total tick | Overlap | Spatial | Notes |
|------------|---------|---------|-------|
| 55000 | 0.923 | 0.968 | Continuing from 50k |
| 60000 | 0.948 | 0.978 | Peak spatial for this segment |
| 65000 | 0.940 | 0.978 | Holding |
| 70000 | 0.861 | 0.915 | Major restructuring dip |
| 75000 | 0.931 | 0.943 | Recovery |
| 80000 | 0.907 | 0.946 | |
| 85000 | 0.908 | 0.935 | |
| 90000 | 0.959 | 0.965 | Second peak |
| 95000 | 0.963 | 0.960 | |
| 100000 | 0.983 | 0.957 | Final |

**Pattern:** No improvement from doubling training time. Spatial accuracy oscillates in 0.92–0.98 range, never breaking past the ~0.96 ceiling hit at 35k. Spatial <3px actually *dropped* from 78.4% to 68.8% — the model is in an oscillating equilibrium where restructuring cycles keep disrupting and rebuilding the same spatial structure. Overlap still never stabilizes (~1500 neurons changing per tick at 100k).

**Conclusion:** Garden's weak signal creates a hard ceiling at ~0.96 KNN spatial / ~78% <3px. The model has extracted all the spatial information available from the garden image's correlation structure. Further training just churns without improving. Compare saccades: 0.99 spatial / 96% <3px at 10k ticks.

### max_hit_ratio Ablation

Tested the effect of `max_hit_ratio` (MHR) on garden convergence. MHR limits what fraction of k_sample candidates can pass the correlation threshold per anchor — higher values let more (potentially noisy) pairs through.

**Quick scan at 5k ticks (garden, step=5):**

| MHR | Pairs | Spatial | <3px | Notes |
|-----|-------|---------|------|-------|
| 0.02 | 508K | 0.058 | 0.4% | Near starvation |
| 0.05 | 44M | 0.112 | 0.4% | Chaotic |
| 0.10 | 107M | 0.063 | 1.0% | Default |
| 0.20 | 93M | 0.087 | 0.6% | |
| 0.50 | 170M | 0.179 | 14.8% | Best early spatial |
| 1.00 | 116M | 0.069 | 1.6% | Noise overwhelms |

### Run 010: Gray garden, step=5, 50k, MHR=1.0 (no safety net)

| Metric | Value |
|--------|-------|
| Total pairs | 1.26B |
| Spatial <3px | 64.1% |
| Spatial <5px | 81.9% |

| Tick | Overlap | Spatial |
|------|---------|---------|
| 5000 | 0.261 | 0.219 |
| 10000 | 0.761 | 0.796 |
| 20000 | 0.858 | 0.932 |
| 30000 | 0.841 | 0.904 |
| 40000 | 0.851 | 0.927 |
| 50000 | 0.809 | 0.867 |

**Pattern:** 1.8× more pairs than MHR=0.1 but worse results. Overlap never exceeds 0.93 (vs 0.99). Noise pairs keep destabilizing the structure — the model reaches the same ~0.96 peak spatial but can't hold it.

### Run 011: Gray garden, step=5, 50k, MHR=0.5

| Metric | Value |
|--------|-------|
| Total pairs | 1.09B |
| Spatial <3px | 81.5% |
| Spatial <5px | 93.7% |

| Tick | Overlap | Spatial |
|------|---------|---------|
| 5000 | 0.261 | 0.219 |
| 10000 | 0.761 | 0.796 |
| 15000 | 0.801 | 0.884 |
| 20000 | 0.858 | 0.932 |
| 25000 | 0.837 | 0.901 |
| 30000 | 0.841 | 0.904 |
| 35000 | 0.851 | 0.932 |
| 40000 | 0.851 | 0.927 |
| 45000 | 0.868 | 0.941 |
| 50000 | 0.897 | 0.973 |

**Pattern:** Best garden result so far. MHR=0.5 lets through enough signal without overwhelming with noise. Peak spatial 0.98, final 0.97. Slightly better than MHR=0.1 (81.5% vs 78.4% <3px).

### Run 012: Gray garden, step=50, 50k, MHR=0.5

Testing whether MHR=0.5 can rescue step=50 (which starved at MHR=0.1).

| Metric | Value |
|--------|-------|
| Total pairs | 6.21B |
| Spatial <3px | 57.5% |
| Spatial <5px | 77.8% |

| Tick | Overlap | Spatial |
|------|---------|---------|
| 5000 | 0.420 | 0.480 |
| 10000 | 0.262 | 0.395 |
| 15000 | 0.537 | 0.599 |
| 20000 | 0.540 | 0.700 |
| 30000 | 0.480 | 0.650 |
| 40000 | 0.506 | 0.649 |
| 50000 | 0.512 | 0.634 |

**Pattern:** MHR=0.5 rescues step=50 from total starvation (3.75M → 6.21B pairs) and gets to 57.5% <3px (vs 0.4% with MHR=0.1). But overlap never stabilizes — oscillates 0.40–0.59 with ~6300/6400 neurons changing every tick. The large saccade step creates too many cross-image-region pairs even with MHR filtering. The model finds some spatial structure but can't consolidate it.

### Threshold Discovery: threshold=0.5 Fixes Garden

All previous garden runs used threshold=0.1 (the original preset). Run 013 tested garden with saccades parameters (threshold=0.5, step=50, MHR=0.1) and achieved **99.6% <3px** — matching saccades quality. The low threshold was the root cause of garden's poor convergence, not the image itself.

### Run 013: Gray garden, step=50, threshold=0.5, 50k (saccade params)

| Metric | Value |
|--------|-------|
| Total pairs | 179M |
| Spatial <3px | 99.6% |
| Spatial <5px | 100.0% |
| KNN spatial | 0.999 |
| KNN overlap | 0.990 (stable, ~450 neurons changing) |

### Saccade Step Ablation with threshold=0.5

Tested step=5 vs step=50 with the corrected threshold=0.5 to isolate the step effect.

### Run 014/016: Gray garden, threshold=0.5, step=5, 50k total

10k (run 014) then warm-started to 50k (run 016).

| Metric | Value |
|--------|-------|
| Total pairs | 8.7M (50k ticks) |
| Spatial <3px | 0.4% |
| KNN spatial | 0.892 |
| KNN overlap | 1.000 (frozen) |
| PCA disparity | 0.999 (random) |

| Total tick | Overlap | Spatial |
|------------|---------|---------|
| 1000 | 0.068 | 0.933 |
| 5000 | 0.926 | 0.982 |
| 10000 | 0.980 | 0.983 |
| 20000 | 0.999 | 0.891 |
| 30000 | 0.999 | 0.891 |
| 40000 | 1.000 | 0.892 |
| 50000 | 1.000 | 0.892 |

**Pattern:** Pair starvation. Only 8.7M pairs across 50k ticks — garden's nearby pixels (within 5px saccade window) almost never correlate above 0.5. KNN spatial of 0.89 is misleading — with so few pairs the embeddings barely moved from random init. PCA disparity 0.999 and 0.4% <3px confirm no real structure was learned. The high KNN spatial reflects random-init coincidence, not learned topology.

### Run 015/017: Gray garden, threshold=0.5, step=50, 50k total

10k (run 015) then warm-started to 50k (run 017).

| Metric | Value |
|--------|-------|
| Total pairs | 179M (50k ticks) |
| Spatial <3px | 99.7% |
| Spatial <5px | 100.0% |
| KNN spatial | 0.998 |
| KNN overlap | 0.972 (~950 neurons changing) |

| Total tick | Overlap | Spatial |
|------------|---------|---------|
| 1000 | 0.001 | 0.263 |
| 5000 | 0.903 | 0.347 |
| 10000 | 0.675 | 0.331 |
| 20000 | 0.958 | 0.993 |
| 30000 | 0.964 | 0.994 |
| 40000 | 0.970 | 0.996 |
| 50000 | 0.972 | 0.998 |

**Pattern:** Matches run 013 (independent run with same params). Needs ~15k ticks to break through, then converges cleanly to 99.7% <3px. Confirms the result is reproducible.

### Step=5 vs Step=50 at threshold=0.5

| | Step=5 | Step=50 |
|---|---|---|
| Pairs (50k ticks) | 8.7M | 179M |
| <3px | 0.4% | 99.7% |
| KNN spatial | 0.89 (frozen random) | 0.998 (learned) |
| PCA disparity | 0.999 | 0.819 |

**Conclusion:** With threshold=0.5, step=5 starves garden completely — the 5px saccade window is too small to find pixel pairs that correlate above 0.5 in this image. Step=50 provides the spatial diversity needed to generate enough above-threshold pairs. Garden presets updated to threshold=0.5 and saccade_step=50 to match saccades.

### Comparative Analysis

| Run | Signal | Step | Thr | MHR | Ticks | Pairs | KNN spatial | <3px | Interpretation |
|-----|--------|------|-----|-----|-------|-------|-------------|------|----------------|
| 005 | Garden | 5 | 0.1 | 0.1 | 10k | 177M | 0.19 | 3.0% | Color similarity phase |
| 007 | Garden | 5 | 0.1 | 0.1 | 50k | 706M | 0.95 | 78.4% | Slow spatial, oscillating |
| 008 | Garden | 5 | 0.1 | 0.1 | 100k | 1.5B | 0.96 | 68.8% | Plateau, no improvement |
| 011 | Garden | 5 | 0.1 | 0.5 | 50k | 1.09B | 0.97 | 81.5% | Best with old threshold |
| 010 | Garden | 5 | 0.1 | 1.0 | 50k | 1.26B | 0.87 | 64.1% | Too much noise |
| 012 | Garden | 50 | 0.1 | 0.5 | 50k | 6.21B | 0.63 | 57.5% | Chaotic |
| 016 | Garden | 5 | 0.5 | 0.1 | 50k | 8.7M | 0.89 | 0.4% | Starved |
| **013** | **Garden** | **50** | **0.5** | **0.1** | **50k** | **179M** | **0.999** | **99.6%** | **Solved** |
| **017** | **Garden** | **50** | **0.5** | **0.1** | **50k** | **179M** | **0.998** | **99.7%** | **Confirmed** |
| 006 | Saccades | 50 | 0.5 | 0.1 | 10k | 213M | 0.99 | 96.2% | Reference |
| 004 | Garden | 50 | 0.1 | 0.1 | 10k | 3.75M | — | 0.4% | Pair starvation |

**Key findings:**

1. **Threshold was the root cause, not the image.** Garden at threshold=0.5 + step=50 achieves 99.7% <3px — better than saccades (96.2%). The old threshold=0.1 let through massive numbers of color-similarity noise pairs that overwhelmed the spatial signal.

2. **KNN spatial accuracy is a valid stopping criterion.** Plateaus at 0.99+ for both signals with correct parameters. Monotonically increasing after the initial restructuring phase.

3. **Step and threshold interact.** Step=5 + threshold=0.5 = starvation (8.7M pairs). Step=50 + threshold=0.1 = starvation (3.75M pairs). Step=50 + threshold=0.5 = correct balance (179M pairs). The large saccade window provides spatial diversity; the high threshold ensures pair quality.

4. **Previous "signal quality ceiling" was actually a threshold misconfiguration.** Runs 005–012 concluded garden's weak signal created a hard ceiling at ~82% <3px. This was wrong — the ceiling was caused by threshold=0.1 admitting noise pairs, not by inherent image properties.

5. **MHR tuning was a red herring.** MHR ablation (runs 010–012) showed marginal improvement (78% → 82%) because it was compensating for the wrong threshold. With threshold=0.5, default MHR=0.1 works perfectly.

6. **Garden presets updated.** Both `gray_80x80_garden.json` and `rgb_80x80_garden.json` changed to threshold=0.5 and saccade_step=50 to match saccades presets.

### KNN Swap Distribution

Analyzed per-neuron swap counts to understand whether the ~950 neurons still changing at convergence (run 017 baseline) are a uniform phenomenon or a tail.

Tracked `top50_swaps` (average swaps for the 50% most stable neurons) and `top90_swaps` (90% most stable) alongside overall overlap.

**Garden, threshold=0.5, step=50, 50k ticks:**

| Tick | Overlap | Spatial | top50 swaps | top90 swaps |
|------|---------|---------|-------------|-------------|
| 5000 | 0.001 | 0.367 | 10.0 | 10.0 |
| 10000 | 0.590 | 0.267 | 1.3 | 3.5 |
| 15000 | 0.330 | 0.299 | 5.2 | 6.4 |
| 20000 | 0.249 | 0.869 | 5.2 | 7.2 |
| 25000 | 0.486 | 0.975 | 2.8 | 4.6 |
| 30000 | 0.691 | 0.991 | 0.7 | 2.5 |
| 35000 | 0.829 | 0.995 | 0.0 | 1.1 |
| 40000 | 0.899 | 0.997 | 0.0 | 0.4 |
| 45000 | 0.848 | 0.998 | 0.0 | 0.8 |
| 50000 | 0.881 | 0.998 | 0.0 | 0.6 |

**Conclusion:** Swapping is entirely a **tail phenomenon**. At convergence (tick 40k+):
- **Top 50%** of neurons: **0.0 swaps** — completely frozen KNN lists
- **Top 90%**: **0.4–0.8 swaps** — near-zero
- **Bottom 10%** (~640 neurons): doing all the swapping, averaging ~5+ swaps per tick

The ~950 neurons still changing at convergence are concentrated in the bottom 10% — likely neurons at spatial region boundaries or areas with ambiguous correlations where multiple nearby pixels are equally valid neighbors. The bulk of the map (90%+) has completely stable, spatially correct KNN lists.

### RGB Multi-Channel Experiments (anchor_batches=3)

Tested both RGB presets (19,200 neurons = 80×80×3) with `--anchor-batches 3` for 50k ticks. Both presets now use threshold=0.5, step=50.

### Run 018: RGB garden, 50k, anchor_batches=3

| Metric | Value |
|--------|-------|
| Total pairs | 1.19B |
| Elapsed | 1449s (24 min) |
| Flat <3px | 40.7% |
| Flat <5px | 78.2% |
| KNN spatial | 0.964 |
| KNN overlap | 0.854 |

| Tick | Overlap | Spatial | top50 | top90 |
|------|---------|---------|-------|-------|
| 5000 | 0.000 | 0.038 | 10.0 | 10.0 |
| 10000 | 0.325 | 0.202 | 5.0 | 6.4 |
| 15000 | 0.272 | 0.661 | 5.4 | 7.0 |
| 20000 | 0.586 | 0.868 | 2.0 | 3.6 |
| 25000 | 0.793 | 0.918 | 0.1 | 1.5 |
| 30000 | 0.849 | 0.936 | 0.0 | 0.9 |
| 35000 | 0.862 | 0.938 | 0.0 | 0.7 |
| 40000 | 0.829 | 0.951 | 0.0 | 1.1 |
| 45000 | 0.872 | 0.960 | 0.0 | 0.7 |
| 50000 | 0.854 | 0.964 | 0.0 | 0.8 |

**Channel analysis:**
- Channel separation: **100%** (all K=10 neighbors are same channel)
- Per-channel mean pixel distance: R=1.70, G=1.69, B=1.69
- Per-channel <5px: **100%** for all channels

![RGB garden 50k](rgb_garden_50k.png)

### Run 019: RGB saccades, 50k, anchor_batches=3

| Metric | Value |
|--------|-------|
| Total pairs | 8.12B |
| Elapsed | 1491s (25 min) |
| Flat <3px | 41.5% |
| Flat <5px | 75.6% |
| KNN spatial | 0.758 |
| KNN overlap | 0.676 |

| Tick | Overlap | Spatial | top50 | top90 |
|------|---------|---------|-------|-------|
| 5000 | 0.000 | 0.364 | 10.0 | 10.0 |
| 10000 | 0.294 | 0.654 | 5.6 | 6.7 |
| 15000 | 0.523 | 0.746 | 2.9 | 4.3 |
| 20000 | 0.673 | 0.712 | 1.5 | 2.8 |
| 25000 | 0.655 | 0.768 | 1.7 | 3.0 |
| 30000 | 0.734 | 0.793 | 0.9 | 2.2 |
| 35000 | 0.729 | 0.777 | 0.9 | 2.2 |
| 40000 | 0.728 | 0.727 | 0.9 | 2.2 |
| 45000 | 0.672 | 0.714 | 1.4 | 2.8 |
| 50000 | 0.676 | 0.758 | 1.5 | 2.8 |

**Channel analysis:**
- Channel separation: **100%** (all K=10 neighbors are same channel)
- Per-channel mean pixel distance: R=1.84, G=1.85, B=1.83
- Per-channel <5px: **100%** for all channels

![RGB saccades 50k](rgb_saccades_50k.png)

### RGB Comparison

| | RGB Garden | RGB Saccades |
|---|---|---|
| Channel separation | 100% | 100% |
| Per-channel mean dist | 1.69 | 1.84 |
| Per-channel <5px | 100% | 100% |
| KNN spatial | 0.964 | 0.758 |
| KNN overlap | 0.854 | 0.676 |
| Pairs | 1.19B | 8.12B |
| top50 swaps (50k) | 0.0 | 1.5 |
| top90 swaps (50k) | 0.8 | 2.8 |

**Key findings:**

1. **Both achieve perfect channel separation and excellent per-channel spatial quality** (mean dist <2px, 100% <5px). The flat eval metrics (40–42% <3px) are misleading for multi-channel models — they penalize cross-channel distance.

2. **RGB garden outperforms RGB saccades on KNN metrics** (spatial 0.964 vs 0.758, overlap 0.854 vs 0.676). Garden has fewer pairs (1.19B vs 8.12B), meaning cleaner signal after threshold filtering. With threshold=0.5, the garden image's correlations are as informative as saccades'.

3. **Saccades still churning at 50k.** Top50 swaps still at 1.5 (vs garden's 0.0), top90 at 2.8 (vs 0.8). The 8.12B pairs create continuous pressure that prevents stabilization, even though the per-channel structure is already correct. May need longer to settle, or the high pair volume is causing perpetual restructuring.

4. **KNN spatial accuracy works for multi-channel** — correctly measures within-channel spatial proximity (the `channels` parameter ensures only same-channel neighbors count).

### lr_decay + normalize_every Interaction

Does the dual annealing mechanism from ts-00014 change KNN convergence patterns?

### Run 021: Gray saccades, 50k, lr_decay=0.8, normalize_every=5000

| Metric | Value |
|--------|-------|
| Total pairs | 1.01B |
| Elapsed | 203.7s |
| Spatial <3px | 97.5% |
| KNN spatial | 0.999 |
| KNN overlap | 0.797 |
| Final lr | 0.000107 |

| Tick | Overlap | Spatial | top50 | top90 | lr |
|------|---------|---------|-------|-------|----|
| 5000 | 0.001 | 0.458 | 10.0 | 10.0 | 0.000800 |
| 10000 | 0.190 | 0.950 | 6.8 | 7.9 | 0.000640 |
| 15000 | 0.590 | 0.992 | 2.2 | 3.6 | 0.000512 |
| 20000 | 0.703 | 0.995 | 0.8 | 2.4 | 0.000410 |
| 30000 | 0.757 | 0.999 | 0.5 | 1.9 | 0.000262 |
| 40000 | 0.790 | 1.000 | 0.2 | 1.5 | 0.000168 |
| 50000 | 0.797 | 0.999 | 0.0 | 1.3 | 0.000107 |

### Run 022: Gray saccades, 50k, no decay (baseline)

| Metric | Value |
|--------|-------|
| Total pairs | 1.03B |
| Spatial <3px | 95.0% |
| KNN spatial | 0.997 |
| KNN overlap | 0.717 |

| Tick | Overlap | Spatial | top50 | top90 |
|------|---------|---------|-------|-------|
| 5000 | 0.001 | 0.478 | 10.0 | 10.0 |
| 10000 | 0.276 | 0.939 | 5.7 | 6.9 |
| 15000 | 0.582 | 0.997 | 2.3 | 3.7 |
| 20000 | 0.690 | 0.997 | 1.1 | 2.6 |
| 30000 | 0.685 | 0.999 | 1.3 | 2.6 |
| 40000 | 0.734 | 0.998 | 0.7 | 2.1 |
| 50000 | 0.717 | 0.997 | 0.8 | 2.3 |

**lr_decay comparison:**

| | No decay | decay=0.8, norm=5000 |
|---|---|---|
| KNN overlap (50k) | 0.717 | **0.797** |
| KNN spatial (50k) | 0.997 | **0.999** |
| <3px | 95.0% | **97.5%** |
| top50 swaps | 0.8 | **0.0** |
| top90 swaps | 2.3 | **1.3** |

**Finding:** lr_decay improves KNN stability moderately — overlap from 0.72 to 0.80, top50 swaps from 0.8 to 0.0. The decaying lr lets the system settle more cleanly in late training. Spatial accuracy is marginally better (0.999 vs 0.997). The effect is real but smaller than the threshold/step corrections. Both configs achieve excellent spatial quality (>95% <3px).

### Grid Size Effect

Do KNN stability dynamics scale with grid size? Tested 40×40, 80×80, 160×160 with saccades, threshold=0.5, step=50, 50k ticks.

### Run 023/024: 40×40 saccades, 50k total (n=1,600, k_sample=50)

| Tick | Overlap | Spatial | top50 | top90 |
|------|---------|---------|-------|-------|
| 10000 | 0.864 | 0.131 | 0.2 | 1.0 |
| 20000 | 0.722 | 0.774 | 1.1 | 2.3 |
| 30000 | 0.718 | 0.900 | 0.8 | 2.3 |
| 40000 | 0.788 | 0.986 | 0.3 | 1.5 |
| 50000 | 0.788 | 0.986 | 0.3 | 1.5 |

Final: pairs=39M, <3px=98.6%, spatial=0.986

### Run 025: 160×160 saccades, 50k (n=25,600, k_sample=800)

| Tick | Overlap | Spatial | top50 | top90 |
|------|---------|---------|-------|-------|
| 10000 | 0.092 | 0.091 | 8.3 | 9.0 |
| 20000 | 0.325 | 0.864 | 5.2 | 6.4 |
| 30000 | 0.700 | 0.955 | 1.4 | 2.6 |
| 40000 | 0.791 | 0.981 | 0.6 | 1.7 |
| 50000 | 0.815 | 0.990 | 0.3 | 1.4 |

Final: pairs=1.11B, <3px=98.9%, spatial=0.990

### Run 068: 320×320 saccades, 50k (n=102,400, k_sample=3200)

First successful 320×320 run. Uses matmul correlation path (previous attempt with gather path hung — 3GB tensor per tick caused GPU thrashing with async render worker).

| Tick | Overlap | Spatial | top50 | top90 |
|------|---------|---------|-------|-------|
| 5000 | 0.000 | 0.085 | 10.0 | 10.0 |
| 10000 | 0.507 | 0.083 | 3.3 | 4.5 |
| 15000 | 0.271 | 0.075 | 5.5 | 7.0 |
| 20000 | 0.103 | 0.097 | 8.1 | 8.9 |
| 25000 | 0.266 | 0.086 | 6.0 | 7.1 |
| 30000 | 0.584 | 0.085 | 2.8 | 3.8 |
| 35000 | 0.720 | 0.089 | 1.5 | 2.5 |
| 40000 | 0.746 | 0.122 | 1.3 | 2.2 |
| 45000 | 0.648 | 0.225 | 1.9 | 3.1 |
| 50000 | 0.553 | 0.427 | 2.8 | 4.1 |

Final: pairs=1.12B, elapsed=336s (6.7 ms/tick), PCA=0.808, <3px=58.2%, <5px=82.8%, spatial=0.427

**320×320 does not converge at 50k ticks.** Overlap is unstable (peaks at 0.746 at tick 40k, then drops), spatial accuracy is only 0.427 (vs 0.99 at smaller grids). With batch_size=256 (default), only 256 anchors are updated per tick — that's 0.25% of 102k neurons. At 80×80, 256/6400 = 4% coverage per tick. The 16× lower coverage rate likely explains the slow convergence.

### Run 069: 320×320 saccades, anchor_batches=4, 50k (n=102,400, k_sample=3200)

Same as run 068 but with `--anchor-batches 4` (1024 anchors/tick, 1% coverage). Logged to wandb: `320x320-ab4-50k`.

| Tick | Overlap | Spatial | top50 | top90 |
|------|---------|---------|-------|-------|
| 5000 | 0.000 | 0.083 | 10.0 | 10.0 |
| 10000 | 0.101 | 0.093 | 8.2 | 8.9 |
| 15000 | 0.258 | 0.574 | 5.6 | 7.1 |
| 20000 | 0.321 | 0.922 | 5.2 | 6.5 |
| 25000 | 0.590 | 0.969 | 2.5 | 3.7 |
| 30000 | 0.708 | 0.981 | 1.4 | 2.5 |
| 35000 | 0.775 | 0.987 | 0.7 | 1.8 |
| 40000 | 0.824 | 0.990 | 0.3 | 1.4 |
| 45000 | 0.846 | 0.992 | 0.2 | 1.1 |
| 50000 | 0.872 | 0.993 | 0.1 | 0.9 |

Final: pairs=4.52B, elapsed=794s (15.9 ms/tick), PCA=0.901, <3px=99.2%, <5px=100%, spatial=0.993

**4× anchors fully restores convergence at 320×320.** Spatial accuracy reaches 0.993 (vs 0.427 with default anchors), <3px jumps from 58.2% to 99.2%, and overlap is monotonically increasing (0.872 vs unstable 0.553). The convergence trajectory closely matches 160×160 run 025.

### Runs 070–071: 320×320 saccades, anchor_batches=16, 50k (4096 anchors/tick, 4% coverage)

Two identical runs to test reproducibility. Uses KNN_STABLE_INSERT (replace-worst-slot instead of topk re-sort). Input image corrected: K_320_g.png resized from K_crop_g.png (1024×1024) via INTER_AREA.

| Tick | Run 070 Overlap | Run 070 Spatial | Run 071 Overlap | Run 071 Spatial |
|------|-----------------|-----------------|-----------------|-----------------|
| 5000 | 0.000 | 0.739 | 0.000 | 0.774 |
| 10000 | 0.318 | 0.981 | 0.353 | 0.976 |
| 15000 | 0.663 | 0.993 | 0.662 | 0.989 |
| 20000 | 0.783 | 0.996 | 0.785 | 0.993 |
| 30000 | 0.831 | 0.998 | 0.818 | 0.998 |
| 40000 | 0.828 | 0.999 | 0.855 | 0.999 |
| 50000 | 0.831 | 1.000 | 0.845 | 1.000 |

Both: elapsed=~3331s (66.6 ms/tick), <3px=99.0–99.5%, <5px=100%, spatial=1.000

**16× anchors reach spatial=1.000** (perfect KNN spatial accuracy), up from 0.993 with ab4. But 66.6 ms/tick is 4.2× slower than ab4 (15.9 ms/tick) — the correlation matmul runs 16 times instead of 4. Diminishing returns: ab4 reaches 0.993, ab16 reaches 1.000 at 4× the wall time.

### Run 072: 320×320 saccades, anchor_batches=8, 50k (2048 anchors/tick, 2% coverage)

Testing the middle ground between ab4 (1024 anchors) and ab16 (4096 anchors).

| Tick | Overlap | Spatial |
|------|---------|---------|
| 5000 | 0.000 | 0.091 |
| 10000 | 0.074 | 0.872 |
| 15000 | 0.470 | 0.979 |
| 20000 | 0.688 | 0.989 |
| 25000 | 0.751 | 0.993 |
| 30000 | 0.809 | 0.995 |
| 35000 | 0.816 | 0.996 |
| 40000 | 0.822 | 0.997 |
| 45000 | 0.831 | 0.998 |
| 50000 | 0.857 | 0.998 |

Final: elapsed=1830s (36.6 ms/tick), PCA=0.881, <3px=99.2%, <5px=100%, spatial=0.998

### Run 073: 320×320 saccades, batch_size=512, anchor_batches=4, 50k (2048 anchors/tick, 2% coverage)

Same anchors/tick as run 072 (2048), but fewer larger batches instead of more smaller ones. Tests whether bigger matmul per batch (512×102400 vs 256×102400) improves throughput.

| Tick | Overlap | Spatial |
|------|---------|---------|
| 5000 | 0.000 | 0.088 |
| 10000 | 0.176 | 0.723 |
| 15000 | 0.389 | 0.972 |
| 20000 | 0.663 | 0.988 |
| 25000 | 0.765 | 0.993 |
| 30000 | 0.795 | 0.995 |
| 35000 | 0.828 | 0.997 |
| 40000 | 0.835 | 0.998 |
| 45000 | 0.846 | 0.998 |
| 50000 | 0.872 | 0.999 |

Final: elapsed=2142s (42.8 ms/tick), PCA=0.979, <3px=99.5%, <5px=100%, spatial=0.999

### 320×320 Anchor Scaling Summary

| Config | Anchors/tick | Coverage | ms/tick | Spatial | <3px | Overlap |
|--------|-------------|----------|---------|---------|------|---------|
| b256×ab1 (run 068) | 256 | 0.25% | 6.7 | 0.427 | 58.2% | 0.553 |
| b256×ab4 (run 069) | 1024 | 1% | 15.9 | 0.993 | 99.2% | 0.872 |
| b256×ab8 (run 072) | 2048 | 2% | 36.6 | 0.998 | 99.2% | 0.857 |
| b512×ab4 (run 073) | 2048 | 2% | 42.8 | 0.999 | 99.5% | 0.872 |
| b256×ab16 (runs 070–071) | 4096 | 4% | 66.6 | 1.000 | 99.0–99.5% | 0.831–0.845 |

**Grid size comparison at 50k ticks:**

| Grid | n | k_sample | Anchors/tick | Coverage | Pairs | Overlap | Spatial | <3px | top50 | top90 |
|------|---|----------|-------------|----------|-------|---------|---------|------|-------|-------|
| 40×40 | 1,600 | 50 | 256 | 16% | 39M | 0.79 | 0.986 | 98.6% | 0.3 | 1.5 |
| 80×80 | 6,400 | 200 | 256 | 4% | 1.03B | 0.72 | 0.997 | 95.0% | 0.8 | 2.3 |
| 160×160 | 25,600 | 800 | 256 | 1% | 1.11B | 0.82 | 0.990 | 98.9% | 0.3 | 1.4 |
| 320×320 | 102,400 | 3,200 | 256 | 0.25% | 1.12B | 0.55 | 0.427 | 58.2% | 2.8 | 4.1 |
| 320×320 | 102,400 | 3,200 | 1024 | 1% | 4.52B | 0.87 | 0.993 | 99.2% | 0.1 | 0.9 |
| 320×320 | 102,400 | 3,200 | 2048 | 2% | 8.9B | 0.86 | 0.998 | 99.2% | — | — |
| 320×320 | 102,400 | 3,200 | 4096 | 4% | 17.9B | 0.84 | 1.000 | 99.5% | — | — |

**Findings:**

1. **Both k_sample and anchor_sample must scale with n.** k_sample at ~3% of n keeps neighbor discovery effective, but anchor_sample must also scale to maintain sufficient per-tick coverage. At 320×320, 256 anchors (0.25% coverage) is insufficient; 1024 anchors (1% coverage) restores convergence.

2. **Coverage rate threshold is ~1%.** 160×160 converges at 1% coverage (256/25600), 320×320 fails at 0.25% but succeeds at 1%. This suggests ~1% per-tick coverage is the minimum for stable convergence at 50k ticks.

3. **Overlap instability is a coverage problem, not a scale problem.** Run 068's non-monotonic overlap (peaking then dropping) disappears entirely with 4× anchors. The embedding space needs enough updates per tick to make coherent progress rather than random walks.

4. **Tail fraction at 320×320 with sufficient anchors** (top50=0.1, top90=0.9) is actually *better* than smaller grids, suggesting the larger embedding space has more room for stable neighbor assignments.

5. **Diminishing returns above 1% coverage.** ab4 (1%) → ab8 (2%) → ab16 (4%) improves spatial from 0.993 → 0.998 → 1.000, but wall time scales linearly (15.9 → 36.6 → 66.6 ms/tick). The sweet spot is ab4 or ab8 — ab16 pays 4× wall time for +0.007 spatial.

6. **Batch size 512 is slower than 256 at same anchor count.** b512×ab4 (42.8 ms/tick) vs b256×ab8 (36.6 ms/tick) — both produce 2048 anchors/tick but b512 is 17% slower. The larger matmul per iteration doesn't compensate for the overhead. b256 is the better batch size for 320×320.

### K Value and Ring Completion

Hypothesis: on a grid, neighbors fall in discrete distance rings (k=4 cardinal, k=8 +diagonals, k=12 +dist-2, k=20 +dist-√5, k=24 +dist-2√2). If K exactly completes a ring, there's no ambiguity about which neighbors belong → overlap should reach 1.0. K=10 sits between rings (8 immediate + 2 from a ring of 4), creating inherent swapping.

Tested K=8 (complete ring), K=10 (mid-ring), K=24 (complete ring) at 50k ticks with saccades:

| K | Ring complete? | Overlap | Spatial | top50 | top90 | % changed |
|---|---------------|---------|---------|-------|-------|-----------|
| 8 | yes | 0.636 | 0.997 | 1.2 | 2.5 | 87% |
| 10 | no | 0.696 | 0.998 | 1.1 | 2.5 | 84% |
| 24 | yes | 0.835 | 0.998 | 1.2 | 3.1 | 82% |

**Result: hypothesis rejected.** Ring-complete K values (8, 24) don't reach higher overlap than mid-ring K=10. K=8 is actually *worse* (0.636 vs 0.696).

**Why:** KNN is computed in **embedding space**, not grid space. Spatially equidistant pixels end up at slightly different embedding distances — the embedding doesn't perfectly preserve the grid metric. Swapping is caused by **embedding-space noise** (nearby pixels have nearly identical embeddings, so dot-product rankings fluctuate with each training update), not grid-ring ambiguity. K=24 has the highest overlap simply because more neighbors per neuron means each swap is a smaller fraction of the total.

### RGB Garden Long-Run Chain (2M ticks)

Warm-start chain: 40 sequential 50k-tick runs, each reusing the previous model. RGB garden preset (19,200 neurons = 80×80×3), anchor_batches=3, KNN report every 5000 ticks, 100 frames saved per run.

**Eval metrics over training:**

| Total ticks | PCA | K10 mean | <3px | <5px | KNN overlap | KNN spatial | top50 | top90 | % changed |
|-------------|-----|----------|------|------|-------------|-------------|-------|-------|-----------|
| 50k | 0.943 | 3.68 | 41.5% | 79.6% | 0.879 | 0.961 | 0.0 | 0.6 | 35% |
| 100k | 0.941 | 3.68 | 40.6% | 79.3% | 0.900 | 0.969 | 0.0 | 0.4 | 28% |
| 250k | 0.932 | 3.67 | 40.2% | 79.3% | 0.940 | 0.974 | 0.0 | 0.1 | 17% |
| 500k | 0.914 | 3.66 | 40.2% | 79.4% | 0.912 | 0.976 | 0.0 | 0.3 | 24% |
| 750k | 0.966 | 3.66 | 40.1% | 79.2% | 0.962 | 0.976 | 0.0 | 0.0 | 11% |
| 1000k | 0.974 | 3.66 | 40.0% | 79.2% | 0.944 | 0.976 | 0.0 | 0.1 | 14% |
| 1500k | 0.972 | 3.66 | 40.0% | 79.2% | 0.931 | 0.976 | 0.0 | 0.2 | 18% |
| 2000k | 0.970 | 3.66 | 40.0% | 79.2% | 0.947 | 0.974 | 0.0 | 0.1 | 14% |

**Findings:**

1. **Spatial accuracy plateaus at 0.975 by ~250k ticks** and is completely stable for the remaining 1.75M ticks. Warm-starting does not degrade quality.

2. **KNN overlap stabilizes around 0.94-0.95** — never reaches 1.0 due to the ~15% tail neurons that keep swapping. This is consistent across all runs and matches the gray saccades behavior.

3. **top50 swaps = 0.0 throughout** — 50% of neurons have completely frozen KNN lists from the first 50k-tick run onwards. top90 oscillates between 0.0 and 0.3.

4. **Eval metrics (<3px, <5px) plateau early and don't improve.** <3px=40% and <5px=79% are the RGB 3-channel eval numbers (measured across all 19,200 neurons including cross-channel distances). The per-channel KNN spatial accuracy (0.975) is the more meaningful quality metric.

5. **PCA disparity climbs from 0.94 to 0.977** between 500k and 750k, then plateaus. The embeddings become more structured over extended training.

6. **Conclusion: training past ~250k ticks is unnecessary for RGB garden.** Spatial quality converges by 250k. PCA disparity gains plateau by 750k. The warm-start chain confirms the model is stable under indefinite continued training — no catastrophic drift or degradation.


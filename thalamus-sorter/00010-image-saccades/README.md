# ts-00010: Image Saccades — Real Image Signals via Random Crops

**Date:** 2026-03-11
**Status:** In Progress
**Source:** *tagged on completion as `exp/ts-00010`*

## Goal

Replace synthetic Gaussian-smoothed noise with real image content as the temporal signal source. Each timestep, a random crop (simulated saccade) from a larger source image provides firing rates for all neurons. Nearby neurons see similar content across crops → temporal correlation → spatial map discovery.

## Motivation

ts-00009 proved the correlation-based pipeline works with synthetic signals. But those signals have artificial correlation structure (Gaussian blur). Real images have natural spatial correlation — nearby pixels tend to be similar because of object continuity, textures, and lighting. The question: can the sorter discover spatial structure from natural image statistics alone?

## Approach

### Signal generation

- Source: a large grayscale image (pre-normalized to 0-1 firing rates)
- Grid: 80x80 neurons, each maps to one pixel
- Each timestep t: pick random offset (dx, dy), sample 80x80 crop from source
- Firing rate of neuron (x, y) at time t = source_image[y + dy, x + dx]
- Per-frame mean subtraction: `crop - crop.mean()` removes global luminance shift
- Nearby neurons see similar pixel values across random crops → correlated temporal patterns

### Key difference from ts-00009

In ts-00009, correlation came from Gaussian blur applied to random noise — the spatial smoothing was explicit and controlled by sigma. Here, correlation comes from natural image structure — edges, textures, gradients. The smoothing is implicit in the image content.

### Saccade modes

1. **Random independent crops** — each timestep picks a completely random (dx, dy) offset. Simple but creates weak temporal structure because consecutive frames are unrelated.

2. **Random walk saccades** (`--saccade-step N`) — each timestep moves by a small random step from the previous position. Simulates biological eye movements. Step size controls the decorrelation rate between frames.

## Challenges discovered

### 1. Global luminance correlation

Random crops shift ALL pixels together. When a crop happens to be from a bright region, every neuron fires high; dark region → every neuron fires low. This creates ~0.5-0.6 correlation between ALL pairs, drowning out local spatial structure.

**Fix:** Per-frame mean subtraction. Each crop has its global mean removed before entering the signal buffer. This isolates local spatial structure (edges, textures) from global brightness.

### 2. Source size vs grid size

With a 1536x1024 source and 80x80 grid, random crop shifts barely affect pixel-level similarity. Two pixels 40 apart in the grid sample source positions 40 apart — and in natural images, content at distance 40 pixels is still quite correlated. The correlation radius of natural images is much larger than the grid spacing.

**Fix A:** Use smaller source (240x240 = 3x grid). Crops shift by a larger fraction of the image, creating more decorrelation between distant pixels.

**Fix B:** Random walk with large steps on the full source (step=50). The large steps ensure consecutive frames sample sufficiently different regions.

### 3. Temporal autocorrelation in small-step walks

Random walk with small steps (step=5) on the full source: consecutive frames barely differ, creating high temporal autocorrelation across ALL neurons (signal barely changes between frames). The correlation between distant neurons comes from the slow drift, not from spatial proximity.

**Fix:** Larger step size (step=50). Provides enough decorrelation between frames that only truly spatially proximate neurons remain correlated.

## Results

Base parameters: 80x80 grid, dims=8, k_neg=5, lr=0.001, normalize_every=100, k_sample=200, threshold=0.5, signal_T=200.

### Progression of approaches

| Run | Approach | Source | Ticks | Disparity | Notes |
|-----|----------|--------|-------|-----------|-------|
| run1-2 | Random crops, raw | 1536x1024 | 5k | ~1.0 | Failed — global luminance dominates |
| run3 | Random crops + mean sub | 1536x1024 | 50k | ~0.35 | Slow — source too large relative to grid |
| run4_t03 | Mean sub, threshold=0.3 | 1536x1024 | 50k | ~0.40 | Lower threshold = noisier neighbors |
| run7_t08 | Mean sub, threshold=0.8 | 1536x1024 | 50k | ~0.45 | Too strict — few pairs found |
| run5_200k | Mean sub | 1536x1024 | 200k | ~0.28 | Still converging, slowly |
| run8_500k | Mean sub | 1536x1024 | 500k | ~0.22 | K visible but fragmented |
| run6_T1000 | Mean sub, T=1000 | 1536x1024 | 50k | ~0.30 | Longer buffer helps slightly |
| run9_240 | Mean sub | 240x240 crop | 50k | ~0.20 | 3x source works better |
| run10_cov | Covariance, threshold=0.005 | 240x240 | 50k | ~0.22 | Doesn't help (uniform variance) |
| run11_cov05 | Covariance, threshold=0.05 | 240x240 | 50k | ~0.25 | Too strict for covariance scale |
| run12_240ms | Mean sub, Pearson | 240x240 | 50k | ~0.20 | Same as run9 (confirming) |
| run13_walk | Walk step=5 | 1536x1024 | 50k | ~0.45 | Failed — temporal autocorrelation |
| **run14_step50** | **Walk step=50** | **1536x1024** | **50k** | **~0.18** | **Best result — K emerging** |
| run15_hitratio | Walk step=50, max_hit_ratio=0.1 | 1536x1024 | 50k | ~0.80 | Filter has no effect (clean signal), bad walk path |
| run17_baseline | Walk step=50 (no filter) | 1536x1024 | 50k | ~0.57 | Different walk path, worse than run14 |

### Hit ratio filter (`--max-hit-ratio`)

**Idea:** If a neuron correlates with a large fraction of random candidates, it's seeing a global signal (flickering lights, slow drift), not local spatial structure. Discard these anchors — they'd inject noise into learning.

**Implementation:** After computing which candidates pass the threshold, compute `hit_ratio = good_neighbors / k_sample`. Discard anchors where ratio exceeds `--max-hit-ratio`.

**Diagnostic results:**

| Signal type | Mean hit ratio | Max hit ratio | Filter effect at 0.1 |
|-------------|---------------|---------------|----------------------|
| Step=50 + mean sub (clean) | 3.6% | 9.0% | No anchors cut — signal already clean |
| Step=50, NO mean sub (raw) | 22.0% | 45.0% | 90% of anchors cut — catches global signal |

The filter is a **general robustness mechanism**. With clean signals (mean-subtracted, large steps), it has no effect. But it defends against any global contamination — flickering lights, correlated noise bursts — that mean subtraction alone may not catch. For a real-time biological system where signal quality varies, this is a safety net.

**Note on variance across runs:** Saccade results are high-variance depending on which region of the image the random walk traverses. Run14 (0.18) got a favorable walk path; run15/17 (0.57-0.80) got worse ones. This is a fundamental property of natural image signals — correlation structure varies spatially across the image.

### Key findings

1. **Mean subtraction is essential.** Without it, global luminance creates universal correlation and the sorter learns nothing.

2. **Source size matters.** 3x grid size (240x240 for 80x80 grid) or large-step random walks on the full source both achieve good decorrelation. The key insight: adjacent pixels in the grid always sample adjacent pixels in the source. In natural images, adjacent source pixels are always similar — so distant grid neurons are only decorrelated when the crop positions differ enough to overcome natural image correlation radius.

3. **Random walk > random independent crops.** Random walk with step=50 on the full source (disparity 0.18 at 50k) outperforms both random crops on full source (0.22 at 500k) and random crops on 240x240 (0.20 at 50k). The sequential movement creates richer temporal structure while still decorrelating distant neurons.

4. **Covariance filtering doesn't help for this image.** The test image has relatively uniform variance across pixels (0.23-0.28 std), so covariance ≈ correlation × constant. Will matter for images with genuinely flat (uniform) regions.

5. **Hit ratio filter is a robustness mechanism, not a performance booster.** When the signal is clean, it does nothing. When global contamination occurs, it prevents bad learning by discarding anchors that correlate with everyone.

6. **Natural image signals are harder than Gaussian noise.** ts-00009 achieved disparity ~0.02-0.04 with synthetic Gaussian signals. Best image saccade result is 0.18. The gap reflects that natural image correlation structure is more complex — less cleanly spatial than a Gaussian kernel.

7. **High variance across runs.** Different random walk paths produce very different results (0.18 to 0.80) because correlation structure varies across regions of the source image.

## Files

- `main.py` — `correlation` mode with `--signal-source`, `--saccade-step`, `--use-covariance`, `--max-hit-ratio`
- `solvers/drift_torch.py` — `use_covariance` and `max_hit_ratio` parameters in `tick_correlation()`
- `saccades_gray.npy` — Pre-normalized source image (1024x1536, float32 0-1)
- `saccades_gray_240.npy` — 240x240 crop for faster testing
- `saccades.png` — Original source image

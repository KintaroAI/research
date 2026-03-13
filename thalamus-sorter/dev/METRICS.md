# Thalamus Sorter — Evaluation Metrics

## Primary metrics

### Procrustes disparity (PCA)

Best-fit rotation/scaling of 2D PCA projection to known grid coordinates, then residual error.

| Range | Meaning |
|-------|---------|
| ~1.0 | Random (no structure) |
| 0.5-0.9 | Early convergence |
| 0.2-0.5 | Recognizable structure |
| 0.05-0.2 | Sharp reconstruction |
| <0.03 | Fully converged |

**Strengths**: Linear projection of actual embedding — faithful to learned structure. Single number, easy to track.

**Weaknesses**: Hides non-planar structure. High disparity can coexist with excellent local neighborhoods if the global layout wraps or folds in high-D. Always pair with K-neighbor metrics.

### K=10 neighbors within N pixels

For each neuron, find K=10 closest neighbors in embedding space. What fraction are within N grid pixels (Manhattan distance)?

| Range (<5px) | Meaning |
|-------------|---------|
| <5% | No spatial structure |
| 50-70% | Moderate, early convergence |
| 90-95% | Good local structure |
| 95-98% | Baseline convergence |
| >98% | Excellent |

Thresholds: `<3px` is stricter (local precision), `<5px` is the standard (local neighborhood).

**Strengths**: Measures what matters — do nearby embeddings correspond to nearby pixels? Robust to global distortions. Works for any embedding dimensionality.

**Weaknesses**: Doesn't capture global layout. A model with perfect local neighborhoods but flipped quadrants scores well here but poorly on Procrustes.

### Mean K=10 neighbor distance

Average grid distance (Manhattan) from each neuron to its K=10 embedding neighbors.

| Range | Meaning |
|-------|---------|
| 1.5-2.5 | Tight neighborhoods, converged |
| 3-5 | Loose but structured |
| 10+ | Poorly sorted |
| ~53 (80x80) | Random |

## Signal quality metrics

### Hit ratio

Per anchor: `good_neighbors / k_sample`. What fraction of random candidates pass the correlation threshold?

| Range | Meaning |
|-------|---------|
| 0% (dead anchor) | No neighbors found, wasted tick |
| 1-5% | Tight threshold, high precision |
| 5-15% | Normal operating range |
| >50% | Threshold too low or global signal |

### Dead anchor rate

Fraction of anchors finding zero neighbors per tick.

| Range | Meaning |
|-------|---------|
| 10-15% | Healthy — matches 80x80 baseline |
| 30-50% | k_sample probably too low for grid size |
| >60% | Broken — most ticks wasted |

**Scaling diagnostic**: If dead anchor rate jumps when increasing grid size, k_sample needs to scale proportionally.

### Discrimination ratio

How well does the scoring metric separate nearby pairs from distant pairs?

- MSE: near (dist<=3) = 0.006, far (dist>=30) = 0.039, ratio **6.3x**
- Deriv-corr: near = 0.820, far = 0.042, ratio **19.3x**

Higher = cleaner signal. Derivative correlation gives ~3x better discrimination than raw MSE.

## Multi-channel metrics

### Channel self-neighbor fraction

For each channel, what fraction of K=10 embedding neighbors are same-channel?

| Range | Meaning |
|-------|---------|
| ~33% (3ch) | No channel separation (random) |
| 50-80% | Partial separation |
| 95-100% | Total channel isolation |

### Within-channel spatial quality

Same as K=10 <Npx, but computed per-channel using only same-channel neurons. Measures spatial sorting quality within each channel cluster independently.

### Cross-channel pixel proximity

For cross-channel neighbor pairs, what is the pixel distance? If embeddings encode pixel position across channels, cross-channel neighbors should be at nearby pixels.

- `same_pixel = 0%` + `mean_dist ~ random` → channels are isolated spatial maps
- `same_pixel > 5%` + `mean_dist << random` → cross-channel pixel structure exists

**Finding (ts-00013)**: Cross-channel pixel proximity is consistently near-zero. The solver never discovers co-located cross-channel neurons.

## Convergence benchmarks

### Grayscale 80x80 (n=6400, k_sample=200, D=8)

| Ticks | PCA disp | <5px |
|-------|----------|------|
| 1k | 0.67 | 73% |
| 5k | 0.25 | 95% |
| 50k | 0.18 | 98% |

### Grayscale 160x160 (n=25600, k_sample=800, D=8)

| Ticks | PCA disp | <5px |
|-------|----------|------|
| 50k | 0.40 | 97% |
| 500k | ~0.20 | 98%+ |

### RGB 80x80 garden.png (n=19200, k_sample=600)

| Ticks | Dims | R <5px | G <5px | B <5px |
|-------|------|--------|--------|--------|
| 50k | 8 | 0.9% | 1.6% | 1.1% |
| 50k | 16 | 1.0% | 1.1% | 1.0% |
| 500k | 8 | 23.2% | 93.2% | 97.1% |
| 500k | 16 | 51.0% | 99.2% | 100% |

## How to run eval

Eval is built into `main.py` via `--eval` flag. Reports PCA disparity and K=10 neighbor stats at the end of training.

For post-hoc channel analysis:
```bash
python analyze_channels.py model.npy -W 80 -H 80 -C 3
```

For post-hoc rendering:
```bash
python render_embeddings.py model.npy -i K_80_g.png -m umap --align
```

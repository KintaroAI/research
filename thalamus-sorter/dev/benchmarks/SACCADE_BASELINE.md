# Saccade Baseline Benchmark

Fundamental test: spatial sorting from image saccade crops.
If this fails, nothing else works.

## Preset

`--preset saccade_baseline`

## Parameters

### Embedding
| Parameter | Value |
|-----------|-------|
| --lr | 0.001 |
| --dims | 8 |
| --k-sample | 200 |
| --k-neg | 5 |
| --threshold | 0.5 |
| --max-hit-ratio | 0.1 |
| --signal-T | 1000 |
| --use-deriv-corr | true |
| --batch-size | 256 |

### Signal
| Parameter | Value |
|-----------|-------|
| --width | 80 |
| --height | 80 |
| --image | K_80_g.png |
| --signal-source | saccades_gray.npy |
| --saccade-step | 50 |

### Clustering
| Parameter | Value |
|-----------|-------|
| --cluster-m | 100 |
| --cluster-max-k | 2 |
| --cluster-k2 | 16 |
| --cluster-split-every | 10 |
| --cluster-knn2-mode | incremental |
| --cluster-centroid-mode | nudge |

### Columns
| Parameter | Value |
|-----------|-------|
| --column-outputs | 4 |
| --column-feedback | true |
| --column-lateral | false |
| --column-lr | 0.05 |
| --column-temperature | 0.2 |
| --column-window | 10 |
| --column-streaming-decay | 0.8 |
| --column-max-inputs | 20 |
| --eligibility | false |

### Global constants (column_manager.py)
| Constant | Value |
|----------|-------|
| COLUMN_MODE | kmeans |
| CONFIDENCE_GATING | False |
| TIREDNESS_RATE | 0.0 |
| ENTROPY_SCALED_LR | True |
| LATERAL_LEARN_MODE | covariance |

### Derived
| Value | Formula |
|-------|---------|
| n_sensory | 6400 (80×80) |
| K (feedback) | 400 (100×4) |
| n_total | 6800 |

## Baseline Results (10k ticks)

| Metric | Value |
|--------|-------|
| Contiguity | 1.000 |
| Diameter | 13.2 |
| Alive clusters | 88/100 |
| K10 within 3px | 96.4% |
| K10 within 5px | 99.9% |
| K10 mean dist | 1.90 |
| Stability | 1.000 (at final tick) |
| Total pairs | 192M |
| Time | 101s |

### Convergence over time

| Tick | Contiguity | Diameter | Stability | K10 <3px |
|------|-----------|----------|-----------|----------|
| 5k | 0.975 | 16.6 | 0.00 | — |
| 10k | 1.000 | 13.2 | 0.47 | 96.4% |

## How to run

```bash
python main.py word2vec --preset saccade_baseline -f 10000 -o output_dir
```

## What it tests

1. **Derivative-correlation** discovers that neighboring pixels change together during saccades
2. **Skip-gram embedding** places correlated neurons nearby in 8D space
3. **Streaming k-means** groups nearby embeddings into spatially coherent clusters
4. **Column feedback** creates V2+ hierarchy (feedback neurons cluster separately)

Failure modes:
- Low contiguity: clusters not spatially coherent (embedding or threshold issue)
- High diameter: clusters too spread out (not enough training or too many clusters)
- Low K10: nearest embedding neighbors not spatial neighbors (embedding quality)

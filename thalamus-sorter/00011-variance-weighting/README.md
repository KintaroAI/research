# ts-00011: Variance Weighting for MSE-Based Neighbor Scoring

**Date:** 2026-03-12
**Status:** Complete
**Source:** `exp/ts-00011`

## Goal

Add a variance component to MSE-based neighbor scoring that distinguishes "both dead" from "both active and co-varying" without compressing the 15x near/far MSE discrimination ratio.

## Motivation

From ts-00010: pure MSE thresholding discovers spatial structure from raw firing rates with no global operations. But MSE has a blind spot:

```
A = [0, 0, 0, 0]    B = [0, 0, 0, 0]    → MSE = 0  (both dead)
A = [0.8, 0.2, 0.5]  B = [0.8, 0.2, 0.5]  → MSE = 0  (genuinely co-varying)
```

Both get MSE=0 and are treated as perfect neighbors. In a cross-modal system where some neurons may be inactive for long stretches, dead pairs would pollute the neighbor graph.

### Why the previous attempt failed

The combined score `sqrt(var_A × var_B) × (1 - MSE)` compressed the discriminating MSE signal. With natural images:
- `(1 - MSE)` ranges 0.95-1.00 (only 5% spread)
- variance multiplier (≈0.05 for all pixels) squashes everything into 0.02-0.04
- The 15x near/far ratio in raw MSE is destroyed

## Approach candidates (score-based)

### 1. Two-stage: MSE gates, variance weights

Keep `MSE < threshold` as the binary neighbor decision (preserves full 15x discrimination). Then weight the *learning update* by `min(var_A, var_B)`.

```
neighbor = MSE(A, B) < threshold
weight   = min(var_A, var_B)
gradient *= weight
```

Dead pairs pass the gate but produce near-zero gradient — they don't learn. Active pairs get full learning signal. The selection criterion is never compressed.

**Pros:** Clean separation of concerns. MSE ratio untouched.
**Cons:** Dead pairs still appear in the neighbor count, potentially affecting hit-ratio diagnostics.

### 2. Additive penalty

```
score = MSE + λ / (var_A + var_B + ε)
neighbor = score < threshold
```

Low-variance pairs get a penalty that pushes their score above threshold. This *shifts* MSE rather than compressing it. The 15x ratio between near (0.003) and far (0.05) is preserved — dead pairs just get bumped up past threshold.

**Pros:** Single score, clean threshold semantics.
**Cons:** Introduces hyperparameter λ. Need to calibrate so the penalty exceeds threshold for dead pairs but is negligible for active ones.

### 3. Hard gate on joint variance

```
neighbor = MSE(A, B) < threshold  AND  (var_A + var_B) > min_var
```

Simple binary filter. With natural images it's a no-op (all pixels have similar variance, which is fine). In cross-modal settings with dead neurons, it catches them.

**Pros:** Simplest. No new hyperparameters if min_var is set conservatively.
**Cons:** Binary — no soft transition. Borderline-variance neurons are either fully in or fully out.

## Approach 4: Derivative correlation (`--use-deriv-corr`)

### Key insight: derivatives encode both activity and similarity

Instead of computing variance separately, work with **temporal derivatives** of the signal. Given signal A = [a_1, a_2, ..., a_T], the derivative is dA = [a_2 - a_1, a_3 - a_2, ...].

**Derivatives as activity proxy:** A dead neuron (constant signal) has dA = [0, 0, 0, ...]. An active neuron has nonzero derivatives. `mean(dA²)` is a natural variance measure — sum of squared changes — purely local, no global info.

**Derivatives as similarity:** If two neurons spike up at the same times and down at the same times, their derivatives align. The dot product `mean(dA * dB)` measures this temporal co-variation.

### Why multiply instead of subtract?

MSE on derivatives would compute `mean((dA - dB)²)` — but dead-dead pairs still get 0. The product `dA[i] * dB[i]` is different:

| Case | dA | dB | Products | mean(dA*dB) |
|------|----|----|----------|-------------|
| Both dead | [0, 0, 0] | [0, 0, 0] | [0, 0, 0] | **0** |
| Dead + active | [0, 0, 0] | [-0.5, 0.3, 0.4] | [0, 0, 0] | **0** |
| Co-varying (near) | [-0.6, 0.3, 0.4] | [-0.5, 0.25, 0.35] | [0.30, 0.075, 0.14] | **+0.17** |
| Uncorrelated (far) | [-0.6, 0.3, 0.4] | [0.1, 0.4, -0.2] | [-0.06, 0.12, -0.08] | **~0** |
| Anti-correlated | [-0.6, 0.3, 0.4] | [0.4, -0.3, -0.5] | [-0.24, -0.09, -0.20] | **-0.18** |

The product naturally combines activity and similarity into one operation:
- Dead neurons produce zero regardless of partner → **no separate variance gate needed**
- Co-varying pairs produce large positive values
- Uncorrelated pairs cancel out to ~0
- Score magnitude scales with both neurons' activity levels

### Normalization → Pearson correlation on derivatives

`mean(dA * dB)` is the unnormalized dot product (covariance of derivatives). Normalizing by the norms gives Pearson correlation on derivatives — bounded [-1, 1]:

```
dA = diff(A),  dB = diff(B)
dA_centered = dA - mean(dA),  dB_centered = dB - mean(dB)
score = dot(dA_centered, dB_centered) / (norm(dA_centered) * norm(dB_centered))
```

Dead neurons: norm = 0, clamped to ε → score = 0. Active co-varying pairs: score → 1. Threshold semantics: `score > threshold` (higher = more similar).

### Properties

- **No global operations.** Each neuron only needs its own temporal trace.
- **Dead-dead = 0.** Solved intrinsically by the product — no separate gate.
- **Biologically plausible.** Neurons respond to changes (derivatives), not absolute levels. Comparing derivative patterns is a natural Hebbian operation.
- **Works on raw signals.** No per-frame mean subtraction needed. Derivatives remove DC offset.

## Results

Base parameters: 80x80 grid, dims=8, k_neg=5, lr=0.001, normalize_every=100, k_sample=200, signal_T=1000, step=50, rolling buffer.

### 10k tick comparison: MSE vs derivative correlation

| Method | Threshold | Total pairs | PCA disp | Mean dist | <3px | <5px |
|--------|-----------|-------------|----------|-----------|------|------|
| MSE | 0.02 | 120M | 0.60 | 2.58 | 80.3% | 94.5% |
| deriv-corr | 0.3 | 795M | 0.56 | 3.29 | 63.4% | 87.4% |
| **deriv-corr** | **0.5** | **222M** | **0.55** | **2.37** | **84.0%** | **97.2%** |

At threshold=0.5, derivative correlation beats MSE at 10k ticks — 97.2% vs 94.5% within 5px. Threshold=0.3 is too loose (795M pairs, too many false neighbors dilute learning). Threshold=0.5 gives a cleaner signal with fewer but higher-quality pairs.

### 50k tick comparison: MSE vs derivative correlation

| Method | Threshold | Ticks | PCA disp | Mean dist | <3px | <5px |
|--------|-----------|-------|----------|-----------|------|------|
| MSE | 0.02 | 50k | 0.27 | 1.94 | 81.5% | 97.1% |
| deriv-corr | 0.5 | 50k | 0.57 | 2.50 | 80.5% | 96.4% |

Very close K-neighbor quality (96.4% vs 97.1%). PCA disparity is higher for deriv-corr but PCA is noisy across runs (different walk paths). The key advantage: deriv-corr handles dead neurons intrinsically.

### Longer runs

| Method | Threshold | Ticks | PCA disp | Mean dist | <3px | <5px |
|--------|-----------|-------|----------|-----------|------|------|
| deriv-corr | 0.5 | 50k | 0.57 | 2.50 | 80.5% | 96.4% |
| deriv-corr | 0.5 | 100k | 0.57 | 2.65 | 76.1% | 94.7% |
| deriv-corr | 0.5 | 1M | 0.57 | 2.55 | 78.8% | 95.7% |

**Stable but plateaued.** PCA disparity stays at 0.57 across all runs. K-neighbor metrics hover 95-96% within 5px — no degradation (unlike T=200 MSE) but no improvement past 50k either.

### Comparison: deriv-corr vs MSE (T=1000)

| Method | Ticks | PCA disp | Mean dist | <3px | <5px |
|--------|-------|----------|-----------|------|------|
| MSE | 50k | 0.27 | 1.94 | 81.5% | 97.1% |
| MSE | 9M | 0.17 | 2.00 | 80.4% | 97.4% |
| deriv-corr | 50k | 0.57 | 2.50 | 80.5% | 96.4% |
| deriv-corr | 1M | 0.57 | 2.55 | 78.8% | 95.7% |

Deriv-corr is ~1-2% below MSE in K-neighbor quality and has worse PCA disparity (0.57 vs 0.17-0.27). MSE continues to improve slowly over millions of ticks; deriv-corr plateaus early. However, deriv-corr handles dead neurons intrinsically — no separate variance gate needed. For cross-modal systems where some neurons are inactive, this property is essential.

### Head-to-head: same walk, same anchors, same candidates

Diagnostic script (`compare_mse_dc.py`) runs both methods on identical signal buffers and anchor/candidate sets.

#### Neighbor overlap

| Category | Count | % | Notes |
|----------|-------|---|-------|
| Both agree (neighbor) | 83,877 | 3.3% | Core consensus |
| MSE only | 40,031 | 1.6% | MSE false positives — DC rejects these |
| DC only | 473 | 0.02% | Almost zero exclusive DC picks |
| Neither | 2,435,619 | 95.1% | Both reject |

**99.4% of deriv-corr neighbors are also MSE neighbors.** DC is a strict subset of MSE — it finds the same good neighbors but filters out MSE's noise. MSE has 40k extra picks that DC correctly rejects (mean grid dist 12.7, 0% within 5px).

#### Discrimination ratio

| Method | Near (dist≤3) mean | Far (dist≥30) mean | Ratio |
|--------|--------------------|--------------------|-------|
| MSE | 0.006 | 0.039 | 6.3x |
| deriv-corr | 0.820 | 0.042 | **19.3x** |

Deriv-corr separates near from far 3x better than MSE.

#### Precision vs true grid neighbors

For each anchor, what fraction of selected neighbors are actually among the K nearest on the real grid?

| True K | MSE precision | DC precision | Both-agree precision |
|--------|---------------|--------------|----------------------|
| 5 | 1.2% | 1.7% | 1.7% |
| 10 | 3.2% | 4.6% | 4.6% |
| 20 | 7.0% | **10.2%** | **10.2%** |

DC precision is ~1.5x better than MSE across all K values. The "both agree" set matches DC precision exactly — the consensus adds nothing beyond what DC already selects.

#### Recall

Of the true K=10 nearest that appeared in the random candidate pool, both methods find **100%**. Neither misses a true neighbor when it's available — the difference is entirely in false positive rate.

#### Interpretation

Deriv-corr doesn't find different neighbors — it finds the same good ones with fewer false positives. The 1-2% lower K-neighbor quality in the embedding runs (96.4% vs 97.1%) may come from having fewer total training pairs rather than worse pair quality. The pairs DC provides are higher precision, but there are fewer of them.

## Conclusions

1. **Derivative correlation solves the variance problem.** `mean(dA * dB)` naturally produces zero for dead neurons — no separate variance gate needed. The product encodes both activity and similarity in one operation.

2. **DC is a strict subset of MSE.** 99.4% of DC neighbors are also MSE neighbors. DC doesn't find different structure — it filters out MSE's false positives (40k noise pairs rejected, mean grid distance 12.7).

3. **DC has 3x better discrimination.** Near/far ratio 19.3x vs MSE's 6.3x. Higher precision at all K values (10.2% vs 7.0% at K=20).

4. **Both methods have 100% recall.** When a true grid neighbor appears in the candidate pool, both always find it. The difference is purely in false positive rate.

5. **DC plateaus slightly below MSE in embedding quality.** 95-96% vs 97% within 5px. Likely because fewer total training pairs — DC's stricter filtering means less data per tick. The pairs are better, but there are fewer of them.

6. **Recommendation:** Use `--use-deriv-corr` for cross-modal systems where dead neurons are expected. Use `--use-mse` for single-modality runs where all neurons are active and maximum embedding quality matters. Both are viable; DC is the more principled choice for biological plausibility.

## Files

- `main.py` — `--use-deriv-corr` flag, raw signal mode for derivative methods
- `solvers/drift_torch.py` — `use_deriv_corr` in `tick_correlation()`

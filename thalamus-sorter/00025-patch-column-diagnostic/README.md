# ts-00025: Patch Column Diagnostic

**Date:** 2026-03-26
**Status:** Complete
**Source:** `exp/ts-00025`
**Depends on:** ts-00024 (modularized building blocks), ts-00023 (hierarchy)

## Goal

Understand how the model's clustering organizes column outputs by hardcoding the first layer. Two questions:

1. Do the 4 outputs from the same patch land in the same model cluster? (bad — means model groups anti-correlated signals)
2. Are the 4 column outputs actually different? (measurable via entropy and correlation)

## Architecture

```
Source image (saccades_gray.npy)
  ↓ saccade walk (step=35, 56×56 crops for 7×7 patches)
  ↓
First layer (HARDCODED, inside benchmark):
  56×56 crop → 64 non-overlapping 7×7 patches (8×8 grid, uses 56×56)
  Each patch → 1 ColumnManager column (4 outputs, k-means mode)
  Total: 64 × 4 = 256 outputs
  ↓
Model (NORMAL pipeline):
  256 "neurons" on 16×16 grid → DriftSolver → ClusterManager (16 clusters)
  ↓
Analysis:
  For each patch, check which model clusters its 4 outputs landed in
```

## Benchmark

Created `benchmarks/patch_column.py` with 4 metrics:

1. **Spread distribution** — for each patch, count unique model clusters among its 4 outputs. Histogram of {all-4-same, 2 clusters, 3 clusters, all-different}.
2. **Output entropy** — per-patch Shannon entropy of the 4 softmax outputs. High = differentiated, low = collapsed.
3. **Inter-output correlation** — per-patch mean pairwise Pearson correlation of the 4 output time series. High = column not differentiating.
4. **Winner distribution** — per-output global win rate and per-patch dominant winner fraction. Detects winner-take-all collapse.

Preset: `presets/patch_column_baseline.json` (16×16 grid, k_sample=50, cluster_m=16, no column_outputs/column_feedback).

```bash
python main.py word2vec --preset patch_column_baseline -f 10000
```

## Results

### Patch size comparison (10k ticks)

| Metric | 3×3 (9 in) | 5×5 (25 in) | 7×7 (49 in) | 9×9 (81 in) |
|---|---|---|---|---|
| Entropy (normalized) | 0.93 | 0.69 | 0.09 | 0.003 |
| Mean spread | — | 3.05 | 2.89 | 2.45 |
| Inter-output corr | — | -0.07 | -0.02 | -0.05 |
| Dominant winner % | — | — | 99.2% | — |

### Spread distribution at 7×7 (10k ticks)

```
all-4-same:    0/64
2 clusters:   15/64
3 clusters:   47/64
all-different:  2/64
Mean spread: 2.80 (1.0=collapsed, 4.0=differentiated)
```

### Winner distribution at 7×7 (10k ticks, 500-tick post-training sample)

```
Global:  28.1% / 20.3% / 34.3% / 17.3%
Per-patch dominant winner: mean=0.992
Output 0 dominates: 18/64 patches
Output 1 dominates: 13/64 patches
Output 2 dominates: 22/64 patches
Output 3 dominates: 11/64 patches
```

### Patch sweep: collapse vs patch size (10k train, last 500 eval)

Created `benchmarks/patch_sweep.py` — standalone sweep script that trains columns for N ticks then evaluates winner distribution over the last 500 ticks only (no cumulative dilution from warmup).

```
patch inputs  grid patches dom_mean     dom_range   >90%   >75% H_norm
---------------------------------------------------------------------------
  2x2      4  40x40    1600    0.510  [0.446,0.566]      0      0  0.806
  3x3      9  26x26     676    0.582  [0.502,0.638]      0      0  0.770
  4x4     16  20x20     400    0.636  [0.548,0.696]      0      0  0.717
  5x5     25  16x16     256    0.673  [0.586,0.748]      0      0  0.677
  6x6     36  13x13     169    0.701  [0.620,0.780]      0      7  0.641
  7x7     49  11x11     121    0.723  [0.646,0.814]      0     22  0.605
  8x8     64  10x10     100    0.739  [0.664,0.830]      0     33  0.586
  9x9     81   8x8       64    0.762  [0.712,0.844]      0     42  0.552
 10x10   100   8x8       64    0.781  [0.732,0.860]      0     56  0.528
```

```bash
python benchmarks/patch_sweep.py                        # defaults: 10k train, 500 eval
python benchmarks/patch_sweep.py --ticks 20000          # longer training
python benchmarks/patch_sweep.py --patches 5,7,9        # subset
```

Key observation: smooth monotonic trend — every extra input pushes columns further toward collapse. At 9×9, 66% of patches have >75% single-winner dominance. No patch size is immune; it's a matter of speed.

Note: earlier runs that averaged over all ticks (including warmup) showed artificially low dominance — 20k even looked *better* than 10k because the early uniform period diluted the average. The eval-window approach captures the true post-training state.

## Findings

### 1. Columns collapse to a single winner

After training, each column's dominant output wins ~76% of ticks (9×9 patches, 500-tick eval window). With the full model pipeline (patch_column benchmark, 500-tick post-training sample), collapse reaches ~99%. This is winner-take-all collapse driven by softmax competitive dynamics with Hebbian learning.

### 2. More inputs → sharper collapse

Smooth monotonic: 2×2 (51% dominant) → 10×10 (78% dominant) at 10k ticks. No threshold or phase transition — just a steady gradient. Columns with more inputs find stronger patterns, which amplifies the winner-take-all feedback loop.

### 3. The embedding algorithm groups silent outputs

The skip-gram DriftSolver measures temporal correlation. Three silent outputs (near-zero simultaneously) have trivially high correlation — they "fire together" by being quiet together. The model embeds them as neighbors and clusters them together. This is why spread is 2.8 not 4.0.

### 4. Sharper columns → worse model separation

Paradoxically, better-differentiated columns lead to worse model spread:
- 5×5 (moderate sharpness): spread 3.05
- 9×9 (fully collapsed): spread 2.45

Because when one output dominates, the 3 losers are even more uniformly silent, making them even more indistinguishable to the embedding algorithm.

### 5. Global winner imbalance

Outputs 0 and 2 win disproportionately (28-34%) vs outputs 1 and 3 (17-21%). This likely stems from the random initialization of k-means centroids — some centroids start closer to common input patterns.

## Root Causes

1. **Column collapse**: softmax + Hebbian learning creates positive feedback loop. Winner gets better → wins more → gets even better. Entropy-scaled lr is on but insufficient.

2. **Embedding blindness to silence**: DriftSolver correlation treats "both near-zero" as "both similar." It can't distinguish shared silence from shared signal.

## ConscienceColumn: the fix

Implemented `ConscienceColumn` (inherits from new `ColumnBase`) based on conscience competitive learning. Key differences from default `ColumnManager`:

- **Hard WTA** with adaptive homeostatic threshold: `theta_k += alpha * (y_k - 1/n_outputs)`. Winners get penalized, losers get helped. Pushes each output toward winning exactly 1/n_outputs of the time.
- **Input normalization**: mean-subtract + L2-normalize. Columns detect shape/pattern, not brightness.
- **Dead-unit reseeding**: if an output hasn't won in `reseed_after` ticks, reinitialize its prototype from current input.
- **Output**: softmax of raw similarities (without theta) for pipeline compatibility. Per-tick output is still peaked, but the winner rotates.

### Architecture refactor

```
ColumnBase          — wire/unwire, slot_map, get_outputs, _gather_input
├── ColumnManager   — softmax WTA, kmeans/variance, lateral, eligibility, tiredness
└── ConscienceColumn — hard WTA, conscience threshold, input normalization, reseeding
```

Selection via `column_config['type']`: `'default'` or `'conscience'`.

### Patch sweep: conscience vs default (10k train, last 500 eval)

```
                  DEFAULT                          CONSCIENCE
patch  dom_mean  >75%  H_norm     dom_mean  >75%  H_norm
  5x5    0.673     0   0.677        0.300     0   0.990
  7x7    0.723    22   0.605        0.302     0   0.989
  9x9    0.762    42   0.552        0.302     0   0.990
```

Conscience columns hit ~30% dominance (near-perfect 25% uniformity) regardless of patch size. Zero collapse.

### Full model pipeline: default vs conscience (7×7, 10k ticks)

| Metric | Default | Conscience |
|---|---|---|
| Spread (of 4.0) | 2.77 | **3.69** |
| Entropy (normalized) | 0.094 | **0.424** |
| Inter-output correlation | -0.024 | **-0.321** |
| Global win % | 28/20/34/17 | **26/25/25/24** |
| Dominant winner % | 99.2% | **37.5%** |
| Model contiguity | 0.635 | 0.283 |
| Total skip-gram pairs | 15M | 0.9M |

Conscience fixes column collapse and the model separates outputs better (spread 3.69 vs 2.77). But the model's contiguity dropped to 0.28 and skip-gram pairs are 15× lower — the rotating winner signal has more temporal variation, making it harder for the embedding algorithm to find stable correlations. May need `threshold` or `k_sample` tuning.

### Typical conscience output (single column, 20 ticks post-training)

```
tick  out0   out1   out2   out3   winner
   0  0.034  0.192  0.587  0.187   [2]
   4  0.179  0.148  0.309  0.364   [3]
   7  0.554  0.131  0.067  0.249   [0]
  18  0.274  0.543  0.095  0.088   [1]
```

Per-tick output is peaked (winner ~0.5-0.7), but winner rotates across ticks. All 4 outputs take turns winning.

### Temporal stability: replay drift test

Created `benchmarks/patch_drift.py` — replays the exact same saccade sequence to measure true prototype drift (vs input variation). Method:

1. Train 10k ticks (first 5k warmup, last 5k recorded)
2. Replay those 5k saccade positions while columns keep learning
3. Compare per-tick winners: same input → same output = stable

```bash
python benchmarks/patch_drift.py --column-type conscience
python benchmarks/patch_drift.py --column-type default
```

Results (7×7 patches, 10k train, 5k replay):

| Metric | Default | Conscience |
|---|---|---|
| Same-input match rate | 65.7% | 63.4% |
| Stable patches (>90% match) | 0/121 | 12/121 |
| Drifting patches (<50% match) | 10/121 | 39/121 |
| Per-output match range | 61-72% | 60-67% |

Both types show ~35% drift over 5k continued learning ticks — prototypes haven't converged, they're still moving. This is not conscience-specific; it's that columns are still actively learning. Conscience has more uniformly distributed drift (all outputs ~63%), while default has output 0 more stable (72%) due to collapse bias.

To address later: learning rate decay, prototype freezing after convergence, or stability-gated learning.

### Performance: vectorized streaming update

Vectorized `streaming_update_v3_gpu` in `cluster_experiments.py` — batch distance computation for all anchors in one `(n_anchors, 1+k2, dims)` operation instead of per-anchor Python loop. Thin apply loop over ~5-20 actual movers handles sequential size mutations.

| | Before | After |
|---|---|---|
| ms/tick (forage 22×8, m=100) | 9.8 | 4.4 |
| Stream update cost | ~7ms (69%) | ~1.5ms |
| 1M run time | ~2.7h | ~1.2h |

Verified identical outputs to scalar reference over 500 ticks. Ref version kept as `streaming_update_v3_gpu_ref`.

### Bug fix: column wiring inconsistency

Fixed stale wiring on in-ring cluster switches. When a neuron switches primary via in-ring swap (both clusters in ring buffer), no wiring event fires. Later LRU eviction of the non-primary slot emits `(neuron, evicted, new)` but the neuron is wired to the old primary, not the evicted cluster — causing stale wiring. Fix: on wiring events, unwire from all ring entries except new primary.

### Column logic benchmarks: pattern classification + XOR chain

Two standalone tests proving columns can learn to compute, not just categorize.

#### Pattern classifier (`benchmarks/column_pattern.py`)

Single column (9 inputs = 3×3 patch, 4 outputs) classifies 4 spatial patterns: vertical line, horizontal line, diagonal, cross. Noisy input, random cycling with hold periods.

```bash
python benchmarks/column_pattern.py --column-type conscience --train 20000 --noise 0.02 --hold 50
```

| Pattern | Default output | Conscience output |
|---|---|---|
| vertical | 3 (99%) | 2 (56%) |
| horizontal | 3 (99%) | 3 (79%) |
| diagonal | 3 (99%) | 1 (70%) |
| cross | 3 (99%) | 0 (75%) |
| **Unique outputs** | **1/4 (collapsed)** | **4/4 PASS** |

Default collapses all patterns to one output. Conscience assigns each to a different output — it actually classifies.

#### XOR chain (`benchmarks/column_xor.py`)

Two-column chain: Column 1 (2 inputs → 4 outputs) creates a representation, Column 2 (4 inputs → 2 or 4 outputs) finds a single output that tracks XOR. XOR is not linearly separable — a single column can't solve it, but a chain can.

```bash
python benchmarks/column_xor.py --column-type conscience --train 20000 --noise 0.02 --hold 50
```

Results with 2 outputs in Column 2 (4→2 chain):

| | Conscience | Default |
|---|---|---|
| XOR separation | **0.260** | 0.001 |
| XOR accuracy | **100%** | 100% (noise) |
| AND accuracy | 75% | 75% |
| OR accuracy | 75% | 75% |

Conscience chain output means — clean XOR computation:
```
(0,0): out0=0.255  out1=0.745   XOR=0
(0,1): out0=0.000  out1=1.000   XOR=1
(1,0): out0=0.000  out1=1.000   XOR=1
(1,1): out0=0.265  out1=0.735   XOR=0
```

Output 1 encodes XOR: saturates to 1.0 when XOR=1, drops to 0.74 when XOR=0. Default chain shows 0.001 separation (all outputs ~0.5 — collapsed, no real signal).

Key insight: conscience is necessary for both tasks. Default columns collapse, destroying the information columns are supposed to extract. The tradeoff from earlier (conscience drifts more) is worth it — without conscience, columns can't compute at all.

## Next Steps

- Address prototype drift: learning rate decay or stability-gated learning
- Tune embedding parameters (`threshold`, `k_sample`) for conscience signal characteristics
- Explore whether conscience columns produce meaningful visual prototypes (edge detectors, etc.)
- Scale XOR chain test: deeper chains (3+ columns), more complex functions

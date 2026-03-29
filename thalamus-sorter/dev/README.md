# Thalamus Sorter — dev

Consolidated codebase for topographic map formation experiments. All algorithms from the parent directories are unified here with shared utilities and a single CLI entry point.

## Setup

```bash
make setup          # creates venv, installs deps
source venv/bin/activate   # activate before running any python commands
```

## Current baseline

Derivative-correlation sorting with rolling saccade buffer on a real image:

```bash
# Using presets (recommended):
python main.py word2vec --preset gray_80x80_saccades -f 50000 -o output_run
python main.py word2vec --preset rgb_80x80_garden -f 50000 -o output_run

# Equivalent explicit flags:
python main.py word2vec --mode correlation \
  -W 80 -H 80 --dims 8 --k-neg 5 --lr 0.001 \
  --k-sample 200 --threshold 0.5 \
  --signal-T 1000 --signal-source saccades_gray.npy \
  --saccade-step 50 --use-deriv-corr \
  --max-hit-ratio 0.1 \
  -i K_80_g.png -f 50000 --save-every 500 \
  --save-model --eval \
  -o output_run --render umap --align
```

Typical 50k results: PCA disparity ~0.2-0.6, K=10 neighbors 95-98% within 5 grid pixels.

**Note for production:** Use `--max-hit-ratio 0.1` in the final system. It filters out anchors that correlate with too many candidates (global signal like flickering lights). No effect with clean signals but essential as a safety net when signal quality varies.

## Performance

Correlation computation uses matmul instead of per-candidate signal gather. Each tick precomputes normalized signals for all n neurons, then `anchor @ normed.T` produces a `(batch, n)` matrix of all pairwise correlations via a single cuBLAS matmul. This is O(n²) — we compute correlations against every neuron even though we only need k_sample of them. But GPU matmul throughput vastly exceeds random memory gather bandwidth, so computing n "unnecessary" scores is faster than fetching k_sample signal vectors from random locations. The old approach allocated `(batch, k_sample, T)` temporaries (3GB+ for 320×320); the matmul path replaces that with a `(batch, n)` output (100MB). **~5x faster** at 80×80.

Add `--fp16` to run the correlation matmul in float16 for additional speedup on large grids:

```bash
python main.py word2vec --preset gray_80x80_saccades -f 50000 --fp16
```

Skip-gram updates and KNN tracking remain in float32.

**TODO: incremental sig_normed.** The full `(n, T-1)` normalized signal matrix is rebuilt every tick, but only one column of the signal buffer changes (the new saccade frame). This affects exactly two derivative entries. Incremental patching of mean/norm would avoid recomputing ~100M elements at 320×320.

**Future: CUDA C kernel.** The matmul trick exists because PyTorch can't express a fused gather-correlate kernel. In CUDA C, a custom kernel could read each anchor signal once, loop over only its k_sample candidates, compute derivative + correlation inline, and write the score — no intermediate `sig_normed` buffer, no wasted O(n²) work. This would be genuinely O(batch × k_sample × T) with ~30x less computation than the current matmul at 320×320 (k_sample=3200 vs n=102400). The matmul is the right solution for PyTorch, but a fused CUDA kernel is the next step if correlation becomes the bottleneck again at larger scales.

## Live clustering, columns & feedback

Streaming k-means clusters neurons in embedding space. Each cluster gets a
SoftWTACell column that learns to differentiate its members' signals. Column
outputs can feed back as input to new "feedback neurons" — closing the loop.

```bash
# Basic clustering (no columns, no feedback)
python main.py word2vec --preset gray_80x80_saccades -f 50000 --cluster-m 100

# Full pipeline with feedback (recommended: use presets, they include these)
python main.py word2vec --preset gray_80x80_saccades -f 50000
```

### Clustering flags

| Flag | Default | Description |
|------|---------|-------------|
| `--cluster-m` | 0 | Number of clusters (0=disabled) |
| `--cluster-neurons-per` | 0 | Auto-compute M: `M = n_sensory / (N - column_outputs)` |
| `--cluster-k2` | 16 | Cluster-level KNN size |
| `--cluster-lr` | 1.0 | Centroid nudge learning rate |
| `--cluster-split-every` | 10 | Dead cluster recovery interval (ticks) |
| `--cluster-hysteresis` | 0.0 | Reassignment resistance (0=none) |
| `--cluster-knn2-mode` | incremental | `incremental` (from pairs) or `knn` (requires --knn-track) |
| `--cluster-centroid-mode` | nudge | `nudge` (lr-based drift) or `exact` (immediate) |
| `--cluster-max-k` | 1 | Ring buffer depth for multi-cluster membership |
| `--cluster-report-every` | 1000 | Save cluster viz + metrics every N ticks |
| `--cluster-render-mode` | color | `color`, `signal`, or `both` |
| `--cluster-track-history` | false | Save per-neuron cluster ID history |
| `--cluster-max-size` | 0 | Max neurons per cluster (0=unlimited). Full clusters swap worst-fit member. |

### Column wiring flags

| Flag | Default | Description |
|------|---------|-------------|
| `--column-outputs` | 0 | Outputs per column (0=disabled, 4=typical) |
| `--column-max-inputs` | 20 | Pre-allocated input slots per column |
| `--column-window` | 10 | Sliding window for streaming variance |
| `--column-lr` | 0.05 | Column Hebbian learning rate |
| `--column-temperature` | 0.2 | Softmax temperature (lower=peakier) |
| `--column-match-threshold` | 0.1 | Dormant reassignment threshold |
| `--column-streaming-decay` | 0.8 | EMA decay (rule of thumb: 1-2/window) |

### Lateral input connections

| Flag | Default | Description |
|------|---------|-------------|
| `--lateral-inputs` | false | Enable permanent column-to-column wiring via feedback neurons |
| `--lateral-input-k` | 4 | Lateral connections sent per column (small-world topology) |

Each column sends `lateral-input-k` outputs to random destination columns,
permanently wired into reserved input slots (bypassing thalamus). The same
feedback neuron signal flows through both the normal thalamus path and the
lateral path. Reserved slots are protected from wire/unwire eviction.
`max_inputs` is expanded by `lateral_input_k * 2` to avoid reducing normal capacity.

### Feedback loop flags

| Flag | Default | Description |
|------|---------|-------------|
| `--column-feedback` | false | Feed column outputs back as feedback neuron signals |
| `--render-mode` | grid | `grid` (default) or `embed` (adds embed scatter plots) |

When `--column-feedback` is enabled, K = M × column_outputs feedback neurons
are added to the signal buffer. Their signals come from the previous tick's
column outputs. Use `--cluster-neurons-per 10` with `--column-outputs 4` to
get M = n_sensory/6 clusters with ~10 total neurons per cluster (6 sensory + 4
feedback).

`--render-mode embed` saves `embed_NNNNNN.png` at each `cluster_report_every`:
PCA scatter of all neurons — sensory (gray dots) and feedback (colored by
column hue). Does not affect normal frame rendering.

### Column learning dynamics

`ENTROPY_SCALED_LR = True` in `column_manager.py`: columns with uniform
outputs (high entropy) learn at full rate; differentiated columns (low entropy)
learn slowly. Prevents early lock-in while allowing gradual re-learning.

### KNN tracking flags

| Flag | Default | Description |
|------|---------|-------------|
| `--knn-track` | 0 | Track per-neuron KNN list of this size (0=off) |
| `--knn-report-every` | 1000 | Report KNN stability/spatial accuracy every N ticks |

## W&B logging

Add `--wandb` to log KNN stability, training summary, and eval metrics to Weights & Biases in real-time:

```bash
python main.py word2vec --preset rgb_80x80_garden -f 50000 --wandb
python main.py word2vec --preset rgb_80x80_garden -f 50000 --wandb --wandb-name "garden-50k" --wandb-project thalamus
python main.py word2vec --preset rgb_80x80_garden -f 50000 --wandb --wandb-tags "rgb,garden" --wandb-group knn-stability
```

Logged metrics:
- `tick/elapsed_s`, `tick/total_pairs`, `tick/ms_per_tick` — every `--log-every` ticks (default 1000)
- `knn/overlap`, `knn/spatial`, `knn/n_changed`, `knn/pct_changed`, `knn/top50_swaps`, `knn/top90_swaps` — per KNN report tick
- `summary/ticks`, `summary/elapsed_s`, `summary/std`, `summary/total_pairs` — at training end
- `eval/pca_disparity`, `eval/k10_mean_dist`, `eval/k10_within_3px`, `eval/k10_within_5px` — if `--eval` is set

Tick progress is also printed to stdout regardless of `--wandb`:

```
  tick 500/3000 (7.9s, 15.9 ms/tick, pairs=7959624)
  tick 1000/3000 (13.6s, 11.3 ms/tick, pairs=15363278)
```

Control the interval with `--log-every N` (default 1000, 0 to disable).

If wandb is not installed, `--wandb` prints a warning and runs without logging.

**TODO: auto-adjust k_sample.** Currently k_sample must be manually scaled with grid size (k_sample=200 for 80x80, 800 for 160x160). Implement adaptive k_sample: track the fraction of anchors that find zero neighbors per tick. If the zero-hit rate exceeds ~15%, double k_sample; if it drops below ~5%, halve it. This keeps the effective sampling fraction constant regardless of grid size. See ts-00012 for data: at 160x160 with k_sample=200, 64.8% of anchors find zero neighbors (dead); k_sample=800 restores the 14.3% baseline.

## Quick start

```bash
# Run tests (headless, no display needed)
make test

# Grid simulation (no camera needed)
make greedy                                   # default 40x40, k=24
make greedy ARGS="--width 80 --height 80 --k 48"
make mst ARGS="--width 20 --height 20"
make sa ARGS="--width 20 --height 20 --temp 200"

# Camera modes (requires webcam + display)
make camera-sa ARGS="--width 32 --height 24"
make camera-spatial ARGS="--width 16 --height 12 --lr 0.001"
```

Press `q` in the OpenCV window to quit any mode.

## Solvers

### `greedy` — Greedy Hebbian drift

Each neuron moves one step toward the centroid of its K highest-affinity neighbors. The most biologically plausible approach: Hebbian attraction with spatial constraints. Iterative — watch the grid self-organize in real time.

| Flag | Default | Description |
|------|---------|-------------|
| `--k` | 24 | Number of neighbors to attract toward |
| `--move-fraction` | 0.9 | Fraction of neurons moved per tick |
| `--weight-type` | decay2d | `decay2d` or `inv1d` |
| `--decay` | 0.1 | Decay rate for decay2d weights |
| `--image` / `-i` | — | Input image to scramble and reconstruct |
| `--frames` / `-f` | 0 | Number of ticks to run (0 = unlimited) |
| `--output-dir` / `-o` | — | Directory to save output frames as PNGs |
| `--save-every` | 1 | Save every Nth frame |

Image mode example — scramble a 40x40 grayscale image and save 200 frames over 10k ticks:

```bash
make greedy ARGS="--image K_40_g.png -W 40 -H 40 --k 24 --frames 10000 --save-every 50 -o output/"
```

### `mst` — Maximum spanning tree

One-shot sorting. Builds a maximum spanning tree on the weight matrix (Kruskal's algorithm), then DFS traversal produces a linear ordering that respects correlation structure. Reshaped into a 2D grid.

### `sa` — Simulated annealing

Random neuron swaps accepted or rejected based on cost change and temperature. Cost = sum of `affinity(i,j) * manhattan_distance(i,j)` over sampled pairs. Temperature decreases by cooling rate each step.

| Flag | Default | Description |
|------|---------|-------------|
| `--temp` | 100 | Initial temperature |
| `--cooling` | 0.99 | Cooling rate per step |
| `--sa-iterations` | 100 | SA steps per display tick |

### `camera-sa` — Online weight learning + MST sort

Uses live camera as a streaming signal source. Captures frames into a temporal buffer, finds nearest neighbors in temporal-signal space (sklearn ball_tree), computes pairwise correlations, and updates a running-average weight matrix. Sorts the learned weights via MST each frame.

| Flag | Default | Description |
|------|---------|-------------|
| `--steps` | 10 | Temporal window size (frames) |

### `camera-spatial` — Total variation gradient descent

Treats sorting as continuous optimization. Maintains a permutation weight matrix and minimizes total variation loss (sum of absolute horizontal + vertical differences in the output grid) via gradient descent. The only differentiable approach.

| Flag | Default | Description |
|------|---------|-------------|
| `--lr` | 0.001 | Learning rate |
| `--epochs` | 1000 | Gradient steps per camera frame |

## Weight matrices

Two built-in weight matrix types define the "ideal" neuron affinity:

- **`decay2d`** (default) — `weight = max(0, 1 - euclidean_distance * decay_rate)` on 2D grid coordinates. Neurons nearby on the grid have high affinity. Controlled by `--decay`.
- **`inv1d`** — `weight = 1/|i-j|` on 1D index distance. Simpler but doesn't capture 2D structure.

The camera modes learn weights online from observed temporal correlations instead of using a pre-defined matrix.

## Code layout

```
dev/
├── main.py              # CLI entry point (5 subcommands)
├── Makefile
├── requirements.txt
├── utils/
│   ├── camera.py        # cv2.VideoCapture wrapper
│   ├── correlation.py   # Pearson correlation (scalar, temporal, matrix)
│   ├── display.py       # cv2 display helpers
│   ├── geometry.py      # coords(), to_index(), move_closer()
│   ├── graph.py         # UnionFind, MST (Kruskal), DFS traversal
│   ├── wandb_logger.py  # W&B integration (--wandb flag)
│   └── weights.py       # Weight matrices + OnlineWeightMatrix
└── solvers/
    ├── greedy_drift.py          # GreedyDrift class
    ├── mst_sort.py              # mst_sort() function
    ├── simulated_annealing.py   # SimulatedAnnealing class
    └── spatial_coherence.py     # SpatialCoherence class
```

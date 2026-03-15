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

## W&B logging

Add `--wandb` to log KNN stability, training summary, and eval metrics to Weights & Biases in real-time:

```bash
python main.py word2vec --preset rgb_80x80_garden -f 50000 --wandb
python main.py word2vec --preset rgb_80x80_garden -f 50000 --wandb --wandb-name "garden-50k" --wandb-project thalamus
python main.py word2vec --preset rgb_80x80_garden -f 50000 --wandb --wandb-tags "rgb,garden" --wandb-group knn-stability
```

Logged metrics:
- `knn/overlap`, `knn/spatial`, `knn/n_changed`, `knn/pct_changed`, `knn/top50_swaps`, `knn/top90_swaps` — per KNN report tick
- `summary/ticks`, `summary/elapsed_s`, `summary/std`, `summary/total_pairs` — at training end
- `eval/pca_disparity`, `eval/k10_mean_dist`, `eval/k10_within_3px`, `eval/k10_within_5px` — if `--eval` is set

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

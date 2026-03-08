# Thalamus Sorter

Prototype implementations of **thalamic topographic map formation** — the first principle from the [Natural Intelligence](https://kintaroai.com/docs/natural-intelligence/) framework.

## What this is

The core question: given a grid of neurons receiving temporal signals, can we rearrange them so that neurons with correlated activity end up spatially close to each other?

This is **topographic map formation** — exactly what the thalamus is theorized to do through activity-dependent self-organization. A live camera feed serves as streaming sensory input. Each pixel is treated as a "neuron" whose temporal signal (brightness over consecutive frames) defines its identity. The system attempts to find an arrangement where co-active neurons are neighbors.

## Connection to the research plan

This work corresponds to an early, literal reading of the Kintaro article's Principle 1: the thalamus sorts signals by temporal correlation into a topographic arrangement. The research plan's Phase 3 (thalamic sorter) has since evolved this into a gated routing mechanism operating on multimodal embeddings, which is more tractable for ML but further from the biological metaphor.

Key differences between this codebase and the Phase 3 thalamic sorter:

| | This codebase | Phase 3 thalamic sorter |
|---|---|---|
| Core operation | Spatial rearrangement of neurons by correlation | Gated routing of multimodal signal vectors |
| What it sorts | Pixel positions on a 2D grid | Modality embeddings (vision, audio, proprioception) |
| Sorting signal | Temporal correlation of brightness traces | Prediction error, top-down context, surprise |
| Output | A topographic map (permutation) | Gated/weighted embeddings + salience score |
| Top-down modulation | None (purely bottom-up) | Context-dependent routing from workspace feedback |

## Common pattern across all approaches

Every implementation shares the same structure:

1. **Signal source.** Camera frames (grayscale pixels) captured over a temporal window (typically 5 frames).

2. **Weight matrix.** Defines "ideal" affinity between neuron pairs — usually `1/|i-j|` (inverse distance) or `1 - decay * distance` (linear decay). This is the target topology: neurons close together should have high affinity.

3. **Correlation matrix.** Computed from temporal signals — Pearson correlation between the brightness traces of neuron pairs. This is the observed structure from data.

4. **Sorting objective.** Rearrange neurons so the observed correlation structure matches the ideal distance-based weight structure. Cost = `sum(correlation(i,j) * distance(i,j))` — high correlation between distant neurons is expensive.

5. **Visualization.** Real-time OpenCV display showing the neuron arrangement converging.

## Directory structure

### `variance/`

Correlation-based analysis with simulated annealing.

- **`variance.py`** — Main implementation. Captures camera frames, computes temporal variance and Pearson correlations, applies simulated annealing to minimize correlation-distance cost. Contains the full pipeline: camera capture, correlation computation, weight matrix creation (`1/|i-j|`), SA with sampled cost function, coordinate-based neuron swapping, and visualization. Also includes an unused simple neural network training function (sigmoid MLP) that appears to be from early experimentation.
- **`variance_threshold.py`** — Variant at smaller resolution (320x240) that filters variance by threshold and computes total variation loss on output.
- **`variance_color.py`** — Extended version returning both grayscale and color frames, creating neon visual effects by merging variance masks with original color image.

### `simulated_annealing/`

Multiple SA variants with increasing sophistication.

- **`simulated_annealing.py`** — Minimal SA: random neuron swaps with temperature-based acceptance, Manhattan distance cost function. Two versions: one with full matrix copy, one with in-place swap/undo.
- **`simulated_annealing_weighted.py`** — Brute-force cost against entire weight matrix. Reference implementation.
- **`simulated_annealing_sampled.py`** — SA with sampled (approximate) cost function for speed.
- **`simulated_annealing_local.py`** — Local neighborhood variant.
- **`nn.py`** — Combines nearest-neighbor search (sklearn, K=8, ball_tree) with SA. Captures temporal input/output from camera, builds dynamic weight matrix from observed correlations, displays both variance and weight matrix in real time. Key addition: the weight matrix is *learned* from streaming data, not pre-specified.
- **`nn_sa_sort.py`**, **`nn_sa_sort_sampled.py`**, **`nn_sa_sort_sampled_weighted.py`** — Variants combining NN lookup with SA sorting at different sampling/weighting strategies.
- **`lsh.py`** — Most complete implementation. Adds MST construction (UnionFind + Kruskal's), greedy sorting with graph traversal, output rebuilding from sorted neurons. Maintains an evolving weight matrix with count tracking (running average over last 10 observations). Compares learned weights against ideal weight matrix. This is the closest to an online self-organizing system.
- **`variance.py`**, **`points.py`** — Helper/placeholder files.

### `greedy/`

Greedy drift algorithms — neurons physically move toward their correlated neighbors.

- **`greedy_drift_neigbors.py`** — **Cleanest implementation.** A `Sorter` class that maintains a neuron grid, computes distance-weighted affinity (`max(0, 1 - decay * distance)`), finds K=24 nearest neighbors by weight, and moves each neuron one step toward the centroid of its neighbors. This is essentially a physical simulation of Hebbian self-organization: correlated units attract.
- **`greedy_drift.py`** — Earlier version using MST-based sorting (UnionFind + DFS traversal) combined with greedy swaps.
- **`greedy_drift_weights.py`** — Weight-based variant of greedy drift.
- **`greedy_drift_cleaned.py`** — Simplified/refactored version.
- **`greedy_swap.py`** — Swap-only variant without drift.
- **`greedy_drift_neigbors_image.py`** — Adds visual output of sorting progress.
- **`greedy_drift_neigbors_all_move.py`** — All neurons move simultaneously per tick.
- **`greedy_drift_neigbors_one_move.py`** — Single neuron moves per tick.
- **`greedy_drift_neigbors_image_gpu.py`** — GPU-accelerated variant.

### `mst/`

Minimum spanning tree approaches.

- **`mst.py`** — Kruskal's MST algorithm on the correlation weight matrix. UnionFind with path compression and union by rank. Builds maximum spanning tree (edges sorted by weight descending), then DFS traversal produces a linear ordering that respects correlation structure. The ordering is reshaped into a 2D grid for display. Clean, standalone implementation.
- **`mst_lsh.py`** — Extended MST with camera integration, LSH-based nearest neighbor lookup, temporal correlation tracking, dynamic weight matrix learning, and real-time visualization comparing learned vs ideal weights.

### `spatial_coherence/`

Gradient-based optimization.

- **`spatial_coherence.py`** — Treats sorting as a differentiable optimization problem. Defines total variation loss (sum of absolute horizontal and vertical differences in the output grid). Backpropagates through TV loss to compute gradients on the permutation weight matrix `W1`. Runs gradient descent for 1000 epochs per camera frame. This is the only approach that uses continuous optimization rather than discrete swaps — conceptually closest to how a differentiable thalamic sorter module would work in a neural network.
- **`init.py`** — Basic initialization helper.

### `demo/`

Archived outputs and simplified demo scripts.

- Simplified versions of greedy drift, LSH, and NN+SA for demonstration
- `join_img_example.py`, `join_img_example_gray.py` — image composition utilities
- `output_*` directories with saved result frames

### `wrong_weights/`

Mirror of the main directory structure containing archived experiments with alternative weight formulations (e.g. using wrong distance metrics). Kept for comparison to show that the weight matrix formulation matters.

## Key technical details

- **Weight matrix `W1`** is typically initialized as a shuffled identity (permutation matrix). "Sorting" means finding the right permutation that aligns observed correlations with spatial proximity.
- **Online weight learning** in `lsh.py` and `nn.py`: running average of observed temporal correlations, capped at a tail window. The system learns topographic structure from streaming data rather than having it pre-specified.
- **Greedy drift** (`greedy_drift_neigbors.py`) is the most biologically plausible: neurons move toward the centroid of their correlated neighbors. This is Hebbian attraction with spatial constraints.

## What this demonstrates

1. **The sorting problem is real and hard.** Multiple optimization strategies were tried (SA, greedy drift, MST, gradient descent), suggesting none trivially solves it at scale.

2. **Greedy Hebbian drift works.** Neurons moving toward their correlated neighbors is the simplest approach that produces coherent topographic maps.

3. **Online weight learning is viable.** Correlation structure can be learned incrementally from streaming data, not pre-computed — supporting the research plan's streaming world requirement.

4. **Total variation loss is a useful regularizer.** Encourages smooth topographic maps and could be applied to learned representations in a differentiable thalamic sorter.

5. **The gap between sorting and routing.** This codebase rearranges a fixed set of neurons (spatial self-organization). The research plan needs dynamic routing — different signals emphasized at different times based on context. These are different operations. Topographic structure may be a prerequisite for meaningful routing, but it is not a substitute for the gating mechanism.

## What's missing (relative to Phase 3)

- **Top-down modulation.** Everything here is purely bottom-up — correlations in the input determine arrangement. The Phase 3 thalamic sorter needs context-dependent routing: same input, different routing based on workspace feedback.
- **Multiple modalities.** Only a single modality (camera grayscale) is used. No cross-modal correlation or fusion.
- **Surprise / salience.** No mechanism for detecting or responding to novel events.
- **Recurrent state.** No hidden state carried across time beyond the temporal correlation window.

## Dependencies

- `numpy` — matrix operations
- `opencv-python` (`cv2`) — camera capture and visualization
- `scikit-learn` — `NearestNeighbors` (used in NN and LSH variants)

## Usage

Most scripts run standalone with a connected camera:

```bash
cd variance && python variance.py
cd greedy && python greedy_drift_neigbors.py
cd mst && python mst.py
cd spatial_coherence && python spatial_coherence.py
```

The greedy drift neighbors script runs without a camera (pure simulation on a random grid). The MST script also runs standalone on a random 10x10 grid.

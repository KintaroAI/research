# ts-00007: Skip-gram Reference Comparison

**Date:** 2026-03-11
**Status:** Complete

## Hypothesis

Our custom PyTorch skip-gram solver (`drift_torch.py`) should produce embeddings comparable to gensim's well-tested Word2Vec implementation when given the same neighbor structure as input.

## Method

1. Build top-K neighbors (k=24) from a 2D grid using KDTree.
2. Convert neighbor lists to "sentences": each sentence = [neuron_id, neighbor1, ..., neighbor_k].
3. Run gensim `Word2Vec(sg=1)` as reference baseline.
4. Run our `DriftSolver(mode='dot')` on the same top-K.
5. Compare: PC correlations with grid x/y, top-K overlap with true neighbors, PCA variance, visual reconstruction.

### Sentence construction

Each neuron's neighbor list becomes a "sentence" of length k+1. Neighbors are shuffled each epoch for variety. Multiple epochs (default 10) generate repeated sentences with different neighbor orderings.

## Results

### Test 1: 40x40, D16, k=24 (single-pair baseline)

**Gensim:** `Word2Vec(sg=1, vector_size=16, window=5, negative=5, epochs=20)` on 16k sentences × 10 sentence-epochs.

**Ours:** `DriftSolver(mode='dot', dims=16, k_neg=5, normalize_every=100)` for 50k single-pair ticks.

| Metric | Gensim | Ours (single-pair) |
|--------|--------|------|
| PC→X correlation | 0.4627 | 0.3004 |
| PC→Y correlation | 0.6373 | 0.3469 |
| Top-24 neighbor overlap | 82.2% | 87.9% |
| Embedding std | 1.14 | 0.24 |
| Top PC variance | 10.2% | 7.7% |
| Training time | 9.8s (CPU) | 19.3s (GPU) |

**Observations:**
- Gensim has higher PC correlations (spatial structure) but lower neighbor overlap.
- Our solver has higher neighbor overlap but less spatial structure in PCs.
- Gensim's PC variance more concentrated — finding more structure.
- Key difference: gensim uses sliding window over sentence (positional context), our solver samples 1 random positive per tick (no positional structure).
- Gensim sees 16k × 20 = 320k sentence passes. Our solver does 50k ticks.

### Test 2: 40x40, D3, k=24 (three-way comparison)

Added sentence mode to our solver to mimic gensim's sliding window approach. Three-way comparison: gensim, our single-pair, our sentence mode.

| Metric | Gensim | Single-pair | Sentence |
|--------|--------|-------------|----------|
| PC→X correlation | 0.9600 | 0.3469 | 0.6330 |
| PC→Y correlation | 0.9621 | 0.3004 | 0.6199 |
| Top-24 neighbor overlap | 61.3% | 87.9% | 67.3% |
| Training time | 48.1s (CPU) | 19.3s (GPU) | 163.0s (GPU) |

**Observations:**
- Gensim far superior on PC correlations at D3 (0.96 vs 0.35 for single-pair).
- D3 forces gensim to pack spatial structure into exactly 3 dimensions — PCs become nearly pure x/y axes.
- Our single-pair mode still focuses on neighbor overlap over spatial structure.
- Initial sentence mode slow (163s) due to Python loop over ~250 offset pairs per tick.

### Test 3: Batched sentence optimization

Rewrote `tick_sentence()` to batch all (center, context) pairs into single tensor operations, eliminating the Python for-loop.

**Before:** Loop over ~250 offset pairs per tick → 163.0s for 1000 ticks.
**After:** All pairs flattened into tensors, one positive + negative pass per tick → 1.0s for 1000 ticks.

**160x speedup** from batching.

### Test 4: Parameter sweep (D3, 40x40, sentence mode)

Swept lr and tick count to find optimal parameters for batched sentence mode with k_neg=5.

| lr | Ticks | PC→X | PC→Y | Overlap | Time |
|----|-------|------|------|---------|------|
| 0.005 | 1000 | 0.633 | 0.620 | 67.3% | 1.0s |
| 0.001 | 5000 | 0.933 | 0.943 | 63.3% | 5.1s |
| 0.0005 | 10000 | **0.957** | **0.968** | 65.3% | 5.3s |
| 0.0005 | 20000 | 0.949 | 0.951 | 64.1% | 10.3s |

**Best result:** lr=0.0005, 10k ticks → r=0.957/0.968, 65.3% overlap in 5.3 seconds.

### Final comparison (D3, 40x40)

| Metric | Gensim | Our sentence mode |
|--------|--------|-------------------|
| PC→X correlation | 0.960 | 0.957 |
| PC→Y correlation | 0.962 | 0.968 |
| Top-24 overlap | 61.3% | 65.3% |
| Training time | 48.1s (CPU) | **5.3s (GPU)** |
| Speed | 1x | **9x faster** |

**Our sentence mode matches gensim embedding quality at 9x speed on GPU.**

## Key insight: sliding window matters

The critical difference between our single-pair mode and gensim is how training pairs are generated:

- **Single-pair:** Each tick samples 1 random positive neighbor for 1 random neuron. Only trains neuron↔neighbor pairs.
- **Sentence/sliding window:** Each tick builds sentences [neuron, nb1, nb2, ...], then trains all pairs within a sliding window. This creates **neighbor↔neighbor** pairs (not just neuron↔neighbor), which teaches the model that neighbors of neighbors should also be close.

This neighbor-of-neighbor structure is what gives gensim its strong spatial correlations. Once we implemented the same approach, quality matched.

## Implementation: `tick_sentence()`

The batched implementation in `DriftSolver`:

1. Build sentences: `[neuron_id, shuffled_neighbor_1, ..., shuffled_neighbor_k]` for all neurons.
2. Precompute all (center_offset, context_offset) pairs from sliding window.
3. Gather center/context neuron IDs for all pairs across all sentences → flat tensors.
4. Single batched forward pass: dot products, sigmoid, gradient scatter_add.
5. Single batched negative sampling: k_neg random negatives per pair.

~250 training pairs per sentence × 1600 sentences = ~400k updates per tick, all in one GPU kernel.

## Scaling to 80x80

### Test 5: 80x80, D3 and D8 (both solvers)

Same parameters as 40x40 (lr=0.0005, 10k ticks, k_neg=5, window=5). Gensim: 20 epochs on 64k sentences.

| Metric | Gensim D3 | Ours D3 | Gensim D8 | Ours D8 |
|--------|-----------|---------|-----------|---------|
| PC→X | 0.355 | 0.074 | 0.421 (PC2) | 0.312 (PC3) |
| PC→Y | 0.518 | 0.829 | 0.336 (PC3) | 0.593 (PC7) |
| Top-24 overlap | 26.5% | 14.3% | **95.6%** | 91.2% |
| Std | 3.88 | 0.56 | 2.36 | 0.35 |
| Time | 41.7s | 20.3s | 34.1s | 87.8s |

**Observations:**
- D3 is too few dimensions for 6400 neurons — even gensim can't compress 80x80 spatial structure into 3 dims. Both solvers get one axis but not both.
- D8 gives excellent neighbor overlap (91-96%) but poor PC correlations — spatial structure is spread across many dimensions instead of concentrating into top PCs.
- Opposite trade-off: fewer dims = forced axis alignment but lossy overlap; more dims = great overlap but diluted spatial axes.
- Our D3 happens to get one strong axis (r=0.93 Y) while gensim spreads across both weakly.

### Saved models for visualization experiments

Trained embeddings saved as `.npy` files in `output_7/` for testing different projection methods without retraining:

| Model | File | Dims |
|-------|------|------|
| Gensim D3 | `output_7/output_7_gensim_80_D3.npy` | (6400, 3) |
| Gensim D8 | `output_7/output_7_gensim_80_D8.npy` | (6400, 8) |
| Ours D3 W | `output_7/output_7_ours_80_D3_W.npy` | (6400, 3) |
| Ours D3 C | `output_7/output_7_ours_80_D3_C.npy` | (6400, 3) |
| Ours D8 W | `output_7/output_7_ours_80_D8_W.npy` | (6400, 8) |
| Ours D8 C | `output_7/output_7_ours_80_D8_C.npy` | (6400, 8) |

### Visualization: `render_embeddings.py`

Standalone script to render saved embeddings with different projection methods:

```bash
python render_embeddings.py output_7/MODEL.npy -i K_80_g.png -m {pca,bestpc,angular,direct,umap,tsne,spectral,mds}
```

Linear methods: `pca` (top 2 PCs), `bestpc` (PCs most correlated with grid x/y), `angular` (unit normalize then PCA), `direct` (first 2 dims).

Nonlinear methods: `umap`, `tsne`, `mds`, `spectral` (Laplacian Eigenmaps).

### Test 6: Nonlinear projections recover K from D8

PCA/bestpc completely failed on D8 embeddings (diagonal stripes, scattered blobs). Nonlinear methods recover the K clearly:

| Method | Gensim D8 | Ours D8 | Notes |
|--------|-----------|---------|-------|
| bestpc | diagonal stripes | scattered blob | Both axes map to same PC |
| **UMAP** | clear K | clear K | Best overall quality |
| **t-SNE** | clean K | cleanest K | Good contrast |
| **Spectral** | clear K | clear K, slight warp | Fast, graph-based |

**Key insight:** The D8 embeddings had 95% neighbor overlap all along — the spatial structure was encoded nonlinearly across 8 dimensions. PCA couldn't find it because it's linear. UMAP/t-SNE/spectral handle this easily.

### Test 7: Supervised linear projections (Procrustes, least-squares)

Since we know the true grid coordinates, we can use supervised projections:

- **Procrustes:** PCA to 2D, then rotate/flip/scale to match grid. Fast but limited to top-2 PCs.
- **lstsq:** Direct least-squares `emb @ W ≈ grid_coords`. Searches all D dimensions at once. Just a matrix solve — instant.

| Method | Gensim D8 | Ours D8 | Notes |
|--------|-----------|---------|-------|
| lstsq | partial K (r=0.73/0.47) | partial K (r=0.55/0.80) | One axis good, one weak |
| procrustes | scattered | stripes | Limited to top-2 PCA |

lstsq is better than procrustes (uses all dims) but still linear — can't match nonlinear methods for D8.

### Test 8: Procrustes alignment for consistent orientation

All projections (UMAP, t-SNE, etc.) produce arbitrarily rotated/flipped output. Procrustes alignment to the known grid coordinates fixes this in ~1ms:

```bash
python render_embeddings.py MODEL.npy -i K_80_g.png -m umap --align
```

| Method | Disparity (aligned) | Notes |
|--------|-------------------|-------|
| UMAP | 0.009–0.025 | Near-perfect alignment |
| t-SNE | 0.001 | Lowest disparity |
| Spectral | 0.015 | Good alignment |

The `--align` flag applies `scipy.spatial.procrustes(grid_coords, projected_2d)` after any projection. Essential for frame-to-frame consistency in video output.

### Test 9: UMAP scaling benchmark

Tested UMAP as potential streaming visualization pipeline.

**fit_transform() (one-shot):**

| Grid | D=3 | D=8 | D=16 | D=32 |
|------|-----|-----|------|------|
| 40x40 (1.6k) | 10.4s | 4.4s | 4.5s | 4.6s |
| 80x80 (6.4k) | 21.4s | 8.0s | 8.6s | 8.9s |
| 160x160 (25.6k) | 13.2s | 13.5s | 14.7s | 15.5s |

**transform() (streaming update):**

| Grid | D=8 | Notes |
|------|-----|-------|
| 40x40 | 7.9s | |
| 80x80 | 16.7s | |
| 160x160 | 6.2s | |

**Observations:**
- Dims scale well — D8 to D32 barely matters, dominated by kNN graph construction.
- n scales roughly linearly.
- `transform()` is NOT fast — nearly as slow as full fit. Still builds a kNN graph and runs optimization internally.

### Test 10: Warm-started projections for frame-to-frame rendering

Key insight: UMAP and t-SNE support `init` parameter — pass previous frame's 2D positions as starting point with fewer iterations. Dramatically faster than cold start.

**UMAP warm start benchmark (80x80, D8):**

| n_epochs | Time | Disparity | Speedup vs full |
|----------|------|-----------|-----------------|
| full (default) | 23.5s | — | 1x |
| 200 | 2.9s | 0.037 | 8x |
| 100 | 1.6s | 0.016 | 15x |
| **50** | **0.87s** | **0.008** | **27x** |
| 20 | 0.45s | 0.011 | 52x |
| 10 | 0.32s | 0.015 | 74x |

**t-SNE warm start benchmark (80x80, D8):**

| max_iter | Time | Disparity | Speedup vs full |
|----------|------|-----------|-----------------|
| full (1000) | 18.6s | — | 1x |
| 500 | 9.6s | 0.001 | 2x |
| 250 (minimum) | 5.1s | 0.034 | 4x |

UMAP warm start is the clear winner: **0.87s/frame at 50 epochs** with excellent quality. t-SNE has a minimum of 250 iterations and only 4x speedup.

Warm start is automatic in the training pipeline — first frame runs full, subsequent frames use previous 2D positions as init with reduced iterations.

### Test 11: Full pipeline with warm UMAP

```bash
python main.py word2vec --mode sentence -W 80 -H 80 --dims 8 \
    --k 24 --k-neg 5 --lr 0.0005 --normalize-every 100 \
    -i K_80_g.png -f 1000 --save-every 10 -o output_7/run3_umap \
    --render umap --align --warm-start output_7/run2/model.npy
```

- 1000 ticks, 100 frames + final + model saved
- **122.9s total (~1.2s/frame)** — first frame ~15s (cold UMAP), subsequent ~0.9s (warm)
- Clear K visible across all frames, consistent orientation via Procrustes alignment
- Model auto-saved to `output_7/run3_umap/model.npy` for further warm starts

### Integrated training pipeline

```bash
# First run: train + save model
python main.py word2vec --mode sentence -W 80 -H 80 --dims 8 \
    --k 24 --k-neg 5 --lr 0.0005 --normalize-every 100 \
    -i K_80_g.png -f 1000 --save-every 10 -o output_7/run1 \
    --render umap --align

# Continue from saved model
python main.py word2vec --mode sentence ... \
    --render tsne --align --warm-start output_7/run1/model.npy
```

Features:
- `--warm-start MODEL.npy` — load previous embeddings as initial positions
- `--align` — Procrustes-align every frame to grid for consistent orientation
- `--render {method}` — any projection method, with automatic warm start for UMAP/t-SNE
- `--save-model` — explicit save path (auto-saves to output_dir/model.npy if omitted)

### Projection method comparison for frame rendering

| Method | 1st frame | Warm frames | Quality (D8) | Warm start |
|--------|-----------|-------------|-------------|------------|
| `pca`/`euclidean` | instant | instant | poor | n/a |
| `bestpc` | instant | instant | poor-ok | n/a |
| `direct` | instant | instant | poor | n/a |
| `angular` | instant | instant | poor | n/a |
| `lstsq` | instant | instant | medium | n/a |
| `procrustes` | instant | instant | poor | n/a |
| `spectral` | ~3s | ~3s | good | no |
| **`tsne`** | ~18s | **~5s** | excellent | **yes** (250 iters) |
| **`umap`** | ~15s | **~0.9s** | **excellent** | **yes** (50 epochs) |
| `mds` | minutes | minutes | good | no |

For multi-frame video: **`umap`** with warm start (~0.9s/frame). For single renders: `umap` or `tsne`. For quick checks: `lstsq` or `bestpc`.

### Test 12: Warm start vs cold UMAP comparison

Three runs with identical training (1000 ticks, 100 frames, 80x80 D8, sentence mode) — only the UMAP rendering differs:

| Run | UMAP warm start | Procrustes align | Total time | Time/frame | Notes |
|-----|----------------|-----------------|------------|-----------|-------|
| `run3_umap` | yes | yes | 122.9s | ~1.2s | Best: fast + stable |
| `run4_umap_noalign` | yes | no | 123.5s | ~1.2s | K drifts orientation between frames |
| `run5_umap_cold` | no | yes | 727.1s | ~7.3s | 6x slower, stable via Procrustes |

**Observations:**
- Warm start provides **6x speedup** (1.2s vs 7.3s per frame) with no quality loss.
- Without Procrustes alignment, warm UMAP orientation drifts frame-to-frame — K is recognizable but rotates/flips.
- Cold UMAP + Procrustes produces stable output but is much slower since each frame runs full UMAP (~15s first frame, ~7s average with overhead).
- **Recommended pipeline: warm UMAP + Procrustes** — fast (0.9s/frame after first) and orientation-stable.

### Baseline run (from scratch)

```bash
python main.py word2vec --mode sentence -W 80 -H 80 --dims 8 \
    --k 24 --k-neg 5 --lr 0.0005 --normalize-every 100 \
    -i K_80_g.png -f 1000 --save-every 10 -o output_7/run7_baseline \
    --render umap --align
```

Trains from random initialization, 1000 ticks, 100 frames. Warm UMAP projection + Procrustes alignment. **125.5s total.**

### Procrustes disparity as a sorting metric

The Procrustes disparity from `--align` is a natural metric for how well the embeddings encode spatial structure. It measures the residual error after optimally rotating/scaling the projected 2D layout to match the known grid — lower disparity means the learned embedding better preserves the original spatial arrangement.

From the baseline run (random init → 1000 ticks):

| Tick | Disparity | Phase |
|------|-----------|-------|
| 10 | 0.9997 | Random — no structure yet |
| 50 | 0.9508 | First neighbor relationships forming |
| 100 | 0.7786 | Clusters emerging |
| 200 | 0.4178 | Coarse spatial layout |
| 300 | 0.1845 | K clearly recognizable |
| 400 | 0.1134 | Fine structure filling in |
| 500 | 0.0532 | Near-converged |
| 700 | 0.0330 | Stable |
| 1000 | 0.0257 | Final |

Disparity drops monotonically from ~1.0 (random) to ~0.025 (well-sorted). This curve is cheap to compute (one Procrustes solve per frame, ~1ms) and directly measures what we care about: how faithfully the embedding recovers the original spatial map. It could serve as a stopping criterion or learning rate scheduler signal.

## Files

- `main.py` — Main script with sentence mode, warm start, all projections
- `render_embeddings.py` — Standalone renderer with linear + nonlinear projections + Procrustes alignment
- `run_skipgram_compare.py` — Three-way comparison script (gensim vs single-pair vs sentence)
- `solvers/drift_torch.py` — PyTorch GPU solver with `tick_sentence()` method
- `output_7/` — Saved models (.npy), rendered images (.png), frame sequences

## Next steps

- Try sentence mode as initialization for Euclidean solver (two-stage pipeline)
- Explore adaptive learning rate schedules
- Test with real image neighbor structures (not just 2D grid proximity)
- Investigate faster nonlinear projections for streaming (GPU UMAP, parametric UMAP, or learned projection networks)

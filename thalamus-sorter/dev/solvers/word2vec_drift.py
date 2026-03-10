"""Word2vec-style drift: neurons update embeddings using skip-gram
with negative sampling, adapted for topographic map formation.

Positive pairs: precomputed top-K neighbors (or random high-similarity peers).
Negative pairs: randomly sampled neurons.
Update: sigmoid-based with per-dimension coefficients from peer vectors.

Key differences from centroid drift (continuous_drift.py):
  - Pairwise updates, not centroid averaging
  - Sigmoid self-regulation: focuses learning on "mistakes"
  - Per-dimension coefficients: each dim updated by peer's value in that dim
  - Explicit negative sampling: push dissimilar pairs apart
"""

import numpy as np

try:
    import cupy as cp
    _HAS_CUPY = True
except ImportError:
    _HAS_CUPY = False


def _get_xp(gpu):
    if gpu and _HAS_CUPY:
        return cp
    return np


class Word2vecDrift:
    """Sort neurons using word2vec-style skip-gram with negative sampling.

    Each tick: for each neuron, sample one positive neighbor and k_neg
    random negatives. Update embeddings with sigmoid objective.
    """

    def __init__(self, width, height, top_k, k=24, lr=0.05, dims=2,
                 k_neg=5, image=None, gpu=False):
        """
        Args:
            width, height: grid dimensions (for rendering)
            top_k: (n, k) precomputed neighbor indices
            k: number of neighbors
            lr: learning rate
            dims: dimensionality of position vectors (>= 2)
            k_neg: number of negative samples per positive
            image: optional (height, width) uint8 array for reconstruction
            gpu: use CuPy GPU acceleration
        """
        self.width = width
        self.height = height
        self.n = width * height
        self.k = k
        self.lr = lr
        self.dims = dims
        self.k_neg = k_neg
        self.gpu = gpu and _HAS_CUPY

        # Position vectors: (n, dims) — small init like word2vec
        self.positions = np.random.normal(0, 0.1, (self.n, dims)).astype(np.float32)

        self.top_k = top_k.astype(np.int32)

        # Pixel values for image reconstruction
        self.image = image
        if image is not None:
            self.pixel_values = np.zeros(self.n, dtype=np.uint8)
            for i in range(self.n):
                ox, oy = i % width, i // width
                self.pixel_values[i] = image[oy, ox]
        else:
            self.pixel_values = None

        # Rendered output
        self.output = None
        self.neurons_matrix = None

        # Move to GPU
        if self.gpu:
            self.positions = cp.asarray(self.positions)
            self.top_k = cp.asarray(self.top_k)

    def tick(self):
        """One iteration: skip-gram with negative sampling for all neurons."""
        xp = _get_xp(self.gpu)

        # --- Positive: sample one random neighbor from top_k ---
        if self.gpu:
            pos_idx = cp.random.randint(0, self.k, self.n, dtype=cp.int32)
        else:
            pos_idx = np.random.randint(0, self.k, self.n).astype(np.int32)
        j_pos = self.top_k[xp.arange(self.n), pos_idx]  # (n,)

        # Euclidean distance to positive peer
        pos_j = self.positions[j_pos]                           # (n, dims)
        delta_pos = pos_j - self.positions                      # (n, dims)
        dist_sq_pos = xp.sum(delta_pos ** 2, axis=1)            # (n,)

        # Sigmoid: σ(dist²) — large when far (needs pull), small when close (done)
        sig_pos = 1.0 / (1.0 + xp.exp(xp.clip(-dist_sq_pos, -20, 20)))  # (n,)

        # Pull TOWARD positive peer, scaled by sigmoid
        self.positions += self.lr * sig_pos[:, None] * delta_pos

        # --- Negative: sample all k_neg at once, vectorized ---
        if self.gpu:
            j_neg = cp.random.randint(0, self.n, (self.n, self.k_neg), dtype=cp.int32)
        else:
            j_neg = np.random.randint(0, self.n, (self.n, self.k_neg)).astype(np.int32)

        neg_j = self.positions[j_neg]                            # (n, k_neg, dims)
        delta_neg = neg_j - self.positions[:, None, :]           # (n, k_neg, dims)
        dist_sq_neg = xp.sum(delta_neg ** 2, axis=2)             # (n, k_neg)

        # Sigmoid: σ(-dist²) — large when close (wrongly similar), small when far (fine)
        sig_neg = 1.0 / (1.0 + xp.exp(xp.clip(dist_sq_neg, -20, 20)))  # (n, k_neg)

        # Push AWAY from negative peers, scaled by sigmoid
        push = xp.sum(sig_neg[:, :, None] * delta_neg, axis=1)  # (n, dims)
        self.positions -= self.lr * push

        self._normalize(xp)

    def _normalize(self, xp):
        """LayerNorm: center at origin, unit variance per dimension."""
        self.positions -= xp.mean(self.positions, axis=0)
        std = xp.std(self.positions, axis=0)
        std = xp.where(std < 1e-8, xp.ones_like(std), std)
        self.positions /= std

    def render(self):
        """Quantize continuous positions to a 2D grid for display.
        Same as ContinuousDrift: PCA + Voronoi KDTree."""
        from scipy.spatial import cKDTree

        pos = self.positions
        if self.gpu:
            pos = cp.asnumpy(pos)

        if self.dims > 2:
            _, _, Vt = np.linalg.svd(pos, full_matrices=False)
            pos_2d = (pos @ Vt[:2].T).astype(np.float64)
        else:
            pos_2d = pos[:, :2].astype(np.float64)

        for d in range(2):
            mn, mx = pos_2d[:, d].min(), pos_2d[:, d].max()
            span = mx - mn if mx - mn > 1e-8 else 1.0
            target = (self.width - 1) if d == 0 else (self.height - 1)
            pos_2d[:, d] = (pos_2d[:, d] - mn) / span * target

        grid_y, grid_x = np.mgrid[0:self.height, 0:self.width]
        grid_points = np.column_stack([
            grid_x.ravel().astype(np.float64),
            grid_y.ravel().astype(np.float64)
        ])

        tree = cKDTree(pos_2d)
        _, nearest = tree.query(grid_points)

        self.neurons_matrix = nearest.reshape(self.height, self.width)

        if self.pixel_values is not None:
            self.output = self.pixel_values[nearest].reshape(
                self.height, self.width)

        return self.output

    def run_gpu(self, n_ticks):
        """Run n ticks, then render to CPU."""
        for _ in range(n_ticks):
            self.tick()
        self.render()

    def position_stats(self):
        """Return position statistics for debugging."""
        xp = _get_xp(self.gpu)
        pos = self.positions
        means = xp.mean(pos, axis=0)
        stds = xp.std(pos, axis=0)
        if self.gpu:
            means, stds = cp.asnumpy(means), cp.asnumpy(stds)
        return {
            'mean': means, 'std': stds,
            'mean_norm': float(np.linalg.norm(means)),
            'std_mean': float(np.mean(stds)),
        }

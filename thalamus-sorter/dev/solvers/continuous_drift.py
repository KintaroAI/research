"""Continuous position drift: neurons have learned float position vectors
that drift toward the centroid of their K nearest neighbors.

No grid, no swaps, no conflict resolution. Each tick is just:
  positions += lr * (centroid_of_neighbors - positions)

Fully parallel — every neuron updates independently.
Rendering quantizes to a 2D grid via nearest-neighbor assignment.
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


class ContinuousDrift:
    """Sort neurons by drifting continuous position vectors toward
    the centroid of each neuron's K highest-affinity neighbors.

    Position vectors are unconstrained floats of configurable dimensionality.
    Rendering to a 2D image uses the first 2 dimensions + nearest-neighbor
    assignment to grid cells.
    """

    def __init__(self, width, height, top_k, k=24, lr=0.05, dims=2,
                 margin=0.1, image=None, gpu=False):
        """
        Args:
            width, height: grid dimensions (for rendering)
            top_k: (n, k) precomputed neighbor indices
            k: number of neighbors
            lr: learning rate (step fraction toward centroid)
            dims: dimensionality of position vectors (>= 2)
            image: optional (height, width) uint8 array for reconstruction
            gpu: use CuPy GPU acceleration
        """
        self.width = width
        self.height = height
        self.n = width * height
        self.k = k
        self.lr = lr
        self.dims = dims
        self.margin = margin
        self.gpu = gpu and _HAS_CUPY

        # Position vectors: (n, dims) — the "embeddings"
        # All dims: random normal, then LayerNorm to unit variance
        self.positions = np.random.normal(0, 1, (self.n, dims)).astype(np.float32)

        self.top_k = top_k.astype(np.int32)

        # Pixel value for each neuron identity (for image reconstruction)
        self.image = image
        if image is not None:
            self.pixel_values = np.zeros(self.n, dtype=np.uint8)
            for i in range(self.n):
                ox, oy = i % width, i // width
                self.pixel_values[i] = image[oy, ox]
        else:
            self.pixel_values = None

        # Rendered output (populated by render())
        self.output = None
        self.neurons_matrix = None

        # Move to GPU
        if self.gpu:
            self.positions = cp.asarray(self.positions)
            self.top_k = cp.asarray(self.top_k)

    def tick(self):
        """One iteration: every neuron lerps toward its neighbors' centroid,
        with a dead zone to prevent over-convergence. Then rescale."""
        xp = _get_xp(self.gpu)
        neighbor_pos = self.positions[self.top_k]    # (n, k, dims)
        centroids = xp.mean(neighbor_pos, axis=1)    # (n, dims)
        delta = centroids - self.positions
        if self.margin > 0:
            dist = xp.sqrt(xp.sum(delta ** 2, axis=1, keepdims=True))
            scale = xp.tanh(dist / self.margin)
            delta = delta * scale
        self.positions += self.lr * delta
        self._normalize(xp)

    def _normalize(self, xp):
        """LayerNorm: center at origin, unit variance per dimension."""
        self.positions -= xp.mean(self.positions, axis=0)
        std = xp.std(self.positions, axis=0)
        std = xp.where(std < 1e-8, xp.ones_like(std), std)
        self.positions /= std

    def _normalize_rmsnorm(self, xp):
        """Center + per-vector unit-length normalization.
        Points live on a hypersphere — needs adapted render for display."""
        self.positions -= xp.mean(self.positions, axis=0)
        norms = xp.sqrt(xp.sum(self.positions ** 2, axis=1, keepdims=True))
        norms = xp.where(norms < 1e-8, xp.ones_like(norms), norms)
        self.positions /= norms

    def render(self):
        """Quantize continuous positions to a 2D grid for display.
        Voronoi-style: each grid cell gets the nearest neuron's pixel value.
        For dims > 2, uses PCA to project to 2D first.
        Fast O(n log n) via KDTree — not bijective but visually clear."""
        from scipy.spatial import cKDTree

        pos = self.positions
        if self.gpu:
            pos = cp.asnumpy(pos)

        if self.dims > 2:
            # PCA projection to 2D
            _, _, Vt = np.linalg.svd(pos, full_matrices=False)
            pos_2d = (pos @ Vt[:2].T).astype(np.float64)
        else:
            pos_2d = pos[:, :2].astype(np.float64)

        # Map from normalized space to grid coordinates
        for d in range(2):
            mn, mx = pos_2d[:, d].min(), pos_2d[:, d].max()
            span = mx - mn if mx - mn > 1e-8 else 1.0
            target = (self.width - 1) if d == 0 else (self.height - 1)
            pos_2d[:, d] = (pos_2d[:, d] - mn) / span * target

        # Grid cell centers
        grid_y, grid_x = np.mgrid[0:self.height, 0:self.width]
        grid_points = np.column_stack([
            grid_x.ravel().astype(np.float64),
            grid_y.ravel().astype(np.float64)
        ])

        # Nearest neuron for each grid cell
        tree = cKDTree(pos_2d)
        _, nearest = tree.query(grid_points)

        self.neurons_matrix = nearest.reshape(self.height, self.width)

        if self.pixel_values is not None:
            self.output = self.pixel_values[nearest].reshape(
                self.height, self.width)

        return self.output

    def displacement(self):
        """Mean displacement: average distance from each neuron's current
        position to its ideal grid position. Lower = better topographic map.
        Computed in position space, independent of rendering."""
        xp = _get_xp(self.gpu)
        # Ideal position for neuron i: (i % width, i // width)
        ideal = xp.zeros((self.n, 2), dtype=xp.float32)
        ids = xp.arange(self.n)
        ideal[:, 0] = ids % self.width
        ideal[:, 1] = ids // self.width
        diff = self.positions[:, :2] - ideal
        dists = xp.sqrt(xp.sum(diff ** 2, axis=1))
        mean_disp = float(xp.mean(dists))
        return mean_disp

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

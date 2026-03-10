"""Temporal correlation solver: neurons discover neighbors from
time-series correlations instead of precomputed top-K lists.

Buffer sources (mutually exclusive):
  - "gaussian": Generate T spatially smooth random fields (weak signal)
  - "embeddings": Run ContinuousDrift to get converged embeddings,
    use as pre-computed feature vectors (strong, clean signal)

During sorting, each neuron randomly samples P peers, computes
correlation from buffers, and updates embeddings: positive correlation
pulls, negative repels.
"""

import numpy as np
from scipy.ndimage import gaussian_filter

try:
    import cupy as cp
    _HAS_CUPY = True
except ImportError:
    _HAS_CUPY = False


def _get_xp(gpu):
    if gpu and _HAS_CUPY:
        return cp
    return np


def _generate_buffers_gaussian(width, height, T, sigma):
    """Generate T spatially smooth random fields, return (n, T) buffer.

    Each field is white noise smoothed with a Gaussian of given sigma.
    Nearby pixels get similar values -> high temporal correlation.
    Distant pixels get uncorrelated values -> low correlation.
    """
    n = width * height
    buf = np.zeros((n, T), dtype=np.float32)
    for t in range(T):
        noise = np.random.randn(height, width).astype(np.float32)
        smooth = gaussian_filter(noise, sigma=sigma)
        # Normalize to zero mean, unit variance
        smooth = (smooth - smooth.mean()) / (smooth.std() + 1e-8)
        buf[:, t] = smooth.ravel()
    return buf


def _generate_buffers_embeddings(width, height, top_k, dims=16,
                                 lr=0.05, ticks=100000):
    """Run ContinuousDrift to convergence, return embeddings as buffers.

    The converged position vectors encode spatial proximity — neurons
    that are close on the grid have similar embeddings. Using these as
    "buffer vectors" gives clean, strong correlations.

    Returns (n, dims) array — each neuron's buffer is its embedding vector.
    """
    from solvers.continuous_drift import ContinuousDrift

    n = width * height
    print(f"  Running ContinuousDrift for {ticks} ticks "
          f"(dims={dims}, lr={lr}) to generate embeddings...")
    solver = ContinuousDrift(width, height, top_k, k=top_k.shape[1],
                             lr=lr, dims=dims, margin=0, gpu=True)
    for _ in range(ticks):
        solver.tick()
    pos = solver.positions
    if _HAS_CUPY:
        import cupy as _cp
        if isinstance(pos, _cp.ndarray):
            pos = _cp.asnumpy(pos)
    return pos


def _generate_buffers_synthetic(width, height):
    """Use (x, y) grid coordinates as 2-element buffers.

    Returns (n, 2) array. Similarity computed via Gaussian RBF kernel
    in tick(), not Pearson.
    """
    n = width * height
    buf = np.zeros((n, 2), dtype=np.float32)
    buf[:, 0] = np.arange(n) % width
    buf[:, 1] = np.arange(n) // width
    return buf


class TemporalCorrelation:
    """Sort neurons by temporal correlation of synthetic time-series.

    Each neuron has a precomputed time-series buffer (filled at init from
    spatially smooth random fields). Each tick, neurons randomly sample P
    peers, compute Pearson correlation from buffers, and update embeddings.
    """

    def __init__(self, width, height, P=1, lr=0.05, dims=2,
                 buf_source="synthetic", T=200, sigma=5.0, threshold=0.0,
                 top_k=None, emb_dims=16, emb_ticks=100000,
                 image=None, gpu=False):
        """
        Args:
            width, height: grid dimensions (for rendering)
            P: number of random peers per neuron per tick
            lr: learning rate for correlation-driven updates
            dims: dimensionality of position vectors (>= 2)
            buf_source: "synthetic", "gaussian", or "embeddings"
            T: buffer length (for gaussian source)
            sigma: spatial smoothing sigma (for gaussian/synthetic RBF kernel)
            top_k: precomputed neighbor indices (for embeddings source)
            emb_dims: embedding dimensionality (for embeddings source)
            emb_ticks: convergence ticks (for embeddings source)
            image: optional (height, width) uint8 array for reconstruction
            gpu: use CuPy GPU acceleration
        """
        self.width = width
        self.height = height
        self.n = width * height
        self.P = P
        self.lr = lr
        self.dims = dims
        self.buf_source = buf_source
        self.sigma = sigma
        self.threshold = threshold
        self.gpu = gpu and _HAS_CUPY

        # Position vectors: (n, dims) — the "embeddings" to be learned
        self.positions = np.random.normal(0, 1, (self.n, dims)).astype(np.float32)

        # Generate buffers based on source
        if buf_source == "synthetic":
            print(f"  Synthetic buffers: (x,y) grid coords, Gaussian RBF sigma={sigma}")
            self.buffers = _generate_buffers_synthetic(width, height)
        elif buf_source == "embeddings":
            if top_k is None:
                raise ValueError("buf_source='embeddings' requires top_k")
            self.buffers = _generate_buffers_embeddings(
                width, height, top_k, dims=emb_dims, ticks=emb_ticks)
        else:
            print(f"  Generating {T} spatially smooth fields (sigma={sigma})...")
            self.buffers = _generate_buffers_gaussian(width, height, T, sigma)
        self.T = self.buffers.shape[1]

        # For non-synthetic: normalize buffers for fast Pearson
        if buf_source != "synthetic":
            buf_mean = self.buffers.mean(axis=1, keepdims=True)
            buf_centered = self.buffers - buf_mean
            buf_std = np.sqrt(
                np.sum(buf_centered ** 2, axis=1, keepdims=True) + 1e-8
            )
            self.buf_normed = buf_centered / buf_std
        else:
            self.buf_normed = None

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
            self.buffers = cp.asarray(self.buffers)
            if self.buf_normed is not None:
                self.buf_normed = cp.asarray(self.buf_normed)

    def _similarity(self, j, xp):
        """Compute similarity between each neuron and its peer j.

        For synthetic buffers: Gaussian RBF on (x,y) distance, shifted by threshold.
        For gaussian/embeddings: Pearson correlation from normalized buffers.
        Returns (n,) similarity values. Positive = pull, negative = repel.
        """
        if self.buf_source == "synthetic":
            # Gaussian RBF: exp(-dist² / (2σ²)) - threshold
            diff = self.buffers - self.buffers[j]       # (n, 2)
            dist_sq = xp.sum(diff ** 2, axis=1)         # (n,)
            sim = xp.exp(-dist_sq / (2 * self.sigma ** 2))  # (n,) in [0, 1]
            return sim - self.threshold
        else:
            # Pearson from pre-normalized buffers
            bi = self.buf_normed
            bj = self.buf_normed[j]
            return xp.sum(bi * bj, axis=1) / self.T

    def tick(self):
        """One iteration: each neuron samples P random peers, computes
        similarity from buffers, updates embeddings."""
        xp = _get_xp(self.gpu)

        # Random peer indices: (n, P)
        if self.gpu:
            peers = cp.random.randint(0, self.n, (self.n, self.P), dtype=cp.int32)
        else:
            peers = np.random.randint(0, self.n, (self.n, self.P)).astype(np.int32)

        for p in range(self.P):
            j = peers[:, p]  # (n,) — peer index for each neuron
            sim = self._similarity(j, xp)  # (n,) in [0, 1] or [-1, 1]

            # Pull toward peer proportional to similarity
            delta = self.positions[j] - self.positions  # (n, dims)
            update = self.lr * sim[:, None] * delta     # (n, dims)
            self.positions += update

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

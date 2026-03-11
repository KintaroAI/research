"""Word2vec-style drift: neurons update embeddings using sigmoid-scaled
pairwise updates, adapted for topographic map formation.

Two modes:
  - tick(): precomputed top-K for positives, random for negatives (skip-gram)
  - tick_similarity(): all-random sampling, similarity decides attract/repel

Key properties:
  - Sigmoid self-regulation: focuses learning on "mistakes", controls scale
  - No normalization needed — push/pull balance stabilizes embeddings
  - Per-dimension coefficients from peer vectors (dot product mode)
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
    """Sort neurons using word2vec-style sigmoid updates.

    Supports two tick modes:
      - tick(): skip-gram with precomputed top-K (positive) + random (negative)
      - tick_similarity(): all random peers, similarity decides attract/repel
    """

    def __init__(self, width, height, top_k=None, k=24, lr=0.05, dims=2,
                 k_neg=5, P=10, sigma=5.0, threshold=0.0,
                 normalize_every=0, image=None, gpu=False):
        """
        Args:
            width, height: grid dimensions (for rendering)
            top_k: (n, k) precomputed neighbor indices (for tick() mode)
            k: number of neighbors
            lr: learning rate
            dims: dimensionality of position vectors (>= 2)
            k_neg: number of negative samples per positive (for tick() mode)
            P: random peers per neuron per tick (for tick_similarity() mode)
            sigma: Gaussian RBF kernel width (for tick_similarity() mode)
            threshold: similarity threshold for attract/repel boundary
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
        self.P = P
        self.sigma = sigma
        self.threshold = threshold
        self.normalize_every = normalize_every
        self._tick_count = 0
        self.gpu = gpu and _HAS_CUPY

        # Word vectors W: (n, dims) — small uniform init like word2vec
        scale = 0.5 / dims
        self.positions = np.random.uniform(-scale, scale, (self.n, dims)).astype(np.float32)

        # Context vectors C: (n, dims) — initialized to zeros like word2vec syn1neg
        self.contexts = np.zeros((self.n, dims), dtype=np.float32)

        if top_k is not None:
            self.top_k = top_k.astype(np.int32)
        else:
            self.top_k = None

        # Synthetic buffers: (x, y) grid coords for similarity computation
        self.grid_coords = np.zeros((self.n, 2), dtype=np.float32)
        self.grid_coords[:, 0] = np.arange(self.n) % width
        self.grid_coords[:, 1] = np.arange(self.n) // width

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
            self.contexts = cp.asarray(self.contexts)
            self.grid_coords = cp.asarray(self.grid_coords)
            if self.top_k is not None:
                self.top_k = cp.asarray(self.top_k)

    def tick(self):
        """Skip-gram mode: precomputed top-K positive + random negative.
        Euclidean: sigmoid on distance², update toward/away from peer."""
        xp = _get_xp(self.gpu)

        # --- Positive: sample one random neighbor from top_k ---
        if self.gpu:
            pos_idx = cp.random.randint(0, self.k, self.n, dtype=cp.int32)
        else:
            pos_idx = np.random.randint(0, self.k, self.n).astype(np.int32)
        j_pos = self.top_k[xp.arange(self.n), pos_idx]

        delta_pos = self.positions[j_pos] - self.positions  # toward peer
        dist_sq_pos = xp.sum(delta_pos ** 2, axis=1)
        # σ(dist²) — large when far apart → strong pull
        sig_pos = 1.0 / (1.0 + xp.exp(xp.clip(-dist_sq_pos, -20, 20)))
        self.positions += self.lr * sig_pos[:, None] * delta_pos

        # --- Negative: sample all k_neg at once ---
        if self.gpu:
            j_neg = cp.random.randint(0, self.n, (self.n, self.k_neg), dtype=cp.int32)
        else:
            j_neg = np.random.randint(0, self.n, (self.n, self.k_neg)).astype(np.int32)

        delta_neg = self.positions[j_neg] - self.positions[:, None, :]  # toward neg peer
        dist_sq_neg = xp.sum(delta_neg ** 2, axis=2)
        # σ(-dist²) — large when close → strong push
        sig_neg = 1.0 / (1.0 + xp.exp(xp.clip(dist_sq_neg, -20, 20)))
        push = xp.sum(sig_neg[:, :, None] * delta_neg, axis=1)
        self.positions -= self.lr * push  # push away from negative peers

    def tick_dual(self):
        """Two-vector dot product skip-gram like real word2vec.
        W (positions) and C (contexts) are separate vectors.
        Updates are always cross: W←C and C←W, never W←W."""
        xp = _get_xp(self.gpu)

        # --- Positive: sample one random neighbor from top_k ---
        if self.gpu:
            pos_idx = cp.random.randint(0, self.k, self.n, dtype=cp.int32)
        else:
            pos_idx = np.random.randint(0, self.k, self.n).astype(np.int32)
        j_pos = self.top_k[xp.arange(self.n), pos_idx]

        # dot(W[i], C[j]) for positive pairs
        c_pos = self.contexts[j_pos]
        dot_pos = xp.sum(self.positions * c_pos, axis=1)
        sig_pos = 1.0 / (1.0 + xp.exp(xp.clip(dot_pos, -20, 20)))
        # W[i] += lr * σ(-dot) * C[j]
        self.positions += self.lr * sig_pos[:, None] * c_pos
        # C[j] += lr * σ(-dot) * W[i] — use pre-update W
        w_for_ctx = self.positions - self.lr * sig_pos[:, None] * c_pos  # undo to get pre-update
        xp.add.at(self.contexts, j_pos, self.lr * sig_pos[:, None] * w_for_ctx)

        # --- Negative: sample k_neg random neurons ---
        if self.gpu:
            j_neg = cp.random.randint(0, self.n, (self.n, self.k_neg), dtype=cp.int32)
        else:
            j_neg = np.random.randint(0, self.n, (self.n, self.k_neg)).astype(np.int32)

        # dot(W[i], C[neg]) for negative pairs
        c_neg = self.contexts[j_neg]  # (n, k_neg, dims)
        dot_neg = xp.sum(self.positions[:, None, :] * c_neg, axis=2)  # (n, k_neg)
        sig_neg = 1.0 / (1.0 + xp.exp(xp.clip(-dot_neg, -20, 20)))
        # W[i] -= lr * σ(dot) * C[neg]
        push_w = xp.sum(sig_neg[:, :, None] * c_neg, axis=1)
        self.positions -= self.lr * push_w
        # C[neg] -= lr * σ(dot) * W[i]
        push_c = sig_neg[:, :, None] * self.positions[:, None, :]  # (n, k_neg, dims)
        for ki in range(self.k_neg):
            xp.add.at(self.contexts, j_neg[:, ki], -self.lr * push_c[:, ki, :])

        self._maybe_normalize()

    def _maybe_normalize(self):
        """Periodically normalize W and C to unit length to prevent magnitude blow-up
        while preserving angular relationships."""
        if self.normalize_every <= 0:
            return
        self._tick_count += 1
        if self._tick_count % self.normalize_every != 0:
            return
        xp = _get_xp(self.gpu)
        for vecs in [self.positions, self.contexts]:
            norms = xp.sqrt(xp.sum(vecs ** 2, axis=1, keepdims=True))
            norms = xp.where(norms < 1e-8, xp.ones_like(norms), norms)
            vecs /= norms

    def tick_dual_xy(self):
        """Two-vector dot product with alternating x/y neighbor signals.
        Even ticks use x-only neighbors, odd ticks use y-only neighbors.
        Forces dimension specialization by providing conflicting 1D signals."""
        if not hasattr(self, '_top_k_x'):
            self._build_xy_topk()
        # Alternate
        self.top_k = self._top_k_x if self._tick_count % 2 == 0 else self._top_k_y
        self.tick_dual()

    def _build_xy_topk(self):
        """Precompute separate top-K for x-primary and y-primary proximity.
        Secondary axis has weight 0.3 — neighbors are close in primary axis
        AND reasonably close in secondary axis, reducing conflicting gradients."""
        from scipy.spatial import cKDTree
        coords = self.grid_coords
        if self.gpu:
            coords = cp.asnumpy(coords)

        # x-neighbors: x dominant, y secondary (weight 0.3)
        coords_x = coords.copy()
        coords_x[:, 1] *= 0.3
        tree_x = cKDTree(coords_x)
        _, idx_x = tree_x.query(coords_x, k=self.k + 1)
        self._top_k_x = idx_x[:, 1:].astype(np.int32)

        # y-neighbors: y dominant, x secondary (weight 0.3)
        coords_y = coords.copy()
        coords_y[:, 0] *= 0.3
        tree_y = cKDTree(coords_y)
        _, idx_y = tree_y.query(coords_y, k=self.k + 1)
        self._top_k_y = idx_y[:, 1:].astype(np.int32)

        if self.gpu:
            self._top_k_x = cp.asarray(self._top_k_x)
            self._top_k_y = cp.asarray(self._top_k_y)

    def tick_similarity(self):
        """Similarity mode: all random peers, similarity decides attract/repel.
        Euclidean: sigmoid on distance², update toward/away from peer."""
        xp = _get_xp(self.gpu)

        # Sample P random peers
        if self.gpu:
            peers = cp.random.randint(0, self.n, (self.n, self.P), dtype=cp.int32)
        else:
            peers = np.random.randint(0, self.n, (self.n, self.P)).astype(np.int32)

        for p in range(self.P):
            j = peers[:, p]

            # Similarity from grid coords: Gaussian RBF
            diff = self.grid_coords - self.grid_coords[j]
            dist_sq = xp.sum(diff ** 2, axis=1)
            sim = xp.exp(-dist_sq / (2 * self.sigma ** 2))  # [0, 1]

            # Shift by threshold: positive = attract, negative = repel
            signal = sim - self.threshold  # positive → pull, negative → push

            # Euclidean: direction toward peer
            delta = self.positions[j] - self.positions
            emb_dist_sq = xp.sum(delta ** 2, axis=1)

            # Sigmoid on embedding distance:
            # attract (signal > 0): σ(dist²) — large when far → pull hard
            # repel (signal < 0): σ(-dist²) — large when close → push hard
            sig = 1.0 / (1.0 + xp.exp(xp.clip(-signal * emb_dist_sq, -20, 20)))

            self.positions += self.lr * (signal * sig)[:, None] * delta

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

    def render_angular(self):
        """Render based on angular similarity: normalize vectors to unit length
        before Voronoi assignment. Use when embeddings encode direction (dot product)
        rather than position (Euclidean distance)."""
        from scipy.spatial import cKDTree

        pos = self.positions
        if self.gpu:
            pos = cp.asnumpy(pos)

        # Normalize to unit vectors — only direction matters
        norms = np.sqrt(np.sum(pos ** 2, axis=1, keepdims=True))
        norms = np.where(norms < 1e-8, 1.0, norms)
        pos = pos / norms

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

    def render_bestpc(self):
        """Render using the two PCs most correlated with grid x/y coordinates.
        For dot-product embeddings where spatial axes may not be PC0/PC1."""
        from scipy.spatial import cKDTree

        pos = self.positions
        if self.gpu:
            pos = cp.asnumpy(pos)

        grid_x = np.arange(self.n) % self.width
        grid_y = np.arange(self.n) // self.width

        _, _, Vt = np.linalg.svd(pos, full_matrices=False)
        pcs = pos @ Vt.T  # (n, dims)

        # Find best PC for each axis
        best_x_pc, best_x_corr = 0, 0
        best_y_pc, best_y_corr = 0, 0
        for i in range(self.dims):
            cx = abs(np.corrcoef(pcs[:, i], grid_x)[0, 1])
            cy = abs(np.corrcoef(pcs[:, i], grid_y)[0, 1])
            if cx > best_x_corr:
                best_x_corr = cx
                best_x_pc = i
            if cy > best_y_corr:
                best_y_corr = cy
                best_y_pc = i

        pos_2d = np.column_stack([pcs[:, best_x_pc], pcs[:, best_y_pc]]).astype(np.float64)
        # Flip if negatively correlated
        if np.corrcoef(pcs[:, best_x_pc], grid_x)[0, 1] < 0:
            pos_2d[:, 0] *= -1
        if np.corrcoef(pcs[:, best_y_pc], grid_y)[0, 1] < 0:
            pos_2d[:, 1] *= -1

        for d in range(2):
            mn, mx = pos_2d[:, d].min(), pos_2d[:, d].max()
            span = mx - mn if mx - mn > 1e-8 else 1.0
            target = (self.width - 1) if d == 0 else (self.height - 1)
            pos_2d[:, d] = (pos_2d[:, d] - mn) / span * target

        grid_yy, grid_xx = np.mgrid[0:self.height, 0:self.width]
        grid_points = np.column_stack([
            grid_xx.ravel().astype(np.float64),
            grid_yy.ravel().astype(np.float64)
        ])

        tree = cKDTree(pos_2d)
        _, nearest = tree.query(grid_points)

        self.neurons_matrix = nearest.reshape(self.height, self.width)

        if self.pixel_values is not None:
            self.output = self.pixel_values[nearest].reshape(
                self.height, self.width)

        return self.output

    def run_gpu(self, n_ticks, mode="similarity"):
        """Run n ticks, then render to CPU."""
        tick_fn = self.tick_similarity if mode == "similarity" else self.tick
        for _ in range(n_ticks):
            tick_fn()
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

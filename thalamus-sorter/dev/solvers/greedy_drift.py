"""Greedy drift solver: neurons move toward the centroid of their
highest-affinity neighbors. Hebbian self-organization with spatial constraints.

Supports GPU acceleration via CuPy (--gpu flag). The vectorized tick()
computes all centroids and directions in one batch, then applies
non-conflicting swaps in parallel rounds.
"""

import numpy as np

try:
    import cupy as cp
    _HAS_CUPY = True
except ImportError:
    _HAS_CUPY = False


def _get_xp(gpu):
    """Return the array module (cupy or numpy)."""
    if gpu and _HAS_CUPY:
        return cp
    return np


class GreedyDrift:
    """Sort a neuron grid by iteratively moving each neuron one step
    toward the average position of its K nearest neighbors (by weight).

    Optionally tracks an image: each neuron identity maps to a pixel value,
    so as neurons sort, the image reconstructs from scrambled state.
    """

    def __init__(self, width, height, weight_matrix, k=24, move_fraction=0.9,
                 image=None, gpu=False):
        """
        Args:
            width, height: grid dimensions
            weight_matrix: (n, n) affinity matrix
            k: number of neighbors to attract toward
            move_fraction: fraction of neurons moved per tick
            image: optional (height, width) uint8 array — pixel values
                   mapped to neuron identities for visual reconstruction
            gpu: use CuPy GPU acceleration if available
        """
        self.width = width
        self.height = height
        self.n = width * height
        self.k = k
        self.move_fraction = move_fraction
        self.gpu = gpu and _HAS_CUPY

        xp = _get_xp(self.gpu)

        # Neuron grid: neurons_matrix[y, x] = neuron identity index
        perm = np.random.permutation(self.n)
        self.neurons_matrix = perm.reshape((height, width))

        # Current (x, y) position of each neuron identity
        self.coordinates = np.zeros((self.n, 2), dtype=np.float32)
        for y in range(height):
            for x in range(width):
                i = self.neurons_matrix[y, x]
                self.coordinates[i] = [x, y]

        # Precompute top-K neighbor indices for ALL neurons as dense array
        self.top_k = np.zeros((self.n, k), dtype=np.int32)
        for i in range(self.n):
            kj = np.argpartition(weight_matrix[i], -k - 1)[-k - 1:-1]
            self.top_k[i] = kj

        # Move to GPU if requested
        if self.gpu:
            self.coordinates = cp.asarray(self.coordinates)
            self.top_k = cp.asarray(self.top_k)

        # Image tracking
        self.image = image
        if image is not None:
            self.output = np.zeros((height, width), dtype=np.uint8)
            for y in range(height):
                for x in range(width):
                    i = self.neurons_matrix[y, x]
                    ox, oy = i % width, i // width
                    self.output[y, x] = image[oy, ox]
        else:
            self.output = None

    def tick(self):
        """One sorting iteration. Vectorized centroid computation,
        then conflict-free batch swaps."""
        xp = _get_xp(self.gpu)
        n_moves = int(self.move_fraction * self.n)

        # Pick random grid positions
        flat_indices = np.random.randint(0, self.n, size=n_moves)
        src_y = flat_indices // self.width
        src_x = flat_indices % self.width

        # Neuron IDs at those positions
        neuron_ids = self.neurons_matrix[src_y, src_x]

        # Top-K neighbor positions -> centroids  (all on GPU if enabled)
        neighbor_ids = self.top_k[neuron_ids]           # (n_moves, k)
        neighbor_pos = self.coordinates[neighbor_ids]    # (n_moves, k, 2)
        centroids = xp.mean(neighbor_pos, axis=1)       # (n_moves, 2)

        # Direction vectors
        if self.gpu:
            cur_x = cp.asarray(src_x, dtype=cp.float32)
            cur_y = cp.asarray(src_y, dtype=cp.float32)
        else:
            cur_x = src_x.astype(np.float32)
            cur_y = src_y.astype(np.float32)

        dx = centroids[:, 0] - cur_x
        dy = centroids[:, 1] - cur_y
        mag = xp.sqrt(dx * dx + dy * dy)

        nonzero = mag > 0
        dx_norm = xp.where(nonzero, dx / xp.maximum(mag, 1e-8), 0)
        dy_norm = xp.where(nonzero, dy / xp.maximum(mag, 1e-8), 0)

        new_x = xp.clip(xp.round(cur_x + dx_norm).astype(xp.int32),
                         0, self.width - 1)
        new_y = xp.clip(xp.round(cur_y + dy_norm).astype(xp.int32),
                         0, self.height - 1)

        # Transfer to CPU for swaps
        if self.gpu:
            new_x = cp.asnumpy(new_x)
            new_y = cp.asnumpy(new_y)
            nonzero = cp.asnumpy(nonzero)

        # Batch swaps: skip conflicts (two moves touching the same cell)
        touched = set()
        for idx in range(n_moves):
            if not nonzero[idx]:
                continue
            sx, sy = int(src_x[idx]), int(src_y[idx])
            nx, ny = int(new_x[idx]), int(new_y[idx])
            if sx == nx and sy == ny:
                continue
            src_key = sy * self.width + sx
            dst_key = ny * self.width + nx
            if src_key in touched or dst_key in touched:
                continue
            touched.add(src_key)
            touched.add(dst_key)
            self._swap(sx, sy, nx, ny)

    def _swap(self, x1, y1, x2, y2):
        """Swap the neurons at two grid positions."""
        i = self.neurons_matrix[y1, x1]
        j = self.neurons_matrix[y2, x2]
        self.neurons_matrix[y1, x1] = j
        self.neurons_matrix[y2, x2] = i
        self.coordinates[i, 0] = x2
        self.coordinates[i, 1] = y2
        self.coordinates[j, 0] = x1
        self.coordinates[j, 1] = y1
        if self.output is not None:
            self.output[y1, x1], self.output[y2, x2] = (
                self.output[y2, x2], self.output[y1, x1])

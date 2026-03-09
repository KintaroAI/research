"""Greedy drift solver: neurons move toward the centroid of their
highest-affinity neighbors. Hebbian self-organization with spatial constraints.

Two execution modes:
  tick()     — CPU reference: vectorized centroids, sequential Python swap loop.
  tick_gpu() — Full GPU: vectorized centroids + parallel conflict resolution.
               No CPU-GPU transfers. Use run_gpu(n) to batch multiple ticks
               and only sync back to CPU at the end.
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

    def __init__(self, width, height, weight_matrix=None, k=24, move_fraction=0.9,
                 image=None, gpu=False, top_k=None):
        """
        Args:
            width, height: grid dimensions
            weight_matrix: (n, n) affinity matrix (legacy — triggers O(n²) memory)
            k: number of neighbors to attract toward
            move_fraction: fraction of neurons moved per tick
            image: optional (height, width) uint8 array — pixel values
                   mapped to neuron identities for visual reconstruction
            gpu: use CuPy GPU acceleration if available
            top_k: (n, k) precomputed neighbor indices — if provided,
                   weight_matrix is not needed (O(nK) memory)
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

        # Top-K neighbor indices: either precomputed or extracted from matrix
        if top_k is not None:
            self.top_k = top_k.astype(np.int32)
        elif weight_matrix is not None:
            self.top_k = np.zeros((self.n, k), dtype=np.int32)
            for i in range(self.n):
                kj = np.argpartition(weight_matrix[i], -k - 1)[-k - 1:-1]
                self.top_k[i] = kj
        else:
            raise ValueError("Either weight_matrix or top_k must be provided")

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

    # ------------------------------------------------------------------
    # Full-GPU path: parallel conflict resolution, no CPU-GPU transfers
    # ------------------------------------------------------------------

    def _init_gpu_state(self):
        """Move neurons_matrix and output to GPU. Called once before
        the first tick_gpu(). After this, tick_gpu() operates entirely
        on GPU arrays; call sync_to_cpu() to read results."""
        if not self.gpu:
            raise RuntimeError("GPU methods require gpu=True")
        if hasattr(self, '_gpu_ready'):
            return
        self._gpu_neurons = cp.asarray(self.neurons_matrix)
        if self.output is not None:
            self._gpu_output = cp.asarray(self.output)
        self._gpu_ready = True

    def sync_to_cpu(self):
        """Copy GPU state back to CPU arrays (neurons_matrix, output)."""
        if not hasattr(self, '_gpu_ready'):
            return
        self.neurons_matrix = cp.asnumpy(self._gpu_neurons)
        self.coordinates = cp.asnumpy(self.coordinates)
        if self.output is not None:
            self.output = cp.asnumpy(self._gpu_output)

    def _restore_gpu_coords(self):
        """Re-upload coordinates to GPU after sync_to_cpu()."""
        if not isinstance(self.coordinates, cp.ndarray):
            self.coordinates = cp.asarray(self.coordinates)

    def tick_gpu(self):
        """One sorting iteration, fully on GPU.

        Parallel conflict resolution: each proposed swap gets a random
        priority. A swap is accepted only if it holds the highest priority
        at BOTH its source and destination cells — guaranteeing no two
        accepted swaps touch the same cell."""
        self._init_gpu_state()
        self._restore_gpu_coords()

        n_moves = int(self.move_fraction * self.n)

        # Pick random grid positions (GPU)
        flat_indices = cp.random.randint(0, self.n, size=n_moves)
        src_y = flat_indices // self.width
        src_x = flat_indices % self.width

        # Neuron IDs at those positions (GPU)
        neuron_ids = self._gpu_neurons[src_y, src_x]

        # Top-K neighbor positions -> centroids (GPU)
        neighbor_ids = self.top_k[neuron_ids]           # (n_moves, k)
        neighbor_pos = self.coordinates[neighbor_ids]    # (n_moves, k, 2)
        centroids = cp.mean(neighbor_pos, axis=1)       # (n_moves, 2)

        # Direction vectors (GPU)
        cur_x = src_x.astype(cp.float32)
        cur_y = src_y.astype(cp.float32)
        dx = centroids[:, 0] - cur_x
        dy = centroids[:, 1] - cur_y
        mag = cp.sqrt(dx * dx + dy * dy)

        nonzero = mag > 0
        dx_norm = cp.where(nonzero, dx / cp.maximum(mag, 1e-8), 0)
        dy_norm = cp.where(nonzero, dy / cp.maximum(mag, 1e-8), 0)

        new_x = cp.clip(cp.round(cur_x + dx_norm).astype(cp.int32),
                         0, self.width - 1)
        new_y = cp.clip(cp.round(cur_y + dy_norm).astype(cp.int32),
                         0, self.height - 1)

        # Flat cell keys
        flat_src = src_y * self.width + src_x
        flat_dst = new_y * self.width + new_x

        # Valid: nonzero movement and different cells
        valid = nonzero & (flat_src != flat_dst)

        # Parallel conflict resolution via random integer priorities.
        # Each move gets a unique random priority (uint32 for exact comparison
        # on GPU — CuPy maximum.at supports uint32 but not int64).
        # For each cell, find the max priority among all moves touching it.
        # A move is accepted iff it wins at both its src and dst cells.
        priorities = cp.random.randint(1, 2**31, size=n_moves, dtype=cp.uint32)
        priorities *= valid  # invalid moves get priority 0

        cell_max = cp.zeros(self.n, dtype=cp.uint32)
        cp.maximum.at(cell_max, flat_src, priorities)
        cp.maximum.at(cell_max, flat_dst, priorities)

        wins_src = (priorities > 0) & (priorities == cell_max[flat_src])
        wins_dst = (priorities > 0) & (priorities == cell_max[flat_dst])
        accepted = wins_src & wins_dst

        # Apply accepted swaps — no conflicts, safe for parallel fancy indexing
        idx = cp.where(accepted)[0]
        if len(idx) > 0:
            a_sy, a_sx = src_y[idx], src_x[idx]
            a_ny, a_nx = new_y[idx], new_x[idx]

            neuron_a = self._gpu_neurons[a_sy, a_sx]
            neuron_b = self._gpu_neurons[a_ny, a_nx]

            # Swap neuron IDs on the grid
            self._gpu_neurons[a_sy, a_sx] = neuron_b
            self._gpu_neurons[a_ny, a_nx] = neuron_a

            # Update coordinate tracking
            self.coordinates[neuron_a, 0] = a_nx.astype(cp.float32)
            self.coordinates[neuron_a, 1] = a_ny.astype(cp.float32)
            self.coordinates[neuron_b, 0] = a_sx.astype(cp.float32)
            self.coordinates[neuron_b, 1] = a_sy.astype(cp.float32)

            # Swap image pixels
            if self.output is not None:
                pixel_a = self._gpu_output[a_sy, a_sx].copy()
                pixel_b = self._gpu_output[a_ny, a_nx].copy()
                self._gpu_output[a_sy, a_sx] = pixel_b
                self._gpu_output[a_ny, a_nx] = pixel_a

    def run_gpu(self, n_ticks):
        """Run n ticks fully on GPU, sync to CPU at the end."""
        self._init_gpu_state()
        self._restore_gpu_coords()
        for _ in range(n_ticks):
            self.tick_gpu()
        self.sync_to_cpu()

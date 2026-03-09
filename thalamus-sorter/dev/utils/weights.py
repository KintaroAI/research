"""Weight matrix construction for neuron affinity.

Two approaches:
- Full matrix: inverse_distance_1d(), decay_distance_2d() — O(n²) memory,
  only feasible for small grids (≤40x40).
- Top-K only: topk_decay2d(), topk_inv1d() — O(nK) memory,
  scales to arbitrary grid sizes. Returns (n, K) index array directly.
"""

import numpy as np


# ---------------------------------------------------------------------------
# Top-K neighbor computation (matrix-free, O(nK) memory)
# ---------------------------------------------------------------------------

def topk_decay2d(width, height, k):
    """Return (n, k) array of top-K neighbor indices by 2D proximity.
    Equivalent to building decay_distance_2d then argpartitioning each row,
    but without ever allocating the n×n matrix.

    Uses scipy KDTree for efficient spatial lookup."""
    from scipy.spatial import cKDTree

    n = width * height
    # Ideal grid positions for each neuron
    coords = np.column_stack([np.arange(n) % width,
                              np.arange(n) // width]).astype(np.float32)
    tree = cKDTree(coords)
    # Query k+1 because the closest neighbor is the neuron itself
    _, indices = tree.query(coords, k=k + 1)
    # Drop self (column 0)
    return indices[:, 1:].astype(np.int32)


def topk_inv1d(n, k):
    """Return (n, k) array of top-K neighbor indices by 1D index distance.
    For neuron i, the closest are i±1, i±2, ... — just the K nearest indices."""
    k = min(k, n - 1)
    top_k = np.zeros((n, k), dtype=np.int32)
    for i in range(n):
        # Candidates sorted by |i - j|, excluding self
        candidates = np.argsort(np.abs(np.arange(n) - i))
        top_k[i] = candidates[1:k + 1]
    return top_k


# ---------------------------------------------------------------------------
# Full matrix construction (O(n²) memory — legacy, used by MST/SA/camera)
# ---------------------------------------------------------------------------

def inverse_distance_1d(n):
    """Weight matrix based on 1D index distance: weight = 1/|i-j|.
    Neurons with nearby indices have high affinity."""
    W = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            w = 1.0 / abs(i - j)
            W[i, j] = w
            W[j, i] = w
    return W


def decay_distance_2d(width, height, decay_rate=0.1):
    """Weight matrix based on 2D Euclidean distance with linear decay:
    weight = max(0, 1 - distance * decay_rate).
    Each neuron index maps to a (x, y) position on the grid."""
    n = width * height
    W = np.zeros((n, n))
    for i in range(n):
        xi, yi = i % width, i // width
        for j in range(i + 1, n):
            xj, yj = j % width, j // width
            dist = ((xi - xj) ** 2 + (yi - yj) ** 2) ** 0.5
            w = max(0.0, 1.0 - dist * decay_rate)
            W[i, j] = w
            W[j, i] = w
        W[i, i] = 1.0
    return W


class OnlineWeightMatrix:
    """Learns a weight matrix online from observed correlations.
    Maintains a running average over the last `tail` observations."""

    def __init__(self, n, tail=10):
        self.n = n
        self.tail = tail
        # Slice 0 = weight, slice 1 = update count
        self._data = np.zeros((n, n, 2))

    @property
    def weights(self):
        return self._data[:, :, 0]

    def update(self, i, j, weight):
        """Update the running average weight between neurons i and j."""
        count = self._data[i, j, 1]
        if count >= self.tail - 1:
            count = self.tail - 1
        new_weight = (self._data[i, j, 0] * count + weight) / (count + 1)
        self._data[i, j, 0] = new_weight
        self._data[i, j, 1] = count + 1
        self._data[j, i, 0] = new_weight
        self._data[j, i, 1] = count + 1

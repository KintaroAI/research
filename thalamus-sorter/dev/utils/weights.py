"""Weight matrix construction for neuron affinity."""

import numpy as np


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

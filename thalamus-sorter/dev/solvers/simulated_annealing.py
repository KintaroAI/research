"""Simulated annealing solver: random neuron swaps with
temperature-based acceptance to minimize correlation * distance cost."""

import math
import random
import numpy as np

from utils.geometry import coords


def cost_sampled(weight_matrix, perm, width, sample_size):
    """Approximate cost by sampling random neuron pairs.
    Cost = sum(affinity(i,j) * manhattan_distance(pos_i, pos_j)).

    perm is a permutation matrix (identity shuffled by columns):
    perm[:, output_slot] has a 1 at the row of the neuron assigned there.
    """
    n = weight_matrix.shape[0]
    cost = 0.0
    for _ in range(sample_size):
        i, j = random.sample(range(n), 2)
        # Find which neuron identities are at output positions i and j
        ni = _neuron_at(perm, i)
        nj = _neuron_at(perm, j)
        if ni is None or nj is None:
            continue
        x1, y1 = coords(i, width)
        x2, y2 = coords(j, width)
        dist = abs(x1 - x2) + abs(y1 - y2)
        cost += weight_matrix[ni, nj] * dist
    return cost


def cost_full(weight_matrix, perm, width):
    """Exact cost over all neuron pairs."""
    n = weight_matrix.shape[0]
    cost = 0.0
    for i in range(n):
        ni = _neuron_at(perm, i)
        if ni is None:
            continue
        for j in range(i + 1, n):
            nj = _neuron_at(perm, j)
            if nj is None:
                continue
            x1, y1 = coords(i, width)
            x2, y2 = coords(j, width)
            dist = abs(x1 - x2) + abs(y1 - y2)
            cost += weight_matrix[ni, nj] * dist
    return cost


def _neuron_at(perm, slot):
    """Which neuron identity occupies output slot `slot` in the permutation matrix."""
    idx = np.where(perm[:, slot] == 1)
    if len(idx[0]) == 0:
        return None
    return idx[0][0]


class SimulatedAnnealing:
    """SA solver operating on a permutation matrix."""

    def __init__(self, weight_matrix, width, height,
                 init_temp=100.0, cooling_rate=0.99, sampled=True):
        self.n = width * height
        self.width = width
        self.height = height
        self.weight_matrix = weight_matrix
        self.temp = init_temp
        self.cooling_rate = cooling_rate
        self.sampled = sampled

        # Permutation matrix: shuffled identity
        self.perm = np.eye(self.n)
        cols = list(range(self.n))
        random.shuffle(cols)
        self.perm = self.perm[:, cols]

        self._current_cost = self._cost()

    def _cost(self):
        if self.sampled:
            return cost_sampled(self.weight_matrix, self.perm,
                                self.width, self.n * self.n // 2)
        return cost_full(self.weight_matrix, self.perm, self.width)

    def tick(self, iterations=100):
        """Run `iterations` SA steps."""
        for _ in range(iterations):
            i = random.randint(0, self.n - 1)
            j = random.randint(0, self.n - 1)
            if i == j:
                continue

            # Swap columns i and j in permutation
            self.perm[:, [i, j]] = self.perm[:, [j, i]]
            new_cost = self._cost()
            delta = new_cost - self._current_cost

            if delta < 0 or (self.temp > 0 and
                             math.exp(-delta / self.temp) > random.random()):
                self._current_cost = new_cost
            else:
                # Undo swap
                self.perm[:, [i, j]] = self.perm[:, [j, i]]

            self.temp *= self.cooling_rate

    @property
    def neurons_matrix(self):
        """Current neuron arrangement as a (height, width) grid."""
        order = [_neuron_at(self.perm, s) for s in range(self.n)]
        return np.array(order).reshape(self.height, self.width)

    @property
    def cost(self):
        return self._current_cost

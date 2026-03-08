"""Greedy drift solver: neurons move toward the centroid of their
highest-affinity neighbors. Hebbian self-organization with spatial constraints."""

import numpy as np
import random


class GreedyDrift:
    """Sort a neuron grid by iteratively moving each neuron one step
    toward the average position of its K nearest neighbors (by weight)."""

    def __init__(self, width, height, weight_matrix, k=24, move_fraction=0.9):
        self.width = width
        self.height = height
        self.n = width * height
        self.k = k
        self.move_fraction = move_fraction

        # Neuron grid: neurons_matrix[y][x] = neuron identity index
        self.neurons_matrix = np.random.permutation(self.n).reshape((height, width))
        self.weight_matrix = weight_matrix

        # Current (x, y) position of each neuron identity
        self.coordinates = np.zeros((self.n, 2))
        for y in range(height):
            for x in range(width):
                i = self.neurons_matrix[y][x]
                self.coordinates[i] = [x, y]

        # Cache for top-K neighbor indices per neuron
        self._k_cache = {}

    def tick(self):
        """One sorting iteration: move a fraction of neurons toward neighbors."""
        n_moves = int(self.move_fraction * self.n)
        for _ in range(n_moves):
            x = random.randint(0, self.width - 1)
            y = random.randint(0, self.height - 1)
            self._move_toward_neighbors(x, y)

    def _get_top_k(self, neuron_id):
        """Return indices of the K neurons with highest affinity to neuron_id."""
        if neuron_id not in self._k_cache:
            # argpartition is O(n) vs O(n log n) for full sort
            kj = np.argpartition(self.weight_matrix[neuron_id], -self.k - 1)
            self._k_cache[neuron_id] = kj[-self.k - 1:-1]
        return self._k_cache[neuron_id]

    def _move_toward_neighbors(self, x, y):
        """Move the neuron at (x, y) one step toward its neighbor centroid."""
        neuron_id = self.neurons_matrix[y][x]
        top_k = self._get_top_k(neuron_id)

        # Centroid of top-K neighbors' current positions
        positions = self.coordinates[top_k]
        avg_x = np.mean(positions[:, 0])
        avg_y = np.mean(positions[:, 1])

        dx = avg_x - x
        dy = avg_y - y
        if dx == 0 and dy == 0:
            return

        # Normalize to one-cell step
        mag = (dx ** 2 + dy ** 2) ** 0.5
        new_x = x + round(dx / mag)
        new_y = y + round(dy / mag)

        # Clamp to grid bounds
        new_x = max(0, min(self.width - 1, new_x))
        new_y = max(0, min(self.height - 1, new_y))

        self._swap(x, y, new_x, new_y)

    def _swap(self, x1, y1, x2, y2):
        """Swap the neurons at two grid positions."""
        i = self.neurons_matrix[y1][x1]
        j = self.neurons_matrix[y2][x2]
        self.neurons_matrix[y1][x1] = j
        self.neurons_matrix[y2][x2] = i
        self.coordinates[i] = [x2, y2]
        self.coordinates[j] = [x1, y1]

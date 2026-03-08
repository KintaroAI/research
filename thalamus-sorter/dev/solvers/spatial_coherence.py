"""Spatial coherence solver: minimize total variation loss through
gradient descent on a permutation weight matrix. The only differentiable
approach — treats sorting as continuous optimization."""

import numpy as np


def total_variation_loss(Y, width, height):
    """Sum of absolute horizontal and vertical differences.
    Measures how spatially smooth the output is."""
    Y2d = Y.reshape(height, width)
    h_diff = np.abs(Y2d[:-1, :] - Y2d[1:, :]).sum()
    v_diff = np.abs(Y2d[:, :-1] - Y2d[:, 1:]).sum()
    return h_diff + v_diff


def tv_gradient(Y, W1, width, height):
    """Gradient of total variation loss with respect to input weights W1."""
    Y2d = Y.reshape(height, width)
    h_diff = Y2d[:-1, :] - Y2d[1:, :]
    v_diff = Y2d[:, :-1] - Y2d[:, 1:]

    dLdY = np.zeros_like(Y2d)
    dLdY[:-1, :] += h_diff
    dLdY[1:, :] -= h_diff
    dLdY[:, :-1] += v_diff
    dLdY[:, 1:] -= v_diff

    dLdW1 = np.dot(np.ravel(dLdY), W1.T)
    return dLdW1


class SpatialCoherence:
    """Optimize a permutation matrix to minimize total variation loss
    of the output signal on a 2D grid."""

    def __init__(self, width, height, lr=0.001):
        self.width = width
        self.height = height
        self.n = width * height
        self.lr = lr

        # Start with shuffled identity (permutation matrix)
        self.W1 = np.eye(self.n)
        np.random.shuffle(self.W1)

    def optimize(self, input_signal, epochs=1000):
        """Run gradient descent on TV loss for the given input signal.
        Returns the optimized output."""
        for _ in range(epochs):
            Y = np.dot(self.W1, input_signal)
            grad = tv_gradient(Y, self.W1, self.width, self.height)
            self.W1 -= self.lr * grad
        return np.dot(self.W1, input_signal)

    @property
    def loss(self):
        return None  # Computed during optimize()

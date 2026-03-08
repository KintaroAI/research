"""Pearson correlation functions for temporal neuron signals."""

import numpy as np


def pearson(x, y):
    """Pearson correlation between two 1D arrays."""
    n = len(x)
    mean_x, mean_y = np.mean(x), np.mean(y)
    cov = np.sum((x - mean_x) * (y - mean_y)) / n
    std_x = np.sqrt(np.sum((x - mean_x) ** 2) / n)
    std_y = np.sqrt(np.sum((y - mean_y) ** 2) / n)
    if std_x * std_y == 0:
        return 0.0
    return cov / (std_x * std_y)


def temporal_correlation(temporal_a, temporal_b, i, j):
    """Pearson correlation between neuron i's trace in temporal_a and
    neuron j's trace in temporal_b. temporal arrays are (n_neurons, n_steps)."""
    if i == j and temporal_a is temporal_b:
        return 1.0
    return pearson(temporal_a[i, :], temporal_b[j, :])


def matrix_correlations(X, Y):
    """Pairwise correlation matrix between all neurons in X and Y.
    X, Y are (rows, cols, n_steps). Returns (rows, cols) correlation matrix."""
    n = X.shape[2]
    C = np.zeros((X.shape[0], Y.shape[0]))
    for i in range(X.shape[0]):
        for j in range(Y.shape[0]):
            C[i, j] = pearson(X[i, j, :], Y[i, j, :])
    return C

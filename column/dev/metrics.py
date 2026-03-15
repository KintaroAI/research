"""Separation quality metrics (SQM) for competitive categorization cells.

Three axes:
  Coverage  — are all output units alive?
  Selectivity — does each unit respond to a distinct pattern?
  Separation — do learned categories match true structure?

All functions take numpy arrays. Use compute_sqm() for the full report.
"""

import numpy as np


# ── Coverage ──────────────────────────────────────────────────────────

def winner_entropy(winners, n_outputs):
    """Shannon entropy of winner distribution, normalized to [0, 1].

    1.0 = perfectly uniform usage, 0.0 = total collapse.
    """
    counts = np.bincount(winners, minlength=n_outputs).astype(float)
    counts = counts[counts > 0]
    p = counts / counts.sum()
    H = -(p * np.log2(p)).sum()
    H_max = np.log2(n_outputs)
    return H / H_max if H_max > 0 else 0.0


def usage_gini(winners, n_outputs):
    """Gini coefficient of winner counts. 0 = uniform, 1 = total collapse."""
    counts = np.bincount(winners, minlength=n_outputs).astype(float)
    counts.sort()
    n = len(counts)
    index = np.arange(1, n + 1)
    return (2 * (index * counts).sum() / (n * counts.sum()) - (n + 1) / n) if counts.sum() > 0 else 0.0


# ── Selectivity ───────────────────────────────────────────────────────

def confidence_gap(probs):
    """Mean difference between top-1 and top-2 output probabilities.

    probs: (N, m) array of output probabilities.
    High gap = sharp decisions, low gap = ambiguous.
    """
    sorted_p = np.sort(probs, axis=1)[:, ::-1]
    gaps = sorted_p[:, 0] - sorted_p[:, 1]
    return float(gaps.mean())


def prototype_spread(prototypes):
    """Mean pairwise cosine distance between prototypes.

    prototypes: (m, n) array. Returns value in [0, 2].
    Higher = better spread. Compare to expected distance for random unit
    vectors: 1.0 (orthogonal on average in high dimensions).
    """
    norms = np.linalg.norm(prototypes, axis=1, keepdims=True)
    normed = prototypes / np.maximum(norms, 1e-8)
    cosine_sim = normed @ normed.T
    m = len(prototypes)
    # Extract upper triangle (exclude diagonal)
    mask = np.triu(np.ones((m, m), dtype=bool), k=1)
    distances = 1.0 - cosine_sim[mask]
    return float(distances.mean()) if len(distances) > 0 else 0.0


# ── Separation (supervised) ───────────────────────────────────────────

def purity(winners, labels, n_outputs):
    """Per-unit purity: fraction of inputs from dominant cluster, averaged."""
    n_clusters = labels.max() + 1
    scores = []
    for w in range(n_outputs):
        mask = winners == w
        if mask.sum() == 0:
            continue
        counts = np.bincount(labels[mask], minlength=n_clusters)
        scores.append(counts.max() / mask.sum())
    return float(np.mean(scores)) if scores else 0.0


def normalized_mutual_info(winners, labels):
    """Normalized mutual information between winners and labels.

    NMI = 2 * I(W;L) / (H(W) + H(L)). Range [0, 1].
    Unlike purity, penalizes many-to-one mappings.
    """
    N = len(winners)
    if N == 0:
        return 0.0

    w_classes = np.unique(winners)
    l_classes = np.unique(labels)

    # Marginal entropies
    def entropy(x, classes):
        counts = np.array([np.sum(x == c) for c in classes], dtype=float)
        p = counts / N
        p = p[p > 0]
        return -(p * np.log(p)).sum()

    H_w = entropy(winners, w_classes)
    H_l = entropy(labels, l_classes)

    if H_w == 0 or H_l == 0:
        return 0.0

    # Mutual information
    MI = 0.0
    for w in w_classes:
        for l in l_classes:
            n_wl = np.sum((winners == w) & (labels == l))
            if n_wl == 0:
                continue
            n_w = np.sum(winners == w)
            n_l = np.sum(labels == l)
            MI += (n_wl / N) * np.log((N * n_wl) / (n_w * n_l))

    return float(2 * MI / (H_w + H_l))


def confusion_matrix(winners, labels, n_outputs):
    """Co-occurrence matrix: (n_clusters, n_outputs).

    Entry [c, w] = number of times cluster c was assigned to winner w.
    """
    n_clusters = labels.max() + 1
    mat = np.zeros((n_clusters, n_outputs), dtype=int)
    for c in range(n_clusters):
        for w in range(n_outputs):
            mat[c, w] = np.sum((labels == c) & (winners == w))
    return mat


# ── Full report ───────────────────────────────────────────────────────

def compute_sqm(winners, probs, prototypes, n_outputs, labels=None):
    """Compute all separation quality metrics.

    Args:
        winners: (N,) int array of winner indices
        probs: (N, m) float array of output probabilities
        prototypes: (m, n) float array of prototype vectors
        n_outputs: number of output units
        labels: (N,) int array of ground-truth cluster labels (optional)

    Returns:
        dict with all metrics
    """
    results = {
        'winner_entropy': winner_entropy(winners, n_outputs),
        'usage_gini': usage_gini(winners, n_outputs),
        'confidence_gap': confidence_gap(probs),
        'prototype_spread': prototype_spread(prototypes),
    }

    if labels is not None:
        results['purity'] = purity(winners, labels, n_outputs)
        results['nmi'] = normalized_mutual_info(winners, labels)

    return results


def format_sqm(sqm):
    """One-line summary string for logging."""
    parts = [
        f"entropy={sqm['winner_entropy']:.3f}",
        f"gini={sqm['usage_gini']:.3f}",
        f"conf_gap={sqm['confidence_gap']:.3f}",
        f"spread={sqm['prototype_spread']:.3f}",
    ]
    if 'nmi' in sqm:
        parts.append(f"nmi={sqm['nmi']:.3f}")
    if 'purity' in sqm:
        parts.append(f"purity={sqm['purity']:.3f}")
    return "  ".join(parts)

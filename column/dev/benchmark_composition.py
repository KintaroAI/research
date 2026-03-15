"""Compositional logic benchmark — multi-cell classification.

Tests whether three cells with the right inter-cell wiring can learn
relational tasks that prototype cells cannot solve alone.

Key insight: the WIRING OPERATION between cells determines what's solvable.
- Outer product / concatenation → fails (exp 00006)
- Absolute difference → solves same/different (comparison)
- Circular convolution → solves sum mod 4 (modular arithmetic)

Architecture:
  input_a (8D) → Cell A (4 outputs) ─┐
                                      ├─ [wiring op] → Cell C → category
  input_b (8D) → Cell B (4 outputs) ─┘

Wiring operations:
  outer:     p_a ⊗ p_b                → 16D  (pairwise products)
  concat:    p_a || p_b               → 8D   (concatenation)
  diff:      |p_a - p_b|              → 4D   (comparison)
  circ_conv: circular_conv(p_a, p_b)  → 4D   (modular addition of distributions)

Usage:
    python benchmark_composition.py [-o output_dir]
"""

import argparse
import json
import os

import numpy as np
import torch

from column import SoftWTACell
from metrics import compute_sqm, format_sqm, normalized_mutual_info


def make_paired_data(n_samples, n_values=4, n_dims=8, noise=0.3, seed=42):
    """Generate paired number inputs with multiple task labels."""
    rng = np.random.default_rng(seed)

    centers = rng.standard_normal((n_values, n_dims)).astype(np.float32)
    centers = centers / np.linalg.norm(centers, axis=1, keepdims=True)

    # Balanced sampling: 50% same, 50% different
    a_vals = rng.integers(n_values, size=n_samples)
    b_vals = np.empty(n_samples, dtype=int)
    for i in range(n_samples):
        if rng.random() < 0.5:
            b_vals[i] = a_vals[i]
        else:
            b_vals[i] = (a_vals[i] + rng.integers(1, n_values)) % n_values

    a_data = np.array([centers[a] + rng.standard_normal(n_dims).astype(np.float32) * noise
                       for a in a_vals])
    b_data = np.array([centers[b] + rng.standard_normal(n_dims).astype(np.float32) * noise
                       for b in b_vals])

    labels = {
        'same_diff': (a_vals == b_vals).astype(int),
        'proximity': (np.minimum(np.abs(a_vals - b_vals),
                                 n_values - np.abs(a_vals - b_vals)) <= 1).astype(int),
        'sum_mod4': (a_vals + b_vals) % n_values,
    }

    return a_data, b_data, a_vals, b_vals, labels, centers


def circular_conv(pa, pb):
    """Circular convolution: conv[k] = sum_j pa[j] * pb[(k-j) mod n].

    Computes the probability distribution over (a+b) mod n.
    """
    n = len(pa)
    result = torch.zeros(n)
    for k in range(n):
        for j in range(n):
            result[k] += pa[j] * pb[(k - j) % n]
    return result


def compare_stats(pa, pb):
    """Comparison statistics — value-independent same/different signal.

    Computes multiple scalar statistics that depend on the RELATIONSHIP
    between pa and pb, not on which specific value they represent.
    """
    cos_sim = (pa * pb).sum()
    l2_dist = ((pa - pb) ** 2).sum()
    max_prod = (pa * pb).max()
    entropy_diff = torch.abs(-(pa * torch.log(pa + 1e-8)).sum()
                             - (-(pb * torch.log(pb + 1e-8)).sum()))
    return torch.stack([cos_sim, l2_dist, max_prod, entropy_diff])


def combine_outputs(pa, pb, method):
    """Apply a wiring operation to combine Cell A and B outputs."""
    if method == 'outer':
        return (pa.unsqueeze(1) * pb.unsqueeze(0)).flatten()
    elif method == 'concat':
        return torch.cat([pa, pb])
    elif method == 'diff':
        return torch.abs(pa - pb)
    elif method == 'circ_conv':
        return circular_conv(pa, pb)
    elif method == 'compare':
        return compare_stats(pa, pb)
    else:
        raise ValueError(f"Unknown combine method: {method}")


def combine_dim(n_values, method):
    """Output dimensionality for each wiring operation."""
    if method == 'outer':
        return n_values * n_values
    elif method == 'concat':
        return n_values * 2
    elif method in ('diff', 'circ_conv'):
        return n_values
    elif method == 'compare':
        return 4
    raise ValueError(f"Unknown method: {method}")


def run_pipeline(a_data, b_data, a_vals, b_vals, labels, task,
                 combine='diff', n_dims=8):
    """Run the three-cell pipeline with specified wiring operation."""
    n_samples = len(a_vals)
    n_values = 4
    task_labels = labels[task]
    n_categories = len(np.unique(task_labels))

    cell_a = SoftWTACell(n_dims, n_values, temperature=0.3, lr=0.05)
    cell_b = SoftWTACell(n_dims, n_values, temperature=0.3, lr=0.05)

    c_inputs = combine_dim(n_values, combine)
    cell_c = SoftWTACell(c_inputs, n_categories, temperature=0.3, lr=0.05)

    # Phase 1: pretrain Cell A and B
    pretrain = n_samples // 3
    for i in range(pretrain):
        xa = torch.from_numpy(a_data[i])
        xb = torch.from_numpy(b_data[i])
        pa = cell_a.forward(xa)
        cell_a.update(xa, pa)
        pb = cell_b.forward(xb)
        cell_b.update(xb, pb)

    # Phase 2: train all three
    winners, probs = [], []
    for i in range(pretrain, n_samples):
        xa = torch.from_numpy(a_data[i])
        xb = torch.from_numpy(b_data[i])

        pa = cell_a.forward(xa)
        cell_a.update(xa, pa)
        pb = cell_b.forward(xb)
        cell_b.update(xb, pb)

        combined = combine_outputs(pa.detach(), pb.detach(), combine)
        pc = cell_c.forward(combined)
        wc, _ = cell_c.update(combined, pc)
        winners.append(wc)
        probs.append(pc.detach().numpy())

    w_arr = np.array(winners)
    p_arr = np.array(probs)
    eval_labels = task_labels[pretrain:]
    sqm = compute_sqm(w_arr, p_arr, cell_c.prototypes.numpy(),
                      n_categories, labels=eval_labels)
    return sqm


def run_single_cell(a_data, b_data, labels, task, n_dims=8):
    """Single cell on concatenated raw input (baseline)."""
    n_samples = len(a_data)
    task_labels = labels[task]
    n_categories = len(np.unique(task_labels))

    cell = SoftWTACell(n_dims * 2, n_categories, temperature=0.3, lr=0.05)

    winners, probs = [], []
    for i in range(n_samples):
        x = torch.from_numpy(np.concatenate([a_data[i], b_data[i]]))
        p = cell.forward(x)
        w, _ = cell.update(x, p)
        winners.append(w)
        probs.append(p.detach().numpy())

    w_arr = np.array(winners)
    p_arr = np.array(probs)
    return compute_sqm(w_arr, p_arr, cell.prototypes.numpy(),
                       n_categories, labels=task_labels)


def run_task(name, a_data, b_data, a_vals, b_vals, labels, methods):
    """Run all architectures on a single task."""
    results = {}
    for method in methods:
        results[method] = run_pipeline(
            a_data, b_data, a_vals, b_vals, labels, name, combine=method)
    results['single'] = run_single_cell(a_data, b_data, labels, name)
    return results


def print_task(results):
    """Print results for one task."""
    scenarios = list(results.keys())
    col_w = 14
    keys = ['nmi', 'purity', 'consistency', 'confidence_gap']

    header = f"  {'metric':<20}" + "".join(f"{s:>{col_w}}" for s in scenarios)
    print(header)
    print("  " + "-" * (len(header) - 2))
    for k in keys:
        row = f"  {k:<20}"
        for s in scenarios:
            v = results[s].get(k, '')
            if isinstance(v, float):
                row += f"{v:>{col_w}.3f}"
            else:
                row += f"{'—':>{col_w}}"
        print(row)


def main():
    parser = argparse.ArgumentParser(description='Compositional logic benchmark')
    parser.add_argument('-o', '--output', type=str, default=None)
    parser.add_argument('--samples', type=int, default=15000)
    parser.add_argument('--noise', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    print("=" * 74)
    print("Compositional Logic Benchmark — Wiring Operations Compared")
    print("=" * 74)
    print(f"Samples: {args.samples}, 4 values as 8D clusters, noise={args.noise}")
    print(f"Wiring: compare, circ_conv, diff, outer, concat, single")

    a_data, b_data, a_vals, b_vals, labels, centers = \
        make_paired_data(args.samples, noise=args.noise, seed=args.seed)

    all_methods = ['compare', 'circ_conv', 'diff', 'outer', 'concat']
    all_results = {}

    for task, desc in [('same_diff', 'Same/Different (a == b?)'),
                       ('proximity', 'Proximity (|a-b| ≤ 1?)'),
                       ('sum_mod4', 'Sum mod 4 ((a+b) % 4)')]:
        print(f"\n{'─' * 74}")
        print(f"Task: {desc}")
        print(f"{'─' * 74}")
        results = run_task(task, a_data, b_data, a_vals, b_vals, labels,
                           all_methods)
        all_results[task] = results
        print_task(results)

    # NMI Summary
    all_archs = all_methods + ['single']
    print(f"\n{'=' * 74}")
    print("NMI Summary")
    print(f"{'=' * 74}")
    col_w = 14
    header = f"  {'task':<16}" + "".join(f"{a:>{col_w}}" for a in all_archs)
    print(header)
    print("  " + "-" * (len(header) - 2))
    for task in ['same_diff', 'proximity', 'sum_mod4']:
        row = f"  {task:<16}"
        for arch in all_archs:
            row += f"{all_results[task][arch]['nmi']:>{col_w}.3f}"
        print(row)

    if args.output:
        os.makedirs(args.output, exist_ok=True)
        with open(os.path.join(args.output, 'benchmark_composition.json'), 'w') as f:
            json.dump(all_results, f, indent=2, default=lambda x: x.tolist()
                      if hasattr(x, 'tolist') else x)
        print(f"\nSaved to {args.output}/benchmark_composition.json")


if __name__ == '__main__':
    main()

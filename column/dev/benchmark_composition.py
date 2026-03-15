"""Compositional logic benchmark — multi-cell classification.

Tests whether three cells with multiplicative interaction can learn
relationships between two input streams that a single cell cannot.

Architecture:
  input_a (8D) → Cell A (4 outputs) ─┐
                                      ├─ outer product (16D) → Cell C → category
  input_b (8D) → Cell B (4 outputs) ─┘

Tasks:
  1. Same/different: is a == b?  (2 categories)
  2. Proximity: is |a-b| mod 4 ≤ 1?  (2 categories)
  3. Sum mod 4: what is (a+b) mod 4?  (4 categories — hard negative result)

Why single cell fails: all categories have identical centroids in the
concatenated input space (balanced residue classes). No prototype can
separate them.

Why outer product helps: the product p_a ⊗ p_b creates features that
encode pairwise interactions. Same-class pairs activate diagonal
positions, different-class pairs activate off-diagonal positions —
giving distinct centroids.

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
    """Generate paired number inputs with multiple task labels.

    Each number 0-3 is encoded as a noisy vector near a cluster center.
    Returns data, number values, and labels for each task.
    """
    rng = np.random.default_rng(seed)

    centers = rng.standard_normal((n_values, n_dims)).astype(np.float32)
    centers = centers / np.linalg.norm(centers, axis=1, keepdims=True)

    # Balanced sampling: 50% same, 50% different pairs
    a_vals = rng.integers(n_values, size=n_samples)
    b_vals = np.empty(n_samples, dtype=int)
    for i in range(n_samples):
        if rng.random() < 0.5:
            b_vals[i] = a_vals[i]  # same
        else:
            b_vals[i] = (a_vals[i] + rng.integers(1, n_values)) % n_values  # different

    a_data = np.array([centers[a] + rng.standard_normal(n_dims).astype(np.float32) * noise
                       for a in a_vals])
    b_data = np.array([centers[b] + rng.standard_normal(n_dims).astype(np.float32) * noise
                       for b in b_vals])

    labels = {
        'same_diff': (a_vals == b_vals).astype(int),  # 0=different, 1=same
        'proximity': (np.minimum(np.abs(a_vals - b_vals),
                                 n_values - np.abs(a_vals - b_vals)) <= 1).astype(int),
        'sum_mod4': (a_vals + b_vals) % n_values,
    }

    return a_data, b_data, a_vals, b_vals, labels, centers


def run_pipeline(a_data, b_data, a_vals, b_vals, labels, task,
                 combine='outer', n_dims=8):
    """Run the three-cell pipeline with specified combination method.

    combine: 'outer' (multiplicative) or 'concat' (additive)
    """
    n_samples = len(a_vals)
    n_values = 4
    task_labels = labels[task]
    n_categories = len(np.unique(task_labels))

    cell_a = SoftWTACell(n_dims, n_values, temperature=0.3, lr=0.05)
    cell_b = SoftWTACell(n_dims, n_values, temperature=0.3, lr=0.05)

    if combine == 'outer':
        c_inputs = n_values * n_values  # 16
    else:
        c_inputs = n_values * 2  # 8

    cell_c = SoftWTACell(c_inputs, n_categories, temperature=0.3, lr=0.05)

    # Phase 1: train Cell A and B alone (Cell C sees nothing)
    pretrain = n_samples // 3
    for i in range(pretrain):
        xa = torch.from_numpy(a_data[i])
        xb = torch.from_numpy(b_data[i])
        pa = cell_a.forward(xa)
        cell_a.update(xa, pa)
        pb = cell_b.forward(xb)
        cell_b.update(xb, pb)

    # Phase 2: train all three cells
    winners, probs = [], []
    for i in range(pretrain, n_samples):
        xa = torch.from_numpy(a_data[i])
        xb = torch.from_numpy(b_data[i])

        pa = cell_a.forward(xa)
        cell_a.update(xa, pa)
        pb = cell_b.forward(xb)
        cell_b.update(xb, pb)

        if combine == 'outer':
            combined = (pa.unsqueeze(1) * pb.unsqueeze(0)).flatten()
        else:
            combined = torch.cat([pa, pb])

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
    """Single cell on concatenated raw input."""
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


def run_task(name, a_data, b_data, a_vals, b_vals, labels):
    """Run all architectures on a single task."""
    results = {}

    results['3cell_outer'] = run_pipeline(
        a_data, b_data, a_vals, b_vals, labels, name, combine='outer')

    results['3cell_concat'] = run_pipeline(
        a_data, b_data, a_vals, b_vals, labels, name, combine='concat')

    results['single'] = run_single_cell(a_data, b_data, labels, name)

    return results


def print_task(task_name, results):
    """Print results for one task."""
    scenarios = list(results.keys())
    col_w = 16
    keys = ['nmi', 'purity', 'consistency', 'confidence_gap', 'winner_entropy']

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
    parser.add_argument('--samples', type=int, default=10000)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    print("=" * 68)
    print("Compositional Logic Benchmark")
    print("=" * 68)
    print(f"Samples: {args.samples}, 4 values encoded as 8D clusters, noise=0.3")
    print(f"Architectures: 3cell_outer (A⊗B→C), 3cell_concat (A||B→C), single (raw)")

    a_data, b_data, a_vals, b_vals, labels, centers = \
        make_paired_data(args.samples, seed=args.seed)

    all_results = {}

    for task, desc in [('same_diff', 'Same/Different (a == b?)'),
                       ('proximity', 'Proximity (|a-b| ≤ 1?)'),
                       ('sum_mod4', 'Sum mod 4 ((a+b) % 4)')]:
        print(f"\n{'─' * 68}")
        print(f"Task: {desc}")
        print(f"{'─' * 68}")
        results = run_task(task, a_data, b_data, a_vals, b_vals, labels)
        all_results[task] = results
        print_task(task, results)

    # Summary
    print(f"\n{'=' * 68}")
    print("NMI Summary")
    print(f"{'=' * 68}")
    print(f"  {'task':<20}{'3cell_outer':>16}{'3cell_concat':>16}{'single':>16}")
    print("  " + "-" * 64)
    for task in ['same_diff', 'proximity', 'sum_mod4']:
        row = f"  {task:<20}"
        for arch in ['3cell_outer', '3cell_concat', 'single']:
            row += f"{all_results[task][arch]['nmi']:>16.3f}"
        print(row)

    if args.output:
        os.makedirs(args.output, exist_ok=True)
        with open(os.path.join(args.output, 'benchmark_composition.json'), 'w') as f:
            json.dump(all_results, f, indent=2, default=lambda x: x.tolist()
                      if hasattr(x, 'tolist') else x)
        print(f"\nSaved to {args.output}/benchmark_composition.json")


if __name__ == '__main__':
    main()

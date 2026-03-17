"""Recurrent ring benchmark — attractor dynamics from looped cells.

Cells connected in a ring: each cell's output feeds into the next.
After multiple cycles, the pattern should stabilize into an attractor.
Different inputs converge to different attractors — the attractor IS
the learned category.

Tests whether recurrent processing finds categories that a single
feedforward pass cannot, particularly for non-convex or overlapping clusters.

Architecture:
  x → Cell A → Cell B → Cell C → Cell A → ... (iterate until stable)

Usage:
    python benchmark_recurrent.py [-o output_dir]
"""

import argparse
import json
import os

import numpy as np
import torch

from column import SoftWTACell
from main import make_clustered_data
from metrics import normalized_mutual_info


class RecurrentRing:
    """Ring of cells where each cell's output feeds into the next."""

    def __init__(self, n_cells, n_inputs, n_outputs, **cell_kwargs):
        self.n_cells = n_cells
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        # First cell takes raw input, subsequent cells take previous cell's output
        self.cells = []
        self.cells.append(SoftWTACell(n_inputs, n_outputs, **cell_kwargs))
        for i in range(1, n_cells):
            self.cells.append(SoftWTACell(n_outputs, n_outputs, **cell_kwargs))

    def forward(self, x, n_cycles=3, update=True):
        """Run input through the ring for n_cycles.

        Returns the final cell's output probabilities and winner after settling.
        """
        # First pass: raw input → Cell 0
        probs = self.cells[0].forward(x)
        if update:
            self.cells[0].update(x, probs)

        # Remaining cells in first cycle
        for i in range(1, self.n_cells):
            probs = self.cells[i].forward(probs)
            if update:
                self.cells[i].update(probs, probs)

        # Additional cycles: feedback through the ring
        # Cell 0 gets concatenated [raw_input, last_cell_output] on recurrent passes
        for cycle in range(1, n_cycles):
            # Cell 0: raw input weighted by feedback — use winner's prototype
            # as the feedback signal projected back to input space
            feedback = self.cells[0].prototypes[probs.argmax().item()]
            combined = x * 0.7 + feedback * 0.3
            probs = self.cells[0].forward(combined)
            if update:
                self.cells[0].update(combined, probs)

            for i in range(1, self.n_cells):
                probs = self.cells[i].forward(probs)
                if update:
                    self.cells[i].update(probs, probs)

        winner = probs.argmax().item()
        return winner, probs


def make_nonconvex_data(n_samples, n_inputs=16, seed=42):
    """Generate data with non-convex cluster structure.

    4 categories, each made of 2 sub-clusters placed on opposite sides.
    A single prototype can't capture both sub-clusters — the centroid
    falls between them where no data exists.
    """
    rng = np.random.default_rng(seed)
    n_categories = 4
    n_subclusters = 2

    # Each category has 2 sub-cluster centers
    centers = []
    for cat in range(n_categories):
        base_dir = rng.standard_normal(n_inputs).astype(np.float32)
        base_dir = base_dir / np.linalg.norm(base_dir)
        # Two sub-clusters: offset in a perpendicular direction
        perp = rng.standard_normal(n_inputs).astype(np.float32)
        perp = perp - np.dot(perp, base_dir) * base_dir  # orthogonalize
        perp = perp / np.linalg.norm(perp)
        centers.append(base_dir + perp * 0.8)
        centers.append(base_dir - perp * 0.8)

    centers = np.array(centers, dtype=np.float32)

    data = []
    labels = []
    for i in range(n_samples):
        cat = rng.integers(n_categories)
        sub = rng.integers(n_subclusters)
        center_idx = cat * n_subclusters + sub
        x = centers[center_idx] + rng.standard_normal(n_inputs).astype(np.float32) * 0.15
        data.append(x)
        labels.append(cat)

    return np.array(data, dtype=np.float32), np.array(labels), centers


def run_ring(data, labels, n_cells, n_outputs, n_cycles, **cell_kwargs):
    """Train and evaluate a recurrent ring."""
    n_inputs = data.shape[1]
    n_frames = len(data)

    ring = RecurrentRing(n_cells, n_inputs, n_outputs, **cell_kwargs)

    winners = []
    for i in range(n_frames):
        x = torch.from_numpy(data[i])
        w, _ = ring.forward(x, n_cycles=n_cycles, update=True)
        winners.append(w)

    eval_start = n_frames // 2
    w_arr = np.array(winners[eval_start:])
    l_arr = labels[eval_start:]
    return normalized_mutual_info(w_arr, l_arr)


def run_single(data, labels, n_outputs, **cell_kwargs):
    """Baseline: single feedforward cell."""
    n_frames = len(data)
    cell = SoftWTACell(data.shape[1], n_outputs, **cell_kwargs)

    winners = []
    for i in range(n_frames):
        x = torch.from_numpy(data[i])
        p = cell.forward(x)
        w, _ = cell.update(x, p)
        winners.append(w)

    eval_start = n_frames // 2
    return normalized_mutual_info(np.array(winners[eval_start:]), labels[eval_start:])


def main():
    parser = argparse.ArgumentParser(description='Recurrent ring benchmark')
    parser.add_argument('-o', '--output', type=str, default=None)
    parser.add_argument('--frames', type=int, default=20000)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    cell_kwargs = dict(temperature=0.3, lr=0.05)
    results = {}

    print("=" * 60)
    print("Recurrent Ring Benchmark")
    print("=" * 60)

    # Test 1: Standard convex clusters
    print(f"\n[Test 1] Standard 8 convex clusters, 16D")
    data_c, labels_c, _ = make_clustered_data(8, 16, args.frames, seed=args.seed)

    print(f"  {'architecture':<35} {'NMI':>8}")
    print("  " + "-" * 43)

    nmi = run_single(data_c, labels_c, 8, **cell_kwargs)
    results['convex_single'] = nmi
    print(f"  {'single cell (m=8)':<35} {nmi:>8.3f}")

    for n_cells in [2, 3]:
        for n_cycles in [1, 3, 5]:
            nmi = run_ring(data_c, labels_c, n_cells, 8, n_cycles, **cell_kwargs)
            name = f"ring {n_cells} cells, {n_cycles} cycles"
            results[f'convex_ring_{n_cells}c_{n_cycles}cy'] = nmi
            print(f"  {name:<35} {nmi:>8.3f}")

    # Test 2: Non-convex clusters (2 sub-clusters per category)
    print(f"\n[Test 2] 4 non-convex categories (2 sub-clusters each), 16D")
    data_nc, labels_nc, _ = make_nonconvex_data(args.frames, seed=args.seed)

    print(f"  {'architecture':<35} {'NMI':>8}")
    print("  " + "-" * 43)

    for m in [4, 8]:
        nmi = run_single(data_nc, labels_nc, m, **cell_kwargs)
        results[f'nonconvex_single_m{m}'] = nmi
        name = f"single cell (m={m})"
        print(f"  {name:<35} {nmi:>8.3f}")

    for n_cells in [2, 3]:
        for n_cycles in [1, 3, 5]:
            nmi = run_ring(data_nc, labels_nc, n_cells, 4, n_cycles, **cell_kwargs)
            name = f"ring {n_cells} cells, {n_cycles} cycles (m=4)"
            results[f'nonconvex_ring_{n_cells}c_{n_cycles}cy'] = nmi
            print(f"  {name:<35} {nmi:>8.3f}")

    # Also try ring with m=8 (can learn sub-clusters, then merge)
    for n_cycles in [3, 5]:
        nmi = run_ring(data_nc, labels_nc, 3, 8, n_cycles, **cell_kwargs)
        name = f"ring 3 cells, {n_cycles} cycles (m=8)"
        results[f'nonconvex_ring_3c_{n_cycles}cy_m8'] = nmi
        print(f"  {name:<35} {nmi:>8.3f}")

    if args.output:
        os.makedirs(args.output, exist_ok=True)
        with open(os.path.join(args.output, 'benchmark_recurrent.json'), 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved to {args.output}/benchmark_recurrent.json")


if __name__ == '__main__':
    main()

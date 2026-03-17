"""Receptive field tiling benchmark — local features + combinations.

Layer 1: multiple cells, each seeing a different subset of inputs (like V1
receptive fields). Layer 2: one cell sees all Layer 1 outputs concatenated,
learning combinations of local features.

Tests whether this detects feature conjunctions that a single cell on the
full input cannot — e.g., "feature A in channels 0-3 AND feature B in
channels 4-7" requires seeing both local features and their co-occurrence.

Usage:
    python benchmark_receptive.py [-o output_dir]
"""

import argparse
import json
import os

import numpy as np
import torch

from column import SoftWTACell
from metrics import normalized_mutual_info


def make_conjunction_data(n_samples, n_inputs=16, n_features=4,
                          n_values=2, noise=0.2, seed=42):
    """Generate data where the category is defined by a conjunction of local features.

    Input is split into n_features groups. Each group independently takes one of
    n_values patterns. The category label is defined by a specific combination rule.

    With n_features=4, n_values=2: each group is 'A' or 'B' pattern.
    Category = XOR of first two features (requires seeing both).
    """
    rng = np.random.default_rng(seed)
    group_size = n_inputs // n_features

    # Random pattern centers per group per value
    patterns = {}
    for f in range(n_features):
        for v in range(n_values):
            p = rng.standard_normal(group_size).astype(np.float32)
            p = p / np.linalg.norm(p)
            patterns[(f, v)] = p

    data = []
    feature_vals = []  # per-feature labels
    conj_labels = []   # conjunction category

    for i in range(n_samples):
        x = np.zeros(n_inputs, dtype=np.float32)
        vals = []
        for f in range(n_features):
            v = rng.integers(n_values)
            vals.append(v)
            start = f * group_size
            end = start + group_size
            x[start:end] = patterns[(f, v)] + rng.standard_normal(group_size).astype(np.float32) * noise

        # Conjunction label: XOR of feature 0 and feature 1
        # (requires seeing both local features to determine)
        conj = vals[0] ^ vals[1]

        # Also make a 4-way label using features 0 and 1
        pair_label = vals[0] * n_values + vals[1]

        data.append(x)
        feature_vals.append(vals)
        conj_labels.append(pair_label)  # 4 categories from 2x2 combinations

    return (np.array(data, dtype=np.float32),
            np.array(feature_vals),
            np.array(conj_labels))


class ReceptiveFieldNet:
    """Layer 1: local cells on input subsets. Layer 2: combination cell."""

    def __init__(self, n_inputs, n_features, n_local_outputs, n_final_outputs,
                 **cell_kwargs):
        self.n_features = n_features
        self.group_size = n_inputs // n_features
        self.n_local = n_local_outputs

        # Layer 1: one cell per receptive field
        self.local_cells = []
        for f in range(n_features):
            cell = SoftWTACell(self.group_size, n_local_outputs, **cell_kwargs)
            self.local_cells.append(cell)

        # Layer 2: sees all local outputs concatenated
        l2_inputs = n_features * n_local_outputs
        self.combo_cell = SoftWTACell(l2_inputs, n_final_outputs, **cell_kwargs)

    def forward_update(self, x):
        """Forward + update through both layers."""
        local_outputs = []
        for f, cell in enumerate(self.local_cells):
            start = f * self.group_size
            end = start + self.group_size
            x_local = x[start:end]
            p = cell.forward(x_local)
            cell.update(x_local, p)
            local_outputs.append(p.detach())

        # Concatenate local outputs → Layer 2
        combined = torch.cat(local_outputs)
        p2 = self.combo_cell.forward(combined)
        w2, _ = self.combo_cell.update(combined, p2)
        return w2

    def forward_only(self, x):
        """Forward only, no update."""
        local_outputs = []
        for f, cell in enumerate(self.local_cells):
            start = f * self.group_size
            end = start + self.group_size
            x_local = x[start:end]
            p = cell.forward(x_local)
            local_outputs.append(p.detach())

        combined = torch.cat(local_outputs)
        p2 = self.combo_cell.forward(combined)
        return p2.argmax().item()


def run_receptive(data, labels, n_features, n_local, n_final, **cell_kwargs):
    """Train and evaluate a receptive field network."""
    n_inputs = data.shape[1]
    net = ReceptiveFieldNet(n_inputs, n_features, n_local, n_final, **cell_kwargs)

    winners = []
    for i in range(len(data)):
        x = torch.from_numpy(data[i])
        w = net.forward_update(x)
        winners.append(w)

    eval_start = len(data) // 2
    w_arr = np.array(winners[eval_start:])
    l_arr = labels[eval_start:]
    return normalized_mutual_info(w_arr, l_arr)


def run_single(data, labels, n_outputs, **cell_kwargs):
    """Single cell on full input."""
    cell = SoftWTACell(data.shape[1], n_outputs, **cell_kwargs)
    winners = []
    for i in range(len(data)):
        x = torch.from_numpy(data[i])
        p = cell.forward(x)
        w, _ = cell.update(x, p)
        winners.append(w)

    eval_start = len(data) // 2
    return normalized_mutual_info(np.array(winners[eval_start:]), labels[eval_start:])


def main():
    parser = argparse.ArgumentParser(description='Receptive field tiling benchmark')
    parser.add_argument('-o', '--output', type=str, default=None)
    parser.add_argument('--frames', type=int, default=20000)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    cell_kwargs = dict(temperature=0.3, lr=0.05)
    results = {}

    print("=" * 64)
    print("Receptive Field Tiling Benchmark")
    print("=" * 64)

    # Test 1: conjunction detection (16 inputs, 4 groups of 4)
    # Category = combination of feature 0 and feature 1 (4 categories)
    print(f"\n[Test 1] Feature conjunction: 16 inputs, 4 groups of 4")
    print(f"  Category = combination of feature 0 × feature 1 (4 categories)")
    data, fvals, labels = make_conjunction_data(args.frames, n_inputs=16,
                                                n_features=4, seed=args.seed)

    print(f"\n  {'architecture':<40} {'NMI':>8}")
    print("  " + "-" * 48)

    # Single cells
    for m in [4, 8]:
        nmi = run_single(data, labels, m, **cell_kwargs)
        name = f"single cell (m={m})"
        results[f'conj_single_m{m}'] = nmi
        print(f"  {name:<40} {nmi:>8.3f}")

    # Receptive field nets
    for n_local in [2, 4]:
        for n_final in [4, 8]:
            nmi = run_receptive(data, labels, 4, n_local, n_final, **cell_kwargs)
            name = f"RF 4×{n_local} → {n_final}"
            results[f'conj_rf_4x{n_local}_f{n_final}'] = nmi
            print(f"  {name:<40} {nmi:>8.3f}")

    # Test 2: higher dimensional (32 inputs, 4 groups of 8)
    print(f"\n[Test 2] Higher dimensional: 32 inputs, 4 groups of 8")
    data32, fvals32, labels32 = make_conjunction_data(args.frames, n_inputs=32,
                                                      n_features=4, seed=args.seed)

    print(f"\n  {'architecture':<40} {'NMI':>8}")
    print("  " + "-" * 48)

    for m in [4, 8]:
        nmi = run_single(data32, labels32, m, **cell_kwargs)
        name = f"single cell (m={m})"
        results[f'hd_single_m{m}'] = nmi
        print(f"  {name:<40} {nmi:>8.3f}")

    for n_local in [2, 4]:
        nmi = run_receptive(data32, labels32, 4, n_local, 4, **cell_kwargs)
        name = f"RF 4×{n_local} → 4"
        results[f'hd_rf_4x{n_local}_f4'] = nmi
        print(f"  {name:<40} {nmi:>8.3f}")

    # Test 3: more features (32 inputs, 8 groups of 4)
    print(f"\n[Test 3] More features: 32 inputs, 8 groups of 4")
    data8g, fvals8g, labels8g = make_conjunction_data(args.frames, n_inputs=32,
                                                       n_features=8, seed=args.seed)

    print(f"\n  {'architecture':<40} {'NMI':>8}")
    print("  " + "-" * 48)

    nmi = run_single(data8g, labels8g, 4, **cell_kwargs)
    results['mg_single_m4'] = nmi
    print(f"  {'single cell (m=4)':<40} {nmi:>8.3f}")

    nmi = run_receptive(data8g, labels8g, 8, 2, 4, **cell_kwargs)
    results['mg_rf_8x2_f4'] = nmi
    print(f"  {'RF 8×2 → 4':<40} {nmi:>8.3f}")

    nmi = run_receptive(data8g, labels8g, 8, 4, 4, **cell_kwargs)
    results['mg_rf_8x4_f4'] = nmi
    print(f"  {'RF 8×4 → 4':<40} {nmi:>8.3f}")

    if args.output:
        os.makedirs(args.output, exist_ok=True)
        with open(os.path.join(args.output, 'benchmark_receptive.json'), 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved to {args.output}/benchmark_receptive.json")


if __name__ == '__main__':
    main()

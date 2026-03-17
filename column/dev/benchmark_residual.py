"""Residual stacking benchmark — break the single-cell category limit.

Each layer categorizes the residual (what the previous layer got wrong).
Combined output (w1, w2, ...) gives m^L effective categories from L layers
of m outputs each.

Usage:
    python benchmark_residual.py [-o output_dir]
"""

import argparse
import json
import os

import numpy as np
import torch

from column import SoftWTACell
from main import make_clustered_data
from metrics import compute_sqm, format_sqm, normalized_mutual_info


class ResidualStack:
    """Chain of SoftWTACells where each layer categorizes the residual."""

    def __init__(self, n_inputs, n_outputs_per_layer, n_layers, **cell_kwargs):
        self.n_layers = n_layers
        self.cells = []
        for i in range(n_layers):
            cell = SoftWTACell(n_inputs, n_outputs_per_layer, **cell_kwargs)
            self.cells.append(cell)

    def forward_update(self, x):
        """Forward + update through all layers. Returns list of winners."""
        winners = []
        residual = x.clone()

        for cell in self.cells:
            probs = cell.forward(residual)
            winner, _ = cell.update(residual, probs)
            winners.append(winner)
            # Residual: what this layer didn't capture
            residual = residual - cell.prototypes[winner]

        return tuple(winners)

    def forward_only(self, x):
        """Forward only (no update). Returns list of winners."""
        winners = []
        residual = x.clone()

        for cell in self.cells:
            probs = cell.forward(residual)
            winner = probs.argmax().item()
            winners.append(winner)
            residual = residual - cell.prototypes[winner]

        return tuple(winners)


def tuple_to_int(winners, m):
    """Convert (w1, w2, ...) tuple to a single integer label."""
    result = 0
    for w in winners:
        result = result * m + w
    return result


def run_residual(data, labels, n_layers, m, n_frames=None):
    """Train a residual stack and evaluate."""
    if n_frames is None:
        n_frames = len(data)
    n_inputs = data.shape[1]

    stack = ResidualStack(n_inputs, m, n_layers, temperature=0.3, lr=0.05)

    all_winners = []
    for i in range(n_frames):
        x = torch.from_numpy(data[i])
        w = stack.forward_update(x)
        all_winners.append(tuple_to_int(w, m))

    # Evaluate on last portion
    eval_start = n_frames // 2
    w_arr = np.array(all_winners[eval_start:])
    l_arr = labels[eval_start:n_frames]
    nmi = normalized_mutual_info(w_arr, l_arr)

    # Also compute per-layer NMI (how much does each layer add?)
    layer_nmis = []
    for depth in range(1, n_layers + 1):
        all_w_partial = []
        residual_track = []
        for i in range(n_frames):
            x = torch.from_numpy(data[i])
            res = x.clone()
            ws = []
            for j in range(depth):
                p = stack.cells[j].forward(res)
                w = p.argmax().item()
                ws.append(w)
                res = res - stack.cells[j].prototypes[w]
            all_w_partial.append(tuple_to_int(tuple(ws), m))
        w_partial = np.array(all_w_partial[eval_start:])
        layer_nmis.append(normalized_mutual_info(w_partial, l_arr))

    return nmi, layer_nmis


def run_single_cell(data, labels, m, n_frames=None):
    """Baseline: single cell with m outputs."""
    if n_frames is None:
        n_frames = len(data)
    n_inputs = data.shape[1]

    cell = SoftWTACell(n_inputs, m, temperature=0.3, lr=0.05)
    winners = []
    for i in range(n_frames):
        x = torch.from_numpy(data[i])
        p = cell.forward(x)
        w, _ = cell.update(x, p)
        winners.append(w)

    eval_start = n_frames // 2
    w_arr = np.array(winners[eval_start:])
    l_arr = labels[eval_start:n_frames]
    return normalized_mutual_info(w_arr, l_arr)


def main():
    parser = argparse.ArgumentParser(description='Residual stacking benchmark')
    parser.add_argument('-o', '--output', type=str, default=None)
    parser.add_argument('--frames', type=int, default=20000)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    print("=" * 60)
    print("Residual Stacking Benchmark")
    print("=" * 60)

    results = {}

    # Test 1: 16 clusters, 16D — can residual stacking match a 16-output cell?
    print(f"\n[Test 1] 16 clusters in 16D, {args.frames} frames")
    data16, labels16, _ = make_clustered_data(16, 16, args.frames, seed=args.seed)

    print(f"\n  {'architecture':<30} {'effective cats':>14} {'NMI':>8}")
    print("  " + "-" * 54)

    # Single cells
    for m in [4, 8, 16]:
        nmi = run_single_cell(data16, labels16, m, args.frames)
        name = f"single cell (m={m})"
        results[f'single_m{m}'] = nmi
        print(f"  {name:<30} {m:>14} {nmi:>8.3f}")

    # Residual stacks
    for n_layers in [2, 3, 4]:
        m = 4
        nmi, layer_nmis = run_residual(data16, labels16, n_layers, m, args.frames)
        eff = m ** n_layers
        name = f"residual {n_layers}x4"
        results[f'residual_{n_layers}x4'] = {'nmi': nmi, 'layer_nmis': layer_nmis}
        layer_str = " → ".join(f"{n:.3f}" for n in layer_nmis)
        print(f"  {name:<30} {eff:>14} {nmi:>8.3f}  layers: {layer_str}")

    # Test 2: 8 clusters, 16D — easier task
    print(f"\n[Test 2] 8 clusters in 16D, {args.frames} frames")
    data8, labels8, _ = make_clustered_data(8, 16, args.frames, seed=args.seed)

    print(f"\n  {'architecture':<30} {'effective cats':>14} {'NMI':>8}")
    print("  " + "-" * 54)

    for m in [4, 8]:
        nmi = run_single_cell(data8, labels8, m, args.frames)
        name = f"single cell (m={m})"
        results[f'8cl_single_m{m}'] = nmi
        print(f"  {name:<30} {m:>14} {nmi:>8.3f}")

    for n_layers in [2, 3]:
        m = 4
        nmi, layer_nmis = run_residual(data8, labels8, n_layers, m, args.frames)
        eff = m ** n_layers
        name = f"residual {n_layers}x4"
        results[f'8cl_residual_{n_layers}x4'] = {'nmi': nmi, 'layer_nmis': layer_nmis}
        layer_str = " → ".join(f"{n:.3f}" for n in layer_nmis)
        print(f"  {name:<30} {eff:>14} {nmi:>8.3f}  layers: {layer_str}")

    # Residual 2x8 for comparison
    nmi, layer_nmis = run_residual(data8, labels8, 2, 8, args.frames)
    results['8cl_residual_2x8'] = {'nmi': nmi, 'layer_nmis': layer_nmis}
    layer_str = " → ".join(f"{n:.3f}" for n in layer_nmis)
    print(f"  {'residual 2x8':<30} {64:>14} {nmi:>8.3f}  layers: {layer_str}")

    if args.output:
        os.makedirs(args.output, exist_ok=True)
        with open(os.path.join(args.output, 'benchmark_residual.json'), 'w') as f:
            json.dump(results, f, indent=2, default=lambda x: x.tolist()
                      if hasattr(x, 'tolist') else x)
        print(f"\nSaved to {args.output}/benchmark_residual.json")


if __name__ == '__main__':
    main()

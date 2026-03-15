"""SQM benchmark — standardized quality evaluation for SoftWTACell.

Runs four scenarios and reports a full metrics table:
  1. Instantaneous clusters — baseline dot-product separation
  2. Temporal co-variation — correlation mode on co-varying signals
  3. Lock-in — measure convergence speed on both modes
  4. Distribution shift — train on clusters A, switch to B, measure adaptation

Usage:
    python benchmark.py [-o output_dir]
"""

import argparse
import json
import os

import numpy as np
import torch

from column import SoftWTACell
from main import make_clustered_data, make_temporal_clustered_data
from metrics import (compute_sqm, format_sqm, lock_in_score, adaptation_speed,
                     normalized_mutual_info)

N_INPUTS = 16
N_OUTPUTS = 4
N_CLUSTERS = 4
SEED = 42


def run_cell(cell, data, labels, n_frames, temporal_reduce=False):
    """Train cell on data, return winners, probs, SQM snapshots."""
    winners, probs, mqs = [], [], []
    sqm_interval = max(1, n_frames // 5)
    snapshots = []

    for i in range(n_frames):
        x = torch.from_numpy(data[i])
        if temporal_reduce and cell.temporal_mode is None:
            x = x.mean(dim=1)
        p = cell.forward(x)
        w, mq = cell.update(x, p)
        winners.append(w)
        probs.append(p.detach().numpy())
        mqs.append(mq)

        if (i + 1) % sqm_interval == 0:
            window = slice(max(0, i + 1 - sqm_interval), i + 1)
            sqm = compute_sqm(np.array(winners[window]), np.array(probs[window]),
                              cell.prototypes.numpy(), cell.n_outputs,
                              labels=labels[window])
            sqm['frame'] = i + 1
            snapshots.append(sqm)

    w_arr = np.array(winners)
    p_arr = np.array(probs)
    final = compute_sqm(w_arr, p_arr, cell.prototypes.numpy(),
                        cell.n_outputs, labels=labels)
    return w_arr, p_arr, final, snapshots


def scenario_instantaneous():
    """Scenario 1: standard Gaussian clusters, instantaneous mode."""
    data, labels, _ = make_clustered_data(N_CLUSTERS, N_INPUTS, 5000, seed=SEED)
    cell = SoftWTACell(N_INPUTS, N_OUTPUTS, temperature=0.3, lr=0.05)
    w, p, final, snaps = run_cell(cell, data, labels, 5000)
    lock_val, lock_frame = lock_in_score(snaps, 'nmi')
    final['lock_in_nmi'] = lock_val
    final['lock_in_frame'] = lock_frame
    return final, snaps


def scenario_temporal():
    """Scenario 2: temporal co-variation clusters, correlation mode."""
    data, labels, _ = make_temporal_clustered_data(
        N_CLUSTERS, N_INPUTS, 5000, T=10, seed=SEED)
    cell = SoftWTACell(N_INPUTS, N_OUTPUTS, temperature=0.5, lr=0.05,
                       match_threshold=0.1, temporal_mode='correlation')
    w, p, final, snaps = run_cell(cell, data, labels, 5000)
    lock_val, lock_frame = lock_in_score(snaps, 'nmi')
    final['lock_in_nmi'] = lock_val
    final['lock_in_frame'] = lock_frame
    return final, snaps


def scenario_temporal_vs_instantaneous():
    """Scenario 3: temporal data processed with instantaneous mode (should fail)."""
    data, labels, _ = make_temporal_clustered_data(
        N_CLUSTERS, N_INPUTS, 5000, T=10, seed=SEED)
    cell = SoftWTACell(N_INPUTS, N_OUTPUTS, temperature=0.3, lr=0.05)
    w, p, final, snaps = run_cell(cell, data, labels, 5000, temporal_reduce=True)
    return final, snaps


def scenario_adaptation():
    """Scenario 4: train on clusters A, switch to clusters B, measure recovery."""
    n_phase = 3000
    # Phase A
    data_a, labels_a, _ = make_clustered_data(
        N_CLUSTERS, N_INPUTS, n_phase, seed=SEED)
    # Phase B — different seed = different cluster centers
    data_b, labels_b, _ = make_clustered_data(
        N_CLUSTERS, N_INPUTS, n_phase, seed=SEED + 100)

    cell = SoftWTACell(N_INPUTS, N_OUTPUTS, temperature=0.3, lr=0.05)

    # Train phase A
    w_a, _, sqm_a, _ = run_cell(cell, data_a, labels_a, n_phase)

    # Train phase B
    w_b, p_b, sqm_b, snaps_b = run_cell(cell, data_b, labels_b, n_phase)

    frames_to_recover, final_nmi = adaptation_speed(
        w_a, w_b, labels_b, N_OUTPUTS, window=200)

    sqm_b['adaptation_frames'] = frames_to_recover
    sqm_b['pre_shift_nmi'] = sqm_a['nmi']
    return sqm_b, snaps_b


def print_table(results):
    """Print a formatted comparison table."""
    scenarios = list(results.keys())
    # Collect all metric keys
    all_keys = []
    for s in scenarios:
        all_keys.extend(results[s].keys())
    # Order: standard SQM first, then extras
    order = ['winner_entropy', 'usage_gini', 'confidence_gap', 'prototype_spread',
             'nmi', 'purity', 'consistency', 'lock_in_nmi', 'lock_in_frame',
             'adaptation_frames', 'pre_shift_nmi']
    keys = [k for k in order if k in set(all_keys)]

    # Header
    col_w = 14
    header = f"{'metric':<22}" + "".join(f"{s:>{col_w}}" for s in scenarios)
    print(header)
    print("-" * len(header))

    for k in keys:
        row = f"{k:<22}"
        for s in scenarios:
            v = results[s].get(k, '')
            if isinstance(v, float):
                row += f"{v:>{col_w}.3f}"
            elif isinstance(v, int):
                row += f"{v:>{col_w}d}"
            else:
                row += f"{'—':>{col_w}}"
        print(row)


def main():
    parser = argparse.ArgumentParser(description='SQM benchmark suite')
    parser.add_argument('-o', '--output', type=str, default=None)
    args = parser.parse_args()

    print("=" * 70)
    print("SQM Benchmark — SoftWTACell Quality Evaluation")
    print("=" * 70)

    results = {}
    snapshots = {}

    print("\n[1/4] Instantaneous clusters...")
    results['instant'], snapshots['instant'] = scenario_instantaneous()
    print(f"  {format_sqm(results['instant'])}")

    print("\n[2/4] Temporal co-variation (correlation mode)...")
    results['temporal'], snapshots['temporal'] = scenario_temporal()
    print(f"  {format_sqm(results['temporal'])}")

    print("\n[3/4] Temporal data + instantaneous mode (control)...")
    results['temp_fail'], snapshots['temp_fail'] = scenario_temporal_vs_instantaneous()
    print(f"  {format_sqm(results['temp_fail'])}")

    print("\n[4/4] Distribution shift adaptation...")
    results['adapt'], snapshots['adapt'] = scenario_adaptation()
    print(f"  {format_sqm(results['adapt'])}")

    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70 + "\n")
    print_table(results)

    if args.output:
        os.makedirs(args.output, exist_ok=True)
        out = {k: v for k, v in results.items()}
        out['snapshots'] = {k: v for k, v in snapshots.items()}
        with open(os.path.join(args.output, 'benchmark.json'), 'w') as f:
            json.dump(out, f, indent=2, default=lambda x: x.tolist()
                      if hasattr(x, 'tolist') else x)
        print(f"\nSaved to {args.output}/benchmark.json")


if __name__ == '__main__':
    main()

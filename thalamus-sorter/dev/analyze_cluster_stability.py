#!/usr/bin/env python3
"""Analyze per-neuron cluster stability from saved history.

Usage:
    python analyze_cluster_stability.py RUN_DIR [--skip N] [--out PREFIX]
    python analyze_cluster_stability.py dir1 dir2 dir3 --compare

Reads history_ticks.npy and history_ids.npy saved by --cluster-track-history.
"""

import argparse
import numpy as np
import os
import sys


def load_history(run_dir):
    """Load cluster history from a run directory."""
    ticks = np.load(os.path.join(run_dir, "history_ticks.npy"))
    ids = np.load(os.path.join(run_dir, "history_ids.npy"))
    return ticks, ids


def analyze(ticks, ids, skip=0):
    """Analyze per-neuron stability.

    Args:
        ticks: (n_snapshots,) tick numbers
        ids: (n_snapshots, n) cluster IDs per neuron per snapshot
        skip: skip first N snapshots (turbulent cluster formation phase)

    Returns dict with:
        changes: (n,) number of cluster changes per neuron
        stability: (n,) fraction of intervals where neuron stayed
        oscillators: (n,) count of A->B->A patterns
        last_change: (n,) snapshot index of most recent change (-1 if never)
    """
    ids = ids[skip:]
    ticks = ticks[skip:]
    n_snaps, n = ids.shape

    if n_snaps < 2:
        print(f"Need at least 2 snapshots after skip={skip}, got {n_snaps}")
        sys.exit(1)

    # Per-neuron changes: count transitions between consecutive snapshots
    diffs = ids[1:] != ids[:-1]  # (n_snaps-1, n)
    changes = diffs.sum(axis=0)  # (n,)
    n_intervals = n_snaps - 1
    stability = 1.0 - changes / n_intervals

    # Last change index
    last_change = np.full(n, -1, dtype=np.int64)
    for i in range(n):
        changed = np.where(diffs[:, i])[0]
        if len(changed) > 0:
            last_change[i] = changed[-1] + skip

    # Oscillation detection: A->B->A pattern
    oscillators = np.zeros(n, dtype=np.int64)
    if n_snaps >= 3:
        for t in range(n_snaps - 2):
            aba = (ids[t] == ids[t + 2]) & (ids[t] != ids[t + 1])
            oscillators += aba

    return {
        'changes': changes,
        'stability': stability,
        'oscillators': oscillators,
        'last_change': last_change,
        'n_intervals': n_intervals,
        'ticks': ticks,
    }


def print_report(results, label=""):
    """Print summary statistics."""
    changes = results['changes']
    stability = results['stability']
    oscillators = results['oscillators']
    n = len(changes)
    n_intervals = results['n_intervals']

    if label:
        print(f"\n=== {label} ===")
    print(f"  Snapshots: {n_intervals + 1}, neurons: {n}")
    print(f"  Changes per neuron: mean={changes.mean():.1f}, "
          f"median={np.median(changes):.0f}, max={changes.max()}")
    print(f"  Stability: mean={stability.mean():.3f}, "
          f"min={stability.min():.3f}")

    # Histogram buckets
    never = (changes == 0).sum()
    low = ((changes >= 1) & (changes <= 3)).sum()
    med = ((changes > 3) & (changes <= 10)).sum()
    high = (changes > 10).sum()
    print(f"  Never changed: {never} ({never/n*100:.1f}%)")
    print(f"  1-3 changes:   {low} ({low/n*100:.1f}%)")
    print(f"  4-10 changes:  {med} ({med/n*100:.1f}%)")
    print(f"  >10 changes:   {high} ({high/n*100:.1f}%)")

    n_osc = (oscillators > 0).sum()
    print(f"  Oscillators (A->B->A): {n_osc} neurons "
          f"({n_osc/n*100:.1f}%), total={oscillators.sum()} patterns")


def save_heatmap(changes, w, h, path):
    """Save spatial heatmap of cluster changes."""
    try:
        import cv2
    except ImportError:
        print(f"  cv2 not available, skipping heatmap")
        return

    grid = changes.reshape(h, w).astype(np.float32)
    if grid.max() > 0:
        grid = grid / grid.max() * 255
    grid = grid.astype(np.uint8)
    heatmap = cv2.applyColorMap(grid, cv2.COLORMAP_HOT)
    cv2.imwrite(path, heatmap)
    print(f"  Heatmap saved: {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dirs", nargs="+", help="Run directories")
    parser.add_argument("--skip", type=int, default=0,
                        help="Skip first N snapshots (cluster formation phase)")
    parser.add_argument("--width", "-W", type=int, default=80)
    parser.add_argument("--height", "-H", type=int, default=80)
    parser.add_argument("--heatmap", action="store_true",
                        help="Save instability heatmap per run")
    args = parser.parse_args()

    for run_dir in args.dirs:
        label = os.path.basename(run_dir)
        ticks, ids = load_history(run_dir)
        results = analyze(ticks, ids, skip=args.skip)
        print_report(results, label=label)

        if args.heatmap:
            path = os.path.join(run_dir, "instability_heatmap.png")
            save_heatmap(results['changes'], args.width, args.height, path)


if __name__ == "__main__":
    main()

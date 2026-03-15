"""3D movement direction benchmark for SoftWTACell.

Tests whether the cell can learn to categorize the direction of a moving
object in 3D space from position traces. Instantaneous position tells you
nothing about direction — only the temporal trace reveals it.

Scenarios:
  1. Cardinal directions (6: ±x, ±y, ±z) — axis-aligned, easy
  2. Diagonal directions (8: all ±x±y±z combos) — off-axis, harder
  3. Fine-grained (12: edges of an icosahedron) — many similar directions
  4. Control: instantaneous mode on same data (should get ~chance NMI)

Usage:
    python benchmark_3d.py [-o output_dir]
"""

import argparse
import json
import os

import numpy as np
import torch

from column import SoftWTACell
from metrics import compute_sqm, format_sqm, lock_in_score


def make_3d_movement_data(directions, n_samples, T=10, speed=1.0, noise=0.3, seed=42):
    """Generate position traces of objects moving along given directions.

    Each trace is a (3, T) matrix of x,y,z positions over time.
    Movement = random walk with drift along a cluster-specific direction.

    Args:
        directions: (K, 3) array of unit direction vectors
        n_samples: number of traces to generate
        T: trace length (timesteps)
        speed: drift speed along direction
        noise: per-step noise std
        seed: random seed
    """
    rng = np.random.default_rng(seed)
    n_dirs = len(directions)

    data = []
    labels = []
    for i in range(n_samples):
        c = rng.integers(n_dirs)
        # Random starting position (large offset hides direction from mean)
        start = rng.standard_normal(3) * 10.0
        # Random walk with directional drift
        steps = directions[c].reshape(3, 1) * speed + rng.standard_normal((3, T)) * noise
        trace = start.reshape(3, 1) + np.cumsum(steps, axis=1)
        data.append(trace.astype(np.float32))
        labels.append(c)

    return data, np.array(labels)


def cardinal_directions():
    """6 axis-aligned directions: ±x, ±y, ±z."""
    return np.array([
        [1, 0, 0], [-1, 0, 0],
        [0, 1, 0], [0, -1, 0],
        [0, 0, 1], [0, 0, -1],
    ], dtype=np.float32)


def diagonal_directions():
    """8 diagonal directions: all ±x±y±z combinations, normalized."""
    dirs = []
    for sx in [-1, 1]:
        for sy in [-1, 1]:
            for sz in [-1, 1]:
                dirs.append([sx, sy, sz])
    dirs = np.array(dirs, dtype=np.float32)
    return dirs / np.linalg.norm(dirs, axis=1, keepdims=True)


def icosahedron_directions():
    """12 directions from icosahedron edges — evenly spread on sphere."""
    phi = (1 + np.sqrt(5)) / 2  # golden ratio
    dirs = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1],
    ], dtype=np.float32)
    return dirs / np.linalg.norm(dirs, axis=1, keepdims=True)


def run_scenario(name, directions, n_outputs, n_frames=5000, T=10,
                 speed=1.0, noise=0.3, temporal_mode='correlation',
                 temperature=0.5, match_threshold=0.1, seed=42):
    """Run a single benchmark scenario and return SQM results."""
    data, labels = make_3d_movement_data(
        directions, n_frames, T=T, speed=speed, noise=noise, seed=seed)

    cell = SoftWTACell(
        n_inputs=3, n_outputs=n_outputs,
        temperature=temperature, lr=0.05,
        match_threshold=match_threshold,
        temporal_mode=temporal_mode if temporal_mode != 'none' else None,
    )

    winners, probs = [], []
    sqm_interval = max(1, n_frames // 5)
    snapshots = []

    for i in range(n_frames):
        x = torch.from_numpy(data[i])
        if temporal_mode == 'none':
            x = x.mean(dim=1)  # reduce to (3,) for instantaneous
        p = cell.forward(x)
        w, _ = cell.update(x, p)
        winners.append(w)
        probs.append(p.detach().numpy())

        if (i + 1) % sqm_interval == 0:
            window = slice(max(0, i + 1 - sqm_interval), i + 1)
            sqm = compute_sqm(np.array(winners[window]), np.array(probs[window]),
                              cell.prototypes.numpy(), n_outputs,
                              labels=labels[window])
            sqm['frame'] = i + 1
            snapshots.append(sqm)

    w_arr = np.array(winners)
    p_arr = np.array(probs)
    final = compute_sqm(w_arr, p_arr, cell.prototypes.numpy(),
                        n_outputs, labels=labels)

    lock_val, lock_frame = lock_in_score(snapshots, 'nmi')
    final['lock_in_nmi'] = lock_val
    final['lock_in_frame'] = lock_frame
    final['n_directions'] = len(directions)

    return final, snapshots


def print_table(results):
    """Print a formatted comparison table."""
    scenarios = list(results.keys())
    col_w = max(14, max(len(s) for s in scenarios) + 2)
    order = ['n_directions', 'winner_entropy', 'usage_gini', 'confidence_gap',
             'prototype_spread', 'nmi', 'purity', 'consistency',
             'lock_in_nmi', 'lock_in_frame']
    all_keys = set()
    for s in scenarios:
        all_keys.update(results[s].keys())
    keys = [k for k in order if k in all_keys]

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
    parser = argparse.ArgumentParser(description='3D movement direction benchmark')
    parser.add_argument('-o', '--output', type=str, default=None)
    parser.add_argument('--frames', type=int, default=5000)
    parser.add_argument('-T', '--temporal-window', type=int, default=10)
    parser.add_argument('--speed', type=float, default=1.0)
    parser.add_argument('--noise', type=float, default=0.3)
    args = parser.parse_args()

    print("=" * 72)
    print("3D Movement Direction Benchmark")
    print("=" * 72)
    print(f"Config: T={args.temporal_window}, speed={args.speed}, "
          f"noise={args.noise}, frames={args.frames}")

    results = {}
    snapshots = {}

    print("\n[1/4] 6 cardinal directions (±x, ±y, ±z)...")
    results['cardinal'], snapshots['cardinal'] = run_scenario(
        'cardinal', cardinal_directions(), n_outputs=6,
        n_frames=args.frames, T=args.temporal_window,
        speed=args.speed, noise=args.noise)
    print(f"  {format_sqm(results['cardinal'])}")

    print("\n[2/4] 8 diagonal directions...")
    results['diagonal'], snapshots['diagonal'] = run_scenario(
        'diagonal', diagonal_directions(), n_outputs=8,
        n_frames=args.frames, T=args.temporal_window,
        speed=args.speed, noise=args.noise)
    print(f"  {format_sqm(results['diagonal'])}")

    print("\n[3/4] 12 icosahedron directions (fine-grained)...")
    results['icosa'], snapshots['icosa'] = run_scenario(
        'icosa', icosahedron_directions(), n_outputs=12,
        n_frames=args.frames * 2, T=args.temporal_window,
        speed=args.speed, noise=args.noise)
    print(f"  {format_sqm(results['icosa'])}")

    print("\n[4/4] 6 cardinal — instantaneous mode (control)...")
    results['control'], snapshots['control'] = run_scenario(
        'control', cardinal_directions(), n_outputs=6,
        n_frames=args.frames, T=args.temporal_window,
        speed=args.speed, noise=args.noise,
        temporal_mode='none', match_threshold=0.5, temperature=0.3)
    print(f"  {format_sqm(results['control'])}")

    print("\n" + "=" * 72)
    print("Summary")
    print("=" * 72 + "\n")
    print_table(results)

    if args.output:
        os.makedirs(args.output, exist_ok=True)
        out = {k: v for k, v in results.items()}
        out['snapshots'] = {k: v for k, v in snapshots.items()}
        with open(os.path.join(args.output, 'benchmark_3d.json'), 'w') as f:
            json.dump(out, f, indent=2, default=lambda x: x.tolist()
                      if hasattr(x, 'tolist') else x)
        print(f"\nSaved to {args.output}/benchmark_3d.json")


if __name__ == '__main__':
    main()

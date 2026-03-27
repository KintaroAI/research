"""Patch size sweep: measure column collapse vs patch size.

Standalone script — no model pipeline, just ColumnManager + saccades.
Tests patch sizes 2×2 through 10×10 on 80×80 retina, 4 outputs each,
reports dominant winner fraction after 10k ticks.

Usage:
    cd thalamus-sorter/dev
    python benchmarks/patch_sweep.py
    python benchmarks/patch_sweep.py --source other_image.npy --ticks 20000
"""

import argparse
import time
import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from column_manager import ColumnManager, ConscienceColumn


def make_columns(patch_sz, retina, n_outputs, window, temperature, lr,
                 column_type='default', alpha=0.01, reseed_after=1000):
    """Create a wired column manager for a given patch size on the retina."""
    grid_n = retina // patch_sz
    n_patches = grid_n * grid_n
    n_inputs = patch_sz * patch_sz
    retina_n = retina * retina

    if column_type == 'conscience':
        cm = ConscienceColumn(
            m=n_patches, n_outputs=n_outputs, max_inputs=n_inputs,
            window=window, temperature=temperature, lr=lr,
            alpha=alpha, reseed_after=reseed_after,
        )
    else:
        cm = ColumnManager(
            m=n_patches, n_outputs=n_outputs, max_inputs=n_inputs,
            window=window, temperature=temperature, lr=lr,
            mode='kmeans', entropy_scaled_lr=True,
        )

    for py in range(grid_n):
        for px in range(grid_n):
            patch_id = py * grid_n + px
            slot = 0
            for dy in range(patch_sz):
                for dx in range(patch_sz):
                    neuron_id = (py * patch_sz + dy) * retina + (px * patch_sz + dx)
                    cm.slot_map[patch_id, slot] = neuron_id
                    slot += 1

    ring = np.zeros((retina_n, window), dtype=np.float32)
    return cm, ring, grid_n, n_patches


def run_one(source, patch_sz, retina, n_outputs, n_ticks, n_eval, step,
            window, temperature, lr, column_type='default',
            alpha=0.01, reseed_after=1000):
    """Run one patch size: train for n_ticks, evaluate on last n_eval ticks."""
    cm, ring, grid_n, n_patches = make_columns(
        patch_sz, retina, n_outputs, window, temperature, lr,
        column_type=column_type, alpha=alpha, reseed_after=reseed_after)

    src_h, src_w = source.shape
    max_dy = src_h - retina
    max_dx = src_w - retina
    rng = np.random.RandomState(42)
    walk_y = rng.randint(0, max_dy + 1)
    walk_x = rng.randint(0, max_dx + 1)

    win_counts = np.zeros((n_patches, n_outputs), dtype=int)
    eval_start = n_ticks - n_eval

    for t in range(n_ticks):
        walk_y = np.clip(walk_y + rng.randint(-step, step + 1), 0, max_dy)
        walk_x = np.clip(walk_x + rng.randint(-step, step + 1), 0, max_dx)
        crop = source[walk_y:walk_y+retina, walk_x:walk_x+retina].ravel()
        ring[:, :-1] = ring[:, 1:]
        ring[:, -1] = crop
        cm.tick(ring)

        if t >= eval_start:
            winners = cm.get_outputs().argmax(axis=1)
            for p in range(n_patches):
                win_counts[p, winners[p]] += 1

    # Metrics (over eval window only)
    win_fracs = win_counts.astype(np.float32) / n_eval
    dominant_frac = win_fracs.max(axis=1)             # per-patch
    global_frac = win_counts.sum(axis=0) / win_counts.sum()

    # Entropy of per-patch winner distribution (not softmax entropy)
    eps = 1e-10
    H_per_patch = -(win_fracs * np.log(win_fracs + eps)).sum(axis=1)
    H_max = np.log(n_outputs)

    return {
        'patch_sz': patch_sz,
        'n_inputs': patch_sz * patch_sz,
        'grid_n': grid_n,
        'n_patches': n_patches,
        'dominant_mean': float(dominant_frac.mean()),
        'dominant_min': float(dominant_frac.min()),
        'dominant_max': float(dominant_frac.max()),
        'collapsed_90': int((dominant_frac > 0.90).sum()),
        'collapsed_75': int((dominant_frac > 0.75).sum()),
        'winner_entropy_norm': float((H_per_patch / H_max).mean()),
        'global_frac': [round(float(f) * 100, 1) for f in global_frac],
    }


def main():
    parser = argparse.ArgumentParser(description='Patch size sweep: column collapse vs patch size')
    parser.add_argument('--source', type=str, default='saccades_gray.npy',
                        help='Source image (.npy or .png)')
    parser.add_argument('--retina', type=int, default=80,
                        help='Retina crop size (default: 80)')
    parser.add_argument('--ticks', type=int, default=10000,
                        help='Training ticks per patch size (default: 10000)')
    parser.add_argument('--eval', type=int, default=500,
                        help='Evaluation window: last N ticks (default: 500)')
    parser.add_argument('--step', type=int, default=50,
                        help='Saccade step size (default: 50)')
    parser.add_argument('--outputs', type=int, default=4,
                        help='Column outputs (default: 4)')
    parser.add_argument('--window', type=int, default=10,
                        help='Temporal window (default: 10)')
    parser.add_argument('--temperature', type=float, default=0.2,
                        help='Softmax temperature (default: 0.2)')
    parser.add_argument('--lr', type=float, default=0.05,
                        help='Column learning rate (default: 0.05)')
    parser.add_argument('--patches', type=str, default='2,3,4,5,6,7,8,9,10',
                        help='Comma-separated patch sizes (default: 2,3,4,5,6,7,8,9,10)')
    parser.add_argument('--column-type', type=str, default='default',
                        help="Column type: 'default' or 'conscience' (default: default)")
    parser.add_argument('--alpha', type=float, default=0.01,
                        help='Conscience threshold learning rate (default: 0.01)')
    parser.add_argument('--reseed-after', type=int, default=1000,
                        help='Reseed dead units after N ticks (default: 1000)')
    args = parser.parse_args()

    # Load source
    if args.source.endswith('.npy'):
        source = np.load(args.source).astype(np.float32)
    else:
        import cv2
        img = cv2.imread(args.source, cv2.IMREAD_GRAYSCALE)
        source = img.astype(np.float32) / 255.0
    if source.ndim == 3:
        source = source.mean(axis=2)

    patch_sizes = [int(x) for x in args.patches.split(',')]

    col_type = getattr(args, 'column_type', 'default')
    print(f"Patch sweep: retina={args.retina}, ticks={args.ticks}, "
          f"eval={args.eval}, step={args.step}, outputs={args.outputs}, "
          f"window={args.window}, temp={args.temperature}, lr={args.lr}, "
          f"type={col_type}"
          + (f", alpha={args.alpha}" if col_type == 'conscience' else ""))
    print(f"Source: {args.source} ({source.shape[1]}x{source.shape[0]})")
    print()

    results = []
    for P in patch_sizes:
        grid_n = args.retina // P
        if grid_n < 1:
            print(f"  {P}x{P}: skip (retina {args.retina} too small)")
            continue
        t0 = time.time()
        r = run_one(source, P, args.retina, args.outputs, args.ticks,
                    args.eval, args.step, args.window, args.temperature,
                    args.lr, column_type=col_type,
                    alpha=args.alpha, reseed_after=args.reseed_after)
        elapsed = time.time() - t0
        results.append(r)
        print(f"  {P}x{P} ({r['n_inputs']:>3d} in, {r['n_patches']:>4d} patches): "
              f"dominant={r['dominant_mean']:.3f}, "
              f">90%={r['collapsed_90']}/{r['n_patches']}, "
              f"H_norm={r['winner_entropy_norm']:.3f}  "
              f"({elapsed:.1f}s)")

    # Summary table
    print()
    print(f"{'patch':>5} {'inputs':>6} {'grid':>5} {'patches':>7} "
          f"{'dom_mean':>8} {'dom_range':>13} "
          f"{'>90%':>6} {'>75%':>6} {'H_norm':>6}")
    print("-" * 75)
    for r in results:
        print(f"{r['patch_sz']:>3}x{r['patch_sz']:<2} {r['n_inputs']:>5}  "
              f"{r['grid_n']:>2}x{r['grid_n']:<2}  {r['n_patches']:>6}  "
              f"{r['dominant_mean']:>7.3f}  "
              f"[{r['dominant_min']:.3f},{r['dominant_max']:.3f}]  "
              f"{r['collapsed_90']:>5}  {r['collapsed_75']:>5}  "
              f"{r['winner_entropy_norm']:>.3f}")


if __name__ == '__main__':
    main()

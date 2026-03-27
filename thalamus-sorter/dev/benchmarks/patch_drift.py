"""Patch drift: measure temporal stability of column winner preferences.

Replays the SAME saccade sequence twice to isolate true drift from
input variation. Steps:

1. Train for N ticks (phase 1: 0..N/2, phase 2: N/2..N)
2. Record saccade positions + per-tick winners during phase 2 (ticks N/2..N)
3. Replay those exact saccade positions (ticks N..N+N/2) — columns keep learning
4. Compare per-tick winners: same input → same output = stable

Metrics:
- match_rate: fraction of ticks where replay winner == original winner (same input)
- per-patch match: which patches are stable vs drifting
- drift by output: does one output drift more than others?

Usage:
    cd thalamus-sorter/dev
    python benchmarks/patch_drift.py
    python benchmarks/patch_drift.py --column-type default  # comparison
    python benchmarks/patch_drift.py --patches 5,7,9        # sweep
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


def run_drift(source, patch_sz, retina, n_outputs, n_train, step,
              window, temperature, lr, column_type, alpha, reseed_after):
    """Train, record, replay, compare."""
    cm, ring, grid_n, n_patches = make_columns(
        patch_sz, retina, n_outputs, window, temperature, lr,
        column_type=column_type, alpha=alpha, reseed_after=reseed_after)

    src_h, src_w = source.shape
    max_dy = src_h - retina
    max_dx = src_w - retina
    rng = np.random.RandomState(42)
    walk_y = rng.randint(0, max_dy + 1)
    walk_x = rng.randint(0, max_dx + 1)
    retina_n = retina * retina

    def do_tick_at(y, x):
        """Run one tick with a specific saccade position."""
        crop = source[y:y+retina, x:x+retina].ravel()
        ring[:, :-1] = ring[:, 1:]
        ring[:, -1] = crop
        cm.tick(ring)

    # --- Phase 1: pure training (first half) ---
    half = n_train // 2
    for t in range(half):
        walk_y = np.clip(walk_y + rng.randint(-step, step + 1), 0, max_dy)
        walk_x = np.clip(walk_x + rng.randint(-step, step + 1), 0, max_dx)
        do_tick_at(walk_y, walk_x)

    # --- Phase 2: record saccade positions + winners (second half) ---
    record_positions = []  # (half, 2) saccade positions
    record_winners = []    # (half, n_patches) per-tick winners

    for t in range(half):
        walk_y = np.clip(walk_y + rng.randint(-step, step + 1), 0, max_dy)
        walk_x = np.clip(walk_x + rng.randint(-step, step + 1), 0, max_dx)
        record_positions.append((walk_y, walk_x))
        do_tick_at(walk_y, walk_x)
        record_winners.append(cm.get_outputs().argmax(axis=1).copy())

    record_winners = np.array(record_winners)  # (half, n_patches)

    # --- Phase 3: replay exact same positions (columns keep learning) ---
    replay_winners = []

    for t in range(half):
        y, x = record_positions[t]
        do_tick_at(y, x)
        replay_winners.append(cm.get_outputs().argmax(axis=1).copy())

    replay_winners = np.array(replay_winners)  # (half, n_patches)

    # --- Metrics ---

    # 1. Per-tick match: same input, same winner?
    matches = (record_winners == replay_winners)        # (half, n_patches) bool
    overall_match = matches.mean()

    # 2. Per-patch match rate
    per_patch_match = matches.mean(axis=0)              # (n_patches,)

    # 3. Per-output stability: when output X won originally, how often does it
    #    still win on replay?
    per_output_match = np.zeros(n_outputs, dtype=np.float32)
    per_output_count = np.zeros(n_outputs, dtype=np.float32)
    for o in range(n_outputs):
        mask = record_winners == o
        per_output_count[o] = mask.sum()
        if per_output_count[o] > 0:
            per_output_match[o] = matches[mask].mean()

    # 4. Transition analysis: when winner changes, where does it go?
    changed = ~matches
    flip_matrix = np.zeros((n_outputs, n_outputs), dtype=int)
    for t in range(half):
        for p in range(n_patches):
            if changed[t, p]:
                flip_matrix[record_winners[t, p], replay_winners[t, p]] += 1

    return {
        'patch_sz': patch_sz,
        'n_patches': n_patches,
        'n_ticks': half,
        'column_type': column_type,
        'overall_match': float(overall_match),
        'per_patch_match_mean': float(per_patch_match.mean()),
        'per_patch_match_range': [float(per_patch_match.min()),
                                   float(per_patch_match.max())],
        'stable_patches': int((per_patch_match > 0.9).sum()),
        'drifting_patches': int((per_patch_match < 0.5).sum()),
        'per_output_match': [float(per_output_match[o]) for o in range(n_outputs)],
        'per_output_count': [int(per_output_count[o]) for o in range(n_outputs)],
        'flip_matrix': flip_matrix,
        'total_flips': int(changed.sum()),
    }


def main():
    parser = argparse.ArgumentParser(
        description='Measure column drift via same-input replay')
    parser.add_argument('--source', type=str, default='saccades_gray.npy')
    parser.add_argument('--retina', type=int, default=80)
    parser.add_argument('--patch', type=int, default=7)
    parser.add_argument('--ticks', type=int, default=10000,
                        help='Total training ticks; second half is recorded (default: 10000)')
    parser.add_argument('--step', type=int, default=50)
    parser.add_argument('--outputs', type=int, default=4)
    parser.add_argument('--temporal-window', type=int, default=10)
    parser.add_argument('--temperature', type=float, default=0.2)
    parser.add_argument('--lr', type=float, default=0.05)
    parser.add_argument('--column-type', type=str, default='conscience')
    parser.add_argument('--alpha', type=float, default=0.01)
    parser.add_argument('--reseed-after', type=int, default=1000)
    parser.add_argument('--patches', type=str, default=None,
                        help='Comma-separated patch sizes for sweep (e.g., 5,7,9)')
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

    patch_sizes = [int(x) for x in args.patches.split(',')] if args.patches else [args.patch]
    half = args.ticks // 2

    print(f"Drift benchmark (replay): type={args.column_type}, "
          f"train={args.ticks} (record last {half}), replay {half}")
    print()

    results = []
    for P in patch_sizes:
        grid_n = args.retina // P
        if grid_n < 1:
            continue
        t0 = time.time()
        r = run_drift(source, P, args.retina, args.outputs,
                      args.ticks, args.step, args.temporal_window,
                      args.temperature, args.lr, args.column_type,
                      args.alpha, args.reseed_after)
        elapsed = time.time() - t0
        results.append(r)
        print(f"  {P}x{P} ({r['n_patches']} patches): "
              f"match={r['overall_match']:.1%}, "
              f"stable(>90%)={r['stable_patches']}/{r['n_patches']}, "
              f"drifting(<50%)={r['drifting_patches']}/{r['n_patches']}  "
              f"({elapsed:.1f}s)")

    # Summary table
    print()
    print(f"{'patch':>5} {'type':>10} {'match':>7} {'stable':>8} "
          f"{'drifting':>8} {'range':>15} {'flips':>6}")
    print("-" * 68)
    for r in results:
        pr = r['per_patch_match_range']
        print(f"{r['patch_sz']:>3}x{r['patch_sz']:<2} {r['column_type']:>10} "
              f"{r['overall_match']:>6.1%}  "
              f"{r['stable_patches']:>4}/{r['n_patches']:<4} "
              f"{r['drifting_patches']:>4}/{r['n_patches']:<4} "
              f"[{pr[0]:.2f}-{pr[1]:.2f}]  "
              f"{r['total_flips']:>5}")

    # Per-output stability
    if results:
        r = results[-1]
        n_out = len(r['per_output_match'])
        print(f"\nPer-output stability ({r['patch_sz']}x{r['patch_sz']} {r['column_type']}):")
        for o in range(n_out):
            cnt = r['per_output_count'][o]
            m = r['per_output_match'][o]
            print(f"  output {o}: match={m:.1%} ({cnt} original wins)")

        print(f"\nFlip matrix (original → replay):")
        print(f"       {'  '.join(f'to_{j}' for j in range(n_out))}")
        for i in range(n_out):
            row = '  '.join(f'{r["flip_matrix"][i,j]:>5}' for j in range(n_out))
            print(f"  from_{i}  {row}")


if __name__ == '__main__':
    main()

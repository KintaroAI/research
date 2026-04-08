"""Standalone patch column tuning tool.

Runs prewired 7x7 patch columns on saccade crops and outputs diagnostics.
No model/clustering — just the first-layer columns + pattern visualization.

Usage:
    cd dev/
    python benchmarks/patch_column.py --source saccades_gray.npy -f 10000
    python benchmarks/patch_column.py --source saccades_gray.npy -f 10000 \
        --column-type conscience --step 3 --outputs 4
"""

import os
import sys
import json
import argparse
import numpy as np

# Allow imports from parent dir
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from cluster_manager import create_column


def build_patch_layer(args):
    retina = args.retina_size
    patch_sz = args.patch_size
    grid_n = retina // patch_sz
    n_patches = grid_n * grid_n
    n_inputs = patch_sz * patch_sz
    n_outputs = args.outputs
    step = args.step

    # Load source
    source_path = args.source
    if source_path.endswith('.npy'):
        source = np.load(source_path).astype(np.float32)
    else:
        import cv2
        img = cv2.imread(source_path, cv2.IMREAD_GRAYSCALE)
        source = img.astype(np.float32) / 255.0
    if source.ndim == 3:
        source = source.mean(axis=2)

    src_h, src_w = source.shape
    assert src_h >= retina and src_w >= retina, \
        f"Source {src_w}x{src_h} too small for {retina}x{retina} retina"

    # Create columns via factory
    n_heads = 7 if n_inputs % 7 == 0 else 1
    col_config = {
        'm': n_patches,
        'n_outputs': n_outputs,
        'max_inputs': n_inputs,
        'window': args.window,
        'temperature': args.temperature,
        'lr': args.lr,
        'alpha': args.alpha,
        'n_heads': n_heads,
        'mode': 'kmeans',
        'entropy_scaled_lr': True,
        'k_active': args.k_active,
        'homeostasis_rate': args.homeostasis_rate,
        'fatigue_strength': args.fatigue_strength,
        'multi_scale': getattr(args, 'multi_scale', False),
    }
    cm = create_column(args.column_type, col_config)

    # Pre-wire patches
    retina_n = retina * retina
    for py in range(grid_n):
        for px in range(grid_n):
            patch_id = py * grid_n + px
            slot = 0
            for dy in range(patch_sz):
                for dx in range(patch_sz):
                    neuron_id = (py * patch_sz + dy) * retina + (px * patch_sz + dx)
                    cm.slot_map[patch_id, slot] = neuron_id
                    slot += 1

    ring = np.zeros((retina_n, args.window), dtype=np.float32)

    rng = np.random.RandomState(42)
    max_dy = src_h - retina
    max_dx = src_w - retina
    walk_y = rng.randint(0, max_dy + 1)
    walk_x = rng.randint(0, max_dx + 1)

    def tick_fn():
        nonlocal walk_y, walk_x
        walk_y = np.clip(walk_y + rng.randint(-step, step + 1), 0, max_dy)
        walk_x = np.clip(walk_x + rng.randint(-step, step + 1), 0, max_dx)
        crop = source[walk_y:walk_y+retina, walk_x:walk_x+retina].ravel()
        ring[:, :-1] = ring[:, 1:]
        ring[:, -1] = crop
        cm.tick(ring)

    return cm, tick_fn, ring, source, {
        'retina': retina, 'patch_sz': patch_sz, 'grid_n': grid_n,
        'n_patches': n_patches, 'n_outputs': n_outputs,
    }


def run(args):
    cm, tick_fn, ring, source, meta = build_patch_layer(args)
    n_patches = meta['n_patches']
    n_outputs = meta['n_outputs']
    patch_sz = meta['patch_sz']
    grid_n = meta['grid_n']
    retina = meta['retina']

    print(f"Patch layer: {n_patches} patches of {patch_sz}x{patch_sz}, "
          f"{n_outputs} outputs, retina={retina}x{retina}, step={args.step}")
    print(f"Column type: {args.column_type}, lr={args.lr}, "
          f"alpha={args.alpha}, temp={args.temperature}")

    # --- Training ---
    frames = args.frames
    log_every = max(frames // 10, 1)
    for t in range(frames):
        tick_fn()
        if (t + 1) % log_every == 0:
            print(f"  tick {t+1}/{frames}")

    # --- Eval: freeze and sample with wide saccades ---
    n_sample = 500
    eval_step = args.eval_step
    saved_lr = cm.lr
    cm.lr = 0.0
    if hasattr(cm, 'set_learn_prob'):
        cm.set_learn_prob(0.0)

    output_history = np.zeros((n_sample, n_patches, n_outputs), dtype=np.float32)
    crop_history = np.zeros((n_sample, retina, retina), dtype=np.float32)

    # Separate saccade walk for eval (wider steps for variety)
    src_h, src_w = source.shape[:2]
    max_dy = src_h - retina
    max_dx = src_w - retina
    eval_rng = np.random.RandomState(123)
    ey = eval_rng.randint(0, max_dy + 1)
    ex = eval_rng.randint(0, max_dx + 1)

    for t in range(n_sample):
        ey = np.clip(ey + eval_rng.randint(-eval_step, eval_step + 1), 0, max_dy)
        ex = np.clip(ex + eval_rng.randint(-eval_step, eval_step + 1), 0, max_dx)
        crop = source[ey:ey+retina, ex:ex+retina].ravel()
        ring[:, :-1] = ring[:, 1:]
        ring[:, -1] = crop
        cm.tick(ring)
        output_history[t] = cm.get_outputs()
        crop_history[t] = ring[:, -1].reshape(retina, retina)

    cm.lr = saved_lr
    if hasattr(cm, 'set_learn_prob'):
        cm.set_learn_prob(1.0)

    # --- Metrics ---
    # Entropy
    eps = 1e-10
    entropies = np.zeros(n_patches, dtype=np.float32)
    for p in range(n_patches):
        probs = output_history[:, p, :]
        H = -(probs * np.log(probs + eps)).sum(axis=1)
        entropies[p] = H.mean()
    H_max = np.log(n_outputs)
    norm_entropy = float(entropies.mean()) / H_max if H_max > 0 else 0.0

    # Inter-output correlation
    mean_corrs = np.zeros(n_patches, dtype=np.float32)
    for p in range(n_patches):
        series = output_history[:, p, :]
        pair_corrs = []
        for i in range(n_outputs):
            for j in range(i + 1, n_outputs):
                si, sj = series[:, i], series[:, j]
                if si.std() < 1e-8 or sj.std() < 1e-8:
                    pair_corrs.append(1.0)
                else:
                    r = np.corrcoef(si, sj)[0, 1]
                    pair_corrs.append(float(r) if not np.isnan(r) else 0.0)
        mean_corrs[p] = np.mean(pair_corrs)

    # Winner distribution
    win_counts = np.zeros((n_patches, n_outputs), dtype=int)
    for t in range(n_sample):
        winners = output_history[t].argmax(axis=1)
        for p in range(n_patches):
            win_counts[p, winners[p]] += 1
    win_fracs = win_counts.astype(np.float32) / n_sample
    global_win = win_counts.sum(axis=0)
    global_frac = global_win / global_win.sum()
    max_frac = win_fracs.max(axis=1)
    dominant = win_counts.argmax(axis=1)

    print(f"\nResults after {frames} ticks:")
    print(f"  Entropy: {entropies.mean():.3f}/{H_max:.3f} "
          f"(normalized: {norm_entropy:.3f})")
    print(f"  Inter-output corr: {mean_corrs.mean():.3f} "
          f"[{mean_corrs.min():.3f}, {mean_corrs.max():.3f}]")
    print(f"  Global winner balance: {' / '.join(f'{f:.1%}' for f in global_frac)}")
    print(f"  Dominant winner: mean={max_frac.mean():.3f} "
          f"[{max_frac.min():.3f}, {max_frac.max():.3f}]")
    for o in range(n_outputs):
        print(f"    Output {o} dominates: {(dominant == o).sum()}/{n_patches}")

    # --- Pattern grid ---
    out_dir = args.output
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    try:
        import cv2
        n_show = 16
        scale = 3
        cell = patch_sz * scale
        gap = 1
        block_h = n_show * (cell + gap) + gap
        block_w = n_outputs * (cell + gap) + gap
        block_gap = 3
        img_h = grid_n * (block_h + block_gap)
        img_w = grid_n * (block_w + block_gap)

        canvas = np.full((img_h, img_w), 40, dtype=np.uint8)
        rng_viz = np.random.RandomState(0)

        for py in range(grid_n):
            for px in range(grid_n):
                p = py * grid_n + px
                winners = output_history[:, p, :].argmax(axis=1)
                by = py * (block_h + block_gap)
                bx = px * (block_w + block_gap)
                canvas[by:by+block_h, bx:bx+block_w] = 80

                for o in range(n_outputs):
                    ticks_won = np.where(winners == o)[0]
                    if len(ticks_won) == 0:
                        continue
                    chosen = rng_viz.choice(
                        ticks_won, min(n_show, len(ticks_won)), replace=False)
                    chosen.sort()

                    for row, t_idx in enumerate(chosen):
                        patch = crop_history[t_idx,
                                             py*patch_sz:(py+1)*patch_sz,
                                             px*patch_sz:(px+1)*patch_sz]
                        patch_u8 = (np.clip(patch, 0, 1) * 255).astype(np.uint8)
                        patch_large = cv2.resize(
                            patch_u8, (cell, cell),
                            interpolation=cv2.INTER_NEAREST)

                        y0 = by + gap + row * (cell + gap)
                        x0 = bx + gap + o * (cell + gap)
                        canvas[y0:y0+cell, x0:x0+cell] = patch_large

        if out_dir:
            path = os.path.join(out_dir, "patch_patterns.png")
        else:
            path = "patch_patterns.png"
        cv2.imwrite(path, canvas)
        print(f"\nPattern grid saved: {path}")
    except ImportError:
        print("\ncv2 not available, skipping pattern grid")

    # --- Save JSON ---
    results = {
        'frames': frames,
        'column_type': args.column_type,
        'n_patches': n_patches,
        'n_outputs': n_outputs,
        'patch_size': patch_sz,
        'step': args.step,
        'entropy': {
            'mean': round(float(entropies.mean()), 4),
            'normalized': round(norm_entropy, 4),
        },
        'inter_output_correlation': round(float(mean_corrs.mean()), 4),
        'winner_distribution': {
            'global_pct': [round(float(f) * 100, 1) for f in global_frac],
            'dominant_winner_mean': round(float(max_frac.mean()), 4),
        },
    }
    if out_dir:
        json_path = os.path.join(out_dir, "patch_column_analysis.json")
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Analysis saved: {json_path}")


def main():
    parser = argparse.ArgumentParser(description='Patch column tuning tool')
    parser.add_argument('--source', type=str, required=True,
                        help='Source image (.npy or .png)')
    parser.add_argument('--retina-size', type=int, default=56)
    parser.add_argument('--patch-size', type=int, default=7)
    parser.add_argument('--outputs', type=int, default=4)
    parser.add_argument('--column-type', type=str, default='conscience')
    parser.add_argument('--lr', type=float, default=0.05)
    parser.add_argument('--alpha', type=float, default=0.01)
    parser.add_argument('--temperature', type=float, default=0.2)
    parser.add_argument('--k-active', type=int, default=2,
                        help='Top-k active outputs for homeostatic-fatigue column')
    parser.add_argument('--homeostasis-rate', type=float, default=0.02)
    parser.add_argument('--fatigue-strength', type=float, default=1.0)
    parser.add_argument('--multi-scale', action='store_true',
                        help='Use multi-scale descriptor [current, mean, delta_1, delta_half]')
    parser.add_argument('--window', type=int, default=10)
    parser.add_argument('--step', type=int, default=3)
    parser.add_argument('--eval-step', type=int, default=50,
                        help='Saccade step during eval sampling (default: 50)')
    parser.add_argument('-f', '--frames', type=int, default=10000)
    parser.add_argument('-o', '--output', type=str, default=None,
                        help='Output directory (default: print only)')
    args = parser.parse_args()
    run(args)


if __name__ == '__main__':
    main()

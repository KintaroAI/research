"""PATCH_COLUMN benchmark: hardcoded first layer diagnostic.

Hardcodes the first layer as 64 non-overlapping 7x7 patches from a 56x56
saccade crop, each processed by its own ColumnManager column (4 outputs).
The 256 column outputs feed into the model's normal pipeline.

Measures:
1. Spread distribution: do patch outputs land in different model clusters?
2. First-layer output entropy: are the 4 outputs per patch differentiated?
3. Inter-output correlation: do the 4 outputs carry different signals?

Usage:
    python main.py word2vec --preset patch_column_baseline -f 10000
"""

import os
import json
import numpy as np

from column_manager import ColumnManager, ConscienceColumn

name = 'patch_column'
description = 'Hardcoded 9x9 patch first layer → model clustering diagnostic'


def add_args(parser):
    parser.add_argument("--patch-source", type=str, default=None,
                        help="Path to source image (.npy or .png)")
    parser.add_argument("--patch-retina-size", type=int, default=56,
                        help="Crop size for saccades (default: 56)")
    parser.add_argument("--patch-size", type=int, default=7,
                        help="Patch dimension (default: 7)")
    parser.add_argument("--patch-column-lr", type=float, default=0.05,
                        help="First-layer column learning rate (default: 0.05)")
    parser.add_argument("--patch-column-window", type=int, default=10,
                        help="Temporal window (default: 10)")
    parser.add_argument("--patch-column-temperature", type=float, default=0.2,
                        help="Softmax temperature (default: 0.2)")
    parser.add_argument("--patch-column-type", type=str, default="default",
                        help="Column type: 'default' or 'conscience' (default: default)")
    parser.add_argument("--patch-column-alpha", type=float, default=0.01,
                        help="Conscience threshold lr (default: 0.01)")


def make_signal(w, h, args):
    n = w * h  # 16*16 = 256 = 64 patches * 4 outputs

    retina = getattr(args, 'patch_retina_size', 56)
    patch_sz = getattr(args, 'patch_size', 7)
    col_lr = getattr(args, 'patch_column_lr', 0.05)
    col_window = getattr(args, 'patch_column_window', 10)
    col_temp = getattr(args, 'patch_column_temperature', 0.2)
    step = getattr(args, 'saccade_step', 50)

    # Grid of patches
    grid_n = retina // patch_sz  # 80 // 9 = 8
    n_patches = grid_n * grid_n  # 64
    n_inputs = patch_sz * patch_sz  # 81
    n_outputs = 4
    assert n_patches * n_outputs == n, \
        f"Grid mismatch: {n_patches}*{n_outputs}={n_patches*n_outputs} != {n}"

    # Load source image
    patch_source = getattr(args, 'patch_source', None) or \
                   getattr(args, 'signal_source', None)
    assert patch_source is not None, "Need --patch-source or --signal-source"

    if patch_source.endswith('.npy'):
        source = np.load(patch_source).astype(np.float32)
    else:
        import cv2
        img = cv2.imread(patch_source, cv2.IMREAD_GRAYSCALE)
        source = img.astype(np.float32) / 255.0

    if source.ndim == 3:
        # Convert to grayscale if multi-channel
        source = source.mean(axis=2)

    src_h, src_w = source.shape
    assert src_h >= retina and src_w >= retina, \
        f"Source {src_w}x{src_h} too small for {retina}x{retina} retina"

    # Create first-layer columns
    col_type = getattr(args, 'patch_column_type', 'default')
    col_alpha = getattr(args, 'patch_column_alpha', 0.01)
    if col_type == 'conscience':
        first_layer_cm = ConscienceColumn(
            m=n_patches, n_outputs=n_outputs, max_inputs=n_inputs,
            window=col_window, temperature=col_temp, lr=col_lr,
            alpha=col_alpha,
        )
    elif col_type == 'predictive':
        from column_manager import PredictiveColumn
        n_heads = 7 if n_inputs % 7 == 0 else 1
        first_layer_cm = PredictiveColumn(
            m=n_patches, n_outputs=n_outputs, max_inputs=n_inputs,
            window=col_window, temperature=col_temp, lr=0.001,
            n_heads=n_heads,
        )
    elif col_type == 'recon':
        from column_manager import ReconColumn
        n_heads = 7 if n_inputs % 7 == 0 else 1
        first_layer_cm = ReconColumn(
            m=n_patches, n_outputs=n_outputs, max_inputs=n_inputs,
            window=col_window, temperature=col_temp, lr=0.001,
            n_heads=n_heads,
        )
    else:
        first_layer_cm = ColumnManager(
            m=n_patches, n_outputs=n_outputs, max_inputs=n_inputs,
            window=col_window, temperature=col_temp, lr=col_lr,
            mode='kmeans', entropy_scaled_lr=True,
        )

    # Pre-wire: patch (px, py) gets neurons at retina positions
    retina_n = retina * retina  # 6400
    for py in range(grid_n):
        for px in range(grid_n):
            patch_id = py * grid_n + px
            slot = 0
            for dy in range(patch_sz):
                for dx in range(patch_sz):
                    neuron_id = (py * patch_sz + dy) * retina + (px * patch_sz + dx)
                    first_layer_cm.slot_map[patch_id, slot] = neuron_id
                    slot += 1

    # Ring buffer: (retina_n, window) for ColumnManager.tick()
    ring = np.zeros((retina_n, col_window), dtype=np.float32)

    # Saccade state
    rng = np.random.RandomState(42)
    max_dy = src_h - retina
    max_dx = src_w - retina
    walk_y = rng.randint(0, max_dy + 1)
    walk_x = rng.randint(0, max_dx + 1)

    def tick_fn(t):
        nonlocal walk_y, walk_x
        # 1. Saccade walk
        walk_y = np.clip(walk_y + rng.randint(-step, step + 1), 0, max_dy)
        walk_x = np.clip(walk_x + rng.randint(-step, step + 1), 0, max_dx)
        # 2. Extract crop
        crop = source[walk_y:walk_y+retina, walk_x:walk_x+retina].ravel()
        # 3. Shift ring buffer, append new frame
        ring[:, :-1] = ring[:, 1:]
        ring[:, -1] = crop
        # 4. First-layer column tick
        first_layer_cm.tick(ring)
        # 5. Return 64*4 = 256 column outputs
        return first_layer_cm.get_outputs().ravel()

    metadata = {
        'w': w, 'h': h, 'n': n,
        'retina': retina,
        'patch_sz': patch_sz,
        'grid_n': grid_n,
        'n_patches': n_patches,
        'n_outputs': n_outputs,
        'first_layer_cm': first_layer_cm,
        'ring': ring,
        'source': source,
    }

    print(f"  signal buffer: PATCH_COLUMN ({n_patches} patches of {patch_sz}x{patch_sz}, "
          f"{n_outputs} outputs each = {n} neurons, "
          f"retina={retina}x{retina}, step={step})")

    return tick_fn, metadata


def analyze(metadata, cluster_mgr, signals, tick_counter, T, output_dir):
    if cluster_mgr is None or not cluster_mgr.initialized:
        print("  PATCH_COLUMN: cluster_mgr not initialized, skipping analysis")
        return

    n_patches = metadata['n_patches']
    n_outputs = metadata['n_outputs']
    n = metadata['n']
    first_layer_cm = metadata['first_layer_cm']

    n_total = cluster_mgr.n
    most_recent = cluster_mgr.cluster_ids[
        np.arange(n_total), cluster_mgr.pointers]

    # ---------------------------------------------------------------
    # Metric 1: Spread distribution
    # For each patch, count unique model clusters among its 4 neurons
    # ---------------------------------------------------------------
    spread_counts = []
    spread_hist = {1: 0, 2: 0, 3: 0, 4: 0}
    for p in range(n_patches):
        neuron_ids = [p * n_outputs + o for o in range(n_outputs)]
        clusters = set()
        for nid in neuron_ids:
            if nid < n_total:
                clusters.add(int(most_recent[nid]))
        n_unique = len(clusters)
        spread_counts.append(n_unique)
        if n_unique in spread_hist:
            spread_hist[n_unique] += 1

    mean_spread = np.mean(spread_counts)

    print(f"  PATCH_COLUMN spread distribution:")
    labels = {1: 'all-4-same', 2: '3+1 or 2+2', 3: '2+1+1', 4: 'all-different'}
    for k in [1, 2, 3, 4]:
        print(f"    {labels[k]}: {spread_hist[k]}/{n_patches}")
    print(f"    Mean spread: {mean_spread:.2f} (1.0=collapsed, 4.0=differentiated)")

    # ---------------------------------------------------------------
    # Metric 2 & 3: Sample 500 ticks post-training
    # ---------------------------------------------------------------
    print("  PATCH_COLUMN: sampling 500 ticks for entropy/correlation...")
    tick_fn = metadata.get('_tick_fn')
    if tick_fn is None:
        print("    (no tick_fn in metadata, skipping Metrics 2-4)")
        results = _save_results(output_dir, spread_hist, mean_spread,
                                None, None, None, None,
                                n_patches, n_outputs)
        return

    total_ticks = metadata.get('_total_ticks', 0)
    n_sample = 500
    # Collect per-patch outputs over time: (n_sample, n_patches, n_outputs)
    output_history = np.zeros((n_sample, n_patches, n_outputs), dtype=np.float32)

    for t in range(n_sample):
        tick_fn(total_ticks + T + t)
        output_history[t] = first_layer_cm.get_outputs()

    # --- Metric 2: Per-patch Shannon entropy of output probabilities ---
    # Average entropy across samples
    eps = 1e-10
    entropies = np.zeros(n_patches, dtype=np.float32)
    for p in range(n_patches):
        probs = output_history[:, p, :]  # (n_sample, n_outputs)
        # Per-tick entropy, then average
        H = -(probs * np.log(probs + eps)).sum(axis=1)
        entropies[p] = H.mean()

    H_max = np.log(n_outputs)
    mean_entropy = float(entropies.mean())
    normalized_entropy = mean_entropy / H_max if H_max > 0 else 0.0

    print(f"  PATCH_COLUMN output entropy:")
    print(f"    Mean entropy: {mean_entropy:.3f} / {H_max:.3f} "
          f"(normalized: {normalized_entropy:.3f})")
    print(f"    Per-patch range: [{entropies.min():.3f}, {entropies.max():.3f}]")

    # --- Metric 3: Inter-output correlation per patch ---
    mean_corrs = np.zeros(n_patches, dtype=np.float32)
    for p in range(n_patches):
        series = output_history[:, p, :]  # (n_sample, n_outputs)
        pair_corrs = []
        for i in range(n_outputs):
            for j in range(i + 1, n_outputs):
                si, sj = series[:, i], series[:, j]
                if si.std() < 1e-8 or sj.std() < 1e-8:
                    pair_corrs.append(1.0)  # constant = maximally correlated
                else:
                    r = np.corrcoef(si, sj)[0, 1]
                    pair_corrs.append(float(r) if not np.isnan(r) else 0.0)
        mean_corrs[p] = np.mean(pair_corrs)

    mean_inter_corr = float(mean_corrs.mean())

    print(f"  PATCH_COLUMN inter-output correlation:")
    print(f"    Mean pairwise r: {mean_inter_corr:.3f} "
          f"(1.0=identical, 0.0=independent)")
    print(f"    Per-patch range: [{mean_corrs.min():.3f}, {mean_corrs.max():.3f}]")

    # --- Metric 4: Per-output winner distribution ---
    win_counts = np.zeros((n_patches, n_outputs), dtype=int)
    for t in range(n_sample):
        winners = output_history[t].argmax(axis=1)  # (n_patches,)
        for p in range(n_patches):
            win_counts[p, winners[p]] += 1

    win_fracs = win_counts.astype(np.float32) / n_sample
    global_win = win_counts.sum(axis=0)
    global_frac = global_win / global_win.sum()
    max_frac = win_fracs.max(axis=1)  # per-patch dominant winner fraction

    print(f"  PATCH_COLUMN winner distribution:")
    print(f"    Global: {' / '.join(f'{f:.1%}' for f in global_frac)}")
    print(f"    Per-patch dominant winner: mean={max_frac.mean():.3f}, "
          f"range=[{max_frac.min():.3f}, {max_frac.max():.3f}]")
    dominant = win_counts.argmax(axis=1)
    for o in range(n_outputs):
        print(f"    Output {o} dominates: {(dominant == o).sum()}/{n_patches} patches")

    _save_results(output_dir, spread_hist, mean_spread,
                  mean_entropy, mean_inter_corr,
                  global_frac, max_frac, n_patches, n_outputs)


def _save_results(output_dir, spread_hist, mean_spread,
                  mean_entropy, mean_inter_corr,
                  global_win_frac, per_patch_max_frac,
                  n_patches, n_outputs):
    results = {
        'n_patches': n_patches,
        'n_outputs': n_outputs,
        'spread': {
            'histogram': {str(k): v for k, v in spread_hist.items()},
            'mean': round(mean_spread, 3),
        },
    }
    if mean_entropy is not None:
        results['entropy'] = {
            'mean': round(mean_entropy, 4),
            'H_max': round(float(np.log(n_outputs)), 4),
            'normalized': round(mean_entropy / max(np.log(n_outputs), 1e-8), 4),
        }
    if mean_inter_corr is not None:
        results['inter_output_correlation'] = {
            'mean_pairwise_r': round(mean_inter_corr, 4),
        }
    if global_win_frac is not None:
        results['winner_distribution'] = {
            'global_pct': [round(float(f) * 100, 1) for f in global_win_frac],
            'dominant_winner_mean': round(float(per_patch_max_frac.mean()), 4),
            'dominant_winner_range': [round(float(per_patch_max_frac.min()), 4),
                                      round(float(per_patch_max_frac.max()), 4)],
        }

    if output_dir:
        path = os.path.join(output_dir, "patch_column_analysis.json")
        with open(path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"  PATCH_COLUMN analysis saved: {path}")

    return results

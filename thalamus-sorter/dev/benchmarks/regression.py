"""REGRESSION benchmark: verify core perception capabilities.

Runs a standardized signal and checks four criteria:
1. TOPO — topographic map: nearby pixels are embedding neighbors
2. COLUMN_DIFF — columns differentiate (clear winners, not uniform)
3. SEPARATION — sensory/feedback neurons form distinct clusters
4. STABILITY — cluster assignments stable over time

Uses a synthetic gradient image for deterministic, fast testing.
All criteria checked after training — single pass/fail output.

Usage:
    python main.py word2vec --signal-source regression -f 5000
"""

import os
import json
import numpy as np

name = 'regression'
description = 'Regression tests: topo, column diff, separation, stability'


def add_args(parser):
    parser.add_argument("--regression-hold", type=int, default=1,
                        help="Unused (for CLI compat)")


def make_signal(w, h, args):
    """Synthetic gradient: smooth spatial structure for topographic learning."""
    n = w * h
    rng = np.random.RandomState(42)

    # Pre-generate a smooth gradient "image" for saccade-like crops
    src_size = 128
    xs = np.linspace(0, 1, src_size)
    ys = np.linspace(0, 1, src_size)
    xx, yy = np.meshgrid(xs, ys)
    source = (0.5 * np.sin(xx * 6) + 0.5 * np.cos(yy * 4) + 0.3 * xx * yy).astype(np.float32)

    max_dy = src_size - h
    max_dx = src_size - w
    walk_y = rng.randint(0, max_dy + 1)
    walk_x = rng.randint(0, max_dx + 1)
    saccade_step = 3

    feature_log = []

    def tick_fn(t):
        nonlocal walk_y, walk_x
        walk_y = np.clip(walk_y + rng.randint(-saccade_step, saccade_step + 1),
                         0, max_dy)
        walk_x = np.clip(walk_x + rng.randint(-saccade_step, saccade_step + 1),
                         0, max_dx)
        crop = source[walk_y:walk_y+h, walk_x:walk_x+w].ravel()
        feature_log.append((t, walk_x, walk_y))
        return crop

    metadata = {
        'feature_log': feature_log,
        'w': w, 'h': h, 'n': n,
    }

    print(f"  signal buffer: REGRESSION synthetic gradient ({src_size}×{src_size})")
    return tick_fn, metadata


def analyze(metadata, cluster_mgr, signals, tick_counter, T, output_dir):
    """Run all regression checks and report pass/fail."""
    import torch

    w, h = metadata['w'], metadata['h']
    n = metadata['n']
    results = {}
    all_pass = True

    print("  REGRESSION checks:")

    # --- 1. TOPO: topographic map quality ---
    if cluster_mgr is not None and cluster_mgr._dsolver is not None:
        dsolver = cluster_mgr._dsolver
        if dsolver.knn_k > 0:
            sig_channels = 1
            spatial = dsolver.knn_spatial_accuracy(w, radius=5, channels=sig_channels,
                                                   n_eval=n)
            topo_pass = spatial > 0.3
            results['topo'] = {'spatial_5px': round(spatial, 4), 'pass': topo_pass}
            status = 'PASS' if topo_pass else 'FAIL'
            print(f"    TOPO: spatial_5px={spatial:.4f} {'✓' if topo_pass else '✗'} (>{0.3})")
            if not topo_pass:
                all_pass = False
        else:
            print(f"    TOPO: skipped (no --knn-track)")
            results['topo'] = {'skip': True}

    # --- 2. COLUMN_DIFF: column differentiation ---
    if cluster_mgr is not None and cluster_mgr.column_mgr is not None:
        cm = cluster_mgr.column_mgr
        # Sample 100 random signals to get output statistics
        rng = np.random.RandomState(123)
        max_probs = []
        for _ in range(100):
            fake_sig = rng.randn(cluster_mgr.n, cm.window).astype(np.float32) * 0.1
            cm.tick(fake_sig)
            probs = cm._outputs
            max_probs.extend(probs.max(axis=1).tolist())
        max_probs = np.array(max_probs)
        frac_differentiated = (max_probs > 0.4).mean()
        diff_pass = frac_differentiated > 0.3
        results['column_diff'] = {
            'frac_gt_04': round(float(frac_differentiated), 4),
            'mean_max_prob': round(float(max_probs.mean()), 4),
            'pass': diff_pass
        }
        status = 'PASS' if diff_pass else 'FAIL'
        print(f"    COLUMN_DIFF: {frac_differentiated*100:.1f}% > 0.4 "
              f"{'✓' if diff_pass else '✗'} (>30%)")
        if not diff_pass:
            all_pass = False

    # --- 3. SEPARATION: sensory vs feedback cluster purity ---
    if cluster_mgr is not None and cluster_mgr.initialized:
        n_sensory = cluster_mgr.n_sensory
        most_recent = cluster_mgr.cluster_ids[
            np.arange(cluster_mgr.n), cluster_mgr.pointers]
        m = cluster_mgr.m
        mixed = 0
        alive = 0
        for c in range(m):
            members = np.where(most_recent == c)[0]
            if len(members) == 0:
                continue
            alive += 1
            has_s = (members < n_sensory).any()
            has_f = (members >= n_sensory).any()
            if has_s and has_f:
                mixed += 1
        mixed_frac = mixed / max(alive, 1)
        sep_pass = mixed_frac < 0.1
        results['separation'] = {
            'mixed_clusters': mixed,
            'alive_clusters': alive,
            'mixed_frac': round(mixed_frac, 4),
            'pass': sep_pass
        }
        status = 'PASS' if sep_pass else 'FAIL'
        print(f"    SEPARATION: {mixed}/{alive} mixed ({mixed_frac*100:.1f}%) "
              f"{'✓' if sep_pass else '✗'} (<10%)")
        if not sep_pass:
            all_pass = False

    # --- 4. STABILITY: cluster assignment stability ---
    if cluster_mgr is not None and cluster_mgr.initialized:
        # Run 200 more ticks and check stability
        tick_fn = metadata.get('_tick_fn')
        if tick_fn is not None:
            prev = cluster_mgr.cluster_ids[
                np.arange(cluster_mgr.n), cluster_mgr.pointers].copy()
            for t in range(200):
                sig_t = tick_fn(metadata['_total_ticks'] + T + t)
                col_t = tick_counter[0] % T
                signals[:n, col_t] = torch.from_numpy(sig_t).to(signals.device)
                tick_counter[0] += 1
            curr = cluster_mgr.cluster_ids[
                np.arange(cluster_mgr.n), cluster_mgr.pointers]
            stability = (prev == curr).mean()
            stab_pass = stability > 0.8
            results['stability'] = {
                'stability': round(float(stability), 4),
                'pass': stab_pass
            }
            status = 'PASS' if stab_pass else 'FAIL'
            print(f"    STABILITY: {stability:.4f} "
                  f"{'✓' if stab_pass else '✗'} (>0.8)")
            if not stab_pass:
                all_pass = False

    # --- Summary ---
    print(f"  REGRESSION: {'ALL PASS ✓' if all_pass else 'SOME FAILED ✗'}")

    if output_dir:
        results['all_pass'] = all_pass
        path = os.path.join(output_dir, "regression.json")
        with open(path, 'w') as f:
            json.dump(results, f, indent=2)

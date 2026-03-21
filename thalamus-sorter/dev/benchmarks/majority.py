"""MAJORITY benchmark: 3 binary inputs, output = majority vote.

Tests whether lateral connections can combine 3 inputs (XOR only needs 2).
MAJ(A,B,C) = 1 if two or more of A,B,C are 1.

Grid layout (2×2 quadrants):
    ┌───┬───┐
    │ A │ B │
    ├───┼───┤
    │ C │MAJ│
    └───┴───┘

Usage:
    python main.py word2vec --signal-source majority --majority-hold 50 --column-lateral ...
"""

import os
import json
import numpy as np

name = 'majority'
description = 'Majority vote: 3 binary inputs A,B,C → MAJ = majority(A,B,C)'


def add_args(parser):
    parser.add_argument("--majority-hold", type=int, default=50,
                        help="Ticks to hold each A,B,C state (default: 50)")
    parser.add_argument("--majority-noise", type=float, default=0.1,
                        help="Per-neuron noise std (default: 0.1)")


def make_signal(w, h, args):
    n = w * h
    half_w = w // 2
    half_h = h // 2
    noise_std = getattr(args, 'majority_noise', 0.1)
    hold_ticks = getattr(args, 'majority_hold', 50)
    rng = np.random.RandomState(42)

    coords_x = np.arange(n) % w
    coords_y = np.arange(n) // w
    region_A = (coords_x < half_w) & (coords_y < half_h)
    region_B = (coords_x >= half_w) & (coords_y < half_h)
    region_C = (coords_x < half_w) & (coords_y >= half_h)
    region_MAJ = (coords_x >= half_w) & (coords_y >= half_h)

    state = {'A': 0, 'B': 0, 'C': 0, 'last_change': -hold_ticks}
    feature_log = []

    def tick_fn(t):
        if t - state['last_change'] >= hold_ticks:
            state['A'] = rng.randint(0, 2)
            state['B'] = rng.randint(0, 2)
            state['C'] = rng.randint(0, 2)
            state['last_change'] = t

        A, B, C = state['A'], state['B'], state['C']
        maj_val = int(A + B + C >= 2)
        feature_log.append((t, A, B, C, maj_val))

        sig = np.zeros(n, dtype=np.float32)
        sig[region_A] = float(A)
        sig[region_B] = float(B)
        sig[region_C] = float(C)
        sig[region_MAJ] = float(maj_val)
        sig += rng.randn(n).astype(np.float32) * noise_std
        return sig

    metadata = {
        'regions': {'A': region_A, 'B': region_B, 'C': region_C, 'MAJ': region_MAJ},
        'region_names': ['A', 'B', 'C', 'MAJ'],
        'feature_log': feature_log,
        'w': w, 'h': h, 'n': n,
        'noise_std': noise_std,
        'hold_ticks': hold_ticks,
    }

    print(f"  signal buffer: MAJORITY synthetic "
          f"(noise={noise_std}, hold={hold_ticks})")

    return tick_fn, metadata


def analyze(metadata, cluster_mgr, signals, tick_counter, T, output_dir):
    import torch

    if cluster_mgr is None or not cluster_mgr.initialized:
        return
    if cluster_mgr.column_mgr is None:
        return

    print("  MAJORITY analysis: sampling 500 ticks...")
    tick_fn = metadata['_tick_fn']
    col_history = []
    n_sensory = metadata['n']

    for t in range(500):
        sig_t = tick_fn(metadata['_total_ticks'] + T + t)
        col_t = tick_counter[0] % T
        signals[:n_sensory, col_t] = torch.from_numpy(sig_t).to(signals.device)
        tick_counter[0] += 1
        if cluster_mgr._signals is not None and cluster_mgr._signal_T > 0:
            cw = cluster_mgr.column_mgr.window
            indices = [(tick_counter[0] - 1 - i) % T for i in range(cw)]
            sw = cluster_mgr._signals[:, indices].cpu().numpy()
            cluster_mgr.column_mgr.tick(sw)
            col_history.append((t, cluster_mgr.column_mgr.get_outputs()))

    log = np.array(metadata['feature_log'])
    if len(log) == 0 or len(col_history) == 0:
        return

    log_ticks = set(log[:, 0].astype(int))
    aligned = []
    for tick, outputs in col_history:
        if tick in log_ticks:
            idx = np.searchsorted(log[:, 0], tick)
            if idx < len(log) and int(log[idx, 0]) == tick:
                aligned.append((log[idx, 1:], outputs))

    if len(aligned) < 10:
        print(f"  MAJORITY: only {len(aligned)} aligned ticks, skipping")
        return

    features = np.array([a[0] for a in aligned])    # (T, 4): A, B, C, MAJ
    all_outputs = np.array([a[1] for a in aligned])  # (T, m, n_out)
    n_ticks, m, n_out = all_outputs.shape
    feature_names = ['A', 'B', 'C', 'MAJ']

    results = np.zeros((m, n_out, 4))
    for fi, fname in enumerate(feature_names):
        feat = features[:, fi]
        if feat.std() < 1e-8:
            continue
        for c in range(m):
            for o in range(n_out):
                out = all_outputs[:, c, o]
                if out.std() < 1e-8:
                    continue
                r = np.corrcoef(feat, out)[0, 1]
                if not np.isnan(r):
                    results[c, o, fi] = r

    best = {}
    for fi, fname in enumerate(feature_names):
        abs_corr = np.abs(results[:, :, fi])
        best_idx = np.unravel_index(abs_corr.argmax(), abs_corr.shape)
        best[fname] = {
            'max_abs_corr': float(abs_corr.max()),
            'best_column': int(best_idx[0]),
            'best_output': int(best_idx[1]),
            'mean_abs_corr': float(abs_corr.mean()),
        }

    print(f"  MAJORITY results ({n_ticks} ticks):")
    for fname, info in best.items():
        print(f"    {fname:3s}: max|r|={info['max_abs_corr']:.3f} "
              f"(col {info['best_column']}, out {info['best_output']}), "
              f"mean|r|={info['mean_abs_corr']:.3f}")

    if output_dir:
        path = os.path.join(output_dir, "majority_analysis.json")
        with open(path, 'w') as f:
            json.dump(best, f, indent=2)
        print(f"  MAJORITY analysis saved: {path}")

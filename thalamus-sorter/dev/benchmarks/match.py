"""MATCH benchmark: spatial comparison of two distant regions.

Tests whether lateral connections can compare multi-valued patterns
across distant clusters. Harder than XOR — requires comparing 4-valued
patterns, not just binary bits.

Grid layout (2×2 quadrants):
    ┌───┬───┐
    │ P1│ P2│   P1, P2 each get one of 4 patterns (0.0, 0.33, 0.67, 1.0)
    ├───┼───┤
    │ EQ│NEQ│   EQ = 1 if P1 == P2, NEQ = 1 if P1 ≠ P2
    └───┴───┘

Usage:
    python main.py word2vec --signal-source match --match-hold 50 --column-lateral ...
"""

import os
import json
import numpy as np

name = 'match'
description = 'Spatial comparison: EQ = (P1 == P2), NEQ = (P1 ≠ P2)'


def add_args(parser):
    parser.add_argument("--match-hold", type=int, default=50,
                        help="Ticks to hold each pattern state (default: 50)")
    parser.add_argument("--match-noise", type=float, default=0.1,
                        help="Per-neuron noise std (default: 0.1)")


def make_signal(w, h, args):
    n = w * h
    half_w = w // 2
    half_h = h // 2
    noise_std = getattr(args, 'match_noise', 0.1)
    hold_ticks = getattr(args, 'match_hold', 50)
    rng = np.random.RandomState(42)
    patterns = [0.0, 0.33, 0.67, 1.0]

    coords_x = np.arange(n) % w
    coords_y = np.arange(n) // w
    region_P1 = (coords_x < half_w) & (coords_y < half_h)
    region_P2 = (coords_x >= half_w) & (coords_y < half_h)
    region_EQ = (coords_x < half_w) & (coords_y >= half_h)
    region_NEQ = (coords_x >= half_w) & (coords_y >= half_h)

    state = {'P1': 0, 'P2': 0, 'last_change': -hold_ticks}
    feature_log = []

    def tick_fn(t):
        if t - state['last_change'] >= hold_ticks:
            state['P1'] = rng.randint(0, len(patterns))
            state['P2'] = rng.randint(0, len(patterns))
            state['last_change'] = t

        p1_idx, p2_idx = state['P1'], state['P2']
        p1_val = patterns[p1_idx]
        p2_val = patterns[p2_idx]
        eq_val = float(p1_idx == p2_idx)
        neq_val = float(p1_idx != p2_idx)
        feature_log.append((t, p1_val, p2_val, eq_val, neq_val))

        sig = np.zeros(n, dtype=np.float32)
        sig[region_P1] = p1_val
        sig[region_P2] = p2_val
        sig[region_EQ] = eq_val
        sig[region_NEQ] = neq_val
        sig += rng.randn(n).astype(np.float32) * noise_std
        return sig

    metadata = {
        'regions': {'P1': region_P1, 'P2': region_P2,
                    'EQ': region_EQ, 'NEQ': region_NEQ},
        'region_names': ['P1', 'P2', 'EQ', 'NEQ'],
        'feature_log': feature_log,
        'w': w, 'h': h, 'n': n,
        'noise_std': noise_std,
        'hold_ticks': hold_ticks,
    }

    print(f"  signal buffer: MATCH synthetic "
          f"(noise={noise_std}, hold={hold_ticks}, 4 patterns)")

    return tick_fn, metadata


def analyze(metadata, cluster_mgr, signals, tick_counter, T, output_dir):
    import torch

    if cluster_mgr is None or not cluster_mgr.initialized:
        return
    if cluster_mgr.column_mgr is None:
        return

    print("  MATCH analysis: sampling 500 ticks...")
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
        print(f"  MATCH: only {len(aligned)} aligned ticks, skipping")
        return

    features = np.array([a[0] for a in aligned])
    all_outputs = np.array([a[1] for a in aligned])
    n_ticks, m, n_out = all_outputs.shape
    feature_names = ['P1', 'P2', 'EQ', 'NEQ']

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

    print(f"  MATCH results ({n_ticks} ticks):")
    for fname, info in best.items():
        print(f"    {fname:3s}: max|r|={info['max_abs_corr']:.3f} "
              f"(col {info['best_column']}, out {info['best_output']}), "
              f"mean|r|={info['mean_abs_corr']:.3f}")

    if output_dir:
        path = os.path.join(output_dir, "match_analysis.json")
        with open(path, 'w') as f:
            json.dump(best, f, indent=2)
        print(f"  MATCH analysis saved: {path}")

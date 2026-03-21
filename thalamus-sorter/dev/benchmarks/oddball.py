"""ODDBALL benchmark: novelty/deviance detection.

Tests whether a column can detect that its region's value differs from
the majority of other regions. Requires lateral context — knowing what
"everyone else" is showing to identify the outlier.

Grid layout (2×2 quadrants):
    ┌───┬───┐
    │ R0│ R1│   3 regions show the SAME random value
    ├───┼───┤   1 region (the oddball) shows a DIFFERENT value
    │ R2│ODD│   ODD region = which region is the oddball (encoded as 0/0.33/0.67/1)
    └───┴───┘

The oddball position rotates randomly each hold period. The ODD output
region encodes WHICH region is the oddball (not whether there IS one —
there always is one).

Usage:
    python main.py word2vec --signal-source oddball --oddball-hold 50 --column-lateral ...
"""

import os
import json
import numpy as np

name = 'oddball'
description = 'Novelty detection: 3 same + 1 different, detect which is odd'


def add_args(parser):
    parser.add_argument("--oddball-hold", type=int, default=50,
                        help="Ticks to hold each state (default: 50)")
    parser.add_argument("--oddball-noise", type=float, default=0.1,
                        help="Per-neuron noise std (default: 0.1)")


def make_signal(w, h, args):
    n = w * h
    half_w = w // 2
    half_h = h // 2
    noise_std = getattr(args, 'oddball_noise', 0.1)
    hold_ticks = getattr(args, 'oddball_hold', 50)
    rng = np.random.RandomState(42)

    coords_x = np.arange(n) % w
    coords_y = np.arange(n) // w
    regions = [
        (coords_x < half_w) & (coords_y < half_h),    # R0: top-left
        (coords_x >= half_w) & (coords_y < half_h),   # R1: top-right
        (coords_x < half_w) & (coords_y >= half_h),   # R2: bottom-left
    ]
    region_ODD = (coords_x >= half_w) & (coords_y >= half_h)  # bottom-right

    # Oddball identity encoded as fractional value
    odd_encoding = {0: 0.0, 1: 0.33, 2: 0.67}

    state = {'majority_val': 0.5, 'odd_val': 0.0, 'odd_idx': 0,
             'last_change': -hold_ticks}
    feature_log = []

    def tick_fn(t):
        if t - state['last_change'] >= hold_ticks:
            state['majority_val'] = float(rng.rand())
            state['odd_val'] = float(rng.rand())
            # Ensure oddball differs noticeably
            while abs(state['odd_val'] - state['majority_val']) < 0.3:
                state['odd_val'] = float(rng.rand())
            state['odd_idx'] = rng.randint(0, 3)
            state['last_change'] = t

        maj_v = state['majority_val']
        odd_v = state['odd_val']
        odd_i = state['odd_idx']
        feature_log.append((t, float(maj_v), float(odd_v), odd_i,
                            odd_encoding[odd_i]))

        sig = np.zeros(n, dtype=np.float32)
        for r in range(3):
            if r == odd_i:
                sig[regions[r]] = odd_v
            else:
                sig[regions[r]] = maj_v
        sig[region_ODD] = odd_encoding[odd_i]
        sig += rng.randn(n).astype(np.float32) * noise_std
        return sig

    metadata = {
        'regions': {'R0': regions[0], 'R1': regions[1], 'R2': regions[2],
                    'ODD': region_ODD},
        'region_names': ['maj_val', 'odd_val', 'odd_idx', 'odd_enc'],
        'feature_log': feature_log,
        'w': w, 'h': h, 'n': n,
        'noise_std': noise_std,
        'hold_ticks': hold_ticks,
    }

    print(f"  signal buffer: ODDBALL synthetic "
          f"(noise={noise_std}, hold={hold_ticks})")

    return tick_fn, metadata


def analyze(metadata, cluster_mgr, signals, tick_counter, T, output_dir):
    import torch

    if cluster_mgr is None or not cluster_mgr.initialized:
        return
    if cluster_mgr.column_mgr is None:
        return

    print("  ODDBALL analysis: sampling 500 ticks...")
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
        print(f"  ODDBALL: only {len(aligned)} aligned ticks, skipping")
        return

    features = np.array([a[0] for a in aligned])
    all_outputs = np.array([a[1] for a in aligned])
    n_ticks, m, n_out = all_outputs.shape

    # Correlate with: odd_idx (which region is odd) and odd_enc (encoded value)
    feature_names = ['maj_val', 'odd_val', 'odd_idx', 'odd_enc']

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
                results[c, o, fi] = np.corrcoef(feat, out)[0, 1]

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

    print(f"  ODDBALL results ({n_ticks} ticks):")
    for fname, info in best.items():
        print(f"    {fname:7s}: max|r|={info['max_abs_corr']:.3f} "
              f"(col {info['best_column']}, out {info['best_output']}), "
              f"mean|r|={info['mean_abs_corr']:.3f}")

    if output_dir:
        path = os.path.join(output_dir, "oddball_analysis.json")
        with open(path, 'w') as f:
            json.dump(best, f, indent=2)
        print(f"  ODDBALL analysis saved: {path}")

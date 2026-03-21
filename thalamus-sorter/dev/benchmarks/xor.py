"""XOR benchmark: 4 quadrants with binary features A, B, XOR=A^B, AND=A&B.

Tests whether columns can detect non-linear cross-cluster features via
lateral connections. Without lateral connections, XOR is undetectable
(max|r|~0.2). With lateral connections, XOR detection reaches r>0.6.

Usage:
    python main.py word2vec --signal-source xor --xor-hold 50 --column-lateral ...
"""

import os
import json
import numpy as np

name = 'xor'
description = 'XOR non-linear feature detection (4 quadrants: A, B, XOR, AND)'


def add_args(parser):
    """Add XOR-specific CLI flags."""
    parser.add_argument("--xor-noise", type=float, default=0.1,
                        help="Per-neuron noise std for XOR signal (default: 0.1)")
    parser.add_argument("--xor-hold", type=int, default=5,
                        help="Ticks to hold each A,B state (default: 5)")


def make_signal(w, h, args):
    """Create XOR signal generator.

    Returns:
        tick_fn: callable(t) -> (n_sensory,) float32 signal
        metadata: dict with region masks, feature log
    """
    n = w * h
    half_w = w // 2
    half_h = h // 2
    noise_std = getattr(args, 'xor_noise', 0.1)
    hold_ticks = getattr(args, 'xor_hold', 5)
    rng = np.random.RandomState(42)

    # Region masks
    coords_x = np.arange(n) % w
    coords_y = np.arange(n) // w
    region1 = (coords_x < half_w) & (coords_y < half_h)   # top-left: A
    region2 = (coords_x >= half_w) & (coords_y < half_h)  # top-right: B
    region3 = (coords_x < half_w) & (coords_y >= half_h)  # bottom-left: XOR
    region4 = (coords_x >= half_w) & (coords_y >= half_h) # bottom-right: AND

    state = {'A': 0, 'B': 0, 'last_change': -hold_ticks}
    feature_log = []

    def tick_fn(t):
        if t - state['last_change'] >= hold_ticks:
            state['A'] = rng.randint(0, 2)
            state['B'] = rng.randint(0, 2)
            state['last_change'] = t

        A, B = state['A'], state['B']
        xor_val = A ^ B
        and_val = A & B
        feature_log.append((t, A, B, xor_val, and_val))

        sig = np.zeros(n, dtype=np.float32)
        sig[region1] = float(A)
        sig[region2] = float(B)
        sig[region3] = float(xor_val)
        sig[region4] = float(and_val)
        sig += rng.randn(n).astype(np.float32) * noise_std
        return sig

    metadata = {
        'regions': {'A': region1, 'B': region2, 'XOR': region3, 'AND': region4},
        'region_names': ['A', 'B', 'XOR', 'AND'],
        'feature_log': feature_log,
        'w': w, 'h': h, 'n': n,
        'noise_std': noise_std,
        'hold_ticks': hold_ticks,
    }

    print(f"  signal buffer: XOR synthetic "
          f"(noise={noise_std}, hold={hold_ticks})")

    return tick_fn, metadata


def analyze(metadata, cluster_mgr, signals, tick_counter, T, output_dir):
    """Post-training XOR analysis: correlate column outputs with features."""
    import torch

    if cluster_mgr is None or not cluster_mgr.initialized:
        return
    if cluster_mgr.column_mgr is None:
        return

    print("  XOR analysis: sampling 500 ticks...")
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

    # Compute correlations
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
        print(f"  XOR: only {len(aligned)} aligned ticks, skipping")
        return

    features = np.array([a[0] for a in aligned])
    all_outputs = np.array([a[1] for a in aligned])
    n_ticks, m, n_out = all_outputs.shape
    feature_names = ['A', 'B', 'XOR', 'AND']

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

    print(f"  XOR results ({n_ticks} ticks):")
    for fname, info in best.items():
        print(f"    {fname:3s}: max|r|={info['max_abs_corr']:.3f} "
              f"(col {info['best_column']}, out {info['best_output']}), "
              f"mean|r|={info['mean_abs_corr']:.3f}")

    if output_dir:
        xor_path = os.path.join(output_dir, "xor_analysis.json")
        with open(xor_path, 'w') as f:
            json.dump(best, f, indent=2)
        print(f"  XOR analysis saved: {xor_path}")

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
    """Post-training XOR analysis: correlate column outputs with features.

    Blind eval: during analysis, XOR and AND quadrants are zeroed out.
    Only A and B are provided as input. If any column output still
    correlates with XOR, the network is computing it, not just reading it.
    """
    import torch

    if cluster_mgr is None or not cluster_mgr.initialized:
        return
    if cluster_mgr.column_mgr is None:
        return

    n_sensory = metadata['n']
    regions = metadata['regions']
    xor_mask = regions['XOR']
    and_mask = regions['AND']

    # --- Classify columns by quadrant membership ---
    n_total = cluster_mgr.n
    n_outputs = cluster_mgr.column_mgr.n_outputs
    m = cluster_mgr.m
    most_recent = cluster_mgr.cluster_ids[
        np.arange(n_total), cluster_mgr.pointers]

    # For each cluster, which quadrants do its sensory neurons belong to?
    cluster_quads = {}
    for c in range(m):
        members = np.where(most_recent == c)[0]
        sensory = members[members < n_sensory]
        quads = set()
        for qname, qmask in regions.items():
            if np.any(qmask[sensory]):
                quads.add(qname)
        cluster_quads[c] = quads

    # Columns that contain NO XOR/AND sensory neurons (can't cheat)
    blind_cols = set(c for c in range(m)
                     if 'XOR' not in cluster_quads.get(c, set())
                     and 'AND' not in cluster_quads.get(c, set()))

    print(f"  XOR analysis: {len(blind_cols)}/{m} columns have no XOR/AND neurons")
    print("  XOR analysis: sampling 500 ticks (XOR+AND quadrants ZEROED)...")

    tick_fn = metadata['_tick_fn']
    col_history = []
    feature_log_eval = []

    for t in range(500):
        sig_t = tick_fn(metadata['_total_ticks'] + T + t)
        # Zero out XOR and AND quadrants — only A and B visible
        sig_blind = sig_t.copy()
        sig_blind[xor_mask] = 0.0
        sig_blind[and_mask] = 0.0

        col_t = tick_counter[0] % T
        signals[:n_sensory, col_t] = torch.from_numpy(sig_blind).to(signals.device)
        tick_counter[0] += 1
        if cluster_mgr._signals is not None and cluster_mgr._signal_T > 0:
            cw = cluster_mgr.column_mgr.window
            indices = [(tick_counter[0] - 1 - i) % T for i in range(cw)]
            sw = cluster_mgr._signals[:, indices].cpu().numpy()
            cluster_mgr.column_mgr.tick(sw)
            col_history.append((t, cluster_mgr.column_mgr.get_outputs()))

        # Log ground truth (even though XOR/AND are zeroed in signal)
        log = metadata['feature_log']
        if log:
            feature_log_eval.append(log[-1][1:])  # (A, B, XOR, AND)

    if len(col_history) == 0 or len(feature_log_eval) == 0:
        return

    all_outputs = np.array([o for _, o in col_history])
    features = np.array(feature_log_eval)
    n_ticks = min(len(features), all_outputs.shape[0])
    all_outputs = all_outputs[:n_ticks]
    features = features[:n_ticks]
    n_ticks, m_out, n_out = all_outputs.shape
    feature_names = ['A', 'B', 'XOR', 'AND']

    # Correlate — report both "all columns" and "blind columns only"
    results_all = np.zeros((m_out, n_out, 4))
    for fi in range(4):
        feat = features[:, fi].astype(np.float32)
        if feat.std() < 1e-8:
            continue
        for c in range(m_out):
            for o in range(n_out):
                out = all_outputs[:, c, o]
                if out.std() < 1e-8:
                    continue
                r = np.corrcoef(feat, out)[0, 1]
                if not np.isnan(r):
                    results_all[c, o, fi] = r

    print(f"  XOR blind eval ({n_ticks} ticks, XOR+AND zeroed in signal):")
    best = {}
    for fi, fname in enumerate(feature_names):
        # Best across ALL columns
        abs_all = np.abs(results_all[:, :, fi])
        idx_all = np.unravel_index(abs_all.argmax(), abs_all.shape)

        # Best across BLIND columns only (no XOR/AND neurons)
        abs_blind = abs_all.copy()
        for c in range(m_out):
            if c not in blind_cols:
                abs_blind[c, :] = 0
        idx_blind = np.unravel_index(abs_blind.argmax(), abs_blind.shape)

        best[fname] = {
            'all_max_r': round(float(abs_all.max()), 4),
            'all_col': int(idx_all[0]),
            'all_out': int(idx_all[1]),
            'blind_max_r': round(float(abs_blind.max()), 4),
            'blind_col': int(idx_blind[0]),
            'blind_out': int(idx_blind[1]),
        }

        quad_info = cluster_quads.get(idx_all[0], set())
        print(f"    {fname:3s}: all max|r|={abs_all.max():.3f} "
              f"(col {idx_all[0]} {quad_info})"
              f"  blind max|r|={abs_blind.max():.3f} "
              f"(col {idx_blind[0]})")

    if output_dir:
        xor_path = os.path.join(output_dir, "xor_analysis.json")
        with open(xor_path, 'w') as f:
            json.dump(best, f, indent=2)
        print(f"  XOR analysis saved: {xor_path}")

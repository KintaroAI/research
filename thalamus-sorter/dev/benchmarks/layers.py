"""LAYERS benchmark: hierarchical feature detection.

Three levels of features, each building on the previous:
- L1: 8 independent binary signals (local features)
- L2: 4 combinations of L1 pairs (composite features)
- L3: 2 combinations of L2 pairs (abstract features)

Grid layout (16×16 = 256 neurons, 8 per group):
    Groups 0-7 (64 neurons): L1 features (8 independent binary signals)
    Groups 8-11 (32 neurons): L2 features (XOR of L1 pairs)
    Groups 12-13 (16 neurons): L3 features (XOR of L2 pairs)
    Remaining 144 neurons: zero

V1 columns should detect L1. V2 columns (via feedback/lateral from V1)
should detect L2. V3 columns should detect L3.

Usage:
    python main.py word2vec --signal-source layers --column-lateral ...
"""

import os
import json
import numpy as np

name = 'layers'
description = 'Hierarchical: L1(8 signals) → L2(4 combos) → L3(2 combos)'


def add_args(parser):
    parser.add_argument("--layers-hold", type=int, default=50,
                        help="Ticks to hold each state (default: 50)")
    parser.add_argument("--layers-noise", type=float, default=0.1,
                        help="Per-neuron noise std (default: 0.1)")


def make_signal(w, h, args):
    n = w * h
    hold = getattr(args, 'layers_hold', 50)
    noise_std = getattr(args, 'layers_noise', 0.1)
    rng = np.random.RandomState(42)
    S = 8  # neurons per group

    # Build group indices
    idx = {}
    offset = 0
    # L1: 8 independent features
    for i in range(8):
        idx[f'L1_{i}'] = list(range(offset, offset + S))
        offset += S
    # L2: 4 composite features (XOR of L1 pairs)
    # L2_0 = L1_0 XOR L1_1, L2_1 = L1_2 XOR L1_3, etc.
    for i in range(4):
        idx[f'L2_{i}'] = list(range(offset, offset + S))
        offset += S
    # L3: 2 abstract features (XOR of L2 pairs)
    # L3_0 = L2_0 XOR L2_1, L3_1 = L2_2 XOR L2_3
    for i in range(2):
        idx[f'L3_{i}'] = list(range(offset, offset + S))
        offset += S

    state = {'L1': np.zeros(8, dtype=int), 'last_change': -hold}
    feature_log = []

    def tick_fn(t):
        if t - state['last_change'] >= hold:
            state['L1'] = rng.randint(0, 2, size=8)
            state['last_change'] = t

        L1 = state['L1']
        # L2: XOR of consecutive L1 pairs
        L2 = np.array([L1[i*2] ^ L1[i*2+1] for i in range(4)])
        # L3: XOR of consecutive L2 pairs
        L3 = np.array([L2[i*2] ^ L2[i*2+1] for i in range(2)])

        feature_log.append((t,) + tuple(L1) + tuple(L2) + tuple(L3))

        sig = np.zeros(n, dtype=np.float32)
        for i in range(8):
            for j in idx[f'L1_{i}']:
                sig[j] = float(L1[i])
        for i in range(4):
            for j in idx[f'L2_{i}']:
                sig[j] = float(L2[i])
        for i in range(2):
            for j in idx[f'L3_{i}']:
                sig[j] = float(L3[i])
        sig += rng.randn(n).astype(np.float32) * noise_std

        return sig

    metadata = {
        'feature_log': feature_log,
        'w': w, 'h': h, 'n': n,
        'hold': hold,
    }

    print(f"  signal buffer: LAYERS synthetic "
          f"(L1=8, L2=4, L3=2, hold={hold})")

    return tick_fn, metadata


def analyze(metadata, cluster_mgr, signals, tick_counter, T, output_dir):
    import torch

    if cluster_mgr is None or not cluster_mgr.initialized:
        return
    if cluster_mgr.column_mgr is None:
        return

    n = metadata['n']
    n_sensory = cluster_mgr.n_sensory

    # --- Layer classification ---
    most_recent = cluster_mgr.cluster_ids[
        np.arange(cluster_mgr.n), cluster_mgr.pointers]
    m = cluster_mgr.m
    n_outputs = cluster_mgr.column_mgr.n_outputs

    # Classify clusters
    v1_clusters = []  # contain sensory neurons
    v2_clusters = []  # contain feedback from V1 columns
    v3_clusters = []  # contain feedback from V2 columns
    fb_only = []

    s_counts = np.zeros(m, dtype=int)
    f_counts = np.zeros(m, dtype=int)
    for c in range(m):
        members = np.where(most_recent == c)[0]
        s_counts[c] = (members < n_sensory).sum()
        f_counts[c] = (members >= n_sensory).sum()

    # V1 = clusters with sensory neurons
    v1_set = set()
    for c in range(m):
        if s_counts[c] > 0:
            v1_clusters.append(c)
            v1_set.add(c)

    # V2 = feedback clusters whose source columns are in V1
    v2_set = set()
    for c in range(m):
        if s_counts[c] == 0 and f_counts[c] > 0:
            fb_members = np.where((most_recent == c) &
                                  (np.arange(cluster_mgr.n) >= n_sensory))[0]
            source_cols = (fb_members - n_sensory) // n_outputs
            if any(sc in v1_set for sc in source_cols):
                v2_clusters.append(c)
                v2_set.add(c)

    # V3 = feedback clusters whose source columns are in V2
    for c in range(m):
        if c not in v1_set and c not in v2_set and f_counts[c] > 0:
            fb_members = np.where((most_recent == c) &
                                  (np.arange(cluster_mgr.n) >= n_sensory))[0]
            source_cols = (fb_members - n_sensory) // n_outputs
            if any(sc in v2_set for sc in source_cols):
                v3_clusters.append(c)

    print(f"  LAYERS hierarchy:")
    print(f"    V1 (sensory): {len(v1_clusters)} clusters")
    print(f"    V2 (from V1): {len(v2_clusters)} clusters")
    print(f"    V3 (from V2): {len(v3_clusters)} clusters")

    # --- Feature correlation per layer ---
    print("  LAYERS analysis: sampling 500 ticks...")
    tick_fn = metadata['_tick_fn']
    col_history = []

    for t in range(500):
        sig_t = tick_fn(metadata['_total_ticks'] + T + t)
        col_t = tick_counter[0] % T
        signals[:n, col_t] = torch.from_numpy(sig_t).to(signals.device)
        tick_counter[0] += 1
        if cluster_mgr._signals is not None:
            cw = cluster_mgr.column_mgr.window
            indices = [(tick_counter[0] - 1 - i) % T for i in range(cw)]
            sw = cluster_mgr._signals[:, indices].cpu().numpy()
            cluster_mgr.column_mgr.tick(sw)
            col_history.append((t, cluster_mgr.column_mgr.get_outputs()))

    log = np.array(metadata['feature_log'])
    if len(log) == 0 or len(col_history) == 0:
        return

    all_outputs = np.array([o for _, o in col_history])
    n_ticks = min(len(log), all_outputs.shape[0])
    all_outputs = all_outputs[:n_ticks]
    features = log[-n_ticks:, 1:]  # skip tick column

    feature_names = ([f'L1_{i}' for i in range(8)] +
                     [f'L2_{i}' for i in range(4)] +
                     [f'L3_{i}' for i in range(2)])

    # Per-feature best correlation, noting which layer the best column is in
    results = {}
    for fi, fname in enumerate(feature_names):
        if fi >= features.shape[1]:
            continue
        feat = features[:, fi]
        if feat.std() < 1e-8:
            continue
        max_corr, best_c, best_o = 0, 0, 0
        for c in range(m):
            for o in range(n_outputs):
                out = all_outputs[:, c, o]
                if out.std() < 1e-8:
                    continue
                r = np.corrcoef(feat, out)[0, 1]
                if not np.isnan(r) and abs(r) > max_corr:
                    max_corr = abs(r)
                    best_c, best_o = c, o
        # Determine layer of best column
        if best_c in v1_set:
            layer = 'V1'
        elif best_c in v2_set:
            layer = 'V2'
        else:
            layer = 'V3+'
        results[fname] = {
            'max_abs_corr': round(max_corr, 4),
            'best_column': best_c,
            'best_output': best_o,
            'detected_by': layer,
        }

    print(f"  LAYERS feature correlations ({n_ticks} ticks):")
    for level in ['L1', 'L2', 'L3']:
        level_results = {k: v for k, v in results.items() if k.startswith(level)}
        if level_results:
            corrs = [v['max_abs_corr'] for v in level_results.values()]
            layers = [v['detected_by'] for v in level_results.values()]
            print(f"    {level}: mean|r|={np.mean(corrs):.3f}, "
                  f"detected by: {', '.join(layers)}")
            for fname, info in level_results.items():
                print(f"      {fname}: r={info['max_abs_corr']:.3f} "
                      f"(col {info['best_column']}/{info['detected_by']}, "
                      f"out {info['best_output']})")

    if output_dir:
        save_results = {
            'hierarchy': {
                'V1': len(v1_clusters),
                'V2': len(v2_clusters),
                'V3': len(v3_clusters),
            },
            'features': results,
        }
        path = os.path.join(output_dir, "layers_analysis.json")
        with open(path, 'w') as f:
            json.dump(save_results, f, indent=2)
        print(f"  LAYERS analysis saved: {path}")

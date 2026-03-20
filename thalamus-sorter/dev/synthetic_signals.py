"""Synthetic signal generators for benchmarking non-linear feature detection.

Usage:
    --signal-source xor      XOR benchmark (requires square grid, w=h)

Each generator returns a callable tick_fn(t) -> (n_sensory,) signal array,
plus metadata for post-hoc analysis.
"""

import numpy as np


def make_xor_signal(w, h, noise_std=0.1, hold_ticks=5, seed=42):
    """XOR benchmark: 4 quadrants with binary features A, B, XOR, AND.

    Grid layout (w×h must be square):
        ┌─────────┬─────────┐
        │ Region 1│ Region 2│
        │  sig=A  │  sig=B  │
        ├─────────┼─────────┤
        │ Region 3│ Region 4│
        │ sig=XOR │ sig=AND │
        └─────────┴─────────┘

    Each tick, binary features A,B ∈ {0,1} are drawn (held for hold_ticks).
    Region signals: base_value + per-neuron noise.

    Returns:
        tick_fn: callable(t) -> (n,) float32 signal
        metadata: dict with region masks, feature log access
    """
    n = w * h
    half_w = w // 2
    half_h = h // 2
    rng = np.random.RandomState(seed)

    # Build region masks (neuron indices for each quadrant)
    coords_x = np.arange(n) % w
    coords_y = np.arange(n) // w
    region1 = (coords_x < half_w) & (coords_y < half_h)   # top-left: A
    region2 = (coords_x >= half_w) & (coords_y < half_h)  # top-right: B
    region3 = (coords_x < half_w) & (coords_y >= half_h)  # bottom-left: XOR
    region4 = (coords_x >= half_w) & (coords_y >= half_h) # bottom-right: AND

    # State
    state = {'A': 0, 'B': 0, 'last_change': -hold_ticks}
    feature_log = []  # (tick, A, B, XOR, AND)

    def tick_fn(t):
        # Change features every hold_ticks
        if t - state['last_change'] >= hold_ticks:
            state['A'] = rng.randint(0, 2)
            state['B'] = rng.randint(0, 2)
            state['last_change'] = t

        A, B = state['A'], state['B']
        xor_val = A ^ B
        and_val = A & B
        feature_log.append((t, A, B, xor_val, and_val))

        # Generate signal
        sig = np.zeros(n, dtype=np.float32)
        sig[region1] = float(A)
        sig[region2] = float(B)
        sig[region3] = float(xor_val)
        sig[region4] = float(and_val)

        # Add per-neuron noise
        sig += rng.randn(n).astype(np.float32) * noise_std

        return sig

    metadata = {
        'regions': {
            'A': region1, 'B': region2,
            'XOR': region3, 'AND': region4,
        },
        'region_names': ['A', 'B', 'XOR', 'AND'],
        'feature_log': feature_log,
        'w': w, 'h': h, 'n': n,
        'noise_std': noise_std,
        'hold_ticks': hold_ticks,
    }

    return tick_fn, metadata


def analyze_xor_columns(metadata, column_outputs_history, tick_range=None):
    """Analyze whether column outputs correlate with XOR/AND features.

    Args:
        metadata: from make_xor_signal
        column_outputs_history: list of (tick, outputs_array) pairs
            where outputs_array is (m, n_outputs)
        tick_range: (start, end) to restrict analysis window

    Returns:
        dict with per-column correlations to each feature
    """
    log = np.array(metadata['feature_log'])  # (T, 5): tick, A, B, XOR, AND
    if tick_range:
        mask = (log[:, 0] >= tick_range[0]) & (log[:, 0] <= tick_range[1])
        log = log[mask]

    if len(log) == 0 or len(column_outputs_history) == 0:
        return {}

    # Align column outputs with feature log by tick
    log_ticks = set(log[:, 0].astype(int))
    aligned = []
    for tick, outputs in column_outputs_history:
        if tick in log_ticks:
            idx = np.searchsorted(log[:, 0], tick)
            if idx < len(log) and int(log[idx, 0]) == tick:
                aligned.append((log[idx, 1:], outputs))  # (4,), (m, n_out)

    if len(aligned) < 10:
        return {'error': f'only {len(aligned)} aligned ticks'}

    features = np.array([a[0] for a in aligned])    # (T, 4): A, B, XOR, AND
    all_outputs = np.array([a[1] for a in aligned])  # (T, m, n_out)

    T, m, n_out = all_outputs.shape
    feature_names = ['A', 'B', 'XOR', 'AND']

    # Per-column, per-output: correlation with each feature
    results = np.zeros((m, n_out, 4))  # (m, n_out, 4 features)
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

    # Summary: best correlation per feature across all (column, output) pairs
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

    return {
        'correlations': results,
        'best_per_feature': best,
        'n_ticks': T,
        'feature_names': feature_names,
    }

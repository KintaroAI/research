"""MIRROR benchmark: action-consequence awareness.

Column 0's outputs are fed back as sensory input on the next tick.
The system must learn that its own output BECOMES its input —
the simplest self-awareness test.

Grid layout (16×16 = 256 neurons):
    8× mirror neurons — receive column 0's 4 outputs (2 neurons each)
    8× stimulus neurons — random changing signal (context)
    240× zero (unused)

The mirror neurons show: what did column 0 output last tick?
If the system learns this, it knows its own actions have consequences.

Usage:
    python main.py word2vec --signal-source mirror --column-lateral ...
"""

import os
import json
import numpy as np

name = 'mirror'
description = 'Action-consequence: column 0 output → next tick sensory input'


def add_args(parser):
    parser.add_argument("--mirror-hold", type=int, default=1,
                        help="Unused (for CLI compat)")


def make_signal(w, h, args):
    n = w * h
    rng = np.random.RandomState(42)

    # Neuron groups
    idx_mirror = list(range(0, 8))   # column 0's outputs fed back
    idx_stim = list(range(8, 16))    # random stimulus

    _refs = {'column_mgr': None}
    prev_col0_out = np.zeros(4, dtype=np.float32)
    stim_state = {'value': 0.5, 'last_change': -20}
    feature_log = []

    def tick_fn(t):
        nonlocal prev_col0_out

        # Read column 0 outputs from previous tick
        col_mgr = _refs['column_mgr']
        if col_mgr is not None:
            curr_out = col_mgr.get_outputs()[0]  # (4,)
        else:
            curr_out = np.zeros(4, dtype=np.float32)

        # Stimulus changes every 20 ticks
        if t - stim_state['last_change'] >= 20:
            stim_state['value'] = float(rng.rand())
            stim_state['last_change'] = t

        feature_log.append((t,
                            prev_col0_out[0], prev_col0_out[1],
                            prev_col0_out[2], prev_col0_out[3],
                            curr_out[0], curr_out[1],
                            curr_out[2], curr_out[3],
                            stim_state['value']))

        sig = np.zeros(n, dtype=np.float32)
        # Mirror: prev tick's column 0 outputs (2 neurons per output)
        for i in range(4):
            sig[idx_mirror[i * 2]] = prev_col0_out[i]
            sig[idx_mirror[i * 2 + 1]] = prev_col0_out[i]
        # Stimulus
        for i in idx_stim:
            sig[i] = stim_state['value']
        sig[:16] += rng.randn(16).astype(np.float32) * 0.02

        # Store current outputs for next tick
        prev_col0_out = curr_out.copy()

        return sig

    metadata = {
        'feature_log': feature_log,
        'w': w, 'h': h, 'n': n,
        '_refs': _refs,
    }

    print(f"  signal buffer: MIRROR synthetic (column 0 → sensory feedback)")
    return tick_fn, metadata


def analyze(metadata, cluster_mgr, signals, tick_counter, T, output_dir):
    import torch

    if cluster_mgr is None or not cluster_mgr.initialized:
        return
    if cluster_mgr.column_mgr is None:
        return

    print("  MIRROR analysis: sampling 500 ticks...")
    tick_fn = metadata['_tick_fn']
    col_history = []
    n = metadata['n']

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

    # Correlate column outputs with mirror signals (prev_col0_out)
    # and current col0 outputs to see if any column predicts col0
    feature_names = ['prev_out0', 'prev_out1', 'prev_out2', 'prev_out3',
                     'curr_out0', 'curr_out1', 'curr_out2', 'curr_out3',
                     'stimulus']
    features = log[-n_ticks:, 1:]

    m, n_out = all_outputs.shape[1], all_outputs.shape[2]
    best = {}
    for fi, fname in enumerate(feature_names):
        if fi >= features.shape[1]:
            continue
        feat = features[:, fi]
        if feat.std() < 1e-8:
            continue
        max_corr, best_c, best_o = 0, 0, 0
        for c in range(m):
            for o in range(n_out):
                out = all_outputs[:, c, o]
                if out.std() < 1e-8:
                    continue
                r = np.corrcoef(feat, out)[0, 1]
                if not np.isnan(r) and abs(r) > max_corr:
                    max_corr = abs(r)
                    best_c, best_o = c, o
        best[fname] = {'max_abs_corr': max_corr, 'best_column': best_c,
                        'best_output': best_o}

    print(f"  MIRROR results ({n_ticks} ticks):")
    for fname, info in best.items():
        print(f"    {fname:10s}: max|r|={info['max_abs_corr']:.3f} "
              f"(col {info['best_column']}, out {info['best_output']})")

    if output_dir:
        path = os.path.join(output_dir, "mirror_analysis.json")
        with open(path, 'w') as f:
            json.dump(best, f, indent=2)

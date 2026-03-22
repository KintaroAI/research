"""ECHO benchmark: temporal prediction.

One neuron group ("voice") gets a random value. Another group ("echo")
receives the SAME value delayed by N ticks. Tests whether the system
learns that voice[t] predicts echo[t+N].

Grid layout (16×16 = 256 neurons):
    8× voice neurons — random value, changes every hold ticks
    8× echo neurons — same value, delayed by delay ticks
    8× noise neurons — random noise (control)
    232× zero (unused)

Usage:
    python main.py word2vec --signal-source echo --column-lateral ...
"""

import os
import json
import numpy as np

name = 'echo'
description = 'Temporal prediction: echo = voice delayed by N ticks'


def add_args(parser):
    parser.add_argument("--echo-delay", type=int, default=10,
                        help="Delay in ticks between voice and echo (default: 10)")
    parser.add_argument("--echo-hold", type=int, default=20,
                        help="Ticks to hold each voice value (default: 20)")


def make_signal(w, h, args):
    n = w * h
    delay = getattr(args, 'echo_delay', 10)
    hold = getattr(args, 'echo_hold', 20)
    rng = np.random.RandomState(42)

    # Neuron groups
    idx_voice = list(range(0, 8))
    idx_echo = list(range(8, 16))
    idx_noise = list(range(16, 24))

    # State
    voice_buffer = np.zeros(delay + 1, dtype=np.float32)
    state = {'value': 0.5, 'last_change': -hold}
    feature_log = []

    def tick_fn(t):
        # New voice value every hold ticks
        if t - state['last_change'] >= hold:
            state['value'] = float(rng.rand())
            state['last_change'] = t

        # Shift buffer, add current voice
        voice_buffer[1:] = voice_buffer[:-1]
        voice_buffer[0] = state['value']
        echo_value = voice_buffer[delay]  # delayed copy

        feature_log.append((t, state['value'], echo_value))

        sig = np.zeros(n, dtype=np.float32)
        for i in idx_voice:
            sig[i] = state['value']
        for i in idx_echo:
            sig[i] = echo_value
        for i in idx_noise:
            sig[i] = float(rng.rand())
        # Small noise on voice/echo
        sig[:24] += rng.randn(24).astype(np.float32) * 0.02

        return sig

    metadata = {
        'feature_log': feature_log,
        'w': w, 'h': h, 'n': n,
        'delay': delay, 'hold': hold,
    }

    print(f"  signal buffer: ECHO synthetic (delay={delay}, hold={hold})")
    return tick_fn, metadata


def analyze(metadata, cluster_mgr, signals, tick_counter, T, output_dir):
    import torch

    if cluster_mgr is None or not cluster_mgr.initialized:
        return
    if cluster_mgr.column_mgr is None:
        return

    print("  ECHO analysis: sampling 500 ticks...")
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
    n_ticks, m, n_out = all_outputs.shape

    n_ticks = min(len(log), all_outputs.shape[0])
    all_outputs = all_outputs[:n_ticks]
    features = log[-n_ticks:, 1:]

    best = {}
    for fi, fname in enumerate(['voice', 'echo']):
        feat = features[:, fi] if fi < features.shape[1] else np.zeros(n_ticks)
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

    print(f"  ECHO results ({n_ticks} ticks):")
    for fname, info in best.items():
        print(f"    {fname:5s}: max|r|={info['max_abs_corr']:.3f} "
              f"(col {info['best_column']}, out {info['best_output']})")

    if output_dir:
        path = os.path.join(output_dir, "echo_analysis.json")
        with open(path, 'w') as f:
            json.dump(best, f, indent=2)

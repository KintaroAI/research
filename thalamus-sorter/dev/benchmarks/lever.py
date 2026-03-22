"""LEVER benchmark: conditional action-reward (operant conditioning).

A "light" turns on randomly. If light=1 AND column 0 output 0 > 0.5 →
reward (hunger resets, pleasure spikes). The system must learn:
"when light is on, press the lever."

Grid layout (16×16 = 256 neurons):
    8× light neurons — 0 or 1, changes every hold ticks
    8× lever neurons — column 0 output 0 fed back (did I press?)
    8× reward neurons — spikes on successful press, decays over 20 ticks
    8× hunger neurons — ramps, resets on reward (pulsated)
    224× zero (unused)

Usage:
    python main.py word2vec --signal-source lever --column-lateral ...
"""

import os
import json
import numpy as np

name = 'lever'
description = 'Operant conditioning: press lever when light is on → reward'


def add_args(parser):
    parser.add_argument("--lever-hold", type=int, default=50,
                        help="Ticks to hold each light state (default: 50)")
    parser.add_argument("--lever-press-threshold", type=float, default=0.5,
                        help="Column 0 output 0 threshold for 'press' (default: 0.5)")


def make_signal(w, h, args):
    n = w * h
    hold = getattr(args, 'lever_hold', 50)
    press_threshold = getattr(args, 'lever_press_threshold', 0.5)
    rng = np.random.RandomState(42)

    idx_light = list(range(0, 8))
    idx_lever = list(range(8, 16))
    idx_reward = list(range(16, 24))
    idx_hunger = list(range(24, 32))

    _refs = {'column_mgr': None, 'dsolver': None}
    state = {'light': 0, 'last_change': -hold, 'reward_decay': 0.0,
             'hunger': 0.0, 'score': 0, 'presses': 0,
             'correct_presses': 0, 'wrong_presses': 0}
    base_lr = [None, None]
    feature_log = []

    def tick_fn(t):
        # Light changes
        if t - state['last_change'] >= hold:
            state['light'] = rng.randint(0, 2)
            state['last_change'] = t

        # Read column 0 output 0 (the "lever")
        col_mgr = _refs['column_mgr']
        lever_value = 0.0
        if col_mgr is not None:
            lever_value = float(col_mgr.get_outputs()[0, 0])

        pressed = lever_value > press_threshold

        # Check reward condition
        if pressed:
            state['presses'] += 1
            if state['light'] == 1:
                state['correct_presses'] += 1
                state['score'] += 1
                state['reward_decay'] = 1.0
                state['hunger'] = 0.0
            else:
                state['wrong_presses'] += 1

        # Decay reward signal
        state['reward_decay'] = max(0.0, state['reward_decay'] * 0.9)

        # Hunger ramps
        state['hunger'] = min(1.0, state['hunger'] + 0.005)

        # Hunger modulates lr
        lr_scale = max(0.01, 1.0 - state['hunger'] * 0.99)
        if col_mgr is not None:
            if base_lr[0] is None:
                base_lr[0] = col_mgr.lr
            col_mgr.lr = base_lr[0] * lr_scale
        dsolver = _refs.get('dsolver')
        if dsolver is not None:
            if base_lr[1] is None:
                base_lr[1] = dsolver.lr
            dsolver.lr = base_lr[1] * lr_scale

        feature_log.append((t, float(state['light']), lever_value,
                            float(pressed), state['reward_decay'],
                            state['hunger']))

        sig = np.zeros(n, dtype=np.float32)
        for i in idx_light:
            sig[i] = float(state['light'])
        for i in idx_lever:
            sig[i] = lever_value
        for i in idx_reward:
            sig[i] = state['reward_decay']
        from benchmarks.forage import pulsate
        for i in idx_hunger:
            sig[i] = pulsate(state['hunger'], t, 17)
        sig[:32] += rng.randn(32).astype(np.float32) * 0.02

        return sig

    metadata = {
        'feature_log': feature_log,
        'w': w, 'h': h, 'n': n,
        '_refs': _refs,
        'state': state,
    }

    print(f"  signal buffer: LEVER synthetic (hold={hold}, threshold={press_threshold})")
    return tick_fn, metadata


def analyze(metadata, cluster_mgr, signals, tick_counter, T, output_dir):
    import torch

    state = metadata['state']
    print(f"  LEVER results:")
    print(f"    Score: {state['score']} correct presses")
    print(f"    Total presses: {state['presses']}")
    print(f"    Correct: {state['correct_presses']}, Wrong: {state['wrong_presses']}")
    if state['presses'] > 0:
        accuracy = state['correct_presses'] / state['presses']
        print(f"    Accuracy: {accuracy:.3f} (0.5 = random)")

    if cluster_mgr is None or not cluster_mgr.initialized:
        return
    if cluster_mgr.column_mgr is None:
        return

    print("  LEVER analysis: sampling 500 ticks...")
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
    features = log[-n_ticks:, 1:]

    feature_names = ['light', 'lever', 'pressed', 'reward', 'hunger']
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

    print(f"  LEVER correlations ({n_ticks} ticks):")
    for fname, info in best.items():
        print(f"    {fname:7s}: max|r|={info['max_abs_corr']:.3f} "
              f"(col {info['best_column']}, out {info['best_output']})")

    if output_dir:
        results = {
            'score': state['score'],
            'presses': state['presses'],
            'correct': state['correct_presses'],
            'wrong': state['wrong_presses'],
            'correlations': best,
        }
        path = os.path.join(output_dir, "lever_analysis.json")
        with open(path, 'w') as f:
            json.dump(results, f, indent=2)

"""SEEK benchmark: directional action-reward.

A target appears at one of 4 positions. The motor (column 0) must push
in the matching direction. Reward when motor output matches target.
Simpler than foraging — no field, just "which direction should I push?"

Grid layout (16×16 = 256 neurons):
    8× target_up, 8× target_down, 8× target_left, 8× target_right
       (one group active = 1.0, others = 0.0)
    8× reward neurons — spikes on correct match, decays
    8× hunger neurons — ramps, resets on reward
    192× zero (unused)

Motor: column 0 outputs [dx+, dx-, dy+, dy-].
Match = argmax(motor) == target_direction → reward.

Usage:
    python main.py word2vec --signal-source seek --column-lateral ...
"""

import os
import json
import numpy as np

name = 'seek'
description = 'Directional: push toward target direction → reward'


def add_args(parser):
    parser.add_argument("--seek-hold", type=int, default=50,
                        help="Ticks to hold each target direction (default: 50)")


def make_signal(w, h, args):
    n = w * h
    hold = getattr(args, 'seek_hold', 50)
    rng = np.random.RandomState(42)

    S = 8  # neurons per signal
    idx = {}
    offset = 0
    for name_s in ['target_right', 'target_left', 'target_down', 'target_up',
                    'reward', 'hunger']:
        idx[name_s] = list(range(offset, offset + S))
        offset += S

    _refs = {'column_mgr': None, 'dsolver': None}
    dir_names = ['right', 'left', 'down', 'up']  # maps to out[0-3]
    state = {'target_dir': 0, 'last_change': -hold,
             'reward_decay': 0.0, 'hunger': 0.0,
             'score': 0, 'attempts': 0, 'correct': 0}
    base_lr = [None, None]
    feature_log = []

    def tick_fn(t):
        # Change target direction
        if t - state['last_change'] >= hold:
            state['target_dir'] = rng.randint(0, 4)
            state['last_change'] = t

        # Read motor output
        col_mgr = _refs['column_mgr']
        motor_out = np.zeros(4, dtype=np.float32)
        if col_mgr is not None:
            motor_out = col_mgr.get_outputs()[0]  # (4,)

        motor_dir = int(motor_out.argmax())
        motor_confidence = float(motor_out.max())

        # Check match — only count if motor is confident enough
        if motor_confidence > 0.35:
            state['attempts'] += 1
            if motor_dir == state['target_dir']:
                state['correct'] += 1
                state['score'] += 1
                state['reward_decay'] = 1.0
                state['hunger'] = 0.0

        # Decay reward
        state['reward_decay'] = max(0.0, state['reward_decay'] * 0.9)

        # Hunger ramps
        state['hunger'] = min(1.0, state['hunger'] + 0.005)

        # Lr modulation
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

        feature_log.append((t, state['target_dir'], motor_dir,
                            motor_confidence, state['reward_decay'],
                            state['hunger']))

        sig = np.zeros(n, dtype=np.float32)
        # Target direction: one-hot across 4 groups
        target_names = ['target_right', 'target_left', 'target_down', 'target_up']
        for d in range(4):
            val = 1.0 if d == state['target_dir'] else 0.0
            for i in idx[target_names[d]]:
                sig[i] = val
        # Reward
        for i in idx['reward']:
            sig[i] = state['reward_decay']
        # Hunger (pulsated)
        from benchmarks.forage import pulsate
        for i in idx['hunger']:
            sig[i] = pulsate(state['hunger'], t, 17)

        sig[:offset] += rng.randn(offset).astype(np.float32) * 0.02

        return sig

    metadata = {
        'feature_log': feature_log,
        'w': w, 'h': h, 'n': n,
        '_refs': _refs,
        'state': state,
        'dir_names': dir_names,
    }

    print(f"  signal buffer: SEEK synthetic (hold={hold})")
    return tick_fn, metadata


def analyze(metadata, cluster_mgr, signals, tick_counter, T, output_dir):
    import torch

    state = metadata['state']
    print(f"  SEEK results:")
    print(f"    Score: {state['score']}")
    print(f"    Attempts: {state['attempts']}")
    if state['attempts'] > 0:
        accuracy = state['correct'] / state['attempts']
        print(f"    Accuracy: {accuracy:.3f} (0.25 = random)")

    if cluster_mgr is None or not cluster_mgr.initialized:
        return
    if cluster_mgr.column_mgr is None:
        return

    print("  SEEK analysis: sampling 500 ticks...")
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

    feature_names = ['target_dir', 'motor_dir', 'confidence', 'reward', 'hunger']
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

    print(f"  SEEK correlations ({n_ticks} ticks):")
    for fname, info in best.items():
        print(f"    {fname:10s}: max|r|={info['max_abs_corr']:.3f} "
              f"(col {info['best_column']}, out {info['best_output']})")

    if output_dir:
        results = {
            'score': state['score'],
            'attempts': state['attempts'],
            'accuracy': state['correct'] / max(1, state['attempts']),
            'correlations': best,
        }
        path = os.path.join(output_dir, "seek_analysis.json")
        with open(path, 'w') as f:
            json.dump(results, f, indent=2)

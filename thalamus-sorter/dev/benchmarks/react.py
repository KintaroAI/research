"""REACT benchmark: stimulus-response learning with reward.

Tests whether the system can learn arbitrary stimulus→action mappings
through teacher signal and reward-based eligibility traces.

4 stimuli × 4 correct responses. Teacher phase shows the answer,
test phase removes it — must respond from learned associations.

Grid layout:
    32 stimulus neurons (4 groups × 8)
    32 answer neurons (4 groups × 8, teacher phase only)
    32 motor feedback neurons (4 groups × 8, echo of column outputs)
    = 96 total (12×8 grid)

Phases:
    1. Teacher (0 to teacher_ticks): stimulus + answer shown, reward on correct
    2. Test-frozen (teacher_ticks to test_end): no answer, lr=0, reward on correct
    3. Test-live (test_end to end): no answer, lr restored, reward on correct

Usage:
    python main.py word2vec --signal-source react -W 12 -H 8 ...
"""

import os
import json
import numpy as np

name = 'react'
description = 'Stimulus-response: learn 4 mappings via teacher + reward'

N_STIMULI = 4
NEURONS_PER = 8


def add_args(parser):
    parser.add_argument("--react-hold", type=int, default=50,
                        help="Ticks to hold each stimulus (default: 50)")
    parser.add_argument("--react-teacher-ticks", type=int, default=5000,
                        help="Teacher phase duration (default: 5000)")
    parser.add_argument("--react-test-ticks", type=int, default=2000,
                        help="Each test phase duration (default: 2000)")
    parser.add_argument("--react-motor-columns", type=str, default="0,1,2,3",
                        help="Columns whose outputs are motor (default: 0,1,2,3)")


def make_signal(w, h, args):
    n = w * h
    hold = getattr(args, 'react_hold', 50)
    teacher_ticks = getattr(args, 'react_teacher_ticks', 5000)
    test_ticks = getattr(args, 'react_test_ticks', 2000)
    motor_cols_str = getattr(args, 'react_motor_columns', '0,1,2,3')
    motor_columns = [int(x) for x in motor_cols_str.split(',')]
    rng = np.random.RandomState(42)

    # Neuron indices
    S = NEURONS_PER
    idx = {}
    offset = 0
    for i in range(N_STIMULI):
        idx[f'stim_{i}'] = list(range(offset, offset + S))
        offset += S
    for i in range(N_STIMULI):
        idx[f'answer_{i}'] = list(range(offset, offset + S))
        offset += S
    for i in range(N_STIMULI):
        idx[f'motor_{i}'] = list(range(offset, offset + S))
        offset += S

    state = {
        'current_stim': 0,
        'last_change': -hold,
    }
    feature_log = []
    score = {'teacher_correct': 0, 'teacher_total': 0,
             'frozen_correct': 0, 'frozen_total': 0,
             'live_correct': 0, 'live_total': 0}
    _refs = {'column_mgr': None, 'dsolver': None}
    base_column_lr = [None]
    base_embed_lr = [None]

    # Phase boundaries
    test_frozen_start = teacher_ticks
    test_live_start = teacher_ticks + test_ticks
    total_test_end = teacher_ticks + 2 * test_ticks

    def tick_fn(t):
        # Switch stimulus every hold ticks
        if t - state['last_change'] >= hold:
            state['current_stim'] = rng.randint(0, N_STIMULI)
            state['last_change'] = t

        stim = state['current_stim']

        # Determine phase
        is_teacher = t < test_frozen_start
        is_frozen = test_frozen_start <= t < test_live_start
        is_live = t >= test_live_start

        # LR control
        col_mgr = _refs.get('column_mgr')
        dsolver = _refs.get('dsolver')
        if col_mgr is not None:
            if base_column_lr[0] is None:
                base_column_lr[0] = col_mgr.lr
            if is_frozen:
                col_mgr.lr = 0.0
            else:
                col_mgr.lr = base_column_lr[0]
        if dsolver is not None:
            if base_embed_lr[0] is None:
                base_embed_lr[0] = dsolver.lr
            if is_frozen:
                dsolver.lr = 0.0
            else:
                dsolver.lr = base_embed_lr[0]

        # Build signal
        sig = np.zeros(n, dtype=np.float32)

        # Stimulus: active group = 1.0, others = 0.0
        for i in range(N_STIMULI):
            val = 1.0 if i == stim else 0.0
            for j in idx[f'stim_{i}']:
                sig[j] = val + rng.randn() * 0.05

        # Answer (teacher phase only)
        if is_teacher:
            for i in range(N_STIMULI):
                val = 1.0 if i == stim else 0.0
                for j in idx[f'answer_{i}']:
                    sig[j] = val + rng.randn() * 0.05
        else:
            # No answer — just noise
            for i in range(N_STIMULI):
                for j in idx[f'answer_{i}']:
                    sig[j] = rng.randn() * 0.05

        # Motor feedback: echo column outputs from motor columns
        if col_mgr is not None and len(motor_columns) > 0:
            all_out = col_mgr.get_outputs()
            m_cols = all_out.shape[0]
            for i, mc in enumerate(motor_columns):
                if i < N_STIMULI and mc < m_cols:
                    # Map column's winner to motor group activation
                    winner = int(all_out[mc].argmax())
                    for j in range(S):
                        sig[idx[f'motor_{i}'][j]] = 1.0 if winner == i else 0.0

        # Check correctness and reward
        if col_mgr is not None and len(motor_columns) > 0:
            # Determine which motor output is strongest
            all_out = col_mgr.get_outputs()
            # Check if correct motor column's winner matches stimulus
            # Simple: does any motor column have winner == stim?
            motor_response = -1
            best_prob = 0.0
            for i, mc in enumerate(motor_columns):
                if i < N_STIMULI and mc < all_out.shape[0]:
                    prob = all_out[mc, stim]  # prob of output matching stim
                    if prob > best_prob:
                        best_prob = prob
                        motor_response = i

            correct = (motor_response == stim) and (best_prob > 0.3)

            # Score tracking
            if is_teacher:
                score['teacher_total'] += 1
                if correct:
                    score['teacher_correct'] += 1
            elif is_frozen:
                score['frozen_total'] += 1
                if correct:
                    score['frozen_correct'] += 1
            elif is_live:
                score['live_total'] += 1
                if correct:
                    score['live_correct'] += 1

            # Reward
            if correct:
                col_mgr.set_reward(1.0)

        feature_log.append((t, stim, 1 if is_teacher else 0))

        return sig

    metadata = {
        'feature_log': feature_log,
        'w': w, 'h': h, 'n': n,
        'idx': idx,
        'score': score,
        'teacher_ticks': teacher_ticks,
        'test_ticks': test_ticks,
        '_refs': _refs,
    }

    print(f"  signal buffer: REACT ({N_STIMULI} stimuli, "
          f"hold={hold}, teacher={teacher_ticks}, test={test_ticks})")

    return tick_fn, metadata


def analyze(metadata, cluster_mgr, signals, tick_counter, T, output_dir):
    if cluster_mgr is None or not cluster_mgr.initialized:
        return
    if cluster_mgr.column_mgr is None:
        return

    score = metadata['score']
    teacher_ticks = metadata['teacher_ticks']
    test_ticks = metadata['test_ticks']

    def _pct(correct, total):
        return f"{100*correct/max(total,1):.1f}%" if total > 0 else "N/A"

    print(f"  REACT results:")
    print(f"    Teacher phase (0-{teacher_ticks}): "
          f"{score['teacher_correct']}/{score['teacher_total']} "
          f"({_pct(score['teacher_correct'], score['teacher_total'])})")
    print(f"    Test frozen (lr=0): "
          f"{score['frozen_correct']}/{score['frozen_total']} "
          f"({_pct(score['frozen_correct'], score['frozen_total'])})")
    print(f"    Test live (lr restored): "
          f"{score['live_correct']}/{score['live_total']} "
          f"({_pct(score['live_correct'], score['live_total'])})")

    if output_dir:
        path = os.path.join(output_dir, "react_analysis.json")
        with open(path, 'w') as f:
            json.dump(score, f, indent=2)
        print(f"  REACT analysis saved: {path}")

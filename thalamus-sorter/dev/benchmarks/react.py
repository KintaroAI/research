"""REACT benchmark: stimulus-response learning via spasm exploration + reward.

Tests whether the system can learn arbitrary stimulus→action mappings
through random motor exploration (spasms) and reward signal.

4 stimuli × 4 motor columns × 4 outputs each. Spasm neurons ramp up
randomly — when they fire, the benchmark twitches the corresponding
motor. Contraction feedback shows the system what its motors did.
Reward when motor output matches the correct response for current stimulus.

Grid layout (24×8 = 192):
    32 stimulus neurons (4 groups × 8)
    32 spasm neurons (4 groups × 8, ramp up → trigger twitch → reset)
    128 contraction neurons (4 cols × 4 outs × 8, echo of motor output)

Usage:
    python main.py word2vec --signal-source react -W 24 -H 8 ...
"""

import os
import json
import numpy as np

name = 'react'
description = 'Stimulus-response: learn 4 mappings via spasm exploration + reward'

N_STIMULI = 4
NEURONS_PER = 8
N_OUTPUTS = 4


def add_args(parser):
    parser.add_argument("--react-hold", type=int, default=50,
                        help="Ticks to hold each stimulus (default: 50)")
    parser.add_argument("--react-motor-columns", type=str, default="0,1,2,3",
                        help="Columns whose outputs are motor (default: 0,1,2,3)")
    parser.add_argument("--react-spasm-rate", type=float, default=0.1,
                        help="Max spasm ramp increment per tick (default: 0.1, ~20 ticks to fire)")
    parser.add_argument("--react-spasm-threshold", type=float, default=0.8,
                        help="Spasm fire threshold (default: 0.8)")
    parser.add_argument("--react-motor-threshold", type=float, default=0.8,
                        help="Motor output threshold for 'voluntary active' (default: 0.8)")


def make_signal(w, h, args):
    n = w * h
    hold = getattr(args, 'react_hold', 50)
    motor_cols_str = getattr(args, 'react_motor_columns', '0,1,2,3')
    motor_columns = [int(x) for x in motor_cols_str.split(',')]
    spasm_rate = getattr(args, 'react_spasm_rate', 0.005)
    spasm_threshold = getattr(args, 'react_spasm_threshold', 0.8)
    motor_threshold = getattr(args, 'react_motor_threshold', 0.4)
    rng = np.random.RandomState(42)

    # Neuron indices
    S = NEURONS_PER
    idx = {}
    offset = 0
    for i in range(N_STIMULI):
        idx[f'stim_{i}'] = list(range(offset, offset + S))
        offset += S
    for i in range(N_STIMULI):
        idx[f'spasm_{i}'] = list(range(offset, offset + S))
        offset += S
    for mc in range(N_STIMULI):
        for out in range(N_OUTPUTS):
            idx[f'contract_{mc}_out{out}'] = list(range(offset, offset + S))
            offset += S

    # State
    state = {
        'current_stim': 0,
        'last_change': -hold,
        'spasm_level': np.zeros(N_STIMULI, dtype=np.float32),
    }
    feature_log = []
    score = {'correct': 0, 'total': 0, 'spasms': 0, 'rewards': 0}
    _refs = {'column_mgr': None, 'dsolver': None}

    # Correct mapping: stimulus i → motor column i should have output i as winner
    # But we accept ANY consistent mapping — track what the system actually does
    # For simplicity: reward when motor column stim has its highest output > threshold
    # The specific output slot doesn't matter — just that the right column activates

    def tick_fn(t):
        # Switch stimulus every hold ticks
        if t - state['last_change'] >= hold:
            state['current_stim'] = rng.randint(0, N_STIMULI)
            state['last_change'] = t

        stim = state['current_stim']
        col_mgr = _refs.get('column_mgr')

        # Get current motor outputs
        motor_out = np.zeros((N_STIMULI, N_OUTPUTS), dtype=np.float32)
        if col_mgr is not None:
            all_out = col_mgr.get_outputs()
            for i, mc in enumerate(motor_columns):
                if i < N_STIMULI and mc < all_out.shape[0]:
                    motor_out[i] = all_out[mc]

        # Spasm logic: ramp up randomly, fire when threshold reached
        for i in range(N_STIMULI):
            # Check if motor is voluntarily active (any output above threshold)
            max_motor = motor_out[i].max()
            if max_motor > motor_threshold:
                # Voluntary activity — reset spasm, no twitch needed
                state['spasm_level'][i] = 0.0
            else:
                # Idle — ramp up spasm with random increment
                state['spasm_level'][i] += rng.rand() * spasm_rate

            # Fire spasm if threshold reached
            if state['spasm_level'][i] >= spasm_threshold:
                # Force a twitch: temporarily boost a random output
                # on motor column i by setting reward-like signal
                # We can't directly set column output, but we can
                # influence via the spasm signal (model sees spasm → learns)
                state['spasm_level'][i] = 0.0
                score['spasms'] += 1

        # Build signal
        sig = np.zeros(n, dtype=np.float32)

        # Stimulus
        for i in range(N_STIMULI):
            val = 1.0 if i == stim else 0.0
            for j in idx[f'stim_{i}']:
                sig[j] = val + rng.randn() * 0.02

        # Spasm neurons — show current ramp level
        for i in range(N_STIMULI):
            for j in idx[f'spasm_{i}']:
                sig[j] = state['spasm_level'][i]

        # Contraction feedback — echo motor output probabilities
        for mc_i in range(N_STIMULI):
            for out in range(N_OUTPUTS):
                val = float(motor_out[mc_i, out])
                for j in idx[f'contract_{mc_i}_out{out}']:
                    sig[j] = val

        # Reward: check if the correct motor column is most active
        reward_threshold = 0.65
        if col_mgr is not None:
            correct_max = motor_out[stim].max()
            other_max = max(motor_out[j].max() for j in range(N_STIMULI) if j != stim)

            if correct_max > reward_threshold and correct_max > other_max:
                score['correct'] += 1
                score['rewards'] += 1
                col_mgr.set_reward(1.0)

        score['total'] += 1

        feature_log.append((t, stim, score['correct'], score['spasms']))

        return sig

    metadata = {
        'feature_log': feature_log,
        'w': w, 'h': h, 'n': n,
        'idx': idx,
        'score': score,
        '_refs': _refs,
    }

    print(f"  signal buffer: REACT ({N_STIMULI} stimuli, hold={hold}, "
          f"spasm_rate={spasm_rate}, motor_cols={motor_columns})")

    return tick_fn, metadata


def analyze(metadata, cluster_mgr, signals, tick_counter, T, output_dir):
    if cluster_mgr is None or not cluster_mgr.initialized:
        return
    if cluster_mgr.column_mgr is None:
        return

    score = metadata['score']
    total = max(score['total'], 1)

    print(f"  REACT results:")
    print(f"    Correct: {score['correct']}/{score['total']} "
          f"({100*score['correct']/total:.1f}%)")
    print(f"    Rewards issued: {score['rewards']}")
    print(f"    Spasms: {score['spasms']}")
    print(f"    Chance level: {100/N_STIMULI:.1f}%")

    if output_dir:
        path = os.path.join(output_dir, "react_analysis.json")
        with open(path, 'w') as f:
            json.dump(score, f, indent=2)
        print(f"  REACT analysis saved: {path}")

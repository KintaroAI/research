"""Column pattern classifier: can a single column learn to categorize spatial patterns?

Feeds 4 distinct 3x3 binary patterns (vertical, horizontal, diagonal, cross)
with noise, cycling randomly. Tests whether the column assigns each pattern
type to a consistent output.

Usage:
    cd thalamus-sorter/dev
    python benchmarks/column_pattern.py
    python benchmarks/column_pattern.py --column-type default
"""

import argparse
import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from column_manager import ColumnManager, ConscienceColumn

# 4 canonical 3x3 patterns
PATTERNS = {
    'vertical': np.array([
        [0, 1, 0],
        [0, 1, 0],
        [0, 1, 0],
    ], dtype=np.float32),
    'horizontal': np.array([
        [0, 0, 0],
        [1, 1, 1],
        [0, 0, 0],
    ], dtype=np.float32),
    'diagonal': np.array([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
    ], dtype=np.float32),
    'cross': np.array([
        [0, 1, 0],
        [1, 1, 1],
        [0, 1, 0],
    ], dtype=np.float32),
}
PATTERN_NAMES = list(PATTERNS.keys())
PATTERN_ARRAYS = [PATTERNS[n].ravel() for n in PATTERN_NAMES]


def run_test(column_type, n_train, n_eval, noise, window, lr, temperature,
             alpha, hold):
    n_inputs = 9  # 3x3
    n_outputs = 4
    n_patterns = 4

    if column_type == 'conscience':
        cm = ConscienceColumn(
            m=1, n_outputs=n_outputs, max_inputs=n_inputs,
            window=window, temperature=temperature, lr=lr, alpha=alpha)
    else:
        cm = ColumnManager(
            m=1, n_outputs=n_outputs, max_inputs=n_inputs,
            window=window, temperature=temperature, lr=lr,
            mode='kmeans', entropy_scaled_lr=True)

    # Wire: single column, 9 inputs
    for i in range(n_inputs):
        cm.slot_map[0, i] = i

    ring = np.zeros((n_inputs, window), dtype=np.float32)
    rng = np.random.RandomState(42)

    # Training
    current_pattern = 0
    hold_counter = 0
    for t in range(n_train):
        if hold_counter <= 0:
            current_pattern = rng.randint(0, n_patterns)
            hold_counter = hold
        hold_counter -= 1

        signal = PATTERN_ARRAYS[current_pattern] + rng.randn(n_inputs).astype(np.float32) * noise
        ring[:, :-1] = ring[:, 1:]
        ring[:, -1] = signal
        cm.tick(ring)

    # Evaluation: for each pattern, run n_eval ticks and record winners
    results = {}
    for pi, pname in enumerate(PATTERN_NAMES):
        # Reset ring buffer for clean eval
        ring[:] = 0
        winners = np.zeros(n_outputs, dtype=int)

        for t in range(n_eval):
            signal = PATTERN_ARRAYS[pi] + rng.randn(n_inputs).astype(np.float32) * noise
            ring[:, :-1] = ring[:, 1:]
            ring[:, -1] = signal
            cm.tick(ring)
            w = cm.get_outputs()[0].argmax()
            winners[w] += 1

        dominant = winners.argmax()
        dominant_frac = winners[dominant] / n_eval
        results[pname] = {
            'dominant_output': int(dominant),
            'dominant_frac': float(dominant_frac),
            'distribution': winners.tolist(),
        }

    # Check: do all 4 patterns map to different outputs?
    dominant_outputs = [results[n]['dominant_output'] for n in PATTERN_NAMES]
    all_different = len(set(dominant_outputs)) == n_patterns
    n_unique = len(set(dominant_outputs))

    return {
        'column_type': column_type,
        'patterns': results,
        'dominant_outputs': dominant_outputs,
        'all_different': all_different,
        'n_unique': n_unique,
    }


def main():
    parser = argparse.ArgumentParser(description='Column pattern classifier test')
    parser.add_argument('--column-type', type=str, default='conscience')
    parser.add_argument('--train', type=int, default=5000)
    parser.add_argument('--eval', type=int, default=500)
    parser.add_argument('--noise', type=float, default=0.1)
    parser.add_argument('--window', type=int, default=10)
    parser.add_argument('--lr', type=float, default=0.05)
    parser.add_argument('--temperature', type=float, default=0.2)
    parser.add_argument('--alpha', type=float, default=0.01)
    parser.add_argument('--hold', type=int, default=20,
                        help='Ticks to hold each pattern before switching (default: 20)')
    args = parser.parse_args()

    print(f"Pattern classifier: type={args.column_type}, train={args.train}, "
          f"eval={args.eval}, noise={args.noise}, hold={args.hold}")

    r = run_test(args.column_type, args.train, args.eval, args.noise,
                 args.window, args.lr, args.temperature, args.alpha, args.hold)

    print()
    for pname in PATTERN_NAMES:
        p = r['patterns'][pname]
        dist = '/'.join(f'{d}' for d in p['distribution'])
        print(f"  {pname:12s} → output {p['dominant_output']} "
              f"({p['dominant_frac']:.0%})  [{dist}]")

    print()
    outputs = r['dominant_outputs']
    print(f"  Mapping: {' / '.join(f'{n}→{o}' for n, o in zip(PATTERN_NAMES, outputs))}")
    if r['all_different']:
        print(f"  PASS: all 4 patterns map to different outputs")
    else:
        print(f"  PARTIAL: {r['n_unique']}/4 unique outputs")


if __name__ == '__main__':
    main()

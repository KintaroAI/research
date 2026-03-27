"""Column XOR: can a chain of two columns compute XOR?

XOR is not linearly separable — a single column (k-means) can't solve it.
But two columns chained together can:

  Column 1: sees 2 binary inputs (A, B), has 4 outputs
    → learns to categorize (0,0), (0,1), (1,0), (1,1) into 4 buckets

  Column 2: sees Column 1's 4 outputs as input, has 2 outputs
    → learns to group {(0,0),(1,1)} vs {(0,1),(1,0)}

If this works, it shows columns can compose non-linear functions through
hierarchy — the first column creates a representation where XOR becomes
linearly separable.

Also tests single-column baseline (should fail at XOR, succeed at AND/OR).

Usage:
    cd thalamus-sorter/dev
    python benchmarks/column_xor.py
    python benchmarks/column_xor.py --column-type default
"""

import argparse
import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from column_manager import ColumnManager, ConscienceColumn


def make_column(n_inputs, n_outputs, column_type, window, lr, temperature, alpha):
    if column_type == 'conscience':
        cm = ConscienceColumn(
            m=1, n_outputs=n_outputs, max_inputs=n_inputs,
            window=window, temperature=temperature, lr=lr, alpha=alpha)
    else:
        cm = ColumnManager(
            m=1, n_outputs=n_outputs, max_inputs=n_inputs,
            window=window, temperature=temperature, lr=lr,
            mode='kmeans', entropy_scaled_lr=True)
    for i in range(n_inputs):
        cm.slot_map[0, i] = i
    return cm


def eval_function(cm1, cm2, ring1, ring2, target_fn, n_eval, noise, rng, window):
    """Evaluate a column chain on a boolean function.

    Returns per-input-combo accuracy and overall accuracy.
    """
    combos = [(0, 0), (0, 1), (1, 0), (1, 1)]
    correct = 0
    total = 0
    per_combo = {}

    # First, determine which output maps to which target value
    # by majority vote over eval ticks
    combo_outputs = {c: [] for c in combos}

    for _ in range(n_eval):
        for a, b in combos:
            # Feed input through chain
            sig1 = np.array([float(a), float(b)], dtype=np.float32)
            sig1 += rng.randn(2).astype(np.float32) * noise
            ring1[:, :-1] = ring1[:, 1:]
            ring1[:, -1] = sig1
            cm1.tick(ring1)
            out1 = cm1.get_outputs()[0]  # (4,)

            if cm2 is not None:
                ring2[:, :-1] = ring2[:, 1:]
                ring2[:, -1] = out1
                cm2.tick(ring2)
                final_out = cm2.get_outputs()[0]  # (2,)
            else:
                final_out = out1

            winner = final_out.argmax()
            combo_outputs[(a, b)].append(winner)

    # Determine output-to-target mapping by majority
    # Group combos by target value
    target_groups = {}
    for c in combos:
        tv = target_fn(c[0], c[1])
        if tv not in target_groups:
            target_groups[tv] = []
        target_groups[tv].append(c)

    # For each target value, find most common output across its combos
    output_to_target = {}
    target_to_output = {}
    n_target_vals = len(target_groups)

    # Count output frequencies per target group
    for tv, group_combos in target_groups.items():
        out_counts = {}
        for c in group_combos:
            for o in combo_outputs[c]:
                out_counts[o] = out_counts.get(o, 0) + 1
        best_out = max(out_counts, key=out_counts.get)
        target_to_output[tv] = best_out

    # Now compute accuracy using this mapping
    for c in combos:
        target = target_fn(c[0], c[1])
        expected_out = target_to_output[target]
        outputs = combo_outputs[c]
        n_correct = sum(1 for o in outputs if o == expected_out)
        per_combo[c] = n_correct / len(outputs)
        correct += n_correct
        total += len(outputs)

    return correct / total, per_combo, target_to_output


def run_test(column_type, n_train, n_eval, noise, window, lr, temperature,
             alpha, hold):
    rng = np.random.RandomState(42)

    # --- Single column baseline (2 inputs → 2 outputs) ---
    # Should work for AND/OR, fail for XOR
    cm_single = make_column(2, 2, column_type, window, lr, temperature, alpha)
    ring_single = np.zeros((2, window), dtype=np.float32)

    # --- Two-column chain (2 → 4 → 2) ---
    cm1 = make_column(2, 4, column_type, window, lr, temperature, alpha)
    cm2 = make_column(4, 2, column_type, window, lr, temperature, alpha)
    ring1 = np.zeros((2, window), dtype=np.float32)
    ring2 = np.zeros((4, window), dtype=np.float32)

    # Training: random (A, B) pairs, held for `hold` ticks
    combos = [(0, 0), (0, 1), (1, 0), (1, 1)]
    current = 0
    hold_counter = 0

    for t in range(n_train):
        if hold_counter <= 0:
            current = rng.randint(0, 4)
            hold_counter = hold
        hold_counter -= 1

        a, b = combos[current]
        sig = np.array([float(a), float(b)], dtype=np.float32)
        sig_noisy = sig + rng.randn(2).astype(np.float32) * noise

        # Train single column
        ring_single[:, :-1] = ring_single[:, 1:]
        ring_single[:, -1] = sig_noisy
        cm_single.tick(ring_single)

        # Train chain: column 1
        ring1[:, :-1] = ring1[:, 1:]
        ring1[:, -1] = sig_noisy
        cm1.tick(ring1)
        out1 = cm1.get_outputs()[0]  # (4,)

        # Train chain: column 2 (fed from column 1 output)
        ring2[:, :-1] = ring2[:, 1:]
        ring2[:, -1] = out1
        cm2.tick(ring2)

    # Evaluation
    fns = {
        'AND': lambda a, b: a & b,
        'OR':  lambda a, b: a | b,
        'XOR': lambda a, b: a ^ b,
    }

    results = {}
    for fn_name, fn in fns.items():
        # Single column
        ring_s_eval = np.zeros((2, window), dtype=np.float32)
        acc_s, per_s, map_s = eval_function(
            cm_single, None, ring_s_eval, None, fn, n_eval, noise, rng, window)

        # Chain
        ring1_eval = np.zeros((2, window), dtype=np.float32)
        ring2_eval = np.zeros((4, window), dtype=np.float32)
        acc_c, per_c, map_c = eval_function(
            cm1, cm2, ring1_eval, ring2_eval, fn, n_eval, noise, rng, window)

        results[fn_name] = {
            'single_acc': acc_s,
            'single_per_combo': per_s,
            'single_mapping': map_s,
            'chain_acc': acc_c,
            'chain_per_combo': per_c,
            'chain_mapping': map_c,
        }

    return results


def main():
    parser = argparse.ArgumentParser(description='Column XOR chain test')
    parser.add_argument('--column-type', type=str, default='conscience')
    parser.add_argument('--train', type=int, default=10000)
    parser.add_argument('--eval', type=int, default=200)
    parser.add_argument('--noise', type=float, default=0.05)
    parser.add_argument('--window', type=int, default=10)
    parser.add_argument('--lr', type=float, default=0.05)
    parser.add_argument('--temperature', type=float, default=0.2)
    parser.add_argument('--alpha', type=float, default=0.01)
    parser.add_argument('--hold', type=int, default=20,
                        help='Ticks to hold each input before switching (default: 20)')
    args = parser.parse_args()

    print(f"Column XOR test: type={args.column_type}, train={args.train}, "
          f"eval={args.eval}, noise={args.noise}, hold={args.hold}")
    print()

    results = run_test(args.column_type, args.train, args.eval, args.noise,
                       args.window, args.lr, args.temperature, args.alpha,
                       args.hold)

    print(f"{'function':>8} {'single':>8} {'chain':>8}  per-combo (chain)")
    print("-" * 60)
    for fn_name in ['AND', 'OR', 'XOR']:
        r = results[fn_name]
        combo_str = '  '.join(
            f'{c}:{r["chain_per_combo"][c]:.0%}'
            for c in [(0,0), (0,1), (1,0), (1,1)])
        single_mark = 'ok' if r['single_acc'] > 0.8 else 'FAIL'
        chain_mark = 'ok' if r['chain_acc'] > 0.8 else 'FAIL'
        print(f"{fn_name:>8} {r['single_acc']:>6.0%} {single_mark:>2}  "
              f"{r['chain_acc']:>5.0%} {chain_mark:>2}   {combo_str}")

    print()
    xor = results['XOR']
    if xor['chain_acc'] > 0.8 and xor['single_acc'] < 0.7:
        print("SUCCESS: chain solves XOR where single column fails!")
    elif xor['chain_acc'] > 0.8:
        print("Chain solves XOR (but single column also does — unexpected)")
    elif xor['single_acc'] > 0.8:
        print("Single column solves XOR?! (unexpected — check noise level)")
    else:
        print(f"Neither solves XOR reliably. Chain={xor['chain_acc']:.0%}, "
              f"Single={xor['single_acc']:.0%}")

    # Show column 1 categorization
    print(f"\nColumn 1 output mapping ({args.column_type}):")
    print(f"  target→output: {xor['chain_mapping']}")


if __name__ == '__main__':
    main()

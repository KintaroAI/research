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


def eval_outputs(cm1, cm2, ring1, ring2, n_eval, noise, rng):
    """Run eval ticks, collect per-output mean values for each input combo.

    Returns:
        combo_means: dict (a,b) → (n_outputs,) mean output values
        combo_raw: dict (a,b) → list of (n_outputs,) arrays per tick
    """
    combos = [(0, 0), (0, 1), (1, 0), (1, 1)]
    combo_sums = {c: None for c in combos}
    combo_counts = {c: 0 for c in combos}
    combo_raw = {c: [] for c in combos}

    for _ in range(n_eval):
        for a, b in combos:
            sig1 = np.array([float(a), float(b)], dtype=np.float32)
            sig1 += rng.randn(2).astype(np.float32) * noise
            ring1[:, :-1] = ring1[:, 1:]
            ring1[:, -1] = sig1
            cm1.tick(ring1)
            out1 = cm1.get_outputs()[0]

            if cm2 is not None:
                ring2[:, :-1] = ring2[:, 1:]
                ring2[:, -1] = out1
                cm2.tick(ring2)
                final_out = cm2.get_outputs()[0].copy()
            else:
                final_out = out1.copy()

            if combo_sums[(a, b)] is None:
                combo_sums[(a, b)] = np.zeros_like(final_out)
            combo_sums[(a, b)] += final_out
            combo_counts[(a, b)] += 1
            combo_raw[(a, b)].append(final_out)

    combo_means = {}
    for c in combos:
        combo_means[c] = combo_sums[c] / combo_counts[c]

    return combo_means, combo_raw


def score_function(combo_means, target_fn):
    """Find which single output best tracks a boolean function.

    For each output, compute how well it separates target=0 vs target=1
    by comparing mean values. Returns best output index and separation score.
    """
    combos = [(0, 0), (0, 1), (1, 0), (1, 1)]
    n_outputs = len(combo_means[(0, 0)])

    best_output = -1
    best_sep = 0.0
    best_inverted = False

    for o in range(n_outputs):
        # Mean output value when target=0 vs target=1
        vals_0 = [combo_means[c][o] for c in combos if target_fn(c[0], c[1]) == 0]
        vals_1 = [combo_means[c][o] for c in combos if target_fn(c[0], c[1]) == 1]
        mean_0 = np.mean(vals_0)
        mean_1 = np.mean(vals_1)
        sep = abs(mean_1 - mean_0)
        if sep > best_sep:
            best_sep = sep
            best_output = o
            best_inverted = mean_0 > mean_1

    # Compute accuracy: for each combo, does the best output correctly
    # predict the target? (threshold at midpoint)
    vals_0 = [combo_means[c][best_output] for c in combos if target_fn(c[0], c[1]) == 0]
    vals_1 = [combo_means[c][best_output] for c in combos if target_fn(c[0], c[1]) == 1]
    threshold = (np.mean(vals_0) + np.mean(vals_1)) / 2

    per_combo = {}
    for c in combos:
        target = target_fn(c[0], c[1])
        val = combo_means[c][best_output]
        if best_inverted:
            pred = 0 if val > threshold else 1
        else:
            pred = 1 if val > threshold else 0
        per_combo[c] = float(pred == target)

    acc = np.mean(list(per_combo.values()))

    return {
        'best_output': best_output,
        'separation': float(best_sep),
        'accuracy': float(acc),
        'per_combo': per_combo,
        'inverted': best_inverted,
        'mean_when_0': float(np.mean(vals_0)),
        'mean_when_1': float(np.mean(vals_1)),
    }


def run_test(column_type, n_train, n_eval, noise, window, lr, temperature,
             alpha, hold):
    rng = np.random.RandomState(42)

    # --- Single column baseline (2 inputs → 2 outputs) ---
    # Should work for AND/OR, fail for XOR
    cm_single = make_column(2, 2, column_type, window, lr, temperature, alpha)
    ring_single = np.zeros((2, window), dtype=np.float32)

    # --- Two-column chain (2 → 4 → 4) ---
    cm1 = make_column(2, 4, column_type, window, lr, temperature, alpha)
    cm2 = make_column(4, 4, column_type, window, lr, temperature, alpha)
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

    # Evaluation: collect mean output values per input combo
    ring_s_eval = np.zeros((2, window), dtype=np.float32)
    single_means, _ = eval_outputs(cm_single, None, ring_s_eval, None,
                                    n_eval, noise, rng)

    ring1_eval = np.zeros((2, window), dtype=np.float32)
    ring2_eval = np.zeros((4, window), dtype=np.float32)
    chain_means, _ = eval_outputs(cm1, cm2, ring1_eval, ring2_eval,
                                   n_eval, noise, rng)

    fns = {
        'AND': lambda a, b: a & b,
        'OR':  lambda a, b: a | b,
        'XOR': lambda a, b: a ^ b,
    }

    results = {}
    for fn_name, fn in fns.items():
        results[fn_name] = {
            'single': score_function(single_means, fn),
            'chain': score_function(chain_means, fn),
        }

    # Also report raw combo means for chain
    results['_chain_means'] = {str(c): chain_means[c].tolist() for c in chain_means}
    results['_single_means'] = {str(c): single_means[c].tolist() for c in single_means}

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

    print(f"{'function':>8} {'single':>20} {'chain':>20}")
    print(f"{'':>8} {'out  sep   acc':>20} {'out  sep   acc':>20}")
    print("-" * 52)
    for fn_name in ['AND', 'OR', 'XOR']:
        s = results[fn_name]['single']
        c = results[fn_name]['chain']
        s_mark = 'ok' if s['accuracy'] >= 1.0 else '--'
        c_mark = 'ok' if c['accuracy'] >= 1.0 else '--'
        print(f"{fn_name:>8}   "
              f"o{s['best_output']} {s['separation']:.3f} {s['accuracy']:.0%} {s_mark:>2}   "
              f"o{c['best_output']} {c['separation']:.3f} {c['accuracy']:.0%} {c_mark:>2}")

    # Detailed chain output values per combo
    print(f"\nChain output means per input combo:")
    combos = [(0,0), (0,1), (1,0), (1,1)]
    cm = results['_chain_means']
    n_out = len(cm[str(combos[0])])
    header = '  '.join(f'out{o}' for o in range(n_out))
    print(f"  {'input':>7}  {header}   XOR")
    for c in combos:
        vals = cm[str(c)]
        xor_val = c[0] ^ c[1]
        val_str = '  '.join(f'{v:.3f}' for v in vals)
        print(f"  {str(c):>7}  {val_str}    {xor_val}")

    print()
    xor_c = results['XOR']['chain']
    xor_s = results['XOR']['single']
    if xor_c['accuracy'] >= 1.0 and xor_s['accuracy'] < 1.0:
        print(f"SUCCESS: chain output {xor_c['best_output']} computes XOR "
              f"(sep={xor_c['separation']:.3f}), single column can't")
    elif xor_c['accuracy'] >= 1.0:
        print(f"Chain output {xor_c['best_output']} computes XOR "
              f"(sep={xor_c['separation']:.3f})")
    else:
        print(f"XOR not cleanly separated. "
              f"Best chain output {xor_c['best_output']} sep={xor_c['separation']:.3f}, "
              f"acc={xor_c['accuracy']:.0%}")


if __name__ == '__main__':
    main()

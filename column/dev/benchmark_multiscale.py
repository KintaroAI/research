"""Multi-scale temporal benchmark — cells at different time scales.

Multiple cells see the same input stream, each with a different streaming_decay.
Fast cell captures rapid changes, slow cell captures trends. Together they
separate patterns that no single time scale can.

Test data: 4 slow trends × 2 fast sub-patterns = 8 total categories.
A slow trend is a direction that persists for many frames. Within each trend,
the signal alternates rapidly between two sub-patterns.

Usage:
    python benchmark_multiscale.py [-o output_dir]
"""

import argparse
import json
import os

import numpy as np
import torch

from column import SoftWTACell
from metrics import normalized_mutual_info


def make_multiscale_data(n_samples, n_inputs=16, n_slow=4, n_fast=2,
                         slow_block=100, fast_block=5, noise=0.3, seed=42):
    """Generate data with slow + fast co-variation at different timescales.

    Slow signal: a latent driver that varies over ~slow_block frames,
    projected onto a slow direction. Changes direction every slow_block frames.

    Fast signal: a latent driver that varies every frame, projected onto
    a fast direction. Changes direction every fast_block frames.

    Both signals produce actual temporal variance (not DC bias), so streaming
    cells with different decays can detect them.
    """
    rng = np.random.default_rng(seed)

    # Slow and fast co-variation directions
    slow_dirs = rng.standard_normal((n_slow, n_inputs)).astype(np.float32)
    slow_dirs = slow_dirs / np.linalg.norm(slow_dirs, axis=1, keepdims=True)
    fast_dirs = rng.standard_normal((n_fast, n_inputs)).astype(np.float32)
    fast_dirs = fast_dirs / np.linalg.norm(fast_dirs, axis=1, keepdims=True)

    data = []
    slow_labels = []
    fast_labels = []
    combined_labels = []

    slow_cat = 0
    fast_cat = 0
    # Slow latent signal: smooth, changes over many frames
    slow_phase = 0.0
    for i in range(n_samples):
        if i % slow_block == 0:
            slow_cat = rng.integers(n_slow)
            slow_phase = rng.random() * 2 * np.pi
        if i % fast_block == 0:
            fast_cat = rng.integers(n_fast)

        # Slow co-variation: sinusoidal driver along slow direction
        slow_driver = np.sin(slow_phase + 2 * np.pi * (i % slow_block) / slow_block)
        slow_signal = slow_dirs[slow_cat] * slow_driver

        # Fast co-variation: random driver along fast direction (new each frame)
        fast_driver = rng.standard_normal()
        fast_signal = fast_dirs[fast_cat] * fast_driver * 0.7

        signal = slow_signal + fast_signal + rng.standard_normal(n_inputs).astype(np.float32) * noise
        data.append(signal.astype(np.float32))
        slow_labels.append(slow_cat)
        fast_labels.append(fast_cat)
        combined_labels.append(slow_cat * n_fast + fast_cat)

    return (np.array(data, dtype=np.float32),
            np.array(slow_labels),
            np.array(fast_labels),
            np.array(combined_labels))


def run_single_scale(data, labels, n_outputs, decay, name=""):
    """Single streaming cell at one time scale."""
    n_inputs = data.shape[1]
    cell = SoftWTACell(n_inputs, n_outputs, temperature=0.5, lr=0.05,
                       match_threshold=0.1, temporal_mode='streaming',
                       streaming_decay=decay)
    winners = []
    for i in range(len(data)):
        x = torch.from_numpy(data[i])
        p = cell.forward(x)
        w, _ = cell.update(x, p)
        winners.append(w)

    eval_start = len(data) // 2
    w = np.array(winners[eval_start:])
    l = labels[eval_start:]
    return normalized_mutual_info(w, l)


def run_multiscale(data, slow_labels, fast_labels, combined_labels,
                   n_outputs_fast=4, n_outputs_slow=4,
                   decay_fast=0.5, decay_slow=0.95):
    """Two cells at different time scales, combined output."""
    n_inputs = data.shape[1]
    n_samples = len(data)

    cell_fast = SoftWTACell(n_inputs, n_outputs_fast, temperature=0.5, lr=0.05,
                            match_threshold=0.1, temporal_mode='streaming',
                            streaming_decay=decay_fast)
    cell_slow = SoftWTACell(n_inputs, n_outputs_slow, temperature=0.5, lr=0.05,
                            match_threshold=0.1, temporal_mode='streaming',
                            streaming_decay=decay_slow)

    fast_winners = []
    slow_winners = []
    combined_winners = []

    for i in range(n_samples):
        x = torch.from_numpy(data[i])

        pf = cell_fast.forward(x)
        wf, _ = cell_fast.update(x, pf)
        fast_winners.append(wf)

        ps = cell_slow.forward(x)
        ws, _ = cell_slow.update(x, ps)
        slow_winners.append(ws)

        combined_winners.append(ws * n_outputs_fast + wf)

    eval_start = n_samples // 2
    fw = np.array(fast_winners[eval_start:])
    sw = np.array(slow_winners[eval_start:])
    cw = np.array(combined_winners[eval_start:])
    sl = slow_labels[eval_start:]
    fl = fast_labels[eval_start:]
    cl = combined_labels[eval_start:]

    return {
        'fast_vs_fast_labels': normalized_mutual_info(fw, fl),
        'fast_vs_slow_labels': normalized_mutual_info(fw, sl),
        'slow_vs_slow_labels': normalized_mutual_info(sw, sl),
        'slow_vs_fast_labels': normalized_mutual_info(sw, fl),
        'combined_vs_combined': normalized_mutual_info(cw, cl),
        'fast_vs_combined': normalized_mutual_info(fw, cl),
        'slow_vs_combined': normalized_mutual_info(sw, cl),
    }


def main():
    parser = argparse.ArgumentParser(description='Multi-scale temporal benchmark')
    parser.add_argument('-o', '--output', type=str, default=None)
    parser.add_argument('--frames', type=int, default=20000)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    print("=" * 64)
    print("Multi-Scale Temporal Benchmark")
    print("=" * 64)
    print(f"4 slow trends (block=100) × 2 fast sub-patterns (block=5)")
    print(f"= 8 combined categories, {args.frames} frames")

    data, slow_labels, fast_labels, combined_labels = \
        make_multiscale_data(args.frames, seed=args.seed)

    results = {}

    # Single cells at various decays
    print(f"\n[1/3] Single cells at different decays")
    print(f"  {'decay':<10}{'vs slow':>10}{'vs fast':>10}{'vs combined':>10}")
    print("  " + "-" * 38)

    for decay in [0.3, 0.5, 0.7, 0.9, 0.95, 0.99]:
        nmi_slow = run_single_scale(data, slow_labels, 4, decay)
        nmi_fast = run_single_scale(data, fast_labels, 2, decay)
        nmi_comb = run_single_scale(data, combined_labels, 8, decay)
        results[f'single_d{decay}'] = {
            'slow': nmi_slow, 'fast': nmi_fast, 'combined': nmi_comb}
        print(f"  {decay:<10}{nmi_slow:>10.3f}{nmi_fast:>10.3f}{nmi_comb:>10.3f}")

    # Multi-scale: fast + slow cells combined
    print(f"\n[2/3] Multi-scale (fast decay=0.5, slow decay=0.95)")
    ms = run_multiscale(data, slow_labels, fast_labels, combined_labels)
    results['multiscale'] = ms

    print(f"  Fast cell  vs fast labels:  {ms['fast_vs_fast_labels']:.3f}")
    print(f"  Fast cell  vs slow labels:  {ms['fast_vs_slow_labels']:.3f}")
    print(f"  Slow cell  vs slow labels:  {ms['slow_vs_slow_labels']:.3f}")
    print(f"  Slow cell  vs fast labels:  {ms['slow_vs_fast_labels']:.3f}")
    print(f"  Combined   vs combined:     {ms['combined_vs_combined']:.3f}")
    print(f"  Fast alone vs combined:     {ms['fast_vs_combined']:.3f}")
    print(f"  Slow alone vs combined:     {ms['slow_vs_combined']:.3f}")

    # Best single cell vs multi-scale
    print(f"\n[3/3] Summary — combined NMI (8 categories)")
    best_single = max(results[f'single_d{d}']['combined']
                      for d in [0.3, 0.5, 0.7, 0.9, 0.95, 0.99])
    print(f"  Best single cell:   {best_single:.3f}")
    print(f"  Multi-scale (2 cells): {ms['combined_vs_combined']:.3f}")

    if args.output:
        os.makedirs(args.output, exist_ok=True)
        with open(os.path.join(args.output, 'benchmark_multiscale.json'), 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved to {args.output}/benchmark_multiscale.json")


if __name__ == '__main__':
    main()

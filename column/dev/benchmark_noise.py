"""Noise robustness benchmark — how does random noise on a subset of
input channels affect separation quality?

Tests whether the cell degrades gracefully when some channels carry
pure noise while others carry meaningful signal. This simulates
real-world scenarios where some sensors are broken/irrelevant.

Sweeps:
  1. Fraction of noisy channels (0% to 90%) at fixed n_inputs=16
  2. Fixed fraction (50%) with varying noise amplitude

Usage:
    python benchmark_noise.py [-o output_dir]
"""

import argparse
import json
import os

import numpy as np
import torch

from column import SoftWTACell
from metrics import compute_sqm, format_sqm


def make_noisy_channel_data(n_clusters, n_inputs, n_samples, n_noisy=0,
                            noise_amplitude=1.0, signal_noise=0.1, seed=42):
    """Generate clustered data where some channels are pure noise.

    First (n_inputs - n_noisy) channels carry cluster signal.
    Last n_noisy channels are replaced with random noise.
    """
    rng = np.random.default_rng(seed)
    n_signal = n_inputs - n_noisy

    # Cluster centers only in signal channels
    centers_signal = rng.standard_normal((n_clusters, n_signal)).astype(np.float32)
    centers_signal = centers_signal / np.linalg.norm(centers_signal, axis=1, keepdims=True)

    data = []
    labels = []
    for i in range(n_samples):
        c = rng.integers(n_clusters)
        # Signal channels: cluster center + small noise
        signal = centers_signal[c] + rng.standard_normal(n_signal).astype(np.float32) * signal_noise
        # Noise channels: pure random
        noise = rng.standard_normal(n_noisy).astype(np.float32) * noise_amplitude
        x = np.concatenate([signal, noise])
        data.append(x)
        labels.append(c)

    return np.array(data, dtype=np.float32), np.array(labels)


def run_cell(data, labels, n_outputs, n_frames=None):
    """Train and evaluate a cell."""
    if n_frames is None:
        n_frames = len(data)
    n_inputs = data.shape[1]

    cell = SoftWTACell(n_inputs, n_outputs, temperature=0.3, lr=0.05)

    winners, probs = [], []
    for i in range(n_frames):
        x = torch.from_numpy(data[i])
        p = cell.forward(x)
        w, _ = cell.update(x, p)
        winners.append(w)
        probs.append(p.detach().numpy())

    w_arr = np.array(winners)
    p_arr = np.array(probs)
    # Evaluate on last half
    half = len(w_arr) // 2
    sqm = compute_sqm(w_arr[half:], p_arr[half:], cell.prototypes.numpy(),
                      n_outputs, labels=labels[half:n_frames])
    return sqm


def main():
    parser = argparse.ArgumentParser(description='Noise robustness benchmark')
    parser.add_argument('-o', '--output', type=str, default=None)
    parser.add_argument('--samples', type=int, default=10000)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    n_inputs = 16
    n_clusters = 4
    n_outputs = 4

    print("=" * 68)
    print("Noise Robustness Benchmark")
    print("=" * 68)
    print(f"{n_inputs} inputs, {n_clusters} clusters, {args.samples} frames")

    all_results = {}

    # Sweep 1: fraction of noisy channels
    print(f"\n{'─' * 68}")
    print("Sweep 1: Noisy channel fraction (noise_amplitude=1.0)")
    print(f"{'─' * 68}")
    print(f"  {'noisy_frac':<14}{'n_noisy':>10}{'NMI':>10}{'purity':>10}"
          f"{'consist':>10}{'conf_gap':>10}")
    print("  " + "-" * 62)

    sweep1 = []
    for n_noisy in [0, 2, 4, 6, 8, 10, 12, 14]:
        frac = n_noisy / n_inputs
        data, labels = make_noisy_channel_data(
            n_clusters, n_inputs, args.samples,
            n_noisy=n_noisy, seed=args.seed)
        sqm = run_cell(data, labels, n_outputs)
        sqm['n_noisy'] = n_noisy
        sqm['noisy_frac'] = frac
        sweep1.append(sqm)
        print(f"  {frac:<14.0%}{n_noisy:>10d}{sqm['nmi']:>10.3f}"
              f"{sqm['purity']:>10.3f}{sqm['consistency']:>10.3f}"
              f"{sqm['confidence_gap']:>10.3f}")

    all_results['sweep_fraction'] = sweep1

    # Sweep 2: noise amplitude at 50% noisy channels
    print(f"\n{'─' * 68}")
    print("Sweep 2: Noise amplitude (50% noisy channels, 8 of 16)")
    print(f"{'─' * 68}")
    print(f"  {'amplitude':<14}{'NMI':>10}{'purity':>10}"
          f"{'consist':>10}{'conf_gap':>10}")
    print("  " + "-" * 52)

    sweep2 = []
    for amp in [0.0, 0.1, 0.3, 0.5, 1.0, 2.0, 5.0]:
        data, labels = make_noisy_channel_data(
            n_clusters, n_inputs, args.samples,
            n_noisy=8, noise_amplitude=amp, seed=args.seed)
        sqm = run_cell(data, labels, n_outputs)
        sqm['noise_amplitude'] = amp
        sweep2.append(sqm)
        print(f"  {amp:<14.1f}{sqm['nmi']:>10.3f}{sqm['purity']:>10.3f}"
              f"{sqm['consistency']:>10.3f}{sqm['confidence_gap']:>10.3f}")

    all_results['sweep_amplitude'] = sweep2

    if args.output:
        os.makedirs(args.output, exist_ok=True)
        with open(os.path.join(args.output, 'benchmark_noise.json'), 'w') as f:
            json.dump(all_results, f, indent=2, default=lambda x: x.tolist()
                      if hasattr(x, 'tolist') else x)
        print(f"\nSaved to {args.output}/benchmark_noise.json")


if __name__ == '__main__':
    main()

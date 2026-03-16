"""Dynamic input benchmark — extend and remove channels on a live cell.

Tests:
  1. Extend: add channels to a trained cell, verify zero disruption
  2. Extend + retrain: new channel carries useful signal, cell learns it
  3. Extend + noise: new channel is noise, cell ignores it
  4. Remove: drop a channel, verify recovery after retraining
  5. Replace: zero out a broken channel, retrain

Usage:
    python benchmark_dynamic_inputs.py [-o output_dir]
"""

import argparse
import json
import os

import numpy as np
import torch

from column import SoftWTACell
from main import make_clustered_data
from metrics import compute_sqm, format_sqm, normalized_mutual_info


def eval_nmi(cell, data, labels, n=1000):
    """Evaluate cell NMI on data (forward only, no update)."""
    start = max(0, len(data) - n)
    w = []
    for i in range(start, len(data)):
        p = cell.forward(torch.from_numpy(data[i]))
        w.append(p.argmax().item())
    return normalized_mutual_info(np.array(w), labels[start:])


def scenario_extend_zero_disruption(seed=42):
    """Add channels with zero init — categorization should be unchanged."""
    data, labels, _ = make_clustered_data(4, 10, 8000, seed=seed)
    cell = SoftWTACell(10, 4, temperature=0.3, lr=0.05)
    for i in range(5000):
        cell.update(torch.from_numpy(data[i]))
    nmi_before = eval_nmi(cell, data[5000:], labels[5000:])

    cell.extend_inputs(4)  # 10 → 14

    rng = np.random.default_rng(seed)
    data14 = np.concatenate([data, rng.standard_normal((8000, 4)).astype(np.float32) * 0.01], axis=1)
    nmi_after = eval_nmi(cell, data14[5000:], labels[5000:])

    return {'nmi_before': nmi_before, 'nmi_after': nmi_after, 'extended_by': 4}


def scenario_extend_useful(seed=42):
    """Add a channel that carries cluster-correlated signal."""
    data, labels, _ = make_clustered_data(4, 10, 10000, seed=seed)
    cell = SoftWTACell(10, 4, temperature=0.3, lr=0.05)
    for i in range(5000):
        cell.update(torch.from_numpy(data[i]))
    nmi_before = eval_nmi(cell, data[5000:7000], labels[5000:7000])

    cell.extend_inputs(1)  # 10 → 11

    rng = np.random.default_rng(seed + 1)
    useful_ch = (labels * 0.5 + rng.standard_normal(10000) * 0.1).astype(np.float32)
    data11 = np.concatenate([data, useful_ch.reshape(-1, 1)], axis=1)

    # Retrain
    for i in range(5000, 8000):
        cell.update(torch.from_numpy(data11[i]))
    nmi_retrained = eval_nmi(cell, data11[8000:], labels[8000:])
    weight_new = abs(cell.prototypes[:, 10].mean().item())

    return {'nmi_before': nmi_before, 'nmi_retrained': nmi_retrained,
            'new_channel_weight': weight_new}


def scenario_extend_noise(seed=42):
    """Add a noisy channel — cell should ignore it."""
    data, labels, _ = make_clustered_data(4, 10, 10000, seed=seed)
    cell = SoftWTACell(10, 4, temperature=0.3, lr=0.05)
    for i in range(5000):
        cell.update(torch.from_numpy(data[i]))
    nmi_before = eval_nmi(cell, data[5000:7000], labels[5000:7000])

    cell.extend_inputs(1)

    rng = np.random.default_rng(seed + 2)
    noise_ch = rng.standard_normal((10000, 1)).astype(np.float32) * 0.3
    data11 = np.concatenate([data, noise_ch], axis=1)

    for i in range(5000, 8000):
        cell.update(torch.from_numpy(data11[i]))
    nmi_after_noise = eval_nmi(cell, data11[8000:], labels[8000:])
    weight_noise = abs(cell.prototypes[:, 10].mean().item())

    return {'nmi_before': nmi_before, 'nmi_after_noise': nmi_after_noise,
            'noise_channel_weight': weight_noise}


def scenario_remove(seed=42):
    """Remove a channel — measure impact and recovery."""
    data, labels, _ = make_clustered_data(4, 10, 10000, seed=seed)
    cell = SoftWTACell(10, 4, temperature=0.3, lr=0.05)
    for i in range(5000):
        cell.update(torch.from_numpy(data[i]))
    nmi_before = eval_nmi(cell, data[5000:7000], labels[5000:7000])

    cell.remove_inputs(0)  # remove first channel
    data9 = data[:, 1:]  # drop first column from data

    nmi_immediately = eval_nmi(cell, data9[5000:7000], labels[5000:7000])

    for i in range(5000, 8000):
        cell.update(torch.from_numpy(data9[i]))
    nmi_retrained = eval_nmi(cell, data9[8000:], labels[8000:])

    return {'nmi_before': nmi_before, 'nmi_after_remove': nmi_immediately,
            'nmi_retrained': nmi_retrained}


def scenario_replace_broken(seed=42):
    """Zero out a broken channel, retrain — simulates sensor replacement."""
    data, labels, _ = make_clustered_data(4, 10, 10000, seed=seed)
    cell = SoftWTACell(10, 4, temperature=0.3, lr=0.05)
    for i in range(5000):
        cell.update(torch.from_numpy(data[i]))
    nmi_before = eval_nmi(cell, data[5000:7000], labels[5000:7000])

    # Zero out channel 0's prototype weight (simulate "forgetting" it)
    cell.prototypes[:, 0] = 0
    cell.prototypes = torch.nn.functional.normalize(cell.prototypes, dim=1)

    nmi_zeroed = eval_nmi(cell, data[5000:7000], labels[5000:7000])

    for i in range(5000, 8000):
        cell.update(torch.from_numpy(data[i]))
    nmi_retrained = eval_nmi(cell, data[8000:], labels[8000:])

    return {'nmi_before': nmi_before, 'nmi_after_zero': nmi_zeroed,
            'nmi_retrained': nmi_retrained}


def main():
    parser = argparse.ArgumentParser(description='Dynamic input benchmark')
    parser.add_argument('-o', '--output', type=str, default=None)
    args = parser.parse_args()

    print("=" * 60)
    print("Dynamic Input Benchmark")
    print("=" * 60)

    results = {}

    print("\n[1/5] Extend — zero disruption...")
    r = scenario_extend_zero_disruption()
    results['extend_zero'] = r
    print(f"  Before: NMI={r['nmi_before']:.3f}")
    print(f"  After +{r['extended_by']} channels (no retrain): NMI={r['nmi_after']:.3f}")

    print("\n[2/5] Extend — useful new channel...")
    r = scenario_extend_useful()
    results['extend_useful'] = r
    print(f"  Before: NMI={r['nmi_before']:.3f}")
    print(f"  After retrain with useful channel: NMI={r['nmi_retrained']:.3f}")
    print(f"  New channel prototype weight: {r['new_channel_weight']:.3f}")

    print("\n[3/5] Extend — noisy new channel...")
    r = scenario_extend_noise()
    results['extend_noise'] = r
    print(f"  Before: NMI={r['nmi_before']:.3f}")
    print(f"  After retrain with noise channel: NMI={r['nmi_after_noise']:.3f}")
    print(f"  Noise channel prototype weight: {r['noise_channel_weight']:.3f}")

    print("\n[4/5] Remove channel...")
    r = scenario_remove()
    results['remove'] = r
    print(f"  Before: NMI={r['nmi_before']:.3f}")
    print(f"  After remove (no retrain): NMI={r['nmi_after_remove']:.3f}")
    print(f"  After retrain: NMI={r['nmi_retrained']:.3f}")

    print("\n[5/5] Replace broken channel...")
    r = scenario_replace_broken()
    results['replace'] = r
    print(f"  Before: NMI={r['nmi_before']:.3f}")
    print(f"  After zeroing channel: NMI={r['nmi_after_zero']:.3f}")
    print(f"  After retrain: NMI={r['nmi_retrained']:.3f}")

    if args.output:
        os.makedirs(args.output, exist_ok=True)
        with open(os.path.join(args.output, 'benchmark_dynamic.json'), 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved to {args.output}/benchmark_dynamic.json")


if __name__ == '__main__':
    main()

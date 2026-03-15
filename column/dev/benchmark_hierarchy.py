"""Hierarchical cell benchmark — sequential stacking of SoftWTACells.

Tests whether two stacked cells can learn movement *patterns* that neither
cell can learn alone:
  - Cell 1 (temporal, on raw 2D positions) → categorizes instantaneous direction
  - Cell 2 (temporal, on Cell 1 output history) → categorizes the movement pattern

Movement patterns:
  - steady:     constant direction
  - oscillate:  alternating between two opposite directions
  - zigzag:     alternating between two perpendicular directions
  - circle:     smoothly rotating direction

Usage:
    python benchmark_hierarchy.py [-o output_dir]
"""

import argparse
import json
import math
import os

import numpy as np
import torch

from column import SoftWTACell
from metrics import compute_sqm, format_sqm, normalized_mutual_info


# ── Data generation ───────────────────────────────────────────────────

def make_movement_patterns(n_samples, T_segment=10, n_segments=8,
                           speed=1.0, noise=0.2, seed=42):
    """Generate 2D trajectories with distinct movement patterns.

    Each sample is a full trajectory: (2, T_segment * n_segments) positions.
    Also returns per-segment direction labels and pattern labels.

    Patterns:
      0: steady    — same direction throughout
      1: oscillate — flip direction every segment
      2: zigzag    — rotate 90° every segment
      3: circle    — rotate smoothly by 360°/n_segments each segment
    """
    rng = np.random.default_rng(seed)
    n_patterns = 4
    T_total = T_segment * n_segments

    # 4 base directions (evenly spaced on unit circle)
    n_dirs = 4
    base_angles = np.linspace(0, 2 * np.pi, n_dirs, endpoint=False)
    base_dirs = np.stack([np.cos(base_angles), np.sin(base_angles)], axis=1)

    traces = []           # (2, T_total) position traces
    pattern_labels = []   # which pattern (0-3)
    segment_dirs = []     # per-segment direction index (for Cell 1 evaluation)

    for i in range(n_samples):
        pattern = rng.integers(n_patterns)
        start_dir = rng.integers(n_dirs)
        start_pos = rng.standard_normal(2) * 5.0

        positions = []
        dirs_for_sample = []
        pos = start_pos.copy()

        for seg in range(n_segments):
            if pattern == 0:  # steady
                d_idx = start_dir
            elif pattern == 1:  # oscillate
                d_idx = start_dir if seg % 2 == 0 else (start_dir + n_dirs // 2) % n_dirs
            elif pattern == 2:  # zigzag
                d_idx = start_dir if seg % 2 == 0 else (start_dir + n_dirs // 4) % n_dirs
            elif pattern == 3:  # circle
                d_idx = (start_dir + seg) % n_dirs

            direction = base_dirs[d_idx]
            dirs_for_sample.append(d_idx)

            for t in range(T_segment):
                pos = pos + direction * speed + rng.standard_normal(2) * noise
                positions.append(pos.copy())

        trace = np.array(positions).T  # (2, T_total)
        traces.append(trace.astype(np.float32))
        pattern_labels.append(pattern)
        segment_dirs.append(dirs_for_sample)

    return (traces, np.array(pattern_labels), np.array(segment_dirs),
            T_segment, n_segments)


# ── Benchmark scenarios ──────────────────────────────────────────────

def run_hierarchy(traces, pattern_labels, segment_dirs, T_seg, n_seg):
    """Run the two-cell hierarchy and evaluate.

    Cell 1: temporal correlation on raw position segments → direction category
    Cell 2: instantaneous on transition matrix from Cell 1 winner sequence

    The transition matrix is direction-invariant: it captures HOW winners change
    over time, not WHICH specific winner fires. "Steady-north" and "steady-east"
    both produce diagonal transition matrices, while "oscillate" produces
    anti-diagonal patterns.
    """
    n_samples = len(traces)
    n_dir_categories = 4

    # Cell 1: learns instantaneous direction from position trace segments
    cell1 = SoftWTACell(
        n_inputs=2, n_outputs=n_dir_categories,
        temperature=0.5, lr=0.05, match_threshold=0.1,
        temporal_mode='correlation'
    )

    # Cell 2: learns movement patterns from Cell 1 transition matrix
    n_trans = n_dir_categories * n_dir_categories
    cell2 = SoftWTACell(
        n_inputs=n_trans, n_outputs=4,
        temperature=0.3, lr=0.05, match_threshold=0.5,
    )

    # Track metrics
    cell2_winners = []
    cell2_probs = []
    cell1_all_segment_winners = []
    cell1_all_segment_labels = []

    for i in range(n_samples):
        trace = traces[i]  # (2, T_total)
        seg_winners = []

        # Phase 1: feed segments through Cell 1
        for seg in range(n_seg):
            start = seg * T_seg
            end = start + T_seg
            segment_trace = torch.from_numpy(trace[:, start:end])
            p1 = cell1.forward(segment_trace)
            w1, _ = cell1.update(segment_trace, p1)
            seg_winners.append(w1)
            cell1_all_segment_winners.append(w1)
            cell1_all_segment_labels.append(segment_dirs[i][seg])

        # Phase 2: estimate transition matrix from Cell 1 winner sequence
        trans = np.zeros((n_dir_categories, n_dir_categories), dtype=np.float32)
        for t in range(len(seg_winners) - 1):
            trans[seg_winners[t], seg_winners[t + 1]] += 1
        row_sums = trans.sum(axis=1, keepdims=True)
        row_sums = np.maximum(row_sums, 1e-8)
        trans = trans / row_sums

        trans_flat = torch.from_numpy(trans.flatten())
        p2 = cell2.forward(trans_flat)
        w2, _ = cell2.update(trans_flat, p2)
        cell2_winners.append(w2)
        cell2_probs.append(p2.detach().numpy())

    # Evaluate Cell 1: direction categorization
    c1_w = np.array(cell1_all_segment_winners)
    c1_l = np.array(cell1_all_segment_labels)
    c1_nmi = normalized_mutual_info(c1_w, c1_l)

    # Evaluate Cell 2: pattern categorization
    c2_w = np.array(cell2_winners)
    c2_p = np.array(cell2_probs)
    c2_sqm = compute_sqm(c2_w, c2_p, cell2.prototypes.numpy(), 4,
                         labels=pattern_labels)

    return {
        'cell1_direction_nmi': c1_nmi,
        'cell2_sqm': c2_sqm,
        'cell1_prototypes': cell1.prototypes.numpy(),
        'cell2_prototypes': cell2.prototypes.numpy(),
    }


def run_cell1_only(traces, pattern_labels, segment_dirs, T_seg, n_seg):
    """Cell 1 alone trying to learn patterns (should fail — only sees direction)."""
    n_samples = len(traces)
    cell = SoftWTACell(
        n_inputs=2, n_outputs=4,
        temperature=0.5, lr=0.05, match_threshold=0.1,
        temporal_mode='correlation'
    )

    winners, probs = [], []
    for i in range(n_samples):
        trace = traces[i]
        # Feed full trajectory as one big trace — Cell 1 sees overall covariance
        # but can't distinguish temporal patterns (steady vs oscillate both have
        # similar long-run covariance when averaged over many segments)
        x = torch.from_numpy(trace)
        p = cell.forward(x)
        w, _ = cell.update(x, p)
        winners.append(w)
        probs.append(p.detach().numpy())

    w_arr = np.array(winners)
    p_arr = np.array(probs)
    sqm = compute_sqm(w_arr, p_arr, cell.prototypes.numpy(), 4,
                      labels=pattern_labels)
    return sqm


def run_cell2_only(traces, pattern_labels, segment_dirs, T_seg, n_seg):
    """Single cell trying to learn patterns from raw positions directly."""
    n_samples = len(traces)

    cell = SoftWTACell(
        n_inputs=2, n_outputs=4,
        temperature=0.5, lr=0.05, match_threshold=0.1,
        temporal_mode='correlation'
    )

    winners, probs = [], []
    for i in range(n_samples):
        trace = traces[i]  # (2, T_total)
        x = torch.from_numpy(trace)
        p = cell.forward(x)
        w, _ = cell.update(x, p)
        winners.append(w)
        probs.append(p.detach().numpy())

    w_arr = np.array(winners)
    p_arr = np.array(probs)
    sqm = compute_sqm(w_arr, p_arr, cell.prototypes.numpy(), 4,
                      labels=pattern_labels)
    return sqm


def print_results(hierarchy, cell1_only, cell2_only):
    """Print formatted comparison."""
    c2h = hierarchy['cell2_sqm']

    print(f"\n{'metric':<22}{'hierarchy':>14}{'cell1_only':>14}{'cell2_only':>14}")
    print("-" * 64)

    print(f"{'cell1_dir_nmi':<22}{hierarchy['cell1_direction_nmi']:>14.3f}"
          f"{'—':>14}{'—':>14}")

    for k in ['winner_entropy', 'usage_gini', 'confidence_gap',
              'prototype_spread', 'nmi', 'purity', 'consistency']:
        v_h = c2h.get(k, 0)
        v_1 = cell1_only.get(k, 0)
        v_2 = cell2_only.get(k, 0)
        print(f"{k:<22}{v_h:>14.3f}{v_1:>14.3f}{v_2:>14.3f}")


def main():
    parser = argparse.ArgumentParser(description='Hierarchical cell benchmark')
    parser.add_argument('-o', '--output', type=str, default=None)
    parser.add_argument('--samples', type=int, default=5000)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    print("=" * 64)
    print("Hierarchical Cell Benchmark")
    print("=" * 64)
    print(f"Patterns: steady, oscillate, zigzag, circle")
    print(f"Samples: {args.samples}, T_segment=10, n_segments=12")

    traces, pattern_labels, segment_dirs, T_seg, n_seg = \
        make_movement_patterns(args.samples, n_segments=12, seed=args.seed)

    print("\n[1/3] Two-cell hierarchy (Cell 1 → Cell 2)...")
    hierarchy = run_hierarchy(traces, pattern_labels, segment_dirs, T_seg, n_seg)
    print(f"  Cell 1 direction NMI: {hierarchy['cell1_direction_nmi']:.3f}")
    print(f"  Cell 2 pattern SQM:   {format_sqm(hierarchy['cell2_sqm'])}")

    print("\n[2/3] Cell 1 alone on patterns (full trajectory)...")
    cell1_only = run_cell1_only(traces, pattern_labels, segment_dirs, T_seg, n_seg)
    print(f"  {format_sqm(cell1_only)}")

    print("\n[3/3] Cell 2 alone on raw positions...")
    cell2_only = run_cell2_only(traces, pattern_labels, segment_dirs, T_seg, n_seg)
    print(f"  {format_sqm(cell2_only)}")

    print_results(hierarchy, cell1_only, cell2_only)

    if args.output:
        os.makedirs(args.output, exist_ok=True)
        out = {
            'hierarchy': {
                'cell1_direction_nmi': hierarchy['cell1_direction_nmi'],
                'cell2_sqm': hierarchy['cell2_sqm'],
            },
            'cell1_only': cell1_only,
            'cell2_only': cell2_only,
        }
        with open(os.path.join(args.output, 'benchmark_hierarchy.json'), 'w') as f:
            json.dump(out, f, indent=2, default=lambda x: x.tolist()
                      if hasattr(x, 'tolist') else x)
        print(f"\nSaved to {args.output}/benchmark_hierarchy.json")


if __name__ == '__main__':
    main()

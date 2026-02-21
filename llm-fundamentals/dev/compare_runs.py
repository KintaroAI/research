#!/usr/bin/env python3
"""
Compare training runs from log files produced by train_gpt2_fp32.

Usage:
    python compare_runs.py log1.txt log2.txt [--thresholds 4.0,3.0,2.5]

Parses log lines:
    s:STEP trl:LOSS          — per-step train loss
    s:STEP tel:LOSS          — val loss
    s:STEP evl:LOSS avt:LOSS gap:LOSS  — eval summary (val, avg_train, gap)
    s:STEP tsl:LOSS          — test loss

Outputs:
    1. Summary table (best val, final val, final test, final gap)
    2. Matched-step val comparison
    3. Matched-train-loss comparison (isolates generalization from speed)
    4. Gap trajectory
"""

import sys
import os
import re


def parse_log(filepath):
    """Parse a log file into structured data."""
    train = []   # [(step, loss), ...]
    val = []     # [(step, loss), ...]
    evals = []   # [(step, val_loss, avg_train, gap), ...]
    test = []    # [(step, loss), ...]

    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            # train loss: s:STEP trl:LOSS
            m = re.match(r's:(\d+)\s+trl:([\d.]+)', line)
            if m:
                train.append((int(m.group(1)), float(m.group(2))))
                continue

            # eval summary: s:STEP evl:LOSS avt:LOSS gap:LOSS
            m = re.match(r's:(\d+)\s+evl:([\d.]+)\s+avt:([\d.]+)\s+gap:([+-]?[\d.]+)', line)
            if m:
                evals.append((int(m.group(1)), float(m.group(2)),
                              float(m.group(3)), float(m.group(4))))
                continue

            # val loss: s:STEP tel:LOSS
            m = re.match(r's:(\d+)\s+tel:([\d.]+)', line)
            if m:
                val.append((int(m.group(1)), float(m.group(2))))
                continue

            # test loss: s:STEP tsl:LOSS
            m = re.match(r's:(\d+)\s+tsl:([\d.]+)', line)
            if m:
                test.append((int(m.group(1)), float(m.group(2))))
                continue

    return {'train': train, 'val': val, 'evals': evals, 'test': test,
            'name': os.path.basename(filepath)}


def find_step_at_train_loss(train_data, threshold):
    """Find the first step where cumulative avg train loss <= threshold."""
    # Use a rolling window matching the eval intervals
    for step, loss in train_data:
        if loss <= threshold:
            return step
    return None


def interpolate_val_at_step(val_data, target_step):
    """Get val loss at or nearest to target_step."""
    if not val_data:
        return None
    best = None
    best_dist = float('inf')
    for step, loss in val_data:
        dist = abs(step - target_step)
        if dist < best_dist:
            best_dist = dist
            best = loss
    return best


def auto_thresholds(runs):
    """Pick thresholds from the range of train losses across runs."""
    all_losses = []
    for r in runs:
        for _, loss in r['train']:
            all_losses.append(loss)
    if not all_losses:
        return []
    lo = min(all_losses)
    hi = min(all_losses[:min(50, len(all_losses))])  # early losses
    # pick 3-4 thresholds spanning the range
    if hi <= lo:
        return []
    step = (hi - lo) / 4
    thresholds = []
    for i in range(1, 4):
        t = round(hi - i * step, 2)
        if t > lo:
            thresholds.append(t)
    return thresholds


def print_summary(runs):
    """Print summary table for each run."""
    print("=" * 78)
    print("SUMMARY")
    print("=" * 78)
    header = f"{'Run':<30} {'Best Val':>10} {'@ Step':>8} {'Final Val':>10} {'Final Test':>11} {'Final Gap':>10}"
    print(header)
    print("-" * 78)

    for r in runs:
        best_val = min(r['val'], key=lambda x: x[1]) if r['val'] else (0, float('nan'))
        final_val = r['val'][-1] if r['val'] else (0, float('nan'))
        final_test = r['test'][-1][1] if r['test'] else float('nan')
        final_gap = r['evals'][-1][3] if r['evals'] else float('nan')

        print(f"{r['name']:<30} {best_val[1]:>10.4f} {best_val[0]:>8d} "
              f"{final_val[1]:>10.4f} {final_test:>11.4f} {final_gap:>+10.4f}")
    print()


def print_matched_step(runs):
    """Print val losses at shared evaluation steps."""
    # Collect all val steps across runs
    all_steps = set()
    for r in runs:
        for step, _ in r['val']:
            all_steps.add(step)

    # Only show steps present in ALL runs
    common_steps = sorted(all_steps)
    step_sets = [set(s for s, _ in r['val']) for r in runs]
    common_steps = [s for s in common_steps if all(s in ss for ss in step_sets)]

    if not common_steps:
        return

    print("=" * 78)
    print("MATCHED-STEP VAL COMPARISON")
    print("=" * 78)

    # Header
    names = [r['name'][:20] for r in runs]
    header = f"{'Step':>8}"
    for n in names:
        header += f" {n:>20}"
    print(header)
    print("-" * (8 + 21 * len(runs)))

    # Build step->loss maps
    val_maps = [{s: l for s, l in r['val']} for r in runs]

    for step in common_steps:
        row = f"{step:>8}"
        for vm in val_maps:
            row += f" {vm[step]:>20.4f}"
        print(row)
    print()


def print_matched_train_loss(runs, thresholds):
    """At matched train loss thresholds, compare val losses."""
    if not thresholds:
        thresholds = auto_thresholds(runs)
    if not thresholds:
        print("(No matched-train-loss comparison: could not determine thresholds)\n")
        return

    print("=" * 78)
    print("MATCHED-TRAIN-LOSS VAL COMPARISON")
    print("  (val loss when each run first reaches a given train loss threshold)")
    print("=" * 78)

    names = [r['name'][:20] for r in runs]
    header = f"{'Threshold':>10}"
    for n in names:
        header += f" {'step':>8} {'val':>10}"
    print(header)
    print("-" * (10 + 19 * len(runs)))

    for thresh in sorted(thresholds, reverse=True):
        row = f"{thresh:>10.2f}"
        for r in runs:
            step = find_step_at_train_loss(r['train'], thresh)
            if step is not None:
                val = interpolate_val_at_step(r['val'], step)
                val_str = f"{val:.4f}" if val is not None else "N/A"
                row += f" {step:>8d} {val_str:>10}"
            else:
                row += f" {'N/A':>8} {'N/A':>10}"
        print(row)
    print()


def print_gap_trajectory(runs):
    """Print gap (val - avg_train) at each eval step."""
    has_evals = any(r['evals'] for r in runs)
    if not has_evals:
        return

    print("=" * 78)
    print("GAP TRAJECTORY (val - avg_train)")
    print("=" * 78)

    names = [r['name'][:20] for r in runs]
    header = f"{'Step':>8}"
    for n in names:
        header += f" {n:>20}"
    print(header)
    print("-" * (8 + 21 * len(runs)))

    # Collect all eval steps
    all_steps = set()
    for r in runs:
        for e in r['evals']:
            all_steps.add(e[0])

    eval_maps = [{e[0]: e[3] for e in r['evals']} for r in runs]

    for step in sorted(all_steps):
        row = f"{step:>8}"
        for em in eval_maps:
            if step in em:
                row += f" {em[step]:>+20.4f}"
            else:
                row += f" {'':>20}"
        print(row)
    print()


def main():
    args = sys.argv[1:]
    thresholds = None

    # Parse --thresholds flag
    log_files = []
    i = 0
    while i < len(args):
        if args[i] == '--thresholds' and i + 1 < len(args):
            thresholds = [float(x) for x in args[i + 1].split(',')]
            i += 2
        else:
            log_files.append(args[i])
            i += 1

    if len(log_files) < 1:
        print("Usage: python compare_runs.py log1.txt [log2.txt ...] [--thresholds 4.0,3.0,2.5]")
        sys.exit(1)

    runs = [parse_log(f) for f in log_files]

    print()
    print_summary(runs)
    print_matched_step(runs)
    print_matched_train_loss(runs, thresholds)
    print_gap_trajectory(runs)


if __name__ == '__main__':
    main()

"""Rotating line benchmark: test whether column outputs learn to represent
angle, rotation direction, speed, and pulsation from a single 7x7 patch.

Generates a line on a 7x7 grid that:
  - Rotates continuously (configurable period, direction)
  - Pulsates in thickness (configurable period)
  - Changes rotation speed over time (optional)

Feeds the patch into a single column and measures whether outputs
correlate with the underlying generative factors (angle, angular velocity,
thickness) without any supervision.

Usage:
    cd dev/
    python benchmarks/rotating_line.py -f 20000 --outputs 8 \
        --column-type temporal_prototype --rotation-period 200
    python benchmarks/rotating_line.py -f 20000 --outputs 8 \
        --column-type conscience --rotation-period 200
"""

import os
import sys
import json
import argparse
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from cluster_manager import create_column


def render_line(size, angle_rad, thickness=1.0, antialias=True):
    """Render a line through the center of a size x size grid.

    Args:
        size: grid dimension (e.g. 7)
        angle_rad: line angle in radians
        thickness: line half-width in pixels (1.0 = thin, 3.0 = fat)
        antialias: smooth edges

    Returns:
        (size, size) float32 array in [0, 1]
    """
    center = (size - 1) / 2.0
    y, x = np.mgrid[:size, :size]
    dx = x - center
    dy = y - center

    # Signed distance from each pixel to the line through center at angle
    # Line direction: (cos(a), sin(a)). Normal: (-sin(a), cos(a))
    nx = -np.sin(angle_rad)
    ny = np.cos(angle_rad)
    dist = np.abs(dx * nx + dy * ny)

    if antialias:
        # Smooth falloff: 1.0 at center, 0.0 at thickness boundary
        img = np.clip(1.0 - (dist - thickness + 0.5), 0.0, 1.0)
    else:
        img = (dist <= thickness).astype(np.float32)

    return img.astype(np.float32)


def generate_sequence(n_frames, size=7, rotation_period=200, pulse_period=80,
                      direction=1, speed_modulation=0.0, reversal_period=0,
                      rng_seed=42):
    """Generate a sequence of rotating+pulsating line frames.

    Args:
        n_frames: number of frames
        size: grid size
        rotation_period: ticks per full rotation
        pulse_period: ticks per thickness cycle
        direction: +1 = counterclockwise, -1 = clockwise (initial)
        speed_modulation: 0.0 = constant speed, >0 = sinusoidal speed variation
        reversal_period: if >0, reverse rotation direction every N ticks

    Returns:
        frames: (n_frames, size, size) float32
        factors: dict of (n_frames,) arrays with ground truth
    """
    rng = np.random.RandomState(rng_seed)

    frames = np.zeros((n_frames, size, size), dtype=np.float32)

    # Ground truth factors
    angles = np.zeros(n_frames, dtype=np.float32)
    angular_velocities = np.zeros(n_frames, dtype=np.float32)
    thicknesses = np.zeros(n_frames, dtype=np.float32)
    thickness_velocities = np.zeros(n_frames, dtype=np.float32)
    sin_angles = np.zeros(n_frames, dtype=np.float32)
    cos_angles = np.zeros(n_frames, dtype=np.float32)
    directions = np.zeros(n_frames, dtype=np.float32)

    base_speed = 2.0 * np.pi / rotation_period
    current_dir = direction

    angle = rng.uniform(0, 2 * np.pi)

    for t in range(n_frames):
        # Direction reversal
        if reversal_period > 0 and t > 0 and t % reversal_period == 0:
            current_dir *= -1

        # Angular velocity: base + optional sinusoidal modulation
        omega = current_dir * base_speed
        if speed_modulation > 0:
            omega *= (1.0 + speed_modulation * np.sin(2 * np.pi * t / (rotation_period * 3)))

        angle += omega
        angle = angle % (2 * np.pi)

        # Thickness pulsation: oscillates between 0.5 and 2.5
        thickness = 1.5 + 1.0 * np.sin(2 * np.pi * t / pulse_period)
        thickness_vel = (2 * np.pi / pulse_period) * np.cos(2 * np.pi * t / pulse_period)

        frames[t] = render_line(size, angle, thickness=max(0.3, thickness))

        angles[t] = angle
        angular_velocities[t] = omega
        thicknesses[t] = thickness
        thickness_velocities[t] = thickness_vel
        sin_angles[t] = np.sin(angle)
        cos_angles[t] = np.cos(angle)
        directions[t] = float(current_dir)

    factors = {
        'angle': angles,
        'sin_angle': sin_angles,
        'cos_angle': cos_angles,
        'direction': directions,
        'angular_velocity': angular_velocities,
        'thickness': thicknesses,
        'thickness_velocity': thickness_velocities,
    }

    return frames, factors


def run(args):
    size = args.patch_size
    n_in = size * size
    n_out = args.outputs
    n_frames = args.frames
    window = args.window

    print(f"Rotating line benchmark: {size}x{size} patch, {n_out} outputs, "
          f"{n_frames} frames")
    print(f"  rotation_period={args.rotation_period}, pulse_period={args.pulse_period}, "
          f"direction={args.direction}, speed_mod={args.speed_modulation}")

    # Generate full sequence
    frames, factors = generate_sequence(
        n_frames + 1000,  # extra for eval
        size=size,
        rotation_period=args.rotation_period,
        pulse_period=args.pulse_period,
        direction=args.direction,
        speed_modulation=args.speed_modulation,
        reversal_period=args.reversal_period,
    )

    # Create column
    col_config = {
        'm': 1,
        'n_outputs': n_out,
        'max_inputs': n_in,
        'window': window,
        'temperature': args.temperature,
        'lr': args.lr,
        'alpha': args.alpha,
        'k_active': args.k_active,
        'homeostasis_rate': args.homeostasis_rate,
        'fatigue_strength': args.fatigue_strength,
        'lr_neg': args.lr_neg,
        'margin_band': args.margin_band,
        'multi_scale': getattr(args, 'multi_scale', False),
        'mode': 'kmeans',
        'entropy_scaled_lr': True,
    }
    cm = create_column(args.column_type, col_config)

    # Wire all inputs to the single column
    for i in range(n_in):
        cm.slot_map[0, i] = i

    # Ring buffer
    ring = np.zeros((n_in, window), dtype=np.float32)

    # --- Training ---
    log_every = max(n_frames // 10, 1)
    print(f"Column type: {args.column_type}, lr={args.lr}, temp={args.temperature}")

    for t in range(n_frames):
        frame = frames[t].ravel()
        ring[:, :-1] = ring[:, 1:]
        ring[:, -1] = frame
        cm.tick(ring)
        if (t + 1) % log_every == 0:
            print(f"  tick {t+1}/{n_frames}")

    # --- Eval: freeze and collect outputs ---
    n_eval = 1000
    saved_lr = cm.lr
    cm.lr = 0.0
    if hasattr(cm, 'set_learn_prob'):
        cm.set_learn_prob(0.0)

    output_history = np.zeros((n_eval, n_out), dtype=np.float32)
    eval_factors = {k: v[n_frames:n_frames+n_eval] for k, v in factors.items()}

    for t in range(n_eval):
        frame = frames[n_frames + t].ravel()
        ring[:, :-1] = ring[:, 1:]
        ring[:, -1] = frame
        cm.tick(ring)
        output_history[t] = cm.get_outputs()[0]  # single column, take row 0

    cm.lr = saved_lr
    if hasattr(cm, 'set_learn_prob'):
        cm.set_learn_prob(1.0)

    # --- Analysis: correlate outputs with ground truth factors ---
    print(f"\nResults after {n_frames} training ticks, {n_eval} eval ticks:")

    # Per-output max |correlation| with each factor
    factor_names = list(eval_factors.keys())
    corr_matrix = np.zeros((n_out, len(factor_names)), dtype=np.float32)

    for fi, fname in enumerate(factor_names):
        fvals = eval_factors[fname]
        for oi in range(n_out):
            ovals = output_history[:, oi]
            if ovals.std() < 1e-8 or fvals.std() < 1e-8:
                corr_matrix[oi, fi] = 0.0
            else:
                r = np.corrcoef(ovals, fvals)[0, 1]
                corr_matrix[oi, fi] = r if not np.isnan(r) else 0.0

    # Print correlation matrix
    header = f"{'output':>8s}" + "".join(f"{fn:>16s}" for fn in factor_names)
    print(f"\n  Correlation matrix (output × factor):")
    print(f"  {header}")
    for oi in range(n_out):
        row = f"  {'out_'+str(oi):>8s}"
        for fi in range(len(factor_names)):
            r = corr_matrix[oi, fi]
            marker = " *" if abs(r) > 0.5 else "  " if abs(r) > 0.3 else ""
            row += f"{r:>14.3f}{marker}"
        print(row)

    # Best factor per output
    print(f"\n  Best factor per output:")
    for oi in range(n_out):
        best_fi = np.argmax(np.abs(corr_matrix[oi]))
        best_r = corr_matrix[oi, best_fi]
        print(f"    output {oi}: {factor_names[best_fi]:>20s}  r={best_r:+.3f}")

    # Best output per factor
    print(f"\n  Best output per factor:")
    for fi, fname in enumerate(factor_names):
        best_oi = np.argmax(np.abs(corr_matrix[:, fi]))
        best_r = corr_matrix[best_oi, fi]
        print(f"    {fname:>20s}: output {best_oi}  r={best_r:+.3f}")

    # Factor coverage: how many factors have |r| > 0.3 with at least one output?
    covered = sum(1 for fi in range(len(factor_names))
                  if np.abs(corr_matrix[:, fi]).max() > 0.3)
    print(f"\n  Factor coverage (single output |r|>0.3): {covered}/{len(factor_names)}")

    # Multi-output R²: how well do ALL outputs jointly predict each factor?
    # This removes the statistical ceiling from sparse single-output correlations.
    print(f"\n  Multi-output R² (all outputs → each factor via linear regression):")
    r2_scores = {}
    for fi, fname in enumerate(factor_names):
        fvals = eval_factors[fname]
        if fvals.std() < 1e-8:
            r2_scores[fname] = 0.0
            continue
        # OLS: R² = 1 - SS_res / SS_tot
        X = output_history  # (n_eval, n_out)
        y = fvals
        # Add bias column
        X_b = np.column_stack([X, np.ones(len(y))])
        # Solve least squares
        beta, _, _, _ = np.linalg.lstsq(X_b, y, rcond=None)
        y_pred = X_b @ beta
        ss_res = ((y - y_pred) ** 2).sum()
        ss_tot = ((y - y.mean()) ** 2).sum()
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
        r2_scores[fname] = float(r2)
        marker = " **" if r2 > 0.5 else " *" if r2 > 0.3 else ""
        print(f"    {fname:>20s}: R²={r2:.3f}{marker}")

    r2_covered = sum(1 for v in r2_scores.values() if v > 0.3)
    print(f"  Factor coverage (multi-output R²>0.3): {r2_covered}/{len(factor_names)}")

    # Output entropy
    eps = 1e-10
    H = -(output_history * np.log(output_history + eps)).sum(axis=1)
    H_max = np.log(n_out)
    print(f"  Output entropy: {H.mean():.3f}/{H_max:.3f} "
          f"(normalized: {H.mean()/H_max:.3f})")

    # Inter-output correlation
    pair_corrs = []
    for i in range(n_out):
        for j in range(i + 1, n_out):
            si, sj = output_history[:, i], output_history[:, j]
            if si.std() < 1e-8 or sj.std() < 1e-8:
                pair_corrs.append(1.0)
            else:
                r = np.corrcoef(si, sj)[0, 1]
                pair_corrs.append(float(r) if not np.isnan(r) else 0.0)
    print(f"  Inter-output correlation: {np.mean(pair_corrs):.3f} "
          f"[{np.min(pair_corrs):.3f}, {np.max(pair_corrs):.3f}]")

    # Winner distribution
    winners = output_history.argmax(axis=1)
    win_counts = np.bincount(winners, minlength=n_out)
    win_frac = win_counts / n_eval
    print(f"  Winner distribution: {' / '.join(f'{f:.1%}' for f in win_frac)}")

    # --- Save ---
    out_dir = args.output
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        results = {
            'frames': n_frames,
            'eval_frames': n_eval,
            'column_type': args.column_type,
            'n_outputs': n_out,
            'patch_size': size,
            'rotation_period': args.rotation_period,
            'pulse_period': args.pulse_period,
            'direction': args.direction,
            'speed_modulation': args.speed_modulation,
            'correlation_matrix': corr_matrix.tolist(),
            'factor_names': factor_names,
            'factor_coverage_single': covered,
            'factor_coverage_r2': r2_covered,
            'r2_scores': {k: round(v, 4) for k, v in r2_scores.items()},
            'entropy': {
                'mean': round(float(H.mean()), 4),
                'normalized': round(float(H.mean() / H_max), 4),
            },
            'inter_output_correlation': round(float(np.mean(pair_corrs)), 4),
            'winner_distribution': [round(float(f), 4) for f in win_frac],
            'best_per_factor': {
                fname: {
                    'output': int(np.argmax(np.abs(corr_matrix[:, fi]))),
                    'r': round(float(corr_matrix[np.argmax(np.abs(corr_matrix[:, fi])), fi]), 4),
                }
                for fi, fname in enumerate(factor_names)
            },
        }
        json_path = os.path.join(out_dir, "rotating_line_analysis.json")
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n  Analysis saved: {json_path}")

        # Save output timeseries + factors for plotting
        np.savez(os.path.join(out_dir, "rotating_line_data.npz"),
                 outputs=output_history,
                 **eval_factors)

        # Visualize: output traces overlaid with factors
        try:
            import cv2
            h_img = 80 * n_out + 80 * len(factor_names) + 40
            w_img = n_eval + 100
            canvas = np.full((h_img, w_img, 3), 255, dtype=np.uint8)

            colors = [(0,0,200), (0,180,0), (200,0,0), (180,0,180),
                      (0,180,180), (180,180,0), (100,100,100), (0,100,200)]

            # Output traces
            y_off = 20
            for oi in range(n_out):
                vals = output_history[:, oi]
                vals_norm = (vals - vals.min()) / (vals.max() - vals.min() + 1e-8)
                for t in range(1, n_eval):
                    y0 = y_off + int((1 - vals_norm[t-1]) * 60)
                    y1 = y_off + int((1 - vals_norm[t]) * 60)
                    cv2.line(canvas, (50+t-1, y0), (50+t, y1),
                             colors[oi % len(colors)], 1)
                cv2.putText(canvas, f"o{oi}", (5, y_off+35),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35,
                            colors[oi % len(colors)], 1)
                y_off += 80

            # Factor traces
            y_off += 20
            for fi, fname in enumerate(factor_names):
                vals = eval_factors[fname]
                vals_norm = (vals - vals.min()) / (vals.max() - vals.min() + 1e-8)
                for t in range(1, n_eval):
                    y0 = y_off + int((1 - vals_norm[t-1]) * 60)
                    y1 = y_off + int((1 - vals_norm[t]) * 60)
                    cv2.line(canvas, (50+t-1, y0), (50+t, y1), (0,0,0), 1)
                label = fname[:12]
                cv2.putText(canvas, label, (5, y_off+35),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0,0,0), 1)
                y_off += 80

            path = os.path.join(out_dir, "rotating_line_traces.png")
            cv2.imwrite(path, canvas)
            print(f"  Traces saved: {path}")
        except ImportError:
            pass


def main():
    parser = argparse.ArgumentParser(description='Rotating line benchmark')
    parser.add_argument('--patch-size', type=int, default=7)
    parser.add_argument('--outputs', type=int, default=8)
    parser.add_argument('--column-type', type=str, default='temporal_prototype')
    parser.add_argument('--lr', type=float, default=0.05)
    parser.add_argument('--alpha', type=float, default=0.01)
    parser.add_argument('--temperature', type=float, default=0.45)
    parser.add_argument('--k-active', type=int, default=1)
    parser.add_argument('--homeostasis-rate', type=float, default=0.5)
    parser.add_argument('--fatigue-strength', type=float, default=0.001)
    parser.add_argument('--lr-neg', type=float, default=0.01)
    parser.add_argument('--margin-band', type=float, default=0.3)
    parser.add_argument('--multi-scale', action='store_true',
                        help='Use multi-scale descriptor (4 parts: current, delta_1, mean_full, delta_half)')
    parser.add_argument('--window', type=int, default=10)
    parser.add_argument('--rotation-period', type=int, default=200,
                        help='Ticks per full rotation (default: 200)')
    parser.add_argument('--pulse-period', type=int, default=80,
                        help='Ticks per thickness cycle (default: 80)')
    parser.add_argument('--direction', type=int, default=1, choices=[-1, 1],
                        help='Rotation direction: 1=CCW, -1=CW')
    parser.add_argument('--speed-modulation', type=float, default=0.0,
                        help='Sinusoidal speed variation amplitude (0=constant)')
    parser.add_argument('--reversal-period', type=int, default=0,
                        help='Reverse rotation direction every N ticks (0=never)')
    parser.add_argument('-f', '--frames', type=int, default=20000)
    parser.add_argument('-o', '--output', type=str, default=None)
    args = parser.parse_args()
    run(args)


if __name__ == '__main__':
    main()

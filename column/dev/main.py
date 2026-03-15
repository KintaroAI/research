"""Train a soft-WTA column cell on synthetic clustered data."""

import argparse
import json
import os

import numpy as np
import torch

from column import SoftWTACell
from metrics import compute_sqm, format_sqm


def make_clustered_data(n_clusters, n_inputs, n_samples, seed=42):
    """Generate Gaussian clusters on the unit sphere."""
    rng = np.random.default_rng(seed)
    centers = rng.standard_normal((n_clusters, n_inputs))
    centers = centers / np.linalg.norm(centers, axis=1, keepdims=True)

    data = []
    labels = []
    for i in range(n_samples):
        c = rng.integers(n_clusters)
        x = centers[c] + rng.standard_normal(n_inputs) * 0.1
        data.append(x)
        labels.append(c)

    return np.array(data, dtype=np.float32), np.array(labels), centers


def run(args):
    os.makedirs(args.output, exist_ok=True)

    n_clusters = args.n_clusters or args.n_outputs
    data, labels, centers = make_clustered_data(
        n_clusters, args.n_inputs, args.frames, seed=args.seed
    )

    cell = SoftWTACell(
        n_inputs=args.n_inputs,
        n_outputs=args.n_outputs,
        temperature=args.temperature,
        lr=args.lr,
        match_threshold=args.match_threshold,
        usage_decay=args.usage_decay,
    )

    all_winners = []
    all_probs = []
    match_qualities = []
    log_interval = max(1, args.frames // 20)
    sqm_interval = max(1, args.frames // 5)  # SQM at 20/40/60/80/100%
    sqm_snapshots = []

    for i in range(args.frames):
        x = torch.from_numpy(data[i])
        probs = cell.forward(x)
        winner, mq = cell.update(x, probs)
        all_winners.append(winner)
        all_probs.append(probs.detach().numpy())
        match_qualities.append(mq)

        if (i + 1) % log_interval == 0:
            recent_mq = np.mean(match_qualities[-log_interval:])
            unique_recent = len(set(all_winners[-log_interval:]))
            print(f"frame {i+1:>6d}/{args.frames}  "
                  f"avg_match={recent_mq:.4f}  "
                  f"unique_winners={unique_recent}/{args.n_outputs}  "
                  f"usage_std={cell.usage.std():.4f}")

        if (i + 1) % sqm_interval == 0:
            window = slice(max(0, i + 1 - sqm_interval), i + 1)
            w_arr = np.array(all_winners[window])
            p_arr = np.array(all_probs[window])
            sqm = compute_sqm(w_arr, p_arr, cell.prototypes.numpy(),
                              args.n_outputs, labels=labels[window])
            sqm['frame'] = i + 1
            sqm_snapshots.append(sqm)
            print(f"  SQM: {format_sqm(sqm)}")

    # Final SQM over all frames
    winners = np.array(all_winners)
    probs_arr = np.array(all_probs)
    final_sqm = compute_sqm(winners, probs_arr, cell.prototypes.numpy(),
                            args.n_outputs, labels=labels)

    results = {
        'args': vars(args),
        'sqm': final_sqm,
        'sqm_snapshots': sqm_snapshots,
        'final_avg_match': float(np.mean(match_qualities[-log_interval:])),
        'usage': cell.usage.tolist(),
    }

    with open(os.path.join(args.output, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    torch.save(cell.state_dict(), os.path.join(args.output, 'cell.pt'))

    print(f"\nFinal SQM: {format_sqm(final_sqm)}")
    print(f"Saved to {args.output}")
    return results


def main():
    parser = argparse.ArgumentParser(description='Train a soft-WTA column cell')
    parser.add_argument('--n-inputs', type=int, default=16)
    parser.add_argument('--n-outputs', type=int, default=8)
    parser.add_argument('--n-clusters', type=int, default=None,
                        help='Number of input clusters (default: n_outputs)')
    parser.add_argument('--frames', type=int, default=10000)
    parser.add_argument('--temperature', type=float, default=0.5)
    parser.add_argument('--lr', type=float, default=0.05)
    parser.add_argument('--match-threshold', type=float, default=0.5)
    parser.add_argument('--usage-decay', type=float, default=0.99)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('-o', '--output', type=str, required=True)
    args = parser.parse_args()
    run(args)


if __name__ == '__main__':
    main()

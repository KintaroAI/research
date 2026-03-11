"""Run 3D reconstruction on GPU using DriftSolver (PyTorch).

Usage:
    python run_3d_gpu.py tree_80.npy -f 5000 --save-every 500 -o output_6_test3
"""

import numpy as np
import argparse
import os
import time
from scipy.spatial import cKDTree


def compute_topk_3d(size, k=24):
    """Compute top-K neighbors for a 3D grid."""
    n = size ** 3
    coords = np.zeros((n, 3), dtype=np.float32)
    idx = np.arange(n)
    coords[:, 0] = idx % size
    coords[:, 1] = (idx // size) % size
    coords[:, 2] = idx // (size * size)

    print(f"Building KDTree for {n} neurons...")
    tree = cKDTree(coords)
    _, top_k = tree.query(coords, k=k + 1)
    top_k = top_k[:, 1:].astype(np.int32)
    return coords, top_k


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("volume", help="Path to .npy volume file")
    parser.add_argument("-f", "--frames", type=int, default=5000)
    parser.add_argument("--save-every", type=int, default=500)
    parser.add_argument("-o", "--output", default="output_3d_gpu")
    parser.add_argument("--k", type=int, default=24)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--margin", type=float, default=0.1)
    parser.add_argument("--mode", default="euclidean", choices=["euclidean", "dot"])
    parser.add_argument("--dims", type=int, default=3)
    parser.add_argument("-s", "--subsample-render", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    # Load volume — scalar (W,H,D) or RGB (W,H,D,3)
    vol = np.load(args.volume)
    is_rgb = vol.ndim == 4 and vol.shape[3] == 3
    size = vol.shape[0]
    if is_rgb:
        assert vol.shape[:3] == (size, size, size), "RGB volume must be cubic"
    else:
        assert vol.shape == (size, size, size), "Volume must be cubic"
    n = size ** 3
    print(f"Volume: {size}x{size}x{size} = {n} neurons, "
          f"{'RGB' if is_rgb else 'scalar'}, device={args.device}")

    # Flatten voxel values — for RGB, flatten to (n, 3)
    if is_rgb:
        voxel_values = vol.reshape(n, 3)
    else:
        voxel_values = vol.ravel().astype(np.float32)

    # Compute 3D top-K
    coords, top_k = compute_topk_3d(size, k=args.k)

    # Create GPU solver
    from solvers.drift_torch import DriftSolver
    import torch

    solver = DriftSolver(n, top_k=top_k, dims=args.dims, lr=args.lr,
                         mode=args.mode, margin=args.margin, device=args.device)

    from view_3d import render_volume

    frame_idx = 0
    t_start = time.time()

    for t in range(1, args.frames + 1):
        solver.tick()

        if t % args.save_every == 0 or t == args.frames:
            torch.cuda.synchronize()
            elapsed = time.time() - t_start

            # Render — for RGB, reconstruct each channel via Voronoi
            if is_rgb:
                recon = solver.render((size, size, size), voxel_values=voxel_values)
            else:
                recon = solver.render((size, size, size), voxel_values=voxel_values)
            path = os.path.join(args.output, f"frame_{frame_idx:04d}.png")
            render_volume(recon, path,
                          title=f"Reconstruction (tick {t})",
                          threshold=0.01, subsample=args.subsample_render,
                          elev=25, azim=45)
            frame_idx += 1

            s = solver.stats()
            print(f"  tick {t}, frame {frame_idx}, std={s['std']:.4f}, "
                  f"elapsed={elapsed:.1f}s")

    elapsed = time.time() - t_start
    print(f"Done: {args.frames} ticks, {frame_idx} frames in {elapsed:.1f}s")


if __name__ == "__main__":
    main()

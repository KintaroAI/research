"""Run ContinuousDrift solver on a 3D volume and render reconstruction.

Usage:
    python run_3d.py tree_80.npy -f 5000 --save-every 500 -o output_6_test1
"""

import numpy as np
import argparse
import os
from scipy.spatial import cKDTree


def compute_topk_3d(size, k=24):
    """Compute top-K neighbors for a 3D grid using 1/r decay."""
    n = size ** 3
    coords = np.zeros((n, 3), dtype=np.float32)
    idx = np.arange(n)
    coords[:, 0] = idx % size
    coords[:, 1] = (idx // size) % size
    coords[:, 2] = idx // (size * size)

    print(f"Building KDTree for {n} neurons...")
    tree = cKDTree(coords)
    _, top_k = tree.query(coords, k=k + 1)
    top_k = top_k[:, 1:].astype(np.int32)  # exclude self
    print(f"Top-K computed: {top_k.shape}")
    return coords, top_k


class Solver3D:
    """ContinuousDrift adapted for 3D volumes."""

    def __init__(self, size, top_k, k=24, lr=0.05, margin=0.1, voxel_values=None):
        self.size = size
        self.n = size ** 3
        self.k = k
        self.lr = lr
        self.margin = margin
        self.top_k = top_k

        # Random initial positions in 3D
        self.positions = np.random.normal(0, 1, (self.n, 3)).astype(np.float32)

        # Voxel values for reconstruction
        self.voxel_values = voxel_values

    def tick(self):
        """One iteration: lerp toward neighbor centroid with dead zone + LayerNorm."""
        neighbor_pos = self.positions[self.top_k]  # (n, k, 3)
        centroids = np.mean(neighbor_pos, axis=1)  # (n, 3)
        delta = centroids - self.positions
        if self.margin > 0:
            dist = np.sqrt(np.sum(delta ** 2, axis=1, keepdims=True))
            scale = np.tanh(dist / self.margin)
            delta = delta * scale
        self.positions += self.lr * delta
        self._normalize()

    def _normalize(self):
        """LayerNorm: center + unit variance per dimension."""
        self.positions -= np.mean(self.positions, axis=0)
        std = np.std(self.positions, axis=0)
        std = np.where(std < 1e-8, 1.0, std)
        self.positions /= std

    def render(self):
        """Reconstruct 3D volume via Voronoi: each grid cell gets nearest neuron's value."""
        pos = self.positions.copy()

        # Map positions to grid coordinates
        for d in range(3):
            mn, mx = pos[:, d].min(), pos[:, d].max()
            span = mx - mn if mx - mn > 1e-8 else 1.0
            pos[:, d] = (pos[:, d] - mn) / span * (self.size - 1)

        # Grid cell centers
        idx = np.arange(self.n)
        grid = np.zeros((self.n, 3), dtype=np.float64)
        grid[:, 0] = idx % self.size
        grid[:, 1] = (idx // self.size) % self.size
        grid[:, 2] = idx // (self.size * self.size)

        # Nearest neuron for each grid cell
        tree = cKDTree(pos.astype(np.float64))
        _, nearest = tree.query(grid)

        if self.voxel_values is not None:
            vol = self.voxel_values[nearest].reshape(
                self.size, self.size, self.size)
            return vol
        return nearest.reshape(self.size, self.size, self.size)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("volume", help="Path to .npy volume file")
    parser.add_argument("-f", "--frames", type=int, default=5000)
    parser.add_argument("--save-every", type=int, default=500)
    parser.add_argument("-o", "--output", default="output_3d")
    parser.add_argument("--k", type=int, default=24)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--margin", type=float, default=0.1)
    parser.add_argument("-s", "--subsample-render", type=int, default=2,
                        help="Subsample for 3D visualization")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    # Load volume
    vol = np.load(args.volume)
    size = vol.shape[0]
    assert vol.shape == (size, size, size), "Volume must be cubic"
    n = size ** 3
    print(f"Volume: {size}x{size}x{size} = {n} neurons")

    # Flatten voxel values
    voxel_values = vol.ravel().astype(np.float32)

    # Compute 3D top-K
    coords, top_k = compute_topk_3d(size, k=args.k)

    # Create solver
    solver = Solver3D(size, top_k, k=args.k, lr=args.lr, margin=args.margin,
                      voxel_values=voxel_values)

    # Import renderer
    from view_3d import render_volume

    frame_idx = 0
    save_interval = max(1, args.frames // (args.frames // args.save_every))

    for t in range(1, args.frames + 1):
        solver.tick()

        if t % args.save_every == 0 or t == args.frames:
            recon = solver.render()
            path = os.path.join(args.output, f"frame_{frame_idx:04d}.png")
            render_volume(recon, path,
                          title=f"Reconstruction (tick {t})",
                          threshold=0.01, subsample=args.subsample_render,
                          elev=25, azim=45)
            frame_idx += 1

            stats = solver.positions
            std = np.std(stats)
            print(f"  tick {t}, frame {frame_idx}, std={std:.4f}")

    print(f"Done: {args.frames} ticks, {frame_idx} frames saved to {args.output}/")


if __name__ == "__main__":
    main()

"""Visualize a 3D volume as a scatter plot of spheres.
Renders from multiple angles and saves as PNG images.

Usage:
    python view_3d.py tree_80.npy                    # view original volume
    python view_3d.py tree_80.npy -o tree_view.png   # save specific output
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import argparse


def render_volume(vol, output_path, title="3D Volume", threshold=0.01,
                  subsample=1, elev=25, azim=45):
    """Render 3D volume as scatter plot.

    Args:
        vol: scalar (W,H,D) with values 0-1, or RGB (W,H,D,3) uint8
        output_path: where to save the PNG
        threshold: minimum value to display (scalar) or min channel sum (RGB)
        subsample: take every Nth point (for speed with large volumes)
        elev, azim: camera angles
    """
    is_rgb = vol.ndim == 4 and vol.shape[3] == 3

    if is_rgb:
        # Threshold on sum of channels (0 = black = empty)
        brightness = vol.astype(np.float32).sum(axis=3)
        mask = brightness > threshold
    else:
        mask = vol > threshold

    if subsample > 1:
        s = subsample
        sub_mask = np.zeros_like(mask)
        sub_mask[::s, ::s, ::s] = mask[::s, ::s, ::s]
        mask = sub_mask

    xs, ys, zs = np.where(mask)

    if len(xs) == 0:
        print("No voxels above threshold!")
        return

    print(f"Rendering {len(xs)} voxels...")

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    if is_rgb:
        # Direct RGB colors from volume
        colors = vol[xs, ys, zs].astype(np.float32) / 255.0
        # Add alpha channel
        alpha = np.full((len(xs), 1), 0.7, dtype=np.float32)
        colors = np.hstack([colors, alpha])
        sizes = np.full(len(xs), 6.0)
    else:
        vals = vol[mask]
        colors = np.zeros((len(vals), 4))
        for i, v in enumerate(vals):
            if v < 0.5:  # trunk
                colors[i] = [0.45, 0.25, 0.1, 0.8]
            elif v >= 0.95:  # star
                colors[i] = [1.0, 0.9, 0.0, 1.0]
            else:  # canopy
                g = 0.3 + 0.5 * v
                colors[i] = [0.1, g, 0.05, 0.6]
        sizes = vals * 8

    # Swap Y↔Z so Y (up in volume) maps to matplotlib's vertical Z axis
    ax.scatter(xs, zs, ys, c=colors, s=sizes, depthshade=True)

    ax.set_xlabel('X')
    ax.set_ylabel('Z')
    ax.set_zlabel('Y (up)')
    ax.set_title(title)
    ax.view_init(elev=elev, azim=azim)

    # Equal aspect ratio
    max_range = max(vol.shape[:3])
    ax.set_xlim(0, max_range)
    ax.set_ylim(0, max_range)
    ax.set_zlim(0, max_range)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def render_multi_angle(vol, prefix, title="3D Volume", threshold=0.01,
                       subsample=1):
    """Render from 3 angles: front, side, top-down."""
    angles = [
        (25, 45, "perspective"),
        (0, 0, "front"),
        (0, 90, "side"),
        (90, 0, "top"),
    ]
    for elev, azim, name in angles:
        path = f"{prefix}_{name}.png"
        render_volume(vol, path, title=f"{title} ({name})",
                      threshold=threshold, subsample=subsample,
                      elev=elev, azim=azim)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("volume", help="Path to .npy volume file")
    parser.add_argument("-o", "--output", default=None, help="Output PNG path")
    parser.add_argument("-t", "--threshold", type=float, default=0.01)
    parser.add_argument("-s", "--subsample", type=int, default=1)
    parser.add_argument("--multi", action="store_true", help="Render multiple angles")
    parser.add_argument("--title", default="3D Volume")
    args = parser.parse_args()

    vol = np.load(args.volume)
    print(f"Volume shape: {vol.shape}, filled: {(vol > args.threshold).sum()}")

    if args.multi:
        prefix = args.output or args.volume.replace(".npy", "")
        render_multi_angle(vol, prefix, title=args.title,
                           threshold=args.threshold, subsample=args.subsample)
    else:
        output = args.output or args.volume.replace(".npy", ".png")
        render_volume(vol, output, title=args.title,
                      threshold=args.threshold, subsample=args.subsample)

"""Analyze channel structure of multi-channel embeddings.

For RGBG mode (--signal-channels 4), each pixel has 4 neurons (R,G,B,GS).
The .ravel() order is row-major on (h, w, C), so neuron i corresponds to:
  pixel_idx = i // C
  channel   = i % C
  px_x      = pixel_idx % crop_w
  px_y      = pixel_idx // crop_w

This script analyzes K=10 embedding neighbors to see whether spatial proximity
or channel identity dominates the learned structure.

Usage:
    python analyze_channels.py output_13_rgbg_50k/model.npy -W 80 -H 80 -C 4
"""

import argparse
import numpy as np
from scipy.spatial import cKDTree


def analyze(emb_path, crop_w, crop_h, num_channels):
    emb = np.load(emb_path)
    n = emb.shape[0]
    n_pixels = crop_w * crop_h
    assert n == n_pixels * num_channels, \
        f"n={n} != {crop_w}*{crop_h}*{num_channels}={n_pixels*num_channels}"

    pixel_idx = np.arange(n) // num_channels
    channel = np.arange(n) % num_channels
    px_x = pixel_idx % crop_w
    px_y = pixel_idx // crop_w

    ch_names = {0: 'R', 1: 'G', 2: 'B', 3: 'GS'}

    print(f"Loaded {emb_path}: {n} neurons, {emb.shape[1]} dims")
    print(f"  {crop_w}x{crop_h} pixels, {num_channels} channels "
          f"({', '.join(ch_names.get(i, str(i)) for i in range(num_channels))})")

    # K=10 neighbors
    tree = cKDTree(emb)
    _, idx = tree.query(emb, k=11)
    idx = idx[:, 1:]  # remove self

    # Classify each neighbor
    same_pixel = (pixel_idx[idx] == pixel_idx[:, None])
    same_channel = (channel[idx] == channel[:, None])
    pixel_dist = np.abs(px_x[idx] - px_x[:, None]) + np.abs(px_y[idx] - px_y[:, None])

    # Overall stats
    print(f"\n=== K=10 neighbor composition (all neurons) ===")
    print(f"  Same pixel (diff channel):    {same_pixel.mean()*100:.1f}%")
    print(f"  Same channel:                 {same_channel.mean()*100:.1f}%")
    print(f"  Same pixel AND same channel:  impossible (self excluded)")
    sp_diffch = same_pixel & ~same_channel
    print(f"  Same pixel, diff channel:     {sp_diffch.mean()*100:.1f}%")
    dp_samech = ~same_pixel & same_channel
    print(f"  Diff pixel, same channel:     {dp_samech.mean()*100:.1f}%")
    dp_diffch = ~same_pixel & ~same_channel
    print(f"  Diff pixel, diff channel:     {dp_diffch.mean()*100:.1f}%")

    print(f"\n  Mean pixel distance of K=10 neighbors: {pixel_dist.mean():.2f}")
    print(f"  Mean pixel distance (same channel):     "
          f"{pixel_dist[same_channel].mean():.2f}" if same_channel.any() else "  N/A")
    print(f"  Mean pixel distance (diff channel):     "
          f"{pixel_dist[~same_channel].mean():.2f}" if (~same_channel).any() else "  N/A")

    # Per-channel breakdown
    print(f"\n=== Per-channel analysis ===")
    for c in range(num_channels):
        mask = channel == c
        c_idx = idx[mask]  # (n_pixels, 10)
        c_same_pixel = (pixel_idx[c_idx] == pixel_idx[mask][:, None])
        c_same_channel = (channel[c_idx] == c)
        c_pixel_dist = np.abs(px_x[c_idx] - px_x[mask][:, None]) + \
                       np.abs(px_y[c_idx] - px_y[mask][:, None])

        name = ch_names.get(c, str(c))
        print(f"\n  Channel {name}:")
        print(f"    Same pixel:   {c_same_pixel.mean()*100:.1f}%")
        print(f"    Same channel: {c_same_channel.mean()*100:.1f}%")
        print(f"    Mean pixel dist: {c_pixel_dist.mean():.2f}")

        # What channels do neighbors belong to?
        for c2 in range(num_channels):
            frac = (channel[c_idx] == c2).mean() * 100
            name2 = ch_names.get(c2, str(c2))
            marker = " <-- self" if c2 == c else ""
            print(f"    -> {name2}: {frac:.1f}%{marker}")

    # Spatial quality per channel (same-channel neighbors only)
    print(f"\n=== Spatial quality (same-channel K=10 neighbors) ===")
    for c in range(num_channels):
        mask = channel == c
        c_emb = emb[mask]
        c_tree = cKDTree(c_emb)
        _, c_idx = c_tree.query(c_emb, k=11)
        c_idx = c_idx[:, 1:]
        c_px_x = px_x[mask]
        c_px_y = px_y[mask]
        c_dists = np.abs(c_px_x[c_idx] - c_px_x[:, None]) + \
                  np.abs(c_px_y[c_idx] - c_px_y[:, None])
        name = ch_names.get(c, str(c))
        within_5 = (c_dists <= 5).mean() * 100
        print(f"  {name}: mean_dist={c_dists.mean():.2f}, <5px={within_5:.1f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model", help="Path to .npy embeddings")
    parser.add_argument("-W", type=int, required=True, help="Crop width (pixels)")
    parser.add_argument("-H", type=int, required=True, help="Crop height (pixels)")
    parser.add_argument("-C", type=int, default=4, help="Number of channels")
    args = parser.parse_args()
    analyze(args.model, args.W, args.H, args.C)

"""Compare CPU vs GPU UMAP rendering quality with COLOR pixel values.

Uses an RGBG model (25600 neurons = 80x80 pixels x 4 channels) rendered
on a 320x80 grid with color-tinted pixels per channel.

Usage:
    python test_umap_compare_color.py output_13_rgbg_50k/model.npy
"""

import argparse
import numpy as np
import cv2
import time
from render_embeddings import align_to_grid, render


def run_cpu(emb, w, h, n_neighbors=15, min_dist=0.1, n_epochs=None):
    import umap
    t0 = time.time()
    reducer = umap.UMAP(n_components=2, n_neighbors=n_neighbors,
                        min_dist=min_dist, n_epochs=n_epochs,
                        random_state=42, low_memory=False)
    pos = reducer.fit_transform(emb).astype(np.float64)
    elapsed = time.time() - t0
    for d, target in enumerate([w - 1, h - 1]):
        mn, mx = pos[:, d].min(), pos[:, d].max()
        span = mx - mn if mx - mn > 1e-8 else 1.0
        pos[:, d] = (pos[:, d] - mn) / span * target
    return pos, elapsed


def run_gpu(emb, w, h, n_neighbors=50, min_dist=0.1, n_epochs=200):
    from cuml.manifold import UMAP as cuUMAP
    t0 = time.time()
    reducer = cuUMAP(n_components=2, n_neighbors=n_neighbors,
                     min_dist=min_dist, n_epochs=n_epochs,
                     random_state=42)
    pos = reducer.fit_transform(emb).astype(np.float64)
    elapsed = time.time() - t0
    for d, target in enumerate([w - 1, h - 1]):
        mn, mx = pos[:, d].min(), pos[:, d].max()
        span = mx - mn if mx - mn > 1e-8 else 1.0
        pos[:, d] = (pos[:, d] - mn) / span * target
    return pos, elapsed


def render_frame(pos, w, h, pixel_values):
    pos_a = align_to_grid(pos, w, h)
    frame = render(pos_a, w, h, pixel_values)
    return cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)


def make_color_pixels(image_gray, w, h, sig_channels=4):
    """Create color-tinted pixel values for RGBG neurons.
    n = w * h * sig_channels neurons, each tinted by channel identity.
    Neuron i -> pixel i // sig_channels, channel i % sig_channels."""
    n = w * h * sig_channels
    pixel_values = np.zeros((n, 3), dtype=np.uint8)
    # BGR tints matching main.py channel_tints
    tints = [
        np.array([0.3, 0.3, 1.0]),  # R -> red (BGR)
        np.array([0.3, 1.0, 0.3]),  # G -> green (BGR)
        np.array([1.0, 0.3, 0.3]),  # B -> blue (BGR)
        np.array([0.8, 0.8, 0.8]),  # GS -> gray (BGR)
    ]
    flat = image_gray.ravel().astype(np.float32)
    for i in range(n):
        px = i // sig_channels
        ch = i % sig_channels
        tint = tints[ch]
        pixel_values[i] = np.clip(flat[px] * tint, 0, 255).astype(np.uint8)
    return pixel_values


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", help="Path to .npy embeddings")
    parser.add_argument("-i", "--image", default="K_80_g.png")
    parser.add_argument("-W", type=int, default=80, help="Pixel crop width")
    parser.add_argument("-H", type=int, default=80, help="Pixel crop height")
    parser.add_argument("-C", type=int, default=4, help="Signal channels")
    args = parser.parse_args()

    emb = np.load(args.model)
    sig_channels = args.C
    pw, ph = args.W, args.H
    # Render grid: width = pixel_width * channels, height = pixel_height
    w = pw * sig_channels
    h = ph
    n = emb.shape[0]
    assert n == pw * ph * sig_channels, \
        f"n={n} != {pw}*{ph}*{sig_channels}={pw*ph*sig_channels}"
    print(f"Model: {n} neurons, {emb.shape[1]} dims")
    print(f"  Pixels: {pw}x{ph}, channels: {sig_channels}, render grid: {w}x{h}")

    img = cv2.imread(args.image, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (pw, ph))
    pixel_values = make_color_pixels(img, pw, ph, sig_channels)
    print(f"Color pixel values: {pixel_values.shape}, dtype={pixel_values.dtype}")

    configs = [
        ("CPU nn=15 md=0.1",        "cpu", dict(n_neighbors=15, min_dist=0.1)),
        ("GPU nn=15 md=0.1 ep=200",  "gpu", dict(n_neighbors=15, min_dist=0.1, n_epochs=200)),
        ("GPU nn=50 md=0.1 ep=200",  "gpu", dict(n_neighbors=50, min_dist=0.1, n_epochs=200)),
        ("GPU nn=100 md=0.1 ep=200", "gpu", dict(n_neighbors=100, min_dist=0.1, n_epochs=200)),
        ("GPU nn=100 md=0.01 ep=200","gpu", dict(n_neighbors=100, min_dist=0.01, n_epochs=200)),
        ("GPU nn=100 md=0.01 ep=500","gpu", dict(n_neighbors=100, min_dist=0.01, n_epochs=500)),
        ("GPU nn=200 md=0.01 ep=500","gpu", dict(n_neighbors=200, min_dist=0.01, n_epochs=500)),
    ]

    frames = []
    for label, mode, kwargs in configs:
        if mode == "cpu":
            pos, elapsed = run_cpu(emb, w, h, **kwargs)
        else:
            pos, elapsed = run_gpu(emb, w, h, **kwargs)
        frame = render_frame(pos, w, h, pixel_values)
        frames.append(frame)
        print(f"  {label}: {elapsed:.2f}s")
        cv2.imwrite(f"test_umap_color_{label.replace(' ', '_').replace('=', '')}.png", frame)

    # Side by side (color frames are 3-channel)
    sep = np.full((h, 2, 3), 128, dtype=np.uint8)
    combined = frames[0]
    for f in frames[1:]:
        combined = np.hstack([combined, sep, f])
    cv2.imwrite("test_umap_color_sweep.png", combined)
    print(f"\nSaved test_umap_color_sweep.png ({combined.shape[1]}x{combined.shape[0]})")
    print("Left to right:", " | ".join(c[0] for c in configs))


if __name__ == "__main__":
    main()

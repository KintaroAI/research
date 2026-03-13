"""Compare CPU vs GPU UMAP rendering quality.

Uses a well-sorted grayscale model to test whether GPU UMAP
produces comparable visual output to CPU UMAP.

Usage:
    python test_umap_compare.py /tmp/test_gray_cpu/model.npy
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
    # Scale to grid
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", help="Path to .npy embeddings")
    parser.add_argument("-i", "--image", default="K_80_g.png")
    parser.add_argument("-W", type=int, default=80)
    parser.add_argument("-H", type=int, default=80)
    args = parser.parse_args()

    emb = np.load(args.model)
    w, h = args.W, args.H
    n = emb.shape[0]
    print(f"Model: {n} neurons, {emb.shape[1]} dims, grid {w}x{h}")

    img = cv2.imread(args.image, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (w, h))
    pixel_values = img.ravel()

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
        cv2.imwrite(f"test_umap_{label.replace(' ', '_').replace('=', '')}.png", frame)

    # Side by side
    sep = np.full((h, 2), 128, dtype=np.uint8)
    combined = frames[0]
    for f in frames[1:]:
        combined = np.hstack([combined, sep, f])
    cv2.imwrite("test_umap_sweep.png", combined)
    print(f"\nSaved test_umap_sweep.png ({combined.shape[1]}x{combined.shape[0]})")
    print("Left to right:", " | ".join(c[0] for c in configs))


if __name__ == "__main__":
    main()

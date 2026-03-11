"""Render saved embeddings to an image using various projection methods.

Usage:
    python render_embeddings.py output_7/output_7_gensim_80_D8.npy -i K_80_g.png -m umap
    python render_embeddings.py output_7/output_7_ours_80_D8_W.npy -i K_80_g.png -m spectral
    python render_embeddings.py output_7/output_7_ours_80_D3_W.npy -i K_80_g.png -m bestpc

Projection methods (linear):
    pca      — Top 2 principal components
    bestpc   — 2 PCs most correlated with grid x/y (flipped to match orientation)
    angular  — Normalize to unit vectors, then PCA
    direct   — First 2 dimensions as-is

Projection methods (supervised — uses known grid coordinates):
    procrustes — PCA to 2D, then Procrustes alignment to grid
    lstsq      — Least-squares optimal linear map from D→2D targeting grid coords

Projection methods (nonlinear):
    umap     — UMAP (preserves local + some global structure)
    tsne     — t-SNE (preserves local structure)
    mds      — Multidimensional scaling (preserves pairwise distances)
    spectral — Laplacian Eigenmaps / Spectral Embedding (graph-based)
"""

import argparse
import numpy as np
import cv2
from scipy.spatial import cKDTree


def project(emb, width, height, method, prev_2d=None):
    """Project embeddings to 2D using the specified method.

    Args:
        emb: (n, dims) embeddings
        width, height: grid dimensions
        method: projection method name
        prev_2d: (n, 2) previous frame's 2D positions for warm start
                 (used by umap/tsne to initialize and reduce iterations)

    Returns:
        (n, 2) float64 array scaled to grid coordinates.
    """
    n, dims = emb.shape

    if method == 'angular':
        norms = np.linalg.norm(emb, axis=1, keepdims=True)
        norms = np.where(norms < 1e-8, 1.0, norms)
        emb = emb / norms

    if method == 'direct':
        pos_2d = emb[:, :2].copy().astype(np.float64)
    elif method in ('pca', 'angular'):
        if dims > 2:
            _, _, Vt = np.linalg.svd(emb, full_matrices=False)
            pos_2d = (emb @ Vt[:2].T).astype(np.float64)
        else:
            pos_2d = emb[:, :2].copy().astype(np.float64)
    elif method == 'bestpc':
        grid_x = np.arange(n) % width
        grid_y = np.arange(n) // width
        _, _, Vt = np.linalg.svd(emb, full_matrices=False)
        pcs = emb @ Vt.T

        best_x_pc, best_x_corr = 0, 0
        best_y_pc, best_y_corr = 0, 0
        for i in range(dims):
            cx = abs(np.corrcoef(pcs[:, i], grid_x)[0, 1])
            cy = abs(np.corrcoef(pcs[:, i], grid_y)[0, 1])
            if cx > best_x_corr:
                best_x_corr, best_x_pc = cx, i
            if cy > best_y_corr:
                best_y_corr, best_y_pc = cy, i

        pos_2d = np.column_stack([pcs[:, best_x_pc], pcs[:, best_y_pc]]).astype(np.float64)
        if np.corrcoef(pcs[:, best_x_pc], grid_x)[0, 1] < 0:
            pos_2d[:, 0] *= -1
        if np.corrcoef(pcs[:, best_y_pc], grid_y)[0, 1] < 0:
            pos_2d[:, 1] *= -1

        print(f"  bestpc: X=PC{best_x_pc} r={best_x_corr:.4f}, "
              f"Y=PC{best_y_pc} r={best_y_corr:.4f}")
    elif method == 'procrustes':
        from scipy.spatial import procrustes as scipy_procrustes
        _, _, Vt = np.linalg.svd(emb, full_matrices=False)
        pca_2d = (emb @ Vt[:2].T).astype(np.float64)
        grid_coords = np.column_stack([
            np.arange(n) % width,
            np.arange(n) // width
        ]).astype(np.float64)
        _, aligned, disparity = scipy_procrustes(grid_coords, pca_2d)
        pos_2d = aligned
        print(f"  procrustes: disparity={disparity:.4f}")
    elif method == 'lstsq':
        grid_coords = np.column_stack([
            np.arange(n) % width,
            np.arange(n) // width
        ]).astype(np.float64)
        W, _, _, _ = np.linalg.lstsq(emb, grid_coords, rcond=None)
        pos_2d = (emb @ W).astype(np.float64)
        r_x = np.corrcoef(pos_2d[:, 0], grid_coords[:, 0])[0, 1]
        r_y = np.corrcoef(pos_2d[:, 1], grid_coords[:, 1])[0, 1]
        print(f"  lstsq: r_x={r_x:.4f}, r_y={r_y:.4f}")
    elif method == 'umap':
        import umap
        init = 'spectral'
        n_epochs = None  # default
        if prev_2d is not None:
            init = prev_2d.astype(np.float32)
            n_epochs = 50
        reducer = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1,
                            init=init, n_epochs=n_epochs, random_state=42)
        pos_2d = reducer.fit_transform(emb).astype(np.float64)
        warm = "warm " if prev_2d is not None else ""
        print(f"  umap: {warm}n_epochs={n_epochs or 'default'}")
    elif method == 'tsne':
        from sklearn.manifold import TSNE
        init = 'pca'
        max_iter = 1000
        if prev_2d is not None:
            init = prev_2d.astype(np.float32)
            max_iter = 250
        reducer = TSNE(n_components=2, perplexity=30, init=init,
                       max_iter=max_iter, random_state=42)
        pos_2d = reducer.fit_transform(emb).astype(np.float64)
        warm = "warm " if prev_2d is not None else ""
        print(f"  tsne: {warm}max_iter={max_iter}")
    elif method == 'mds':
        from sklearn.manifold import MDS
        reducer = MDS(n_components=2, random_state=42, normalized_stress='auto')
        pos_2d = reducer.fit_transform(emb).astype(np.float64)
        print(f"  mds: SMACOF")
    elif method == 'spectral':
        from sklearn.manifold import SpectralEmbedding
        reducer = SpectralEmbedding(n_components=2, n_neighbors=24,
                                    random_state=42)
        pos_2d = reducer.fit_transform(emb).astype(np.float64)
        print(f"  spectral: n_neighbors=24")
    else:
        raise ValueError(f"Unknown method: {method}")

    # Scale to grid coordinates
    for d, target in enumerate([width - 1, height - 1]):
        mn, mx = pos_2d[:, d].min(), pos_2d[:, d].max()
        span = mx - mn if mx - mn > 1e-8 else 1.0
        pos_2d[:, d] = (pos_2d[:, d] - mn) / span * target

    return pos_2d


def align_to_grid(pos_2d, width, height):
    """Procrustes-align projected 2D positions to the known grid coordinates.
    Fixes arbitrary rotation, reflection, and scaling from the projection."""
    from scipy.spatial import procrustes as scipy_procrustes
    n = pos_2d.shape[0]
    grid_coords = np.column_stack([
        np.arange(n) % width,
        np.arange(n) // width
    ]).astype(np.float64)
    _, aligned, disparity = scipy_procrustes(grid_coords, pos_2d)
    # Rescale aligned output back to grid range
    for d, target in enumerate([width - 1, height - 1]):
        mn, mx = aligned[:, d].min(), aligned[:, d].max()
        span = mx - mn if mx - mn > 1e-8 else 1.0
        aligned[:, d] = (aligned[:, d] - mn) / span * target
    print(f"  procrustes align: disparity={disparity:.4f}")
    return aligned


def render(pos_2d, width, height, pixel_values):
    """Voronoi assignment: each grid cell gets nearest neuron's pixel value."""
    grid_y, grid_x = np.mgrid[0:height, 0:width]
    grid_points = np.column_stack([
        grid_x.ravel().astype(np.float64),
        grid_y.ravel().astype(np.float64)
    ])
    tree = cKDTree(pos_2d)
    _, nearest = tree.query(grid_points)
    return pixel_values[nearest].reshape(height, width)


def main():
    parser = argparse.ArgumentParser(
        description="Render saved embeddings to an image")
    parser.add_argument("model", help="Path to .npy embeddings file")
    parser.add_argument("-i", "--image", required=True,
                        help="Source image (for pixel values)")
    parser.add_argument("-m", "--method", default="bestpc",
                        choices=["pca", "bestpc", "angular", "direct",
                                 "procrustes", "lstsq",
                                 "umap", "tsne", "mds", "spectral"],
                        help="Projection method (default: bestpc)")
    parser.add_argument("-o", "--output", default=None,
                        help="Output path (default: auto-generated)")
    parser.add_argument("-s", "--size", type=int, default=None,
                        help="Override grid size (default: inferred from sqrt(n))")
    parser.add_argument("--align", action="store_true",
                        help="Procrustes-align output to grid (fixes rotation/flip)")
    args = parser.parse_args()

    emb = np.load(args.model)
    n, dims = emb.shape
    print(f"Loaded {args.model}: {n} neurons, {dims} dims")

    # Infer grid size
    if args.size:
        w = h = args.size
    else:
        side = int(np.sqrt(n))
        assert side * side == n, f"n={n} is not a perfect square, use --size"
        w = h = side
    print(f"Grid: {w}x{h}, method: {args.method}")

    # Load image and extract pixel values
    img = cv2.imread(args.image, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Error: could not load {args.image}")
        return
    img = cv2.resize(img, (w, h))
    pixel_values = img.ravel()

    # Project and render
    pos_2d = project(emb, w, h, args.method)
    if args.align:
        pos_2d = align_to_grid(pos_2d, w, h)
    output = render(pos_2d, w, h, pixel_values)

    # Save
    if args.output is None:
        model_name = args.model.replace('.npy', '')
        args.output = f"{model_name}_{args.method}.png"
    cv2.imwrite(args.output, output)
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()

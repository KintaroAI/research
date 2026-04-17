"""Scatter of cluster centroids colored by modality × depth × specificity.

Projects cluster centroids to 2D (PCA or UMAP) and colors each cluster by:
  - hue: weighted circular mean of ancestor sensory hues. Cross-modal
    or cross-region clusters show blended hues automatically.
  - lightness: layer depth (shallow dark → deep light).
  - saturation: ancestry concentration (narrow lineage = saturated,
    broadly mixed = desaturated).

Layer depth is assigned by BFS on the feedback graph: any cluster with
direct sensory input is at depth 1; a cluster whose feedback sources
include a depth-d cluster is at depth d+1.

The module is modality-agnostic: callers provide a per-sensory-neuron
hue and modality id. Helpers build these for common layouts (grid for
visual, linear for audio/freq).

Usage (Python):
    from layer_scatter import render_layer_scatter, make_grid_hue
    sensory_hue = make_grid_hue(w, h, hue_center=0.60, hue_spread=0.15)
    img = render_layer_scatter(most_recent, n_sensory, n_outputs, centroids,
                                sensory_hue=sensory_hue)

Usage (standalone):
    python layer_scatter.py --state-dir /path/to/saved/state --out scatter.png \
        --grid-w 80 --grid-h 80 --n-outputs 4
"""

import argparse
import colorsys
import os
import numpy as np
import cv2


LAYER_LIGHTNESS = [0.22, 0.36, 0.50, 0.64, 0.76, 0.86]


def make_grid_hue(grid_w, grid_h, hue_center=0.60, hue_spread=0.15):
    """Per-pixel hue for a retinotopic modality.

    Gradient along the diagonal so adjacent pixels have similar hues.
    """
    n = grid_w * grid_h
    x = (np.arange(n) % grid_w) / max(grid_w - 1, 1)
    y = (np.arange(n) // grid_w) / max(grid_h - 1, 1)
    t = (x + y) * 0.5
    return (hue_center + (t - 0.5) * hue_spread) % 1.0


def make_linear_hue(n_channels, hue_center, hue_spread=0.15):
    """Per-channel hue for a 1D modality (audio bands, etc.)."""
    t = np.linspace(0.0, 1.0, n_channels)
    return (hue_center + (t - 0.5) * hue_spread) % 1.0


def compute_layer_colors(most_recent, n_sensory, n_outputs, m,
                         sensory_hue):
    """Returns (hue, sat, lightness, layer) arrays of shape (m,).

    Ancestry is propagated layer-by-layer on the unit circle so mixing
    produces circular-mean blends (blue + red → purple, etc.).
    """
    n_total = len(most_recent)
    mr = np.asarray(most_recent)

    cos_sum = np.zeros(m, dtype=np.float64)
    sin_sum = np.zeros(m, dtype=np.float64)
    weight = np.zeros(m, dtype=np.float64)
    sensory_count = np.zeros(m, dtype=np.int64)

    sens_mr = mr[:n_sensory]
    valid = sens_mr >= 0
    angles = 2.0 * np.pi * np.asarray(sensory_hue, dtype=np.float64)
    np.add.at(cos_sum, sens_mr[valid], np.cos(angles[valid]))
    np.add.at(sin_sum, sens_mr[valid], np.sin(angles[valid]))
    np.add.at(weight, sens_mr[valid], 1.0)
    np.add.at(sensory_count, sens_mr[valid], 1)

    fb_idx = np.arange(n_sensory, n_total)
    fb_src = (fb_idx - n_sensory) // n_outputs
    fb_dst = mr[n_sensory:n_total]
    fb_valid = fb_dst >= 0
    fb_src = fb_src[fb_valid]
    fb_dst = fb_dst[fb_valid]

    edge_key = fb_dst.astype(np.int64) * m + fb_src.astype(np.int64)
    uniq_key, counts = np.unique(edge_key, return_counts=True)
    edge_dst = (uniq_key // m).astype(np.int64)
    edge_src = (uniq_key % m).astype(np.int64)
    edge_w = counts.astype(np.float64)

    layer = np.full(m, -1, dtype=np.int32)
    layer[sensory_count > 0] = 1

    prev_set = set(np.where(layer == 1)[0].tolist())
    depth = 2
    while prev_set and depth < 32:
        src_in_prev = np.isin(edge_src, list(prev_set))
        candidates = np.unique(edge_dst[src_in_prev])
        new_set = set(int(c) for c in candidates if layer[c] < 0)
        if not new_set:
            break
        for c in new_set:
            layer[c] = depth
        prev_set = new_set
        depth += 1

    max_depth = int(layer.max()) if (layer > 0).any() else 1

    for d in range(2, max_depth + 1):
        dst_mask = layer[edge_dst] == d
        src_mask = (layer[edge_src] > 0) & (layer[edge_src] < d)
        active = dst_mask & src_mask
        if not active.any():
            continue
        src_sel = edge_src[active]
        dst_sel = edge_dst[active]
        w_sel = edge_w[active]

        src_w = weight[src_sel]
        nonzero = src_w > 1e-9
        if not nonzero.any():
            continue
        src_sel = src_sel[nonzero]
        dst_sel = dst_sel[nonzero]
        w_sel = w_sel[nonzero]
        src_w = src_w[nonzero]

        scale = w_sel / src_w
        np.add.at(cos_sum, dst_sel, cos_sum[src_sel] * scale)
        np.add.at(sin_sum, dst_sel, sin_sum[src_sel] * scale)
        np.add.at(weight, dst_sel, w_sel)

    hue = np.zeros(m, dtype=np.float64)
    sat = np.zeros(m, dtype=np.float64)
    mag = np.sqrt(cos_sum * cos_sum + sin_sum * sin_sum)
    good = weight > 1e-9
    hue[good] = (np.arctan2(sin_sum[good], cos_sum[good]) / (2.0 * np.pi)) % 1.0
    sat[good] = np.clip(mag[good] / weight[good], 0.0, 1.0)

    light = np.full(m, 0.5, dtype=np.float64)
    for c in range(m):
        d = int(layer[c]) if layer[c] > 0 else 1
        idx = min(d - 1, len(LAYER_LIGHTNESS) - 1)
        light[c] = LAYER_LIGHTNESS[idx]

    return hue, sat, light, layer


def _project_2d(centroids, method='pca'):
    """Return (m, 2) projection. Missing/zero centroids land at origin."""
    c = centroids.astype(np.float64)
    n, d = c.shape
    if d <= 2:
        out = np.zeros((n, 2), dtype=np.float64)
        out[:, :min(d, 2)] = c[:, :min(d, 2)]
        return out
    mean = c.mean(axis=0, keepdims=True)
    cz = c - mean
    if method == 'umap':
        try:
            import umap
            reducer = umap.UMAP(n_components=2, n_neighbors=15,
                                min_dist=0.15, random_state=42)
            return reducer.fit_transform(cz).astype(np.float64)
        except Exception as e:
            print(f"  umap unavailable ({e}); falling back to pca")
    _, _, Vt = np.linalg.svd(cz, full_matrices=False)
    return (cz @ Vt[:2].T).astype(np.float64)


def render_layer_scatter(most_recent, n_sensory, n_outputs, centroids,
                         sensory_hue=None, sensory_modality=None,
                         modality_names=None, img_size=1200,
                         method='pca', title=None, dot_scale=1.0):
    """Render scatter image (BGR uint8 array)."""
    centroids = np.asarray(centroids)
    m = centroids.shape[0]
    most_recent = np.asarray(most_recent)

    if sensory_hue is None:
        sensory_hue = np.full(n_sensory, 0.60, dtype=np.float64)
    else:
        sensory_hue = np.asarray(sensory_hue, dtype=np.float64)
    if sensory_modality is None:
        sensory_modality = np.zeros(n_sensory, dtype=np.int32)

    sizes = np.bincount(most_recent[most_recent >= 0], minlength=m)
    alive = sizes > 0

    hue, sat, light, layer = compute_layer_colors(
        most_recent, n_sensory, n_outputs, m, sensory_hue)

    pos = _project_2d(centroids, method=method)

    img = np.full((img_size, img_size, 3), 22, dtype=np.uint8)

    if not alive.any():
        return img

    margin = 80
    p_alive = pos[alive]
    pmin = p_alive.min(axis=0)
    pmax = p_alive.max(axis=0)
    center = (pmin + pmax) * 0.5
    span = max((pmax - pmin).max(), 1e-6)
    scale = (img_size - 2 * margin) / span

    cx, cy = img_size / 2, img_size / 2
    draw_order = np.argsort(layer)

    for c in draw_order:
        if not alive[c] or layer[c] <= 0:
            continue
        r, g, b = colorsys.hls_to_rgb(hue[c], light[c], sat[c])
        bgr = (int(b * 255), int(g * 255), int(r * 255))
        px = int(cx + (pos[c, 0] - center[0]) * scale)
        py = int(cy + (pos[c, 1] - center[1]) * scale)
        radius = max(4, int(np.sqrt(max(sizes[c], 1)) * dot_scale * 1.8))
        cv2.circle(img, (px, py), radius, bgr, -1, lineType=cv2.LINE_AA)
        cv2.circle(img, (px, py), radius, (60, 60, 60), 1, lineType=cv2.LINE_AA)

    _draw_legend(img, layer, hue, sat, alive, modality_names)
    if title:
        cv2.putText(img, title, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (230, 230, 230), 1, cv2.LINE_AA)
    return img


def _draw_legend(img, layer, hue, sat, alive, modality_names):
    """Draw depth legend (lightness swatch per layer) in bottom-left."""
    y0 = img.shape[0] - 30
    depths = sorted(set(int(d) for d in layer[alive] if d > 0))
    for d in depths:
        mask = alive & (layer == d)
        count = int(mask.sum())
        idx = min(d - 1, len(LAYER_LIGHTNESS) - 1)
        # Representative hue = mean hue vector of this depth's clusters
        if mask.any():
            angles = 2 * np.pi * hue[mask]
            cm = np.cos(angles).mean()
            sm = np.sin(angles).mean()
            rep_hue = (np.arctan2(sm, cm) / (2 * np.pi)) % 1.0
            rep_sat = np.clip(np.sqrt(cm * cm + sm * sm), 0.0, 1.0)
        else:
            rep_hue, rep_sat = 0.6, 0.7
        r, g, b = colorsys.hls_to_rgb(rep_hue, LAYER_LIGHTNESS[idx], max(rep_sat, 0.3))
        bgr = (int(b * 255), int(g * 255), int(r * 255))
        cv2.circle(img, (30, y0), 10, bgr, -1, lineType=cv2.LINE_AA)
        label = f"L{d} ({count})"
        if modality_names and 0 in modality_names:
            label = f"{modality_names[0]}{d} ({count})"
        cv2.putText(img, label, (50, y0 + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220),
                    1, cv2.LINE_AA)
        y0 -= 26


def _load_state(state_dir):
    cluster_ids = np.load(os.path.join(state_dir, "cluster_ids.npy"))
    pointers = np.load(os.path.join(state_dir, "pointers.npy"))
    centroids = np.load(os.path.join(state_dir, "centroids.npy"))
    n = cluster_ids.shape[0]
    most_recent = cluster_ids[np.arange(n), pointers]
    return most_recent, centroids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--state-dir", required=True,
                    help="directory with cluster_ids.npy, pointers.npy, centroids.npy")
    ap.add_argument("--out", default="layer_scatter.png")
    ap.add_argument("--n-outputs", type=int, required=True,
                    help="outputs per column (n_outputs in cluster manager)")
    ap.add_argument("--grid-w", type=int, default=None,
                    help="visual grid width (for retinotopic hue)")
    ap.add_argument("--grid-h", type=int, default=None)
    ap.add_argument("--n-sensory", type=int, default=None,
                    help="sensory boundary; defaults to grid-w*grid-h")
    ap.add_argument("--hue-center", type=float, default=0.60)
    ap.add_argument("--hue-spread", type=float, default=0.15)
    ap.add_argument("--method", default="pca", choices=["pca", "umap"])
    ap.add_argument("--img-size", type=int, default=1200)
    ap.add_argument("--dot-scale", type=float, default=1.0)
    ap.add_argument("--title", default=None)
    args = ap.parse_args()

    most_recent, centroids = _load_state(args.state_dir)

    if args.n_sensory is not None:
        n_sensory = args.n_sensory
    elif args.grid_w and args.grid_h:
        n_sensory = args.grid_w * args.grid_h
    else:
        raise SystemExit("provide --n-sensory or --grid-w and --grid-h")

    if args.grid_w and args.grid_h:
        sensory_hue = make_grid_hue(args.grid_w, args.grid_h,
                                    args.hue_center, args.hue_spread)
    else:
        sensory_hue = np.full(n_sensory, args.hue_center)

    img = render_layer_scatter(
        most_recent, n_sensory, args.n_outputs, centroids,
        sensory_hue=sensory_hue, method=args.method,
        title=args.title, img_size=args.img_size,
        dot_scale=args.dot_scale,
        modality_names={0: "V"})
    cv2.imwrite(args.out, img)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()

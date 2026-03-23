"""SHAPES benchmark: visual hierarchy + categorical association.

A large image contains distinct shapes (~20×20 pixels each). When the
saccade crop overlaps a shape, 8 label neurons encode which shape is
visible. Tests whether V1/V2 visual features associate with shape labels.

Grid: 20×16 = 320 neurons.
  First 256: 16×16 pixel crop from image
  Last 64: 8 shape categories × 8 neurons each (one-hot label)

Usage:
    python main.py word2vec --signal-source shapes -W 20 -H 16 ...
"""

import os
import json
import numpy as np

name = 'shapes'
description = 'Shape recognition: visual features + categorical labels'

CROP_W = 16
CROP_H = 16
N_SHAPES = 8
LABEL_NEURONS_PER = 8


def add_args(parser):
    parser.add_argument("--shapes-step", type=int, default=3,
                        help="Saccade step size (default: 3)")
    parser.add_argument("--shapes-threshold", type=float, default=0.3,
                        help="Overlap fraction to activate label (default: 0.3)")


def _draw_shape(img, shape_id, cx, cy, size=20):
    """Draw a distinct shape on the image."""
    h, w = img.shape
    r = size // 2

    if shape_id == 0:  # filled square
        y0, y1 = max(0, cy-r), min(h, cy+r)
        x0, x1 = max(0, cx-r), min(w, cx+r)
        img[y0:y1, x0:x1] = 0.9

    elif shape_id == 1:  # hollow square
        for dy in range(-r, r):
            for dx in range(-r, r):
                y, x = cy+dy, cx+dx
                if 0 <= y < h and 0 <= x < w:
                    if abs(dy) >= r-2 or abs(dx) >= r-2:
                        img[y, x] = 0.8

    elif shape_id == 2:  # filled circle
        for dy in range(-r, r):
            for dx in range(-r, r):
                if dx*dx + dy*dy < r*r:
                    y, x = cy+dy, cx+dx
                    if 0 <= y < h and 0 <= x < w:
                        img[y, x] = 0.85

    elif shape_id == 3:  # horizontal lines
        for dy in range(-r, r):
            if dy % 4 < 2:
                for dx in range(-r, r):
                    y, x = cy+dy, cx+dx
                    if 0 <= y < h and 0 <= x < w:
                        img[y, x] = 0.75

    elif shape_id == 4:  # vertical lines
        for dx in range(-r, r):
            if dx % 4 < 2:
                for dy in range(-r, r):
                    y, x = cy+dy, cx+dx
                    if 0 <= y < h and 0 <= x < w:
                        img[y, x] = 0.7

    elif shape_id == 5:  # cross
        for d in range(-r, r):
            if 0 <= cy+d < h and 0 <= cx < w:
                img[cy+d, max(0,cx-1):min(w,cx+2)] = 0.95
            if 0 <= cy < h and 0 <= cx+d < w:
                img[max(0,cy-1):min(h,cy+2), cx+d] = 0.95

    elif shape_id == 6:  # diagonal
        for d in range(-r, r):
            y, x = cy+d, cx+d
            if 0 <= y < h and 0 <= x < w:
                img[max(0,y-1):min(h,y+2), max(0,x-1):min(w,x+2)] = 0.65

    elif shape_id == 7:  # checkerboard patch
        for dy in range(-r, r):
            for dx in range(-r, r):
                y, x = cy+dy, cx+dx
                if 0 <= y < h and 0 <= x < w:
                    if ((dx+r)//3 + (dy+r)//3) % 2 == 0:
                        img[y, x] = 0.8


def make_signal(w, h, args):
    n = w * h
    step = getattr(args, 'shapes_step', 3)
    overlap_threshold = getattr(args, 'shapes_threshold', 0.3)
    rng = np.random.RandomState(42)

    # Generate image with shapes
    src_size = 160
    img = np.full((src_size, src_size), 0.1, dtype=np.float32)

    # Place 8 shapes at fixed positions (spread across the image)
    shape_positions = [
        (30, 30), (30, 80), (30, 130),
        (70, 50), (70, 110),
        (110, 30), (110, 80), (110, 130),
    ]
    shape_regions = []  # (y0, y1, x0, x1, shape_id)
    shape_size = 20

    for sid, (cy, cx) in enumerate(shape_positions):
        _draw_shape(img, sid, cx, cy, shape_size)
        r = shape_size // 2
        shape_regions.append((
            max(0, cy-r), min(src_size, cy+r),
            max(0, cx-r), min(src_size, cx+r),
            sid
        ))

    # Label neuron indices (after pixel neurons)
    n_pixels = CROP_W * CROP_H  # 256
    label_idx = {}
    offset = n_pixels
    for sid in range(N_SHAPES):
        label_idx[sid] = list(range(offset, offset + LABEL_NEURONS_PER))
        offset += LABEL_NEURONS_PER

    # Saccade state
    max_dy = src_size - CROP_H
    max_dx = src_size - CROP_W
    walk_y = rng.randint(0, max_dy + 1)
    walk_x = rng.randint(0, max_dx + 1)

    feature_log = []

    def tick_fn(t):
        nonlocal walk_y, walk_x
        walk_y = np.clip(walk_y + rng.randint(-step, step + 1), 0, max_dy)
        walk_x = np.clip(walk_x + rng.randint(-step, step + 1), 0, max_dx)

        # Pixel crop
        crop = img[walk_y:walk_y+CROP_H, walk_x:walk_x+CROP_W].ravel()

        # Determine which shape the crop overlaps
        crop_y0, crop_y1 = walk_y, walk_y + CROP_H
        crop_x0, crop_x1 = walk_x, walk_x + CROP_W
        active_shape = -1
        best_overlap = 0.0

        for sy0, sy1, sx0, sx1, sid in shape_regions:
            # Intersection area
            iy0 = max(crop_y0, sy0)
            iy1 = min(crop_y1, sy1)
            ix0 = max(crop_x0, sx0)
            ix1 = min(crop_x1, sx1)
            if iy1 > iy0 and ix1 > ix0:
                overlap = (iy1 - iy0) * (ix1 - ix0)
                shape_area = (sy1 - sy0) * (sx1 - sx0)
                frac = overlap / max(shape_area, 1)
                if frac > best_overlap:
                    best_overlap = frac
                    active_shape = sid

        if best_overlap < overlap_threshold:
            active_shape = -1

        feature_log.append((t, walk_x, walk_y, active_shape, best_overlap))

        # Build signal
        sig = np.zeros(n, dtype=np.float32)
        sig[:n_pixels] = crop

        # One-hot shape label
        for sid in range(N_SHAPES):
            val = 1.0 if sid == active_shape else 0.0
            for i in label_idx[sid]:
                sig[i] = val

        sig += rng.randn(n).astype(np.float32) * 0.02

        return sig

    metadata = {
        'feature_log': feature_log,
        'w': w, 'h': h, 'n': n,
        'src_size': src_size,
        'img': img,
        'shape_positions': shape_positions,
        'n_pixels': n_pixels,
        'label_idx': label_idx,
    }

    print(f"  signal buffer: SHAPES ({src_size}×{src_size}, {N_SHAPES} shapes, "
          f"crop={CROP_W}×{CROP_H}, labels={N_SHAPES}×{LABEL_NEURONS_PER})")

    return tick_fn, metadata


def analyze(metadata, cluster_mgr, signals, tick_counter, T, output_dir):
    import torch

    if cluster_mgr is None or not cluster_mgr.initialized:
        return
    if cluster_mgr.column_mgr is None:
        return

    n = metadata['n']
    n_sensory = cluster_mgr.n_sensory
    n_total = cluster_mgr.n
    n_outputs = cluster_mgr.column_mgr.n_outputs
    m = cluster_mgr.m
    n_pixels = metadata['n_pixels']
    label_idx = metadata['label_idx']

    most_recent = cluster_mgr.cluster_ids[
        np.arange(n_total), cluster_mgr.pointers]

    # --- Split sensory into V1 (pixels) and L1 (labels) ---
    # All label neuron indices (256-319)
    all_label_neurons = set()
    for neurons in label_idx.values():
        all_label_neurons.update(neurons)

    # Per-cluster: count pixel neurons, label neurons, feedback neurons
    pix_counts = np.zeros(m, dtype=int)
    lab_counts = np.zeros(m, dtype=int)
    fb_counts = np.zeros(m, dtype=int)
    for c in range(m):
        members = np.where(most_recent == c)[0]
        for idx in members:
            if idx >= n_sensory:
                fb_counts[c] += 1
            elif idx in all_label_neurons:
                lab_counts[c] += 1
            else:
                pix_counts[c] += 1

    # V1 = clusters with pixel neurons (visual sensory)
    # L1 = clusters with label neurons (categorical sensory)
    # Some clusters may have both (mixed) — classify by majority
    v1 = set()
    l1 = set()
    for c in range(m):
        if pix_counts[c] > 0 and lab_counts[c] == 0:
            v1.add(c)
        elif lab_counts[c] > 0 and pix_counts[c] == 0:
            l1.add(c)
        elif pix_counts[c] > 0 and lab_counts[c] > 0:
            # Mixed — count as V1 (it has visual neurons)
            v1.add(c)
            l1.add(c)  # also L1 since it has labels

    # Build visual hierarchy: V2 from V1, V3 from V2, etc.
    assigned = v1 | l1
    layers = [v1]  # layers[0] = V1
    layer_names = ['V1']
    for depth in range(10):  # up to V11
        prev = layers[-1]
        next_layer = set()
        for c in range(m):
            if c in assigned or fb_counts[c] == 0:
                continue
            fb = np.where((most_recent == c) & (np.arange(n_total) >= n_sensory))[0]
            src = set((fb - n_sensory) // n_outputs)
            if src & prev:
                next_layer.add(c)
        if not next_layer:
            break
        layers.append(next_layer)
        assigned |= next_layer
        layer_names.append(f'V{depth + 2}')

    print(f"  SHAPES hierarchy (V1=visual, L1=labels):")
    print(f"    V1 (pixels): {len(v1)} clusters")
    print(f"    L1 (labels): {len(l1)} clusters")
    for i, (name, layer) in enumerate(zip(layer_names[1:], layers[1:]), 1):
        print(f"    {name} (from {layer_names[i-1]}): {len(layer)} clusters")

    # --- Which L1 clusters per shape? ---
    label_clusters = {}
    for sid, neurons in label_idx.items():
        clusters = set(most_recent[ni] for ni in neurons if ni < n_total)
        label_clusters[sid] = clusters
        in_l1 = [c for c in clusters if c in l1]
        in_v1 = [c for c in clusters if c in v1 and c not in l1]
        print(f"    Shape {sid} labels → L1 clusters {sorted(in_l1)}"
              + (f", mixed V1 {sorted(in_v1)}" if in_v1 else ""))

    # --- Sample column outputs ---
    print("  SHAPES analysis: sampling 500 ticks...")
    tick_fn = metadata['_tick_fn']
    col_history = []

    for t in range(500):
        sig_t = tick_fn(metadata['_total_ticks'] + T + t)
        col_t = tick_counter[0] % T
        signals[:n, col_t] = torch.from_numpy(sig_t).to(signals.device)
        tick_counter[0] += 1
        if cluster_mgr._signals is not None:
            cw = cluster_mgr.column_mgr.window
            indices = [(tick_counter[0] - 1 - i) % T for i in range(cw)]
            sw = cluster_mgr._signals[:, indices].cpu().numpy()
            cluster_mgr.column_mgr.tick(sw)
            col_history.append((t, cluster_mgr.column_mgr.get_outputs()))

    log = np.array(metadata['feature_log'])
    all_outputs = np.array([o for _, o in col_history])
    n_ticks = min(len(log), all_outputs.shape[0])
    all_outputs = all_outputs[:n_ticks]
    active_shapes = log[-n_ticks:, 3]  # shape id or -1

    # --- Correlate Vx column outputs with shape identity ---
    # Search ONLY in visual hierarchy (V2+), not in L1 or V1
    # V1 correlation is trivial (label neurons are sensory)
    # We want: does the visual processing path learn shape identity?
    all_visual = set()
    for layer in layers:
        all_visual |= layer

    def _layer_of(c):
        for name, layer in zip(layer_names, layers):
            if c in layer:
                return name
        if c in l1:
            return 'L1'
        return '?'

    print(f"\n  SHAPES: visual hierarchy correlations with shape identity ({n_ticks} ticks)")
    print(f"  (searching V2+ only — V1/L1 correlations are trivial)")

    results = {}
    for sid in range(N_SHAPES):
        indicator = (active_shapes == sid).astype(np.float32)
        if indicator.std() < 1e-8:
            results[f'shape_{sid}'] = {
                'best_v1': {'r': 0, 'col': -1, 'out': -1},
                'best_vx': {'r': 0, 'col': -1, 'out': -1, 'layer': '?'},
            }
            continue

        # Best in V1 (for reference)
        best_v1_r, best_v1_c, best_v1_o = 0, -1, -1
        # Best in V2+ (the interesting result)
        best_vx_r, best_vx_c, best_vx_o = 0, -1, -1

        for c in range(m):
            for o in range(n_outputs):
                out = all_outputs[:, c, o]
                if out.std() < 1e-8:
                    continue
                r = np.corrcoef(indicator, out)[0, 1]
                if np.isnan(r):
                    continue
                ar = abs(r)
                if c in v1 and c not in l1 and ar > best_v1_r:
                    best_v1_r, best_v1_c, best_v1_o = ar, c, o
                # V2+ only (not V1, not L1)
                if c not in v1 and c not in l1 and ar > best_vx_r:
                    best_vx_r, best_vx_c, best_vx_o = ar, c, o

        vx_layer = _layer_of(best_vx_c) if best_vx_c >= 0 else '?'
        results[f'shape_{sid}'] = {
            'best_v1': {'r': round(best_v1_r, 4), 'col': best_v1_c, 'out': best_v1_o},
            'best_vx': {'r': round(best_vx_r, 4), 'col': best_vx_c,
                        'out': best_vx_o, 'layer': vx_layer},
        }
        print(f"    shape {sid}: V1 r={best_v1_r:.3f} (col {best_v1_c})"
              f"  |  V2+ r={best_vx_r:.3f} (col {best_vx_c}/{vx_layer})")

    # --- Cross-correlate L1 column outputs with Vx column outputs ---
    print(f"\n  SHAPES: L1-to-Vx column output correlation")
    print(f"  (do visual hierarchy columns track label columns?)")

    # Get L1 cluster column output time series
    l1_list = sorted(l1 - v1)  # pure L1 clusters (not mixed)
    if l1_list:
        for l1c in l1_list:
            # Which shape does this L1 cluster represent?
            shape_for_cluster = -1
            for sid, clusters in label_clusters.items():
                if l1c in clusters:
                    shape_for_cluster = sid
                    break

            # Find best Vx correlation with each L1 column output
            for l1o in range(n_outputs):
                l1_ts = all_outputs[:, l1c, l1o]
                if l1_ts.std() < 1e-8:
                    continue
                best_r, best_c, best_o = 0, -1, -1
                for c in range(m):
                    if c in v1 or c in l1:
                        continue
                    for o in range(n_outputs):
                        vx_ts = all_outputs[:, c, o]
                        if vx_ts.std() < 1e-8:
                            continue
                        r = np.corrcoef(l1_ts, vx_ts)[0, 1]
                        if not np.isnan(r) and abs(r) > best_r:
                            best_r, best_c, best_o = abs(r), c, o
                if best_r > 0.2:
                    layer = _layer_of(best_c)
                    print(f"    L1 cluster {l1c} (shape {shape_for_cluster}) "
                          f"out {l1o} → {layer} col {best_c} out {best_o} "
                          f"r={best_r:.3f}")

    if output_dir:
        save_data = {
            'hierarchy': {
                'V1': len(v1), 'L1': len(l1),
                **{name: len(layer) for name, layer in zip(layer_names[1:], layers[1:])},
            },
            'shape_correlations': results,
        }
        path = os.path.join(output_dir, "shapes_analysis.json")
        with open(path, 'w') as f:
            json.dump(save_data, f, indent=2)

        try:
            import cv2
            img_uint8 = (metadata['img'] * 255).astype(np.uint8)
            cv2.imwrite(os.path.join(output_dir, "shapes_source.png"), img_uint8)
        except ImportError:
            pass

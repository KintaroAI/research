"""EDGES benchmark: visual hierarchy from spatial edge features.

Generates a synthetic image with sharp edges (checkerboard + lines +
shapes) and uses saccade crops as input. Tests whether the feedback
loop naturally forms V1→V2→V3 hierarchy:

- V1: clusters of nearby pixels detect local edges/gradients
- V2: feedback from V1 columns detect combinations (corners, T-junctions)
- V3: feedback from V2 detect complex shapes

The signal is real spatial — pixels that are neighbors AND share an
edge will correlate. Derivative-correlation naturally discovers edges.

Usage:
    python main.py word2vec --signal-source edges -W 16 -H 16 ...
"""

import os
import json
import numpy as np

name = 'edges'
description = 'Visual hierarchy: edges → corners → shapes from saccade crops'


def add_args(parser):
    parser.add_argument("--edges-step", type=int, default=3,
                        help="Saccade step size in pixels (default: 3)")


def make_signal(w, h, args):
    n = w * h
    step = getattr(args, 'edges_step', 3)
    rng = np.random.RandomState(42)

    # Generate a rich synthetic image with clear edges
    src_size = 128
    img = np.zeros((src_size, src_size), dtype=np.float32)

    # Checkerboard (strong edges in both directions)
    check_size = 16
    for y in range(src_size):
        for x in range(src_size):
            if ((x // check_size) + (y // check_size)) % 2 == 0:
                img[y, x] = 0.8

    # Horizontal lines
    for y in [20, 50, 80, 110]:
        img[y:y+2, :] = 1.0

    # Vertical lines
    for x in [30, 60, 90]:
        img[:, x:x+2] = 1.0

    # Filled rectangles (create corners and enclosed regions)
    img[10:30, 40:70] = 0.6
    img[70:100, 10:40] = 0.4
    img[50:80, 80:110] = 0.9

    # Circles (curved edges)
    cy, cx, r = 40, 100, 15
    for y in range(src_size):
        for x in range(src_size):
            if (x - cx)**2 + (y - cy)**2 < r**2:
                img[y, x] = 0.7

    # Random walk saccade
    max_dy = src_size - h
    max_dx = src_size - w
    walk_y = rng.randint(0, max_dy + 1)
    walk_x = rng.randint(0, max_dx + 1)

    feature_log = []

    def tick_fn(t):
        nonlocal walk_y, walk_x
        walk_y = np.clip(walk_y + rng.randint(-step, step + 1), 0, max_dy)
        walk_x = np.clip(walk_x + rng.randint(-step, step + 1), 0, max_dx)
        crop = img[walk_y:walk_y+h, walk_x:walk_x+w].ravel()
        feature_log.append((t, walk_x, walk_y))
        return crop

    metadata = {
        'feature_log': feature_log,
        'w': w, 'h': h, 'n': n,
        'src_size': src_size,
        'img': img,
    }

    print(f"  signal buffer: EDGES synthetic ({src_size}×{src_size}, "
          f"crop={w}×{h}, step={step})")

    return tick_fn, metadata


def analyze(metadata, cluster_mgr, signals, tick_counter, T, output_dir):
    import torch

    if cluster_mgr is None or not cluster_mgr.initialized:
        return
    if cluster_mgr.column_mgr is None:
        return

    w, h = metadata['w'], metadata['h']
    n = metadata['n']
    n_sensory = cluster_mgr.n_sensory
    n_total = cluster_mgr.n
    n_outputs = cluster_mgr.column_mgr.n_outputs
    m = cluster_mgr.m

    most_recent = cluster_mgr.cluster_ids[
        np.arange(n_total), cluster_mgr.pointers]

    # --- Layer classification ---
    s_counts = np.zeros(m, dtype=int)
    f_counts = np.zeros(m, dtype=int)
    for c in range(m):
        members = np.where(most_recent == c)[0]
        s_counts[c] = (members < n_sensory).sum()
        f_counts[c] = (members >= n_sensory).sum()

    v1_set = set(c for c in range(m) if s_counts[c] > 0)
    v2_set = set()
    for c in range(m):
        if s_counts[c] == 0 and f_counts[c] > 0:
            fb_members = np.where((most_recent == c) &
                                  (np.arange(n_total) >= n_sensory))[0]
            source_cols = set((fb_members - n_sensory) // n_outputs)
            if source_cols & v1_set:
                v2_set.add(c)
    v3_set = set()
    for c in range(m):
        if c not in v1_set and c not in v2_set and f_counts[c] > 0:
            fb_members = np.where((most_recent == c) &
                                  (np.arange(n_total) >= n_sensory))[0]
            source_cols = set((fb_members - n_sensory) // n_outputs)
            if source_cols & v2_set:
                v3_set.add(c)

    print(f"  EDGES hierarchy:")
    print(f"    V1 (sensory): {len(v1_set)} clusters")
    print(f"    V2 (from V1): {len(v2_set)} clusters")
    print(f"    V3 (from V2): {len(v3_set)} clusters")
    alive = set(c for c in range(m) if s_counts[c] + f_counts[c] > 0)
    unclassified = alive - v1_set - v2_set - v3_set
    print(f"    Unclassified: {len(unclassified)}/{len(alive)} alive clusters")

    # --- V1 spatial coherence: do sensory clusters contain nearby pixels? ---
    v1_diameters = []
    for c in v1_set:
        members = np.where((most_recent == c) &
                           (np.arange(n_total) < n_sensory))[0]
        if len(members) < 2:
            continue
        xs = members % w
        ys = members // w
        dx = xs.max() - xs.min()
        dy = ys.max() - ys.min()
        diameter = max(dx, dy)
        v1_diameters.append(diameter)

    if v1_diameters:
        mean_diam = np.mean(v1_diameters)
        print(f"    V1 spatial diameter: mean={mean_diam:.1f} "
              f"(max grid={max(w,h)-1}, smaller=more local)")

    # --- V2 receptive field: union of V1 source cluster positions ---
    v2_rf_sizes = []
    for c in v2_set:
        fb_members = np.where((most_recent == c) &
                              (np.arange(n_total) >= n_sensory))[0]
        source_cols = set((fb_members - n_sensory) // n_outputs)
        # Gather all sensory positions from source V1 clusters
        all_xs, all_ys = [], []
        for sc in source_cols:
            if sc in v1_set:
                s_members = np.where((most_recent == sc) &
                                     (np.arange(n_total) < n_sensory))[0]
                all_xs.extend((s_members % w).tolist())
                all_ys.extend((s_members // w).tolist())
        if all_xs:
            rf_size = max(max(all_xs) - min(all_xs),
                         max(all_ys) - min(all_ys))
            v2_rf_sizes.append(rf_size)

    if v2_rf_sizes:
        mean_rf = np.mean(v2_rf_sizes)
        print(f"    V2 receptive field: mean={mean_rf:.1f} "
              f"(should be > V1 diameter)")

    # --- V2 fan-in: how many V1 sources per V2 cluster? ---
    v2_fan_ins = []
    v2_all_fb = set()
    for c in v2_set:
        fb_members = np.where((most_recent == c) &
                              (np.arange(n_total) >= n_sensory))[0]
        source_cols = set((fb_members - n_sensory) // n_outputs)
        v1_sources = source_cols & v1_set
        v2_fan_ins.append(len(v1_sources))
        v2_all_fb.update(fb_members.tolist())

    if v2_fan_ins:
        print(f"    V2 fan-in from V1: mean={np.mean(v2_fan_ins):.1f}, "
              f"min={min(v2_fan_ins)}, max={max(v2_fan_ins)}")

    # --- V1→V2 coverage: do V1 column outputs end up in V2 clusters? ---
    v1_feeding_v2 = 0
    for c in v1_set:
        col_fb = set(n_sensory + c * n_outputs + o for o in range(n_outputs))
        if col_fb & v2_all_fb:
            v1_feeding_v2 += 1
    print(f"    V1→V2 coverage: {v1_feeding_v2}/{len(v1_set)} V1 clusters feed V2")

    # --- Column differentiation per layer ---
    cm = cluster_mgr.column_mgr
    # Quick differentiation check
    rng_test = np.random.RandomState(99)
    v1_diff, v2_diff = [], []
    for _ in range(50):
        fake_sig = rng_test.randn(n_total, cm.window).astype(np.float32) * 0.1
        cm.tick(fake_sig)
        probs = cm._outputs
        for c in v1_set:
            v1_diff.append(float(probs[c].max()))
        for c in v2_set:
            v2_diff.append(float(probs[c].max()))

    if v1_diff:
        print(f"    V1 column max_prob: {np.mean(v1_diff):.3f}")
    if v2_diff:
        print(f"    V2 column max_prob: {np.mean(v2_diff):.3f}")

    if output_dir:
        results = {
            'hierarchy': {
                'V1': len(v1_set),
                'V2': len(v2_set),
                'V3': len(v3_set),
                'unclassified': len(unclassified),
                'alive': len(alive),
            },
            'v1_mean_diameter': float(np.mean(v1_diameters)) if v1_diameters else None,
            'v2_mean_receptive_field': float(np.mean(v2_rf_sizes)) if v2_rf_sizes else None,
            'v2_fan_in': {
                'mean': float(np.mean(v2_fan_ins)) if v2_fan_ins else None,
                'min': min(v2_fan_ins) if v2_fan_ins else None,
                'max': max(v2_fan_ins) if v2_fan_ins else None,
            },
            'v1_v2_coverage': f"{v1_feeding_v2}/{len(v1_set)}",
        }
        path = os.path.join(output_dir, "edges_analysis.json")
        with open(path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"  EDGES analysis saved: {path}")

        # Save the source image and V1 cluster map
        try:
            import cv2
            img_uint8 = (metadata['img'] * 255).astype(np.uint8)
            cv2.imwrite(os.path.join(output_dir, "edges_source.png"), img_uint8)

            # V1 cluster map: color each pixel by its cluster assignment
            # Golden-ratio hue spacing for max visual distinction
            v1_list = sorted(v1_set)
            color_map = {}
            golden = 0.618033988749895
            for i, c in enumerate(v1_list):
                hue = int(180 * ((i * golden) % 1.0))
                sat = 180 + (i % 3) * 30       # vary saturation
                val = 200 + ((i // 3) % 2) * 40  # vary brightness
                color_map[c] = cv2.cvtColor(
                    np.array([[[hue, sat, val]]], dtype=np.uint8),
                    cv2.COLOR_HSV2BGR)[0, 0].tolist()

            grid = np.full((h, w, 3), 40, dtype=np.uint8)
            for idx in range(n_sensory):
                py, px = idx // w, idx % w
                cid = most_recent[idx]
                if cid in color_map:
                    grid[py, px] = color_map[cid]

            scale = 16
            grid_large = cv2.resize(grid, (w * scale, h * scale),
                                    interpolation=cv2.INTER_NEAREST)
            # Grid lines
            for i in range(1, w):
                cv2.line(grid_large, (i * scale, 0),
                         (i * scale, h * scale), (80, 80, 80), 1)
            for i in range(1, h):
                cv2.line(grid_large, (0, i * scale),
                         (w * scale, i * scale), (80, 80, 80), 1)

            map_path = os.path.join(output_dir, "edges_v1_map.png")
            cv2.imwrite(map_path, grid_large)
            print(f"  V1 cluster map saved: {map_path}")
        except ImportError:
            pass

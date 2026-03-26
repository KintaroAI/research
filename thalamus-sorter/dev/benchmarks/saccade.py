"""SACCADE benchmark: baseline spatial sorting from image saccades.

Tests whether derivative-correlation embedding + clustering produces
spatially coherent topographic maps from saccade crops of a natural image.

Metrics:
    - Contiguity: fraction of adjacent pixels in the same cluster
    - Diameter: mean spatial extent of clusters (smaller = tighter)
    - K10 within 3px: % of 10 nearest embedding neighbors within 3 pixels
    - K10 within 5px: % of 10 nearest embedding neighbors within 5 pixels
    - Stability: fraction of neurons that stayed in same cluster since last report

This is the fundamental test — if spatial sorting fails, nothing else works.

Usage:
    python main.py word2vec --preset saccade_baseline -f 50000 -o output_dir

Or manually:
    python main.py word2vec --signal-source saccade -W 80 -H 80 --image K_80_g.png \\
        --saccade-step 50 --cluster-m 100 --eval -f 50000 -o output_dir
"""

import os
import json
import numpy as np

name = 'saccade'
description = 'Baseline spatial sorting from image saccade crops'


def add_args(parser):
    pass  # uses standard --image, --saccade-step, etc.


def make_signal(w, h, args):
    """Saccade signal uses the built-in image loader, not a custom tick_fn.

    Return None to signal that the benchmark uses the default image pipeline.
    The analyze() function runs post-hoc with cluster/embedding data.
    """
    return None, None


def analyze(metadata, cluster_mgr, signals, tick_counter, T, output_dir):
    """Post-training analysis: measure spatial sorting quality."""
    if cluster_mgr is None or not cluster_mgr.initialized:
        return

    n_sensory = cluster_mgr.n_sensory
    n_total = cluster_mgr.n
    w = cluster_mgr.w
    h = cluster_mgr.h
    m = cluster_mgr.m

    most_recent = cluster_mgr.cluster_ids[
        np.arange(n_total), cluster_mgr.pointers]
    sensory_ids = most_recent[:n_sensory]

    # --- Contiguity: fraction of adjacent pixel pairs in same cluster ---
    same = 0
    total = 0
    for y in range(h):
        for x in range(w):
            idx = y * w + x
            if x + 1 < w:
                total += 1
                if sensory_ids[idx] == sensory_ids[idx + 1]:
                    same += 1
            if y + 1 < h:
                total += 1
                if sensory_ids[idx] == sensory_ids[idx + w]:
                    same += 1
    contiguity = same / max(total, 1)

    # --- Diameter: mean spatial extent per cluster ---
    diameters = []
    for c in range(m):
        members = np.where(sensory_ids == c)[0]
        if len(members) < 2:
            continue
        xs = members % w
        ys = members // w
        diam = max(xs.max() - xs.min(), ys.max() - ys.min())
        diameters.append(diam)
    mean_diam = float(np.mean(diameters)) if diameters else 0

    # --- Cluster sizes ---
    sizes = np.bincount(sensory_ids, minlength=m)
    alive = int((sizes > 0).sum())
    mean_size = float(sizes[sizes > 0].mean()) if alive > 0 else 0

    # --- K10 spatial accuracy (from embeddings) ---
    k10_3px = 0.0
    k10_5px = 0.0
    if cluster_mgr._dsolver is not None:
        import torch
        emb = cluster_mgr._dsolver.get_positions()[:n_sensory]
        # Compute pairwise distances in embedding space
        # For each neuron, find 10 nearest and check spatial distance
        n = n_sensory
        batch = 1000
        within_3 = 0
        within_5 = 0
        total_neighbors = 0
        for start in range(0, n, batch):
            end = min(start + batch, n)
            chunk = emb[start:end]  # (batch, dims)
            dists = torch.cdist(chunk, emb)  # (batch, n)
            # Zero out self
            for i in range(end - start):
                dists[i, start + i] = float('inf')
            _, topk_idx = dists.topk(10, largest=False)  # (batch, 10)
            topk_idx = topk_idx.cpu().numpy()
            for i in range(end - start):
                neuron = start + i
                nx, ny = neuron % w, neuron // w
                for j in range(10):
                    nb = topk_idx[i, j]
                    bx, by = nb % w, nb // w
                    dist = max(abs(nx - bx), abs(ny - by))
                    if dist <= 3:
                        within_3 += 1
                    if dist <= 5:
                        within_5 += 1
                    total_neighbors += 1
        if total_neighbors > 0:
            k10_3px = 100.0 * within_3 / total_neighbors
            k10_5px = 100.0 * within_5 / total_neighbors

    # --- Hierarchy (if feedback enabled) ---
    hierarchy = {}
    if n_total > n_sensory:
        n_outputs = cluster_mgr.column_mgr.n_outputs if cluster_mgr.column_mgr else 4
        v1 = set(c for c in range(m) if sizes[c] > 0)
        # Count sensory vs feedback per cluster
        fb_counts = np.zeros(m, dtype=int)
        for c in range(m):
            members = np.where(most_recent == c)[0]
            fb_counts[c] = (members >= n_sensory).sum()

        v1_only = set(c for c in range(m)
                      if sizes[c] > 0 and fb_counts[c] == 0)
        # Not a full layer trace — just count V1 vs feedback clusters
        fb_clusters = set(c for c in range(m)
                         if sizes[c] == 0 and fb_counts[c] > 0)
        mixed = set(c for c in range(m)
                   if sizes[c] > 0 and fb_counts[c] > 0)
        hierarchy = {
            'V1_pure': len(v1_only),
            'feedback_only': len(fb_clusters),
            'mixed': len(mixed),
        }

    print(f"  SACCADE results:")
    print(f"    Contiguity: {contiguity:.4f}")
    print(f"    Diameter: {mean_diam:.1f}")
    print(f"    Alive: {alive}/{m}, mean size: {mean_size:.1f}")
    print(f"    K10 within 3px: {k10_3px:.1f}%")
    print(f"    K10 within 5px: {k10_5px:.1f}%")
    if hierarchy:
        print(f"    Hierarchy: {hierarchy}")

    if output_dir:
        results = {
            'contiguity': round(contiguity, 4),
            'diameter': round(mean_diam, 1),
            'alive': alive,
            'total_clusters': m,
            'mean_cluster_size': round(mean_size, 1),
            'k10_within_3px': round(k10_3px, 1),
            'k10_within_5px': round(k10_5px, 1),
            'hierarchy': hierarchy,
        }
        path = os.path.join(output_dir, "saccade_analysis.json")
        with open(path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"  SACCADE analysis saved: {path}")

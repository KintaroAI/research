#!/usr/bin/env python3
"""ts-00016: KNN hierarchy clustering experiments.

Standalone script for clustering experiments on saved DriftSolver models.
Tests k-means clustering on learned embeddings, frequency-based cluster KNN
selection, and streaming cluster maintenance.

Usage:
    python cluster_experiments.py offline --model model.npy --knn knn_lists.npy -W 80 -H 80 --m 100
    python cluster_experiments.py offline --model model.npy --knn knn_lists.npy -W 80 -H 80 --m 25,100,400,1600,6400
"""

import argparse
import numpy as np
import time
import os


# ---------------------------------------------------------------------------
# Core clustering functions
# ---------------------------------------------------------------------------

def kmeans_cluster(embeddings, m, max_iters=100, n_restarts=5, seed=42):
    """Batch k-means on (n, dims) embeddings. Returns (cluster_ids, centroids).

    Uses k-means++ initialization with multiple restarts, keeps best by inertia.
    For large m (>= n/4), uses random init with 1 restart to avoid O(n*m) init cost.
    """
    n, dims = embeddings.shape
    if m >= n:
        # Degenerate: each neuron is its own cluster (or more clusters than neurons)
        cluster_ids = np.arange(n) % m
        centroids = np.zeros((m, dims), dtype=embeddings.dtype)
        for c in range(min(m, n)):
            mask = cluster_ids == c
            if mask.any():
                centroids[c] = embeddings[mask].mean(axis=0)
        return cluster_ids, centroids

    rng = np.random.RandomState(seed)
    best_ids, best_centroids, best_inertia = None, None, float('inf')

    # Scale down restarts for large m
    use_pp = m < n // 4
    actual_restarts = n_restarts if use_pp else 1

    for restart in range(actual_restarts):
        if use_pp:
            # k-means++ init
            centroids = np.empty((m, dims), dtype=embeddings.dtype)
            centroids[0] = embeddings[rng.randint(n)]
            for j in range(1, m):
                dists = np.min(np.sum((embeddings[:, None, :] - centroids[None, :j, :]) ** 2, axis=2), axis=1)
                probs = dists / dists.sum()
                centroids[j] = embeddings[rng.choice(n, p=probs)]
        else:
            # Random init: pick m unique points
            idx = rng.choice(n, size=m, replace=False)
            centroids = embeddings[idx].copy()

        # Lloyd's iterations
        for it in range(max_iters):
            cluster_ids = _assign_clusters(embeddings, centroids)

            # Update centroids
            new_centroids = np.zeros_like(centroids)
            counts = np.zeros(m)
            for c in range(m):
                mask = cluster_ids == c
                cnt = mask.sum()
                if cnt > 0:
                    new_centroids[c] = embeddings[mask].mean(axis=0)
                    counts[c] = cnt
                else:
                    new_centroids[c] = embeddings[rng.randint(n)]
                    counts[c] = 0

            if np.allclose(new_centroids, centroids, atol=1e-6):
                centroids = new_centroids
                break
            centroids = new_centroids

        # Inertia
        diffs = embeddings - centroids[cluster_ids]
        inertia = np.sum(diffs ** 2)

        if inertia < best_inertia:
            best_inertia = inertia
            best_ids = cluster_ids.copy()
            best_centroids = centroids.copy()

    return best_ids, best_centroids


def _assign_clusters(embeddings, centroids):
    """Assign each embedding to nearest centroid. Memory-efficient chunked."""
    n = embeddings.shape[0]
    m = centroids.shape[0]
    cluster_ids = np.empty(n, dtype=np.int64)
    chunk = max(1, min(n, 50000000 // (m * 8)))  # ~50MB working memory
    for i in range(0, n, chunk):
        end = min(i + chunk, n)
        dists = np.sum((embeddings[i:end, None, :] - centroids[None, :, :]) ** 2, axis=2)
        cluster_ids[i:end] = dists.argmin(axis=1)
    return cluster_ids


def frequency_knn(knn_lists, cluster_ids, m, k2):
    """Frequency-based cluster-level KNN selection.

    For each cluster, pool all members' KNN entries, count occurrences,
    remove self-cluster entries, return top-k2 by frequency.

    Returns: knn2 (m, k2) — cluster-level KNN (neuron IDs of consensus neighbors)
    """
    n, k = knn_lists.shape
    knn2 = np.full((m, k2), -1, dtype=np.int64)
    knn2_counts = np.zeros((m, k2), dtype=np.int64)  # frequency of each selected neighbor

    for c in range(m):
        members = np.where(cluster_ids == c)[0]
        if len(members) == 0:
            continue

        # Pool all members' KNN entries
        pooled = knn_lists[members].flatten()

        # Count unique entries
        ids, counts = np.unique(pooled, return_counts=True)

        # Remove self-cluster entries
        not_self = cluster_ids[ids] != c
        ids = ids[not_self]
        counts = counts[not_self]

        if len(ids) == 0:
            continue

        # Top-k2 by frequency
        top = min(k2, len(ids))
        if top == len(ids):
            top_idx = np.argsort(-counts)[:top]
        else:
            top_idx = np.argpartition(-counts, top)[:top]
            top_idx = top_idx[np.argsort(-counts[top_idx])]
        knn2[c, :top] = ids[top_idx]
        knn2_counts[c, :top] = counts[top_idx]

    return knn2, knn2_counts


def cluster_adjacency(knn2, cluster_ids, m):
    """Build cluster-to-cluster adjacency matrix from knn2.

    Returns: adj (m, m) — edge weight = number of knn2 entries pointing to that cluster.
    """
    adj = np.zeros((m, m), dtype=np.int64)
    for c in range(m):
        valid = knn2[c] >= 0
        if not valid.any():
            continue
        neighbor_clusters = cluster_ids[knn2[c, valid]]
        for nc in neighbor_clusters:
            if nc != c:
                adj[c, nc] += 1
    return adj


# ---------------------------------------------------------------------------
# Evaluation metrics
# ---------------------------------------------------------------------------

def eval_clusters(cluster_ids, centroids, knn2, width, height, knn_lists=None):
    """Evaluate cluster quality.

    Returns dict with:
      - size_stats: min, max, mean, std of cluster sizes
      - spatial_diameter: mean/max grid diameter per cluster
      - spatial_contiguity: fraction of members within radius of cluster's grid center
      - knn2_spatial: mean grid distance between cluster center and knn2 target clusters
      - knn2_agreement: fraction of knn2 entries that appear in original KNN lists
    """
    n = len(cluster_ids)
    m = centroids.shape[0]

    # Grid coordinates
    coords = np.stack([np.arange(n) % width, np.arange(n) // width], axis=1).astype(float)

    # Cluster sizes
    sizes = np.bincount(cluster_ids, minlength=m)
    nonempty = sizes > 0

    results = {
        'n': n, 'm': m,
        'n_empty': int((sizes == 0).sum()),
        'size_min': int(sizes[nonempty].min()) if nonempty.any() else 0,
        'size_max': int(sizes[nonempty].max()),
        'size_mean': float(sizes[nonempty].mean()),
        'size_std': float(sizes[nonempty].std()),
    }

    # Spatial diameter: max pairwise grid distance within each cluster
    diameters = []
    cluster_centers = np.zeros((m, 2))
    for c in range(m):
        members = np.where(cluster_ids == c)[0]
        if len(members) == 0:
            continue
        member_coords = coords[members]
        cluster_centers[c] = member_coords.mean(axis=0)
        if len(members) == 1:
            diameters.append(0)
            continue
        # Max distance from center (faster than all-pairs)
        diffs = member_coords - cluster_centers[c]
        dists = np.sqrt((diffs ** 2).sum(axis=1))
        diameters.append(float(dists.max()) * 2)  # diameter = 2 * max_radius

    results['diameter_mean'] = float(np.mean(diameters)) if diameters else 0
    results['diameter_max'] = float(np.max(diameters)) if diameters else 0
    results['diameter_median'] = float(np.median(diameters)) if diameters else 0

    # Spatial contiguity: what fraction of members are within sqrt(n/m) of center?
    ideal_radius = np.sqrt(n / m / np.pi)  # radius of circle with area = ideal cluster size
    contiguity = []
    for c in range(m):
        members = np.where(cluster_ids == c)[0]
        if len(members) == 0:
            continue
        diffs = coords[members] - cluster_centers[c]
        dists = np.sqrt((diffs ** 2).sum(axis=1))
        frac = (dists <= ideal_radius * 2).mean()  # within 2× ideal radius
        contiguity.append(float(frac))
    results['contiguity_mean'] = float(np.mean(contiguity)) if contiguity else 0

    # knn2 spatial: grid distance between cluster centers via knn2
    if knn2 is not None:
        knn2_dists = []
        for c in range(m):
            valid = knn2[c] >= 0
            if not valid.any():
                continue
            target_clusters = cluster_ids[knn2[c, valid]]
            unique_targets = np.unique(target_clusters)
            for tc in unique_targets:
                if tc != c and sizes[tc] > 0:
                    d = np.sqrt(((cluster_centers[c] - cluster_centers[tc]) ** 2).sum())
                    knn2_dists.append(d)
        results['knn2_center_dist_mean'] = float(np.mean(knn2_dists)) if knn2_dists else 0
        results['knn2_center_dist_max'] = float(np.max(knn2_dists)) if knn2_dists else 0

    # knn2 agreement with original KNN
    if knn_lists is not None and knn2 is not None:
        hits = 0
        total = 0
        for c in range(m):
            members = np.where(cluster_ids == c)[0]
            if len(members) == 0:
                continue
            valid = knn2[c] >= 0
            knn2_set = set(knn2[c, valid].tolist())
            if not knn2_set:
                continue
            # Check: what fraction of knn2 entries appear in any member's KNN?
            member_knn_set = set(knn_lists[members].flatten().tolist())
            overlap = len(knn2_set & member_knn_set)
            hits += overlap
            total += len(knn2_set)
        results['knn2_agreement'] = hits / total if total > 0 else 0

    return results


def visualize_clusters(cluster_ids, width, height, path=None):
    """Save color-coded cluster visualization as PNG."""
    try:
        import cv2
    except ImportError:
        print("  (cv2 not available, skipping visualization)")
        return

    m = cluster_ids.max() + 1
    # Generate distinct colors via golden ratio hue spacing
    colors = np.zeros((m, 3), dtype=np.uint8)
    for i in range(m):
        hue = int((i * 137.508) % 180)  # golden angle in degrees/2 for OpenCV
        colors[i] = [hue, 200, 200]
    # Convert HSV to BGR
    hsv_img = colors.reshape(1, m, 3)
    bgr_colors = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2BGR).reshape(m, 3)

    img = bgr_colors[cluster_ids].reshape(height, width, 3)

    # Scale up for visibility
    scale = max(1, 512 // max(width, height))
    if scale > 1:
        img = cv2.resize(img, (width * scale, height * scale), interpolation=cv2.INTER_NEAREST)

    if path:
        cv2.imwrite(path, img)
        print(f"  cluster map saved: {path}")
    return img


# ---------------------------------------------------------------------------
# Phase 1: Offline k-means
# ---------------------------------------------------------------------------

def run_offline(args):
    """Run offline k-means clustering on saved model."""
    embeddings = np.load(args.model)
    knn_lists = np.load(args.knn)
    n = embeddings.shape[0]
    k = knn_lists.shape[1]
    W, H = args.width, args.height
    assert n == W * H, f"Model size {n} != grid {W}x{H}={W*H}"

    m_values = [int(x) for x in args.m.split(',')]
    k2 = args.k2 if args.k2 else k  # default: same as original KNN k

    print(f"Offline k-means: n={n} ({W}x{H}), dims={embeddings.shape[1]}, k={k}, k2={k2}")
    print(f"Testing m = {m_values}\n")

    out_dir = args.output_dir
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    all_results = []

    for m in m_values:
        print(f"--- m={m} (n/m={n/m:.1f} neurons/cluster) ---")
        t0 = time.time()
        cluster_ids, centroids = kmeans_cluster(embeddings, m, seed=args.seed)
        t_km = time.time() - t0

        t0 = time.time()
        knn2, knn2_counts = frequency_knn(knn_lists, cluster_ids, m, k2)
        t_freq = time.time() - t0

        t0 = time.time()
        metrics = eval_clusters(cluster_ids, centroids, knn2, W, H, knn_lists)
        t_eval = time.time() - t0

        metrics['m'] = m
        metrics['k2'] = k2
        metrics['t_kmeans'] = t_km
        metrics['t_frequency'] = t_freq
        all_results.append(metrics)

        print(f"  k-means: {t_km:.3f}s, freq knn: {t_freq:.3f}s, eval: {t_eval:.3f}s")
        print(f"  clusters: {m - metrics['n_empty']} non-empty ({metrics['n_empty']} empty)")
        print(f"  sizes: min={metrics['size_min']} max={metrics['size_max']} "
              f"mean={metrics['size_mean']:.1f} std={metrics['size_std']:.1f}")
        print(f"  diameter: mean={metrics['diameter_mean']:.1f} "
              f"median={metrics['diameter_median']:.1f} max={metrics['diameter_max']:.1f}")
        print(f"  contiguity: {metrics['contiguity_mean']:.3f}")
        if 'knn2_center_dist_mean' in metrics:
            print(f"  knn2 center dist: mean={metrics['knn2_center_dist_mean']:.1f} "
                  f"max={metrics['knn2_center_dist_max']:.1f}")
        if 'knn2_agreement' in metrics:
            print(f"  knn2 agreement with original KNN: {metrics['knn2_agreement']:.3f}")

        if out_dir:
            visualize_clusters(cluster_ids, W, H,
                               os.path.join(out_dir, f"clusters_m{m}.png"))
            np.save(os.path.join(out_dir, f"cluster_ids_m{m}.npy"), cluster_ids)
            np.save(os.path.join(out_dir, f"centroids_m{m}.npy"), centroids)
            np.save(os.path.join(out_dir, f"knn2_m{m}.npy"), knn2)

        print()

    # Summary table
    if len(m_values) > 1:
        print("=== Summary ===")
        print(f"{'m':>6} {'n/m':>6} {'empty':>5} {'size_std':>8} {'diam_mean':>9} "
              f"{'contig':>7} {'knn2_dist':>9} {'knn2_agr':>8} {'t_km':>6}")
        for r in all_results:
            print(f"{r['m']:>6} {r['n']/r['m']:>6.1f} {r['n_empty']:>5} "
                  f"{r['size_std']:>8.1f} {r['diameter_mean']:>9.1f} "
                  f"{r['contiguity_mean']:>7.3f} "
                  f"{r.get('knn2_center_dist_mean', 0):>9.1f} "
                  f"{r.get('knn2_agreement', 0):>8.3f} "
                  f"{r['t_kmeans']:>6.2f}s")


# ---------------------------------------------------------------------------
# Phase 2: Streaming from converged state
# ---------------------------------------------------------------------------

def streaming_update(embeddings, centroids, cluster_ids, anchors, threshold,
                     lr=0.01, sizes=None):
    """Single streaming update step.

    For each anchor:
      1. Compute distance to current centroid
      2. If distance > threshold: find nearest centroid, reassign
      3. Nudge centroids of affected clusters

    Returns: n_reassigned, affected_clusters
    """
    m = centroids.shape[0]
    n = embeddings.shape[0]

    if sizes is None:
        sizes = np.bincount(cluster_ids, minlength=m)

    anchor_emb = embeddings[anchors]                          # (batch, dims)
    cur_cluster = cluster_ids[anchors]                        # (batch,)
    cur_centroid = centroids[cur_cluster]                     # (batch, dims)
    cur_dist = np.sqrt(((anchor_emb - cur_centroid) ** 2).sum(axis=1))  # (batch,)

    # Find anchors that drifted past threshold
    drifted_mask = cur_dist > threshold
    drifted_idx = np.where(drifted_mask)[0]

    n_reassigned = 0
    affected = set()

    if len(drifted_idx) > 0:
        drifted_emb = anchor_emb[drifted_idx]                # (drifted, dims)
        # Find nearest centroid for each drifted anchor
        all_dists = np.sum((drifted_emb[:, None, :] - centroids[None, :, :]) ** 2, axis=2)
        new_cluster = all_dists.argmin(axis=1)                # (drifted,)
        old_cluster = cur_cluster[drifted_idx]

        for i in range(len(drifted_idx)):
            oc, nc = old_cluster[i], new_cluster[i]
            if nc != oc:
                neuron = anchors[drifted_idx[i]]
                cluster_ids[neuron] = nc
                sizes[oc] -= 1
                sizes[nc] += 1
                affected.add(oc)
                affected.add(nc)
                n_reassigned += 1

    # Nudge centroids for all clusters that have anchors
    anchor_clusters = np.unique(cluster_ids[anchors])
    for c in anchor_clusters:
        members = np.where(cluster_ids == c)[0]
        if len(members) > 0:
            member_mean = embeddings[members].mean(axis=0)
            centroids[c] += lr * (member_mean - centroids[c])

    return n_reassigned, affected, sizes


def cluster_agreement(ids_a, ids_b, n, m):
    """Measure agreement between two clusterings using Adjusted Rand Index.

    Also returns simpler metric: fraction of neurons in the same cluster as
    their nearest neighbor under clustering A that are also in the same cluster
    under clustering B.
    """
    # Build co-assignment matrix for both clusterings
    # For each pair of neurons, do they agree on same/different cluster?
    # Full ARI is O(n²), so use a sampling approach for large n
    sample_size = min(n, 5000)
    rng = np.random.RandomState(0)
    sample = rng.choice(n, size=sample_size, replace=False)

    agree = 0
    total = 0
    for i in range(0, len(sample), 100):
        batch = sample[i:i+100]
        for j in range(i+100, len(sample)):
            s_a = ids_a[batch] == ids_a[sample[j]]
            s_b = ids_b[batch] == ids_b[sample[j]]
            agree += (s_a == s_b).sum()
            total += len(batch)

    return agree / total if total > 0 else 0


def run_streaming(args):
    """Phase 2: streaming cluster updates from random init on converged embeddings."""
    embeddings = np.load(args.model)
    knn_lists = np.load(args.knn)
    n, dims = embeddings.shape
    k = knn_lists.shape[1]
    W, H = args.width, args.height
    m = args.m
    k2 = args.k2 if args.k2 else k
    assert n == W * H

    print(f"Streaming clustering: n={n} ({W}x{H}), m={m}, dims={dims}, k2={k2}")
    print(f"  batch_size={args.batch_size}, threshold={args.threshold}, "
          f"lr={args.lr}, iters={args.iters}")

    out_dir = args.output_dir
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    # Offline baseline for comparison
    print("\nComputing offline baseline...")
    baseline_ids, baseline_centroids = kmeans_cluster(embeddings, m, seed=42)
    baseline_knn2, _ = frequency_knn(knn_lists, baseline_ids, m, k2)
    baseline_metrics = eval_clusters(baseline_ids, baseline_centroids, baseline_knn2,
                                     W, H, knn_lists)
    print(f"  baseline: contiguity={baseline_metrics['contiguity_mean']:.3f} "
          f"diam={baseline_metrics['diameter_mean']:.1f} "
          f"knn2_agr={baseline_metrics.get('knn2_agreement', 0):.3f}")

    # Random initialization
    rng = np.random.RandomState(args.seed)
    idx = rng.choice(n, size=m, replace=False)
    centroids = embeddings[idx].copy()
    cluster_ids = _assign_clusters(embeddings, centroids)
    sizes = np.bincount(cluster_ids, minlength=m)

    print(f"\nRandom init: {(sizes == 0).sum()} empty clusters, "
          f"size range [{sizes[sizes>0].min()}, {sizes.max()}]")

    # Streaming loop
    print(f"\n{'iter':>6} {'reassigned':>10} {'affected':>8} {'empty':>5} "
          f"{'size_std':>8} {'diam':>6} {'contig':>7} {'knn2_agr':>8} {'agree':>6}")

    report_every = max(1, args.iters // 20)

    for it in range(args.iters):
        # Sample anchors (simulate per-tick anchor batch)
        anchors = rng.choice(n, size=args.batch_size, replace=False)

        n_reassigned, affected, sizes = streaming_update(
            embeddings, centroids, cluster_ids, anchors,
            threshold=args.threshold, lr=args.lr, sizes=sizes)

        if it % report_every == 0 or it == args.iters - 1:
            knn2, _ = frequency_knn(knn_lists, cluster_ids, m, k2)
            metrics = eval_clusters(cluster_ids, centroids, knn2, W, H, knn_lists)
            agree = cluster_agreement(cluster_ids, baseline_ids, n, m)
            print(f"{it:>6} {n_reassigned:>10} {len(affected):>8} "
                  f"{metrics['n_empty']:>5} {metrics['size_std']:>8.1f} "
                  f"{metrics['diameter_mean']:>6.1f} {metrics['contiguity_mean']:>7.3f} "
                  f"{metrics.get('knn2_agreement', 0):>8.3f} {agree:>6.3f}")

    # Final eval
    print("\n--- Final ---")
    knn2, knn2_counts = frequency_knn(knn_lists, cluster_ids, m, k2)
    final_metrics = eval_clusters(cluster_ids, centroids, knn2, W, H, knn_lists)
    agree = cluster_agreement(cluster_ids, baseline_ids, n, m)

    print(f"  clusters: {m - final_metrics['n_empty']} non-empty "
          f"({final_metrics['n_empty']} empty)")
    print(f"  sizes: min={final_metrics['size_min']} max={final_metrics['size_max']} "
          f"mean={final_metrics['size_mean']:.1f} std={final_metrics['size_std']:.1f}")
    print(f"  diameter: mean={final_metrics['diameter_mean']:.1f} "
          f"median={final_metrics['diameter_median']:.1f}")
    print(f"  contiguity: {final_metrics['contiguity_mean']:.3f}")
    print(f"  knn2 agreement: {final_metrics.get('knn2_agreement', 0):.3f}")
    print(f"  clustering agreement with baseline: {agree:.3f}")

    if out_dir:
        visualize_clusters(cluster_ids, W, H,
                           os.path.join(out_dir, f"clusters_streaming_m{m}.png"))
        visualize_clusters(baseline_ids, W, H,
                           os.path.join(out_dir, f"clusters_baseline_m{m}.png"))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="KNN hierarchy clustering experiments (ts-00016)")
    sub = parser.add_subparsers(dest='command')

    # Offline
    p_off = sub.add_parser('offline', help='Phase 1: offline k-means on saved model')
    p_off.add_argument('--model', required=True, help='Path to model.npy (n, dims)')
    p_off.add_argument('--knn', required=True, help='Path to knn_lists.npy (n, k)')
    p_off.add_argument('-W', '--width', type=int, required=True)
    p_off.add_argument('-H', '--height', type=int, required=True)
    p_off.add_argument('--m', type=str, default='100', help='Comma-separated m values to test')
    p_off.add_argument('--k2', type=int, default=None, help='KNN size for clusters (default: same as k)')
    p_off.add_argument('--seed', type=int, default=42)
    p_off.add_argument('-o', '--output-dir', type=str, default=None)

    # Streaming
    p_str = sub.add_parser('streaming', help='Phase 2: streaming updates from random init')
    p_str.add_argument('--model', required=True, help='Path to model.npy (n, dims)')
    p_str.add_argument('--knn', required=True, help='Path to knn_lists.npy (n, k)')
    p_str.add_argument('-W', '--width', type=int, required=True)
    p_str.add_argument('-H', '--height', type=int, required=True)
    p_str.add_argument('--m', type=int, default=100)
    p_str.add_argument('--k2', type=int, default=None)
    p_str.add_argument('--batch-size', type=int, default=256,
                       help='Anchors per iteration (simulates per-tick batch)')
    p_str.add_argument('--threshold', type=float, default=0.5,
                       help='Distance threshold for reassignment')
    p_str.add_argument('--lr', type=float, default=0.1,
                       help='Centroid nudge learning rate')
    p_str.add_argument('--iters', type=int, default=200,
                       help='Number of streaming iterations')
    p_str.add_argument('--seed', type=int, default=42)
    p_str.add_argument('-o', '--output-dir', type=str, default=None)

    args = parser.parse_args()
    if args.command == 'offline':
        run_offline(args)
    elif args.command == 'streaming':
        run_streaming(args)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()

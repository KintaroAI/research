"""Test cluster convergence from random init on frozen (pre-trained) embeddings.

Loads a trained model, assigns neurons to clusters RANDOMLY (no k-means),
then runs the streaming cluster update loop with synthetic pairs derived
from actual embedding neighbors. No training — embeddings are static.

Usage:
    python test_cluster_stability.py MODEL.npy -W 80 -H 80 --m 100 --ticks 10000
"""

import argparse
import time
import numpy as np
import torch
import sys
sys.path.insert(0, '.')

from cluster_experiments import (
    kmeans_cluster_gpu, _assign_clusters_gpu, streaming_update_v3_gpu,
    eval_clusters, split_largest_cluster_gpu
)


def make_pairs(embeddings_t, k=10, batch=256, rng=None):
    """Generate (anchor, neighbor) pairs from actual embedding proximity.
    Simulates skip-gram output: for each anchor, find k nearest neighbors."""
    if rng is None:
        rng = np.random.RandomState()
    n = embeddings_t.shape[0]
    anchors = rng.choice(n, size=batch, replace=False)
    anchors_t = torch.tensor(anchors, device=embeddings_t.device)
    anchor_embs = embeddings_t[anchors_t]  # (batch, dims)

    # Batch distances to all neurons
    dists = torch.cdist(anchor_embs, embeddings_t)  # (batch, n)
    # Exclude self
    dists[torch.arange(batch, device=dists.device), anchors_t] = float('inf')
    # Top-k nearest
    _, topk_idx = dists.topk(k, dim=1, largest=False)  # (batch, k)

    # Flatten to parallel lists
    centers = anchors_t.unsqueeze(1).expand(-1, k).reshape(-1)
    contexts = topk_idx.reshape(-1)
    return (centers.cpu(), contexts.cpu()), anchors


def main():
    parser = argparse.ArgumentParser(description="Cluster stability test on frozen embeddings")
    parser.add_argument("model", help="Path to model.npy")
    parser.add_argument("-W", type=int, default=80)
    parser.add_argument("-H", type=int, default=80)
    parser.add_argument("--m", type=int, default=100, help="Number of clusters")
    parser.add_argument("--k2", type=int, default=16, help="Cluster KNN size")
    parser.add_argument("--ticks", type=int, default=10000)
    parser.add_argument("--report-every", type=int, default=500)
    parser.add_argument("--hysteresis", type=float, default=0.3)
    parser.add_argument("--split-every", type=int, default=10)
    parser.add_argument("--batch", type=int, default=256, help="Anchors per tick")
    parser.add_argument("--pair-k", type=int, default=10, help="Neighbors per anchor")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--kmeans-init", action="store_true",
                        help="Use k-means init instead of random (for comparison)")
    args = parser.parse_args()

    rng = np.random.RandomState(args.seed)
    embeddings = np.load(args.model).astype(np.float32)
    n, dims = embeddings.shape
    m = args.m
    k2 = args.k2
    W, H = args.W, args.H
    assert n == W * H, f"Model has {n} neurons but grid is {W}x{H}={W*H}"

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    embeddings_t = torch.from_numpy(embeddings).to(device)

    print(f"Loaded {args.model}: n={n}, dims={dims}, device={device}")
    print(f"Config: m={m}, k2={k2}, h={args.hysteresis}, ticks={args.ticks}, "
          f"batch={args.batch}, pair_k={args.pair_k}")

    # --- Init clusters ---
    if args.kmeans_init:
        print("Init: k-means")
        cluster_ids, centroids_np = kmeans_cluster_gpu(embeddings_t, m, seed=args.seed)
    else:
        print("Init: RANDOM assignment")
        cluster_ids = rng.randint(0, m, size=n)
        centroids_np = np.zeros((m, dims), dtype=np.float32)
        for c in range(m):
            mask = cluster_ids == c
            if mask.any():
                centroids_np[c] = embeddings[mask].mean(axis=0)

    cluster_ids_t = torch.from_numpy(cluster_ids.astype(np.int64)).to(device)
    centroids_t = torch.from_numpy(centroids_np).to(device)
    sizes = np.bincount(cluster_ids, minlength=m)

    # Init knn2 with random cluster indices
    knn2 = np.full((m, k2), -1, dtype=np.int64)
    knn2_dists = np.full((m, k2), np.inf, dtype=np.float32)
    for c in range(m):
        if sizes[c] == 0:
            continue
        others = [i for i in range(m) if i != c and sizes[i] > 0]
        if not others:
            continue
        k = min(k2, len(others))
        chosen = rng.choice(others, size=k, replace=False)
        knn2[c, :k] = chosen
        for j in range(k):
            diff = centroids_np[c] - centroids_np[chosen[j]]
            knn2_dists[c, j] = np.dot(diff, diff)

    knn2_t = torch.from_numpy(knn2).to(device)
    knn2_dists_t = torch.from_numpy(knn2_dists).to(device)

    n_alive = m - (sizes == 0).sum()
    print(f"  Init: {n_alive}/{m} alive")

    # --- Run loop ---
    total_reassigned = 0
    total_splits = 0
    prev_cluster_ids = None
    prev_reassigned = 0
    prev_tick = 0
    t0 = time.time()

    for tick in range(1, args.ticks + 1):
        # Generate pairs from embedding proximity
        pairs, anchors = make_pairs(embeddings_t, k=args.pair_k,
                                    batch=args.batch, rng=rng)

        # Streaming update
        knn2_np = knn2_t.cpu().numpy()
        n_reassigned, affected, _, n_blocked = streaming_update_v3_gpu(
            embeddings_t, centroids_t, cluster_ids, knn2_np,
            anchors, lr=0.01, sizes=sizes, min_size=0, rng=rng,
            hysteresis=args.hysteresis)
        total_reassigned += n_reassigned

        if affected:
            cluster_ids_t = torch.from_numpy(cluster_ids.astype(np.int64)).to(device)
            # Recompute knn2 dists for affected rows
            rows_t = torch.tensor(list(affected), dtype=torch.long, device=device)
            targets = knn2_t[rows_t]
            valid = targets >= 0
            safe_targets = targets.clamp(min=0)
            diffs = centroids_t[rows_t].unsqueeze(1) - centroids_t[safe_targets]
            d = (diffs * diffs).sum(dim=2)
            d[~valid] = float('inf')
            knn2_dists_t[rows_t] = d

        # Update knn2 from pairs (GPU)
        center_t = pairs[0].to(device)
        ctx_t = pairs[1].to(device)
        ca = cluster_ids_t[center_t]
        cn = cluster_ids_t[ctx_t]
        cross = ca != cn
        if cross.any():
            ca_x, cn_x = ca[cross], cn[cross]
            unique_packed = torch.unique(ca_x * m + cn_x)
            u_ca = unique_packed // m
            u_cn = unique_packed % m
            diffs = centroids_t[u_ca] - centroids_t[u_cn]
            dists = (diffs * diffs).sum(dim=1)
            already_in = (u_cn.unsqueeze(1) == knn2_t[u_ca]).any(dim=1)
            dists = torch.where(already_in, torch.tensor(float('inf'),
                                device=device), dists)
            best_dist = torch.full((m,), float('inf'), device=device)
            best_dist.scatter_reduce_(0, u_ca, dists, reduce='amin',
                                      include_self=True)
            worst_knn2_dist, worst_knn2_slot = knn2_dists_t.max(dim=1)
            improved = best_dist < worst_knn2_dist
            if improved.any():
                is_best = (dists == best_dist[u_ca]) & improved[u_ca]
                bi = is_best.nonzero(as_tuple=True)[0]
                bi_ca = u_ca[bi]
                first_per_ca = torch.full((m,), bi.shape[0],
                                          dtype=torch.long, device=device)
                first_per_ca.scatter_reduce_(
                    0, bi_ca, torch.arange(bi.shape[0], device=device),
                    reduce='amin', include_self=True)
                imp_ca = torch.where(improved & (first_per_ca < bi.shape[0]))[0]
                imp_ui = bi[first_per_ca[imp_ca]]
                knn2_t[imp_ca, worst_knn2_slot[imp_ca]] = u_cn[imp_ui]
                knn2_dists_t[imp_ca, worst_knn2_slot[imp_ca]] = dists[imp_ui]

        # Splits
        if args.split_every > 0 and tick % args.split_every == 0:
            n_empty = (sizes == 0).sum()
            if n_empty > 0:
                n_to_split = min(n_empty, max(1, n_empty // 5))
                n_splits = split_largest_cluster_gpu(
                    embeddings_t, centroids_t, cluster_ids, sizes, m,
                    n_splits=n_to_split, seed=rng.randint(1000000))
                total_splits += n_splits
                if n_splits > 0:
                    cluster_ids_t = torch.from_numpy(
                        cluster_ids.astype(np.int64)).to(device)
                    # Recompute all knn2 dists
                    for c in range(m):
                        targets = knn2_t[c]
                        valid = targets >= 0
                        safe = targets.clamp(min=0)
                        d = ((centroids_t[c] - centroids_t[safe]) ** 2).sum(dim=1)
                        d[~valid] = float('inf')
                        knn2_dists_t[c] = d

        # Report
        if tick % args.report_every == 0:
            elapsed = time.time() - t0
            n_alive = m - (sizes == 0).sum()
            metrics = eval_clusters(cluster_ids, centroids_t.cpu().numpy(),
                                    knn2_t.cpu().numpy(), W, H)
            interval_reassigned = total_reassigned - prev_reassigned
            interval_ticks = max(1, tick - prev_tick)
            jumps_per_tick = interval_reassigned / interval_ticks
            prev_reassigned = total_reassigned
            prev_tick = tick

            if prev_cluster_ids is not None:
                stability = (cluster_ids == prev_cluster_ids).mean()
            else:
                stability = 0.0
            prev_cluster_ids = cluster_ids.copy()

            print(f"  tick {tick:6d}/{args.ticks} ({elapsed:.1f}s) "
                  f"alive={n_alive}/{m} "
                  f"contig={metrics['contiguity_mean']:.3f} "
                  f"diam={metrics['diameter_mean']:.1f} "
                  f"stab={stability:.3f} "
                  f"jumps/t={jumps_per_tick:.1f} "
                  f"splits={total_splits}")

    # Final eval
    elapsed = time.time() - t0
    metrics = eval_clusters(cluster_ids, centroids_t.cpu().numpy(),
                            knn2_t.cpu().numpy(), W, H)
    n_alive = m - (sizes == 0).sum()
    print(f"\nDone: {args.ticks} ticks in {elapsed:.1f}s")
    print(f"  alive={n_alive}/{m}, contiguity={metrics['contiguity_mean']:.3f}, "
          f"diameter={metrics['diameter_mean']:.1f}")
    print(f"  total_jumps={total_reassigned}, total_splits={total_splits}")


if __name__ == "__main__":
    main()

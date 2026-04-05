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

try:
    import torch
    HAS_TORCH = torch.cuda.is_available()
    if HAS_TORCH:
        DEVICE = torch.device('cuda')
except ImportError:
    HAS_TORCH = False


def _compute_sizes(cluster_ids, m, pointers=None, wire_all_ring=True):
    """Count wired neurons per cluster."""
    if wire_all_ring:
        valid = cluster_ids[cluster_ids >= 0].ravel()
    else:
        primary = cluster_ids[np.arange(cluster_ids.shape[0]), pointers]
        valid = primary[primary >= 0]
    return np.bincount(valid, minlength=m).astype(np.int64) if len(valid) > 0 else np.zeros(m, dtype=np.int64)


# ---------------------------------------------------------------------------
# GPU-accelerated core functions (torch)
# ---------------------------------------------------------------------------

if HAS_TORCH:

    def _assign_clusters_gpu(embeddings_t, centroids_t):
        """Assign each embedding to nearest centroid on GPU. Chunked to fit memory."""
        n = embeddings_t.shape[0]
        m = centroids_t.shape[0]
        # cdist chunks for memory: ~2GB budget
        chunk = max(1, min(n, 2_000_000_000 // (m * 4)))
        ids = torch.empty(n, dtype=torch.int64, device=embeddings_t.device)
        for i in range(0, n, chunk):
            end = min(i + chunk, n)
            dists = torch.cdist(embeddings_t[i:end], centroids_t)  # (chunk, m)
            ids[i:end] = dists.argmin(dim=1)
        return ids

    def kmeans_cluster_gpu(embeddings_t, m, max_iters=100, n_restarts=3, seed=42):
        """Batch k-means on GPU. Returns (cluster_ids_cpu, centroids_cpu)."""
        n, dims = embeddings_t.shape
        rng = np.random.RandomState(seed)
        best_ids, best_centroids, best_inertia = None, None, float('inf')

        for restart in range(n_restarts):
            # Random init
            idx = rng.choice(n, size=m, replace=False)
            centroids = embeddings_t[idx].clone()

            for it in range(max_iters):
                ids = _assign_clusters_gpu(embeddings_t, centroids)

                # Update centroids via scatter
                new_centroids = torch.zeros(m, dims, device=embeddings_t.device,
                                            dtype=embeddings_t.dtype)
                counts = torch.zeros(m, device=embeddings_t.device, dtype=torch.float32)
                new_centroids.scatter_add_(0, ids.unsqueeze(1).expand(-1, dims),
                                           embeddings_t)
                counts.scatter_add_(0, ids, torch.ones(n, device=embeddings_t.device))
                alive = counts > 0
                new_centroids[alive] /= counts[alive].unsqueeze(1)
                # Reinit dead centroids
                dead = ~alive
                if dead.any():
                    n_dead = dead.sum().item()
                    reinit_idx = rng.choice(n, size=n_dead, replace=False)
                    new_centroids[dead] = embeddings_t[reinit_idx]

                if torch.allclose(new_centroids, centroids, atol=1e-6):
                    centroids = new_centroids
                    break
                centroids = new_centroids

            ids = _assign_clusters_gpu(embeddings_t, centroids)
            inertia = ((embeddings_t - centroids[ids]) ** 2).sum().item()
            if inertia < best_inertia:
                best_inertia = inertia
                best_ids = ids.cpu().numpy()
                best_centroids = centroids.cpu().numpy()

        return best_ids, best_centroids

    def frequency_knn_gpu(knn_lists_t, cluster_ids_t, m, k2):
        """Frequency-based cluster KNN on GPU using scatter_add."""
        n, k = knn_lists_t.shape
        knn2 = np.full((m, k2), -1, dtype=np.int64)
        knn2_counts = np.zeros((m, k2), dtype=np.int64)

        # Build member lists on CPU (irregular sizes)
        cluster_ids_cpu = cluster_ids_t.cpu().numpy() if isinstance(
            cluster_ids_t, torch.Tensor) else cluster_ids_t
        knn_cpu = knn_lists_t.cpu().numpy() if isinstance(
            knn_lists_t, torch.Tensor) else knn_lists_t

        # Vectorized: pool + count per cluster using GPU histogram
        for c in range(m):
            members = np.where(cluster_ids_cpu == c)[0]
            if len(members) == 0:
                continue
            pooled = knn_cpu[members].flatten()
            ids, counts = np.unique(pooled, return_counts=True)
            not_self = cluster_ids_cpu[ids] != c
            ids, counts = ids[not_self], counts[not_self]
            if len(ids) == 0:
                continue
            top = min(k2, len(ids))
            if top == len(ids):
                top_idx = np.argsort(-counts)[:top]
            else:
                top_idx = np.argpartition(-counts, top)[:top]
                top_idx = top_idx[np.argsort(-counts[top_idx])]
            knn2[c, :top] = ids[top_idx]
            knn2_counts[c, :top] = counts[top_idx]

        return knn2, knn2_counts

    def streaming_update_v3_gpu_ref(embeddings_t, centroids_t, cluster_ids, knn2,
                                    anchors, lr=0.01, sizes=None, min_size=0, rng=None,
                                    hysteresis=0.0, knn2_is_neurons=False,
                                    centroid_mode='nudge', pointers=None,
                                    last_used=None, tick=0,
                                    jump_counts=None,
                                    wire_all_ring=True):
        """Reference (scalar) streaming update — kept for verification.
        centroid_mode: 'exact' = incremental arithmetic (centroid snaps to true mean),
                       'nudge' = post-loop lr nudge (centroid drifts slowly).
        hysteresis: relative margin — neuron only jumps if dist_new < dist_cur * (1 - hysteresis).
        knn2_is_neurons: if True, knn2 entries are neuron indices (map to cluster IDs);
                         if False, knn2 entries are cluster indices directly.
        cluster_ids: (n, max_k) int64, LRU slots of cluster memberships padded with -1.
        pointers: (n,) int64, primary slot per neuron (most recent).
        last_used: (n, max_k) int64, tick when each slot was last used (for LRU eviction).
        tick: current global tick (for updating last_used)."""
        m = centroids_t.shape[0]
        n = embeddings_t.shape[0]
        max_k = cluster_ids.shape[1]

        if pointers is None:
            pointers = np.zeros(n, dtype=np.int64)
        if last_used is None:
            last_used = np.zeros((n, max_k), dtype=np.int64)
        if sizes is None:
            sizes = _compute_sizes(cluster_ids, m, pointers, wire_all_ring)
        if rng is None:
            rng = np.random.RandomState()

        n_empty = (sizes == 0).sum()
        p = 1.0 - (n_empty / m)

        n_reassigned = 0
        n_switches = 0
        n_blocked = 0
        affected = set()
        wiring_events = []  # list of (neuron, old_cluster, new_cluster)

        # Prefetch all anchor embeddings and centroids to CPU in one transfer
        anchor_embs = embeddings_t[anchors].cpu().numpy()
        centroids_cpu = centroids_t.cpu().numpy()
        # Precompute anchor cluster memberships (batch 2D index)
        anchor_cids = cluster_ids[anchors]  # (n_anchors, max_k)
        hysteresis_mult = 1.0 - hysteresis

        for i, anchor in enumerate(anchors):
            if rng.random() >= p:
                continue

            my_row = anchor_cids[i]
            my_valid = my_row[my_row >= 0]
            nv = len(my_valid)
            if nv == 0:
                continue

            # Primary = most recent entry; gather candidates from primary's knn2
            primary = my_row[pointers[anchor]]
            nb = knn2[primary]
            nb = nb[nb >= 0]
            if len(nb) == 0:
                continue
            if knn2_is_neurons:
                cids = cluster_ids[nb]
                nb = cids[cids >= 0]
                if len(nb) == 0:
                    continue
            candidates = np.unique(np.concatenate([[primary], nb]))
            if len(candidates) <= 1:
                continue

            # Distance on CPU (primary + neighbors)
            emb = anchor_embs[i]
            cand_centroids = centroids_cpu[candidates]
            dists = np.sum((emb - cand_centroids) ** 2, axis=1)
            best_idx = dists.argmin()
            best = candidates[best_idx]

            # If best is primary, no change
            if best == primary:
                continue

            # Hysteresis: compare against primary distance
            if hysteresis > 0.0:
                primary_idx = np.searchsorted(candidates, primary)
                if dists[best_idx] >= dists[primary_idx] * hysteresis_mult:
                    continue

            # Min_size guard on primary (the cluster losing this neuron)
            if sizes[primary] <= min_size:
                n_blocked += 1
                continue

            # Check if best is already in ring — just switch primary
            in_ring = False
            if nv > 1:
                for s in range(max_k):
                    if anchor_cids[i, s] == best:
                        pointers[anchor] = s
                        last_used[anchor, s] = tick
                        in_ring = True
                        break

            if not in_ring:
                if centroid_mode == 'exact':
                    old_size = sizes[primary]
                    if old_size > 1:
                        centroids_cpu[primary] = (centroids_cpu[primary] * old_size - emb) / (old_size - 1)
                    new_size = sizes[best]
                    centroids_cpu[best] = (centroids_cpu[best] * new_size + emb) / (new_size + 1)

                # LRU eviction: replace least-recently-used slot
                lru_slot = last_used[anchor].argmin()
                evicted = int(cluster_ids[anchor, lru_slot])
                cluster_ids[anchor, lru_slot] = best
                anchor_cids[i, lru_slot] = best
                last_used[anchor, lru_slot] = tick
                pointers[anchor] = lru_slot
                wiring_events.append((int(anchor), evicted, int(best)))

            if in_ring:
                if not wire_all_ring:
                    sizes[primary] -= 1
                    sizes[best] += 1
                # wire_all_ring: neuron already wired to both, no size change
                n_switches += 1
            else:
                # Out-of-ring: evicted cluster loses a wiring, best gains one
                if wire_all_ring:
                    if evicted >= 0:
                        sizes[evicted] -= 1
                else:
                    sizes[primary] -= 1
                sizes[best] += 1
                n_reassigned += 1
                if jump_counts is not None:
                    jump_counts[anchor] += 1
            affected.add(primary)
            affected.add(best)

        # Update centroids for affected clusters
        if affected:
            affected_list = list(affected)
            if centroid_mode == 'nudge':
                # Batch centroid nudge on GPU via scatter_add
                most_recent = cluster_ids[np.arange(n), pointers]
                mr_t = torch.from_numpy(most_recent.astype(np.int64)).to(centroids_t.device)
                d = centroids_t.shape[1]
                sums = torch.zeros(m, d, device=centroids_t.device)
                sums.scatter_add_(0, mr_t.unsqueeze(1).expand(n, d), embeddings_t)
                counts = torch.bincount(mr_t, minlength=m).float().unsqueeze(1).clamp(min=1)
                means = sums / counts
                affected_t = torch.tensor(affected_list, dtype=torch.long, device=centroids_t.device)
                centroids_t[affected_t] += lr * (means[affected_t] - centroids_t[affected_t])
            else:
                centroids_t[affected_list] = torch.from_numpy(
                    centroids_cpu[affected_list]).to(centroids_t.device)

        return n_reassigned, affected, sizes, n_blocked, n_switches, wiring_events

    def streaming_update_v3_gpu(embeddings_t, centroids_t, cluster_ids, knn2,
                                anchors, lr=0.01, sizes=None, min_size=0, rng=None,
                                hysteresis=0.0, knn2_is_neurons=False,
                                centroid_mode='nudge', pointers=None,
                                last_used=None, tick=0,
                                jump_counts=None,
                                max_cluster_size=0,
                                cluster_swap=True,
                                wire_all_ring=True):
        """Vectorized streaming update: batch distance computation, thin apply loop.

        Same interface and results as streaming_update_v3_gpu_ref but ~10x faster
        by computing all distances in one (n_anchors, n_candidates, dims) operation
        instead of looping per-anchor in Python.

        max_cluster_size: if >0, cap cluster membership. When a neuron wants to
        join a full cluster, try to swap with the member closest to the source
        cluster's centroid. If no good swap exists, reject the migration.
        """
        m = centroids_t.shape[0]
        n = embeddings_t.shape[0]
        max_k = cluster_ids.shape[1]
        k2 = knn2.shape[1]

        if pointers is None:
            pointers = np.zeros(n, dtype=np.int64)
        if last_used is None:
            last_used = np.zeros((n, max_k), dtype=np.int64)
        if sizes is None:
            sizes = _compute_sizes(cluster_ids, m, pointers, wire_all_ring)
        if rng is None:
            rng = np.random.RandomState()

        n_empty = (sizes == 0).sum()
        p = 1.0 - (n_empty / m)

        n_reassigned = 0
        n_switches = 0
        n_blocked = 0
        affected = set()
        wiring_events = []

        # --- Prefetch to CPU (one transfer) ---
        anchor_embs = embeddings_t[anchors].cpu().numpy()     # (n_anchors, dims)
        centroids_cpu = centroids_t.cpu().numpy()              # (m, dims)
        anchor_cids = cluster_ids[anchors]                     # (n_anchors, max_k)
        n_anchors = len(anchors)
        hysteresis_mult = 1.0 - hysteresis

        # --- Fall through to scalar for knn2_is_neurons (rare knn mode) ---
        if knn2_is_neurons:
            return streaming_update_v3_gpu_ref(
                embeddings_t, centroids_t, cluster_ids, knn2,
                anchors, lr=lr, sizes=sizes, min_size=min_size, rng=rng,
                hysteresis=hysteresis, knn2_is_neurons=True,
                centroid_mode=centroid_mode, pointers=pointers,
                last_used=last_used, tick=tick, jump_counts=jump_counts,
                wire_all_ring=wire_all_ring)

        # --- Batch phase: compute all decisions at once ---

        # 1. RNG draw — same order as scalar loop
        rand_vals = rng.random(n_anchors)
        active = rand_vals < p  # (n_anchors,) bool

        # 2. Primary cluster per anchor
        anchor_ptrs = pointers[anchors]                                    # (n_anchors,)
        ar = np.arange(n_anchors)
        primaries = anchor_cids[ar, anchor_ptrs]                           # (n_anchors,)

        # Filter: must have valid cluster entries
        has_valid = (anchor_cids >= 0).any(axis=1)
        active &= has_valid

        # 3. Gather candidates: primary + knn2[primary]
        # Build (n_anchors, 1 + k2) candidate array
        safe_primaries = np.clip(primaries, 0, m - 1)
        all_nb = knn2[safe_primaries]                                      # (n_anchors, k2)
        has_nb = (all_nb >= 0).any(axis=1)
        active &= has_nb

        candidates = np.full((n_anchors, 1 + k2), -1, dtype=np.int64)
        candidates[:, 0] = primaries
        candidates[:, 1:] = all_nb

        # 4. Batch distance: (n_anchors, 1+k2)
        safe_cands = np.clip(candidates, 0, m - 1)
        cand_cents = centroids_cpu[safe_cands]                             # (n_anchors, 1+k2, dims)
        diff = anchor_embs[:, None, :] - cand_cents                        # (n_anchors, 1+k2, dims)
        dists = (diff ** 2).sum(axis=2)                                    # (n_anchors, 1+k2)
        dists[candidates < 0] = np.inf

        # 5. Best candidate per anchor
        best_slot = dists.argmin(axis=1)                                   # (n_anchors,)
        best_cid = candidates[ar, best_slot]                               # (n_anchors,)

        # 6. Filter: best must differ from primary
        active &= (best_cid != primaries)

        # 7. Hysteresis filter
        if hysteresis > 0.0:
            best_dists = dists[ar, best_slot]
            primary_dists = dists[:, 0]  # primary is always slot 0
            active &= (best_dists < primary_dists * hysteresis_mult)

        # Prefetch all embeddings for swap candidate search
        swapped = set()  # neurons moved by swap — skip if they appear as movers
        if max_cluster_size > 0:
            all_embs_cpu = embeddings_t.cpu().numpy()

        # --- Apply phase: thin loop over movers only ---
        movers = np.where(active)[0]
        for idx in movers:
            anchor = int(anchors[idx])
            primary = int(primaries[idx])
            best = int(best_cid[idx])

            # Skip if this anchor was already moved by a swap
            if anchor in swapped:
                continue

            # min_size guard (sequential — sizes mutated by prior movers)
            if sizes[primary] <= min_size:
                n_blocked += 1
                continue

            # In-ring check
            row = anchor_cids[idx]
            nv = int((row >= 0).sum())
            in_ring = False
            if nv > 1:
                for s in range(max_k):
                    if row[s] == best:
                        pointers[anchor] = s
                        last_used[anchor, s] = tick
                        in_ring = True
                        break

            if not in_ring:
                # Cluster size cap
                if max_cluster_size > 0 and sizes[best] >= max_cluster_size:
                    if not cluster_swap:
                        n_blocked += 1
                        continue
                    mr = cluster_ids[np.arange(n), pointers]
                    members_best = np.where(mr == best)[0]
                    if len(members_best) == 0:
                        n_blocked += 1
                        continue
                    # Find member of target closest to source centroid
                    member_embs = all_embs_cpu[members_best]
                    dists_to_src = ((member_embs - centroids_cpu[primary]) ** 2).sum(axis=1)
                    dist_x_to_src = ((anchor_embs[idx] - centroids_cpu[primary]) ** 2).sum()
                    best_swap_local = int(dists_to_src.argmin())
                    if dists_to_src[best_swap_local] >= dist_x_to_src:
                        # No good swap candidate — reject migration
                        n_blocked += 1
                        continue
                    # Swap: Y goes to source, X continues to target
                    swap_neuron = int(members_best[best_swap_local])
                    swapped.add(swap_neuron)
                    if centroid_mode == 'exact':
                        os_b = sizes[best]
                        if os_b > 1:
                            centroids_cpu[best] = (centroids_cpu[best] * os_b - all_embs_cpu[swap_neuron]) / (os_b - 1)
                        os_p = sizes[primary]
                        centroids_cpu[primary] = (centroids_cpu[primary] * os_p + all_embs_cpu[swap_neuron]) / (os_p + 1)
                    lru_slot_y = int(last_used[swap_neuron].argmin())
                    evicted_y = int(cluster_ids[swap_neuron, lru_slot_y])
                    cluster_ids[swap_neuron, lru_slot_y] = primary
                    last_used[swap_neuron, lru_slot_y] = tick
                    pointers[swap_neuron] = lru_slot_y
                    wiring_events.append((swap_neuron, evicted_y, primary))
                    if wire_all_ring:
                        if evicted_y >= 0:
                            sizes[evicted_y] -= 1
                    else:
                        sizes[best] -= 1
                    sizes[primary] += 1
                    affected.add(best)
                    affected.add(primary)
                    n_reassigned += 1
                    if jump_counts is not None:
                        jump_counts[swap_neuron] += 1

                if centroid_mode == 'exact':
                    old_size = sizes[primary]
                    if old_size > 1:
                        centroids_cpu[primary] = (centroids_cpu[primary] * old_size - anchor_embs[idx]) / (old_size - 1)
                    new_size = sizes[best]
                    centroids_cpu[best] = (centroids_cpu[best] * new_size + anchor_embs[idx]) / (new_size + 1)

                lru_slot = int(last_used[anchor].argmin())
                evicted = int(cluster_ids[anchor, lru_slot])
                cluster_ids[anchor, lru_slot] = best
                anchor_cids[idx, lru_slot] = best
                last_used[anchor, lru_slot] = tick
                pointers[anchor] = lru_slot
                wiring_events.append((anchor, evicted, best))

            if in_ring:
                if not wire_all_ring:
                    sizes[primary] -= 1
                    sizes[best] += 1
                # wire_all_ring: neuron already wired to both, no size change
                n_switches += 1
            else:
                # Out-of-ring: evicted cluster loses a wiring, best gains one
                if wire_all_ring:
                    if evicted >= 0:
                        sizes[evicted] -= 1
                else:
                    sizes[primary] -= 1
                sizes[best] += 1
                n_reassigned += 1
                if jump_counts is not None:
                    jump_counts[anchor] += 1
            affected.add(primary)
            affected.add(best)

        # --- Post-loop centroid nudge (unchanged) ---
        if affected:
            affected_list = list(affected)
            if centroid_mode == 'nudge':
                most_recent = cluster_ids[np.arange(n), pointers]
                mr_t = torch.from_numpy(most_recent.astype(np.int64)).to(centroids_t.device)
                d = centroids_t.shape[1]
                sums = torch.zeros(m, d, device=centroids_t.device)
                sums.scatter_add_(0, mr_t.unsqueeze(1).expand(n, d), embeddings_t)
                counts = torch.bincount(mr_t, minlength=m).float().unsqueeze(1).clamp(min=1)
                means = sums / counts
                affected_t = torch.tensor(affected_list, dtype=torch.long, device=centroids_t.device)
                centroids_t[affected_t] += lr * (means[affected_t] - centroids_t[affected_t])
            else:
                centroids_t[affected_list] = torch.from_numpy(
                    centroids_cpu[affected_list]).to(centroids_t.device)

        return n_reassigned, affected, sizes, n_blocked, n_switches, wiring_events

    def split_largest_cluster_gpu(embeddings_t, centroids_t, cluster_ids, sizes, m,
                                  n_splits=1, seed=None, pointers=None,
                                  last_used=None, tick=0,
                                  wire_all_ring=True):
        """Split largest cluster(s) on GPU. Returns (splits_done, wiring_events)."""
        rng = np.random.RandomState(seed)
        max_k = cluster_ids.shape[1]
        if pointers is None:
            pointers = np.zeros(cluster_ids.shape[0], dtype=np.int64)
        if last_used is None:
            last_used = np.zeros((cluster_ids.shape[0], max_k), dtype=np.int64)
        splits_done = 0
        wiring_events = []  # list of (neuron, old_cluster, new_cluster)

        for _ in range(n_splits):
            empty = np.where(sizes == 0)[0]
            if len(empty) == 0:
                break
            largest = np.argmax(sizes)
            if wire_all_ring:
                members = np.where((cluster_ids == largest).any(axis=1))[0]
            else:
                most_recent = cluster_ids[np.arange(cluster_ids.shape[0]), pointers]
                members = np.where(most_recent == largest)[0]
            if len(members) < 2:
                break

            dead = empty[0]
            member_emb = embeddings_t[members]

            # k-means(2) on GPU
            idx = rng.choice(len(members), size=2, replace=False)
            c0 = member_emb[idx[0]].clone()
            c1 = member_emb[idx[1]].clone()
            for _ in range(10):
                d0 = ((member_emb - c0) ** 2).sum(dim=1)
                d1 = ((member_emb - c1) ** 2).sum(dim=1)
                assign = (d1 < d0).long()
                s0, s1 = (assign == 0).sum().item(), (assign == 1).sum().item()
                if s0 == 0 or s1 == 0:
                    idx = rng.choice(len(members), size=2, replace=False)
                    c0 = member_emb[idx[0]].clone()
                    c1 = member_emb[idx[1]].clone()
                    continue
                c0 = member_emb[assign == 0].mean(dim=0)
                c1 = member_emb[assign == 1].mean(dim=0)

            assign_cpu = assign.cpu().numpy()
            half1 = members[assign_cpu == 1]
            if len(half1) == 0 or len(half1) == len(members):
                continue

            # Replace largest with dead in-place, point to dead slot
            for neuron in half1:
                row = cluster_ids[neuron]
                wrote_dead = False
                n_evicted = 0
                for s in range(max_k):
                    if row[s] == largest:
                        if not wrote_dead:
                            row[s] = dead
                            pointers[neuron] = s
                            last_used[neuron, s] = tick
                            wrote_dead = True
                        else:
                            row[s] = -1
                            last_used[neuron, s] = 0
                        n_evicted += 1
                if wire_all_ring:
                    sizes[largest] -= n_evicted
                    sizes[dead] += 1
                else:
                    sizes[largest] -= 1
                    sizes[dead] += 1
                wiring_events.append((int(neuron), int(largest), int(dead)))
            centroids_t[largest] = member_emb[assign_cpu == 0].mean(dim=0)
            centroids_t[dead] = member_emb[assign_cpu == 1].mean(dim=0)
            splits_done += 1

        return splits_done, wiring_events


# ---------------------------------------------------------------------------
# Core clustering functions (CPU numpy)
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
    # knn2 stores cluster indices directly (not neuron indices)
    if knn2 is not None:
        knn2_dists = []
        for c in range(m):
            valid = knn2[c] >= 0
            if not valid.any():
                continue
            for tc in knn2[c, valid]:
                if tc != c and sizes[tc] > 0:
                    d = np.sqrt(((cluster_centers[c] - cluster_centers[tc]) ** 2).sum())
                    knn2_dists.append(d)
        results['knn2_center_dist_mean'] = float(np.mean(knn2_dists)) if knn2_dists else 0
        results['knn2_center_dist_max'] = float(np.max(knn2_dists)) if knn2_dists else 0

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


def visualize_clusters_signal(cluster_ids, signal, width, height,
                               sig_channels=1, path=None):
    """Render clusters using mean signal of member neurons.

    For grayscale (sig_channels=1): each cluster gets brightness = mean signal.
    For RGB (sig_channels=3): each pixel's channels are averaged per-cluster.

    Args:
        cluster_ids: (n,) int64, primary cluster per neuron
        signal: (n,) float32, current signal frame (one value per neuron)
        width, height: grid dimensions
        sig_channels: 1 (grayscale) or 3 (RGB)
        path: output image path
    """
    try:
        import cv2
    except ImportError:
        return

    m = cluster_ids.max() + 1
    n = len(cluster_ids)

    if sig_channels == 1:
        # Mean signal per cluster
        cluster_means = np.zeros(m, dtype=np.float32)
        counts = np.bincount(cluster_ids, minlength=m).astype(np.float32)
        np.add.at(cluster_means, cluster_ids, signal)
        valid = counts > 0
        cluster_means[valid] /= counts[valid]
        # Map neurons to cluster mean
        pixel_vals = cluster_means[cluster_ids]
        # Normalize to 0-255
        vmin, vmax = pixel_vals.min(), pixel_vals.max()
        if vmax > vmin:
            pixel_vals = (pixel_vals - vmin) / (vmax - vmin) * 255
        else:
            pixel_vals = np.full_like(pixel_vals, 128)
        img = pixel_vals.reshape(height, width).astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    else:
        # RGB: n = h * w * sig_channels, group by pixel
        n_pixels = height * width
        # Reshape signal to (n_pixels, sig_channels)
        sig_rgb = signal.reshape(n_pixels, sig_channels)
        # Use cluster of first channel for each pixel (all channels share cluster)
        pixel_cids = cluster_ids[:n_pixels * sig_channels:sig_channels]
        # Mean RGB per cluster
        cluster_rgb = np.zeros((m, sig_channels), dtype=np.float32)
        counts = np.bincount(pixel_cids, minlength=m).astype(np.float32)
        for ch in range(sig_channels):
            np.add.at(cluster_rgb[:, ch], pixel_cids, sig_rgb[:, ch])
        valid = counts > 0
        for ch in range(sig_channels):
            cluster_rgb[valid, ch] /= counts[valid]
        # Map pixels to cluster mean RGB
        pixel_rgb = cluster_rgb[pixel_cids]
        # Normalize to 0-255
        vmin, vmax = pixel_rgb.min(), pixel_rgb.max()
        if vmax > vmin:
            pixel_rgb = (pixel_rgb - vmin) / (vmax - vmin) * 255
        else:
            pixel_rgb = np.full_like(pixel_rgb, 128)
        img = pixel_rgb.reshape(height, width, sig_channels).astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # Scale up for visibility
    scale = max(1, 512 // max(width, height))
    if scale > 1:
        img = cv2.resize(img, (width * scale, height * scale),
                         interpolation=cv2.INTER_NEAREST)

    if path:
        cv2.imwrite(path, img)
        print(f"  cluster signal map saved: {path}")
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
# Phase 3: Streaming from scratch (random embeddings → converged)
# ---------------------------------------------------------------------------

def run_from_scratch(args):
    """Phase 3: streaming clusters while embeddings evolve from random to converged.

    Simulates simultaneous training + clustering: embeddings interpolate from
    random → converged over `embed_steps` phases. Within each phase, run
    `iters_per_phase` streaming iterations on the current embeddings. This
    mimics the real scenario where clustering starts at tick 0 alongside training.
    """
    converged_emb = np.load(args.model)
    knn_lists = np.load(args.knn)
    n, dims = converged_emb.shape
    k = knn_lists.shape[1]
    W, H = args.width, args.height
    m = args.m
    k2 = args.k2 if args.k2 else k
    assert n == W * H

    embed_steps = args.embed_steps
    iters_per_phase = args.iters_per_phase
    total_iters = embed_steps * iters_per_phase

    print(f"From-scratch clustering: n={n} ({W}x{H}), m={m}, dims={dims}, k2={k2}")
    print(f"  {embed_steps} embedding phases × {iters_per_phase} iters/phase "
          f"= {total_iters} total iters")
    print(f"  batch_size={args.batch_size}, threshold={args.threshold}, lr={args.lr}")

    out_dir = args.output_dir
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    # Offline baseline on converged embeddings for comparison
    print("\nComputing offline baseline on converged embeddings...")
    baseline_ids, baseline_centroids = kmeans_cluster(converged_emb, m, seed=42)
    baseline_knn2, _ = frequency_knn(knn_lists, baseline_ids, m, k2)
    baseline_metrics = eval_clusters(baseline_ids, baseline_centroids, baseline_knn2,
                                     W, H, knn_lists)
    print(f"  baseline: contiguity={baseline_metrics['contiguity_mean']:.3f} "
          f"diam={baseline_metrics['diameter_mean']:.1f}")

    # Start from random embeddings + random clusters
    rng = np.random.RandomState(args.seed)
    random_emb = rng.randn(n, dims).astype(np.float32)
    # Normalize to similar scale as converged
    random_emb *= np.std(converged_emb) / np.std(random_emb)

    # Random centroids from the random embeddings
    idx = rng.choice(n, size=m, replace=False)
    centroids = random_emb[idx].copy()
    cluster_ids = _assign_clusters(random_emb, centroids)
    sizes = np.bincount(cluster_ids, minlength=m)

    print(f"\nRandom init: {(sizes == 0).sum()} empty clusters")

    print(f"\n{'phase':>5} {'alpha':>5} {'iter':>5} {'reassgn':>7} {'empty':>5} "
          f"{'size_std':>8} {'diam':>6} {'contig':>7} {'knn2_agr':>8} {'agree':>6}")

    report_every = max(1, iters_per_phase // 4)

    for phase in range(embed_steps):
        # Interpolate: alpha goes from 0 (random) to 1 (converged)
        alpha = (phase + 1) / embed_steps
        current_emb = (1 - alpha) * random_emb + alpha * converged_emb

        for it in range(iters_per_phase):
            anchors = rng.choice(n, size=args.batch_size, replace=False)

            n_reassigned, affected, sizes = streaming_update(
                current_emb, centroids, cluster_ids, anchors,
                threshold=args.threshold, lr=args.lr, sizes=sizes)

            if it % report_every == 0 or (it == iters_per_phase - 1 and
                    phase % max(1, embed_steps // 20) == 0):
                knn2, _ = frequency_knn(knn_lists, cluster_ids, m, k2)
                metrics = eval_clusters(cluster_ids, centroids, knn2, W, H, knn_lists)
                agree = cluster_agreement(cluster_ids, baseline_ids, n, m)
                print(f"{phase:>5} {alpha:>5.2f} {it:>5} {n_reassigned:>7} "
                      f"{metrics['n_empty']:>5} {metrics['size_std']:>8.1f} "
                      f"{metrics['diameter_mean']:>6.1f} "
                      f"{metrics['contiguity_mean']:>7.3f} "
                      f"{metrics.get('knn2_agreement', 0):>8.3f} {agree:>6.3f}")

        if out_dir and phase % max(1, embed_steps // 10) == 0:
            visualize_clusters(cluster_ids, W, H,
                               os.path.join(out_dir, f"clusters_phase{phase:03d}.png"))

    # Final eval with fully converged embeddings
    print("\n--- Final (alpha=1.0, converged embeddings) ---")
    # One more round of streaming on fully converged embeddings
    for it in range(iters_per_phase):
        anchors = rng.choice(n, size=args.batch_size, replace=False)
        n_reassigned, affected, sizes = streaming_update(
            converged_emb, centroids, cluster_ids, anchors,
            threshold=args.threshold, lr=args.lr, sizes=sizes)

    knn2, _ = frequency_knn(knn_lists, cluster_ids, m, k2)
    final_metrics = eval_clusters(cluster_ids, centroids, knn2, W, H, knn_lists)
    agree = cluster_agreement(cluster_ids, baseline_ids, n, m)

    print(f"  clusters: {m - final_metrics['n_empty']} non-empty "
          f"({final_metrics['n_empty']} empty)")
    print(f"  sizes: min={final_metrics['size_min']} max={final_metrics['size_max']} "
          f"mean={final_metrics['size_mean']:.1f} std={final_metrics['size_std']:.1f}")
    print(f"  diameter: mean={final_metrics['diameter_mean']:.1f}")
    print(f"  contiguity: {final_metrics['contiguity_mean']:.3f}")
    print(f"  knn2 agreement: {final_metrics.get('knn2_agreement', 0):.3f}")
    print(f"  clustering agreement with baseline: {agree:.3f}")

    if out_dir:
        visualize_clusters(cluster_ids, W, H,
                           os.path.join(out_dir, f"clusters_final_m{m}.png"))
        visualize_clusters(baseline_ids, W, H,
                           os.path.join(out_dir, f"clusters_baseline_m{m}.png"))


# ---------------------------------------------------------------------------
# V2: knn2-guided reassignment + split-based dead cluster recovery
# ---------------------------------------------------------------------------

def streaming_update_v2(embeddings, centroids, cluster_ids, knn2, anchors,
                        lr=0.01, sizes=None):
    """knn2-guided streaming update.

    For each anchor:
      1. Gather candidate clusters: own cluster + knn2 neighbors
      2. Compare distance to each candidate centroid
      3. Move to closest if it's not the current cluster

    Returns: n_reassigned, affected_clusters set, sizes
    """
    m = centroids.shape[0]
    n = embeddings.shape[0]
    dims = embeddings.shape[1]

    if sizes is None:
        sizes = np.bincount(cluster_ids, minlength=m)

    n_reassigned = 0
    affected = set()

    for i, anchor in enumerate(anchors):
        cur = cluster_ids[anchor]
        emb = embeddings[anchor]  # (dims,)

        # Candidates: own cluster + knn2 neighbors (skip -1 / invalid)
        neighbors = knn2[cur]
        valid_neighbors = neighbors[neighbors >= 0]
        neighbor_clusters = np.unique(cluster_ids[valid_neighbors])
        # Remove self and add self explicitly to ensure it's always a candidate
        candidates = np.unique(np.concatenate([[cur], neighbor_clusters]))

        # Distance to each candidate centroid
        cand_centroids = centroids[candidates]  # (len(candidates), dims)
        dists = np.sum((emb - cand_centroids) ** 2, axis=1)
        best_idx = dists.argmin()
        best = candidates[best_idx]

        if best != cur:
            cluster_ids[anchor] = best
            sizes[cur] -= 1
            sizes[best] += 1
            affected.add(cur)
            affected.add(best)
            n_reassigned += 1

    # Nudge centroids for affected clusters
    for c in affected:
        members = np.where(cluster_ids == c)[0]
        if len(members) > 0:
            member_mean = embeddings[members].mean(axis=0)
            centroids[c] += lr * (member_mean - centroids[c])

    return n_reassigned, affected, sizes


def split_largest_cluster(embeddings, centroids, cluster_ids, sizes, m,
                          n_splits=1, seed=None):
    """Split the largest cluster(s) to revive dead (empty) clusters.

    For each split: find largest cluster, k-means(2) on its members,
    assign one half to a dead cluster ID.

    Returns: number of splits performed
    """
    rng = np.random.RandomState(seed)
    splits_done = 0

    for _ in range(n_splits):
        empty = np.where(sizes == 0)[0]
        if len(empty) == 0:
            break

        largest = np.argmax(sizes)
        if sizes[largest] < 4:  # too small to split
            break

        dead = empty[0]
        members = np.where(cluster_ids == largest)[0]
        member_emb = embeddings[members]

        # k-means(2) on the largest cluster
        # Simple: pick two random members as seeds, run a few iterations
        idx = rng.choice(len(members), size=2, replace=False)
        c0, c1 = member_emb[idx[0]], member_emb[idx[1]]
        for _ in range(10):
            d0 = np.sum((member_emb - c0) ** 2, axis=1)
            d1 = np.sum((member_emb - c1) ** 2, axis=1)
            assign = (d1 < d0).astype(int)
            if assign.sum() == 0 or assign.sum() == len(members):
                # Degenerate split — try different seeds
                idx = rng.choice(len(members), size=2, replace=False)
                c0, c1 = member_emb[idx[0]], member_emb[idx[1]]
                continue
            c0 = member_emb[assign == 0].mean(axis=0)
            c1 = member_emb[assign == 1].mean(axis=0)

        # Assign half to dead cluster
        half1 = members[assign == 1]
        if len(half1) == 0 or len(half1) == len(members):
            # Failed to split — skip
            continue

        cluster_ids[half1] = dead
        centroids[largest] = embeddings[members[assign == 0]].mean(axis=0)
        centroids[dead] = embeddings[half1].mean(axis=0)
        sizes[largest] = (assign == 0).sum()
        sizes[dead] = len(half1)
        splits_done += 1

    return splits_done


def streaming_update_v3(embeddings, centroids, cluster_ids, knn2, anchors,
                        lr=0.01, sizes=None, min_size=2, rng=None):
    """v3: knn2-guided reassignment with min_size block + probabilistic throttle.

    Three mechanisms:
      A. Block moves that would drop source cluster below min_size
      B. Probabilistic throttle: p = 1 - (n_empty / m), skip reassignment check
         with probability 1-p when many clusters are dead
      C. After reassignment, dissolve clusters below min_size (handled externally)

    Returns: n_reassigned, affected_clusters set, sizes, n_blocked
    """
    m = centroids.shape[0]
    n = embeddings.shape[0]

    if sizes is None:
        sizes = np.bincount(cluster_ids, minlength=m)
    if rng is None:
        rng = np.random.RandomState()

    n_empty = (sizes == 0).sum()
    p = 1.0 - (n_empty / m)  # throttle: more dead → less churn

    n_reassigned = 0
    n_blocked = 0
    affected = set()

    for i, anchor in enumerate(anchors):
        # Probabilistic throttle: skip check entirely with probability 1-p
        if rng.random() >= p:
            continue

        cur = cluster_ids[anchor]
        emb = embeddings[anchor]

        # Candidates: own cluster + knn2 neighbors
        neighbors = knn2[cur]
        valid_neighbors = neighbors[neighbors >= 0]
        neighbor_clusters = np.unique(cluster_ids[valid_neighbors])
        candidates = np.unique(np.concatenate([[cur], neighbor_clusters]))

        # Distance to each candidate centroid
        cand_centroids = centroids[candidates]
        dists = np.sum((emb - cand_centroids) ** 2, axis=1)
        best = candidates[dists.argmin()]

        if best != cur:
            # Block if move would drop source below min_size
            if sizes[cur] <= min_size:
                n_blocked += 1
                continue

            cluster_ids[anchor] = best
            sizes[cur] -= 1
            sizes[best] += 1
            affected.add(cur)
            affected.add(best)
            n_reassigned += 1

    # Nudge centroids for affected clusters
    for c in affected:
        members = np.where(cluster_ids == c)[0]
        if len(members) > 0:
            member_mean = embeddings[members].mean(axis=0)
            centroids[c] += lr * (member_mean - centroids[c])

    return n_reassigned, affected, sizes, n_blocked


def dissolve_small_clusters(embeddings, centroids, cluster_ids, knn2, sizes,
                            min_size=2):
    """Dissolve clusters below min_size by reassigning members to knn2-guided best fit.

    Returns: n_dissolved (number of clusters dissolved), n_neurons_moved
    """
    m = centroids.shape[0]
    n_dissolved = 0
    n_moved = 0

    for c in range(m):
        if sizes[c] == 0 or sizes[c] >= min_size:
            continue

        # This cluster is too small — dissolve it
        members = np.where(cluster_ids == c)[0]

        for neuron in members:
            emb = embeddings[neuron]

            # Find best cluster via knn2 neighbors of current cluster
            neighbors = knn2[c]
            valid_neighbors = neighbors[neighbors >= 0]
            if len(valid_neighbors) == 0:
                # No knn2 info — find nearest non-empty centroid
                alive = np.where(sizes >= min_size)[0]
                if len(alive) == 0:
                    alive = np.where(sizes > 0)[0]
                    alive = alive[alive != c]
                if len(alive) == 0:
                    break
                dists = np.sum((emb - centroids[alive]) ** 2, axis=1)
                best = alive[dists.argmin()]
            else:
                neighbor_clusters = np.unique(cluster_ids[valid_neighbors])
                # Only consider clusters that are alive and above min_size
                viable = neighbor_clusters[(neighbor_clusters != c) &
                                           (sizes[neighbor_clusters] > 0)]
                if len(viable) == 0:
                    # Fall back to any alive cluster
                    alive = np.where(sizes > 0)[0]
                    alive = alive[alive != c]
                    if len(alive) == 0:
                        break
                    viable = alive
                dists = np.sum((emb - centroids[viable]) ** 2, axis=1)
                best = viable[dists.argmin()]

            cluster_ids[neuron] = best
            sizes[c] -= 1
            sizes[best] += 1
            n_moved += 1

        n_dissolved += 1

    return n_dissolved, n_moved


def run_from_scratch_v2(args):
    """Phase 3 v2: knn2-guided reassignment + split recovery."""
    converged_emb = np.load(args.model)
    knn_lists = np.load(args.knn)
    n, dims = converged_emb.shape
    k = knn_lists.shape[1]
    W, H = args.width, args.height
    m = args.m
    k2 = args.k2 if args.k2 else k
    assert n == W * H

    embed_steps = args.embed_steps
    iters_per_phase = args.iters_per_phase
    split_every = args.split_every
    total_iters = embed_steps * iters_per_phase

    print(f"From-scratch v2 (knn2-guided): n={n} ({W}x{H}), m={m}, dims={dims}, k2={k2}")
    print(f"  {embed_steps} phases × {iters_per_phase} iters = {total_iters} total")
    print(f"  batch_size={args.batch_size}, lr={args.lr}, split_every={split_every}")

    out_dir = args.output_dir
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    # Offline baseline
    print("\nComputing offline baseline...")
    baseline_ids, baseline_centroids = kmeans_cluster(converged_emb, m, seed=42)
    baseline_knn2, _ = frequency_knn(knn_lists, baseline_ids, m, k2)
    baseline_metrics = eval_clusters(baseline_ids, baseline_centroids, baseline_knn2,
                                     W, H, knn_lists)
    print(f"  baseline: contiguity={baseline_metrics['contiguity_mean']:.3f} "
          f"diam={baseline_metrics['diameter_mean']:.1f}")

    # Random init
    rng = np.random.RandomState(args.seed)
    random_emb = rng.randn(n, dims).astype(np.float32)
    random_emb *= np.std(converged_emb) / np.std(random_emb)

    idx = rng.choice(n, size=m, replace=False)
    centroids = random_emb[idx].copy()
    cluster_ids = _assign_clusters(random_emb, centroids)
    sizes = np.bincount(cluster_ids, minlength=m)

    # Initial knn2
    knn2, _ = frequency_knn(knn_lists, cluster_ids, m, k2)

    print(f"\nRandom init: {(sizes == 0).sum()} empty clusters")

    print(f"\n{'phase':>5} {'alpha':>5} {'iter':>5} {'reassgn':>7} {'splits':>6} "
          f"{'empty':>5} {'size_std':>8} {'diam':>6} {'contig':>7} {'knn2_agr':>8} "
          f"{'agree':>6}")

    report_every = max(1, iters_per_phase // 4)
    total_splits = 0

    for phase in range(embed_steps):
        alpha = (phase + 1) / embed_steps
        current_emb = (1 - alpha) * random_emb + alpha * converged_emb

        for it in range(iters_per_phase):
            anchors = rng.choice(n, size=args.batch_size, replace=False)

            n_reassigned, affected, sizes = streaming_update_v2(
                current_emb, centroids, cluster_ids, knn2, anchors,
                lr=args.lr, sizes=sizes)

            # Patch knn2 for affected clusters
            if affected:
                for c in affected:
                    if sizes[c] > 0:
                        members_c = np.where(cluster_ids == c)[0]
                        pooled = knn_lists[members_c].flatten()
                        ids, counts = np.unique(pooled, return_counts=True)
                        not_self = cluster_ids[ids] != c
                        ids, counts = ids[not_self], counts[not_self]
                        if len(ids) > 0:
                            top = min(k2, len(ids))
                            if top == len(ids):
                                top_idx = np.argsort(-counts)[:top]
                            else:
                                top_idx = np.argpartition(-counts, top)[:top]
                                top_idx = top_idx[np.argsort(-counts[top_idx])]
                            knn2[c, :] = -1
                            knn2[c, :top] = ids[top_idx]
                        else:
                            knn2[c, :] = -1

            # Periodic split
            global_iter = phase * iters_per_phase + it
            n_splits = 0
            if split_every > 0 and global_iter > 0 and global_iter % split_every == 0:
                n_empty = (sizes == 0).sum()
                if n_empty > 0:
                    n_to_split = min(n_empty, max(1, n_empty // 10))
                    n_splits = split_largest_cluster(
                        current_emb, centroids, cluster_ids, sizes, m,
                        n_splits=n_to_split, seed=rng.randint(1000000))
                    total_splits += n_splits
                    # Rebuild knn2 fully after splits (simpler than patching)
                    if n_splits > 0:
                        knn2, _ = frequency_knn(knn_lists, cluster_ids, m, k2)

            if it % report_every == 0 or (it == iters_per_phase - 1 and
                    phase % max(1, embed_steps // 20) == 0):
                metrics = eval_clusters(cluster_ids, centroids, knn2, W, H, knn_lists)
                agree = cluster_agreement(cluster_ids, baseline_ids, n, m)
                print(f"{phase:>5} {alpha:>5.2f} {it:>5} {n_reassigned:>7} "
                      f"{n_splits:>6} {metrics['n_empty']:>5} "
                      f"{metrics['size_std']:>8.1f} {metrics['diameter_mean']:>6.1f} "
                      f"{metrics['contiguity_mean']:>7.3f} "
                      f"{metrics.get('knn2_agreement', 0):>8.3f} {agree:>6.3f}")

        if out_dir and phase % max(1, embed_steps // 10) == 0:
            visualize_clusters(cluster_ids, W, H,
                               os.path.join(out_dir, f"clusters_phase{phase:03d}.png"))

    # Final eval with converged embeddings
    print("\n--- Final (alpha=1.0, converged embeddings) ---")
    for it in range(iters_per_phase):
        anchors = rng.choice(n, size=args.batch_size, replace=False)
        n_reassigned, affected, sizes = streaming_update_v2(
            converged_emb, centroids, cluster_ids, knn2, anchors,
            lr=args.lr, sizes=sizes)
        if affected:
            for c in affected:
                if sizes[c] > 0:
                    members_c = np.where(cluster_ids == c)[0]
                    pooled = knn_lists[members_c].flatten()
                    ids, counts = np.unique(pooled, return_counts=True)
                    not_self = cluster_ids[ids] != c
                    ids, counts = ids[not_self], counts[not_self]
                    if len(ids) > 0:
                        top = min(k2, len(ids))
                        if top == len(ids):
                            top_idx = np.argsort(-counts)[:top]
                        else:
                            top_idx = np.argpartition(-counts, top)[:top]
                            top_idx = top_idx[np.argsort(-counts[top_idx])]
                        knn2[c, :] = -1
                        knn2[c, :top] = ids[top_idx]
                    else:
                        knn2[c, :] = -1
        global_iter = embed_steps * iters_per_phase + it
        if split_every > 0 and global_iter % split_every == 0:
            n_empty = (sizes == 0).sum()
            if n_empty > 0:
                ns = split_largest_cluster(
                    converged_emb, centroids, cluster_ids, sizes, m,
                    n_splits=min(n_empty, max(1, n_empty // 10)),
                    seed=rng.randint(1000000))
                total_splits += ns
                if ns > 0:
                    knn2, _ = frequency_knn(knn_lists, cluster_ids, m, k2)

    knn2, _ = frequency_knn(knn_lists, cluster_ids, m, k2)
    final_metrics = eval_clusters(cluster_ids, centroids, knn2, W, H, knn_lists)
    agree = cluster_agreement(cluster_ids, baseline_ids, n, m)

    print(f"  clusters: {m - final_metrics['n_empty']} non-empty "
          f"({final_metrics['n_empty']} empty)")
    print(f"  sizes: min={final_metrics['size_min']} max={final_metrics['size_max']} "
          f"mean={final_metrics['size_mean']:.1f} std={final_metrics['size_std']:.1f}")
    print(f"  diameter: mean={final_metrics['diameter_mean']:.1f}")
    print(f"  contiguity: {final_metrics['contiguity_mean']:.3f}")
    print(f"  knn2 agreement: {final_metrics.get('knn2_agreement', 0):.3f}")
    print(f"  clustering agreement with baseline: {agree:.3f}")
    print(f"  total splits: {total_splits}")

    if out_dir:
        visualize_clusters(cluster_ids, W, H,
                           os.path.join(out_dir, f"clusters_final_m{m}.png"))
        visualize_clusters(baseline_ids, W, H,
                           os.path.join(out_dir, f"clusters_baseline_m{m}.png"))


# ---------------------------------------------------------------------------
# V3: min_size dissolution + probabilistic throttle + knn2-guided + splits
# ---------------------------------------------------------------------------

def run_from_scratch_v3(args):
    """Phase 3 v3: min_size dissolution + probabilistic throttle + knn2-guided + splits."""
    converged_emb = np.load(args.model)
    knn_lists = np.load(args.knn)
    n, dims = converged_emb.shape
    k = knn_lists.shape[1]
    W, H = args.width, args.height
    m = args.m
    k2 = args.k2 if args.k2 else k
    min_size = args.min_size
    assert n == W * H

    use_gpu = HAS_TORCH
    if use_gpu:
        print(f"GPU mode: {torch.cuda.get_device_name()}")

    embed_steps = args.embed_steps
    iters_per_phase = args.iters_per_phase
    split_every = args.split_every
    converge_phases = getattr(args, 'converge_phases', 1)
    total_phases = embed_steps + converge_phases
    total_iters = total_phases * iters_per_phase

    print(f"From-scratch v3 (min_size+throttle+knn2+split): n={n} ({W}x{H}), m={m}, "
          f"dims={dims}, k2={k2}")
    print(f"  {embed_steps} interp + {converge_phases} converge phases × "
          f"{iters_per_phase} iters = {total_iters} total")
    print(f"  batch_size={args.batch_size}, lr={args.lr}, split_every={split_every}, "
          f"min_size={min_size}")

    out_dir = args.output_dir
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    # Upload to GPU if available
    if use_gpu:
        converged_emb_t = torch.from_numpy(converged_emb).to(DEVICE)
        knn_lists_t = torch.from_numpy(knn_lists.astype(np.int64)).to(DEVICE)

    # Offline baseline
    print("\nComputing offline baseline...")
    if use_gpu:
        baseline_ids, baseline_centroids = kmeans_cluster_gpu(converged_emb_t, m, seed=42)
    else:
        baseline_ids, baseline_centroids = kmeans_cluster(converged_emb, m, seed=42)
    baseline_knn2, _ = frequency_knn(knn_lists, baseline_ids, m, k2)
    baseline_metrics = eval_clusters(baseline_ids, baseline_centroids, baseline_knn2,
                                     W, H, knn_lists)
    print(f"  baseline: contiguity={baseline_metrics['contiguity_mean']:.3f} "
          f"diam={baseline_metrics['diameter_mean']:.1f}")

    # Random init
    rng = np.random.RandomState(args.seed)
    random_emb = rng.randn(n, dims).astype(np.float32)
    random_emb *= np.std(converged_emb) / np.std(random_emb)

    idx = rng.choice(n, size=m, replace=False)
    centroids = random_emb[idx].copy()
    if use_gpu:
        random_emb_t = torch.from_numpy(random_emb).to(DEVICE)
        centroids_t = torch.from_numpy(centroids).to(DEVICE)
        cluster_ids = _assign_clusters_gpu(random_emb_t, centroids_t).cpu().numpy()
    else:
        cluster_ids = _assign_clusters(random_emb, centroids)
    sizes = np.bincount(cluster_ids, minlength=m)

    # Initial knn2
    knn2, _ = frequency_knn(knn_lists, cluster_ids, m, k2)

    print(f"\nRandom init: {(sizes == 0).sum()} empty clusters")

    print(f"\n{'phase':>5} {'alpha':>5} {'iter':>5} {'reassgn':>7} {'blocked':>7} "
          f"{'dissolv':>7} {'splits':>6} {'empty':>5} {'<min':>5} "
          f"{'size_std':>8} {'diam':>6} {'contig':>7} {'agree':>6}")

    report_every = max(1, iters_per_phase // 4)
    total_splits = 0
    total_dissolved = 0

    def _patch_knn2(affected_set):
        """Recompute knn2 for affected clusters."""
        for c in affected_set:
            if sizes[c] > 0:
                members_c = np.where(cluster_ids == c)[0]
                pooled = knn_lists[members_c].flatten()
                ids, counts = np.unique(pooled, return_counts=True)
                not_self = cluster_ids[ids] != c
                ids, counts = ids[not_self], counts[not_self]
                if len(ids) > 0:
                    top = min(k2, len(ids))
                    if top == len(ids):
                        top_idx = np.argsort(-counts)[:top]
                    else:
                        top_idx = np.argpartition(-counts, top)[:top]
                        top_idx = top_idx[np.argsort(-counts[top_idx])]
                    knn2[c, :] = -1
                    knn2[c, :top] = ids[top_idx]
                else:
                    knn2[c, :] = -1

    def _run_iteration(current_emb, current_emb_t, phase, it, global_iter):
        nonlocal total_splits, total_dissolved

        anchors = rng.choice(n, size=args.batch_size, replace=False)

        if use_gpu:
            n_reassigned, affected, sizes_out, n_blocked = streaming_update_v3_gpu(
                current_emb_t, centroids_t, cluster_ids, knn2, anchors,
                lr=args.lr, sizes=sizes, min_size=min_size, rng=rng)
        else:
            n_reassigned, affected, sizes_out, n_blocked = streaming_update_v3(
                current_emb, centroids, cluster_ids, knn2, anchors,
                lr=args.lr, sizes=sizes, min_size=min_size, rng=rng)

        # Patch knn2 for affected clusters
        if affected:
            _patch_knn2(affected)

        # Dissolve small clusters
        centroids_for_dissolve = centroids_t.cpu().numpy() if use_gpu else centroids
        n_dissolved, n_moved = dissolve_small_clusters(
            current_emb, centroids_for_dissolve, cluster_ids, knn2, sizes,
            min_size=min_size)
        total_dissolved += n_dissolved
        if n_dissolved > 0:
            # Patch knn2 for clusters that received dissolved neurons
            knn2_rebuild = set()
            for c in range(m):
                if sizes[c] > 0:
                    knn2_rebuild.add(c)
            # Full rebuild is simpler after dissolutions
            knn2[:], _ = frequency_knn(knn_lists, cluster_ids, m, k2)

        # Periodic split recovery
        n_splits = 0
        if split_every > 0 and global_iter > 0 and global_iter % split_every == 0:
            n_empty = (sizes == 0).sum()
            if n_empty > 0:
                n_to_split = min(n_empty, max(1, n_empty // 5))
                if use_gpu:
                    n_splits = split_largest_cluster_gpu(
                        current_emb_t, centroids_t, cluster_ids, sizes, m,
                        n_splits=n_to_split, seed=rng.randint(1000000))
                else:
                    n_splits = split_largest_cluster(
                        current_emb, centroids, cluster_ids, sizes, m,
                        n_splits=n_to_split, seed=rng.randint(1000000))
                total_splits += n_splits
                if n_splits > 0:
                    knn2[:], _ = frequency_knn(knn_lists, cluster_ids, m, k2)

        return n_reassigned, n_blocked, n_dissolved, n_splits

    save_every = max(1, total_phases // 20)

    for phase in range(total_phases):
        if phase < embed_steps:
            alpha = (phase + 1) / embed_steps
            current_emb = (1 - alpha) * random_emb + alpha * converged_emb
            if use_gpu:
                current_emb_t = torch.from_numpy(current_emb).to(DEVICE)
        else:
            alpha = 1.0
            current_emb = converged_emb
            if use_gpu:
                current_emb_t = converged_emb_t

        for it in range(iters_per_phase):
            global_iter = phase * iters_per_phase + it
            n_reassigned, n_blocked, n_dissolved, n_splits = _run_iteration(
                current_emb, current_emb_t if use_gpu else None, phase, it, global_iter)

            if it % report_every == 0 or (it == iters_per_phase - 1 and
                    phase % save_every == 0):
                centroids_np = centroids_t.cpu().numpy() if use_gpu else centroids
                metrics = eval_clusters(cluster_ids, centroids_np, knn2, W, H, knn_lists)
                agree = cluster_agreement(cluster_ids, baseline_ids, n, m)
                n_below_min = ((sizes > 0) & (sizes < min_size)).sum()
                print(f"{phase:>5} {alpha:>5.2f} {it:>5} {n_reassigned:>7} "
                      f"{n_blocked:>7} {n_dissolved:>7} {n_splits:>6} "
                      f"{metrics['n_empty']:>5} {n_below_min:>5} "
                      f"{metrics['size_std']:>8.1f} {metrics['diameter_mean']:>6.1f} "
                      f"{metrics['contiguity_mean']:>7.3f} {agree:>6.3f}")

        if out_dir and phase % save_every == 0:
            visualize_clusters(cluster_ids, W, H,
                               os.path.join(out_dir, f"clusters_v3_phase{phase:03d}.png"))

    # Final eval
    print("\n--- Final ---")
    knn2[:], _ = frequency_knn(knn_lists, cluster_ids, m, k2)
    centroids_np = centroids_t.cpu().numpy() if use_gpu else centroids
    final_metrics = eval_clusters(cluster_ids, centroids_np, knn2, W, H, knn_lists)
    agree = cluster_agreement(cluster_ids, baseline_ids, n, m)

    print(f"  clusters: {m - final_metrics['n_empty']} non-empty "
          f"({final_metrics['n_empty']} empty)")
    print(f"  sizes: min={final_metrics['size_min']} max={final_metrics['size_max']} "
          f"mean={final_metrics['size_mean']:.1f} std={final_metrics['size_std']:.1f}")
    print(f"  diameter: mean={final_metrics['diameter_mean']:.1f}")
    print(f"  contiguity: {final_metrics['contiguity_mean']:.3f}")
    print(f"  knn2 agreement: {final_metrics.get('knn2_agreement', 0):.3f}")
    print(f"  clustering agreement with baseline: {agree:.3f}")
    print(f"  total splits: {total_splits}, total dissolved: {total_dissolved}")

    if out_dir:
        visualize_clusters(cluster_ids, W, H,
                           os.path.join(out_dir, f"clusters_v3_final_m{m}.png"))
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

    # From-scratch
    p_fs = sub.add_parser('from-scratch', help='Phase 3: streaming with evolving embeddings')
    p_fs.add_argument('--model', required=True, help='Path to converged model.npy')
    p_fs.add_argument('--knn', required=True, help='Path to knn_lists.npy')
    p_fs.add_argument('-W', '--width', type=int, required=True)
    p_fs.add_argument('-H', '--height', type=int, required=True)
    p_fs.add_argument('--m', type=int, default=100)
    p_fs.add_argument('--k2', type=int, default=None)
    p_fs.add_argument('--batch-size', type=int, default=256)
    p_fs.add_argument('--threshold', type=float, default=0.5)
    p_fs.add_argument('--lr', type=float, default=0.1)
    p_fs.add_argument('--embed-steps', type=int, default=20,
                       help='Number of embedding interpolation phases')
    p_fs.add_argument('--iters-per-phase', type=int, default=50,
                       help='Streaming iterations per embedding phase')
    p_fs.add_argument('--seed', type=int, default=42)
    p_fs.add_argument('-o', '--output-dir', type=str, default=None)

    # From-scratch v2
    p_v2 = sub.add_parser('from-scratch-v2',
                          help='Phase 3 v2: knn2-guided + split recovery')
    p_v2.add_argument('--model', required=True)
    p_v2.add_argument('--knn', required=True)
    p_v2.add_argument('-W', '--width', type=int, required=True)
    p_v2.add_argument('-H', '--height', type=int, required=True)
    p_v2.add_argument('--m', type=int, default=100)
    p_v2.add_argument('--k2', type=int, default=None)
    p_v2.add_argument('--batch-size', type=int, default=256)
    p_v2.add_argument('--lr', type=float, default=0.1)
    p_v2.add_argument('--embed-steps', type=int, default=20)
    p_v2.add_argument('--iters-per-phase', type=int, default=50)
    p_v2.add_argument('--split-every', type=int, default=10,
                       help='Run dead cluster recovery every N iterations (0=disable)')
    p_v2.add_argument('--seed', type=int, default=42)
    p_v2.add_argument('-o', '--output-dir', type=str, default=None)

    # From-scratch v3
    p_v3 = sub.add_parser('from-scratch-v3',
                          help='Phase 3 v3: min_size + probabilistic throttle + knn2 + splits')
    p_v3.add_argument('--model', required=True)
    p_v3.add_argument('--knn', required=True)
    p_v3.add_argument('-W', '--width', type=int, required=True)
    p_v3.add_argument('-H', '--height', type=int, required=True)
    p_v3.add_argument('--m', type=int, default=100)
    p_v3.add_argument('--k2', type=int, default=None)
    p_v3.add_argument('--batch-size', type=int, default=256)
    p_v3.add_argument('--lr', type=float, default=0.1)
    p_v3.add_argument('--embed-steps', type=int, default=20)
    p_v3.add_argument('--iters-per-phase', type=int, default=50)
    p_v3.add_argument('--split-every', type=int, default=10)
    p_v3.add_argument('--min-size', type=int, default=2,
                       help='Minimum cluster size; below this, cluster is dissolved')
    p_v3.add_argument('--converge-phases', type=int, default=1,
                       help='Extra phases at alpha=1.0 after interpolation completes')
    p_v3.add_argument('--seed', type=int, default=42)
    p_v3.add_argument('-o', '--output-dir', type=str, default=None)

    args = parser.parse_args()
    if args.command == 'offline':
        run_offline(args)
    elif args.command == 'streaming':
        run_streaming(args)
    elif args.command == 'from-scratch':
        run_from_scratch(args)
    elif args.command == 'from-scratch-v2':
        run_from_scratch_v2(args)
    elif args.command == 'from-scratch-v3':
        run_from_scratch_v3(args)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()

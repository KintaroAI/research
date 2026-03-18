#!/usr/bin/env python3
"""Thalamus sorter: topographic map formation via correlation-based
neuron sorting. Explores multiple algorithms for arranging neurons
so that temporally correlated units become spatial neighbors.

Grid mode (no camera):
    python main.py greedy --width 80 --height 80 --k 24
    python main.py continuous --width 80 --height 80 --k 24 --lr 0.05
    python main.py mst --width 20 --height 20
    python main.py sa --width 20 --height 20

Camera mode (live video input):
    python main.py camera-sa --width 32 --height 24
    python main.py camera-spatial --width 16 --height 12
"""

import argparse
import os
import sys
import numpy as np
import cv2

from utils.weights import topk_decay2d
from utils.display import show_grid, wait, poll_quit
from solvers.drift_torch import DriftSolver
from legacy_runners import (run_greedy, run_continuous, run_temporal,
                             run_mst, run_sa, run_camera_sa,
                             run_camera_spatial, run_word2vec_legacy)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_image(path, width, height):
    """Load an image, resize to (width, height), convert to grayscale."""
    img = cv2.imread(path)
    if img is None:
        print(f"Error: could not load image '{path}'")
        sys.exit(1)
    resized = cv2.resize(img, (width, height))
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    return gray


def _eval_embeddings(emb, w, h):
    """Evaluate embedding quality: PCA Procrustes disparity + K-neighbor metrics."""
    from scipy.spatial import procrustes as scipy_procrustes, cKDTree
    n = emb.shape[0]
    grid = np.column_stack([np.arange(n) % w, np.arange(n) // w]).astype(np.float64)

    # PCA Procrustes
    _, _, Vt = np.linalg.svd(emb, full_matrices=False)
    pca_2d = (emb @ Vt[:2].T).astype(np.float64)
    _, _, pca_disp = scipy_procrustes(grid, pca_2d)

    # K=10 neighbors in embedding space → grid distance
    tree = cKDTree(emb)
    _, idx = tree.query(emb, k=11)
    idx = idx[:, 1:]  # remove self
    gx = np.arange(n) % w
    gy = np.arange(n) // w
    dists = np.abs(gx[idx] - gx[:, None]) + np.abs(gy[idx] - gy[:, None])
    mean_dist = float(dists.mean())
    within_3 = float((dists <= 3).mean())
    within_5 = float((dists <= 5).mean())

    result = {
        "pca_disparity": round(pca_disp, 4),
        "k10_mean_dist": round(mean_dist, 2),
        "k10_within_3px": round(within_3 * 100, 1),
        "k10_within_5px": round(within_5 * 100, 1),
    }
    print(f"  eval: PCA={pca_disp:.4f} K10: mean={mean_dist:.2f} "
          f"<3px={within_3*100:.1f}% <5px={within_5*100:.1f}%")
    return result


def _save_results_and_model(output_dir, args, dsolver, w, h, t0, max_frames,
                            total_pairs=None, wlog=None):
    """Common end-of-run: eval, info.json, model save."""
    import time
    if output_dir:
        s = dsolver.stats()
        results = {
            "ticks": max_frames,
            "std": round(s['std'], 4),
            "elapsed": round(time.time() - t0, 1),
        }
        if total_pairs is not None:
            results["total_pairs"] = total_pairs
        if getattr(args, 'eval', False):
            emb = dsolver.get_positions()
            results["eval"] = _eval_embeddings(emb, w, h)
            if wlog:
                e = results["eval"]
                wlog.log_eval(e["pca_disparity"], e["k10_mean_dist"],
                              e["k10_within_3px"], e["k10_within_5px"])
        if dsolver.knn_k > 0:
            results["knn"] = {
                "K": dsolver.knn_k,
                "history": dsolver.get_knn_history(),
            }
        _save_run_info(output_dir, args, results=results, wlog=wlog)

    # Save KNN lists alongside model
    if output_dir and dsolver.knn_k > 0:
        knn_path = os.path.join(output_dir, "knn_lists.npy")
        np.save(knn_path, dsolver.get_knn_lists())
        print(f"  knn saved: {knn_path}")

    if args.save_model or args.save_model_path:
        save_path = args.save_model_path
        if not save_path and output_dir:
            save_path = os.path.join(output_dir, "model.npy")
        if save_path:
            np.save(save_path, dsolver.get_positions())
            print(f"  model saved: {save_path}")


def _save_run_info(output_dir, args, results=None, wlog=None):
    """Save/update info.json with command, parameters, and results."""
    import json, subprocess, datetime
    info_path = os.path.join(output_dir, "info.json")

    if os.path.exists(info_path):
        with open(info_path) as f:
            info = json.load(f)
    else:
        # Git hash for reproducibility
        try:
            git_hash = subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                stderr=subprocess.DEVNULL).decode().strip()
        except Exception:
            git_hash = None

        info = {
            "command": " ".join(sys.argv),
            "timestamp": datetime.datetime.now().isoformat(),
            "git_hash": git_hash,
            "args": {k: v for k, v in vars(args).items()
                     if not k.startswith("_") and k != "func"},
        }

    if results:
        info["results"] = results

    if wlog and wlog.run_url:
        info["wandb_url"] = wlog.run_url

    with open(info_path, "w") as f:
        json.dump(info, f, indent=2, default=str)


class _ClusterManager:
    """Live streaming cluster maintenance during training."""

    def __init__(self, n, m, w, h, k2, lr, split_every, output_dir, wlog=None,
                 hysteresis=0.0, knn2_mode='incremental', centroid_mode='nudge',
                 max_k=1, track_history=False):
        import torch
        from cluster_experiments import (
            kmeans_cluster_gpu, _assign_clusters_gpu,
            streaming_update_v3_gpu, split_largest_cluster_gpu,
            visualize_clusters, eval_clusters,
        )
        self.n, self.m, self.w, self.h = n, m, w, h
        self.k2, self.lr, self.split_every = k2, lr, split_every
        self.max_k = max_k
        self.output_dir = output_dir
        self.initialized = False
        self.knn2_mode = knn2_mode  # 'incremental' or 'knn'
        self.centroid_mode = centroid_mode  # 'exact' or 'nudge'
        # Store function refs
        self._kmeans = kmeans_cluster_gpu
        self._assign = _assign_clusters_gpu
        self._stream_update = streaming_update_v3_gpu
        self._split = split_largest_cluster_gpu
        self._visualize = visualize_clusters
        self._eval = eval_clusters
        if knn2_mode == 'knn':
            from cluster_experiments import frequency_knn
            self._freq_knn = frequency_knn
        self.wlog = wlog
        self.hysteresis = hysteresis
        self.rng = np.random.RandomState(42)
        self.total_reassigned = 0
        self.total_switches = 0
        self.total_splits = 0
        self._prev_reassigned = 0
        self._prev_switches = 0
        self._prev_report_tick = 0
        self._prev_cluster_ids = None
        self.track_history = track_history
        self._history = [] if track_history else None

    def init_clusters(self, embeddings_t, knn_lists_np=None):
        """Initialize clusters via GPU k-means on current embeddings."""
        import torch
        self.device = embeddings_t.device
        ids_np, centroids_np = self._kmeans(embeddings_t, self.m, seed=42)
        self.cluster_ids = np.full((self.n, self.max_k), -1, dtype=np.int64)
        self.cluster_ids[:, 0] = ids_np
        self.pointers = np.zeros(self.n, dtype=np.int64)
        self.last_used = np.zeros((self.n, self.max_k), dtype=np.int64)
        self.cluster_ids_t = torch.from_numpy(ids_np.astype(np.int64)).to(self.device)
        self.centroids_t = torch.from_numpy(centroids_np).to(self.device)
        self.sizes = np.bincount(ids_np, minlength=self.m)

        if self.knn2_mode == 'knn':
            # Build knn2 from neuron-level KNN lists
            self.knn_lists = knn_lists_np
            self.knn2, _ = self._freq_knn(knn_lists_np, ids_np,
                                          self.m, self.k2)
        else:
            # Init knn2 with random cluster indices + infinity distances (GPU)
            knn2_np = np.full((self.m, self.k2), -1, dtype=np.int64)
            knn2_dists_np = np.full((self.m, self.k2), np.inf, dtype=np.float32)
            for c in range(self.m):
                if self.sizes[c] == 0:
                    continue
                others = [i for i in range(self.m) if i != c and self.sizes[i] > 0]
                if len(others) == 0:
                    continue
                k = min(self.k2, len(others))
                chosen = self.rng.choice(others, size=k, replace=False)
                knn2_np[c, :k] = chosen
                for j in range(k):
                    diff = centroids_np[c] - centroids_np[chosen[j]]
                    knn2_dists_np[c, j] = np.dot(diff, diff)
            self.knn2_t = torch.from_numpy(knn2_np).to(self.device)
            self.knn2_dists_t = torch.from_numpy(knn2_dists_np).to(self.device)

        self.initialized = True
        n_empty = (self.sizes == 0).sum()
        print(f"  Clusters initialized: m={self.m}, {self.m - n_empty} alive, "
              f"k2={self.k2}, knn2_mode={self.knn2_mode}, "
              f"centroid_mode={self.centroid_mode}")

    def tick(self, embeddings_t, anchors_np, pairs, global_tick,
             knn_lists_np=None):
        """One streaming cluster maintenance step."""
        import torch
        if not self.initialized:
            return

        effective_lr = self.lr

        if self.knn2_mode == 'knn':
            # knn mode: knn2 is neuron-index based (numpy)
            n_reassigned, affected, _, n_blocked, n_switches = self._stream_update(
                embeddings_t, self.centroids_t, self.cluster_ids, self.knn2,
                anchors_np, lr=effective_lr, sizes=self.sizes, min_size=0,
                rng=self.rng, hysteresis=self.hysteresis,
                knn2_is_neurons=True, centroid_mode=self.centroid_mode,
                pointers=self.pointers, last_used=self.last_used,
                tick=global_tick)
            self.total_reassigned += n_reassigned
            self.total_switches += n_switches
            # Patch knn2 for affected clusters from neuron-level KNN
            if affected and self.knn_lists is not None:
                most_recent = self.cluster_ids[np.arange(self.n), self.pointers]
                for c in affected:
                    if self.sizes[c] > 0:
                        members_c = np.where(most_recent == c)[0]
                        pooled = self.knn_lists[members_c].flatten()
                        ids, counts = np.unique(pooled, return_counts=True)
                        not_self = most_recent[ids] != c
                        ids, counts = ids[not_self], counts[not_self]
                        if len(ids) > 0:
                            top = min(self.k2, len(ids))
                            if top == len(ids):
                                top_idx = np.argsort(-counts)[:top]
                            else:
                                top_idx = np.argpartition(-counts, top)[:top]
                                top_idx = top_idx[np.argsort(-counts[top_idx])]
                            self.knn2[c, :] = -1
                            self.knn2[c, :top] = ids[top_idx]
                        else:
                            self.knn2[c, :] = -1
        else:
            # incremental mode: knn2 is cluster-index based (GPU tensors)
            knn2_np = self.knn2_t.cpu().numpy()
            n_reassigned, affected, _, n_blocked, n_switches = self._stream_update(
                embeddings_t, self.centroids_t, self.cluster_ids, knn2_np,
                anchors_np, lr=effective_lr, sizes=self.sizes, min_size=0,
                rng=self.rng, hysteresis=self.hysteresis,
                centroid_mode=self.centroid_mode,
                pointers=self.pointers, last_used=self.last_used,
                tick=global_tick)
            self.total_reassigned += n_reassigned
            self.total_switches += n_switches
            if affected:
                most_recent = self.cluster_ids[np.arange(self.n), self.pointers]
                self.cluster_ids_t = torch.from_numpy(
                    most_recent.astype(np.int64)).to(self.device)
                self._recompute_knn2_dists(list(affected))
            if pairs is not None:
                self._update_knn2_gpu(pairs)

        # Periodic split recovery
        if (self.split_every > 0 and global_tick > 0 and
                global_tick % self.split_every == 0):
            n_empty = (self.sizes == 0).sum()
            if n_empty > 0:
                n_to_split = min(n_empty, max(1, n_empty // 5))
                n_splits = self._split(
                    embeddings_t, self.centroids_t, self.cluster_ids,
                    self.sizes, self.m, n_splits=n_to_split,
                    seed=self.rng.randint(1000000),
                    pointers=self.pointers, last_used=self.last_used,
                    tick=global_tick)
                self.total_splits += n_splits
                if n_splits > 0:
                    most_recent = self.cluster_ids[np.arange(self.n), self.pointers]
                    if self.knn2_mode == 'knn' and self.knn_lists is not None:
                        self.knn2[:], _ = self._freq_knn(
                            self.knn_lists, most_recent, self.m, self.k2)
                    else:
                        self.cluster_ids_t = torch.from_numpy(
                            most_recent.astype(np.int64)).to(self.device)
                        self._recompute_knn2_dists()

    def _update_knn2_gpu(self, pairs):
        """Update knn2 from skip-gram pairs — fully vectorized on GPU."""
        import torch
        center_t = pairs[0].to(self.device)
        ctx_t = pairs[1].to(self.device)
        ca = self.cluster_ids_t[center_t]
        cn = self.cluster_ids_t[ctx_t]
        cross = ca != cn
        if not cross.any():
            return
        ca_x, cn_x = ca[cross], cn[cross]
        unique_packed = torch.unique(ca_x * self.m + cn_x)
        u_ca = unique_packed // self.m
        u_cn = unique_packed % self.m

        # Batch centroid distances
        diffs = self.centroids_t[u_ca] - self.centroids_t[u_cn]
        dists = (diffs * diffs).sum(dim=1)

        # Mask out already-in-knn2
        already_in = (u_cn.unsqueeze(1) == self.knn2_t[u_ca]).any(dim=1)
        dists = torch.where(already_in, torch.tensor(float('inf'),
                            device=self.device), dists)

        # Per-c_a best new candidate via scatter_reduce
        best_dist = torch.full((self.m,), float('inf'), device=self.device)
        best_dist.scatter_reduce_(0, u_ca, dists, reduce='amin',
                                  include_self=True)

        worst_knn2_dist, worst_knn2_slot = self.knn2_dists_t.max(dim=1)
        improved = best_dist < worst_knn2_dist
        if not improved.any():
            return

        # Find which unique entry achieved best_dist for each improved c_a
        is_best = (dists == best_dist[u_ca]) & improved[u_ca]
        bi = is_best.nonzero(as_tuple=True)[0]
        bi_ca = u_ca[bi]
        # First match per c_a
        first_per_ca = torch.full((self.m,), bi.shape[0],
                                  dtype=torch.long, device=self.device)
        first_per_ca.scatter_reduce_(
            0, bi_ca, torch.arange(bi.shape[0], device=self.device),
            reduce='amin', include_self=True)

        imp_ca = torch.where(improved & (first_per_ca < bi.shape[0]))[0]
        imp_ui = bi[first_per_ca[imp_ca]]
        self.knn2_t[imp_ca, worst_knn2_slot[imp_ca]] = u_cn[imp_ui]
        self.knn2_dists_t[imp_ca, worst_knn2_slot[imp_ca]] = dists[imp_ui]

    def _recompute_knn2_dists(self, rows=None):
        """Recompute knn2 distances on GPU. rows=None → all rows."""
        import torch
        if rows is None:
            rows_t = torch.arange(self.m, device=self.device)
        else:
            rows_t = torch.tensor(rows, dtype=torch.long, device=self.device)
        targets = self.knn2_t[rows_t]  # (len, k2)
        valid = targets >= 0
        # Clamp for gather (invalid entries get dummy centroid, overwritten with inf)
        safe_targets = targets.clamp(min=0)
        diffs = self.centroids_t[rows_t].unsqueeze(1) - self.centroids_t[safe_targets]
        d = (diffs * diffs).sum(dim=2)
        d[~valid] = float('inf')
        self.knn2_dists_t[rows_t] = d

    def report(self, tick):
        """Print cluster metrics and save visualization."""
        if not self.initialized:
            return
        most_recent = self.cluster_ids[np.arange(self.n), self.pointers]
        centroids_np = self.centroids_t.cpu().numpy()
        if self.knn2_mode == 'knn':
            # Convert neuron-index knn2 to cluster-index for eval
            knn2_np = self.knn2.copy()
            valid = knn2_np >= 0
            knn2_np[valid] = most_recent[knn2_np[valid]]
            metrics = self._eval(most_recent, centroids_np, knn2_np,
                                 self.w, self.h, self.knn_lists)
        else:
            knn2_np = self.knn2_t.cpu().numpy()
            metrics = self._eval(most_recent, centroids_np, knn2_np,
                                 self.w, self.h)
        n_empty = metrics['n_empty']
        interval_reassigned = self.total_reassigned - self._prev_reassigned
        interval_ticks = max(1, tick - self._prev_report_tick)
        jumps_per_tick = interval_reassigned / interval_ticks
        self._prev_reassigned = self.total_reassigned
        self._prev_report_tick = tick
        n_alive = self.m - n_empty
        # Neuron stability: fraction that stayed in same cluster since last report
        if self._prev_cluster_ids is not None:
            stability = (most_recent == self._prev_cluster_ids).mean()
        else:
            stability = 0.0
        self._prev_cluster_ids = most_recent.copy()
        if self._history is not None:
            self._history.append((tick, most_recent.copy()))
        interval_switches = self.total_switches - self._prev_switches
        switches_per_tick = interval_switches / interval_ticks
        self._prev_switches = self.total_switches
        print(f"  Clusters @ tick {tick}: {n_alive}/{self.m} alive, "
              f"contiguity={metrics['contiguity_mean']:.3f}, "
              f"diam={metrics['diameter_mean']:.1f}, "
              f"stability={stability:.3f}, "
              f"jumps/tick={jumps_per_tick:.1f}, "
              f"switches/tick={switches_per_tick:.1f}, "
              f"total_jumps={self.total_reassigned}, "
              f"total_switches={self.total_switches}, splits={self.total_splits}")
        if self.wlog:
            self.wlog.log_clusters(
                tick, n_alive, self.m, metrics['contiguity_mean'],
                metrics['diameter_mean'], jumps_per_tick,
                self.total_reassigned, self.total_splits,
                stability=stability,
                switches_per_tick=switches_per_tick,
                total_switches=self.total_switches)
        if self.output_dir:
            path = os.path.join(self.output_dir, f"clusters_{tick:06d}.png")
            self._visualize(most_recent, self.w, self.h, path)

    def save(self, output_dir):
        """Save cluster state at end of run."""
        if not self.initialized:
            return
        np.save(os.path.join(output_dir, "cluster_ids.npy"), self.cluster_ids)
        np.save(os.path.join(output_dir, "pointers.npy"), self.pointers)
        np.save(os.path.join(output_dir, "last_used.npy"), self.last_used)
        np.save(os.path.join(output_dir, "centroids.npy"),
                self.centroids_t.cpu().numpy())
        if self.knn2_mode == 'knn':
            np.save(os.path.join(output_dir, "knn2.npy"), self.knn2)
        else:
            np.save(os.path.join(output_dir, "knn2.npy"), self.knn2_t.cpu().numpy())
        if self._history:
            ticks = np.array([t for t, _ in self._history], dtype=np.int64)
            ids = np.stack([h for _, h in self._history])
            np.save(os.path.join(output_dir, "history_ticks.npy"), ticks)
            np.save(os.path.join(output_dir, "history_ids.npy"), ids)
            print(f"  cluster history saved: {len(self._history)} snapshots")
        print(f"  cluster state saved to {output_dir}")


def _run_training_loop(do_tick, dsolver, args, n, w, sig_channels, wlog,
                       on_save=None, on_display=None, can_break=None,
                       cluster_mgr=None):
    """Shared training loop for async and sync render paths.

    Callbacks:
        on_save(tick, total_pairs) — called every save_every ticks
        on_display(tick) — called every tick (caller filters as needed)
        can_break() — return True to exit loop early

    Returns (t0, total_pairs).
    """
    import time
    t0 = time.time()
    max_frames = args.frames
    total_pairs = 0
    prev_pairs = 0
    knn_report_every = getattr(args, 'knn_report_every', 1000)
    log_every = getattr(args, 'log_every', 1000)
    cluster_report_every = getattr(args, 'cluster_report_every', 1000)
    t_log = t0

    for tick in range(1, max_frames + 1):
        total_pairs += do_tick()

        # Live cluster maintenance
        if cluster_mgr is not None:
            if not cluster_mgr.initialized:
                if cluster_mgr.knn2_mode == 'knn':
                    if dsolver.knn_k > 0:
                        knn_np = dsolver.get_knn_lists()
                        cluster_mgr.init_clusters(dsolver.positions, knn_np)
                    else:
                        print("  Warning: knn2_mode='knn' requires --knn-track, "
                              "falling back to incremental")
                        cluster_mgr.knn2_mode = 'incremental'
                        cluster_mgr.init_clusters(dsolver.positions)
                else:
                    cluster_mgr.init_clusters(dsolver.positions)
            if cluster_mgr.initialized:
                anchors_np = dsolver._last_anchors.cpu().numpy()
                pairs = getattr(dsolver, '_last_pairs', None)
                cluster_mgr.tick(dsolver.positions, anchors_np, pairs, tick)
                if tick % cluster_report_every == 0:
                    # Refresh knn_lists for knn mode
                    if cluster_mgr.knn2_mode == 'knn' and dsolver.knn_k > 0:
                        cluster_mgr.knn_lists = dsolver.get_knn_lists()
                    cluster_mgr.report(tick)

        # Tick progress logging
        if log_every > 0 and tick % log_every == 0:
            now = time.time()
            elapsed = now - t0
            ms_tick = (now - t_log) / log_every * 1000
            pairs_per_tick = (total_pairs - prev_pairs) / log_every
            print(f"  tick {tick}/{max_frames} "
                  f"({elapsed:.1f}s, {ms_tick:.1f} ms/tick, "
                  f"pairs={total_pairs}, {pairs_per_tick:.0f}/tick)")
            wlog.log_tick(tick, elapsed, total_pairs, ms_tick, pairs_per_tick)
            t_log = now
            prev_pairs = total_pairs

        # KNN stability reporting
        if dsolver.knn_k > 0 and tick % knn_report_every == 0:
            dsolver._refresh_knn_dists()
            overlap, n_changed, top50_swaps, top90_swaps = dsolver.knn_stability()
            spatial_acc = dsolver.knn_spatial_accuracy(w, radius=3, channels=sig_channels)
            dsolver._knn_overlap_history.append((tick, overlap, spatial_acc))
            lr_str = f" lr={dsolver.lr:.6f}" if dsolver.lr_decay < 1.0 else ""
            print(f"  KNN @ tick {tick}: overlap={overlap:.3f} "
                  f"spatial={spatial_acc:.3f} "
                  f"({n_changed}/{n} changed) "
                  f"swaps: top50={top50_swaps:.1f} top90={top90_swaps:.1f}{lr_str}")
            wlog.log_knn(tick, overlap, spatial_acc, n_changed, n,
                         top50_swaps, top90_swaps,
                         lr=dsolver.lr if dsolver.lr_decay < 1.0 else None)

        if on_save is not None and tick % args.save_every == 0:
            on_save(tick, total_pairs)

        if on_display is not None:
            on_display(tick)

        if can_break is not None and can_break():
            break

    return t0, total_pairs


def _render_worker(shm_buf0, shm_buf1, shm_active, shm_tick,
                   shm_done, n, dims, w, h, pixel_values,
                   render_method, do_align, cold_proj, output_dir,
                   gpu=False):
    """Pull-based render worker. Reads from active double-buffer slot."""
    import time
    import os
    import cv2
    import numpy as np
    from render_embeddings import (project, align_to_grid,
                                   render as render_emb)
    bufs = [shm_buf0, shm_buf1]
    prev_2d = None
    last_rendered_tick = 0
    frame_idx = 0

    while True:
        cur_tick = shm_tick.value
        done = shm_done.value

        if cur_tick > last_rendered_tick:
            slot = shm_active.value
            emb = np.frombuffer(bufs[slot].get_obj(),
                                dtype=np.float32).reshape(n, dims).copy()
            last_rendered_tick = cur_tick

            warm = None if cold_proj else prev_2d
            pos_2d = project(emb, w, h, render_method, prev_2d=warm,
                             gpu=gpu)
            if do_align:
                pos_2d = align_to_grid(pos_2d, w, h)
            if not cold_proj:
                prev_2d = pos_2d.copy()
            frame = render_emb(pos_2d, w, h, pixel_values)

            if frame is not None:
                normalized = cv2.normalize(
                    frame, None, 0, 255, cv2.NORM_MINMAX,
                    dtype=cv2.CV_8U)
                path = os.path.join(output_dir,
                                    f"frame_{frame_idx:06d}.png")
                cv2.imwrite(path, normalized)
                frame_idx += 1

        elif done:
            break
        else:
            time.sleep(0.001)

    print(f"  render worker: {frame_idx} frames saved, "
          f"last tick={last_rendered_tick}")


def run_word2vec(args):
    from utils.wandb_logger import WandbLogger
    # Parse comma-separated tags
    if getattr(args, 'wandb_tags', None):
        args.wandb_tags = [t.strip() for t in args.wandb_tags.split(',')]
    wlog = WandbLogger(args)

    w, h = args.width, args.height

    image = None
    if args.image:
        image = _load_image(args.image, w, h)

    mode = args.mode
    norm_every = args.normalize_every

    assert not getattr(args, 'use_mse', False), \
        "--use-mse is deprecated. Use --use-deriv-corr with threshold=0.5 instead."

    # Resolve anchor_sample from --anchor-sample or --anchor-batches
    if getattr(args, 'anchor_sample', None) is not None:
        anchor_sample = args.anchor_sample
    elif getattr(args, 'anchor_batches', None) is not None:
        anchor_sample = args.anchor_batches * args.batch_size
    else:
        anchor_sample = args.batch_size  # default: 1 batch

    if mode in ("sentence", "correlation"):
        from render_embeddings import project, align_to_grid, render as render_emb
        import time
        import torch

        sig_channels = getattr(args, 'signal_channels', 1)
        n = w * h * sig_channels

        # --- Pixel values for rendering (no model dependency) ---
        pixel_values = None
        if image is not None:
            if sig_channels > 1:
                channel_tints = {
                    0: np.array([0.3, 0.3, 1.0]),  # R -> red (BGR)
                    1: np.array([0.3, 1.0, 0.3]),  # G -> green (BGR)
                    2: np.array([1.0, 0.3, 0.3]),  # B -> blue (BGR)
                    3: np.array([0.8, 0.8, 0.8]),  # GS -> gray (BGR)
                }
                pixel_values = np.zeros((n, 3), dtype=np.uint8)
                for i in range(n):
                    px = i // sig_channels
                    ch = i % sig_channels
                    gray = float(image[px // w, px % w])
                    tint = channel_tints.get(ch, np.array([1.0, 1.0, 1.0]))
                    pixel_values[i] = np.clip(gray * tint, 0, 255).astype(np.uint8)
            else:
                pixel_values = np.zeros(n, dtype=np.uint8)
                for i in range(n):
                    pixel_values[i] = image[i // w, i % w]

        render_w = w * sig_channels
        render_h = h

        output_dir = args.output_dir
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            _save_run_info(output_dir, args)

        render_method = args.render
        if render_method == 'euclidean':
            render_method = 'pca'

        if getattr(args, 'sync_render', False):
            args.async_render = False

        # --- Spawn render worker early so its imports overlap with model init ---
        worker = None
        bufs = None
        shm_active = shm_tick = shm_done = None
        if args.async_render and output_dir:
            import multiprocessing as mp
            try:
                mp.set_start_method('spawn')
            except RuntimeError:
                pass  # already set

            buf_size = n * args.dims
            shm_buf0 = mp.Array('f', buf_size)
            shm_buf1 = mp.Array('f', buf_size)
            shm_active = mp.Value('i', 0)
            shm_tick = mp.Value('i', 0)
            shm_done = mp.Value('i', 0)

            bufs = [shm_buf0, shm_buf1]
            worker = mp.Process(target=_render_worker,
                                args=(shm_buf0, shm_buf1, shm_active, shm_tick,
                                      shm_done, n, args.dims,
                                      render_w, render_h, pixel_values,
                                      render_method,
                                      args.align, args.cold_projection,
                                      output_dir, args.render_gpu))
            worker.start()

        # --- Create solver (imports torch/cuml in main process) ---
        if mode == "sentence":
            k = min(args.k, n - 1)
            top_k = topk_decay2d(w, h, k)
            device = 'cuda' if (args.gpu and torch.cuda.is_available()) else 'cpu'
            dsolver = DriftSolver(n, top_k=top_k, dims=args.dims, lr=args.lr,
                                  mode='dot', k_neg=args.k_neg,
                                  normalize_every=norm_every, device=device)
        else:
            knn_k = getattr(args, 'knn_track', 0)
            lr_decay = getattr(args, 'lr_decay', 1.0)
            knn_nofn = getattr(args, 'knn_nofn', False)
            device = 'cuda' if (args.gpu and torch.cuda.is_available()) else 'cpu'
            dsolver = DriftSolver(n, top_k=None, k=args.k, dims=args.dims,
                                  lr=args.lr, mode='dot', k_neg=args.k_neg,
                                  normalize_every=norm_every, device=device,
                                  knn_k=knn_k, lr_decay=lr_decay,
                                  knn_nofn=knn_nofn)

        if args.warm_start:
            warm = np.load(args.warm_start)
            dsolver.positions = torch.from_numpy(warm).to(dsolver.device)
            print(f"Warm start from {args.warm_start} ({warm.shape})")

        # --- Correlation mode: build signal buffer ---
        saccade_source = None
        if mode == "correlation":
            T = args.signal_T
            sigma = args.signal_sigma
            print(f"Word2vec drift (correlation): {w}x{h} grid, "
                  f"k_sample={args.k_sample}, threshold={args.threshold}, "
                  f"k_neg={args.k_neg}, lr={args.lr}, dims={args.dims}, "
                  f"signal: T={T}, sigma={sigma}, "
                  f"normalize_every={norm_every}, align={args.align}")

            signals_np = np.zeros((n, T), dtype=np.float32)

            if args.signal_source:
                if args.signal_source.endswith('.png') or args.signal_source.endswith('.jpg'):
                    img_bgr = cv2.imread(args.signal_source)
                    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
                    if sig_channels == 4:
                        gs = 0.299 * img_rgb[:,:,0] + 0.587 * img_rgb[:,:,1] + 0.114 * img_rgb[:,:,2]
                        source = np.concatenate([img_rgb, gs[:,:,np.newaxis]], axis=2)
                    elif sig_channels == 3:
                        source = img_rgb
                    else:
                        source = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
                else:
                    source = np.load(args.signal_source)

                if source.ndim == 3:
                    src_h, src_w, src_c = source.shape
                    assert src_c == sig_channels, \
                        f"Source has {src_c} channels but --signal-channels={sig_channels}"
                    crop_h, crop_w = h, w
                    print(f"  multi-channel: {sig_channels}ch, "
                          f"crop={crop_w}x{crop_h} pixels -> {n} neurons")
                else:
                    src_h, src_w = source.shape
                    crop_h, crop_w = h, w

                max_dy = src_h - crop_h
                max_dx = src_w - crop_w
                assert max_dy > 0 and max_dx > 0, \
                    f"Source {src_w}x{src_h} too small for {crop_w}x{crop_h} crop"
                saccade_step = args.saccade_step
                use_raw = args.use_mse or args.use_deriv_corr
                walk_dy = np.random.randint(0, max_dy + 1)
                walk_dx = np.random.randint(0, max_dx + 1)
                for t in range(T):
                    walk_dy = np.clip(walk_dy + np.random.randint(-saccade_step, saccade_step + 1),
                                      0, max_dy)
                    walk_dx = np.clip(walk_dx + np.random.randint(-saccade_step, saccade_step + 1),
                                      0, max_dx)
                    crop = source[walk_dy:walk_dy+crop_h, walk_dx:walk_dx+crop_w].ravel()
                    if use_raw:
                        signals_np[:, t] = crop
                    else:
                        signals_np[:, t] = crop - crop.mean()
                src_desc = f"{src_w}x{src_h}" + (f"x{sig_channels}" if source.ndim == 3 else "")
                crop_desc = f"{crop_w}x{crop_h}" + (f"x{sig_channels}" if source.ndim == 3 else "")
                mean_sub = "raw" if use_raw else "mean-subtracted"
                print(f"  signal buffer: ({n}, {T}), rolling saccades from "
                      f"{args.signal_source} ({src_desc}), "
                      f"crop={crop_desc}, step={saccade_step}, {mean_sub}")
                saccade_source = torch.from_numpy(source).to(dsolver.device)
            else:
                from scipy.ndimage import gaussian_filter
                for t in range(T):
                    noise = np.random.randn(h, w).astype(np.float32)
                    smoothed = gaussian_filter(noise, sigma=sigma)
                    signals_np[:, t] = smoothed.ravel()
                print(f"  signal buffer: ({n}, {T}), spatial sigma={sigma}")

            signals = torch.from_numpy(signals_np).to(dsolver.device)
            tick_counter = [0]
        else:
            print(f"Word2vec drift (sentence): {w}x{h} grid, k={args.k}, "
                  f"k_neg={args.k_neg}, lr={args.lr}, dims={args.dims}, "
                  f"window={args.window}, normalize_every={norm_every}, "
                  f"align={args.align}, async={args.async_render}")

        # --- Tick function ---
        if mode == "correlation":
            def do_tick():
                if saccade_source is not None:
                    nonlocal walk_dy, walk_dx
                    walk_dy = np.clip(walk_dy + np.random.randint(-saccade_step, saccade_step + 1),
                                      0, max_dy)
                    walk_dx = np.clip(walk_dx + np.random.randint(-saccade_step, saccade_step + 1),
                                      0, max_dx)
                    crop = saccade_source[walk_dy:walk_dy+crop_h, walk_dx:walk_dx+crop_w].reshape(-1)
                    col = tick_counter[0] % T
                    if use_raw:
                        signals[:, col] = crop
                    else:
                        signals[:, col] = crop - crop.mean()
                    tick_counter[0] += 1
                return dsolver.tick_correlation(
                    signals, k_sample=args.k_sample,
                    threshold=args.threshold, window=args.window,
                    anchor_only=args.anchor_only,
                    use_covariance=args.use_covariance,
                    use_mse=args.use_mse,
                    use_deriv_corr=args.use_deriv_corr,
                    max_hit_ratio=args.max_hit_ratio,
                    batch_size=args.batch_size,
                    anchor_sample=anchor_sample,
                    fp16=getattr(args, 'fp16', False),
                    matmul_corr=getattr(args, 'matmul_corr', True))
        else:
            def do_tick():
                dsolver.tick_sentence(window=args.window)
                return 0

        # --- Live clustering ---
        cluster_mgr = None
        cluster_m = getattr(args, 'cluster_m', 0)
        if cluster_m > 0:
            cluster_k2 = getattr(args, 'cluster_k2', 16)
            cluster_hyst = getattr(args, 'cluster_hysteresis', 0.0)
            knn2_mode = getattr(args, 'cluster_knn2_mode', 'incremental')
            centroid_mode = getattr(args, 'cluster_centroid_mode', 'nudge')
            cluster_max_k = getattr(args, 'cluster_max_k', 1)
            cluster_mgr = _ClusterManager(
                n, cluster_m, w, h, k2=cluster_k2,
                lr=getattr(args, 'cluster_lr', 1.0),
                split_every=getattr(args, 'cluster_split_every', 10),
                output_dir=output_dir, wlog=wlog,
                hysteresis=cluster_hyst, knn2_mode=knn2_mode,
                centroid_mode=centroid_mode,
                max_k=cluster_max_k,
                track_history=getattr(args, 'cluster_track_history', False))
            print(f"Live clustering enabled: m={cluster_m}, k2={cluster_k2}, "
                  f"max_k={cluster_max_k}, "
                  f"hysteresis={cluster_hyst}, knn2={knn2_mode}, "
                  f"centroid={centroid_mode}, "
                  f"report_every={getattr(args, 'cluster_report_every', 1000)}")

        # --- Training + rendering ---
        if worker is not None:
            write_slot = [1]

            def on_save_async(tick, total_pairs):
                emb = dsolver.get_positions()
                dst = np.frombuffer(bufs[write_slot[0]].get_obj(),
                                    dtype=np.float32)
                np.copyto(dst, emb.ravel())
                shm_active.value = write_slot[0]
                shm_tick.value = tick
                write_slot[0] = 1 - write_slot[0]

            t0, total_pairs = _run_training_loop(
                do_tick, dsolver, args, n, w, sig_channels, wlog,
                on_save=on_save_async, cluster_mgr=cluster_mgr)

            # Final snapshot
            emb = dsolver.get_positions()
            dst = np.frombuffer(bufs[write_slot[0]].get_obj(), dtype=np.float32)
            np.copyto(dst, emb.ravel())
            shm_active.value = write_slot[0]
            shm_tick.value = args.frames + 1
            time.sleep(0.01)
            shm_done.value = 1

            elapsed_train = time.time() - t0
            s = dsolver.stats()
            print(f"Training done: {args.frames} ticks in {elapsed_train:.1f}s, "
                  f"std={s['std']:.4f}, total_pairs={total_pairs}")
            wlog.log_done(args.frames, elapsed_train, s['std'], total_pairs)

            worker.join()
            elapsed_total = time.time() - t0
            print(f"Total (train + render drain): {elapsed_total:.1f}s")

        else:
            prev_2d = None

            def render_frame():
                nonlocal prev_2d
                emb = dsolver.get_positions()
                warm = None if args.cold_projection else prev_2d
                pos_2d = project(emb, render_w, render_h, render_method,
                                 prev_2d=warm, gpu=args.render_gpu)
                if args.align:
                    pos_2d = align_to_grid(pos_2d, render_w, render_h)
                if not args.cold_projection:
                    prev_2d = pos_2d.copy()
                return render_emb(pos_2d, render_w, render_h, pixel_values)

            saved = [0]

            def on_save_sync(tick, total_pairs):
                frame = render_frame()
                if frame is not None:
                    normalized = cv2.normalize(
                        frame, None, 0, 255, cv2.NORM_MINMAX,
                        dtype=cv2.CV_8U)
                    path = os.path.join(output_dir, f"frame_{saved[0]:06d}.png")
                    cv2.imwrite(path, normalized)
                    saved[0] += 1
                    if saved[0] % 100 == 0:
                        s = dsolver.stats()
                        print(f"  tick {tick}, saved {saved[0]} frames, "
                              f"std={s['std']:.4f}, pairs={total_pairs}")

            def on_display_sync(tick):
                if tick % 10 == 0:
                    if pixel_values is not None:
                        frame = render_frame()
                        if frame is not None:
                            show_grid(f"{mode} skip-gram", frame)
                    wait()

            t0, total_pairs = _run_training_loop(
                do_tick, dsolver, args, n, w, sig_channels, wlog,
                on_save=on_save_sync if output_dir else None,
                on_display=on_display_sync if not output_dir else None,
                can_break=poll_quit, cluster_mgr=cluster_mgr)

            elapsed = time.time() - t0
            s = dsolver.stats()
            print(f"Done: {args.frames} ticks in {elapsed:.1f}s, "
                  f"std={s['std']:.4f}, total_pairs={total_pairs}")
            wlog.log_done(args.frames, elapsed, s['std'], total_pairs)

            if output_dir:
                frame = render_frame()
                if frame is not None:
                    normalized = cv2.normalize(
                        frame, None, 0, 255, cv2.NORM_MINMAX,
                        dtype=cv2.CV_8U)
                    path = os.path.join(output_dir, f"frame_{saved[0]:06d}.png")
                    cv2.imwrite(path, normalized)
                    print(f"  final frame saved: {path}")

        if cluster_mgr is not None and cluster_mgr.initialized:
            cluster_mgr.report(args.frames)
            if output_dir:
                cluster_mgr.save(output_dir)

        _save_results_and_model(output_dir, args, dsolver, render_w, render_h,
                               t0, args.frames, total_pairs=total_pairs,
                               wlog=wlog)
        wlog.finish()
        return

    # Legacy modes: similarity, dual, dual-xy, skipgram
    run_word2vec_legacy(args)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Thalamus sorter: topographic map formation algorithms")
    sub = parser.add_subparsers(dest="command", required=True)

    # Common args added to each subparser
    def add_common(p):
        p.add_argument("--width", "-W", type=int, default=40)
        p.add_argument("--height", "-H", type=int, default=40)
        p.add_argument("--weight-type", choices=["inv1d", "decay2d"],
                        default="decay2d",
                        help="Weight matrix type (default: decay2d)")
        p.add_argument("--decay", type=float, default=0.1,
                        help="Decay rate for decay2d weights (default: 0.1)")

    # --- greedy ---
    p_greedy = sub.add_parser("greedy", help="Greedy drift sorting")
    add_common(p_greedy)
    p_greedy.add_argument("--k", type=int, default=24,
                          help="Number of nearest neighbors (default: 24)")
    p_greedy.add_argument("--move-fraction", type=float, default=0.9,
                          help="Fraction of neurons to move per tick (default: 0.9)")
    p_greedy.add_argument("--image", "-i", type=str, default=None,
                          help="Input image to scramble and reconstruct")
    p_greedy.add_argument("--frames", "-f", type=int, default=0,
                          help="Number of frames to run (0 = unlimited)")
    p_greedy.add_argument("--output-dir", "-o", type=str, default=None,
                          help="Directory to save output frames as PNGs")
    p_greedy.add_argument("--save-every", type=int, default=1000,
                          help="Save every Nth frame (default: 1)")
    p_greedy.add_argument("--gpu", action="store_true",
                          help="Use GPU acceleration via CuPy")
    p_greedy.set_defaults(func=run_greedy)

    # --- continuous ---
    p_cont = sub.add_parser("continuous",
                            help="Continuous position drift (embedding-style)")
    add_common(p_cont)
    p_cont.add_argument("--k", type=int, default=24,
                        help="Number of nearest neighbors (default: 24)")
    p_cont.add_argument("--lr", type=float, default=0.05,
                        help="Learning rate / step fraction (default: 0.05)")
    p_cont.add_argument("--dims", type=int, default=2,
                        help="Position vector dimensionality (default: 2)")
    p_cont.add_argument("--margin", type=float, default=0.1,
                        help="Dead zone radius around centroid (default: 0.1)")
    p_cont.add_argument("--image", "-i", type=str, default=None,
                        help="Input image to scramble and reconstruct")
    p_cont.add_argument("--frames", "-f", type=int, default=0,
                        help="Number of frames to run (0 = unlimited)")
    p_cont.add_argument("--output-dir", "-o", type=str, default=None,
                        help="Directory to save output frames as PNGs")
    p_cont.add_argument("--save-every", type=int, default=1000,
                        help="Save every Nth frame (default: 1)")
    p_cont.add_argument("--gpu", action="store_true",
                        help="Use GPU acceleration via CuPy")
    p_cont.set_defaults(func=run_continuous)

    # --- word2vec ---
    p_w2v = sub.add_parser("word2vec",
                           help="Word2vec-style drift (skip-gram or similarity mode)")
    p_w2v.add_argument("--preset", type=str, default=None,
                       help="Load parameter preset from presets/ directory (e.g. 'gray_80x80'). "
                            "CLI args override preset values.")
    p_w2v.add_argument("--width", "-W", type=int, default=40,
                       help="Grid width (default: 40)")
    p_w2v.add_argument("--height", "-H", type=int, default=40,
                       help="Grid height (default: 40)")
    p_w2v.add_argument("--mode", choices=["skipgram", "similarity", "dual", "dual-xy", "sentence", "correlation"],
                       default="similarity",
                       help="Update mode: skipgram, similarity, dual, dual-xy, sentence (precomputed neighbors), correlation (online neighbor discovery from signals) (default: similarity)")
    p_w2v.add_argument("--window", type=int, default=5,
                       help="Sliding window size for sentence mode (default: 5)")
    p_w2v.add_argument("--normalize-every", type=int, default=0,
                       help="Normalize W/C vectors every N ticks (0 = disabled, default: 0)")
    # skipgram mode args
    p_w2v.add_argument("--k", type=int, default=24,
                       help="Number of nearest neighbors for positive sampling (skipgram mode, default: 24)")
    p_w2v.add_argument("--k-neg", type=int, default=5,
                       help="Number of negative samples per positive (skipgram mode, default: 5)")
    # similarity mode args
    p_w2v.add_argument("--P", type=int, default=10,
                       help="Random peers per neuron per tick (similarity mode, default: 10)")
    p_w2v.add_argument("--sigma", type=float, default=5.0,
                       help="Gaussian RBF kernel width (similarity mode, default: 5.0)")
    p_w2v.add_argument("--threshold", type=float, default=0.0,
                       help="Similarity/correlation threshold (default: 0.0)")
    # correlation mode args
    p_w2v.add_argument("--k-sample", type=int, default=50,
                       help="Random candidates to check per neuron (correlation mode, default: 50)")
    p_w2v.add_argument("--batch-size", type=int, default=256,
                       help="GPU batch size for anchor processing (default: 256)")
    anchor_group = p_w2v.add_mutually_exclusive_group()
    anchor_group.add_argument("--anchor-sample", type=int, default=None,
                       help="Total unique anchor neurons per tick (split into batch_size chunks)")
    anchor_group.add_argument("--anchor-batches", type=int, default=None,
                       help="Number of batches per tick (anchor_sample = anchor_batches * batch_size)")
    p_w2v.add_argument("--signal-T", type=int, default=100,
                       help="Temporal signal buffer length (correlation mode, default: 100)")
    p_w2v.add_argument("--signal-sigma", type=float, default=3.0,
                       help="Gaussian smoothing sigma for signal generation (correlation mode, default: 3.0)")
    p_w2v.add_argument("--signal-source", type=str, default=None,
                       help="Path to signal source: .npy (grayscale) or .png/.jpg (auto RGB)")
    p_w2v.add_argument("--signal-channels", type=int, default=1,
                       help="Channels per pixel from source: 1=gray, 3=RGB, 4=RGBG (default: 1). "
                            "n = W*H*channels. PNG auto-loads as RGB.")
    p_w2v.add_argument("--saccade-step", type=int, default=5,
                       help="Max pixels to shift per timestep in saccade mode (default: 5)")
    p_w2v.add_argument("--use-covariance", action="store_true",
                       help="Use covariance (corr×std1×std2) instead of Pearson correlation; downweights flat regions")
    p_w2v.add_argument("--use-mse", action="store_true",
                       help="Use MSE as distance metric (lower=more similar). "
                            "No per-frame global mean needed. Threshold ~0.02.")
    p_w2v.add_argument("--use-deriv-corr", action="store_true",
                       help="Pearson correlation on temporal derivatives. "
                            "Dead neurons get score=0 (no variance gate needed). "
                            "Threshold ~0.3-0.5 (higher=more similar).")
    p_w2v.add_argument("--max-hit-ratio", type=float, default=None,
                       help="Discard anchors where neighbors/k_sample exceeds this ratio (e.g. 0.1). "
                            "Filters out global signals — if a neuron correlates with everyone, skip it.")
    p_w2v.add_argument("--anchor-only", action="store_true",
                       help="Correlation mode: only (anchor, neighbor) pairs, no transitive sliding window")
    p_w2v.add_argument("--knn-track", type=int, default=0,
                       help="Track per-neuron KNN list of this size (0=off). "
                            "Monitors embedding convergence via neighbor stability.")
    p_w2v.add_argument("--knn-report-every", type=int, default=1000,
                       help="Report KNN stability every N ticks (default: 1000)")
    p_w2v.add_argument("--log-every", type=int, default=1000,
                       help="Print tick progress every N ticks (default: 1000)")
    p_w2v.add_argument("--lr-decay", type=float, default=1.0,
                       help="Multiply lr by this factor at each normalization event (default: 1.0 = no decay)")
    p_w2v.add_argument("--knn-nofn", action="store_true",
                       help="Add neighbor-of-neighbor candidates to correlation probing. "
                            "Requires --knn-track. Breaks O(n²) scaling.")
    p_w2v.add_argument("--render", choices=["euclidean", "angular", "bestpc",
                                            "direct", "procrustes", "lstsq",
                                            "umap", "tsne", "spectral", "mds"],
                       default="euclidean",
                       help="Render projection: euclidean/pca (top-2 PCs), "
                            "bestpc (grid-correlated PCs), angular (unit norm + PCA), "
                            "direct (first 2 dims), procrustes/lstsq (supervised linear), "
                            "umap/tsne/spectral/mds (nonlinear) (default: euclidean)")
    p_w2v.add_argument("--align", action="store_true",
                       help="Procrustes-align rendered output to grid (fixes rotation/flip)")
    p_w2v.add_argument("--warm-start", type=str, default=None,
                       help="Load .npy embeddings as initial positions (warm start)")
    p_w2v.add_argument("--cold-projection", action=argparse.BooleanOptionalAction, default=True,
                       help="Run UMAP/t-SNE from scratch each frame (--no-cold-projection for warm start)")
    p_w2v.add_argument("--async-render", action="store_true", default=True,
                       help="Render in separate process (default: on)")
    p_w2v.add_argument("--sync-render", action="store_true",
                       help="Force synchronous rendering")
    p_w2v.add_argument("--eval", action="store_true",
                       help="Evaluate embeddings (PCA Procrustes + K-neighbor) and save to info.json")
    p_w2v.add_argument("--save-model", action="store_true",
                       help="Save final embeddings to .npy file")
    p_w2v.add_argument("--save-model-path", type=str, default=None,
                       help="Path for saved model (default: output_dir/model.npy)")
    # common args
    p_w2v.add_argument("--lr", type=float, default=0.05,
                       help="Learning rate (default: 0.05)")
    p_w2v.add_argument("--dims", type=int, default=2,
                       help="Position vector dimensionality (default: 2)")
    p_w2v.add_argument("--image", "-i", type=str, default=None,
                       help="Input image to scramble and reconstruct")
    p_w2v.add_argument("--frames", "-f", type=int, default=0,
                       help="Number of frames to run (0 = unlimited)")
    p_w2v.add_argument("--output-dir", "-o", type=str, default=None,
                       help="Directory to save output frames as PNGs")
    p_w2v.add_argument("--save-every", type=int, default=1000,
                       help="Save every Nth frame (default: 1000)")
    p_w2v.add_argument("--gpu", action=argparse.BooleanOptionalAction, default=True,
                       help="Use GPU for solver (--no-gpu for CPU)")
    p_w2v.add_argument("--render-gpu", action=argparse.BooleanOptionalAction, default=True,
                       help="Use CuPy GPU for rendering (--no-render-gpu for CPU, default: CPU)")
    p_w2v.add_argument("--fp16", action="store_true",
                       help="Use float16 for correlation computation (faster on GPU, slight precision loss)")
    p_w2v.add_argument("--matmul-corr", action=argparse.BooleanOptionalAction, default=True,
                       help="Use matmul for correlation (default: on). "
                            "--no-matmul-corr uses gather path (less memory, better on CPU)")
    # live clustering
    p_w2v.add_argument("--cluster-m", type=int, default=0,
                       help="Number of clusters (0=disabled)")
    p_w2v.add_argument("--cluster-k2", type=int, default=16,
                       help="Cluster-level KNN size (default: 16)")
    p_w2v.add_argument("--cluster-lr", type=float, default=1.0,
                       help="Centroid nudge learning rate (default: 1.0)")
    p_w2v.add_argument("--cluster-report-every", type=int, default=1000,
                       help="Save cluster visualization every N ticks (default: 1000)")
    p_w2v.add_argument("--cluster-split-every", type=int, default=10,
                       help="Attempt dead cluster recovery every N ticks (default: 10)")
    p_w2v.add_argument("--cluster-hysteresis", type=float, default=0.0,
                       help="Reassignment resistance: neuron must be (1-h)*dist closer to jump (default: 0.0)")
    p_w2v.add_argument("--cluster-knn2-mode", type=str, default='incremental',
                       choices=['incremental', 'knn'],
                       help="knn2 update strategy: 'incremental' (from pairs, no --knn-track needed) "
                            "or 'knn' (from neuron-level KNN lists, requires --knn-track)")
    p_w2v.add_argument("--cluster-centroid-mode", type=str, default='nudge',
                       choices=['exact', 'nudge'],
                       help="Centroid update: 'nudge' (lr-based drift toward member mean, default) "
                            "or 'exact' (incremental arithmetic, immediate — causes churn)")
    p_w2v.add_argument("--cluster-max-k", type=int, default=1,
                       help="Ring buffer depth for multi-cluster membership (default: 1)")
    p_w2v.add_argument("--cluster-track-history", action="store_true",
                       help="Save per-neuron cluster ID at each report interval")
    # wandb logging
    p_w2v.add_argument("--wandb", action="store_true",
                       help="Log metrics to Weights & Biases")
    p_w2v.add_argument("--wandb-project", type=str, default="thalamus-sorter",
                       help="W&B project name (default: thalamus-sorter)")
    p_w2v.add_argument("--wandb-name", type=str, default=None,
                       help="W&B run name")
    p_w2v.add_argument("--wandb-group", type=str, default=None,
                       help="W&B run group")
    p_w2v.add_argument("--wandb-tags", type=str, default=None,
                       help="Comma-separated W&B tags")
    p_w2v.add_argument("--wandb-entity", type=str, default=None,
                       help="W&B entity (team/user)")
    p_w2v.set_defaults(func=run_word2vec)

    # --- temporal ---
    p_temp = sub.add_parser("temporal",
                            help="Temporal correlation sorting (no precomputed neighbors)")
    p_temp.add_argument("--width", "-W", type=int, default=40,
                        help="Grid width (default: 40)")
    p_temp.add_argument("--height", "-H", type=int, default=40,
                        help="Grid height (default: 40)")
    p_temp.add_argument("--buf-source", choices=["synthetic", "gaussian", "embeddings"],
                        default="synthetic",
                        help="Buffer source: gaussian fields or converged embeddings (default: embeddings)")
    p_temp.add_argument("--T", type=int, default=200,
                        help="Buffer length for gaussian source (default: 200)")
    p_temp.add_argument("--P", type=int, default=1,
                        help="Random peers per neuron per tick (default: 1)")
    p_temp.add_argument("--sigma", type=float, default=5.0,
                        help="Spatial smoothing sigma / RBF kernel width (default: 5.0)")
    p_temp.add_argument("--threshold", type=float, default=0.0,
                        help="Similarity threshold: sim > threshold pulls, < repels (default: 0.0)")
    p_temp.add_argument("--emb-k", type=int, default=25,
                        help="K neighbors for embedding generation (default: 25)")
    p_temp.add_argument("--emb-dims", type=int, default=16,
                        help="Dimensionality of source embeddings (default: 16)")
    p_temp.add_argument("--emb-ticks", type=int, default=100000,
                        help="Convergence ticks for embedding generation (default: 100000)")
    p_temp.add_argument("--lr", type=float, default=0.05,
                        help="Learning rate (default: 0.05)")
    p_temp.add_argument("--dims", type=int, default=2,
                        help="Position vector dimensionality (default: 2)")
    p_temp.add_argument("--image", "-i", type=str, default=None,
                        help="Input image to scramble and reconstruct")
    p_temp.add_argument("--frames", "-f", type=int, default=0,
                        help="Number of frames to run (0 = unlimited)")
    p_temp.add_argument("--output-dir", "-o", type=str, default=None,
                        help="Directory to save output frames as PNGs")
    p_temp.add_argument("--save-every", type=int, default=1000,
                        help="Save every Nth frame (default: 1)")
    p_temp.add_argument("--gpu", action="store_true",
                        help="Use GPU acceleration via CuPy")
    p_temp.set_defaults(func=run_temporal)

    # --- mst ---
    p_mst = sub.add_parser("mst", help="MST-based one-shot sorting")
    add_common(p_mst)
    p_mst.set_defaults(func=run_mst)

    # --- sa ---
    p_sa = sub.add_parser("sa", help="Simulated annealing sorting")
    add_common(p_sa)
    p_sa.add_argument("--temp", type=float, default=100.0,
                      help="Initial temperature (default: 100)")
    p_sa.add_argument("--cooling", type=float, default=0.99,
                      help="Cooling rate (default: 0.99)")
    p_sa.add_argument("--sa-iterations", type=int, default=100,
                      help="SA iterations per tick (default: 100)")
    p_sa.set_defaults(func=run_sa)

    # --- camera-sa ---
    p_csa = sub.add_parser("camera-sa",
                           help="Camera: learn weights online, sort with MST")
    p_csa.add_argument("--width", "-W", type=int, default=32)
    p_csa.add_argument("--height", "-H", type=int, default=24)
    p_csa.add_argument("--steps", type=int, default=10,
                       help="Temporal window size (default: 10)")
    p_csa.set_defaults(func=run_camera_sa)

    # --- camera-spatial ---
    p_csp = sub.add_parser("camera-spatial",
                           help="Camera: spatial coherence via TV loss")
    p_csp.add_argument("--width", "-W", type=int, default=16)
    p_csp.add_argument("--height", "-H", type=int, default=12)
    p_csp.add_argument("--lr", type=float, default=0.001,
                       help="Learning rate (default: 0.001)")
    p_csp.add_argument("--epochs", type=int, default=1000,
                       help="Optimization epochs per frame (default: 1000)")
    p_csp.set_defaults(func=run_camera_spatial)

    # Two-pass parsing: first extract --preset, then apply defaults from
    # preset file, then re-parse so CLI args override preset values.
    args, remaining = parser.parse_known_args()

    if hasattr(args, 'preset') and args.preset:
        import json
        preset_path = args.preset
        if not os.path.isabs(preset_path) and not os.path.exists(preset_path):
            # Look in presets/ subdirectory
            preset_path = os.path.join(os.path.dirname(__file__), 'presets', preset_path)
            if not preset_path.endswith('.json'):
                preset_path += '.json'
        with open(preset_path) as f:
            preset = json.load(f)
        print(f"Preset: {args.preset} -> {preset}")
        # Apply preset as defaults, CLI args will override
        # Find the subparser that matches the command
        subparser = sub.choices.get(args.command)
        if subparser:
            subparser.set_defaults(**preset)
        # Re-parse with preset defaults applied
        args = parser.parse_args()

    args.func(args)


if __name__ == "__main__":
    main()

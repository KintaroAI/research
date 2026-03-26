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
                            total_pairs=None, wlog=None, n_sensory=None):
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
            if n_sensory is not None:
                emb = emb[:n_sensory]
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
                 max_k=1, track_history=False, render_mode='color',
                 column_outputs=0, column_max_inputs=20, column_window=4,
                 column_lr=0.05, column_temperature=0.5,
                 column_match_threshold=0.1, column_streaming_decay=0.5,
                 column_lateral=False, lateral_k=6,
                 eligibility=False, trace_decay=0.95,
                 n_sensory=None, embed_render=False, embed_method='pca',
                 column_n_outputs=0, renderer=None):
        import torch
        from cluster_experiments import (
            kmeans_cluster_gpu, _assign_clusters_gpu,
            streaming_update_v3_gpu, split_largest_cluster_gpu,
            visualize_clusters, visualize_clusters_signal, eval_clusters,
        )
        self.n, self.m, self.w, self.h = n, m, w, h
        self.n_sensory = n_sensory if n_sensory is not None else n
        self.embed_render = embed_render
        self.embed_method = embed_method
        self.column_n_outputs = column_n_outputs
        self._dsolver = None  # set externally for embed rendering
        self._pixel_values = None
        self._renderer = renderer
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
        self._visualize_signal = visualize_clusters_signal
        self._eval = eval_clusters
        self.render_mode = render_mode  # 'color', 'signal', 'both'
        self._signals = None
        self._signal_T = 0
        self._sig_channels = 1
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
        self._jump_counts = None  # per-neuron new-cluster jump counter
        # Column wiring
        self.column_mgr = None
        if column_outputs > 0:
            from column_manager import ColumnManager
            self.column_mgr = ColumnManager(
                m, n_outputs=column_outputs, max_inputs=column_max_inputs,
                window=column_window, temperature=column_temperature,
                lr=column_lr, match_threshold=column_match_threshold,
                streaming_decay=column_streaming_decay,
                lateral=column_lateral, lateral_k=lateral_k,
                eligibility=eligibility, trace_decay=trace_decay)

    def set_signals(self, signals_t, sig_channels, T):
        """Store signal tensor reference for signal-based rendering."""
        self._signals = signals_t
        self._sig_channels = sig_channels
        self._signal_T = T

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

        if self.track_history:
            self._jump_counts = np.zeros(self.n, dtype=np.int64)
        self.initialized = True
        # Wire all neurons to their initial cluster columns
        if self.column_mgr:
            for neuron in range(self.n):
                for s in range(self.max_k):
                    c = self.cluster_ids[neuron, s]
                    if c >= 0:
                        self.column_mgr.wire(c, neuron)
            n_wired = (self.column_mgr.slot_map >= 0).sum()
            print(f"  Columns: {n_wired} initial wirings across {self.m} columns")
            # Sync lateral connections with knn2
            if self.column_mgr.lateral and self.knn2_mode != 'knn':
                knn2_np = self.knn2_t.cpu().numpy()
                self.column_mgr.sync_lateral_knn2(knn2_np)
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
            n_reassigned, affected, _, n_blocked, n_switches, wiring_events = self._stream_update(
                embeddings_t, self.centroids_t, self.cluster_ids, self.knn2,
                anchors_np, lr=effective_lr, sizes=self.sizes, min_size=0,
                rng=self.rng, hysteresis=self.hysteresis,
                knn2_is_neurons=True, centroid_mode=self.centroid_mode,
                pointers=self.pointers, last_used=self.last_used,
                tick=global_tick, jump_counts=self._jump_counts)
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
            n_reassigned, affected, _, n_blocked, n_switches, wiring_events = self._stream_update(
                embeddings_t, self.centroids_t, self.cluster_ids, knn2_np,
                anchors_np, lr=effective_lr, sizes=self.sizes, min_size=0,
                rng=self.rng, hysteresis=self.hysteresis,
                centroid_mode=self.centroid_mode,
                pointers=self.pointers, last_used=self.last_used,
                tick=global_tick, jump_counts=self._jump_counts)
            self.total_reassigned += n_reassigned
            self.total_switches += n_switches
            if affected:
                most_recent = self.cluster_ids[np.arange(self.n), self.pointers]
                self.cluster_ids_t = torch.from_numpy(
                    most_recent.astype(np.int64)).to(self.device)
                self._recompute_knn2_dists(list(affected))
            if pairs is not None:
                self._update_knn2_gpu(pairs)

        # Column wiring: process stream update events before splits
        if self.column_mgr and wiring_events:
            for neuron, old_c, new_c in wiring_events:
                if old_c >= 0:
                    self.column_mgr.unwire(old_c, neuron)
                self.column_mgr.wire(new_c, neuron)

        # Periodic split recovery
        if (self.split_every > 0 and global_tick > 0 and
                global_tick % self.split_every == 0):
            n_empty = (self.sizes == 0).sum()
            if n_empty > 0:
                n_to_split = min(n_empty, max(1, n_empty // 5))
                n_splits, split_events = self._split(
                    embeddings_t, self.centroids_t, self.cluster_ids,
                    self.sizes, self.m, n_splits=n_to_split,
                    seed=self.rng.randint(1000000),
                    pointers=self.pointers, last_used=self.last_used,
                    tick=global_tick)
                self.total_splits += n_splits
                # Process split wiring events
                if self.column_mgr and split_events:
                    for neuron, old_c, new_c in split_events:
                        self.column_mgr.unwire(old_c, neuron)
                        self.column_mgr.wire(new_c, neuron)
                if n_splits > 0:
                    most_recent = self.cluster_ids[np.arange(self.n), self.pointers]
                    if self.knn2_mode == 'knn' and self.knn_lists is not None:
                        self.knn2[:], _ = self._freq_knn(
                            self.knn_lists, most_recent, self.m, self.k2)
                    else:
                        self.cluster_ids_t = torch.from_numpy(
                            most_recent.astype(np.int64)).to(self.device)
                        self._recompute_knn2_dists()
                    # Note: lateral knn2 sync happens at report intervals,
                    # not every split (too disruptive to learned weights)

        # Column tick (after all wiring is settled)
        if self.column_mgr:
            if self._signals is not None and self._signal_T > 0:
                # Sliding window from circular buffer: col 0 = most recent
                w = self.column_mgr.window
                indices = [(global_tick - i) % self._signal_T for i in range(w)]
                signal_window = self._signals[:, indices].cpu().numpy()  # (n, w)
                knn2_np = None
                if self.column_mgr.lateral and self.knn2_mode != 'knn':
                    knn2_np = self.knn2_t.cpu().numpy()
                self.column_mgr.tick(signal_window, knn2=knn2_np)
                # Write column outputs to feedback neuron rows for next tick
                if self.n_sensory < self.n:
                    outputs = self.column_mgr.get_outputs()  # (m, n_outputs)
                    fb = torch.from_numpy(
                        outputs.ravel().astype(np.float32)).to(self._signals.device)
                    next_col = (global_tick + 1) % self._signal_T
                    self._signals[self.n_sensory:, next_col] = fb

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
        sensory_ids = most_recent[:self.n_sensory]
        if self.knn2_mode == 'knn':
            # Convert neuron-index knn2 to cluster-index for eval
            knn2_np = self.knn2.copy()
            valid = knn2_np >= 0
            knn2_np[valid] = most_recent[knn2_np[valid]]
            metrics = self._eval(sensory_ids, centroids_np, knn2_np,
                                 self.w, self.h, self.knn_lists)
        else:
            knn2_np = self.knn2_t.cpu().numpy()
            metrics = self._eval(sensory_ids, centroids_np, knn2_np,
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
            self._history.append((tick, most_recent.copy(), self._jump_counts.copy()))
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
        if self.column_mgr:
            # Consistency check: every slot_map wiring must have a matching ring entry
            # (reverse is not required — columns can be full, not all neurons wired)
            slot_map = self.column_mgr.slot_map
            n_stale = 0
            for c in range(self.m):
                for s in range(self.column_mgr.max_inputs):
                    neuron = int(slot_map[c, s])
                    if neuron >= 0:
                        if c not in self.cluster_ids[neuron]:
                            n_stale += 1
            assert n_stale == 0, (
                f"Column wiring inconsistency: {n_stale} stale slot_map entries "
                f"(neuron wired to column but not in cluster ring) at tick {tick}")
            outputs = self.column_mgr.get_outputs()
            # Count wired neurons per column
            wired = (slot_map >= 0).sum(axis=1)
            n_wired_cols = (wired > 0).sum()
            # Winner distribution across all active columns
            if n_wired_cols > 0:
                active_out = outputs[wired > 0]
                winners = active_out.argmax(axis=1)
                winner_dist = np.bincount(winners, minlength=self.column_mgr.n_outputs)
                dist_str = '/'.join(str(int(x)) for x in winner_dist)
                print(f"  Columns @ tick {tick}: {n_wired_cols}/{self.m} wired, "
                      f"winner_dist=[{dist_str}]")
        if self.wlog:
            self.wlog.log_clusters(
                tick, n_alive, self.m, metrics['contiguity_mean'],
                metrics['diameter_mean'], jumps_per_tick,
                self.total_reassigned, self.total_splits,
                stability=stability,
                switches_per_tick=switches_per_tick,
                total_switches=self.total_switches)
        if self._renderer is not None:
            ns = self.n_sensory
            sensory_ids = most_recent[:ns]
            if self.render_mode in ('color', 'both'):
                self._renderer.cluster(tick, sensory_ids)
            if self.render_mode in ('signal', 'both') and self._signals is not None:
                t = tick % self._signal_T
                signal = self._signals[:ns, t].cpu().numpy()
                self._renderer.cluster_signal(tick, sensory_ids, signal)
                self._renderer.signal(tick, signal)
            if self.embed_render and self._dsolver is not None:
                emb_all = self._dsolver.get_positions()
                self._renderer.embed(tick, emb_all, ns,
                                     pixel_values=self._pixel_values,
                                     cluster_ids=most_recent,
                                     n_outputs=self.column_n_outputs,
                                     method=self.embed_method)

    def load_state(self, state_dir, embeddings_t):
        """Load saved cluster + column state for warm restart."""
        import torch
        self.device = embeddings_t.device

        self.cluster_ids = np.load(os.path.join(state_dir, "cluster_ids.npy"))
        self.pointers = np.load(os.path.join(state_dir, "pointers.npy"))
        self.last_used = np.load(os.path.join(state_dir, "last_used.npy"))
        centroids_np = np.load(os.path.join(state_dir, "centroids.npy"))
        self.centroids_t = torch.from_numpy(centroids_np).to(self.device)
        most_recent = self.cluster_ids[np.arange(self.n), self.pointers]
        self.cluster_ids_t = torch.from_numpy(
            most_recent.astype(np.int64)).to(self.device)
        self.sizes = np.bincount(most_recent[most_recent >= 0],
                                 minlength=self.m).astype(np.int64)

        knn2_np = np.load(os.path.join(state_dir, "knn2.npy"))
        if self.knn2_mode == 'knn':
            self.knn2 = knn2_np
        else:
            self.knn2_t = torch.from_numpy(knn2_np).to(self.device)
            knn2_dists_np = np.full_like(knn2_np, np.inf, dtype=np.float32)
            self.knn2_dists_t = torch.from_numpy(knn2_dists_np).to(self.device)
            self._recompute_knn2_dists()

        if self.track_history:
            self._jump_counts = np.zeros(self.n, dtype=np.int64)

        # Load column state
        if self.column_mgr:
            col_state_path = os.path.join(state_dir, "column_states.pt")
            slot_map_path = os.path.join(state_dir, "column_slot_map.npy")
            if os.path.exists(col_state_path) and os.path.exists(slot_map_path):
                state = torch.load(col_state_path, weights_only=True)
                def _to_np(t):
                    return t.numpy() if hasattr(t, 'numpy') else np.array(t)
                self.column_mgr.prototypes = _to_np(state['prototypes'])
                self.column_mgr.usage = _to_np(state['usage'])
                self.column_mgr.proj_mean = _to_np(state.get('proj_mean',
                    torch.zeros(self.m, self.column_mgr.n_outputs)))
                self.column_mgr.proj_var = _to_np(state.get('proj_var',
                    torch.zeros(self.m, self.column_mgr.n_outputs)))
                if self.column_mgr.lateral and state.get('lateral_protos') is not None:
                    self.column_mgr.lateral_protos = _to_np(state['lateral_protos'])
                if self.column_mgr.traces is not None and state.get('traces') is not None:
                    self.column_mgr.traces = _to_np(state['traces'])
                if state.get('output_tiredness') is not None:
                    self.column_mgr.output_tiredness = _to_np(state['output_tiredness'])
                self.column_mgr.slot_map = np.load(slot_map_path)
                n_wired = (self.column_mgr.slot_map >= 0).sum()
                print(f"  Columns restored: {n_wired} wirings")

        self.initialized = True
        n_empty = (self.sizes == 0).sum()
        print(f"  Clusters restored from {state_dir}: "
              f"{self.m - n_empty}/{self.m} alive")

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
            ticks = np.array([t for t, _, _ in self._history], dtype=np.int64)
            ids = np.stack([h for _, h, _ in self._history])
            jumps = np.stack([j for _, _, j in self._history])
            np.save(os.path.join(output_dir, "history_ticks.npy"), ticks)
            np.save(os.path.join(output_dir, "history_ids.npy"), ids)
            np.save(os.path.join(output_dir, "history_jumps.npy"), jumps)
            print(f"  cluster history saved: {len(self._history)} snapshots")
        if self.column_mgr:
            self.column_mgr.save(output_dir)
        print(f"  cluster state saved to {output_dir}")


def _run_training_loop(do_tick, dsolver, args, n, w, sig_channels, wlog,
                       on_save=None, on_display=None, can_break=None,
                       cluster_mgr=None, n_sensory=None,
                       bench_metadata=None):
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
                # Send graph to viz app (independent of report/save)
                viz_every = getattr(args, 'viz_every', 0)
                if (viz_every > 0 and tick % viz_every == 0
                        and cluster_mgr._renderer is not None
                        and cluster_mgr._renderer.viz_address
                        and cluster_mgr.column_mgr):
                    most_recent = cluster_mgr.cluster_ids[
                        np.arange(cluster_mgr.n), cluster_mgr.pointers]
                    ns = cluster_mgr.n_sensory
                    lateral_adj = None
                    if cluster_mgr.column_mgr.lateral:
                        lateral_adj = cluster_mgr.column_mgr.lateral_adj
                    knn2_viz = (cluster_mgr.knn2 if cluster_mgr.knn2_mode == 'knn'
                                else cluster_mgr.knn2_t.cpu().numpy())
                    cluster_mgr._renderer.graph(
                        tick, most_recent, ns, cluster_mgr.column_n_outputs,
                        lateral_adj=lateral_adj,
                        column_outputs=cluster_mgr.column_mgr.get_outputs(),
                        knn2=knn2_viz,
                        centroids=cluster_mgr.centroids_t.cpu().numpy())

                # Send field data to field viz
                if (viz_every > 0 and tick % viz_every == 0
                        and cluster_mgr._renderer is not None
                        and cluster_mgr._renderer.field_address
                        and bench_metadata is not None
                        and 'pos' in bench_metadata):
                    bm = bench_metadata
                    bm_state = bm.get('state', {})
                    hunger_val = bm_state.get('hunger', [0])[0]
                    pois_arr = bm_state.get('pois', np.empty((0, 2)))
                    cluster_mgr._renderer.field_live(
                        tick, bm['pos'].copy(), pois_arr.copy(),
                        bm['field_size'],
                        hunger=float(hunger_val),
                        collect_radius=bm.get('collect_radius', 5.0),
                        score=int(bm.get('score', [0])[0]))

                if tick % cluster_report_every == 0:
                    # Refresh knn_lists for knn mode
                    if cluster_mgr.knn2_mode == 'knn' and dsolver.knn_k > 0:
                        cluster_mgr.knn_lists = dsolver.get_knn_lists()
                    cluster_mgr.report(tick)
                    # Periodic lateral knn2 sync (not every split — too disruptive)
                    if (cluster_mgr.column_mgr and cluster_mgr.column_mgr.lateral
                            and cluster_mgr.knn2_mode != 'knn'):
                        knn2_np = cluster_mgr.knn2_t.cpu().numpy()
                        cluster_mgr.column_mgr.sync_lateral_knn2(knn2_np)

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
            spatial_acc = dsolver.knn_spatial_accuracy(w, radius=3, channels=sig_channels,
                                                       n_eval=n_sensory)
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
        import time
        import torch

        sig_channels = getattr(args, 'signal_channels', 1)
        n = w * h * sig_channels
        n_sensory = n

        # --- Feedback loop: column outputs → feedback neurons ---
        column_outputs = getattr(args, 'column_outputs', 0)
        column_feedback = getattr(args, 'column_feedback', False)
        cluster_m = getattr(args, 'cluster_m', 0)
        neurons_per = getattr(args, 'cluster_neurons_per', 0)

        # Auto-compute M from neurons-per-cluster target
        if cluster_m == 0 and neurons_per > 0 and column_outputs > 0:
            cluster_m = n_sensory // (neurons_per - column_outputs)
            args.cluster_m = cluster_m

        K = 0
        if column_feedback and cluster_m > 0 and column_outputs > 0:
            K = cluster_m * column_outputs
            n = n_sensory + K
            print(f"Column feedback: K={K} feedback neurons, "
                  f"n_sensory={n_sensory}, n_total={n}")

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
                pixel_values = np.zeros((n_sensory, 3), dtype=np.uint8)
                for i in range(n_sensory):
                    px = i // sig_channels
                    ch = i % sig_channels
                    gray = float(image[px // w, px % w])
                    tint = channel_tints.get(ch, np.array([1.0, 1.0, 1.0]))
                    pixel_values[i] = np.clip(gray * tint, 0, 255).astype(np.uint8)
            else:
                pixel_values = np.zeros(n_sensory, dtype=np.uint8)
                for i in range(n_sensory):
                    pixel_values[i] = image[i // w, i % w]
        elif args.signal_source and (
                args.signal_source.endswith('.png') or
                args.signal_source.endswith('.jpg') or
                args.signal_source.endswith('.npy')):
            # Center crop from signal source as static pixel values
            if args.signal_source.endswith('.npy'):
                src = np.load(args.signal_source)
            else:
                img_bgr = cv2.imread(args.signal_source)
                src = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
            if src.ndim == 3:
                src_h, src_w = src.shape[:2]
            else:
                src_h, src_w = src.shape
            cy, cx = (src_h - h) // 2, (src_w - w) // 2
            crop = src[cy:cy+h, cx:cx+w]
            raw = crop.ravel()[:n_sensory]
            vmin, vmax = raw.min(), raw.max()
            if vmax > vmin:
                pixel_values = ((raw - vmin) / (vmax - vmin) * 255).astype(np.uint8)
            else:
                pixel_values = np.full(n_sensory, 128, dtype=np.uint8)

        render_w = w * sig_channels
        render_h = h

        output_dir = args.output_dir
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            _save_run_info(output_dir, args)

        render_method = args.render
        if render_method == 'euclidean':
            render_method = 'pca'

        # --- Create renderer ---
        from render_server import Renderer
        viz_address = getattr(args, 'viz_address', None)
        field_address = getattr(args, 'field_address', None)
        renderer = Renderer(output_dir, render_w, render_h,
                            sig_channels=sig_channels,
                            viz_address=viz_address,
                            field_address=field_address) if output_dir else None

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
            if K > 0 and warm.shape[0] == n_sensory:
                fb_init = np.random.randn(K, args.dims).astype(np.float32) * 0.01
                warm = np.concatenate([warm, fb_init], axis=0)
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

            signals_np = np.random.rand(n, T).astype(np.float32) if K > 0 \
                else np.zeros((n, T), dtype=np.float32)

            # Synthetic benchmark signals
            from benchmarks import get_benchmark
            bench = get_benchmark(args.signal_source) if args.signal_source else None
            bench_signal = None
            bench_metadata = None
            if bench is not None:
                bench_signal, bench_metadata = bench.make_signal(w, h, args)
                if bench_signal is not None:
                    if '_refs' in bench_metadata and renderer is not None:
                        bench_metadata['_refs']['renderer'] = renderer
                    for t in range(T):
                        signals_np[:n_sensory, t] = bench_signal(t)

            if (bench is None or bench_signal is None) and args.signal_source:
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
                        signals_np[:n_sensory, t] = crop
                    else:
                        signals_np[:n_sensory, t] = crop - crop.mean()
                src_desc = f"{src_w}x{src_h}" + (f"x{sig_channels}" if source.ndim == 3 else "")
                crop_desc = f"{crop_w}x{crop_h}" + (f"x{sig_channels}" if source.ndim == 3 else "")
                mean_sub = "raw" if use_raw else "mean-subtracted"
                print(f"  signal buffer: ({n}, {T}), rolling saccades from "
                      f"{args.signal_source} ({src_desc}), "
                      f"crop={crop_desc}, step={saccade_step}, {mean_sub}")
                saccade_source = torch.from_numpy(source).to(dsolver.device)
            elif bench_signal is None:
                from scipy.ndimage import gaussian_filter
                for t in range(T):
                    noise = np.random.randn(h, w).astype(np.float32)
                    smoothed = gaussian_filter(noise, sigma=sigma)
                    signals_np[:n_sensory, t] = smoothed.ravel()
                print(f"  signal buffer: ({n}, {T}), spatial sigma={sigma}")

            signals = torch.from_numpy(signals_np).to(dsolver.device)
            tick_counter = [0]
        else:
            print(f"Word2vec drift (sentence): {w}x{h} grid, k={args.k}, "
                  f"k_neg={args.k_neg}, lr={args.lr}, dims={args.dims}, "
                  f"window={args.window}, normalize_every={norm_every}, "
                  f"align={args.align}, async={args.async_render}")

        # --- Motor control setup ---
        motor_col_id = getattr(args, 'motor_column', -1)
        motor_scale = getattr(args, 'motor_scale', 5.0)
        motor_log = [] if motor_col_id >= 0 else None
        # Proprioception: 6 override neurons (last 6 sensory neurons)
        # [0,1] = position_x, position_y normalized
        # [2,3,4,5] = urgency for dx+, dx-, dy+, dy-
        motor_proprio = motor_col_id >= 0
        if motor_proprio:
            proprio_idx = list(range(n_sensory - 6, n_sensory))
            urgency = np.zeros(4, dtype=np.float32)  # dx+, dx-, dy+, dy-
            urgency_rate = 0.005  # ramp per tick when not moving (~200 ticks to 1.0)
            prev_walk = [walk_dx if saccade_source is not None else 0,
                         walk_dy if saccade_source is not None else 0]
            print(f"Motor proprioception: neurons {proprio_idx}, "
                  f"urgency_rate={urgency_rate}")

        # --- Predictive shift mixing ---
        _pred_shift_base = getattr(args, 'predictive_shift', 0)
        _pred_mix = getattr(args, 'predictive_mix', 0.0)
        def _tick_predictive_shift():
            if _pred_mix > 0 and _pred_shift_base > 0:
                return _pred_shift_base if np.random.random() < _pred_mix else 0
            return _pred_shift_base

        # --- Tick function ---
        if mode == "correlation":
            def do_tick():
                if bench_signal is not None:
                    col = tick_counter[0] % T
                    signals[:n_sensory, col] = torch.from_numpy(
                        bench_signal(tick_counter[0] + T)).to(signals.device)
                    tick_counter[0] += 1
                elif saccade_source is not None:
                    nonlocal walk_dy, walk_dx
                    # Motor bias from designated column
                    motor_dx, motor_dy = 0.0, 0.0
                    if (motor_col_id >= 0 and cluster_mgr is not None
                            and cluster_mgr.column_mgr is not None):
                        out = cluster_mgr.column_mgr.get_outputs()[motor_col_id]
                        motor_dx = (out[0] - out[1]) * motor_scale
                        motor_dy = (out[2] - out[3]) * motor_scale
                        if motor_log is not None:
                            motor_log.append((tick_counter[0], walk_dx, walk_dy,
                                              float(motor_dx), float(motor_dy),
                                              out.tolist()))
                    old_dx, old_dy = walk_dx, walk_dy
                    rand_dy = np.random.randint(-saccade_step, saccade_step + 1)
                    rand_dx = np.random.randint(-saccade_step, saccade_step + 1)
                    # Motor confidence suppresses random walk
                    if motor_proprio:
                        motor_mag = np.sqrt(motor_dx**2 + motor_dy**2)
                        confidence = min(1.0, motor_mag / motor_scale)
                        rand_scale = 1.0 - confidence
                        rand_dy = int(round(rand_dy * rand_scale))
                        rand_dx = int(round(rand_dx * rand_scale))
                    walk_dy = np.clip(walk_dy + rand_dy + int(round(motor_dy)),
                                      0, max_dy)
                    walk_dx = np.clip(walk_dx + rand_dx + int(round(motor_dx)),
                                      0, max_dx)
                    crop = saccade_source[walk_dy:walk_dy+crop_h, walk_dx:walk_dx+crop_w].reshape(-1)
                    col = tick_counter[0] % T
                    if use_raw:
                        signals[:n_sensory, col] = crop
                    else:
                        signals[:n_sensory, col] = crop - crop.mean()
                    # Override proprioception neurons
                    if motor_proprio:
                        dx_moved = walk_dx - old_dx
                        dy_moved = walk_dy - old_dy
                        # Update urgency: ramp up when not moving, reset on move
                        # [0]=dx+, [1]=dx-, [2]=dy+, [3]=dy-
                        moves = [dx_moved > 0, dx_moved < 0,
                                 dy_moved > 0, dy_moved < 0]
                        for i in range(4):
                            if moves[i]:
                                urgency[i] = 0.0
                            else:
                                urgency[i] = min(1.0, urgency[i] + urgency_rate)
                        # Write to signal buffer
                        pos_x = walk_dx / max(max_dx, 1)
                        pos_y = walk_dy / max(max_dy, 1)
                        proprio_vals = torch.tensor(
                            [pos_x, pos_y] + urgency.tolist(),
                            dtype=torch.float32, device=signals.device)
                        signals[proprio_idx, col] = proprio_vals
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
                    matmul_corr=getattr(args, 'matmul_corr', True),
                    predictive_shift=_tick_predictive_shift())
        else:
            def do_tick():
                dsolver.tick_sentence(window=args.window)
                return 0

        # --- Live clustering ---
        embed_render_mode = getattr(args, 'render_mode', 'grid') == 'embed'
        cluster_mgr = None
        cluster_m = getattr(args, 'cluster_m', 0)
        if cluster_m > 0:
            cluster_k2 = getattr(args, 'cluster_k2', 16)
            cluster_hyst = getattr(args, 'cluster_hysteresis', 0.0)
            knn2_mode = getattr(args, 'cluster_knn2_mode', 'incremental')
            centroid_mode = getattr(args, 'cluster_centroid_mode', 'nudge')
            cluster_max_k = getattr(args, 'cluster_max_k', 1)
            column_outputs = getattr(args, 'column_outputs', 0)
            cluster_mgr = _ClusterManager(
                n, cluster_m, w, h, k2=cluster_k2,
                lr=getattr(args, 'cluster_lr', 1.0),
                split_every=getattr(args, 'cluster_split_every', 10),
                output_dir=output_dir, wlog=wlog,
                hysteresis=cluster_hyst, knn2_mode=knn2_mode,
                centroid_mode=centroid_mode,
                max_k=cluster_max_k,
                track_history=getattr(args, 'cluster_track_history', False),
                render_mode=getattr(args, 'cluster_render_mode', 'color'),
                column_outputs=column_outputs,
                column_max_inputs=getattr(args, 'column_max_inputs', 20),
                column_window=getattr(args, 'column_window', 4),
                column_lr=getattr(args, 'column_lr', 0.05),
                column_temperature=getattr(args, 'column_temperature', 0.5),
                column_match_threshold=getattr(args, 'column_match_threshold', 0.1),
                column_streaming_decay=getattr(args, 'column_streaming_decay', 0.5),
                column_lateral=getattr(args, 'column_lateral', False),
                lateral_k=getattr(args, 'lateral_k', 6),
                eligibility=getattr(args, 'eligibility', False),
                trace_decay=getattr(args, 'trace_decay', 0.95),
                n_sensory=n_sensory,
                embed_render=embed_render_mode,
                embed_method=render_method,
                column_n_outputs=column_outputs,
                renderer=renderer)
            render_mode = getattr(args, 'cluster_render_mode', 'color')
            if render_mode in ('signal', 'both') or column_outputs > 0:
                cluster_mgr.set_signals(signals, sig_channels, T)
            cluster_mgr._pixel_values = pixel_values
            cluster_mgr._dsolver = dsolver
            # Give benchmark access to column manager for motor control
            if bench_metadata is not None and '_refs' in bench_metadata:
                bench_metadata['_refs']['column_mgr'] = cluster_mgr.column_mgr
                bench_metadata['_refs']['renderer'] = renderer
                bench_metadata['_refs']['dsolver'] = dsolver
            col_str = f", columns={column_outputs}out" if column_outputs > 0 else ""
            print(f"Live clustering enabled: m={cluster_m}, k2={cluster_k2}, "
                  f"max_k={cluster_max_k}, "
                  f"hysteresis={cluster_hyst}, knn2={knn2_mode}, "
                  f"centroid={centroid_mode}, render={render_mode}{col_str}, "
                  f"report_every={getattr(args, 'cluster_report_every', 1000)}")
            if motor_col_id >= 0:
                print(f"Motor control: column {motor_col_id}, scale={motor_scale}")
            # Warm-start cluster + column state
            warm_clusters = getattr(args, 'warm_start_clusters', None)
            if warm_clusters and os.path.isdir(warm_clusters):
                cluster_mgr.load_state(warm_clusters, dsolver.positions)

        # --- Training + rendering ---
        def on_save(tick, total_pairs):
            if renderer is not None and pixel_values is not None:
                emb = dsolver.get_positions()[:n_sensory]
                renderer.grid(tick, emb, pixel_values,
                              method=render_method,
                              align=args.align, gpu=args.render_gpu)

        t0, total_pairs = _run_training_loop(
            do_tick, dsolver, args, n, w, sig_channels, wlog,
            on_save=on_save if output_dir else None,
            can_break=poll_quit, cluster_mgr=cluster_mgr,
            n_sensory=n_sensory if K > 0 else None,
            bench_metadata=bench_metadata)

        elapsed = time.time() - t0
        s = dsolver.stats()
        print(f"Done: {args.frames} ticks in {elapsed:.1f}s, "
              f"std={s['std']:.4f}, total_pairs={total_pairs}")
        wlog.log_done(args.frames, elapsed, s['std'], total_pairs)

        # Final frame
        if renderer is not None and pixel_values is not None:
            emb = dsolver.get_positions()[:n_sensory]
            renderer.grid(args.frames, emb, pixel_values,
                          method=render_method,
                          align=args.align, gpu=args.render_gpu)

        if cluster_mgr is not None and cluster_mgr.initialized:
            cluster_mgr.report(args.frames)
            if output_dir:
                cluster_mgr.save(output_dir)

        # Benchmark analysis
        if bench is not None and bench_metadata is not None:
            bench_metadata['_tick_fn'] = bench_signal
            bench_metadata['_total_ticks'] = args.frames
            bench.analyze(bench_metadata, cluster_mgr, signals,
                          tick_counter, T, output_dir)

        # Motor log analysis
        if motor_log and output_dir:
            log_arr = np.array([(t, x, y, mdx, mdy) for t, x, y, mdx, mdy, _ in motor_log],
                               dtype=np.float32)
            np.save(os.path.join(output_dir, "motor_log.npy"), log_arr)
            positions = log_arr[:, 1:3].astype(int)
            if len(positions) > 0:
                motor_mag = np.sqrt(log_arr[:, 3]**2 + log_arr[:, 4]**2)
                hist_flat = np.histogram2d(positions[:, 1], positions[:, 0],
                                           bins=[int(positions[:, 1].max()) + 1,
                                                 int(positions[:, 0].max()) + 1])[0].ravel()
                uniformity = hist_flat.std() / max(hist_flat.mean(), 1e-8)
                print(f"  Motor: {len(motor_log)} ticks logged, "
                      f"mean|motor|={motor_mag.mean():.2f}, "
                      f"position uniformity={uniformity:.3f} "
                      f"(0=uniform, higher=concentrated)")
                if renderer is not None:
                    renderer.heatmap(positions)

        _save_results_and_model(output_dir, args, dsolver, render_w, render_h,
                               t0, args.frames, total_pairs=total_pairs,
                               wlog=wlog, n_sensory=n_sensory if K > 0 else None)
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
    p_w2v.add_argument("--threshold", type=float, default=0.5,
                       help="Similarity/correlation threshold (default: 0.0)")
    # correlation mode args
    p_w2v.add_argument("--k-sample", type=int, default=200,
                       help="Random candidates to check per neuron (correlation mode, default: 50)")
    p_w2v.add_argument("--batch-size", type=int, default=256,
                       help="GPU batch size for anchor processing (default: 256)")
    anchor_group = p_w2v.add_mutually_exclusive_group()
    anchor_group.add_argument("--anchor-sample", type=int, default=None,
                       help="Total unique anchor neurons per tick (split into batch_size chunks)")
    anchor_group.add_argument("--anchor-batches", type=int, default=None,
                       help="Number of batches per tick (anchor_sample = anchor_batches * batch_size)")
    p_w2v.add_argument("--signal-T", type=int, default=1000,
                       help="Temporal signal buffer length (correlation mode, default: 100)")
    p_w2v.add_argument("--signal-sigma", type=float, default=3.0,
                       help="Gaussian smoothing sigma for signal generation (correlation mode, default: 3.0)")
    p_w2v.add_argument("--signal-source", type=str, default=None,
                       help="Path to signal source: .npy/.png/.jpg, or benchmark name (e.g. 'xor')")
    # Register benchmark-specific args
    from benchmarks import list_benchmarks, get_benchmark
    for bname in list_benchmarks():
        bmod = get_benchmark(bname)
        if bmod and hasattr(bmod, 'add_args'):
            bmod.add_args(p_w2v)
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
    p_w2v.add_argument("--use-deriv-corr", action=argparse.BooleanOptionalAction, default=True,
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
    p_w2v.add_argument("--render-mode", choices=["grid", "embed"],
                       default="grid",
                       help="'grid' (default), 'embed' saves additional embed_NNNNNN.png "
                            "scatter plots at cluster_report_every intervals")
    p_w2v.add_argument("--viz-address", type=str, default=None,
                       help="host:port for live graph visualization app (e.g., 192.168.1.5:9100)")
    p_w2v.add_argument("--viz-every", type=int, default=0,
                       help="Send graph to viz app every N ticks (0=disabled, 1=every tick)")
    p_w2v.add_argument("--field-address", type=str, default=None,
                       help="host:port for live field visualization (e.g., 192.168.1.5:9101)")
    p_w2v.add_argument("--align", action="store_true",
                       help="Procrustes-align rendered output to grid (fixes rotation/flip)")
    p_w2v.add_argument("--warm-start", type=str, default=None,
                       help="Load .npy embeddings as initial positions (warm start)")
    p_w2v.add_argument("--warm-start-clusters", type=str, default=None,
                       help="Load cluster+column state from this directory (full resume)")
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
    p_w2v.add_argument("--lr", type=float, default=0.001,
                       help="Learning rate (default: 0.001)")
    p_w2v.add_argument("--dims", type=int, default=8,
                       help="Position vector dimensionality (default: 8)")
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
    p_w2v.add_argument("--predictive-shift", type=int, default=0,
                       help="Predictive correlation: shift anchor signal by N ticks "
                            "(0=co-occurrence, 1=causal prediction)")
    p_w2v.add_argument("--predictive-mix", type=float, default=0.0,
                       help="Probability of using predictive shift per tick "
                            "(0.0=always co-occurrence, 0.1=10%% predictive, 1.0=always predictive)")
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
    p_w2v.add_argument("--cluster-max-k", type=int, default=2,
                       help="Ring buffer depth for multi-cluster membership (default: 2)")
    p_w2v.add_argument("--cluster-track-history", action="store_true",
                       help="Save per-neuron cluster ID at each report interval")
    p_w2v.add_argument("--cluster-render-mode", type=str, default='color',
                       choices=['color', 'signal', 'both'],
                       help="Cluster visualization: 'color' (ID-based), 'signal' (mean neuron signal), 'both'")
    # column wiring (thalamus-to-cortex)
    p_w2v.add_argument("--column-outputs", type=int, default=4,
                       help="Column outputs per cluster (0=disabled, 4=enable with 4 outputs)")
    p_w2v.add_argument("--column-max-inputs", type=int, default=20,
                       help="Pre-allocated input slots per column (default: 20)")
    p_w2v.add_argument("--column-window", type=int, default=10,
                       help="Sliding window size for streaming columns (default: 10)")
    p_w2v.add_argument("--column-lr", type=float, default=0.05,
                       help="Column learning rate (default: 0.05)")
    p_w2v.add_argument("--column-temperature", type=float, default=0.2,
                       help="Column softmax temperature (default: 0.2)")
    p_w2v.add_argument("--column-match-threshold", type=float, default=0.1,
                       help="Column match threshold for dormant reassignment (default: 0.1)")
    p_w2v.add_argument("--column-streaming-decay", type=float, default=0.8,
                       help="Column streaming EMA decay (default: 0.8, rule of thumb: 1-2/window)")
    p_w2v.add_argument("--column-feedback", action="store_true",
                       help="Feed column outputs back as signal for feedback neurons")
    p_w2v.add_argument("--column-lateral", action="store_true",
                       help="Enable lateral connections between columns")
    p_w2v.add_argument("--lateral-k", type=int, default=2,
                       help="Lateral connections per column (default: 2)")
    p_w2v.add_argument("--lateral-sparsity", type=float, default=1.0,
                       help="Fraction of lateral connections to keep (1.0=full, 0.1=10%%)")
    p_w2v.add_argument("--eligibility", action="store_true",
                       help="Enable eligibility traces on columns (reward-gated learning)")
    p_w2v.add_argument("--trace-decay", type=float, default=0.95,
                       help="Eligibility trace decay per tick (default: 0.95, ~20 tick window)")
    p_w2v.add_argument("--cluster-neurons-per", type=int, default=0,
                       help="Target neurons per cluster (auto-computes M from formula)")
    p_w2v.add_argument("--motor-column", type=int, default=-1,
                       help="Cluster whose column outputs steer saccade (-1=disabled, 0=first cluster)")
    p_w2v.add_argument("--motor-scale", type=float, default=5.0,
                       help="Motor output scale in pixels (default: 5.0)")
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

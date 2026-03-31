"""ClusterManager: live streaming cluster maintenance during training.

Extracted from main.py — manages k-means clusters over evolving embeddings,
with optional column wiring (SoftWTACell per cluster) and visualization.
"""

import os
import numpy as np


class ClusterManager:
    """Live streaming cluster maintenance during training."""

    def __init__(self, n, m, w, h, k2=16, lr=1.0, split_every=10,
                 max_k=1, knn2_mode='incremental', centroid_mode='nudge',
                 hysteresis=0.0, track_history=False, render_mode='color',
                 n_sensory=None, embed_render=False, embed_method='pca',
                 column_config=None, renderer=None,
                 output_dir=None, wlog=None,
                 max_cluster_size=0, cluster_swap=True):
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
        self.column_n_outputs = 0
        self._dsolver = None  # set externally for embed rendering
        self._pixel_values = None
        self._renderer = renderer
        self.k2, self.lr, self.split_every = k2, lr, split_every
        self.max_cluster_size = max_cluster_size
        self.cluster_swap = cluster_swap
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
        if column_config is not None and column_config.get('n_outputs', 0) > 0:
            self.column_n_outputs = column_config['n_outputs']
            column_type = column_config.get('type', 'default')
            if column_type == 'conscience':
                from column_manager import ConscienceColumn
                self.column_mgr = ConscienceColumn(
                    m,
                    n_outputs=column_config.get('n_outputs', 4),
                    max_inputs=column_config.get('max_inputs', 20),
                    window=column_config.get('window', 4),
                    lr=column_config.get('lr', 0.05),
                    alpha=column_config.get('alpha', 0.01),
                    temperature=column_config.get('temperature', 0.5),
                    reseed_after=column_config.get('reseed_after', 1000),
                    wta_mode=column_config.get('wta_mode', 'none'),
                    lateral_inputs=column_config.get('lateral_inputs', False),
                    lateral_input_k=column_config.get('lateral_input_k', 4))
            elif column_type == 'recon':
                from column_manager import ReconColumn
                self.column_mgr = ReconColumn(
                    m,
                    n_outputs=column_config.get('n_outputs', 4),
                    max_inputs=column_config.get('max_inputs', 20),
                    window=column_config.get('window', 10),
                    lr=column_config.get('lr', 1e-3),
                    temperature=column_config.get('temperature', 0.5),
                    n_heads=column_config.get('n_heads', 2),
                    lambda_sharp=column_config.get('lambda_sharp', 0.01),
                    lambda_balance=column_config.get('lambda_balance', 0.1),
                    alpha=column_config.get('alpha', 0.0),
                    tiredness_rate=column_config.get('tiredness_rate', 0.0),
                    tiredness_recovery=column_config.get('tiredness_recovery', 0.0),
                    wta_mode=column_config.get('wta_mode', 'none'),
                    lateral_inputs=column_config.get('lateral_inputs', False),
                    lateral_input_k=column_config.get('lateral_input_k', 4))
            elif column_type == 'predictive':
                from column_manager import PredictiveColumn
                self.column_mgr = PredictiveColumn(
                    m,
                    n_outputs=column_config.get('n_outputs', 4),
                    max_inputs=column_config.get('max_inputs', 20),
                    window=column_config.get('window', 10),
                    lr=column_config.get('lr', 1e-3),
                    temperature=column_config.get('temperature', 0.5),
                    n_heads=column_config.get('n_heads', 2),
                    lambda_sharp=column_config.get('lambda_sharp', 0.01),
                    lambda_balance=column_config.get('lambda_balance', 0.1),
                    train_every=column_config.get('train_every', 10),
                    alpha=column_config.get('alpha', 0.0),
                    tiredness_rate=column_config.get('tiredness_rate', 0.0),
                    tiredness_recovery=column_config.get('tiredness_recovery', 0.0),
                    wta_mode=column_config.get('wta_mode', 'none'),
                    lateral_inputs=column_config.get('lateral_inputs', False),
                    lateral_input_k=column_config.get('lateral_input_k', 4))
            else:
                from column_manager import ColumnManager
                self.column_mgr = ColumnManager(
                    m,
                    n_outputs=column_config.get('n_outputs', 4),
                    max_inputs=column_config.get('max_inputs', 20),
                    window=column_config.get('window', 4),
                    temperature=column_config.get('temperature', 0.5),
                    lr=column_config.get('lr', 0.05),
                    match_threshold=column_config.get('match_threshold', 0.1),
                    streaming_decay=column_config.get('streaming_decay', 0.5),
                    lateral=column_config.get('lateral', False),
                    lateral_k=column_config.get('lateral_k', 6),
                    eligibility=column_config.get('eligibility', False),
                    trace_decay=column_config.get('trace_decay', 0.95),
                    mode=column_config.get('mode', 'kmeans'),
                    confidence_gating=column_config.get('confidence_gating', False),
                    confidence_floor=column_config.get('confidence_floor', 0.3),
                    tiredness_rate=column_config.get('tiredness_rate', 0.0),
                    tiredness_recovery=column_config.get('tiredness_recovery', 0.0005),
                    entropy_scaled_lr=column_config.get('entropy_scaled_lr', True),
                    lateral_mode=column_config.get('lateral_mode', 'covariance'),
                    reward_lr=column_config.get('reward_lr', 0.01),
                    lateral_inputs=column_config.get('lateral_inputs', False),
                    lateral_input_k=column_config.get('lateral_input_k', 4))

    def set_signals(self, signals_t, sig_channels, T):
        """Store signal tensor reference for signal-based rendering."""
        self._signals = signals_t
        self._sig_channels = sig_channels
        self._signal_T = T

    def init_clusters(self, embeddings_t, knn_lists_np=None):
        """Initialize clusters — fast random assignment (O(n), no k-means)."""
        import torch
        self.device = embeddings_t.device
        n = embeddings_t.shape[0]

        # Random assignment: neuron i → cluster i % m
        ids_np = (np.arange(n) % self.m).astype(np.int64)
        self.rng.shuffle(ids_np)  # shuffle so spatial neighbors aren't in same cluster
        self.cluster_ids = np.full((self.n, self.max_k), -1, dtype=np.int64)
        self.cluster_ids[:, 0] = ids_np
        self.pointers = np.zeros(self.n, dtype=np.int64)
        self.last_used = np.zeros((self.n, self.max_k), dtype=np.int64)
        self.cluster_ids_t = torch.from_numpy(ids_np.astype(np.int64)).to(self.device)
        self.sizes = np.bincount(ids_np, minlength=self.m)

        # Centroids: pick a random member as initial centroid (O(n), no per-cluster loop)
        emb_np = embeddings_t.cpu().numpy()
        dims = emb_np.shape[1]
        centroids_np = np.zeros((self.m, dims), dtype=np.float32)
        # First member per cluster as centroid (streaming update will refine)
        first_seen = np.full(self.m, -1, dtype=np.int64)
        for i in range(n):
            c = ids_np[i]
            if first_seen[c] < 0:
                first_seen[c] = i
                centroids_np[c] = emb_np[i]
        self.centroids_t = torch.from_numpy(centroids_np).to(self.device)

        if self.knn2_mode == 'knn':
            self.knn_lists = knn_lists_np
            self.knn2, _ = self._freq_knn(knn_lists_np, ids_np,
                                          self.m, self.k2)
        else:
            # Random knn2 neighbors — O(m * k2), no O(m²) inner loop
            k = min(self.k2, self.m - 1)
            # Each cluster gets k random neighbors (may include self, fixed below)
            knn2_np = self.rng.randint(0, self.m, size=(self.m, k)).astype(np.int64)
            # Fix self-references: replace with (c+1) % m
            for j in range(k):
                self_ref = knn2_np[np.arange(self.m), j] == np.arange(self.m)
                knn2_np[self_ref, j] = (np.where(self_ref)[0] + 1) % self.m
            # Pad to k2 columns if k < k2
            if k < self.k2:
                pad = np.full((self.m, self.k2 - k), -1, dtype=np.int64)
                knn2_np = np.concatenate([knn2_np, pad], axis=1)
            # Batch distances
            knn2_dists_np = np.full((self.m, self.k2), np.inf, dtype=np.float32)
            for j in range(k):
                valid = knn2_np[:, j] >= 0
                diffs = centroids_np - centroids_np[knn2_np[:, j]]
                knn2_dists_np[:, j] = np.where(valid, (diffs * diffs).sum(axis=1), np.inf)
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
            # Lateral input wiring (permanent column-to-column connections)
            if getattr(self.column_mgr, '_lateral_inputs', False):
                lateral_k = self.column_mgr._lateral_input_k
                edges = self._generate_lateral_edges(lateral_k)
                self.column_mgr.init_lateral_wiring(
                    edges, self.n_sensory, self.column_n_outputs)
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

        _dt = os.environ.get('DEBUG_TICK_TIMING')
        if _dt:
            import time as _time
            _t0 = _time.perf_counter()

        effective_lr = self.lr

        if self.knn2_mode == 'knn':
            # knn mode: knn2 is neuron-index based (numpy)
            n_reassigned, affected, _, n_blocked, n_switches, wiring_events = self._stream_update(
                embeddings_t, self.centroids_t, self.cluster_ids, self.knn2,
                anchors_np, lr=effective_lr, sizes=self.sizes, min_size=0,
                rng=self.rng, hysteresis=self.hysteresis,
                knn2_is_neurons=True, centroid_mode=self.centroid_mode,
                pointers=self.pointers, last_used=self.last_used,
                tick=global_tick, jump_counts=self._jump_counts,
                max_cluster_size=self.max_cluster_size,
                cluster_swap=self.cluster_swap)
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
                tick=global_tick, jump_counts=self._jump_counts,
                max_cluster_size=self.max_cluster_size,
                cluster_swap=self.cluster_swap)
            self.total_reassigned += n_reassigned
            self.total_switches += n_switches
            if affected:
                most_recent = self.cluster_ids[np.arange(self.n), self.pointers]
                self.cluster_ids_t = torch.from_numpy(
                    most_recent.astype(np.int64)).to(self.device)
                self._recompute_knn2_dists(list(affected))
            if pairs is not None:
                self._update_knn2_gpu(pairs)

        if _dt:
            _t_stream = _time.perf_counter()

        # Periodic split recovery (before wiring reconciliation)
        split_events = []
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

        if _dt:
            _t_split = _time.perf_counter()

        # Unified column wiring reconciliation: process all streaming + split
        # events together using final cluster_ids state. Dedup per neuron,
        # then verify slot_map consistency for affected neurons only.
        if self.column_mgr and (wiring_events or split_events):
            neuron_evicted = {}
            affected_cols = set()
            for neuron, old_c, new_c in wiring_events:
                if neuron not in neuron_evicted:
                    neuron_evicted[neuron] = set()
                if old_c >= 0:
                    neuron_evicted[neuron].add(old_c)
                    affected_cols.add(old_c)
                if new_c >= 0:
                    affected_cols.add(new_c)
            for neuron, old_c, new_c in split_events:
                if neuron not in neuron_evicted:
                    neuron_evicted[neuron] = set()
                if old_c >= 0:
                    neuron_evicted[neuron].add(old_c)
                    affected_cols.add(old_c)
                if new_c >= 0:
                    affected_cols.add(new_c)
            for neuron, evicted_set in neuron_evicted.items():
                primary = int(self.cluster_ids[neuron, self.pointers[neuron]])
                ring = self.cluster_ids[neuron]
                for cid in ring:
                    if cid >= 0 and cid != primary:
                        self.column_mgr.unwire(int(cid), neuron)
                for old_c in evicted_set:
                    if old_c not in ring and old_c != primary:
                        self.column_mgr.unwire(int(old_c), neuron)
                self.column_mgr.wire(primary, neuron)
            # Scan only affected columns for stale entries from swap races
            slot_map = self.column_mgr.slot_map
            reserved = self.column_mgr._reserved_mask
            for c in affected_cols:
                for s in range(self.column_mgr.max_inputs):
                    if reserved[c, s]:
                        continue
                    neuron = int(slot_map[c, s])
                    if neuron >= 0 and neuron in neuron_evicted:
                        if c not in self.cluster_ids[neuron]:
                            slot_map[c, s] = -1

        if _dt:
            _t_wiring = _time.perf_counter()

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

        if _dt:
            _t_col = _time.perf_counter()
            _total = (_t_col - _t0) * 1000
            if _total > 15.0:
                print(f"    cluster_tick {global_tick}: {_total:.1f}ms "
                      f"stream={(_t_stream-_t0)*1000:.1f} "
                      f"split={(_t_split-_t_stream)*1000:.1f} "
                      f"wiring={(_t_wiring-_t_split)*1000:.1f} "
                      f"column={(_t_col-_t_wiring)*1000:.1f}")

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
        """Recompute knn2 distances on GPU. rows=None -> all rows."""
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

    def _generate_lateral_edges(self, lateral_k):
        """Generate small-world lateral topology: each column sends lateral_k outputs."""
        edges = []
        rng = np.random.RandomState(42)
        for src_col in range(self.m):
            others = [c for c in range(self.m) if c != src_col]
            dst_cols = rng.choice(others, size=min(lateral_k, len(others)),
                                  replace=False)
            for dst_col in dst_cols:
                src_out = rng.randint(0, self.column_n_outputs)
                edges.append((int(src_col), int(src_out), int(dst_col)))
        return edges

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
            # (reverse is not required -- columns can be full, not all neurons wired)
            slot_map = self.column_mgr.slot_map
            n_stale = 0
            for c in range(self.m):
                for s in range(self.column_mgr.max_inputs):
                    neuron = int(slot_map[c, s])
                    if neuron >= 0:
                        if self.column_mgr._reserved_mask[c, s]:
                            continue  # lateral reserved slot, skip check
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
                slot_map = np.load(slot_map_path)
                self.column_mgr.load_state(state, slot_map)
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

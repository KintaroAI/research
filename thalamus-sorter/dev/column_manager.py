"""Column wiring: thalamus-to-cortex connection via SoftWTACell columns.

Each cluster gets its own SoftWTACell column. Neurons wire/unwire to
their cluster's column as they enter/leave clusters via the ring buffer.
The column's input is a sliding window of raw signal (grayscale pixel
intensity) of each wired neuron, processed in streaming variance mode.

Architecture:
    saccade crop -> N neurons -> M clusters
        each cluster -> 1 SoftWTACell(n_inputs=max_inputs, n_outputs=configurable)
            input = (max_inputs, window) signal trace of wired neurons
            output = soft-WTA probabilities (streaming variance similarity)

Rule of thumb: streaming_decay ≈ 1 - 2/window
    window=4  -> decay=0.5
    window=8  -> decay=0.75
    window=16 -> decay=0.875
"""

import os
import numpy as np
import torch
import torch.nn.functional as F

# Entropy-scaled learning rate: columns with uniform outputs learn faster,
# differentiated columns learn slower (stable but can re-learn gradually).
# Set to False to use the original usage-only lr scaling.
ENTROPY_SCALED_LR = True

# Column similarity/learning mode:
#   'variance'  — streaming variance of prototype projections + power iteration
#                 (linear: finds principal components of signal covariance)
#   'kmeans'    — negative squared distance to centroids + centroid nudge
#                 (non-linear: partitions input space by nearest centroid)
COLUMN_MODE = 'kmeans'

# Lateral learning mode:
#   'contrastive' — winner pulls toward prev_outputs, losers push away
#                   (associative: memorizes co-occurring lateral patterns)
#   'covariance'  — power iteration on cross-covariance between lateral
#                   input and local projection (variance-based: finds which
#                   lateral direction correlates with local state changes)
LATERAL_LEARN_MODE = 'covariance'

# ---------------------------------------------------------------------------
# SoftWTACell — from column/dev/column.py, instantaneous + streaming modes
# (kept for standalone use and benchmarks; ColumnManager uses batched ops)
# ---------------------------------------------------------------------------

class SoftWTACell:
    """Soft winner-take-all competitive categorization cell.

    Each output unit holds a prototype vector. Inputs are compared to prototypes
    via dot-product similarity (instantaneous) or streaming variance of prototype
    projections (streaming mode), passed through softmax with temperature control.
    The winning unit's prototype moves toward the input (Hebbian pull).
    Usage counters gate plasticity to prevent collapse.

    Temporal modes:
      - None: instantaneous dot-product similarity, input (n,)
      - 'streaming': variance-based similarity, input (n,) or (n, T)
    """

    def __init__(self, n_inputs, n_outputs, temperature=0.2, lr=0.05,
                 match_threshold=0.1, usage_decay=0.99,
                 temporal_mode='streaming', streaming_decay=0.5):
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.temperature = temperature
        self.lr = lr
        self.match_threshold = match_threshold
        self.usage_decay = usage_decay
        self.temporal_mode = temporal_mode
        self.streaming_decay = streaming_decay

        # Prototype vectors (m x n), initialized on unit sphere
        self.prototypes = F.normalize(torch.randn(n_outputs, n_inputs), dim=1)

        # Usage counters — EMA of win frequency, starts uniform
        self.usage = torch.full((n_outputs,), 1.0 / n_outputs)

        # Streaming state — EMA of projection mean and variance per prototype
        if temporal_mode == 'streaming':
            self.proj_mean = torch.zeros(n_outputs)
            self.proj_var = torch.zeros(n_outputs)

    def _compute_similarity(self, x):
        """Compute per-prototype similarity scores.

        Args:
            x: (n,) for instantaneous/streaming single-step
               (n, T) for streaming windowed

        Returns:
            similarity: (m,) scores
        """
        if self.temporal_mode is None:
            x_norm = F.normalize(x.unsqueeze(0), dim=1)
            return (x_norm @ self.prototypes.T).squeeze(0)

        # streaming mode
        if x.dim() == 2:
            # (n, T) trace: project, then within-window variance — O(mnT)
            proj = self.prototypes @ x  # (m, T)
            proj_c = proj - proj.mean(dim=1, keepdim=True)
            sim = (proj_c ** 2).mean(dim=1)  # (m,)
            # Update EMA from batch
            proj_mean_batch = proj.mean(dim=1)
            d = self.streaming_decay
            self.proj_mean = d * self.proj_mean + (1 - d) * proj_mean_batch
            self.proj_var = d * self.proj_var + (1 - d) * sim
            return sim

        # (n,) single sample: EMA-based
        proj = self.prototypes @ x
        d = self.streaming_decay
        self.proj_mean = d * self.proj_mean + (1 - d) * proj
        diff = proj - self.proj_mean
        self.proj_var = d * self.proj_var + (1 - d) * diff * diff
        return self.proj_var

    def forward(self, x):
        """Compute output probabilities for input x.

        Args:
            x: (n,) for instantaneous, (n,) or (n, T) for streaming

        Returns:
            probabilities: (m,)
        """
        sim = self._compute_similarity(x)
        return F.softmax(sim / self.temperature, dim=0)

    def _update_instantaneous(self, x, probs):
        """Update for instantaneous input (n,)."""
        x_norm = F.normalize(x.unsqueeze(0), dim=1).squeeze(0)

        winner = probs.argmax().item()
        match_quality = (x_norm * self.prototypes[winner]).sum().item()

        effective_lr = self.lr / (1.0 + self.n_outputs * self.usage[winner])

        if match_quality < self.match_threshold:
            dormant = self.usage.argmin().item()
            if dormant != winner:
                winner = dormant
                effective_lr = self.lr

        self.prototypes[winner] += effective_lr * (x_norm - self.prototypes[winner])
        self.prototypes[winner] = F.normalize(self.prototypes[winner], dim=0)

        return winner, match_quality

    def _update_streaming(self, x, probs):
        """Update for streaming input — (n,) or (n, T)."""
        winner = probs.argmax().item()

        # Match quality: fraction of total variance from this prototype
        total_var = self.proj_var.sum().item()
        match_quality = self.proj_var[winner].item() / max(total_var, 1e-8)

        effective_lr = self.lr / (1.0 + self.n_outputs * self.usage[winner])

        if match_quality < self.match_threshold:
            dormant = self.usage.argmin().item()
            if dormant != winner:
                winner = dormant
                effective_lr = self.lr

        if x.dim() == 2:
            # (n, T) trace: efficient power iteration without full C
            # C @ proto = x_c @ (x_c.T @ proto) / (T-1)  — O(nT) not O(n²)
            x_c = x - x.mean(dim=1, keepdim=True)
            T = x.shape[1]
            inner = x_c.T @ self.prototypes[winner]  # (T,)
            target = x_c @ inner / max(T - 1, 1)     # (n,)
            target_norm = target.norm()
            if target_norm > 1e-8:
                target = target / target_norm
                self.prototypes[winner] += effective_lr * (target - self.prototypes[winner])
                self.prototypes[winner] = F.normalize(self.prototypes[winner], dim=0)
        else:
            # (n,) single sample: Oja's rule
            proj = (self.prototypes[winner] * x).sum()
            if proj.abs() > 1e-8:
                oja_delta = proj * (x - proj * self.prototypes[winner])
                self.prototypes[winner] += effective_lr * oja_delta
                self.prototypes[winner] = F.normalize(self.prototypes[winner], dim=0)

        return winner, match_quality

    def update(self, x, probs=None):
        """Online Hebbian update for a single input.

        Args:
            x: (n,) for instantaneous, (n,) or (n, T) for streaming

        Returns:
            winner: index of winning unit
            match_quality: cosine similarity (instantaneous) or fraction
                          of variance explained (streaming)
        """
        if probs is None:
            probs = self.forward(x)

        if self.temporal_mode == 'streaming':
            winner, match_quality = self._update_streaming(x, probs)
        else:
            winner, match_quality = self._update_instantaneous(x, probs)

        # Update usage counters
        target = torch.zeros(self.n_outputs)
        target[winner] = 1.0
        self.usage = self.usage * self.usage_decay + target * (1 - self.usage_decay)

        return winner, match_quality

    def state_dict(self):
        sd = {
            'prototypes': self.prototypes.clone(),
            'usage': self.usage.clone(),
            'n_inputs': self.n_inputs,
            'n_outputs': self.n_outputs,
            'temperature': self.temperature,
            'lr': self.lr,
            'match_threshold': self.match_threshold,
            'usage_decay': self.usage_decay,
            'temporal_mode': self.temporal_mode,
            'streaming_decay': self.streaming_decay,
        }
        if self.temporal_mode == 'streaming':
            sd['proj_mean'] = self.proj_mean.clone()
            sd['proj_var'] = self.proj_var.clone()
        return sd

    @classmethod
    def from_state_dict(cls, state):
        cell = cls(state['n_inputs'], state['n_outputs'],
                   temperature=state['temperature'], lr=state['lr'],
                   match_threshold=state['match_threshold'],
                   usage_decay=state['usage_decay'],
                   temporal_mode=state.get('temporal_mode', 'streaming'),
                   streaming_decay=state.get('streaming_decay', 0.5))
        cell.prototypes = state['prototypes']
        cell.usage = state['usage']
        if cell.temporal_mode == 'streaming':
            cell.proj_mean = state.get('proj_mean', torch.zeros(cell.n_outputs))
            cell.proj_var = state.get('proj_var', torch.zeros(cell.n_outputs))
        return cell


# ---------------------------------------------------------------------------
# ColumnManager — batched tensor ops, no per-column Python loop
# ---------------------------------------------------------------------------

class ColumnManager:
    """Manages M SoftWTACell columns, one per cluster.

    All M columns are stored as batched tensors and processed in a single
    forward+update pass (one bmm instead of M matmuls).

    Wiring map: slot_map[cluster_c, slot_s] = neuron_id (-1 = empty).
    Pre-allocates max_inputs per column (fixed size, no resizing).

    Each tick receives a sliding window of signal frames (n, window) from the
    circular signal buffer. Column 0 = most recent frame.
    """

    def __init__(self, m, n_outputs=4, max_inputs=20, window=4,
                 temperature=0.2, lr=0.05, match_threshold=0.1,
                 streaming_decay=0.5, lateral=False, lateral_k=6,
                 eligibility=False, trace_decay=0.95):
        self.m = m
        self.n_outputs = n_outputs
        self.max_inputs = max_inputs
        self.window = window
        self.temperature = temperature
        self.lr = lr
        self.match_threshold = match_threshold
        self.usage_decay = 0.99
        self.streaming_decay = streaming_decay
        self.lateral = lateral
        self.eligibility = eligibility
        self.trace_decay = trace_decay

        # Batched state: all numpy for minimal overhead on small tensors
        rng = np.random.RandomState(42)
        protos = rng.randn(m, n_outputs, max_inputs).astype(np.float32)
        if COLUMN_MODE == 'kmeans':
            protos *= 0.1  # small random centroids in input space
        else:
            norms = np.linalg.norm(protos, axis=2, keepdims=True).clip(1e-8)
            protos = protos / norms
        self.prototypes = protos
        self.usage = np.full((m, n_outputs), 1.0 / n_outputs, dtype=np.float32)
        self.proj_mean = np.zeros((m, n_outputs), dtype=np.float32)
        self.proj_var = np.zeros((m, n_outputs), dtype=np.float32)

        # Eligibility traces: deferred learning applied when reward arrives
        if eligibility:
            self.traces = np.zeros_like(protos)  # (m, n_outputs, n_inputs)
        else:
            self.traces = None
        self._pending_reward = 0.0

        # Output tiredness: consecutive wins decay the output value,
        # forcing exploration of other categories
        self.output_tiredness = np.zeros((m, n_outputs), dtype=np.float32)
        self.tiredness_rate = 0.02   # gain per tick as winner
        self.tiredness_recovery = 0.005  # recovery per tick as loser

        self.slot_map = np.full((m, max_inputs), -1, dtype=np.int64)
        self._outputs = np.zeros((m, n_outputs), dtype=np.float32)
        self._arange_m = np.arange(m)

        # Lateral connections: small-world graph with K neighbors per column.
        # K/2 from knn2 (embedding neighbors) + K/2 random long-range.
        # Adjacency list (M, K) — O(M*K) storage, not O(M²).
        # Lateral weights: (M, n_outputs, K * n_outputs).
        self.lateral_K = lateral_k  # connections per column
        if lateral:
            K_lat = min(self.lateral_K, m - 1)
            self.lateral_K = K_lat
            self.lateral_K_near = K_lat // 2  # first half = knn2 neighbors
            # Init with random wiring — knn2 synced later via sync_lateral_knn2()
            self.lateral_adj = np.full((m, K_lat), -1, dtype=np.int64)
            for c in range(m):
                others = [i for i in range(m) if i != c]
                chosen = list(rng.choice(others, size=min(K_lat, len(others)),
                                         replace=False))
                self.lateral_adj[c, :len(chosen)] = chosen

            lat_dim = K_lat * n_outputs
            lat = rng.randn(m, n_outputs, lat_dim).astype(np.float32)
            lat_norms = np.linalg.norm(lat, axis=2, keepdims=True).clip(1e-8)
            self.lateral_protos = lat / lat_norms
            self._prev_outputs = np.zeros((m, n_outputs), dtype=np.float32)
            self._lateral_rng = np.random.RandomState(123)
            self._evict_threshold = 0.1

            # Future enhancement: biased candidate selection for long-range.
            # Track per-column co-activation EMA:
            #   co_act[c, c_other] = decay * co_act + (1-decay) * both_active
            # When evicting long-range, pick most co-activated non-connected
            # column. Discovers functional relationships (A↔XOR).

    def sync_lateral_knn2(self, knn2):
        """Sync lateral adjacency with cluster knn2 neighbors.

        Replaces the first K/2 lateral connections with knn2 neighbors.
        Remaining K/2 slots keep their current (random/evolved) connections.
        Lateral weights for replaced slots are re-initialized.

        Args:
            knn2: (M, k2) array of cluster neighbor indices
        """
        if not self.lateral:
            return 0
        m, K_lat, n_out = self.m, self.lateral_K, self.n_outputs
        half = self.lateral_K_near
        n_synced = 0
        for c in range(m):
            old_near = set(self.lateral_adj[c, :half])
            new_near = []
            for j in range(knn2.shape[1]):
                nb = int(knn2[c, j])
                if nb >= 0 and nb != c and nb not in new_near:
                    new_near.append(nb)
                    if len(new_near) >= half:
                        break
            # Check if random slots duplicate any new near slots
            random_slots = self.lateral_adj[c, half:]
            for s in range(len(random_slots)):
                if random_slots[s] in new_near:
                    # Replace duplicate with a fresh random
                    connected = set(new_near) | set(random_slots)
                    cands = [i for i in range(m) if i != c and i not in connected]
                    if cands:
                        random_slots[s] = self._lateral_rng.choice(cands)
            # Write near slots
            for s in range(min(half, len(new_near))):
                if self.lateral_adj[c, s] != new_near[s]:
                    self.lateral_adj[c, s] = new_near[s]
                    # Re-init weights for changed slot
                    slot_start = s * n_out
                    slot_end = slot_start + n_out
                    for o in range(n_out):
                        w = self._lateral_rng.randn(n_out).astype(np.float32)
                        self.lateral_protos[c, o, slot_start:slot_end] = w
                        norm = np.linalg.norm(self.lateral_protos[c, o]).clip(1e-8)
                        self.lateral_protos[c, o] /= norm
                    n_synced += 1
        return n_synced

    def wire(self, cluster_id, neuron_id):
        """Wire a neuron to a cluster's column (lowest empty slot)."""
        row = self.slot_map[cluster_id]
        empty = np.where(row == -1)[0]
        if len(empty) == 0:
            return  # cluster full
        row[empty[0]] = neuron_id

    def unwire(self, cluster_id, neuron_id):
        """Unwire a neuron from a cluster's column."""
        row = self.slot_map[cluster_id]
        matches = np.where(row == neuron_id)[0]
        if len(matches) > 0:
            row[matches[0]] = -1

    # ------------------------------------------------------------------
    # Similarity + learning strategies (dispatched by COLUMN_MODE)
    # ------------------------------------------------------------------

    def _sim_variance(self, X):
        """Streaming variance similarity: project input onto prototypes,
        measure variance over time window. Linear — finds PCA directions."""
        proj = self.prototypes @ X                    # (m, n_out, window)
        proj_c = proj - proj.mean(axis=2, keepdims=True)
        sim = (proj_c ** 2).mean(axis=2)              # (m, n_out)
        d = self.streaming_decay
        self.proj_mean = d * self.proj_mean + (1 - d) * proj.mean(axis=2)
        self.proj_var = d * self.proj_var + (1 - d) * sim
        return sim

    def _sim_kmeans(self, X):
        """K-means similarity: negative squared distance from mean input
        pattern to each centroid. Non-linear — partitions by nearest."""
        x_mean = X.mean(axis=2)                       # (m, n_in)
        # prototypes (m, n_out, n_in), x_mean (m, n_in) → diff (m, n_out, n_in)
        diff = self.prototypes - x_mean[:, None, :]
        sim = -(diff ** 2).sum(axis=2)                # (m, n_out) negative distance
        # Track in proj_var for match quality (use abs similarity)
        d = self.streaming_decay
        self.proj_var = d * self.proj_var + (1 - d) * (-sim)
        return sim

    def _update_variance(self, X, actual_winners, lr_eff):
        """Power iteration: move winner prototype toward 1st eigenvector
        of input covariance. Linear learning rule."""
        m, n_out, n_in = self.m, self.n_outputs, self.max_inputs
        w = self.window
        ar = self._arange_m

        X_c = X - X.mean(axis=2, keepdims=True)       # (m, n_in, w)
        w_proto = self.prototypes[ar, actual_winners]  # (m, n_in)
        inner = np.einsum('miw,mi->mw', X_c, w_proto)
        target = np.einsum('miw,mw->mi', X_c, inner)
        target = target / max(w - 1, 1)
        target_norm = np.linalg.norm(target, axis=1, keepdims=True).clip(1e-8)
        target = target / target_norm

        do_update = target_norm.squeeze(1) > 1e-8
        new_proto = w_proto + lr_eff[:, None] * (target - w_proto)
        norms = np.linalg.norm(new_proto, axis=1, keepdims=True).clip(1e-8)
        new_proto = new_proto / norms
        update_idx = ar[do_update]
        self.prototypes[update_idx, actual_winners[do_update]] = new_proto[do_update]

    def _update_kmeans(self, X, actual_winners, lr_eff):
        """Centroid nudge: move winner prototype toward mean input pattern.
        No normalization — centroids live in input space."""
        ar = self._arange_m
        x_mean = X.mean(axis=2)                       # (m, n_in)
        w_proto = self.prototypes[ar, actual_winners]  # (m, n_in)
        new_proto = w_proto + lr_eff[:, None] * (x_mean - w_proto)
        self.prototypes[ar, actual_winners] = new_proto

    def set_reward(self, value):
        """Set pending reward. Applied on next tick(), then reset to 0."""
        self._pending_reward = value

    def tick(self, signal_window, knn2=None):
        """Batched forward + learn for all M columns. Pure numpy, no torch."""
        m, n_out, n_in = self.m, self.n_outputs, self.max_inputs
        w = self.window
        ar = self._arange_m

        # --- Build batched input (m, max_inputs, window) via gather ---
        safe_sm = np.clip(self.slot_map, 0, None)
        valid = self.slot_map >= 0                    # (m, max_inputs)
        X = signal_window[safe_sm]                    # (m, max_inputs, window)
        X[~valid] = 0.0

        # --- Similarity (mode-dependent) ---
        if COLUMN_MODE == 'kmeans':
            sim = self._sim_kmeans(X)
        else:
            sim = self._sim_variance(X)

        # Lateral input: gather from K neighbors only
        if self.lateral:
            neighbor_out = self._prev_outputs[self.lateral_adj]  # (m, K, n_out)
            lat_input = neighbor_out.reshape(m, -1)              # (m, K*n_out)
            lat_sim = np.einsum('moi,mi->mo', self.lateral_protos, lat_input)
            sim = sim + lat_sim

        # Save pre-tiredness sim for lateral learning (avoid NaN from tiredness perturbation)
        sim_for_lateral = sim.copy() if self.lateral else None

        # Apply tiredness penalty — tired outputs get suppressed
        sim_range = sim.max(axis=1, keepdims=True) - sim.min(axis=1, keepdims=True)
        sim = sim - self.output_tiredness * sim_range.clip(1e-8)

        # Softmax
        sim_scaled = sim / self.temperature
        sim_scaled -= sim_scaled.max(axis=1, keepdims=True)
        e = np.exp(sim_scaled)
        probs = e / e.sum(axis=1, keepdims=True)      # (m, n_out)

        # --- Batched update ---
        original_winners = probs.argmax(axis=1)        # (m,)

        # Match quality
        total_var = self.proj_var.sum(axis=1).clip(1e-8)
        match_q = self.proj_var[ar, original_winners] / total_var

        # Dormant reassignment
        dormant = self.usage.argmin(axis=1)
        needs_reassign = (match_q < self.match_threshold) & (dormant != original_winners)
        actual_winners = np.where(needs_reassign, dormant, original_winners)

        # Per-column learning rate
        usage_at_orig = self.usage[ar, original_winners]
        lr_normal = self.lr / (1.0 + n_out * usage_at_orig)
        lr_eff = np.where(needs_reassign, self.lr, lr_normal)

        # Entropy-scaled lr
        if ENTROPY_SCALED_LR:
            H = -(probs * np.log(probs + 1e-10)).sum(axis=1)
            H_max = np.log(n_out)
            lr_eff = lr_eff * (H / H_max)

        # --- Update prototypes (mode-dependent) ---
        if COLUMN_MODE == 'kmeans':
            self._update_kmeans(X, actual_winners, lr_eff)
        else:
            self._update_variance(X, actual_winners, lr_eff)

        # --- Eligibility traces ---
        if self.traces is not None:
            x_mean = X.mean(axis=2)                       # (m, n_in)
            w_proto = self.prototypes[ar, actual_winners]  # (m, n_in)
            direction = x_mean - w_proto                   # (m, n_in)

            # Decay all traces
            self.traces *= self.trace_decay

            # Accumulate winner's direction
            self.traces[ar, actual_winners] += direction

            # Apply when reward is pending
            if self._pending_reward != 0.0:
                self.prototypes += self.lr * self._pending_reward * self.traces
                self._pending_reward = 0.0

        # Lateral weight update
        if self.lateral:
            neighbor_out = self._prev_outputs[self.lateral_adj]  # (m, K, n_out)
            lat_in = neighbor_out.reshape(m, -1)                 # (m, K*n_out)

            if LATERAL_LEARN_MODE == 'contrastive':
                sign = np.full((m, n_out), -1.0 / (n_out - 1), dtype=np.float32)
                sign[ar, actual_winners] = 1.0
                delta = lat_in[:, None, :] - self.lateral_protos
                new_lat = self.lateral_protos + lr_eff[:, None, None] * sign[:, :, None] * delta

            elif LATERAL_LEARN_MODE == 'covariance':
                sim_c = sim_for_lateral - sim_for_lateral.mean(axis=1, keepdims=True)
                lat_target = sim_c[:, :, None] * lat_in[:, None, :]
                lat_target_norm = np.linalg.norm(lat_target, axis=2, keepdims=True).clip(1e-8)
                lat_target = lat_target / lat_target_norm
                new_lat = self.lateral_protos + lr_eff[:, None, None] * (lat_target - self.lateral_protos)

            # Guard against NaN from numerical issues
            if np.any(np.isnan(new_lat)):
                new_lat = np.nan_to_num(new_lat, nan=0.0)
            lat_norms = np.linalg.norm(new_lat, axis=2, keepdims=True).clip(1e-8)
            self.lateral_protos = new_lat / lat_norms

            # Streaming eviction: one random column per tick.
            # Near slots (< K_near) rewire to knn2 neighbor.
            # Far slots (>= K_near) rewire to random non-connected column.
            c = self._lateral_rng.randint(m)
            weight_mags = np.abs(self.lateral_protos[c]).sum(axis=0)  # (K*n_out,)
            slot_mags = weight_mags.reshape(self.lateral_K, n_out).sum(axis=1)
            weakest = int(slot_mags.argmin())
            if slot_mags[weakest] < self._evict_threshold:
                connected = set(self.lateral_adj[c])
                is_near = weakest < self.lateral_K_near
                if is_near and knn2 is not None:
                    # Replace with next knn2 neighbor not already connected
                    candidates = [int(nb) for nb in knn2[c]
                                  if nb >= 0 and nb != c and nb not in connected]
                else:
                    # Replace with random non-connected column
                    candidates = [i for i in range(m)
                                  if i != c and i not in connected]
                if candidates:
                    new_neighbor = self._lateral_rng.choice(candidates)
                    self.lateral_adj[c, weakest] = new_neighbor
                    slot_start = weakest * n_out
                    slot_end = slot_start + n_out
                    new_w = self._lateral_rng.randn(n_out, n_out).astype(np.float32)
                    new_w /= np.linalg.norm(new_w, axis=1, keepdims=True).clip(1e-8)
                    for o in range(n_out):
                        self.lateral_protos[c, o, slot_start:slot_end] = new_w[o]
                        norm = np.linalg.norm(self.lateral_protos[c, o]).clip(1e-8)
                        self.lateral_protos[c, o] /= norm

            self._prev_outputs = probs.copy()

        # Usage EMA
        usage_target = np.zeros((m, n_out), dtype=np.float32)
        usage_target[ar, actual_winners] = 1.0
        self.usage = self.usage * self.usage_decay + usage_target * (1 - self.usage_decay)

        # Output tiredness: winners get tired, losers recover
        winner_mask = np.zeros((m, n_out), dtype=np.float32)
        winner_mask[ar, actual_winners] = 1.0
        self.output_tiredness += winner_mask * self.tiredness_rate
        self.output_tiredness -= (1.0 - winner_mask) * self.tiredness_recovery
        self.output_tiredness = self.output_tiredness.clip(0.0, 0.9)

        # Store outputs
        assert not np.any(np.isnan(probs)), "NaN in column outputs"
        self._outputs = probs.astype(np.float32)

    def get_outputs(self):
        """Return (m, n_outputs) array of all column outputs."""
        return self._outputs.copy()

    def save(self, output_dir):
        """Save slot_map + batched column state."""
        import torch as _torch
        np.save(os.path.join(output_dir, "column_slot_map.npy"), self.slot_map)
        _torch.save({
            'prototypes': _torch.from_numpy(self.prototypes),
            'usage': _torch.from_numpy(self.usage),
            'proj_mean': _torch.from_numpy(self.proj_mean),
            'proj_var': _torch.from_numpy(self.proj_var),
            'm': self.m,
            'n_outputs': self.n_outputs,
            'max_inputs': self.max_inputs,
            'window': self.window,
            'temperature': self.temperature,
            'lr': self.lr,
            'match_threshold': self.match_threshold,
            'streaming_decay': self.streaming_decay,
            'lateral': self.lateral,
            'lateral_protos': _torch.from_numpy(self.lateral_protos) if self.lateral else None,
            'lateral_adj': _torch.from_numpy(self.lateral_adj) if self.lateral else None,
            'eligibility': self.eligibility,
            'trace_decay': self.trace_decay,
            'traces': _torch.from_numpy(self.traces) if self.traces is not None else None,
            'output_tiredness': _torch.from_numpy(self.output_tiredness),
        }, os.path.join(output_dir, "column_states.pt"))
        print(f"  column state saved to {output_dir}")

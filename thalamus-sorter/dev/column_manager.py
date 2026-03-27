"""Column wiring: thalamus-to-cortex connection via competitive learning columns.

Each cluster gets its own column. Neurons wire/unwire to their cluster's
column as they enter/leave clusters via the ring buffer. The column's input
is a sliding window of raw signal of each wired neuron.

Column types (selected via `type` key in column_config):
    'default'    — ColumnManager: softmax WTA with kmeans/variance similarity,
                   optional lateral connections, eligibility traces, tiredness.
    'conscience' — ConscienceColumn: hard WTA with homeostatic threshold that
                   prevents winner collapse. Normalized inputs, dead-unit reseeding.

All column types inherit from ColumnBase which provides:
    wire/unwire, slot_map, get_outputs, save/load interface.

Architecture:
    saccade crop -> N neurons -> M clusters
        each cluster -> 1 column(n_inputs=max_inputs, n_outputs=configurable)
            input = (max_inputs, window) signal trace of wired neurons
            output = competitive category probabilities

Rule of thumb for streaming_decay ≈ 1 - 2/window:
    window=4  -> decay=0.5
    window=8  -> decay=0.75
    window=16 -> decay=0.875
"""

import os
import numpy as np
import torch
import torch.nn.functional as F

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
# ColumnBase — shared interface for all column types
# ---------------------------------------------------------------------------

class ColumnBase:
    """Base class for column managers.

    Owns wiring state (slot_map), output buffer, and common parameters.
    Subclasses implement tick() with their learning algorithm.

    Interface used by ClusterManager:
        wire(cluster_id, neuron_id)
        unwire(cluster_id, neuron_id)
        tick(signal_window, knn2=None)
        get_outputs() -> (m, n_outputs) float32
        save(output_dir)
        load_state(state_dict, slot_map)

    Attributes accessed by ClusterManager:
        m, n_outputs, max_inputs, window, lateral, slot_map, prototypes, usage
    """

    def __init__(self, m, n_outputs=4, max_inputs=20, window=4, lr=0.05):
        self.m = m
        self.n_outputs = n_outputs
        self.max_inputs = max_inputs
        self.window = window
        self.lr = lr
        self.lateral = False  # subclass overrides if supported

        self.slot_map = np.full((m, max_inputs), -1, dtype=np.int64)
        self._outputs = np.zeros((m, n_outputs), dtype=np.float32)
        self._arange_m = np.arange(m)

    def wire(self, cluster_id, neuron_id):
        """Wire a neuron to a cluster's column (lowest empty slot)."""
        row = self.slot_map[cluster_id]
        empty = np.where(row == -1)[0]
        if len(empty) == 0:
            return
        row[empty[0]] = neuron_id

    def unwire(self, cluster_id, neuron_id):
        """Unwire a neuron from a cluster's column."""
        row = self.slot_map[cluster_id]
        matches = np.where(row == neuron_id)[0]
        if len(matches) > 0:
            row[matches[0]] = -1

    def get_outputs(self):
        """Return (m, n_outputs) array of all column outputs."""
        return self._outputs.copy()

    def _gather_input(self, signal_window):
        """Gather batched input (m, max_inputs, window) from signal buffer."""
        safe_sm = np.clip(self.slot_map, 0, None)
        valid = self.slot_map >= 0
        X = signal_window[safe_sm]                    # (m, max_inputs, window)
        X[~valid] = 0.0
        return X

    def tick(self, signal_window, knn2=None):
        raise NotImplementedError

    def save(self, output_dir):
        raise NotImplementedError

    def load_state(self, state, slot_map):
        raise NotImplementedError


# ---------------------------------------------------------------------------
# ColumnManager — batched tensor ops, no per-column Python loop
# ---------------------------------------------------------------------------

class ColumnManager(ColumnBase):
    """Softmax WTA columns with kmeans/variance similarity.

    All M columns are stored as batched tensors and processed in a single
    forward+update pass (one bmm instead of M matmuls). Supports optional
    lateral connections, eligibility traces, confidence gating, and tiredness.
    """

    def __init__(self, m, n_outputs=4, max_inputs=20, window=4,
                 temperature=0.2, lr=0.05, match_threshold=0.1,
                 streaming_decay=0.5, lateral=False, lateral_k=6,
                 eligibility=False, trace_decay=0.95,
                 mode='kmeans',
                 confidence_gating=False,
                 confidence_floor=0.3,
                 tiredness_rate=0.0,
                 tiredness_recovery=0.0005,
                 entropy_scaled_lr=True,
                 lateral_mode='covariance',
                 reward_lr=0.01):
        super().__init__(m, n_outputs, max_inputs, window, lr)
        self.temperature = temperature
        self.match_threshold = match_threshold
        self.usage_decay = 0.99
        self.streaming_decay = streaming_decay
        self.lateral = lateral
        self.eligibility = eligibility
        self.trace_decay = trace_decay
        self.mode = mode
        self.confidence_gating = confidence_gating
        self.confidence_floor = confidence_floor
        self.tiredness_rate = tiredness_rate
        self.tiredness_recovery = tiredness_recovery
        self.entropy_scaled_lr = entropy_scaled_lr
        self.lateral_mode = lateral_mode

        # Batched state: all numpy for minimal overhead on small tensors
        rng = np.random.RandomState(42)
        protos = rng.randn(m, n_outputs, max_inputs).astype(np.float32)
        if self.mode == 'kmeans':
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
        self.reward_lr = reward_lr

        # Output tiredness: consecutive wins decay the output value,
        # forcing exploration of other categories
        self.output_tiredness = np.zeros((m, n_outputs), dtype=np.float32)

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

    # ------------------------------------------------------------------
    # Similarity + learning strategies (dispatched by self.mode)
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
        X = self._gather_input(signal_window)

        # --- Similarity (mode-dependent) ---
        if self.mode == 'kmeans':
            sim = self._sim_kmeans(X)
        else:
            sim = self._sim_variance(X)

        # Lateral input: gather from K neighbors only
        if self.lateral:
            neighbor_out = self._prev_outputs[self.lateral_adj]  # (m, K, n_out)
            lat_input = neighbor_out.reshape(m, -1)              # (m, K*n_out)
            lat_sim = np.einsum('moi,mi->mo', self.lateral_protos, lat_input)
            sim = sim + lat_sim

        # Apply tiredness penalty — tired outputs get suppressed
        sim_range = sim.max(axis=1, keepdims=True) - sim.min(axis=1, keepdims=True)
        sim = sim - self.output_tiredness * sim_range.clip(1e-8)

        # Softmax
        sim_scaled = sim / self.temperature
        sim_scaled -= sim_scaled.max(axis=1, keepdims=True)
        e = np.exp(sim_scaled)
        probs = e / e.sum(axis=1, keepdims=True)      # (m, n_out)

        # Confidence gating: scale outputs by how peaked the distribution is.
        # confidence = floor + (1-floor) * (1 - H/H_max)
        # floor ensures columns always output some signal even when unsure
        if self.confidence_gating:
            H = -(probs * np.log(probs + 1e-10)).sum(axis=1)  # (m,)
            H_max = np.log(n_out)
            raw_conf = (1.0 - H / H_max).clip(0.0, 1.0)      # (m,)
            confidence = self.confidence_floor + (1.0 - self.confidence_floor) * raw_conf
            probs = probs * confidence[:, None]

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
        if self.entropy_scaled_lr:
            H = -(probs * np.log(probs + 1e-10)).sum(axis=1)
            H_max = np.log(n_out)
            lr_eff = lr_eff * (H / H_max)

        # Gate learning by input variance (disabled — needs more testing)
        # if CONFIDENCE_GATING:
        #     input_var = X.var(axis=2).mean(axis=1)
        #     var_gate = (input_var / (input_var + 0.01)).clip(0.0, 1.0)
        #     lr_eff = lr_eff * var_gate

        # --- Update prototypes (mode-dependent) ---
        if self.mode == 'kmeans':
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

            # Accumulate winner's direction, scaled by confidence if gating enabled
            if self.confidence_gating:
                self.traces[ar, actual_winners] += confidence[:, None] * direction
            else:
                self.traces[ar, actual_winners] += direction

            # Apply when reward is pending (fixed scale, independent of lr)
            if self._pending_reward != 0.0:
                self.prototypes += self.reward_lr * self._pending_reward * self.traces
                self._pending_reward = 0.0

        # Lateral weight update
        if self.lateral:
            neighbor_out = self._prev_outputs[self.lateral_adj]  # (m, K, n_out)
            lat_in = neighbor_out.reshape(m, -1)                 # (m, K*n_out)

            if self.lateral_mode == 'contrastive':
                sign = np.full((m, n_out), -1.0 / (n_out - 1), dtype=np.float32)
                sign[ar, actual_winners] = 1.0
                delta = lat_in[:, None, :] - self.lateral_protos
                new_lat = self.lateral_protos + lr_eff[:, None, None] * sign[:, :, None] * delta

            elif self.lateral_mode == 'covariance':
                sim_c = sim - sim.mean(axis=1, keepdims=True)
                lat_target = sim_c[:, :, None] * lat_in[:, None, :]
                lat_target_norm = np.linalg.norm(lat_target, axis=2, keepdims=True).clip(1e-8)
                lat_target = lat_target / lat_target_norm
                new_lat = self.lateral_protos + lr_eff[:, None, None] * (lat_target - self.lateral_protos)

            lat_norms = np.linalg.norm(new_lat, axis=2, keepdims=True).clip(1e-8)
            self.lateral_protos = new_lat / lat_norms
            assert not np.any(np.isnan(self.lateral_protos)), \
                "NaN in lateral_protos after update"

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

    def load_state(self, state, slot_map):
        """Restore column state from saved dict + slot_map array."""
        def _to_np(t):
            return t.numpy() if hasattr(t, 'numpy') else np.array(t)
        self.prototypes = _to_np(state['prototypes'])
        self.usage = _to_np(state['usage'])
        self.proj_mean = _to_np(state.get('proj_mean',
            torch.zeros(self.m, self.n_outputs)))
        self.proj_var = _to_np(state.get('proj_var',
            torch.zeros(self.m, self.n_outputs)))
        if self.lateral and state.get('lateral_protos') is not None:
            self.lateral_protos = _to_np(state['lateral_protos'])
        if self.traces is not None and state.get('traces') is not None:
            self.traces = _to_np(state['traces'])
        if state.get('output_tiredness') is not None:
            self.output_tiredness = _to_np(state['output_tiredness'])
        self.slot_map = slot_map


# ---------------------------------------------------------------------------
# ConscienceColumn — hard WTA with homeostatic threshold
# ---------------------------------------------------------------------------

class ConscienceColumn(ColumnBase):
    """Hard WTA columns with conscience mechanism to prevent collapse.

    Each column has m prototypes competing for inputs. Instead of softmax,
    uses hard winner-take-all with an adaptive threshold (conscience) that
    penalizes frequent winners and helps dormant units recover.

    Key differences from ColumnManager:
        - Input normalization: mean-subtract + L2 normalize (pattern, not brightness)
        - Hard WTA: argmax of (similarity - theta), not softmax
        - Conscience threshold: theta_k += alpha * (y_k - 1/n_outputs)
        - Dead-unit reseeding: replace units that haven't won in reseed_after ticks
        - Output: softmax of raw similarities (for pipeline compatibility)
    """

    def __init__(self, m, n_outputs=4, max_inputs=20, window=4,
                 lr=0.05, alpha=0.01, temperature=0.5,
                 reseed_after=1000, **kwargs):
        super().__init__(m, n_outputs, max_inputs, window, lr)
        self.alpha = alpha
        self.temperature = temperature
        self.reseed_after = reseed_after

        # Prototypes: unit-normalized random vectors
        rng = np.random.RandomState(42)
        protos = rng.randn(m, n_outputs, max_inputs).astype(np.float32)
        norms = np.linalg.norm(protos, axis=2, keepdims=True).clip(1e-8)
        self.prototypes = protos / norms

        self.usage = np.full((m, n_outputs), 1.0 / n_outputs, dtype=np.float32)

        # Conscience thresholds: one per output per column
        self.theta = np.zeros((m, n_outputs), dtype=np.float32)

        # Dead-unit tracking
        self.last_won = np.zeros((m, n_outputs), dtype=np.int64)
        self._tick_count = 0

    def _normalize_input(self, X):
        """Mean-subtract and L2-normalize the temporal mean of each column's input.

        Args:
            X: (m, max_inputs, window) raw signal window

        Returns:
            x: (m, max_inputs) normalized input pattern
        """
        x = X.mean(axis=2)                                     # (m, max_inputs)
        x = x - x.mean(axis=1, keepdims=True)                  # mean-subtract
        norms = np.linalg.norm(x, axis=1, keepdims=True).clip(1e-8)
        return x / norms                                        # L2-normalize

    def tick(self, signal_window, knn2=None):
        """Hard WTA with conscience. Pure numpy."""
        m, n_out, n_in = self.m, self.n_outputs, self.max_inputs
        ar = self._arange_m
        self._tick_count += 1

        # --- Gather and normalize input ---
        X = self._gather_input(signal_window)
        x = self._normalize_input(X)                           # (m, n_in)

        # --- Similarity: cosine (protos are unit-norm) ---
        sim = np.einsum('moi,mi->mo', self.prototypes, x)     # (m, n_out)

        # --- Hard WTA with conscience threshold ---
        scores = sim - self.theta
        winners = scores.argmax(axis=1)                        # (m,)

        # --- Winner prototype update: pull toward input, renormalize ---
        w_proto = self.prototypes[ar, winners]                 # (m, n_in)
        new_proto = (1.0 - self.lr) * w_proto + self.lr * x
        norms = np.linalg.norm(new_proto, axis=1, keepdims=True).clip(1e-8)
        self.prototypes[ar, winners] = new_proto / norms

        # --- Conscience threshold: push toward 1/n_out usage ---
        y = np.zeros((m, n_out), dtype=np.float32)
        y[ar, winners] = 1.0
        self.theta += self.alpha * (y - 1.0 / n_out)

        # --- Usage EMA ---
        self.usage = self.usage * 0.99 + y * 0.01

        # --- Dead-unit reseeding ---
        self.last_won[ar, winners] = self._tick_count
        if self.reseed_after > 0:
            dead_ticks = self._tick_count - self.last_won
            dead_cols, dead_outs = np.where(dead_ticks > self.reseed_after)
            if len(dead_cols) > 0:
                for c, o in zip(dead_cols, dead_outs):
                    self.prototypes[c, o] = x[c]
                    self.theta[c, o] = 0.0
                    self.last_won[c, o] = self._tick_count

        # --- Output: softmax of raw similarities (pipeline compatibility) ---
        sim_scaled = sim / self.temperature
        sim_scaled -= sim_scaled.max(axis=1, keepdims=True)
        e = np.exp(sim_scaled)
        probs = e / e.sum(axis=1, keepdims=True)

        self._outputs = probs.astype(np.float32)

    def save(self, output_dir):
        """Save column state."""
        import torch as _torch
        np.save(os.path.join(output_dir, "column_slot_map.npy"), self.slot_map)
        _torch.save({
            'type': 'conscience',
            'prototypes': _torch.from_numpy(self.prototypes),
            'usage': _torch.from_numpy(self.usage),
            'theta': _torch.from_numpy(self.theta),
            'last_won': _torch.from_numpy(self.last_won.astype(np.float32)),
            'tick_count': self._tick_count,
            'm': self.m,
            'n_outputs': self.n_outputs,
            'max_inputs': self.max_inputs,
            'window': self.window,
            'lr': self.lr,
            'alpha': self.alpha,
            'temperature': self.temperature,
            'reseed_after': self.reseed_after,
        }, os.path.join(output_dir, "column_states.pt"))
        print(f"  conscience column state saved to {output_dir}")

    def load_state(self, state, slot_map):
        """Restore column state."""
        def _to_np(t):
            return t.numpy() if hasattr(t, 'numpy') else np.array(t)
        self.prototypes = _to_np(state['prototypes'])
        self.usage = _to_np(state['usage'])
        if 'theta' in state:
            self.theta = _to_np(state['theta'])
        if 'last_won' in state:
            self.last_won = _to_np(state['last_won']).astype(np.int64)
        if 'tick_count' in state:
            self._tick_count = int(state['tick_count'])
        self.slot_map = slot_map

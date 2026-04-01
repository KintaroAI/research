"""Column wiring: thalamus-to-cortex connection via competitive learning columns.

Each cluster gets its own column. Neurons wire/unwire to their cluster's
column as they enter/leave clusters via the ring buffer. The column's input
is a sliding window of raw signal of each wired neuron.

Column types (selected via `type` key in column_config):
    'default'    — ColumnManager: softmax WTA with kmeans/variance similarity,
                   optional lateral connections, eligibility traces, tiredness.
    'conscience' — ConscienceColumn: hard WTA with homeostatic threshold that
                   prevents winner collapse. Normalized inputs, dead-unit reseeding.
    'predictive' — PredictiveColumn: 1-layer causal transformer encoder with
                   category bottleneck. Learns via next-frame prediction loss +
                   entropy regularization. No conscience hack needed.

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
        self._reserved_mask = np.zeros((m, max_inputs), dtype=bool)
        self._outputs = np.zeros((m, n_outputs), dtype=np.float32)
        self._arange_m = np.arange(m)

        # Output modifiers (all off by default)
        self.theta = np.zeros((m, n_outputs), dtype=np.float32)
        self.alpha = 0.0              # conscience rate (0=disabled)
        self.output_tiredness = np.zeros((m, n_outputs), dtype=np.float32)
        self.tiredness_rate = 0.0     # 0=disabled
        self.tiredness_recovery = 0.0
        self.wta_mode = 'none'        # 'none', 'soft', 'hard', 'confidence'

    def wire(self, cluster_id, neuron_id):
        """Wire a neuron to a cluster's column (lowest empty unreserved slot)."""
        row = self.slot_map[cluster_id]
        empty = np.where((row == -1) & ~self._reserved_mask[cluster_id])[0]
        if len(empty) == 0:
            return
        row[empty[0]] = neuron_id

    def unwire(self, cluster_id, neuron_id):
        """Unwire a neuron from a cluster's column (skip reserved slots)."""
        row = self.slot_map[cluster_id]
        matches = np.where((row == neuron_id) & ~self._reserved_mask[cluster_id])[0]
        if len(matches) > 0:
            row[matches[0]] = -1

    def init_lateral_wiring(self, edges, n_sensory, n_outputs):
        """Wire lateral connections permanently into reserved slots.

        Args:
            edges: list of (src_col, src_out, dst_col)
            n_sensory: number of sensory neurons (to compute feedback neuron IDs)
            n_outputs: outputs per column
        """
        n_wired = 0
        n_dropped = 0
        for src_col, src_out, dst_col in edges:
            fb_neuron = n_sensory + src_col * n_outputs + src_out
            row = self.slot_map[dst_col]
            wired = False
            for s in range(self.max_inputs - 1, -1, -1):
                if row[s] == -1 and not self._reserved_mask[dst_col, s]:
                    row[s] = fb_neuron
                    self._reserved_mask[dst_col, s] = True
                    wired = True
                    n_wired += 1
                    break
            if not wired:
                n_dropped += 1
        print(f"  Lateral wiring: {n_wired} permanent connections")
        if n_dropped > 0:
            print(f"  WARNING: {n_dropped} lateral connections dropped (no free slots)")

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

    def apply_output_rotation(self, sim):
        """Apply conscience theta + tiredness penalty to similarity scores.

        Args:
            sim: (m, n_outputs) raw similarity scores (numpy)
        Returns:
            sim: (m, n_outputs) adjusted scores
        """
        if self.alpha != 0.0:
            sim = sim - self.theta
        if self.tiredness_rate > 0.0:
            sim_range = sim.max(axis=1, keepdims=True) - sim.min(axis=1, keepdims=True)
            sim = sim - self.output_tiredness * np.clip(sim_range, 1e-8, None)
        return sim

    def update_output_rotation(self, winners):
        """Update conscience theta + tiredness after determining winners.

        Args:
            winners: (m,) int array of winning output indices
        """
        m, n_out = self.m, self.n_outputs
        winner_mask = np.zeros((m, n_out), dtype=np.float32)
        winner_mask[np.arange(m), winners] = 1.0
        if self.alpha != 0.0:
            self.theta += self.alpha * (winner_mask - 1.0 / n_out)
        if self.tiredness_rate > 0.0:
            self.output_tiredness += winner_mask * self.tiredness_rate
            self.output_tiredness -= (1.0 - winner_mask) * self.tiredness_recovery
            self.output_tiredness = self.output_tiredness.clip(0.0, 0.9)

    def apply_wta(self, p):
        """Apply WTA to probability output.

        Args:
            p: (m, n_outputs) soft probabilities (numpy)
        Returns:
            (m, n_outputs) — mode-dependent transformation
        """
        if self.wta_mode == 'hard':
            out = np.zeros_like(p)
            out[np.arange(self.m), p.argmax(axis=1)] = 1.0
            return out
        elif self.wta_mode == 'soft':
            log_p = np.log(p + 1e-10)
            log_p -= log_p.max(axis=1, keepdims=True)
            e = np.exp(log_p * 3.0)
            return (e / e.sum(axis=1, keepdims=True)).astype(np.float32)
        elif self.wta_mode == 'confidence':
            # Scale by decisiveness: 0 when uniform, 1 when one-hot
            H = -(p * np.log(p + 1e-10)).sum(axis=1, keepdims=True)  # (m, 1)
            H_max = np.log(self.n_outputs)
            confidence = (1.0 - H / H_max).clip(0.0, 1.0)  # (m, 1)
            return (p * confidence).astype(np.float32)
        return p

    def save_rotation_state(self):
        """Return dict of output modifier state for save()."""
        return {
            'theta': torch.from_numpy(self.theta),
            'alpha': self.alpha,
            'output_tiredness': torch.from_numpy(self.output_tiredness),
            'tiredness_rate': self.tiredness_rate,
            'tiredness_recovery': self.tiredness_recovery,
            'wta_mode': self.wta_mode,
        }

    def load_rotation_state(self, state):
        """Restore output modifier state from loaded dict."""
        if 'theta' in state:
            t = state['theta']
            self.theta = t.numpy().astype(np.float32) if hasattr(t, 'numpy') else np.array(t, dtype=np.float32)
        if 'alpha' in state:
            self.alpha = float(state['alpha'])
        if 'output_tiredness' in state:
            t = state['output_tiredness']
            self.output_tiredness = t.numpy().astype(np.float32) if hasattr(t, 'numpy') else np.array(t, dtype=np.float32)
        if 'tiredness_rate' in state:
            self.tiredness_rate = float(state['tiredness_rate'])
        if 'tiredness_recovery' in state:
            self.tiredness_recovery = float(state['tiredness_recovery'])
        if 'wta_mode' in state:
            self.wta_mode = state['wta_mode']

    def set_reward(self, value):
        """Set pending reward (no-op in base; subclasses may use for eligibility)."""
        pass

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
                 reward_lr=0.01,
                 lateral_inputs=False, lateral_input_k=4):
        self._lateral_inputs = lateral_inputs
        self._lateral_input_k = lateral_input_k
        if lateral_inputs:
            max_inputs = max_inputs + lateral_input_k * 2
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
            '_reserved_mask': _torch.from_numpy(self._reserved_mask.astype(np.uint8)),
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
        if state.get('_reserved_mask') is not None:
            self._reserved_mask = _to_np(state['_reserved_mask']).astype(bool)
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
        wta_mode = kwargs.pop('wta_mode', 'none')
        self._lateral_inputs = kwargs.pop('lateral_inputs', False)
        self._lateral_input_k = kwargs.pop('lateral_input_k', 4)
        if self._lateral_inputs:
            max_inputs = max_inputs + self._lateral_input_k * 2
        super().__init__(m, n_outputs, max_inputs, window, lr)
        self.alpha = alpha
        self.temperature = temperature
        self.reseed_after = reseed_after
        self.wta_mode = wta_mode

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

        # GPU path available but CPU is faster at current scales (transfer overhead)
        self._device = torch.device('cpu')
        self._protos_gpu = None
        self._theta_gpu = None

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
        """Hard WTA with conscience. GPU-accelerated if available."""
        self._tick_count += 1

        if self._device.type == 'cuda':
            self._tick_gpu(signal_window)
        else:
            self._tick_cpu(signal_window)

    def _tick_gpu(self, signal_window):
        """GPU path: gather on CPU, everything else on GPU."""
        m, n_out, n_in = self.m, self.n_outputs, self.max_inputs
        dev = self._device

        # Gather + normalize (CPU — slot_map indexing is irregular)
        X = self._gather_input(signal_window)
        x_np = self._normalize_input(X)
        x = torch.from_numpy(x_np).to(dev)

        # Cosine similarity: (m, n_out, n_in) @ (m, n_in, 1) -> (m, n_out)
        sim = torch.bmm(self._protos_gpu, x.unsqueeze(2)).squeeze(2)

        # Hard WTA with conscience
        scores = sim - self._theta_gpu
        winners = scores.argmax(dim=1)  # (m,)

        # Winner prototype update
        ar = torch.arange(m, device=dev)
        w_proto = self._protos_gpu[ar, winners]  # (m, n_in)
        new_proto = (1.0 - self.lr) * w_proto + self.lr * x
        new_proto = new_proto / new_proto.norm(dim=1, keepdim=True).clamp(min=1e-8)
        self._protos_gpu[ar, winners] = new_proto

        # Conscience threshold
        y = torch.zeros(m, n_out, device=dev)
        y[ar, winners] = 1.0
        self._theta_gpu += self.alpha * (y - 1.0 / n_out)

        # Dead-unit reseeding (CPU — rare, branchy)
        winners_np = winners.cpu().numpy()
        self.last_won[self._arange_m, winners_np] = self._tick_count
        if self.reseed_after > 0:
            dead_ticks = self._tick_count - self.last_won
            dead_cols, dead_outs = np.where(dead_ticks > self.reseed_after)
            if len(dead_cols) > 0:
                for c, o in zip(dead_cols, dead_outs):
                    self._protos_gpu[c, o] = x[c]
                    self._theta_gpu[c, o] = 0.0
                    self.last_won[c, o] = self._tick_count

        # Output softmax
        sim_scaled = sim / self.temperature
        sim_scaled -= sim_scaled.max(dim=1, keepdim=True).values
        probs = torch.softmax(sim_scaled, dim=1)

        # Usage EMA + output (minimal CPU sync)
        self.usage = self.usage * 0.99 + y.cpu().numpy() * 0.01
        self._outputs = self.apply_wta(probs.cpu().numpy().astype(np.float32))

    def _tick_cpu(self, signal_window):
        """Original numpy path."""
        m, n_out, n_in = self.m, self.n_outputs, self.max_inputs
        ar = self._arange_m

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

        self._outputs = self.apply_wta(probs.astype(np.float32))

    def _sync_from_gpu(self):
        """Sync GPU state back to numpy arrays."""
        if self._device.type == 'cuda':
            self.prototypes = self._protos_gpu.cpu().numpy()
            self.theta = self._theta_gpu.cpu().numpy()

    def save(self, output_dir):
        """Save column state."""
        self._sync_from_gpu()
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
            '_reserved_mask': _torch.from_numpy(self._reserved_mask.astype(np.uint8)),
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
        if state.get('_reserved_mask') is not None:
            self._reserved_mask = _to_np(state['_reserved_mask']).astype(bool)
        self.slot_map = slot_map
        # Re-upload to GPU
        if self._device.type == 'cuda':
            self._protos_gpu = torch.from_numpy(self.prototypes).to(self._device)
            self._theta_gpu = torch.from_numpy(self.theta).to(self._device)



# ---------------------------------------------------------------------------
# TransformerColumn — unified transformer encoder with category bottleneck
# ---------------------------------------------------------------------------

class TransformerColumn(ColumnBase):
    """Transformer encoder with category bottleneck and configurable loss.

    1-layer causal transformer encodes signal window -> summary -> category
    probabilities via cosine similarity to learned embeddings. Prediction/
    reconstruction flows through the category bottleneck (normalized values).

    loss_mode:
        'predictive' - predict next frame at every window position (T-1 targets)
        'recon'      - reconstruct current frame from last position (1 target)

    Anti-collapse: EMA balance loss, orthogonality on category embeddings,
    cosine logits. No L_sharp by default.
    """

    def __init__(self, m, n_outputs=4, max_inputs=20, window=4,
                 lr=1e-3, temperature=2.0, n_heads=2,
                 lambda_sharp=0.0, lambda_balance=0.1,
                 lambda_ortho=0.01, alpha=0.0,
                 tiredness_rate=0.0, tiredness_recovery=0.0,
                 loss_mode='predictive', **kwargs):
        wta_mode = kwargs.pop('wta_mode', 'none')
        self._lateral_inputs = kwargs.pop('lateral_inputs', False)
        self._lateral_input_k = kwargs.pop('lateral_input_k', 4)
        if self._lateral_inputs:
            max_inputs = max_inputs + self._lateral_input_k * 2
        super().__init__(m, n_outputs, max_inputs, window, lr)
        self.temperature = temperature
        self.n_heads = n_heads
        self.lambda_sharp = lambda_sharp
        self.lambda_balance = lambda_balance
        self.lambda_ortho = lambda_ortho
        self.loss_mode = loss_mode
        self.alpha = alpha
        self.tiredness_rate = tiredness_rate
        self.tiredness_recovery = tiredness_recovery
        self.wta_mode = wta_mode

        d_model = max_inputs
        d_ff = 4 * d_model
        self.d_model = d_model
        self.d_ff = d_ff
        assert d_model % n_heads == 0, \
            f"d_model={d_model} must be divisible by n_heads={n_heads}"

        # --- Encoder parameters (all with leading m dimension) ---
        self.pos_emb = torch.nn.Parameter(torch.randn(m, window, d_model) * 0.02)
        self.ln1_g = torch.nn.Parameter(torch.ones(m, d_model))
        self.ln1_b = torch.nn.Parameter(torch.zeros(m, d_model))
        self.W_qkv = torch.nn.Parameter(torch.randn(m, d_model, 3 * d_model) * (d_model ** -0.5))
        self.W_proj = torch.nn.Parameter(torch.randn(m, d_model, d_model) * (d_model ** -0.5))
        self.ln2_g = torch.nn.Parameter(torch.ones(m, d_model))
        self.ln2_b = torch.nn.Parameter(torch.zeros(m, d_model))
        self.W_fc1 = torch.nn.Parameter(torch.randn(m, d_model, d_ff) * (d_model ** -0.5))
        self.b_fc1 = torch.nn.Parameter(torch.zeros(m, d_ff))
        self.W_fc2 = torch.nn.Parameter(torch.randn(m, d_ff, d_model) * (d_ff ** -0.5))
        self.b_fc2 = torch.nn.Parameter(torch.zeros(m, d_model))

        # --- Head: prediction/reconstruction through bottleneck ---
        self.cat_embs = torch.nn.Parameter(torch.randn(m, n_outputs, d_model) * 0.1)
        self.W_head = torch.nn.Parameter(torch.randn(m, d_model, d_model) * (d_model ** -0.5))
        self.b_head = torch.nn.Parameter(torch.zeros(m, d_model))

        self._causal_mask_np = np.tril(np.ones((window, window), dtype=bool))
        self._np_dirty = True

        # --- State ---
        self._prev_prediction = np.zeros((m, d_model), dtype=np.float32)
        self._surprise = np.zeros(m, dtype=np.float32)
        self._tick_count = 0
        self._learn_prob = 1.0
        self._rng = np.random.RandomState(42)
        self._usage_ema = np.full((m, n_outputs), 1.0 / n_outputs, dtype=np.float32)

        self._params = [
            self.pos_emb, self.ln1_g, self.ln1_b,
            self.W_qkv, self.W_proj,
            self.ln2_g, self.ln2_b,
            self.W_fc1, self.b_fc1, self.W_fc2, self.b_fc2,
            self.cat_embs, self.W_head, self.b_head,
        ]
        self._param_names = [
            'pos_emb', 'ln1_g', 'ln1_b', 'W_qkv', 'W_proj',
            'ln2_g', 'ln2_b', 'W_fc1', 'b_fc1', 'W_fc2', 'b_fc2',
            'cat_embs', 'W_head', 'b_head',
        ]
        self._optimizer = torch.optim.Adam(self._params, lr=lr)

        self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if self._device.type == 'cuda':
            for p in self._params:
                p.data = p.data.to(self._device)
            self._optimizer = torch.optim.Adam(self._params, lr=lr)
        self._causal_mask = torch.tril(torch.ones(window, window, device=self._device))
        self._inv_sqrt_hs = (d_model // n_heads) ** -0.5

    def _layer_norm(self, x, g, b):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        x = (x - mean) / (var + 1e-5).sqrt()
        return x * g.unsqueeze(1) + b.unsqueeze(1)

    def _encode_gpu(self, x_gpu):
        """Encode window -> all hidden states. x_gpu: (m, T, d) on device.
        Returns x: (m, T, d) hidden states at all positions."""
        m, T, d = x_gpu.shape
        n_heads = self.n_heads
        hs = d // n_heads

        x = x_gpu + self.pos_emb
        h = self._layer_norm(x, self.ln1_g, self.ln1_b)
        qkv = torch.bmm(h, self.W_qkv)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(m, T, n_heads, hs).transpose(1, 2)
        k = k.view(m, T, n_heads, hs).transpose(1, 2)
        v = v.view(m, T, n_heads, hs).transpose(1, 2)
        attn = torch.matmul(q, k.transpose(-2, -1)) * self._inv_sqrt_hs
        mask = self._causal_mask[:T, :T]
        attn = attn.masked_fill(mask == 0, -1e9)
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(m, T, d)
        out = torch.bmm(out, self.W_proj)
        x = x + out

        h = self._layer_norm(x, self.ln2_g, self.ln2_b)
        ff = torch.bmm(h, self.W_fc1) + self.b_fc1.unsqueeze(1)
        ff = F.gelu(ff)
        ff = torch.bmm(ff, self.W_fc2) + self.b_fc2.unsqueeze(1)
        x = x + ff
        return x

    def _categorize_gpu(self, z):
        """Cosine logits: z (m, d) or (m, T, d) -> p, c_n."""
        c_n = self.cat_embs / self.cat_embs.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        if z.dim() == 2:
            z_n = z / z.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            sim = torch.bmm(z_n.unsqueeze(1), c_n.transpose(1, 2)).squeeze(1)
        else:
            z_n = z / z.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            sim = torch.bmm(z_n, c_n.transpose(1, 2))
        p = F.softmax(sim / self.temperature, dim=-1)
        return p, c_n

    def _compute_loss(self, x_hidden, x_gpu, dev):
        """Compute training loss based on loss_mode."""
        m, n_out = self.m, self.n_outputs
        c_n = self.cat_embs / self.cat_embs.norm(dim=-1, keepdim=True).clamp(min=1e-8)

        valid_gpu = torch.from_numpy(
            (self.slot_map >= 0).astype(np.float32)).to(dev)
        n_valid_gpu = valid_gpu.sum(dim=1).clamp(min=1.0)

        if self.loss_mode == 'predictive':
            # Next-frame prediction at all T-1 positions
            h_ctx = x_hidden[:, :-1, :]
            targets = x_gpu[:, 1:, :]
            p_all, _ = self._categorize_gpu(h_ctx)
            z_q_all = torch.bmm(p_all, c_n)
            pred_all = torch.bmm(z_q_all, self.W_head) + self.b_head.unsqueeze(1)
            diff_all = (pred_all - targets) ** 2
            diff_masked = diff_all * valid_gpu.unsqueeze(1)
            L_main = diff_masked.sum(dim=2).div(n_valid_gpu.unsqueeze(1)).mean()
            p_for_balance = p_all.mean(dim=1)  # (m, n_out) avg over positions
        else:  # recon
            # Reconstruct current frame from last position
            z = x_hidden[:, -1, :]
            p, _ = self._categorize_gpu(z)
            z_q = torch.bmm(p.unsqueeze(1), c_n).squeeze(1)
            recon = torch.bmm(z_q.unsqueeze(1), self.W_head).squeeze(1) + self.b_head
            target = x_gpu[:, -1, :]
            L_main = ((recon - target) ** 2 * valid_gpu).sum(dim=1).div(n_valid_gpu).mean()
            p_for_balance = p  # (m, n_out) single position

        # L_sharp (disabled by default)
        L_sharp = torch.tensor(0.0, device=dev)
        if self.lambda_sharp > 0:
            H = -(p_for_balance * (p_for_balance + 1e-10).log()).sum(dim=-1)
            L_sharp = H.mean()

        # L_balance: EMA usage per column -> uniform (live grad)
        usage_ema = torch.from_numpy(self._usage_ema).to(dev)
        beta = 0.9
        usage_est = beta * usage_ema + (1.0 - beta) * p_for_balance
        uniform = torch.full_like(usage_est, 1.0 / n_out)
        L_balance = ((usage_est - uniform) ** 2).sum(dim=-1).mean()
        self._usage_ema = (
            beta * self._usage_ema
            + (1.0 - beta) * p_for_balance.detach().cpu().numpy()
        ).astype(np.float32)

        # L_ortho: push category embeddings apart
        L_ortho = torch.tensor(0.0, device=dev)
        if self.lambda_ortho > 0:
            gram = torch.bmm(c_n, c_n.transpose(1, 2))
            eye = torch.eye(n_out, device=dev).unsqueeze(0)
            L_ortho = ((gram - eye) ** 2).mean()

        return (L_main + self.lambda_sharp * L_sharp
                + self.lambda_balance * L_balance
                + self.lambda_ortho * L_ortho)

    def _forward_np(self, x):
        """Pure numpy encoder forward. Returns (sim, p, predicted).
        x: (m, T, d_model) numpy float32."""
        m, T, d = x.shape
        n_heads = self.n_heads
        hs = d // n_heads

        if not hasattr(self, '_np_cache') or self._np_dirty:
            self._np_cache = {n: self._params[i].detach().cpu().numpy()
                              for i, n in enumerate(self._param_names)}
            self._np_dirty = False
        c = self._np_cache

        # Encode
        x = x + c['pos_emb']
        mean = x.mean(axis=-1, keepdims=True)
        var = x.var(axis=-1, keepdims=True)
        h = (x - mean) / np.sqrt(var + 1e-5)
        h = h * c['ln1_g'][:, None, :] + c['ln1_b'][:, None, :]

        qkv = h @ c['W_qkv']
        q, k, v = np.split(qkv, 3, axis=-1)
        q = q.reshape(m, T, n_heads, hs).transpose(0, 2, 1, 3)
        k = k.reshape(m, T, n_heads, hs).transpose(0, 2, 1, 3)
        v = v.reshape(m, T, n_heads, hs).transpose(0, 2, 1, 3)
        attn = (q @ k.transpose(0, 1, 3, 2)) / np.sqrt(hs)
        causal = self._causal_mask_np[:T, :T]
        attn = np.where(causal[None, None, :, :], attn, -1e9)
        attn_max = attn.max(axis=-1, keepdims=True)
        e = np.exp(attn - attn_max)
        attn = e / e.sum(axis=-1, keepdims=True)
        out = attn @ v
        out = out.transpose(0, 2, 1, 3).reshape(m, T, d)
        out = out @ c['W_proj']
        x = x + out

        mean = x.mean(axis=-1, keepdims=True)
        var = x.var(axis=-1, keepdims=True)
        h = (x - mean) / np.sqrt(var + 1e-5)
        h = h * c['ln2_g'][:, None, :] + c['ln2_b'][:, None, :]
        ff = h @ c['W_fc1'] + c['b_fc1'][:, None, :]
        ff = ff * 0.5 * (1.0 + np.tanh(0.7978845608 * (ff + 0.044715 * ff ** 3)))
        ff = ff @ c['W_fc2'] + c['b_fc2'][:, None, :]
        x = x + ff

        z = x[:, -1, :]
        # Cosine logits + normalized values
        z_n = z / np.clip(np.linalg.norm(z, axis=-1, keepdims=True), 1e-8, None)
        c_n = c['cat_embs'] / np.clip(
            np.linalg.norm(c['cat_embs'], axis=-1, keepdims=True), 1e-8, None)
        sim = (z_n[:, None, :] @ c_n.transpose(0, 2, 1))[:, 0, :]
        sim_scaled = sim / self.temperature
        sim_scaled -= sim_scaled.max(axis=-1, keepdims=True)
        e = np.exp(sim_scaled)
        p = e / e.sum(axis=-1, keepdims=True)

        z_q = (p[:, None, :] @ c_n)[:, 0, :]
        predicted = (z_q[:, None, :] @ c['W_head'])[:, 0, :] + c['b_head']

        return sim.astype(np.float32), p.astype(np.float32), predicted.astype(np.float32)

    def tick(self, signal_window, knn2=None):
        m, n_out, d = self.m, self.n_outputs, self.d_model
        dev = self._device

        X = self._gather_input(signal_window)
        x_np = X.transpose(0, 2, 1).astype(np.float32)

        # Surprise from previous prediction
        if self._tick_count > 0:
            current_frame_np = x_np[:, -1, :]
            valid = (self.slot_map >= 0).astype(np.float32)
            n_valid = valid.sum(axis=1).clip(min=1.0)
            diff = (self._prev_prediction - current_frame_np) ** 2
            self._surprise = (diff * valid).sum(axis=1) / n_valid

        if dev.type == 'cuda':
            do_train = self._rng.rand() < self._learn_prob
            x_gpu = torch.from_numpy(x_np).to(dev)

            if do_train:
                x_hidden = self._encode_gpu(x_gpu)
                loss = self._compute_loss(x_hidden, x_gpu, dev)
                self._optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self._params, 1.0)
                self._optimizer.step()
                self._np_dirty = True

            # Rerun forward with current weights
            with torch.no_grad():
                x_hidden = self._encode_gpu(x_gpu)

            # Output: unrotated for prediction, rotated for external
            with torch.no_grad():
                z_last = x_hidden[:, -1, :]
                p_clean, c_n = self._categorize_gpu(z_last)
                z_q = torch.bmm(p_clean.unsqueeze(1), c_n).squeeze(1)
                pred_last = torch.bmm(z_q.unsqueeze(1),
                                      self.W_head).squeeze(1) + self.b_head

                sim_last = torch.bmm(
                    (z_last / z_last.norm(dim=-1, keepdim=True).clamp(min=1e-8)).unsqueeze(1),
                    c_n.transpose(1, 2)).squeeze(1)
                sim_np = sim_last.cpu().numpy()
                sim_np = self.apply_output_rotation(sim_np)
                sim_adj = torch.from_numpy(sim_np).to(dev)
                p_out = F.softmax(sim_adj / self.temperature, dim=-1)

            p_np = p_out.cpu().numpy().astype(np.float32)
            self.update_output_rotation(p_np.argmax(axis=1))
            self._outputs = self.apply_wta(p_np)
            self._prev_prediction = pred_last.cpu().numpy().astype(np.float32)
        else:
            # CPU fallback with rotation
            sim, p, predicted = self._forward_np(x_np)
            sim_rot = self.apply_output_rotation(sim)
            sim_scaled = sim_rot / self.temperature
            sim_scaled -= sim_scaled.max(axis=-1, keepdims=True)
            e = np.exp(sim_scaled)
            p_out = (e / e.sum(axis=-1, keepdims=True)).astype(np.float32)
            self.update_output_rotation(p_out.argmax(axis=1))
            self._outputs = self.apply_wta(p_out)
            self._prev_prediction = predicted

        self._tick_count += 1

    def get_surprise(self):
        return self._surprise.copy()

    def set_learn_prob(self, value):
        self._learn_prob = float(value)

    def save(self, output_dir):
        np.save(os.path.join(output_dir, "column_slot_map.npy"), self.slot_map)
        state = {
            'type': self.loss_mode,
            'tick_count': self._tick_count,
            'm': self.m, 'n_outputs': self.n_outputs,
            'max_inputs': self.max_inputs, 'window': self.window,
            'lr': self.lr, 'temperature': self.temperature,
            'n_heads': self.n_heads, 'loss_mode': self.loss_mode,
            'lambda_sharp': self.lambda_sharp,
            'lambda_balance': self.lambda_balance,
            'lambda_ortho': self.lambda_ortho,
            'd_model': self.d_model, 'd_ff': self.d_ff,
            '_reserved_mask': torch.from_numpy(self._reserved_mask.astype(np.uint8)),
            '_prev_prediction': torch.from_numpy(self._prev_prediction),
            '_usage_ema': torch.from_numpy(self._usage_ema),
        }
        state.update(self.save_rotation_state())
        for i, name in enumerate(self._param_names):
            state[name] = self._params[i].detach().cpu()
        state['optimizer_state'] = self._optimizer.state_dict()
        torch.save(state, os.path.join(output_dir, "column_states.pt"))
        print(f"  {self.loss_mode} column state saved to {output_dir}")

    def load_state(self, state, slot_map):
        self.slot_map = slot_map
        if state.get('_reserved_mask') is not None:
            t = state['_reserved_mask']
            arr = t.numpy() if hasattr(t, 'numpy') else np.array(t)
            self._reserved_mask = arr.astype(bool)
        if 'tick_count' in state:
            self._tick_count = int(state['tick_count'])
        if '_prev_prediction' in state:
            t = state['_prev_prediction']
            self._prev_prediction = t.numpy().astype(np.float32) if hasattr(t, 'numpy') else np.array(t, dtype=np.float32)
        if '_usage_ema' in state:
            t = state['_usage_ema']
            self._usage_ema = t.numpy().astype(np.float32) if hasattr(t, 'numpy') else np.array(t, dtype=np.float32)
        # Handle legacy param names (W_pred/b_pred -> W_head/b_head)
        for i, name in enumerate(self._param_names):
            legacy = name.replace('W_head', 'W_pred').replace('b_head', 'b_pred')
            legacy2 = name.replace('W_head', 'W_recon').replace('b_head', 'b_recon')
            if name in state:
                self._params[i].data.copy_(state[name])
            elif legacy in state:
                self._params[i].data.copy_(state[legacy])
            elif legacy2 in state:
                self._params[i].data.copy_(state[legacy2])
        self._np_dirty = True
        self.load_rotation_state(state)
        if 'optimizer_state' in state:
            self._optimizer.load_state_dict(state['optimizer_state'])
            for opt_state in self._optimizer.state.values():
                for k, v in opt_state.items():
                    if torch.is_tensor(v):
                        opt_state[k] = v.to(self._device)


# Thin wrappers for backward compatibility
class PredictiveColumn(TransformerColumn):
    """Next-frame prediction through category bottleneck."""
    def __init__(self, *args, **kwargs):
        kwargs.setdefault('loss_mode', 'predictive')
        super().__init__(*args, **kwargs)


class ReconColumn(TransformerColumn):
    """Spatial reconstruction through category bottleneck."""
    def __init__(self, *args, **kwargs):
        kwargs.setdefault('loss_mode', 'recon')
        super().__init__(*args, **kwargs)


# ---------------------------------------------------------------------------
# ConsciencePredictiveColumn — hybrid: conscience state + predictive validation
# ---------------------------------------------------------------------------

class ConsciencePredictiveColumn(TransformerColumn):
    """Hybrid column: conscience head defines state, predictive head validates it.

    State head:
        - transformer encoder -> state descriptor
        - cosine similarity to learned prototypes (cat_embs)
        - conscience/tiredness rotation from ColumnBase
        - output probabilities = current-state belief

    Predictive head:
        - one predictor per category
        - next-frame prediction = soft mixture weighted by p_state
        - per-category prediction errors produce a corrective target q that
          nudges p_state toward states that both match now and predict well

    Extra anchor:
        - shared reconstruction head (W_head / b_head from TransformerColumn)
          reconstructs current frame from the categorical bottleneck so states
          stay interpretable, not purely predictive.
    """

    def __init__(self, m, n_outputs=4, max_inputs=20, window=4,
                 lr=1e-3, temperature=1.5, n_heads=2,
                 lambda_balance=0.1, lambda_ortho=0.01,
                 lambda_now=0.25, lambda_nudge=0.10,
                 state_input_scale=0.5, validation_beta=4.0,
                 proto_lr=0.05, reseed_after=1000,
                 usage_decay=0.99, **kwargs):
        kwargs.setdefault('loss_mode', 'predictive')
        super().__init__(m, n_outputs, max_inputs, window,
                         lr=lr, temperature=temperature, n_heads=n_heads,
                         lambda_sharp=0.0,
                         lambda_balance=lambda_balance,
                         lambda_ortho=lambda_ortho,
                         **kwargs)

        self.lambda_now = float(lambda_now)
        self.lambda_nudge = float(lambda_nudge)
        self.state_input_scale = float(state_input_scale)
        self.validation_beta = float(validation_beta)
        self.proto_lr = float(proto_lr)
        self.reseed_after = int(reseed_after)
        self.usage_decay = float(usage_decay)

        # Per-category transition models: state_k -> predicted next frame
        d = self.d_model
        self.W_pred_bank = torch.nn.Parameter(
            torch.randn(m, n_outputs, d, d) * (d ** -0.5))
        self.b_pred_bank = torch.nn.Parameter(torch.zeros(m, n_outputs, d))
        self.last_won = np.zeros((m, n_outputs), dtype=np.int64)
        self.usage = np.full((m, n_outputs), 1.0 / n_outputs, dtype=np.float32)

        self._params.extend([self.W_pred_bank, self.b_pred_bank])
        self._param_names.extend(['W_pred_bank', 'b_pred_bank'])
        if self._device.type == 'cuda':
            self.W_pred_bank.data = self.W_pred_bank.data.to(self._device)
            self.b_pred_bank.data = self.b_pred_bank.data.to(self._device)
        self._optimizer = torch.optim.Adam(self._params, lr=lr)

    def _normalize_t(self, x):
        return x / x.norm(dim=-1, keepdim=True).clamp(min=1e-8)

    def _present_descriptor_torch(self, x_gpu):
        x_now = x_gpu.mean(dim=1)
        x_now = x_now - x_now.mean(dim=1, keepdim=True)
        return self._normalize_t(x_now)

    def _apply_rotation_torch(self, sim):
        out = sim
        if self.alpha != 0.0:
            theta = torch.from_numpy(self.theta).to(sim.device)
            out = out - (theta.unsqueeze(1) if sim.dim() == 3 else theta)
        if self.tiredness_rate > 0.0:
            tired = torch.from_numpy(self.output_tiredness).to(sim.device)
            sim_range = out.max(dim=-1, keepdim=True).values - out.min(dim=-1, keepdim=True).values
            out = out - (tired.unsqueeze(1) if sim.dim() == 3 else tired) * sim_range.clamp(min=1e-8)
        return out

    def _state_from_hidden_torch(self, x_hidden, x_gpu):
        # Training states for positions 0..T-2
        x_now_all = x_gpu[:, :-1, :]
        x_now_all = x_now_all - x_now_all.mean(dim=-1, keepdim=True)
        x_now_all = self._normalize_t(x_now_all)
        z_all = self._normalize_t(x_hidden[:, :-1, :])
        state_all = self._normalize_t(z_all + self.state_input_scale * x_now_all)

        # External output state from full current window
        x_now_last = self._present_descriptor_torch(x_gpu)
        z_last = self._normalize_t(x_hidden[:, -1, :])
        state_last = self._normalize_t(z_last + self.state_input_scale * x_now_last)
        return state_all, state_last

    def _categorize_state_torch(self, state):
        c_n = self._normalize_t(self.cat_embs)
        if state.dim() == 2:
            sim = torch.bmm(state.unsqueeze(1), c_n.transpose(1, 2)).squeeze(1)
        else:
            sim = torch.bmm(state, c_n.transpose(1, 2))
        scores = self._apply_rotation_torch(sim)
        p_state = F.softmax(scores / self.temperature, dim=-1)
        return sim, scores, p_state, c_n

    def _compute_hybrid_loss(self, x_hidden, x_gpu, dev):
        valid_gpu = torch.from_numpy((self.slot_map >= 0).astype(np.float32)).to(dev)
        n_valid_gpu = valid_gpu.sum(dim=1).clamp(min=1.0)
        targets_next = x_gpu[:, 1:, :]
        targets_now = x_gpu[:, :-1, :]

        state_all, _ = self._state_from_hidden_torch(x_hidden, x_gpu)
        sim_all, scores_all, p_state_all, c_n = self._categorize_state_torch(state_all)

        # Per-category prediction of next frame
        pred_per_cat = torch.einsum('mtd,mkdf->mtkf', state_all, self.W_pred_bank)
        pred_per_cat = pred_per_cat + self.b_pred_bank[:, None, :, :]
        pred_mix = (p_state_all.unsqueeze(-1) * pred_per_cat).sum(dim=2)
        diff_next = (pred_mix - targets_next) ** 2
        L_pred = (diff_next * valid_gpu.unsqueeze(1)).sum(dim=2).div(n_valid_gpu.unsqueeze(1)).mean()

        # Current-frame reconstruction anchor
        z_q_all = torch.bmm(p_state_all, c_n)
        recon_now = torch.bmm(z_q_all, self.W_head) + self.b_head.unsqueeze(1)
        diff_now = (recon_now - targets_now) ** 2
        L_now = (diff_now * valid_gpu.unsqueeze(1)).sum(dim=2).div(n_valid_gpu.unsqueeze(1)).mean()

        # Predictive validation nudge
        err_per_cat = (pred_per_cat - targets_next.unsqueeze(2)) ** 2
        err_per_cat = (err_per_cat * valid_gpu[:, None, None, :]).sum(dim=-1)
        err_per_cat = err_per_cat / n_valid_gpu[:, None, None]
        err_rel = err_per_cat - err_per_cat.min(dim=-1, keepdim=True).values
        q = F.softmax(torch.log(p_state_all + 1e-10) - self.validation_beta * err_rel.detach(), dim=-1)
        L_nudge = -(q * torch.log(p_state_all + 1e-10)).sum(dim=-1).mean()

        # EMA balance
        p_for_balance = p_state_all.mean(dim=1)
        usage_ema = torch.from_numpy(self._usage_ema).to(dev)
        beta = 0.9
        usage_est = beta * usage_ema + (1.0 - beta) * p_for_balance
        uniform = torch.full_like(usage_est, 1.0 / self.n_outputs)
        L_balance = ((usage_est - uniform) ** 2).sum(dim=-1).mean()
        self._usage_ema = (
            beta * self._usage_ema + (1.0 - beta) * p_for_balance.detach().cpu().numpy()
        ).astype(np.float32)

        # Orthogonality
        L_ortho = torch.tensor(0.0, device=dev)
        if self.lambda_ortho > 0:
            gram = torch.bmm(c_n, c_n.transpose(1, 2))
            eye = torch.eye(self.n_outputs, device=dev).unsqueeze(0)
            L_ortho = ((gram - eye) ** 2).mean()

        return (L_pred
                + self.lambda_now * L_now
                + self.lambda_nudge * L_nudge
                + self.lambda_balance * L_balance
                + self.lambda_ortho * L_ortho)

    def tick(self, signal_window, knn2=None):
        dev = self._device

        X = self._gather_input(signal_window)
        x_np = X.transpose(0, 2, 1).astype(np.float32)
        x_t = torch.from_numpy(x_np).to(dev)

        # Surprise from previous prediction
        if self._tick_count > 0:
            current_frame_np = x_np[:, -1, :]
            valid = (self.slot_map >= 0).astype(np.float32)
            n_valid = valid.sum(axis=1).clip(min=1.0)
            diff = (self._prev_prediction - current_frame_np) ** 2
            self._surprise = (diff * valid).sum(axis=1) / n_valid

        do_train = self._rng.rand() < self._learn_prob
        if do_train:
            x_hidden = self._encode_gpu(x_t)
            loss = self._compute_hybrid_loss(x_hidden, x_t, dev)
            self._optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self._params, 1.0)
            self._optimizer.step()
            self._np_dirty = True

        with torch.no_grad():
            x_hidden = self._encode_gpu(x_t)
            _, state_last = self._state_from_hidden_torch(x_hidden, x_t)
            sim_last, scores_last, p_state_last, c_n = self._categorize_state_torch(state_last)

            # Per-category prediction from current state
            pred_per_cat_last = torch.einsum('md,mkdf->mkf', state_last, self.W_pred_bank)
            pred_per_cat_last = pred_per_cat_last + self.b_pred_bank
            pred_last = (p_state_last.unsqueeze(-1) * pred_per_cat_last).sum(dim=1)

            # Hebbian prototype pull + dead-unit recovery
            winners = scores_last.argmax(dim=1).cpu().numpy()
            ar = torch.arange(self.m, device=dev)
            w_proto = self.cat_embs[ar, winners]
            new_proto = (1.0 - self.proto_lr) * w_proto + self.proto_lr * state_last
            self.cat_embs[ar, winners] = self._normalize_t(new_proto)

            self.last_won[self._arange_m, winners] = self._tick_count
            if self.reseed_after > 0:
                dead_ticks = self._tick_count - self.last_won
                dead_cols, dead_outs = np.where(dead_ticks > self.reseed_after)
                if len(dead_cols) > 0:
                    self.cat_embs[dead_cols, dead_outs] = state_last[dead_cols]
                    self.last_won[dead_cols, dead_outs] = self._tick_count

            # Output with rotation
            sim_np = sim_last.cpu().numpy().astype(np.float32)
            sim_np = self.apply_output_rotation(sim_np)
            sim_scaled = sim_np / self.temperature
            sim_scaled -= sim_scaled.max(axis=1, keepdims=True)
            e = np.exp(sim_scaled)
            p_out = (e / e.sum(axis=1, keepdims=True)).astype(np.float32)

        self.update_output_rotation(p_out.argmax(axis=1))
        y = np.zeros((self.m, self.n_outputs), dtype=np.float32)
        y[self._arange_m, winners] = 1.0
        self.usage = self.usage * self.usage_decay + y * (1.0 - self.usage_decay)

        self._outputs = self.apply_wta(p_out)
        self._prev_prediction = pred_last.cpu().numpy().astype(np.float32)
        self._tick_count += 1

    def save(self, output_dir):
        np.save(os.path.join(output_dir, "column_slot_map.npy"), self.slot_map)
        state = {
            'type': 'conscience_predictive',
            'tick_count': self._tick_count,
            'm': self.m, 'n_outputs': self.n_outputs,
            'max_inputs': self.max_inputs, 'window': self.window,
            'lr': self.lr, 'temperature': self.temperature,
            'n_heads': self.n_heads, 'lambda_balance': self.lambda_balance,
            'lambda_ortho': self.lambda_ortho, 'lambda_now': self.lambda_now,
            'lambda_nudge': self.lambda_nudge,
            'state_input_scale': self.state_input_scale,
            'validation_beta': self.validation_beta,
            'proto_lr': self.proto_lr, 'reseed_after': self.reseed_after,
            'usage_decay': self.usage_decay,
            'd_model': self.d_model, 'd_ff': self.d_ff,
            '_reserved_mask': torch.from_numpy(self._reserved_mask.astype(np.uint8)),
            '_prev_prediction': torch.from_numpy(self._prev_prediction),
            '_usage_ema': torch.from_numpy(self._usage_ema),
            'usage': torch.from_numpy(self.usage),
            'last_won': torch.from_numpy(self.last_won.astype(np.float32)),
        }
        state.update(self.save_rotation_state())
        for i, name in enumerate(self._param_names):
            state[name] = self._params[i].detach().cpu()
        state['optimizer_state'] = self._optimizer.state_dict()
        torch.save(state, os.path.join(output_dir, "column_states.pt"))
        print(f"  conscience_predictive column state saved to {output_dir}")

    def load_state(self, state, slot_map):
        self.slot_map = slot_map
        if state.get('_reserved_mask') is not None:
            t = state['_reserved_mask']
            arr = t.numpy() if hasattr(t, 'numpy') else np.array(t)
            self._reserved_mask = arr.astype(bool)
        if 'tick_count' in state:
            self._tick_count = int(state['tick_count'])
        if '_prev_prediction' in state:
            t = state['_prev_prediction']
            self._prev_prediction = t.numpy().astype(np.float32) if hasattr(t, 'numpy') else np.array(t, dtype=np.float32)
        if '_usage_ema' in state:
            t = state['_usage_ema']
            self._usage_ema = t.numpy().astype(np.float32) if hasattr(t, 'numpy') else np.array(t, dtype=np.float32)
        if 'usage' in state:
            t = state['usage']
            self.usage = t.numpy().astype(np.float32) if hasattr(t, 'numpy') else np.array(t, dtype=np.float32)
        if 'last_won' in state:
            t = state['last_won']
            arr = t.numpy() if hasattr(t, 'numpy') else np.array(t)
            self.last_won = arr.astype(np.int64)
        for i, name in enumerate(self._param_names):
            if name in state:
                self._params[i].data.copy_(state[name])
        self._np_dirty = True
        self.load_rotation_state(state)
        if 'optimizer_state' in state:
            self._optimizer.load_state_dict(state['optimizer_state'])
            for opt_state in self._optimizer.state.values():
                for k, v in opt_state.items():
                    if torch.is_tensor(v):
                        opt_state[k] = v.to(self._device)

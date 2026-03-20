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
                 streaming_decay=0.5, lateral=False):
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

        # Batched state: (m, n_outputs, max_inputs) prototypes on unit sphere
        self.prototypes = F.normalize(
            torch.randn(m, n_outputs, max_inputs), dim=2)
        # (m, n_outputs) usage counters
        self.usage = torch.full((m, n_outputs), 1.0 / n_outputs)
        # (m, n_outputs) streaming EMA state
        self.proj_mean = torch.zeros(m, n_outputs)
        self.proj_var = torch.zeros(m, n_outputs)

        self.slot_map = np.full((m, max_inputs), -1, dtype=np.int64)
        self._outputs = np.zeros((m, n_outputs), dtype=np.float32)
        self._arange_m = torch.arange(m)

        # Lateral connections: each column receives all other columns' outputs
        self.lateral_sparsity = 1.0  # fraction of connections to KEEP
        if lateral:
            lateral_dim = m * n_outputs
            self.lateral_protos = F.normalize(
                torch.randn(m, n_outputs, lateral_dim), dim=2)
            self._prev_outputs = torch.zeros(m * n_outputs)
            # Sparse mask: set via set_lateral_sparsity() after init
            self._lateral_mask = None  # None = full connectivity

    def wire(self, cluster_id, neuron_id):
        """Wire a neuron to a cluster's column (lowest empty slot)."""
        row = self.slot_map[cluster_id]
        empty = np.where(row == -1)[0]
        if len(empty) == 0:
            return  # cluster full
        row[empty[0]] = neuron_id

    def set_lateral_sparsity(self, keep_fraction, seed=42):
        """Randomly prune lateral connections. keep_fraction=1.0 is full."""
        if not self.lateral:
            return
        self.lateral_sparsity = keep_fraction
        if keep_fraction >= 1.0:
            self._lateral_mask = None
            return
        rng = torch.manual_seed(seed)
        lateral_dim = self.m * self.n_outputs
        mask = (torch.rand(self.m, lateral_dim) < keep_fraction).float()
        self._lateral_mask = mask
        n_kept = int(mask.sum().item())
        n_total = self.m * lateral_dim
        print(f"  Lateral sparsity: keeping {n_kept}/{n_total} "
              f"({keep_fraction*100:.0f}%) connections")

    def unwire(self, cluster_id, neuron_id):
        """Unwire a neuron from a cluster's column."""
        row = self.slot_map[cluster_id]
        matches = np.where(row == neuron_id)[0]
        if len(matches) > 0:
            row[matches[0]] = -1

    def tick(self, signal_window):
        """Batched forward + learn for all M columns in one pass.

        Args:
            signal_window: (n, window) signal trace — column 0 is most recent.
        """
        m, n_out, n_in = self.m, self.n_outputs, self.max_inputs
        w = self.window
        ar = self._arange_m

        # --- Build batched input (m, max_inputs, window) via gather ---
        sm = torch.from_numpy(self.slot_map)        # (m, max_inputs)
        valid = sm >= 0                              # (m, max_inputs)
        safe_sm = sm.clamp(min=0)                    # safe for indexing
        sig = torch.from_numpy(signal_window)        # (n, window)
        X = sig[safe_sm]                             # (m, max_inputs, window)
        X[~valid] = 0.0

        # --- Batched forward: streaming variance ---
        # proj: (m, n_outputs, window) = prototypes @ X
        proj = torch.bmm(self.prototypes, X)
        proj_c = proj - proj.mean(dim=2, keepdim=True)
        sim = (proj_c ** 2).mean(dim=2)              # (m, n_outputs)

        # EMA update
        d = self.streaming_decay
        proj_mean_batch = proj.mean(dim=2)
        self.proj_mean = d * self.proj_mean + (1 - d) * proj_mean_batch
        self.proj_var = d * self.proj_var + (1 - d) * sim

        # Lateral input: add similarity from other columns' previous outputs
        if self.lateral:
            lat_input = self._prev_outputs.unsqueeze(0).expand(m, -1)
            if self._lateral_mask is not None:
                lat_input = lat_input * self._lateral_mask
            lat_sim = torch.bmm(
                self.lateral_protos,
                lat_input.unsqueeze(2)
            ).squeeze(2)  # (m, n_outputs)
            sim = sim + lat_sim

        # Softmax probabilities
        probs = F.softmax(sim / self.temperature, dim=1)  # (m, n_outputs)

        # --- Batched update ---
        original_winners = probs.argmax(dim=1)       # (m,)

        # Match quality: fraction of total variance from winner prototype
        total_var = self.proj_var.sum(dim=1).clamp(min=1e-8)
        match_q = self.proj_var[ar, original_winners] / total_var

        # Dormant reassignment
        dormant = self.usage.argmin(dim=1)
        needs_reassign = (match_q < self.match_threshold) & (dormant != original_winners)
        actual_winners = torch.where(needs_reassign, dormant, original_winners)

        # Per-column learning rate
        usage_at_orig = self.usage[ar, original_winners]
        lr_normal = self.lr / (1.0 + n_out * usage_at_orig)
        lr_eff = torch.where(needs_reassign, self.lr, lr_normal)  # (m,)

        # Entropy-scaled lr: uniform columns learn fast, differentiated slow
        if ENTROPY_SCALED_LR:
            H = -(probs * torch.log(probs + 1e-10)).sum(dim=1)  # (m,)
            H_max = np.log(n_out)
            lr_eff = lr_eff * (H / H_max)

        # Power iteration targets
        X_c = X - X.mean(dim=2, keepdim=True)       # (m, n_in, w)
        w_proto = self.prototypes[ar, actual_winners] # (m, n_in)
        inner = torch.einsum('miw,mi->mw', X_c, w_proto)  # (m, w)
        target = torch.einsum('miw,mw->mi', X_c, inner)   # (m, n_in)
        target = target / max(w - 1, 1)
        target_norm = target.norm(dim=1, keepdim=True)
        target = target / target_norm.clamp(min=1e-8)

        # Update winner prototypes where target is nonzero
        do_update = (target_norm.squeeze(1) > 1e-8)
        new_proto = w_proto + lr_eff.unsqueeze(1) * (target - w_proto)
        new_proto = F.normalize(new_proto, dim=1)
        idx = ar[do_update]
        self.prototypes[idx, actual_winners[do_update]] = new_proto[do_update]

        # Lateral weight update: contrastive — winner pulls, losers push
        if self.lateral:
            lat_input = self._prev_outputs.unsqueeze(0).expand(m, -1)  # (m, lat_dim)
            # Build per-output sign: +1 for winner, -1/(n_out-1) for losers
            # This makes each output specialize on different lateral patterns
            winner_mask = torch.zeros(m, n_out)
            winner_mask[ar, actual_winners] = 1.0
            sign = torch.where(winner_mask > 0,
                               torch.ones(m, n_out),
                               torch.full((m, n_out), -1.0 / (n_out - 1)))
            # Update all outputs simultaneously
            delta = lat_input.unsqueeze(1) - self.lateral_protos  # (m, n_out, lat_dim)
            lr_3d = lr_eff.unsqueeze(1).unsqueeze(2)  # (m, 1, 1)
            sign_3d = sign.unsqueeze(2)                # (m, n_out, 1)
            new_lat = self.lateral_protos + lr_3d * sign_3d * delta
            self.lateral_protos = F.normalize(new_lat, dim=2)
            # Store current outputs for next tick
            self._prev_outputs = probs.detach().flatten()

        # Usage EMA
        usage_target = torch.zeros(m, n_out)
        usage_target[ar, actual_winners] = 1.0
        self.usage = self.usage * self.usage_decay + usage_target * (1 - self.usage_decay)

        # Store outputs
        self._outputs = probs.detach().numpy()

    def get_outputs(self):
        """Return (m, n_outputs) array of all column outputs."""
        return self._outputs.copy()

    def save(self, output_dir):
        """Save slot_map + batched column state."""
        np.save(os.path.join(output_dir, "column_slot_map.npy"), self.slot_map)
        torch.save({
            'prototypes': self.prototypes,
            'usage': self.usage,
            'proj_mean': self.proj_mean,
            'proj_var': self.proj_var,
            'm': self.m,
            'n_outputs': self.n_outputs,
            'max_inputs': self.max_inputs,
            'window': self.window,
            'temperature': self.temperature,
            'lr': self.lr,
            'match_threshold': self.match_threshold,
            'streaming_decay': self.streaming_decay,
            'lateral': self.lateral,
            'lateral_protos': self.lateral_protos if self.lateral else None,
        }, os.path.join(output_dir, "column_states.pt"))
        print(f"  column state saved to {output_dir}")

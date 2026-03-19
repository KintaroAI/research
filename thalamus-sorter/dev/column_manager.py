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


# ---------------------------------------------------------------------------
# SoftWTACell — from column/dev/column.py, instantaneous + streaming modes
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

    def __init__(self, n_inputs, n_outputs, temperature=0.5, lr=0.05,
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
# ColumnManager
# ---------------------------------------------------------------------------

class ColumnManager:
    """Manages M SoftWTACell columns, one per cluster.

    Wiring map: slot_map[cluster_c, slot_s] = neuron_id (-1 = empty).
    Pre-allocates max_inputs per column (fixed size, no resizing).

    Each tick receives a sliding window of signal frames (n, window) from the
    circular signal buffer. Column 0 = most recent frame.
    """

    def __init__(self, m, n_outputs=4, max_inputs=20, window=4,
                 temperature=0.5, lr=0.05, match_threshold=0.1,
                 streaming_decay=0.5):
        self.m = m
        self.n_outputs = n_outputs
        self.max_inputs = max_inputs
        self.window = window
        self.columns = [SoftWTACell(max_inputs, n_outputs,
                                    temperature=temperature, lr=lr,
                                    match_threshold=match_threshold,
                                    temporal_mode='streaming',
                                    streaming_decay=streaming_decay)
                        for _ in range(m)]
        self.slot_map = np.full((m, max_inputs), -1, dtype=np.int64)
        self._outputs = np.zeros((m, n_outputs), dtype=np.float32)

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

    def tick(self, signal_window):
        """Forward + learn for all columns.

        Args:
            signal_window: (n, window) signal trace — column 0 is most recent.
        """
        for c in range(self.m):
            row = self.slot_map[c]
            valid = row >= 0
            if not valid.any():
                self._outputs[c] = 0.0
                continue
            # Build (max_inputs, window) input matrix
            inp = np.zeros((self.max_inputs, self.window), dtype=np.float32)
            inp[valid] = signal_window[row[valid]]
            x = torch.from_numpy(inp)
            probs = self.columns[c].forward(x)
            self.columns[c].update(x, probs)
            self._outputs[c] = probs.detach().numpy()

    def get_outputs(self):
        """Return (m, n_outputs) array of all column outputs."""
        return self._outputs.copy()

    def save(self, output_dir):
        """Save slot_map + all column state_dicts."""
        np.save(os.path.join(output_dir, "column_slot_map.npy"), self.slot_map)
        state_dicts = [col.state_dict() for col in self.columns]
        torch.save(state_dicts, os.path.join(output_dir, "column_states.pt"))
        print(f"  column state saved to {output_dir}")

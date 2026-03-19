"""Column wiring: thalamus-to-cortex connection via SoftWTACell columns.

Each cluster gets its own SoftWTACell column. Neurons wire/unwire to
their cluster's column as they enter/leave clusters via the ring buffer.
The column's input is the raw signal (grayscale pixel intensity) of each
wired neuron. Grid size, cluster count, and output count are all
configurable.

Architecture:
    saccade crop -> N neurons -> M clusters
        each cluster -> 1 SoftWTACell(n_inputs=max_inputs, n_outputs=configurable)
            input = raw signal of wired neurons (empty slots = 0)
            output = soft-WTA probabilities
"""

import os
import numpy as np
import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# SoftWTACell — copied from column/dev/column.py
# ---------------------------------------------------------------------------

class SoftWTACell:
    """Soft winner-take-all competitive categorization cell.

    Each output unit holds a prototype vector. Inputs are compared to prototypes
    via dot-product similarity, passed through softmax with temperature control.
    The winning unit's prototype moves toward the input (Hebbian pull).
    Usage counters gate plasticity to prevent collapse.
    """

    def __init__(self, n_inputs, n_outputs, temperature=0.5, lr=0.05,
                 match_threshold=0.5, usage_decay=0.99):
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.temperature = temperature
        self.lr = lr
        self.match_threshold = match_threshold
        self.usage_decay = usage_decay

        # Prototype vectors (m x n), initialized on unit sphere
        self.prototypes = F.normalize(torch.randn(n_outputs, n_inputs), dim=1)

        # Usage counters — EMA of win frequency, starts uniform
        self.usage = torch.full((n_outputs,), 1.0 / n_outputs)

    def forward(self, x):
        """Compute output probabilities for input x. x: (n,)"""
        x_norm = F.normalize(x.unsqueeze(0), dim=1)
        sim = (x_norm @ self.prototypes.T).squeeze(0)  # (m,)
        return F.softmax(sim / self.temperature, dim=0)

    def update(self, x, probs=None):
        """Online Hebbian update for a single input x: (n,)"""
        if probs is None:
            probs = self.forward(x)

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

        # Update usage counters
        target = torch.zeros(self.n_outputs)
        target[winner] = 1.0
        self.usage = self.usage * self.usage_decay + target * (1 - self.usage_decay)

        return winner, match_quality

    def state_dict(self):
        return {
            'prototypes': self.prototypes.clone(),
            'usage': self.usage.clone(),
            'n_inputs': self.n_inputs,
            'n_outputs': self.n_outputs,
            'temperature': self.temperature,
            'lr': self.lr,
            'match_threshold': self.match_threshold,
            'usage_decay': self.usage_decay,
        }

    @classmethod
    def from_state_dict(cls, state):
        cell = cls(state['n_inputs'], state['n_outputs'],
                   temperature=state['temperature'], lr=state['lr'],
                   match_threshold=state['match_threshold'],
                   usage_decay=state['usage_decay'])
        cell.prototypes = state['prototypes']
        cell.usage = state['usage']
        return cell


# ---------------------------------------------------------------------------
# ColumnManager
# ---------------------------------------------------------------------------

class ColumnManager:
    """Manages M SoftWTACell columns, one per cluster.

    Wiring map: slot_map[cluster_c, slot_s] = neuron_id (-1 = empty).
    Pre-allocates max_inputs per column (fixed size, no resizing).
    """

    def __init__(self, m, n_outputs=4, max_inputs=20, temperature=0.5, lr=0.05):
        self.m = m
        self.n_outputs = n_outputs
        self.max_inputs = max_inputs
        self.columns = [SoftWTACell(max_inputs, n_outputs,
                                    temperature=temperature, lr=lr)
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

    def tick(self, signal):
        """Forward + learn for all columns.

        Args:
            signal: (n,) current signal frame (raw pixel intensities).
        """
        for c in range(self.m):
            row = self.slot_map[c]
            valid = row >= 0
            if not valid.any():
                self._outputs[c] = 0.0
                continue
            inp = np.zeros(self.max_inputs, dtype=np.float32)
            inp[valid] = signal[row[valid]]
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

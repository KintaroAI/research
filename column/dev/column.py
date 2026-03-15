"""Soft winner-take-all competitive categorization cell.

Each output unit holds a prototype vector. Inputs are compared to prototypes
via dot-product similarity, passed through softmax with temperature control.
The winning unit's prototype moves toward the input (Hebbian pull).
Usage counters gate plasticity to prevent collapse.

Supports optional temporal context: input can be (n, T) matrix. In correlation
mode, similarity is based on covariance structure rather than instantaneous values.
"""

import torch
import torch.nn.functional as F


class SoftWTACell:

    def __init__(self, n_inputs, n_outputs, temperature=0.5, lr=0.05,
                 match_threshold=0.5, usage_decay=0.99, temporal_mode=None):
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.temperature = temperature
        self.lr = lr
        self.match_threshold = match_threshold
        self.usage_decay = usage_decay
        self.temporal_mode = temporal_mode  # None or 'correlation'

        # Prototype vectors (m x n), initialized on unit sphere
        self.prototypes = F.normalize(torch.randn(n_outputs, n_inputs), dim=1)

        # Usage counters — EMA of win frequency, starts uniform
        self.usage = torch.full((n_outputs,), 1.0 / n_outputs)

    def _compute_similarity(self, x):
        """Compute per-prototype similarity scores.

        Args:
            x: (n,) for instantaneous, (n, T) for temporal

        Returns:
            similarity: (m,) scores
        """
        if self.temporal_mode is None:
            x_norm = F.normalize(x.unsqueeze(0) if x.dim() == 1 else x, dim=-1)
            sim = x_norm @ self.prototypes.T
            if x.dim() == 1:
                sim = sim.squeeze(0)
            return sim

        # correlation mode: x is (n, T)
        x_c = x - x.mean(dim=1, keepdim=True)
        T = x.shape[1]
        C = x_c @ x_c.T / max(T - 1, 1)  # (n, n) covariance
        # Per-prototype variance: sim_i = proto_i @ C @ proto_i
        Cp = self.prototypes @ C  # (m, n)
        sim = (Cp * self.prototypes).sum(dim=1)  # (m,)
        return sim

    def forward(self, x):
        """Compute output probabilities for input x.

        Args:
            x: (n,) or (batch, n) for instantaneous mode
               (n, T) for correlation mode

        Returns:
            probabilities: (m,) or (batch, m)
        """
        sim = self._compute_similarity(x)
        if sim.dim() == 1:
            return F.softmax(sim / self.temperature, dim=0)
        return F.softmax(sim / self.temperature, dim=1)

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

    def _update_correlation(self, x, probs):
        """Update for temporal input (n, T) using covariance structure."""
        x_c = x - x.mean(dim=1, keepdim=True)
        T = x.shape[1]
        C = x_c @ x_c.T / max(T - 1, 1)

        winner = probs.argmax().item()

        # Match quality: fraction of total variance explained by winner
        pCp = (self.prototypes[winner] @ C @ self.prototypes[winner]).item()
        trC = C.trace().item()
        match_quality = pCp / max(trC, 1e-8)

        effective_lr = self.lr / (1.0 + self.n_outputs * self.usage[winner])

        if match_quality < self.match_threshold:
            dormant = self.usage.argmin().item()
            if dormant != winner:
                winner = dormant
                effective_lr = self.lr

        # Power iteration: rotate prototype toward dominant eigenvector of C
        target = C @ self.prototypes[winner]
        target_norm = target.norm()
        if target_norm > 1e-8:
            target = target / target_norm
            self.prototypes[winner] += effective_lr * (target - self.prototypes[winner])
            self.prototypes[winner] = F.normalize(self.prototypes[winner], dim=0)

        return winner, match_quality

    def update(self, x, probs=None):
        """Online Hebbian update for a single input.

        Args:
            x: (n,) for instantaneous, (n, T) for correlation mode
            probs: precomputed probabilities (optional)

        Returns:
            winner: index of winning unit
            match_quality: cosine similarity (instantaneous) or fraction
                          of variance explained (correlation)
        """
        if probs is None:
            probs = self.forward(x)

        if self.temporal_mode is None:
            winner, match_quality = self._update_instantaneous(x, probs)
        else:
            winner, match_quality = self._update_correlation(x, probs)

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
            'temporal_mode': self.temporal_mode,
        }

    @classmethod
    def from_state_dict(cls, state):
        cell = cls(state['n_inputs'], state['n_outputs'],
                   temperature=state['temperature'], lr=state['lr'],
                   match_threshold=state['match_threshold'],
                   usage_decay=state['usage_decay'],
                   temporal_mode=state.get('temporal_mode'))
        cell.prototypes = state['prototypes']
        cell.usage = state['usage']
        return cell

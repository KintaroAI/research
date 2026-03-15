"""Soft winner-take-all competitive categorization cell.

Each output unit holds a prototype vector. Inputs are compared to prototypes
via dot-product similarity, passed through softmax with temperature control.
The winning unit's prototype moves toward the input (Hebbian pull).
Usage counters gate plasticity to prevent collapse.
"""

import torch
import torch.nn.functional as F


class SoftWTACell:

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
        """Compute output probabilities for input x.

        Args:
            x: input vector (n_inputs,) or batch (batch, n_inputs)

        Returns:
            probabilities: (n_outputs,) or (batch, n_outputs)
        """
        single = x.dim() == 1
        if single:
            x = x.unsqueeze(0)

        x_norm = F.normalize(x, dim=1)
        similarity = x_norm @ self.prototypes.T
        probs = F.softmax(similarity / self.temperature, dim=1)

        if single:
            probs = probs.squeeze(0)
        return probs

    def update(self, x, probs=None):
        """Online Hebbian update for a single input.

        Args:
            x: input vector (n_inputs,)
            probs: precomputed probabilities (optional)

        Returns:
            winner: index of winning unit
            match_quality: cosine similarity of winner to input
        """
        if probs is None:
            probs = self.forward(x)

        x_norm = F.normalize(x.unsqueeze(0), dim=1).squeeze(0)

        winner = probs.argmax().item()
        match_quality = (x_norm * self.prototypes[winner]).sum().item()

        # Usage-gated learning rate: frequent winners learn slower
        effective_lr = self.lr / (1.0 + self.n_outputs * self.usage[winner])

        # Poor match — recruit least-used unit instead
        if match_quality < self.match_threshold:
            dormant = self.usage.argmin().item()
            if dormant != winner:
                winner = dormant
                effective_lr = self.lr  # full LR for recruitment

        # Hebbian pull: move winner's prototype toward input
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

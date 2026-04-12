"""GPU-accelerated TemporalPrototypeColumn.

Drop-in replacement with hot path on GPU (descriptor, similarity,
postprocess, prototype update, fatigue, homeostasis). Saves ~13× at m=2000.

CPU-only operations: _gather_input (irregular slot_map indexing),
decorrelation (infrequent, has loops), loser repulsion (rare), reseeding.
"""

import os
import numpy as np
import torch
import torch.nn.functional as F

from temporal_prototype_column import TemporalPrototypeColumn


class TemporalPrototypeColumnGPU(TemporalPrototypeColumn):
    """GPU-accelerated version. Inherits all logic, overrides hot path."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Move heavy state to GPU
        self._protos_t = torch.from_numpy(self.prototypes).to(self._dev)
        self._theta_t = torch.from_numpy(self.theta).to(self._dev)
        self._fatigue_t = torch.from_numpy(self.fast_fatigue).to(self._dev)
        self._usage_t = torch.from_numpy(self.usage).to(self._dev)

        # Precompute constants
        self._target_usage_t = torch.tensor(self.target_usage, device=self._dev)
        self._arange_m_t = torch.arange(self.m, device=self._dev)
        self._pending_pred_t = None  # GPU prediction buffer

        # Weights for descriptor parts
        self._cw = float(self.current_weight)
        self._mw = float(self.mean_weight)
        self._dw = float(self.delta_weight)

    # ------------------------------------------------------------------
    # GPU descriptor
    # ------------------------------------------------------------------

    def _state_descriptor_gpu(self, X_t):
        """X_t: (m, n_in, T) on GPU. Returns (m, d_model) on GPU."""
        T = X_t.shape[2]
        current = X_t[:, :, -1]

        if self.multi_scale:
            delta_1 = X_t[:, :, -1] - X_t[:, :, -2] if T > 1 else torch.zeros_like(current)
            mean_full = X_t.mean(dim=2)
            half = max(2, T // 2)
            delta_half = X_t[:, :, -1] - X_t[:, :, -half] if T > half else torch.zeros_like(current)
            parts = [
                self._cw * self._part_norm_t(current),
                self._mw * self._part_norm_t(mean_full),
                self._dw * self._part_norm_t(delta_1),
                self._dw * self._part_norm_t(delta_half),
            ]
        else:
            mean = X_t.mean(dim=2)
            delta = X_t[:, :, -1] - X_t[:, :, -2] if T > 1 else torch.zeros_like(current)
            parts = [
                self._cw * self._part_norm_t(current),
                self._mw * self._part_norm_t(mean),
                self._dw * self._part_norm_t(delta),
            ]

        desc = torch.cat(parts, dim=1)
        return F.normalize(desc, dim=1)

    @staticmethod
    def _part_norm_t(x):
        x = x - x.mean(dim=1, keepdim=True)
        return F.normalize(x, dim=1)

    # ------------------------------------------------------------------
    # GPU similarity
    # ------------------------------------------------------------------

    def _raw_similarity_gpu(self, desc_t):
        """desc_t: (m, d_model). Returns (m, n_out) on GPU."""
        protos_n = F.normalize(self._protos_t, dim=2)
        return torch.bmm(protos_n, desc_t.unsqueeze(2)).squeeze(2)

    # ------------------------------------------------------------------
    # GPU postprocess
    # ------------------------------------------------------------------

    def _postprocess_probs_gpu(self, scores_t):
        """scores_t: (m, n_out) on GPU. Returns (m, n_out) on GPU."""
        z = scores_t / max(self.temperature, 1e-8)

        if self.activation == 'entmax15':
            return self._entmax15_gpu(z)
        elif self.activation == 'sparsemax':
            return self._sparsemax_gpu(z)
        else:
            # softmax + topk
            p = F.softmax(z, dim=1)
            if 0 < self.k_active < self.n_outputs:
                _, topk_idx = z.topk(self.k_active, dim=1)
                mask = torch.zeros_like(p)
                mask.scatter_(1, topk_idx, 1.0)
                p = p * mask
                p_sum = p.sum(dim=1, keepdim=True).clamp(min=1e-8)
                p = p / p_sum
            return p

    @staticmethod
    def _entmax15_gpu(z):
        """Vectorized entmax-1.5 on GPU."""
        m, n = z.shape
        z_max = z.max(dim=1, keepdim=True).values
        lo = z_max - 1.0
        hi = z_max

        for _ in range(30):
            mid = (lo + hi) * 0.5
            vals = torch.clamp(z - mid, min=0.0)
            s = (vals ** 2).sum(dim=1, keepdim=True)
            too_low = s > 1.0
            lo = torch.where(too_low, mid, lo)
            hi = torch.where(too_low, hi, mid)

        tau = (lo + hi) * 0.5
        p = torch.clamp(z - tau, min=0.0)
        p_sum = p.sum(dim=1, keepdim=True)
        # Fallback for empty rows
        empty = p_sum.squeeze(1) <= 1e-8
        if empty.any():
            winners = z.argmax(dim=1)
            p[empty] = 0.0
            p[empty, winners[empty]] = 1.0
            p_sum = p.sum(dim=1, keepdim=True)
        return p / p_sum.clamp(min=1e-8)

    @staticmethod
    def _sparsemax_gpu(z):
        """Vectorized sparsemax on GPU."""
        sorted_z, _ = z.sort(dim=1, descending=True)
        cumsum = sorted_z.cumsum(dim=1)
        k_arr = torch.arange(1, z.shape[1] + 1, device=z.device, dtype=z.dtype)
        support = sorted_z > (cumsum - 1.0) / k_arr
        k = support.sum(dim=1).long()
        tau = (cumsum[torch.arange(z.shape[0], device=z.device), k - 1] - 1.0) / k.float()
        return torch.clamp(z - tau.unsqueeze(1), min=0.0)

    # ------------------------------------------------------------------
    # GPU rotation
    # ------------------------------------------------------------------

    def _apply_rotation_gpu(self, sim_t):
        return sim_t - self._theta_t - self.fatigue_strength * self._fatigue_t

    # ------------------------------------------------------------------
    # GPU prototype update
    # ------------------------------------------------------------------

    def _update_prototypes_gpu(self, desc_t, p_active_t, scores_t, sim_raw_t, surprise_np):
        # Decoupled learning
        if self.theta_learn_scale < 1.0:
            sim_learn = sim_raw_t - self.theta_learn_scale * self._theta_t
            p_learn = self._postprocess_probs_gpu(sim_learn)
        else:
            p_learn = p_active_t

        if not self.learn_from_probs:
            winners = p_learn.argmax(dim=1)
            learn = torch.zeros_like(p_learn)
            learn[self._arange_m_t, winners] = 1.0
        else:
            learn = p_learn

        # Usage-scaled LR
        target = max(self.target_usage, 1e-6)
        underuse = ((target - self._usage_t) / target).clamp(0.0, 1.0)
        lr_scale = self.min_lr_scale + (self.max_lr_scale - self.min_lr_scale) * underuse

        # Surprise modulation
        if surprise_np is not None and self.surprise_alpha > 0:
            surprise_t = torch.from_numpy(surprise_np).to(self._dev)
            lr_scale = lr_scale * (1.0 + self.surprise_alpha * surprise_t).unsqueeze(1)

        step = (self.proto_lr * lr_scale * learn).unsqueeze(2)  # (m, n_out, 1)
        self._protos_t += step * (desc_t.unsqueeze(1) - self._protos_t)

        # Track last_won (CPU — sparse, branchy)
        learn_np = learn.cpu().numpy()
        active = learn_np > 1e-4
        self.last_won[active] = self._tick_count

        # Renormalize
        self._protos_t = F.normalize(self._protos_t, dim=2)

    # ------------------------------------------------------------------
    # GPU fatigue + homeostasis
    # ------------------------------------------------------------------

    def _update_fatigue_gpu(self, p_active_t):
        self._fatigue_t = self.fatigue_decay * self._fatigue_t + self.fatigue_rate * p_active_t

    def _update_homeostasis_gpu(self, p_active_t):
        self._usage_t = self.usage_decay * self._usage_t + (1.0 - self.usage_decay) * p_active_t
        self._theta_t += self.homeostasis_rate * (self._usage_t - self._target_usage_t)
        self._theta_t.clamp_(-self.theta_clip, self.theta_clip)

    # ------------------------------------------------------------------
    # Main tick — GPU hot path
    # ------------------------------------------------------------------

    def tick(self, signal_window, knn2=None):
        self._tick_count += 1

        # Gather on CPU (irregular slot_map indexing), transfer once
        X_np = self._gather_input(signal_window).astype(np.float32)
        X_t = torch.from_numpy(X_np).to(self._dev, non_blocking=True)
        current_frame_t = X_t[:, :, -1]

        # Descriptor on GPU
        desc_t = self._state_descriptor_gpu(X_t)

        # Prediction check on GPU
        surprise = self._check_prediction_gpu(current_frame_t)

        # Similarity + rotation + postprocess on GPU
        sim_raw_t = self._raw_similarity_gpu(desc_t)
        scores_t = self._apply_rotation_gpu(sim_raw_t)
        p_active_t = self._postprocess_probs_gpu(scores_t)

        # Prototype update on GPU
        if self._rng.rand() < self._learn_prob:
            self._update_prototypes_gpu(desc_t, p_active_t, scores_t, sim_raw_t, surprise)
            self._update_predictor_gpu(current_frame_t)
            # Reseeding: only sync when actually needed (rare)
            if self.reseed_after > 0 and self._tick_count % 100 == 0:
                self.prototypes = self._protos_t.cpu().numpy()
                self._reseed_dead_units(desc_t.cpu().numpy())
                self._protos_t = torch.from_numpy(self.prototypes).to(self._dev)

        # Fatigue + homeostasis on GPU
        self._update_fatigue_gpu(p_active_t)
        self._update_homeostasis_gpu(p_active_t)

        # Predict next frame on GPU
        self._make_prediction_gpu(desc_t, p_active_t)

        # Single transfer: p_active for output + diagnostics
        p_active_np = p_active_t.cpu().numpy()
        self._update_corr_diagnostics(p_active_np)
        self._outputs = self.apply_wta(p_active_np)

    # ------------------------------------------------------------------
    # GPU prediction (avoids CPU round-trip)
    # ------------------------------------------------------------------

    def _check_prediction_gpu(self, current_frame_t):
        """Compare prediction to actual on GPU. Returns numpy surprise."""
        if self._pending_pred_t is None:
            return np.zeros(self.m, dtype=np.float32)
        err = ((self._pending_pred_t - current_frame_t) ** 2).mean(dim=1)
        err_np = err.cpu().numpy()
        self._surprise_raw = err_np
        self._surprise_ema = (self.surprise_beta * self._surprise_ema
                              + (1.0 - self.surprise_beta) * err_np)
        norm = np.clip(err_np / self._surprise_ema.clip(1e-8), 0.0, 5.0)
        self.last_pred_error = float(err_np.mean())
        return norm.astype(np.float32)

    def _make_prediction_gpu(self, desc_t, p_active_t):
        """Predict next frame entirely on GPU."""
        pred_input = torch.cat([desc_t, p_active_t], dim=1)
        self._pending_pred_t = torch.bmm(
            pred_input.unsqueeze(1), self._pred_W).squeeze(1) + self._pred_b
        self._last_pred_input_t = pred_input

    def _update_predictor_gpu(self, current_frame_t):
        """Train predictor on GPU."""
        if self._pending_pred_t is None or self._last_pred_input_t is None:
            return
        error = self._pending_pred_t - current_frame_t
        grad_W = torch.bmm(self._last_pred_input_t.unsqueeze(2), error.unsqueeze(1))
        self._pred_W -= self.pred_lr * grad_W
        self._pred_b -= self.pred_lr * error

    # ------------------------------------------------------------------
    # Sync helpers for save/load
    # ------------------------------------------------------------------

    def save(self, output_dir):
        # Ensure numpy state is current
        self.prototypes = self._protos_t.cpu().numpy()
        self.theta = self._theta_t.cpu().numpy()
        self.fast_fatigue = self._fatigue_t.cpu().numpy()
        self.usage = self._usage_t.cpu().numpy()
        super().save(output_dir)

    def load_state(self, state, slot_map):
        super().load_state(state, slot_map)
        # Re-sync to GPU
        self._protos_t = torch.from_numpy(self.prototypes).to(self._dev)
        self._theta_t = torch.from_numpy(self.theta).to(self._dev)
        self._fatigue_t = torch.from_numpy(self.fast_fatigue).to(self._dev)
        self._usage_t = torch.from_numpy(self.usage).to(self._dev)

import os
import numpy as np
import torch

from column_manager import ColumnBase


class TemporalPrototypeColumn(ColumnBase):
    """
    Temporal prototype column with:
      - streaming descriptor [current, mean, delta]
      - soft/top-k competition
      - fast fatigue (short-term anti-monopoly)
      - slow homeostatic thresholds (long-term anti-collapse)
      - winner Hebbian pull + targeted loser repulsion
      - rolling output correlation diagnostics (measure only)

    Builds on ConscienceHomeostaticFatigueColumn, adding:
      - Hunger-gated runner-up repulsion (only when representational, not regulatory)
      - Output correlation ring buffer for diagnostic logging

    Semantics:
      - self._outputs      : final emitted distribution
      - self.theta         : slow homeostatic threshold per output
      - self.fast_fatigue  : short-term activity penalty per output
    """

    def __init__(
        self,
        m,
        n_outputs=4,
        max_inputs=20,
        window=4,
        lr=0.05,
        proto_lr=None,
        temperature=0.45,
        usage_decay=0.995,
        current_weight=1.0,
        mean_weight=0.7,
        delta_weight=0.8,
        k_active=2,
        homeostasis_rate=0.02,
        target_usage=None,
        theta_clip=2.5,
        fatigue_decay=0.90,
        fatigue_rate=0.75,
        fatigue_strength=1.0,
        learn_from_probs=True,
        reseed_after=1000,
        min_lr_scale=0.75,
        max_lr_scale=1.75,
        dead_usage_threshold=0.02,
        sparse_floor=1e-8,
        # --- New: loser repulsion ---
        lr_neg=0.01,
        margin_band=0.3,
        # --- New: correlation diagnostics ---
        corr_window=None,  # defaults to `window` to match learning timescale
        # --- New: multi-scale descriptor ---
        multi_scale=False,
        # --- New: prediction-error-modulated learning ---
        pred_lr=0.01,
        surprise_alpha=0.5,
        surprise_beta=0.99,
        # --- New: decorrelation ---
        decorrelation=True,
        decor_lr=0.01,
        # --- New: activation mode ---
        activation='entmax15',  # 'topk', 'sparsemax', 'entmax15'
        # --- New: decouple learning from emission ---
        theta_learn_scale=0.0,  # 0=fully decoupled, 1=coupled (old behavior)
        **kwargs,
    ):
        wta_mode = kwargs.pop("wta_mode", "none")
        self._lateral_inputs = kwargs.pop("lateral_inputs", False)
        self._lateral_input_k = kwargs.pop("lateral_input_k", 4)
        if self._lateral_inputs:
            max_inputs = max_inputs + self._lateral_input_k * 2

        super().__init__(m, n_outputs, max_inputs, window, lr)
        self.wta_mode = wta_mode

        self.temperature = float(temperature)
        self.proto_lr = float(lr if proto_lr is None else proto_lr)
        self.usage_decay = float(usage_decay)

        self.current_weight = float(current_weight)
        self.mean_weight = float(mean_weight)
        self.delta_weight = float(delta_weight)

        self.k_active = int(k_active)
        self.homeostasis_rate = float(homeostasis_rate)
        self.target_usage = float(
            (1.0 / n_outputs) if target_usage is None else target_usage
        )
        self.theta_clip = float(theta_clip)

        self.fatigue_decay = float(fatigue_decay)
        self.fatigue_rate = float(fatigue_rate)
        self.fatigue_strength = float(fatigue_strength)

        self.learn_from_probs = bool(learn_from_probs)
        self.reseed_after = int(reseed_after)
        self.min_lr_scale = float(min_lr_scale)
        self.max_lr_scale = float(max_lr_scale)
        self.dead_usage_threshold = float(dead_usage_threshold)
        self.sparse_floor = float(sparse_floor)

        # Loser repulsion params
        self.lr_neg = float(lr_neg)
        self.margin_band = float(margin_band)

        # Correlation diagnostics — default to 2× learning window
        self.corr_window = int(window * 2 if corr_window is None else corr_window)
        self._corr_buf = np.zeros((self.corr_window, m, n_outputs), dtype=np.float32)
        self._corr_buf_idx = 0
        self._corr_buf_full = False
        self.last_raw_corr = None       # (m, n_out, n_out) or None
        self.last_centered_corr = None  # (m, n_out, n_out) or None

        # Multi-scale descriptor: [current, delta_1, mean_full, delta_half]
        self.multi_scale = bool(multi_scale)

        # Prediction-error-modulated learning
        self.pred_lr = float(pred_lr)
        self.surprise_alpha = float(surprise_alpha)
        self.surprise_beta = float(surprise_beta)

        # Decorrelation: push correlated output prototypes apart
        self.decorrelation = bool(decorrelation)
        self.decor_lr = float(decor_lr)

        # Activation mode
        self.activation = activation
        self.theta_learn_scale = float(theta_learn_scale)

        # External signals (set by benchmark tick_fn)
        self._hunger_level = 0.0  # informational only, not used for learning

        self.d_model = self.max_inputs * (4 if self.multi_scale else 3)

        rng = np.random.RandomState(42)
        protos = rng.randn(m, n_outputs, self.d_model).astype(np.float32)
        protos /= np.linalg.norm(protos, axis=2, keepdims=True).clip(1e-8)
        self.prototypes = protos

        self.usage = np.full((m, n_outputs), 1.0 / n_outputs, dtype=np.float32)
        self.theta = np.zeros((m, n_outputs), dtype=np.float32)
        self.fast_fatigue = np.zeros((m, n_outputs), dtype=np.float32)
        self.last_won = np.zeros((m, n_outputs), dtype=np.int64)

        # Predictor: linear map from (d_model + n_outputs) → max_inputs
        # Predicts next raw frame from current descriptor + current output
        # GPU-accelerated: weights + forward/backward on GPU, surprise on CPU
        pred_in = self.d_model + n_outputs
        pred_out = self.max_inputs
        self._pred_dim = pred_in
        self._pred_out = pred_out
        self._pred_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._pred_W = torch.randn(m, pred_in, pred_out, device=self._pred_device) * (0.01 / np.sqrt(pred_in))
        self._pred_b = torch.zeros(m, pred_out, device=self._pred_device)
        self._pending_pred_np = None        # (m, max_inputs) numpy or None
        self._last_pred_input_t = None      # (m, pred_in) torch tensor for gradient
        self._surprise_ema = np.ones(m, dtype=np.float32)
        self._surprise_raw = np.zeros(m, dtype=np.float32)
        self.last_pred_error = 0.0

        self._tick_count = 0
        self._learn_prob = 1.0
        self._rng = np.random.RandomState(123)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _normalize_np(self, x):
        return x / np.linalg.norm(x, axis=-1, keepdims=True).clip(1e-8)

    def _part_norm(self, x):
        x = x - x.mean(axis=1, keepdims=True)
        return self._normalize_np(x).astype(np.float32)

    def _state_descriptor_np(self, X):
        """
        X: (m, n_in, T)
        Returns: (m, d_model) where d_model = n_in * (4 if multi_scale else 3)
        """
        T = X.shape[2]
        current = X[:, :, -1]

        if self.multi_scale:
            # Multi-scale: current + mean + fast delta + slow delta
            delta_1 = X[:, :, -1] - X[:, :, -2] if T > 1 else np.zeros_like(current)
            mean_full = X.mean(axis=2)
            half = max(2, T // 2)
            delta_half = X[:, :, -1] - X[:, :, -half] if T > half else np.zeros_like(current)

            parts = [
                self.current_weight * self._part_norm(current),
                self.mean_weight * self._part_norm(mean_full),
                self.delta_weight * self._part_norm(delta_1),
                self.delta_weight * self._part_norm(delta_half),
            ]
        else:
            mean = X.mean(axis=2)
            delta = X[:, :, -1] - X[:, :, -2] if T > 1 else np.zeros_like(current)

            parts = [
                self.current_weight * self._part_norm(current),
                self.mean_weight * self._part_norm(mean),
                self.delta_weight * self._part_norm(delta),
            ]

        desc = np.concatenate(parts, axis=1)
        return self._normalize_np(desc).astype(np.float32)

    def _raw_similarity(self, state_np):
        protos = self.prototypes / np.linalg.norm(
            self.prototypes, axis=2, keepdims=True
        ).clip(1e-8)
        sim = np.einsum("moi,mi->mo", protos, state_np)
        return sim.astype(np.float32)

    def _topk_mask(self, scores):
        if self.k_active <= 0 or self.k_active >= self.n_outputs:
            return np.ones_like(scores, dtype=np.float32)
        idx = np.argpartition(scores, -self.k_active, axis=1)[:, -self.k_active :]
        mask = np.zeros_like(scores, dtype=np.float32)
        mask[self._arange_m[:, None], idx] = 1.0
        return mask

    @staticmethod
    def _entmax15(z):
        """α-entmax with α=1.5. Batched: z is (m, n_out).

        For α=1.5, the support set is determined by finding the largest k such that:
            1 + k * z_sorted[k-1] > sum(z_sorted[:k]) + sqrt(k * (1 + sum(z_sorted[:k]^2) - sum(z_sorted[:k])^2))
        Then τ = (sum(z_sorted[:k]) - sqrt(k * (1 + sum(z_sorted[:k]^2) - sum(z_sorted[:k])^2) - ...)) / k

        Simplified exact formula per Peters et al.:
            τ such that sum_i max(z_i - τ, 0)^2 = 1
            → on the support of size k: tau = (sum_k(z) - sqrt(k + sum_k(z)^2 - k*sum_k(z^2))) / (−k)... ugh
        Vectorized via bisection across all rows in parallel.
        """
        m, n = z.shape
        # Bisection bounds: tau is in [max(z) - 1, max(z)]
        z_max = z.max(axis=1)  # (m,)
        lo = (z_max - 1.0)[:, None]  # (m, 1)
        hi = z_max[:, None]           # (m, 1)

        for _ in range(30):  # 30 iterations for ~1e-9 precision
            mid = (lo + hi) * 0.5                  # (m, 1)
            vals = np.maximum(z - mid, 0.0)         # (m, n)
            s = (vals ** 2).sum(axis=1, keepdims=True)  # (m, 1)
            # If s > 1, tau is too low (need higher tau to reduce support mass)
            too_low = s > 1.0
            lo = np.where(too_low, mid, lo)
            hi = np.where(too_low, hi, mid)

        tau = (lo + hi) * 0.5  # (m, 1)
        p = np.maximum(z - tau, 0.0)  # (m, n)
        # Normalize to sum=1 (entmax projects onto simplex; small numerical drift)
        p_sum = p.sum(axis=1, keepdims=True)
        empty = p_sum[:, 0] <= 1e-8
        if empty.any():
            # Fallback: one-hot on argmax for degenerate rows
            winners = z.argmax(axis=1)
            for row in np.where(empty)[0]:
                p[row] = 0.0
                p[row, winners[row]] = 1.0
                p_sum[row] = 1.0
        p = p / p_sum
        return p.astype(np.float32)

    @staticmethod
    def _sparsemax(z):
        """Sparsemax activation. Batched: z is (m, n_out)."""
        m, n = z.shape
        sorted_z = np.sort(z, axis=1)[:, ::-1]
        cumsum = np.cumsum(sorted_z, axis=1)
        k_arr = np.arange(1, n + 1, dtype=np.float32)
        support = sorted_z > (cumsum - 1.0) / k_arr
        # k = number of support elements per row
        k = support.sum(axis=1).astype(int)  # (m,)
        tau = (cumsum[np.arange(m), k - 1] - 1.0) / k
        p = np.maximum(z - tau[:, None], 0.0)
        return p.astype(np.float32)

    def _postprocess_probs(self, scores):
        z = scores / max(self.temperature, 1e-8)

        if self.activation == 'entmax15':
            return self._entmax15(z)
        elif self.activation == 'sparsemax':
            return self._sparsemax(z)
        else:
            # Original: softmax + top-k mask
            logits = z - z.max(axis=1, keepdims=True)
            e = np.exp(logits)
            p = e / e.sum(axis=1, keepdims=True).clip(1e-8)

            mask = self._topk_mask(scores)
            p = p * mask
            p_sum = p.sum(axis=1, keepdims=True)

            empty = p_sum[:, 0] <= self.sparse_floor
            if np.any(empty):
                winners = scores.argmax(axis=1)
                p[empty] = 0.0
                p[empty, winners[empty]] = 1.0
                p_sum = p.sum(axis=1, keepdims=True)

            p = p / p_sum.clip(1e-8)
            return p.astype(np.float32)

    def _update_fast_fatigue(self, p_active):
        self.fast_fatigue = (
            self.fatigue_decay * self.fast_fatigue + self.fatigue_rate * p_active
        ).astype(np.float32)

    def _update_homeostasis(self, p_active):
        self.usage = (
            self.usage_decay * self.usage + (1.0 - self.usage_decay) * p_active
        ).astype(np.float32)
        self.theta += self.homeostasis_rate * (self.usage - self.target_usage)
        self.theta = np.clip(self.theta, -self.theta_clip, self.theta_clip).astype(
            np.float32
        )

    def _lr_scale_from_usage(self):
        target = max(self.target_usage, 1e-6)
        underuse = np.clip((target - self.usage) / target, 0.0, 1.0)
        return (
            self.min_lr_scale
            + (self.max_lr_scale - self.min_lr_scale) * underuse
        ).astype(np.float32)

    def _update_prototypes(self, state_np, p_active, scores, sim_raw, surprise=None):
        ar = self._arange_m

        # --- Winner pull ---
        # Decoupled: learn from raw similarity (no theta bias)
        # Coupled: learn from emitted p_active (theta-biased)
        if self.theta_learn_scale < 1.0:
            sim_learn = sim_raw - self.theta_learn_scale * self.theta
            p_learn = self._postprocess_probs(sim_learn)
        else:
            p_learn = p_active

        if not self.learn_from_probs:
            winners = p_learn.argmax(axis=1)
            learn = np.zeros_like(p_learn, dtype=np.float32)
            learn[ar, winners] = 1.0
        else:
            learn = p_learn

        lr_scale = self._lr_scale_from_usage()
        # Surprise modulation: higher surprise → faster learning
        if surprise is not None and self.surprise_alpha > 0:
            surprise_boost = 1.0 + self.surprise_alpha * surprise  # (m,)
            lr_scale = lr_scale * surprise_boost[:, None]
        step = (self.proto_lr * lr_scale * learn)[:, :, None]
        self.prototypes += step * (state_np[:, None, :] - self.prototypes)

        active_mask = learn > 1e-4
        self.last_won[active_mask] = self._tick_count

        # --- Loser repulsion (new) ---
        if self.lr_neg > 0:
            # Hunger-gated: repulsion strongest when satiated, weakest when hungry
            lr_neg_eff = self.lr_neg
            if lr_neg_eff > 1e-8:
                winners = scores.argmax(axis=1)                    # (m,)
                winner_scores = scores[ar, winners]                # (m,)

                # Number of losers to repel scales with k_active
                n_repel = max(1, self.k_active)

                # Mask out winners to find losers
                loser_scores = scores.copy()
                loser_scores[ar, winners] = -np.inf
                # For k_active > 1, also mask other active outputs
                if self.k_active > 1:
                    active_mask_k = self._topk_mask(scores)  # (m, n_out)
                    loser_scores[active_mask_k > 0] = -np.inf

                # Select top-n_repel losers per column
                n_repel = min(n_repel, self.n_outputs - max(1, self.k_active))
                if n_repel > 0:
                    if n_repel >= loser_scores.shape[1]:
                        loser_idx = np.argsort(loser_scores, axis=1)[:, -n_repel:]
                    else:
                        loser_idx = np.argpartition(
                            loser_scores, -n_repel, axis=1
                        )[:, -n_repel:]

                    # Weakest winner score (for band check)
                    if self.k_active > 1:
                        active_scores = scores.copy()
                        active_scores[active_mask_k == 0] = np.inf
                        weakest_winner = active_scores.min(axis=1)  # (m,)
                    else:
                        weakest_winner = winner_scores

                    # Apply repulsion per loser, gated by similarity band
                    for li in range(n_repel):
                        l_idx = loser_idx[:, li]                   # (m,)
                        l_scores = scores[ar, l_idx]               # (m,)
                        # Only repel if within margin_band of weakest winner
                        close = l_scores > (weakest_winner - self.margin_band)
                        if close.any():
                            close_idx = np.where(close)[0]
                            l_out = l_idx[close_idx]
                            # Repel: push prototype away from descriptor
                            diff = state_np[close_idx] - self.prototypes[close_idx, l_out]
                            self.prototypes[close_idx, l_out] -= lr_neg_eff * diff

        # Renormalize all prototypes
        norms = np.linalg.norm(self.prototypes, axis=2, keepdims=True).clip(1e-8)
        self.prototypes = (self.prototypes / norms).astype(np.float32)

    def _reseed_dead_units(self, state_np):
        if self.reseed_after <= 0:
            return
        dead_ticks = self._tick_count - self.last_won
        dead_mask = (dead_ticks > self.reseed_after) & (
            self.usage < self.dead_usage_threshold
        )
        dead_cols, dead_outs = np.where(dead_mask)
        for c, o in zip(dead_cols, dead_outs):
            self.prototypes[c, o] = state_np[c]
            self.theta[c, o] = 0.0
            self.fast_fatigue[c, o] = 0.0
            self.last_won[c, o] = self._tick_count

    def _update_corr_diagnostics(self, p_active):
        """Store output history and periodically compute correlation diagnostics."""
        self._corr_buf[self._corr_buf_idx] = p_active
        self._corr_buf_idx = (self._corr_buf_idx + 1) % self.corr_window
        if self._corr_buf_idx == 0:
            self._corr_buf_full = True

        # Compute every corr_window ticks once buffer is full
        if self._corr_buf_full and self._corr_buf_idx == 0:
            A = self._corr_buf  # (corr_window, m, n_out)

            # Raw output correlation per column
            # A_t: (m, corr_window, n_out) for per-column covariance
            A_t = A.transpose(1, 0, 2)  # (m, corr_window, n_out)
            A_mean = A_t.mean(axis=1, keepdims=True)
            A_c = A_t - A_mean
            # (m, n_out, n_out) covariance
            raw_cov = np.einsum('mti,mtj->mij', A_c, A_c) / (self.corr_window - 1)

            # Normalize to correlation
            std = np.sqrt(np.diagonal(raw_cov, axis1=1, axis2=2)).clip(1e-8)
            raw_corr = raw_cov / (std[:, :, None] * std[:, None, :])
            self.last_raw_corr = raw_corr

            # Centered by running mean usage (removes "both popular" baseline)
            usage_mean = self.usage[:, None, :]  # (m, 1, n_out)
            A_centered = A_t - usage_mean
            cen_cov = np.einsum('mti,mtj->mij', A_centered, A_centered) / (self.corr_window - 1)
            std_c = np.sqrt(np.diagonal(cen_cov, axis1=1, axis2=2)).clip(1e-8)
            cen_corr = cen_cov / (std_c[:, :, None] * std_c[:, None, :])
            self.last_centered_corr = cen_corr

            # Decorrelation: push prototypes of correlated outputs apart
            if self.decorrelation and self.decor_lr > 0:
                n_out = self.n_outputs
                # Zero the diagonal (self-correlation is always 1, ignore it)
                offdiag = cen_corr.copy()
                offdiag[:, np.arange(n_out), np.arange(n_out)] = 0.0
                # For each pair (i,j) with positive centered correlation,
                # push prototype[i] and prototype[j] apart proportionally
                for i in range(n_out):
                    for j in range(i + 1, n_out):
                        corr_ij = offdiag[:, i, j]  # (m,)
                        # Only push apart positively correlated pairs
                        push_mask = corr_ij > 0.1
                        if push_mask.any():
                            idx = np.where(push_mask)[0]
                            strength = self.decor_lr * corr_ij[idx, None]  # (n_push, 1)
                            diff = self.prototypes[idx, i] - self.prototypes[idx, j]  # (n_push, d_model)
                            self.prototypes[idx, i] += strength * diff
                            self.prototypes[idx, j] -= strength * diff
                # Renormalize
                norms = np.linalg.norm(self.prototypes, axis=2, keepdims=True).clip(1e-8)
                self.prototypes = (self.prototypes / norms).astype(np.float32)

    def _check_prediction(self, current_frame):
        """Compare last tick's prediction to actual raw frame. Returns per-column normalized surprise."""
        if self._pending_pred_np is None:
            return np.zeros(self.m, dtype=np.float32)
        err = ((self._pending_pred_np - current_frame) ** 2).mean(axis=1)  # (m,)
        self._surprise_raw = err
        self._surprise_ema = (self.surprise_beta * self._surprise_ema
                              + (1.0 - self.surprise_beta) * err)
        norm = np.clip(err / self._surprise_ema.clip(1e-8), 0.0, 5.0)
        self.last_pred_error = float(err.mean())
        return norm.astype(np.float32)

    def _make_prediction(self, desc_t, a_t):
        """Predict next raw frame from current descriptor + output. GPU-accelerated."""
        pred_input = np.concatenate([desc_t, a_t], axis=1)  # (m, pred_in)
        inp_t = torch.from_numpy(pred_input).to(self._pred_device)
        pred_t = torch.bmm(inp_t.unsqueeze(1), self._pred_W).squeeze(1) + self._pred_b
        self._pending_pred_np = pred_t.cpu().numpy()
        self._last_pred_input_t = inp_t

    def _update_predictor(self, current_frame):
        """Train predictor via gradient descent on GPU."""
        if self._pending_pred_np is None or self._last_pred_input_t is None:
            return
        target_t = torch.from_numpy(current_frame).to(self._pred_device)
        pred_t = torch.from_numpy(self._pending_pred_np).to(self._pred_device)
        error = pred_t - target_t  # (m, max_inputs)
        # grad_W = input^T @ error: (m, pred_in, 1) @ (m, 1, max_inputs)
        grad_W = torch.bmm(self._last_pred_input_t.unsqueeze(2), error.unsqueeze(1))
        self._pred_W -= self.pred_lr * grad_W
        self._pred_b -= self.pred_lr * error

    # ------------------------------------------------------------------
    # Output rotation hooks — fully owned by this class
    # ------------------------------------------------------------------

    def apply_output_rotation(self, sim):
        """Subtract slow homeostatic theta + fast activity fatigue from scores."""
        return sim - self.theta - self.fatigue_strength * self.fast_fatigue

    def save_rotation_state(self):
        """Rotation state owned by this class (theta, fast_fatigue, wta_mode)."""
        return {
            "theta": torch.from_numpy(self.theta),
            "fast_fatigue": torch.from_numpy(self.fast_fatigue),
            "wta_mode": self.wta_mode,
        }

    def load_rotation_state(self, state):
        if "theta" in state:
            t = state["theta"]
            self.theta = t.numpy().astype(np.float32) if hasattr(t, "numpy") else np.array(t, dtype=np.float32)
        if "fast_fatigue" in state:
            t = state["fast_fatigue"]
            self.fast_fatigue = t.numpy().astype(np.float32) if hasattr(t, "numpy") else np.array(t, dtype=np.float32)
        if "wta_mode" in state:
            self.wta_mode = state["wta_mode"]

    # ------------------------------------------------------------------
    # Main tick
    # ------------------------------------------------------------------

    def tick(self, signal_window, knn2=None):
        self._tick_count += 1

        X = self._gather_input(signal_window).astype(np.float32)
        current_frame = X[:, :, -1]  # raw frame for predictor target
        state_np = self._state_descriptor_np(X)

        # Check last tick's prediction against actual raw frame
        surprise = self._check_prediction(current_frame)

        sim_raw = self._raw_similarity(state_np)

        # Our rotation: homeostatic theta + fast fatigue
        scores = self.apply_output_rotation(sim_raw.copy())

        p_active = self._postprocess_probs(scores)

        if self._rng.rand() < self._learn_prob:
            self._update_prototypes(state_np, p_active, scores, sim_raw, surprise)
            self._update_predictor(current_frame)
            self._reseed_dead_units(state_np)

        self._update_fast_fatigue(p_active)
        self._update_homeostasis(p_active)
        self._update_corr_diagnostics(p_active)

        # Predict next raw frame from current descriptor + output
        self._make_prediction(state_np, p_active)

        self._outputs = self.apply_wta(p_active)

    # ------------------------------------------------------------------
    # API helpers
    # ------------------------------------------------------------------

    def set_learn_prob(self, value):
        self._learn_prob = float(np.clip(value, 0.0, 1.0))

    def set_hunger(self, value):
        """Set external hunger level for gating loser repulsion."""
        self._hunger_level = float(np.clip(value, 0.0, 1.0))

    def get_fast_fatigue(self):
        return self.fast_fatigue.copy()

    def get_corr_diagnostics(self):
        """Return (raw_corr, centered_corr) or (None, None) if not ready."""
        return self.last_raw_corr, self.last_centered_corr

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, output_dir):
        np.save(os.path.join(output_dir, "column_slot_map.npy"), self.slot_map)
        state = {
            "type": "temporal_prototype",
            "tick_count": self._tick_count,
            "m": self.m,
            "n_outputs": self.n_outputs,
            "max_inputs": self.max_inputs,
            "window": self.window,
            "lr": self.lr,
            "proto_lr": self.proto_lr,
            "temperature": self.temperature,
            "usage_decay": self.usage_decay,
            "current_weight": self.current_weight,
            "mean_weight": self.mean_weight,
            "delta_weight": self.delta_weight,
            "k_active": self.k_active,
            "homeostasis_rate": self.homeostasis_rate,
            "target_usage": self.target_usage,
            "theta_clip": self.theta_clip,
            "fatigue_decay": self.fatigue_decay,
            "fatigue_rate": self.fatigue_rate,
            "fatigue_strength": self.fatigue_strength,
            "learn_from_probs": self.learn_from_probs,
            "reseed_after": self.reseed_after,
            "min_lr_scale": self.min_lr_scale,
            "max_lr_scale": self.max_lr_scale,
            "dead_usage_threshold": self.dead_usage_threshold,
            "sparse_floor": self.sparse_floor,
            "lr_neg": self.lr_neg,
            "margin_band": self.margin_band,
            "corr_window": self.corr_window,
            "d_model": self.d_model,
            "prototypes": torch.from_numpy(self.prototypes),
            "usage": torch.from_numpy(self.usage),
            "theta": torch.from_numpy(self.theta),
            "fast_fatigue": torch.from_numpy(self.fast_fatigue),
            "last_won": torch.from_numpy(self.last_won),
            "_reserved_mask": torch.from_numpy(self._reserved_mask.astype(np.uint8)),
            "wta_mode": self.wta_mode,
            # Predictor state
            "pred_W": self._pred_W.cpu(),
            "pred_b": self._pred_b.cpu(),
            "surprise_ema": torch.from_numpy(self._surprise_ema),
            "pred_lr": self.pred_lr,
            "surprise_alpha": self.surprise_alpha,
            "surprise_beta": self.surprise_beta,
        }
        torch.save(state, os.path.join(output_dir, "column_states.pt"))

        # Save correlation diagnostics separately (large arrays, optional analysis)
        if self.last_raw_corr is not None:
            np.savez(os.path.join(output_dir, "column_corr_diagnostics.npz"),
                     raw_corr=self.last_raw_corr,
                     centered_corr=self.last_centered_corr)

        # Print summary
        pred_err = f", pred_err={self.last_pred_error:.4f}" if self._pending_pred_np is not None else ""
        corr_summary = ""
        if self.last_raw_corr is not None:
            n_out = self.n_outputs
            offdiag_mask = ~np.eye(n_out, dtype=bool)
            mean_offdiag = np.abs(self.last_raw_corr[:, offdiag_mask]).mean()
            corr_summary = f", output_corr={mean_offdiag:.3f}"
        print(f"  temporal_prototype column state saved to {output_dir}"
              f"{pred_err}{corr_summary}")

    def load_state(self, state, slot_map):
        def _to_np(t):
            return t.numpy() if hasattr(t, "numpy") else np.array(t)

        self.slot_map = slot_map
        if state.get("_reserved_mask") is not None:
            self._reserved_mask = _to_np(state["_reserved_mask"]).astype(bool)

        if "tick_count" in state:
            self._tick_count = int(state["tick_count"])
        if "prototypes" in state:
            self.prototypes = _to_np(state["prototypes"]).astype(np.float32)
        if "usage" in state:
            self.usage = _to_np(state["usage"]).astype(np.float32)
        if "theta" in state:
            self.theta = _to_np(state["theta"]).astype(np.float32)
        if "fast_fatigue" in state:
            self.fast_fatigue = _to_np(state["fast_fatigue"]).astype(np.float32)
        if "last_won" in state:
            self.last_won = _to_np(state["last_won"]).astype(np.int64)
        if "wta_mode" in state:
            self.wta_mode = state["wta_mode"]
        if "pred_W" in state:
            t = state["pred_W"]
            self._pred_W = (t if isinstance(t, torch.Tensor) else torch.from_numpy(_to_np(t))).float().to(self._pred_device)
        if "pred_b" in state:
            t = state["pred_b"]
            self._pred_b = (t if isinstance(t, torch.Tensor) else torch.from_numpy(_to_np(t))).float().to(self._pred_device)
        if "surprise_ema" in state:
            self._surprise_ema = _to_np(state["surprise_ema"]).astype(np.float32)

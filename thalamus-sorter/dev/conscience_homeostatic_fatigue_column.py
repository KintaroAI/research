import os
import numpy as np
import torch

from column_manager import ColumnBase


class ConscienceHomeostaticFatigueColumn(ColumnBase):
    """
    Conscience-style prototype column with:
      - streaming descriptor [current, mean, delta]
      - soft/top-k competition
      - fast fatigue (short-term anti-monopoly)
      - slow homeostatic thresholds (long-term anti-collapse)
      - winner/near-winner Hebbian prototype updates

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

        self.d_model = self.max_inputs * 3

        rng = np.random.RandomState(42)
        protos = rng.randn(m, n_outputs, self.d_model).astype(np.float32)
        protos /= np.linalg.norm(protos, axis=2, keepdims=True).clip(1e-8)
        self.prototypes = protos

        self.usage = np.full((m, n_outputs), 1.0 / n_outputs, dtype=np.float32)
        self.theta = np.zeros((m, n_outputs), dtype=np.float32)
        self.fast_fatigue = np.zeros((m, n_outputs), dtype=np.float32)
        self.last_won = np.zeros((m, n_outputs), dtype=np.int64)

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
        Returns: (m, 3 * n_in)
        """
        current = X[:, :, -1]
        mean = X.mean(axis=2)
        if X.shape[2] > 1:
            delta = X[:, :, -1] - X[:, :, -2]
        else:
            delta = np.zeros_like(current)

        cur_n = self._part_norm(current)
        mean_n = self._part_norm(mean)
        delta_n = self._part_norm(delta)

        desc = np.concatenate(
            [
                self.current_weight * cur_n,
                self.mean_weight * mean_n,
                self.delta_weight * delta_n,
            ],
            axis=1,
        )
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

    def _postprocess_probs(self, scores):
        logits = scores - scores.max(axis=1, keepdims=True)
        e = np.exp(logits / max(self.temperature, 1e-8))
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

    def _update_prototypes(self, state_np, p_active):
        if not self.learn_from_probs:
            winners = p_active.argmax(axis=1)
            learn = np.zeros_like(p_active, dtype=np.float32)
            learn[self._arange_m, winners] = 1.0
        else:
            learn = p_active

        lr_scale = self._lr_scale_from_usage()
        step = (self.proto_lr * lr_scale * learn)[:, :, None]
        self.prototypes += step * (state_np[:, None, :] - self.prototypes)

        norms = np.linalg.norm(self.prototypes, axis=2, keepdims=True).clip(1e-8)
        self.prototypes = (self.prototypes / norms).astype(np.float32)

        active_mask = learn > 1e-4
        self.last_won[active_mask] = self._tick_count

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
            self.output_tiredness[c, o] = 0.0
            self.last_won[c, o] = self._tick_count

    # ------------------------------------------------------------------
    # Main tick
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Output rotation hooks — fully owned by this class
    # ------------------------------------------------------------------

    def apply_output_rotation(self, sim):
        """Subtract slow homeostatic theta + fast activity fatigue from scores."""
        return sim - self.theta - self.fatigue_strength * self.fast_fatigue

    def update_output_rotation(self, winners):
        """No-op: homeostasis + fatigue are p_active-based and updated in tick()."""
        pass

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

    def tick(self, signal_window, knn2=None):
        self._tick_count += 1

        X = self._gather_input(signal_window).astype(np.float32)
        state_np = self._state_descriptor_np(X)

        sim_raw = self._raw_similarity(state_np)

        # Our rotation: homeostatic theta + fast fatigue
        scores = self.apply_output_rotation(sim_raw.copy())

        p_active = self._postprocess_probs(scores)
        winners = scores.argmax(axis=1)

        if self._rng.rand() < self._learn_prob:
            self._update_prototypes(state_np, p_active)
            self._reseed_dead_units(state_np)

        self._update_fast_fatigue(p_active)
        self._update_homeostasis(p_active)

        self.update_output_rotation(winners)

        self._outputs = self.apply_wta(p_active)

    # ------------------------------------------------------------------
    # API helpers
    # ------------------------------------------------------------------

    def set_learn_prob(self, value):
        self._learn_prob = float(np.clip(value, 0.0, 1.0))

    def get_fast_fatigue(self):
        return self.fast_fatigue.copy()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, output_dir):
        np.save(os.path.join(output_dir, "column_slot_map.npy"), self.slot_map)
        state = {
            "type": "conscience_homeostatic_fatigue",
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
            "d_model": self.d_model,
            "prototypes": torch.from_numpy(self.prototypes),
            "usage": torch.from_numpy(self.usage),
            "theta": torch.from_numpy(self.theta),
            "fast_fatigue": torch.from_numpy(self.fast_fatigue),
            "last_won": torch.from_numpy(self.last_won.astype(np.float32)),
            "_reserved_mask": torch.from_numpy(self._reserved_mask.astype(np.uint8)),
            "wta_mode": self.wta_mode,
        }
        torch.save(state, os.path.join(output_dir, "column_states.pt"))
        print(f"  conscience_homeostatic_fatigue column state saved to {output_dir}")

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

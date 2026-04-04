import os
import numpy as np
import torch
import torch.nn.functional as F

from column_manager import ColumnBase


class ConscienceWithPredictiveOverrideColumn(ColumnBase):
    """Conscience-first column with transformer predictor and trust-gated override.

    Design:
        - The conscience head is the primary state emitter.
        - A transformer predictor learns in parallel from the full temporal window.
        - Predictor trust comes from EMA next-frame prediction error.
        - Predictor can only bias the emitted logits when trust is high.
        - Conscience prototype ecology is updated from conscience-only winners,
          so top-down prediction cannot hijack category identity too early.

    Output semantics:
        - self._outputs      : final emitted state distribution (conscience + bias)
        - self._surprise     : per-column previous next-frame prediction error
        - self._gate         : per-column trust gate in [0, gate_max]
        - self._prev_prediction : previous next-frame prediction

    Notes:
        - This class intentionally uses torch on both CPU and GPU, because the
          predictor path is transformer-based anyway.
        - The conscience head uses a simple handcrafted descriptor
          [current, mean, delta] rather than a learned encoder, to preserve
          exploration and output churn.
    """

    def __init__(
        self,
        m,
        n_outputs=4,
        max_inputs=20,
        window=4,
        lr=0.05,
        pred_lr=1e-3,
        alpha=0.01,
        temperature=0.5,
        n_heads=2,
        reseed_after=1000,
        usage_decay=0.99,
        proto_lr=None,
        current_weight=1.0,
        mean_weight=0.6,
        delta_weight=0.8,
        lambda_pred=1.0,
        lambda_state=0.2,
        beta_override=1.0,
        gate_gamma=8.0,
        gate_max=0.25,
        gate_decay=0.95,
        pred_lr_gain=1.0,
        pred_lr_scale_max=3.0,
        tiredness_rate=0.0,
        tiredness_recovery=0.0,
        **kwargs,
    ):
        wta_mode = kwargs.pop('wta_mode', 'none')
        self._lateral_inputs = kwargs.pop('lateral_inputs', False)
        self._lateral_input_k = kwargs.pop('lateral_input_k', 4)
        if self._lateral_inputs:
            max_inputs = max_inputs + self._lateral_input_k * 2
        super().__init__(m, n_outputs, max_inputs, window, lr)

        self.alpha = float(alpha)
        self.temperature = float(temperature)
        self.n_heads = int(n_heads)
        self.reseed_after = int(reseed_after)
        self.usage_decay = float(usage_decay)
        self.proto_lr = float(lr if proto_lr is None else proto_lr)
        self.current_weight = float(current_weight)
        self.mean_weight = float(mean_weight)
        self.delta_weight = float(delta_weight)
        self.lambda_pred = float(lambda_pred)
        self.lambda_state = float(lambda_state)
        self.beta_override = float(beta_override)
        self.gate_gamma = float(gate_gamma)
        self.gate_max = float(gate_max)
        self.gate_decay = float(gate_decay)
        self.pred_lr = float(pred_lr)
        self.pred_lr_gain = float(pred_lr_gain)
        self.pred_lr_scale_max = float(pred_lr_scale_max)
        self.tiredness_rate = float(tiredness_rate)
        self.tiredness_recovery = float(tiredness_recovery)
        self.wta_mode = wta_mode

        self.d_model = max_inputs
        assert self.d_model % self.n_heads == 0, (
            f"d_model={self.d_model} must be divisible by n_heads={self.n_heads}"
        )
        self.d_ff = 4 * self.d_model

        # ------------------------------------------------------------------
        # Conscience state head: same descriptor as ConscienceColumn
        # ------------------------------------------------------------------
        rng = np.random.RandomState(42)
        protos = rng.randn(m, n_outputs, self.d_model).astype(np.float32)
        protos /= np.linalg.norm(protos, axis=2, keepdims=True).clip(1e-8)
        self.prototypes = protos
        self.usage = np.full((m, n_outputs), 1.0 / n_outputs, dtype=np.float32)
        self.theta = np.zeros((m, n_outputs), dtype=np.float32)
        self.last_won = np.zeros((m, n_outputs), dtype=np.int64)

        # ------------------------------------------------------------------
        # Predictor: 1-layer causal transformer + heads
        # ------------------------------------------------------------------
        d = self.d_model
        d_ff = self.d_ff
        self.pos_emb = torch.nn.Parameter(torch.randn(m, window, d) * 0.02)
        self.ln1_g = torch.nn.Parameter(torch.ones(m, d))
        self.ln1_b = torch.nn.Parameter(torch.zeros(m, d))
        self.W_qkv = torch.nn.Parameter(torch.randn(m, d, 3 * d) * (d ** -0.5))
        self.W_proj = torch.nn.Parameter(torch.randn(m, d, d) * (d ** -0.5))
        self.ln2_g = torch.nn.Parameter(torch.ones(m, d))
        self.ln2_b = torch.nn.Parameter(torch.zeros(m, d))
        self.W_fc1 = torch.nn.Parameter(torch.randn(m, d, d_ff) * (d ** -0.5))
        self.b_fc1 = torch.nn.Parameter(torch.zeros(m, d_ff))
        self.W_fc2 = torch.nn.Parameter(torch.randn(m, d_ff, d) * (d_ff ** -0.5))
        self.b_fc2 = torch.nn.Parameter(torch.zeros(m, d))

        # Predict next frame x_{t+1} from predictor hidden state.
        self.W_pred_x = torch.nn.Parameter(torch.randn(m, d, d) * (d ** -0.5))
        self.b_pred_x = torch.nn.Parameter(torch.zeros(m, d))
        # Predict state-bias logits from predictor hidden state.
        self.W_pred_state = torch.nn.Parameter(torch.randn(m, d, n_outputs) * (d ** -0.5))
        self.b_pred_state = torch.nn.Parameter(torch.zeros(m, n_outputs))

        self._params = [
            self.pos_emb, self.ln1_g, self.ln1_b,
            self.W_qkv, self.W_proj,
            self.ln2_g, self.ln2_b,
            self.W_fc1, self.b_fc1, self.W_fc2, self.b_fc2,
            self.W_pred_x, self.b_pred_x,
            self.W_pred_state, self.b_pred_state,
        ]
        self._param_names = [
            'pos_emb', 'ln1_g', 'ln1_b',
            'W_qkv', 'W_proj',
            'ln2_g', 'ln2_b',
            'W_fc1', 'b_fc1', 'W_fc2', 'b_fc2',
            'W_pred_x', 'b_pred_x',
            'W_pred_state', 'b_pred_state',
        ]

        self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if self._device.type == 'cuda':
            for p in self._params:
                p.data = p.data.to(self._device)
        self._optimizer = torch.optim.Adam(self._params, lr=self.pred_lr)

        self._causal_mask = torch.tril(torch.ones(window, window, device=self._device))
        self._inv_sqrt_hs = (self.d_model // self.n_heads) ** -0.5

        # ------------------------------------------------------------------
        # Runtime state
        # ------------------------------------------------------------------
        self._tick_count = 0
        self._learn_prob = 1.0
        self._external_lr_scale = 1.0
        self._rng = np.random.RandomState(42)
        self._prev_prediction = np.zeros((m, self.d_model), dtype=np.float32)
        self._surprise = np.zeros(m, dtype=np.float32)
        self._pred_err_ema = np.ones(m, dtype=np.float32)
        self._gate = np.zeros(m, dtype=np.float32)

    # ------------------------------------------------------------------
    # Helpers: conscience state head
    # ------------------------------------------------------------------
    def _normalize_np(self, x):
        return x / np.linalg.norm(x, axis=-1, keepdims=True).clip(1e-8)

    def _normalize_t(self, x):
        return x / x.norm(dim=-1, keepdim=True).clamp(min=1e-8)

    def _state_descriptor_np(self, X):
        """Same descriptor as ConscienceColumn: mean(window) -> mean-sub -> L2-norm.

        X: (m, n_in, T)
        Returns: (m, n_in)
        """
        x = X.mean(axis=2)
        x = x - x.mean(axis=1, keepdims=True)
        norms = np.linalg.norm(x, axis=1, keepdims=True).clip(1e-8)
        return (x / norms).astype(np.float32)

    def _state_descriptor_torch_targets(self, x_gpu):
        """Prefix conscience descriptors for positions 1..T-1.
        Same as ConscienceColumn: mean(frames[:t]) -> mean-sub -> L2-norm.

        x_gpu: (m, T, d)
        Returns: (m, T-1, d)
        """
        csum = x_gpu.cumsum(dim=1)
        counts = torch.arange(
            1, x_gpu.shape[1], device=x_gpu.device, dtype=x_gpu.dtype
        ).view(1, -1, 1)
        x = csum[:, :-1, :] / counts
        x = x - x.mean(dim=-1, keepdim=True)
        return self._normalize_t(x)

    def _conscience_logits_np(self, state_np, apply_rotation=True):
        protos = self.prototypes / np.linalg.norm(self.prototypes, axis=2, keepdims=True).clip(1e-8)
        sim = np.einsum('moi,mi->mo', protos, state_np)
        scores = self.apply_output_rotation(sim.copy()) if apply_rotation else sim
        return sim.astype(np.float32), scores.astype(np.float32)

    def _conscience_winner_np(self, state_np):
        _, scores = self._conscience_logits_np(state_np, apply_rotation=True)
        return scores.argmax(axis=1), scores

    # ------------------------------------------------------------------
    # Helpers: predictor head
    # ------------------------------------------------------------------
    def _layer_norm(self, x, g, b):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        x = (x - mean) / (var + 1e-5).sqrt()
        return x * g.unsqueeze(1) + b.unsqueeze(1)

    def _encode_torch(self, x_gpu):
        """x_gpu: (m, T, d) -> hidden states (m, T, d)."""
        m, T, d = x_gpu.shape
        n_heads = self.n_heads
        hs = d // n_heads

        x = x_gpu + self.pos_emb[:, :T, :]
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

    def _predict_next_and_state(self, h):
        """h: (m, Tctx, d) or (m, d).
        Returns next-frame pred and state-bias logits.
        """
        if h.dim() == 2:
            pred_x = torch.bmm(h.unsqueeze(1), self.W_pred_x).squeeze(1) + self.b_pred_x
            logits_state = torch.bmm(h.unsqueeze(1), self.W_pred_state).squeeze(1) + self.b_pred_state
        else:
            pred_x = torch.bmm(h, self.W_pred_x) + self.b_pred_x.unsqueeze(1)
            logits_state = torch.bmm(h, self.W_pred_state) + self.b_pred_state.unsqueeze(1)
        return pred_x, logits_state

    def _gate_from_error(self):
        g = np.exp(-self.gate_gamma * self._pred_err_ema)
        g = np.clip(g, 0.0, self.gate_max)
        self._gate = g.astype(np.float32)
        return self._gate

    def _set_optimizer_lr(self):
        if self.gate_max <= 1e-8:
            uncertainty = 1.0
        else:
            uncertainty = 1.0 - float(np.clip(self._gate.mean() / self.gate_max, 0.0, 1.0))
        scale = 1.0 + self.pred_lr_gain * uncertainty
        scale = float(np.clip(scale, 1.0, self.pred_lr_scale_max))
        scale *= self._external_lr_scale
        for group in self._optimizer.param_groups:
            group['lr'] = self.pred_lr * scale

    # ------------------------------------------------------------------
    # Main tick
    # ------------------------------------------------------------------
    def tick(self, signal_window, knn2=None):
        self._tick_count += 1

        X = self._gather_input(signal_window).astype(np.float32)   # (m, n_in, T)
        x_np = X.transpose(0, 2, 1).astype(np.float32)             # (m, T, d)
        current_frame_np = x_np[:, -1, :]
        valid = (self.slot_map >= 0).astype(np.float32)
        n_valid = valid.sum(axis=1).clip(min=1.0)

        # ------------------------------------------------------------------
        # Evaluate previous prediction -> surprise/trust update
        # ------------------------------------------------------------------
        if self._tick_count > 1:
            diff = (self._prev_prediction - current_frame_np) ** 2
            err = (diff * valid).sum(axis=1) / n_valid
            self._surprise = err.astype(np.float32)
            self._pred_err_ema = (
                self.gate_decay * self._pred_err_ema + (1.0 - self.gate_decay) * err
            ).astype(np.float32)
        else:
            self._surprise.fill(0.0)
        gate = self._gate_from_error()

        # ------------------------------------------------------------------
        # Conscience head (numpy): current state + raw current output identity
        # ------------------------------------------------------------------
        state_np = self._state_descriptor_np(X)
        sim_c_np, scores_c_np = self._conscience_logits_np(state_np, apply_rotation=True)
        winners_c = scores_c_np.argmax(axis=1)

        # ------------------------------------------------------------------
        # Predictor training/inference (torch)
        # ------------------------------------------------------------------
        x_t = torch.from_numpy(x_np).to(self._device)
        do_train = self._rng.rand() < self._learn_prob

        if do_train and x_t.shape[1] > 1:
            self._set_optimizer_lr()
            h_all = self._encode_torch(x_t)
            h_ctx = h_all[:, :-1, :]
            pred_x_all, logits_state_all = self._predict_next_and_state(h_ctx)

            # Train next-frame prediction.
            targets_x = x_t[:, 1:, :]
            valid_gpu = torch.from_numpy(valid).to(self._device)
            n_valid_gpu = valid_gpu.sum(dim=1).clamp(min=1.0)
            loss_x = (((pred_x_all - targets_x) ** 2) * valid_gpu.unsqueeze(1)).sum(dim=2)
            loss_x = loss_x.div(n_valid_gpu.unsqueeze(1)).mean()

            # Train state-bias logits to predict next conscience winner.
            target_states = self._state_descriptor_torch_targets(x_t)
            protos_t = torch.from_numpy(self.prototypes).to(self._device)
            protos_t = protos_t / protos_t.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            sim_targets = torch.bmm(target_states, protos_t.transpose(1, 2))
            target_winners = sim_targets.argmax(dim=-1)
            loss_state = F.cross_entropy(
                logits_state_all.reshape(-1, self.n_outputs),
                target_winners.reshape(-1),
            )

            loss = self.lambda_pred * loss_x + self.lambda_state * loss_state
            self._optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self._params, 1.0)
            self._optimizer.step()

        with torch.no_grad():
            h_all = self._encode_torch(x_t)
            h_last = h_all[:, -1, :]
            pred_last_t, logits_p_t = self._predict_next_and_state(h_last)

        logits_p_np = logits_p_t.cpu().numpy().astype(np.float32)
        logits_p_np = logits_p_np - logits_p_np.mean(axis=1, keepdims=True)

        # ------------------------------------------------------------------
        # Final emitted output: softmax on raw sims + trusted predictive bias
        # (rotation only affects winner selection above, not output distribution)
        # ------------------------------------------------------------------
        final_logits = sim_c_np + (self.beta_override * gate)[:, None] * logits_p_np
        final_logits = final_logits - final_logits.max(axis=1, keepdims=True)
        e = np.exp(final_logits / max(self.temperature, 1e-8))
        p_final = (e / e.sum(axis=1, keepdims=True)).astype(np.float32)

        # ------------------------------------------------------------------
        # Conscience ecology update uses conscience-only winners, not biased ones
        # ------------------------------------------------------------------
        ar = self._arange_m
        w_proto = self.prototypes[ar, winners_c]
        new_proto = (1.0 - self.proto_lr) * w_proto + self.proto_lr * state_np
        new_proto /= np.linalg.norm(new_proto, axis=1, keepdims=True).clip(1e-8)
        self.prototypes[ar, winners_c] = new_proto

        self.last_won[ar, winners_c] = self._tick_count
        if self.reseed_after > 0:
            dead_ticks = self._tick_count - self.last_won
            dead_cols, dead_outs = np.where(dead_ticks > self.reseed_after)
            if len(dead_cols) > 0:
                for c, o in zip(dead_cols, dead_outs):
                    self.prototypes[c, o] = state_np[c]
                    self.theta[c, o] = 0.0
                    self.last_won[c, o] = self._tick_count

        # Update conscience rotation/tiredness from conscience-only winners.
        self.update_output_rotation(winners_c)

        # Usage EMA from conscience winners.
        y = np.zeros((self.m, self.n_outputs), dtype=np.float32)
        y[ar, winners_c] = 1.0
        self.usage = self.usage * self.usage_decay + y * (1.0 - self.usage_decay)

        self._outputs = self.apply_wta(p_final)
        self._prev_prediction = pred_last_t.cpu().numpy().astype(np.float32)

    # ------------------------------------------------------------------
    # API helpers
    # ------------------------------------------------------------------
    def get_surprise(self):
        return self._surprise.copy()

    def get_gate(self):
        return self._gate.copy()

    def set_learn_prob(self, value):
        self._learn_prob = float(value)

    def set_pred_lr_scale(self, scale):
        """Set external LR scale factor. Composed with internal adaptive scale in _set_optimizer_lr."""
        self._external_lr_scale = float(np.clip(scale, 0.0, self.pred_lr_scale_max))

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save(self, output_dir):
        np.save(os.path.join(output_dir, "column_slot_map.npy"), self.slot_map)
        state = {
            'type': 'conscience_predictive_override',
            'tick_count': self._tick_count,
            'm': self.m,
            'n_outputs': self.n_outputs,
            'max_inputs': self.max_inputs,
            'window': self.window,
            'lr': self.lr,
            'pred_lr': self.pred_lr,
            'alpha': self.alpha,
            'temperature': self.temperature,
            'n_heads': self.n_heads,
            'reseed_after': self.reseed_after,
            'usage_decay': self.usage_decay,
            'proto_lr': self.proto_lr,
            'current_weight': self.current_weight,
            'mean_weight': self.mean_weight,
            'delta_weight': self.delta_weight,
            'lambda_pred': self.lambda_pred,
            'lambda_state': self.lambda_state,
            'beta_override': self.beta_override,
            'gate_gamma': self.gate_gamma,
            'gate_max': self.gate_max,
            'gate_decay': self.gate_decay,
            'pred_lr_gain': self.pred_lr_gain,
            'pred_lr_scale_max': self.pred_lr_scale_max,
            'd_model': self.d_model,
            'd_ff': self.d_ff,
            # state_dim removed — prototypes now use d_model like ConscienceColumn
            'prototypes': torch.from_numpy(self.prototypes),
            'usage': torch.from_numpy(self.usage),
            'theta': torch.from_numpy(self.theta),
            'last_won': torch.from_numpy(self.last_won.astype(np.float32)),
            '_reserved_mask': torch.from_numpy(self._reserved_mask.astype(np.uint8)),
            '_prev_prediction': torch.from_numpy(self._prev_prediction),
            '_surprise': torch.from_numpy(self._surprise),
            '_pred_err_ema': torch.from_numpy(self._pred_err_ema),
            '_gate': torch.from_numpy(self._gate),
        }
        state.update(self.save_rotation_state())
        for i, name in enumerate(self._param_names):
            state[name] = self._params[i].detach().cpu()
        state['optimizer_state'] = self._optimizer.state_dict()
        torch.save(state, os.path.join(output_dir, "column_states.pt"))
        print(f"  conscience_predictive_override column state saved to {output_dir}")

    def load_state(self, state, slot_map):
        def _to_np(t):
            return t.numpy() if hasattr(t, 'numpy') else np.array(t)

        self.slot_map = slot_map
        if state.get('_reserved_mask') is not None:
            self._reserved_mask = _to_np(state['_reserved_mask']).astype(bool)
        if 'tick_count' in state:
            self._tick_count = int(state['tick_count'])
        if 'prototypes' in state:
            self.prototypes = _to_np(state['prototypes']).astype(np.float32)
        if 'usage' in state:
            self.usage = _to_np(state['usage']).astype(np.float32)
        if 'theta' in state:
            self.theta = _to_np(state['theta']).astype(np.float32)
        if 'last_won' in state:
            self.last_won = _to_np(state['last_won']).astype(np.int64)
        if '_prev_prediction' in state:
            self._prev_prediction = _to_np(state['_prev_prediction']).astype(np.float32)
        if '_surprise' in state:
            self._surprise = _to_np(state['_surprise']).astype(np.float32)
        if '_pred_err_ema' in state:
            self._pred_err_ema = _to_np(state['_pred_err_ema']).astype(np.float32)
        if '_gate' in state:
            self._gate = _to_np(state['_gate']).astype(np.float32)

        for i, name in enumerate(self._param_names):
            if name in state:
                self._params[i].data.copy_(state[name])
        self.load_rotation_state(state)
        if 'optimizer_state' in state:
            self._optimizer.load_state_dict(state['optimizer_state'])
            for opt_state in self._optimizer.state.values():
                for k, v in opt_state.items():
                    if torch.is_tensor(v):
                        opt_state[k] = v.to(self._device)

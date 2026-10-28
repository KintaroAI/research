"""ctypes wrapper around libvmagent.so — the C++ visual–motor agent.

Python owns the world (world/circle_world.py) and the outer loop (train.py). Everything that learns
(weights, acting with sampling, rollout memory, GAE, PPO, AdamW, checkpoints) lives in C++; this
module only passes numpy arrays across the C ABI as raw pointers (zero copy).

The agent's inputs are exactly: the rendered image (uint8 HWC), the two drive values, the previous
executed action and (optionally) the previous reward. Nothing else from the world reaches the model.
"""
import ctypes as C
import os

import numpy as np

LIB_PATH = os.environ.get("VM_AGENT_LIB") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "libvmagent.so")

ACT_DETERMINISTIC = 1
ACT_PEEK = 2


class VMConfig(C.Structure):
    _fields_ = [(n, C.c_int) for n in ("img", "c1", "c2", "c3", "feat", "d_model", "n_layer", "n_head", "d_ff",
                                       "ctx", "pred_img", "pc1", "pc2", "pred_hidden", "use_prev_reward")]
    _fields_ += [("logstd_init", C.c_float), ("coord", C.c_int), ("pb", C.c_int), ("pred_residual", C.c_int),
                 ("pred_skip", C.c_int), ("inv_hidden", C.c_int), ("dyn_hidden", C.c_int), ("one_frame", C.c_int),
                 ("pi_hidden", C.c_int), ("patch", C.c_int), ("act_token", C.c_int), ("universal", C.c_int),
                 ("bins", C.c_int), ("gray", C.c_int), ("in_bins", C.c_int), ("attn_split", C.c_int),
                 ("attn_overlap", C.c_int), ("attn_pred_global", C.c_int), ("attn_cols", C.c_int),
                 ("attn_checker", C.c_int)]


class PPOConfig(C.Structure):
    _fields_ = [(n, C.c_float) for n in ("gamma", "lam", "clip_eps", "vf_coef", "ent_coef", "pred_coef",
                                         "drive_coef", "pred_backbone_scale", "max_grad_norm", "lr", "beta1",
                                         "beta2", "adam_eps", "weight_decay", "target_kl", "reward_scale",
                                         "vclip_eps", "pred_changed_weight", "inv_coef", "dyn_coef", "dyn_h_coef",
                                         "dyn_x_coef")]
    _fields_ += [(n, C.c_int) for n in ("epochs", "minibatches", "use_ppo", "norm_adv", "clip_vloss", "dyn_horizon",
                                        "dyn_h_horizon")]
    _fields_ += [("move_coef", C.c_float), ("pi_detach", C.c_int), ("pi_own_clip", C.c_int),
                ("drive_changed_weight", C.c_float), ("drive_change_eps", C.c_float), ("act_coef", C.c_float)]


class PPOStats(C.Structure):
    _fields_ = [(n, C.c_float) for n in ("pg_loss", "v_loss", "entropy", "approx_kl", "clipfrac", "explained_var",
                                         "pred_mse", "pred_changed_mse", "copy_mse", "copy_changed_mse",
                                         "changed_frac", "drive_mse", "grad_norm", "std_x", "std_y", "click_p",
                                         "value_mean", "return_mean", "adv_mean", "adv_std", "gn_backbone",
                                         "gn_policy", "gn_pred", "inv_mse", "inv_r2", "dyn_mse", "dyn_hunger_mse",
                                         "dyn_recon_mse")]
    _fields_ += [(n, C.c_int) for n in ("epochs_done", "updates_done")]
    _fields_ += [("move_nll", C.c_float), ("act_drive_mse", C.c_float), ("act_frame_mse", C.c_float)]
    _fields_ += [(n, C.c_float) for n in ("uni_ce_body", "uni_ce_patch", "uni_ce_move", "uni_mse_body",
                                         "uni_mse_patch", "uni_mse_move")]


DEFAULT_MODEL = dict(img=64, c1=16, c2=32, c3=32, feat=128, d_model=128, n_layer=2, n_head=4, d_ff=512, ctx=32,
                     pred_img=32, pc1=16, pc2=32, pred_hidden=256, use_prev_reward=0, logstd_init=-0.5, coord=0, pb=32,
                     pred_residual=0, pred_skip=0, inv_hidden=0, dyn_hidden=0, one_frame=0, pi_hidden=0, patch=0,
                     act_token=0, universal=0, bins=32, gray=0, in_bins=0, attn_split=0, attn_overlap=0, attn_pred_global=0, attn_cols=0,
                     attn_checker=0)

DEFAULT_PPO = dict(gamma=0.99, lam=0.95, clip_eps=0.2, vf_coef=0.5, ent_coef=0.01, pred_coef=0.0, drive_coef=0.0,
                   pred_backbone_scale=1.0, max_grad_norm=0.5, lr=2.5e-4, beta1=0.9, beta2=0.999, adam_eps=1e-5,
                   weight_decay=0.0, target_kl=0.0, reward_scale=0.1, vclip_eps=0.2, pred_changed_weight=0.0, inv_coef=0.0,
                   dyn_coef=0.0, dyn_h_coef=0.0, dyn_x_coef=0.0,
                   epochs=4, minibatches=4, use_ppo=1, norm_adv=1, clip_vloss=0, dyn_horizon=0, dyn_h_horizon=0,
                   move_coef=0.0, pi_detach=0, pi_own_clip=0, drive_changed_weight=0.0,
                   drive_change_eps=0.01, act_coef=0.0)


def _struct(cls, values):
    s = cls()
    for name, _ in cls._fields_:
        setattr(s, name, values[name])
    return s


def _as_dict(s):
    return {name: getattr(s, name) for name, _ in s._fields_}


def _load_lib(path=LIB_PATH):
    lib = C.CDLL(path)
    vp, i32, i64, f32, u64 = C.c_void_p, C.c_int, C.c_long, C.c_float, C.c_uint64
    sig = {
        "vm_create": (vp, [C.POINTER(VMConfig), u64]),
        "vm_load": (vp, [C.c_char_p]),
        "vm_save": (i32, [vp, C.c_char_p, i32]),
        "vm_free": (None, [vp]),
        "vm_num_params": (i64, [vp]),
        "vm_get_config": (None, [vp, C.POINTER(VMConfig)]),
        "vm_params": (C.POINTER(C.c_float), [vp]),
        "vm_grads": (C.POINTER(C.c_float), [vp]),
        "vm_param_layout": (i32, [vp, i32, C.c_char_p, i32, C.POINTER(C.c_long), C.POINTER(C.c_long)]),
        "vm_params_changed": (None, [vp]),
        "vm_set_threads": (None, [i32]),
        "vm_get_threads": (i32, []),
        "vm_ctx_create": (vp, [vp, i32, u64]),
        "vm_ctx_free": (None, [vp]),
        "vm_ctx_reset": (None, [vp, vp]),
        "vm_act": (None, [vp, vp, vp, vp, vp, vp, vp, i32, vp, vp, vp, vp, vp, vp, i32]),
        "vm_rollout_create": (vp, [vp, i32, i32]),
        "vm_rollout_free": (None, [vp]),
        "vm_rollout_outcome": (None, [vp, i32, vp, vp, vp, vp]),
        "vm_rollout_set_action": (None, [vp, i32, vp]),
        "vm_rollout_store": (None, [vp, i32, vp, vp, vp, vp, vp, vp]),
        "vm_update": (None, [vp, vp, vp, C.POINTER(PPOConfig), C.POINTER(PPOStats)]),
        "vm_debug_grads": (f32, [vp, vp, vp, C.POINTER(PPOConfig), vp, i32, C.POINTER(PPOStats)]),
        "vm_forward": (None, [vp, i32, i32, vp, vp, vp, vp, vp, vp, vp, vp, vp, vp, vp, vp]),
        "vm_forward_uni": (None, [vp, i32, i32, vp, vp, vp, vp, vp, vp, vp]),
        "vm_attention": (None, [vp, i32, vp, vp, vp, vp, vp, vp]),
        "vm_encode": (None, [vp, i32, vp, vp, vp]),
        "vm_inverse": (None, [vp, i32, vp, vp, vp]),
        "vm_imagine": (None, [vp, i32, i32, vp, vp, vp, vp, vp, vp]),
        "vm_decode": (None, [vp, i32, vp, vp]),
    }
    for name, (res, args) in sig.items():
        if not hasattr(lib, name):   # an older build without this function (e.g. one a running experiment loaded)
            continue
        fn = getattr(lib, name)
        fn.restype = res
        fn.argtypes = args
    return lib


_LIB = None


def lib():
    global _LIB
    if _LIB is None:
        _LIB = _load_lib()
    return _LIB


def _ptr(a, dtype):
    """Pointer to a C-contiguous array of the given dtype (None passes through as NULL)."""
    if a is None:
        return None
    assert isinstance(a, np.ndarray) and a.dtype == dtype and a.flags["C_CONTIGUOUS"], (a.dtype, dtype)
    return a.ctypes.data


def _arr(a, dtype):
    return None if a is None else np.ascontiguousarray(a, dtype=dtype)


def set_threads(n):
    lib().vm_set_threads(int(n))


def get_threads():
    return lib().vm_get_threads()


class ActContext:
    """KV caches + per-env sampling RNGs for n_envs parallel environments."""

    def __init__(self, agent, n_envs, seed):
        self.agent, self.n_envs = agent, n_envs
        self.handle = lib().vm_ctx_create(agent.handle, n_envs, seed)

    def reset(self, mask=None):
        """Restart contexts (mask: bool [N]; None = all envs and the block phase)."""
        m = None if mask is None else np.ascontiguousarray(mask, dtype=np.uint8)
        lib().vm_ctx_reset(self.handle, _ptr(m, np.uint8))

    def close(self):
        if getattr(self, "handle", None):
            lib().vm_ctx_free(self.handle)
            self.handle = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


class Rollout:
    """C++-side rollout memory for T steps x N envs (T must be a multiple of the context length)."""

    def __init__(self, agent, T, N):
        self.agent, self.T, self.N = agent, T, N
        self.handle = lib().vm_rollout_create(agent.handle, T, N)
        if not self.handle:
            raise ValueError("rollout length T=%d must be a multiple of ctx=%d" % (T, agent.config["ctx"]))

    def outcome(self, t, reward, done, next_images, next_drives):
        r = _arr(reward, np.float32)
        d = _arr(done, np.uint8)
        ni = _arr(next_images, np.uint8)
        nd = _arr(next_drives, np.float32)
        lib().vm_rollout_outcome(self.handle, t, _ptr(r, np.float32), _ptr(d, np.uint8), _ptr(ni, np.uint8),
                                 _ptr(nd, np.float32))

    def store(self, t, images, drives, prev_action, prev_reward, ep_start, action):
        """Write the inputs of step t for all N sequences (training from replayed experience)."""
        img = _arr(images, np.uint8)
        drv = _arr(drives, np.float32)
        pa = _arr(prev_action, np.float32)
        pr = _arr(prev_reward, np.float32)
        es = _arr(ep_start, np.uint8)
        act = _arr(action, np.float32)
        assert img.shape[0] == self.N and drv.shape == (self.N, 2) and pa.shape == (self.N, 3) and act.shape == (self.N, 3)
        lib().vm_rollout_store(self.handle, t, _ptr(img, np.uint8), _ptr(drv, np.float32), _ptr(pa, np.float32),
                               _ptr(pr, np.float32), _ptr(es, np.uint8), _ptr(act, np.float32))

    def set_action(self, t, action):
        """Replace the stored action of step t ([N,3] raw) by one chosen outside the policy (needs ppo.use_ppo=0)."""
        a = _arr(action, np.float32)
        assert a.shape == (self.N, 3)
        lib().vm_rollout_set_action(self.handle, t, _ptr(a, np.float32))

    def close(self):
        if getattr(self, "handle", None):
            lib().vm_rollout_free(self.handle)
            self.handle = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


class Agent:
    """Owns a C++ model. Create fresh (model_cfg, seed) or load from a checkpoint path."""

    def __init__(self, model_cfg=None, seed=0, path=None):
        if path is not None:
            self.handle = lib().vm_load(os.fsencode(path))
            if not self.handle:
                raise IOError("could not load checkpoint %s" % path)
        else:
            cfg = dict(DEFAULT_MODEL)
            cfg.update(model_cfg or {})
            c = _struct(VMConfig, cfg)
            self.handle = lib().vm_create(C.byref(c), seed)
            if not self.handle:
                raise ValueError("invalid model config %r" % cfg)
        c = VMConfig()
        lib().vm_get_config(self.handle, C.byref(c))
        self.config = _as_dict(c)
        self.num_params = lib().vm_num_params(self.handle)

    @classmethod
    def load(cls, path):
        return cls(path=path)

    def save(self, path, with_optimizer=True):
        if lib().vm_save(self.handle, os.fsencode(path), 1 if with_optimizer else 0) != 0:
            raise IOError("could not save checkpoint %s" % path)

    def params(self):
        """Live float32 view of the C++ parameter array (call params_changed() after writing to it)."""
        return np.ctypeslib.as_array(lib().vm_params(self.handle), shape=(self.num_params,))

    def grads(self):
        return np.ctypeslib.as_array(lib().vm_grads(self.handle), shape=(self.num_params,))

    def params_changed(self):
        lib().vm_params_changed(self.handle)

    def param_layout(self):
        out, i = [], 0
        name = C.create_string_buffer(64)
        off, size = C.c_long(), C.c_long()
        while lib().vm_param_layout(self.handle, i, name, 64, C.byref(off), C.byref(size)) == 0:
            out.append((name.value.decode(), off.value, size.value))
            i += 1
        return out

    def new_context(self, n_envs, seed=0):
        return ActContext(self, n_envs, seed)

    def new_rollout(self, T, N):
        return Rollout(self, T, N)

    def act(self, ctx, images, drives, prev_action, prev_reward, ep_start=None, deterministic=False, peek=False,
            rollout=None, t=0):
        """One step for all envs of ctx. Returns dict(action [N,3] raw, logp, value, mu [N,2], click_p)."""
        if self.config["patch"]:
            raise ValueError("act: the patch-token model acts through forward() on its context window")
        N = ctx.n_envs
        img = _arr(images, np.uint8)
        drv = _arr(drives, np.float32)
        pa = _arr(prev_action, np.float32)
        pr = _arr(prev_reward, np.float32)
        es = _arr(ep_start, np.uint8)
        assert img.shape[0] == N and drv.shape == (N, 2) and pa.shape == (N, 3) and pr.shape == (N,)
        out = dict(action=np.empty((N, 3), np.float32), logp=np.empty(N, np.float32),
                   value=np.empty(N, np.float32), mu=np.empty((N, 2), np.float32),
                   click_p=np.empty(N, np.float32))
        flags = (ACT_DETERMINISTIC if deterministic else 0) | (ACT_PEEK if peek else 0)
        lib().vm_act(self.handle, ctx.handle, _ptr(img, np.uint8), _ptr(drv, np.float32), _ptr(pa, np.float32),
                     _ptr(pr, np.float32), _ptr(es, np.uint8), flags, _ptr(out["action"], np.float32),
                     _ptr(out["logp"], np.float32), _ptr(out["value"], np.float32), _ptr(out["mu"], np.float32),
                     _ptr(out["click_p"], np.float32), rollout.handle if rollout is not None else None, t)
        return out

    def update(self, rollout, last_value, ppo_cfg):
        cfg = dict(DEFAULT_PPO)
        cfg.update(ppo_cfg or {})
        pc = _struct(PPOConfig, cfg)
        st = PPOStats()
        lv = _arr(last_value, np.float32)
        lib().vm_update(self.handle, rollout.handle, _ptr(lv, np.float32), C.byref(pc), C.byref(st))
        return _as_dict(st)

    def debug_grads(self, rollout, last_value, ppo_cfg, seq_ids):
        cfg = dict(DEFAULT_PPO)
        cfg.update(ppo_cfg or {})
        pc = _struct(PPOConfig, cfg)
        st = PPOStats()
        lv = _arr(last_value, np.float32)
        ids = np.ascontiguousarray(seq_ids, dtype=np.int32)
        loss = lib().vm_debug_grads(self.handle, rollout.handle, _ptr(lv, np.float32), C.byref(pc),
                                    ids.ctypes.data, len(ids), C.byref(st))
        return loss, _as_dict(st)

    def forward(self, images, drives, prev_action, prev_reward, ep_start=None, pred_actions=None):
        """Batched forward over S sequences of T steps ([S,T,...] inputs). Returns a dict of outputs."""
        img = _arr(images, np.uint8)
        S, T = img.shape[:2]
        drv = _arr(drives, np.float32)
        pa = _arr(prev_action, np.float32)
        pr = _arr(prev_reward, np.float32)
        es = _arr(ep_start, np.uint8)
        act = _arr(pred_actions, np.float32)
        d, p = self.config["d_model"], self.config["pred_img"]
        out = dict(z=np.empty((S, T, d), np.float32), value=np.empty((S, T), np.float32),
                   mu=np.empty((S, T, 2), np.float32), click_p=np.empty((S, T), np.float32))
        if act is not None:
            out["pred_frames"] = np.empty((S, T, p, p, 3), np.float32)
            out["pred_drives"] = np.empty((S, T, 2), np.float32)
        lib().vm_forward(self.handle, S, T, _ptr(img, np.uint8), _ptr(drv, np.float32), _ptr(pa, np.float32),
                         _ptr(pr, np.float32), _ptr(es, np.uint8), _ptr(act, np.float32),
                         _ptr(out["z"], np.float32), _ptr(out["value"], np.float32), _ptr(out["mu"], np.float32),
                         _ptr(out["click_p"], np.float32), _ptr(out.get("pred_frames"), np.float32),
                         _ptr(out.get("pred_drives"), np.float32))
        return out

    def forward_uni(self, images, drives, actions, ep_start=None, values=True, move_probs=True, next_body=False):
        """The universal model (model.universal=1) over S sequences of T steps: images u8 [S,T,H,W,3], drives [S,T,2],
        actions [S,T,3] (the move made at each step; the last one may be a candidate). Returns values [S, T, ntok, V]
        (every token's expected prediction of the next token's values; tokens of a step: body, patches, move) and
        move_probs [S, T, 3, bins] (the predicted move's bin probabilities: (dx+1)/2, (dy+1)/2, click) and next_body
        [S, T, 2] (the move tokens' expected prediction of the drives after their step)."""
        img = _arr(images, np.uint8)
        S, T = img.shape[:2]
        c = self.config
        nt, V, K = (c["img"] // c["patch"]) ** 2 + 2, c["patch"] ** 2 * (1 if c.get("gray") else 3), c["bins"]
        out = {}
        if values:
            out["values"] = np.empty((S, T, nt, V), np.float32)
        if move_probs:
            out["move_probs"] = np.empty((S, T, 3, K), np.float32)
        if next_body:
            out["next_body"] = np.empty((S, T, 2), np.float32)
        lib().vm_forward_uni(self.handle, S, T, _ptr(img, np.uint8), _ptr(_arr(drives, np.float32), np.float32),
                             _ptr(_arr(actions, np.float32), np.float32), _ptr(_arr(ep_start, np.uint8), np.uint8),
                             _ptr(out.get("values"), np.float32), _ptr(out.get("move_probs"), np.float32),
                             _ptr(out.get("next_body"), np.float32))
        return out

    def attention(self, images, drives, prev_action, prev_reward, ep_start=None):
        """Attention weights of one sequence of T steps ([T,...] inputs): [n_layer, n_head, Tt, Tt], Tt = T, or
        T * tokens per step with patch tokens (tokens of a step: the patches row by row, drives, move, command)."""
        img = _arr(images, np.uint8)
        T, c = img.shape[0], self.config
        tt = T * ((c["img"] // c["patch"]) ** 2 + 3 if c["patch"] else 1)
        att = np.empty((c["n_layer"], c["n_head"], tt, tt), np.float32)
        lib().vm_attention(self.handle, T, _ptr(img, np.uint8), _ptr(_arr(drives, np.float32), np.float32),
                           _ptr(_arr(prev_action, np.float32), np.float32), _ptr(_arr(prev_reward, np.float32), np.float32),
                           _ptr(_arr(ep_start, np.uint8), np.uint8), _ptr(att, np.float32))
        return att

    def encode(self, images):
        """Encoder features of [n,H,W,3] u8 frames: dict(a3 [n, c3*(img/8)^2], vf [n, feat])."""
        if self.config["patch"]:
            raise ValueError("encode: the patch-token model has no CNN encoder")
        img = _arr(images, np.uint8)
        n = img.shape[0]
        c = self.config
        a3 = np.empty((n, c["c3"] * (c["img"] // 8) ** 2), np.float32)
        vf = np.empty((n, c["feat"]), np.float32)
        lib().vm_encode(self.handle, n, _ptr(img, np.uint8), _ptr(a3, np.float32), _ptr(vf, np.float32))
        return dict(a3=a3, vf=vf)

    def inverse(self, img0, img1):
        """Inverse-dynamics prediction of the executed (dx, dy) for frame pairs [n,H,W,3] u8 -> [n,2]."""
        a0 = _arr(img0, np.uint8)
        a1 = _arr(img1, np.uint8)
        out = np.empty((a0.shape[0], 2), np.float32)
        lib().vm_inverse(self.handle, a0.shape[0], _ptr(a0, np.uint8), _ptr(a1, np.uint8), _ptr(out, np.float32))
        return out

    def imagine(self, s0, h0, actions, latents=False, frames=False):
        """Latent-dynamics rollouts (dyn_hidden > 0): s0 [n,feat] visual features, h0 [n] hunger, actions [n,H,3] raw.
        Returns dict(hunger [n,H], optional latents [n,H,feat], frames [n,H,r,r,3] with r = pred_img // 2)."""
        s0 = _arr(s0, np.float32)
        h0 = _arr(h0, np.float32)
        act = _arr(actions, np.float32)
        n, H = act.shape[:2]
        c = self.config
        r = c["pred_img"] // 2
        out = dict(hunger=np.empty((n, H), np.float32))
        if latents:
            out["latents"] = np.empty((n, H, c["feat"]), np.float32)
        if frames:
            out["frames"] = np.empty((n, H, r, r, 3), np.float32)
        lib().vm_imagine(self.handle, n, H, _ptr(s0, np.float32), _ptr(h0, np.float32), _ptr(act, np.float32),
                         _ptr(out["hunger"], np.float32), _ptr(out.get("latents"), np.float32),
                         _ptr(out.get("frames"), np.float32))
        return out

    def decode(self, latents):
        """Coarse frames [n,r,r,3] (float) decoded from latents [n,feat] (dyn_hidden > 0)."""
        s = _arr(latents, np.float32)
        r = self.config["pred_img"] // 2
        out = np.empty((s.shape[0], r, r, 3), np.float32)
        lib().vm_decode(self.handle, s.shape[0], _ptr(s, np.float32), _ptr(out, np.float32))
        return out

    def close(self):
        if getattr(self, "handle", None):
            lib().vm_free(self.handle)
            self.handle = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

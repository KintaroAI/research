"""ctypes wrapper of libvmcuda.so: the basic agent's update on the GPU (task 00011, src/vm_cuda.cu).

The model is created from a C++ agent (agent_lib.Agent): the same config and flat parameter layout, copied to the
device. A batch is S sequences of ctx steps as [S, ctx, ...] arrays (as tests/test_model_reference.seq_batch builds
them); prev_reward is the command slot.

    m = CudaModel(agent)
    m.set_batch(images, drives, prev_action, prev_reward, ep_start, action, next_small, next_drives)
    stats = m.update(ppo_cfg)          # forward, backward, clip, AdamW on the device
    params = m.params()                # back to the host (agent.params()[:] = params; agent.params_changed())

online.py (online.device=cuda) runs the whole lifetime on the GPU: it acts and computes the surprise with
forward() (the outputs of agent_lib.Agent.forward), fills a cuda_lib.Rollout like the C++ one and updates with
update_rollout(). After an update only the log-std goes back to the C++ agent (moves are sampled with it);
sync_to(agent) copies all the weights before a checkpoint. The optimizer state stays on the device.
"""
import ctypes as C
import os

import numpy as np

import agent_lib
from agent_lib import PPOConfig, VMConfig, DEFAULT_PPO, _struct

LIB_PATH = os.environ.get("VM_CUDA_LIB") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "libvmcuda.so")


class VMCStats(C.Structure):
    _fields_ = [(n, C.c_float) for n in ("loss", "pred_mse", "drive_mse", "move_nll", "click_p", "pred_changed_mse",
                                         "grad_norm", "gn_backbone", "gn_policy", "gn_pred", "inv_mse", "inv_r2",
                                         "dyn_mse", "dyn_hunger_mse", "dyn_recon_mse", "act_drive_mse",
                                         "act_frame_mse", "uni_ce_body", "uni_ce_patch", "uni_ce_move",
                                         "uni_mse_body", "uni_mse_patch", "uni_mse_move")]


_lib = None


def lib():
    global _lib
    if _lib is None:
        _lib = C.CDLL(LIB_PATH)
        vp, i32, fp, u8p = C.c_void_p, C.c_int, C.POINTER(C.c_float), C.POINTER(C.c_uint8)
        lp = C.POINTER(C.c_long)
        sig = {
            "vmc_create": (vp, [C.POINTER(VMConfig), fp, C.c_long, lp, lp, i32]),
            "vmc_free": (None, [vp]),
            "vmc_set_batch": (i32, [vp, i32, u8p, fp, fp, fp, u8p, fp, u8p, fp, i32]),
            "vmc_set_uni_inputs": (i32, [vp, i32, u8p, fp]),
            "vmc_loss_and_grads": (i32, [vp, C.POINTER(PPOConfig)]),
            "vmc_update": (i32, [vp, C.POINTER(PPOConfig)]),
            "vmc_stats": (i32, [vp, C.POINTER(VMCStats)]),
            "vmc_params": (i32, [vp, fp]),
            "vmc_grads": (i32, [vp, fp]),
            "vmc_set_params": (i32, [vp, fp]),
            "vmc_sync": (i32, [vp]),
            "vmc_forward": (i32, [vp, i32, i32, u8p, fp, fp, fp, u8p, fp, fp, fp, fp, fp, fp, fp]),
            "vmc_forward_uni": (i32, [vp, i32, i32, u8p, fp, fp, u8p, fp, fp, fp]),
            "vmc_uni_candidates": (i32, [vp, i32, fp, fp]),
            "vmc_uni_token_ce": (i32, [vp, i32, fp]),
            "vmc_forward_uni_step": (i32, [vp, i32, i32, u8p, fp, fp, u8p, i32, i32, fp, fp]),
            "vmc_get_range": (i32, [vp, C.c_long, C.c_long, fp]),
            "vmc_encode": (i32, [vp, i32, u8p, fp]),
            "vmc_imagine": (i32, [vp, i32, i32, fp, fp, fp, fp, fp, fp]),
            "vmc_decode": (i32, [vp, i32, fp, fp]),
        }
        for name, (res, args) in sig.items():
            if not hasattr(_lib, name):   # an older build without this function (e.g. one a running job loaded)
                continue
            f = getattr(_lib, name)
            f.restype, f.argtypes = res, args
    return _lib


def available():
    """True if libvmcuda.so loads (a GPU is needed to create a model)."""
    try:
        lib()
        return True
    except OSError:
        return False


class Rollout:
    """The store / outcome interface of agent_lib's rollout ([T, N]: N sequences of T = ctx steps), kept in numpy for
    the GPU update. The next frames are kept full size; the GPU downsamples them as the C++ rollout does."""

    def __init__(self, T, N, img, next_frames=True):
        self.T, self.N = T, N
        self.img = np.zeros((T, N, img, img, 3), np.uint8)
        # the next frames (next_frames=False: not kept, e.g. for the universal model, whose targets are in the window)
        self.nimg = np.zeros((T, N, img, img, 3), np.uint8) if next_frames else None
        self.drv, self.ndrv = np.zeros((T, N, 2), np.float32), np.zeros((T, N, 2), np.float32)
        self.pa, self.act = np.zeros((T, N, 3), np.float32), np.zeros((T, N, 3), np.float32)
        self.pr = np.zeros((T, N), np.float32)
        self.es = np.zeros((T, N), np.uint8)

    def store(self, t, images, drives, prev_action, prev_reward, ep_start, action):
        self.img[t], self.drv[t], self.pa[t], self.pr[t], self.es[t], self.act[t] = (
            images, drives, prev_action, prev_reward, ep_start, action)

    def outcome(self, t, reward, done, next_images, next_drives):
        if self.nimg is not None:
            self.nimg[t] = next_images
        self.ndrv[t] = next_drives

    def batch(self):
        return [None if a is None else np.ascontiguousarray(a.swapaxes(0, 1))
                for a in (self.img, self.drv, self.pa, self.pr, self.es, self.act, self.nimg, self.ndrv)]


def has(name):
    """Whether the loaded libvmcuda.so has a function (older builds lack the newer ones)."""
    return hasattr(lib(), name)


def _p(a, ct):
    return a.ctypes.data_as(C.POINTER(ct))


def _check(rc, what):
    if rc != 0:
        raise RuntimeError("vm_cuda: %s failed" % what)


class CudaModel:
    def __init__(self, agent):
        cfg = _struct(VMConfig, agent.config)
        layout = agent.param_layout()
        self.n_params = agent.num_params
        params = np.ascontiguousarray(agent.params(), np.float32)
        offs = np.array([o for _, o, _ in layout], np.int64)
        sizes = np.array([s for _, _, s in layout], np.int64)
        self.handle = lib().vmc_create(C.byref(cfg), _p(params, C.c_float), self.n_params, _p(offs, C.c_long),
                                       _p(sizes, C.c_long), len(layout))
        if not self.handle:
            raise ValueError("vm_cuda does not support this model config")
        self.ctx = agent.config["ctx"]
        self.config = dict(agent.config)
        self.logstd_off = {name: o for name, o, _ in layout}["logstd"]

    def set_batch(self, images, drives, prev_action, prev_reward, ep_start, action, next_small, next_drives):
        """next_small: [S, ctx, pred_img, pred_img, 3], or the full-size next frames (downsampled on the GPU)."""
        a = [None if x is None else np.ascontiguousarray(x, t) for x, t in ((images, np.uint8), (drives, np.float32),
                                                     (prev_action, np.float32), (prev_reward, np.float32),
                                                     (ep_start, np.uint8), (action, np.float32),
                                                     (next_small, np.uint8), (next_drives, np.float32))]
        S, T = a[0].shape[:2]
        if T != self.ctx:
            raise ValueError("sequences must be ctx = %d steps" % self.ctx)
        cts = (C.c_uint8, C.c_float, C.c_float, C.c_float, C.c_uint8, C.c_float, C.c_uint8, C.c_float)
        self._batch = a   # keep alive until copied
        full = int(next_small is not None and a[6].shape[2] == a[0].shape[2])
        _check(lib().vmc_set_batch(self.handle, S, *[None if x is None else _p(x, ct) for x, ct in zip(a, cts)], full),
               "set_batch")

    def set_uni_inputs(self, images, drives):
        """The universal model's token inputs (frames [S, ctx, H, W, 3], bodies [S, ctx, 2]) for the current batch,
        instead of its own; the targets stay the batch's (self-forcing, task 00039). Cleared by the next set_batch."""
        img, drv = np.ascontiguousarray(images, np.uint8), np.ascontiguousarray(drives, np.float32)
        self._inputs = (img, drv)
        _check(lib().vmc_set_uni_inputs(self.handle, img.shape[0], _p(img, C.c_uint8), _p(drv, C.c_float)),
               "set_uni_inputs")

    def _pc(self, ppo_cfg):
        cfg = dict(DEFAULT_PPO)
        cfg.update(ppo_cfg or {})
        return _struct(PPOConfig, cfg)

    def loss_and_grads(self, ppo_cfg):
        _check(lib().vmc_loss_and_grads(self.handle, C.byref(self._pc(ppo_cfg))), "loss_and_grads")
        return self.stats()

    def update(self, ppo_cfg, stats=True):
        _check(lib().vmc_update(self.handle, C.byref(self._pc(ppo_cfg))), "update")
        return self.stats() if stats else None

    def update_rollout(self, ro, ppo_cfg, agent, sync=True, stats=True, inputs=None):
        """One update from a filled Rollout. sync: all the weights are copied back to the C++ agent; else only the
        log-std (for sampling moves; call sync_to(agent) before saving it). inputs: (frames, bodies) [N, T, ...] as the
        universal model's token inputs instead of the rollout's (set_uni_inputs). Returns the stats under
        agent.update()'s names."""
        self.set_batch(*ro.batch())
        if inputs is not None:
            self.set_uni_inputs(*inputs)
        st = self.update(ppo_cfg, stats=stats)   # stats=False: no wait for the GPU here
        p = agent.params()
        if sync:
            self.sync_to(agent)
        elif agent.config.get("universal"):   # no policy log-std to sample with
            return st
        else:
            ls = np.empty(2, np.float32)
            _check(lib().vmc_get_range(self.handle, self.logstd_off, 2, _p(ls, C.c_float)), "get_range")
            p[self.logstd_off:self.logstd_off + 2] = ls
        if st is not None:
            st.update(std_x=float(np.exp(np.clip(p[self.logstd_off], -5.0, 1.0))))
        return st

    def sync_to(self, agent):
        """Copies all the weights to the C++ agent."""
        agent.params()[:] = self.params()
        agent.params_changed()

    def encode(self, images):
        """agent_lib.Agent.encode on the GPU (v only): dict(vf [n, feat])."""
        img = np.ascontiguousarray(images, np.uint8)
        vf = np.empty((img.shape[0], self.config["feat"]), np.float32)
        _check(lib().vmc_encode(self.handle, img.shape[0], _p(img, C.c_uint8), _p(vf, C.c_float)), "encode")
        return dict(vf=vf)

    def imagine(self, s0, h0, actions, latents=False, frames=False):
        """agent_lib.Agent.imagine on the GPU: latent-dynamics rollouts from s0 [n, feat], h0 [n], actions [n, H, 3]."""
        s0 = np.ascontiguousarray(s0, np.float32)
        h0 = np.ascontiguousarray(h0, np.float32)
        act = np.ascontiguousarray(actions, np.float32)
        n, H = act.shape[:2]
        c = self.config
        r = c["pred_img"] // 2
        out = dict(hunger=np.empty((n, H), np.float32))
        if latents:
            out["latents"] = np.empty((n, H, c["feat"]), np.float32)
        if frames:
            out["frames"] = np.empty((n, H, r, r, 3), np.float32)
        opt = lambda a: None if a is None else _p(a, C.c_float)   # noqa: E731
        _check(lib().vmc_imagine(self.handle, n, H, _p(s0, C.c_float), _p(h0, C.c_float), _p(act, C.c_float),
                                 _p(out["hunger"], C.c_float), opt(out.get("latents")), opt(out.get("frames"))),
               "imagine")
        return out

    def decode(self, latents):
        """agent_lib.Agent.decode on the GPU: coarse frames [n, r, r, 3] from latents [n, feat]."""
        s = np.ascontiguousarray(latents, np.float32)
        r = self.config["pred_img"] // 2
        out = np.empty((s.shape[0], r, r, 3), np.float32)
        _check(lib().vmc_decode(self.handle, s.shape[0], _p(s, C.c_float), _p(out, C.c_float)), "decode")
        return out

    def forward(self, images, drives, prev_action, prev_reward, ep_start=None, pred_actions=None):
        """agent_lib.Agent.forward on the GPU: S sequences of T <= ctx steps ([S, T, ...] inputs)."""
        img = np.ascontiguousarray(images, np.uint8)
        S, T = img.shape[:2]
        drv, pa = np.ascontiguousarray(drives, np.float32), np.ascontiguousarray(prev_action, np.float32)
        pr = np.ascontiguousarray(prev_reward, np.float32)
        es = None if ep_start is None else np.ascontiguousarray(ep_start, np.uint8)
        act = None if pred_actions is None else np.ascontiguousarray(pred_actions, np.float32)
        d, p = self.config["d_model"], self.config["pred_img"]
        out = dict(z=np.empty((S, T, d), np.float32), value=np.empty((S, T), np.float32),
                   mu=np.empty((S, T, 2), np.float32), click_p=np.empty((S, T), np.float32))
        if act is not None:
            out["pred_frames"] = np.empty((S, T, p, p, 3), np.float32)
            out["pred_drives"] = np.empty((S, T, 2), np.float32)
        opt = lambda a, ct: None if a is None else _p(a, ct)   # noqa: E731
        _check(lib().vmc_forward(self.handle, S, T, _p(img, C.c_uint8), _p(drv, C.c_float), _p(pa, C.c_float),
                                 _p(pr, C.c_float), opt(es, C.c_uint8), opt(act, C.c_float),
                                 _p(out["z"], C.c_float), _p(out["value"], C.c_float), _p(out["mu"], C.c_float),
                                 _p(out["click_p"], C.c_float), opt(out.get("pred_frames"), C.c_float),
                                 opt(out.get("pred_drives"), C.c_float)), "forward")
        return out

    def forward_uni(self, images, drives, actions, ep_start=None, values=True, move_probs=True, next_body=False):
        """agent_lib.Agent.forward_uni on the GPU (the universal model)."""
        img = np.ascontiguousarray(images, np.uint8)
        S, T = img.shape[:2]
        c = self.config
        nt, V, K = (c["img"] // c["patch"]) ** 2 + 2, c["patch"] ** 2 * (1 if c.get("gray") else 3), c["bins"]
        drv, act = np.ascontiguousarray(drives, np.float32), np.ascontiguousarray(actions, np.float32)
        es = None if ep_start is None else np.ascontiguousarray(ep_start, np.uint8)
        out = {}
        if values:
            out["values"] = np.empty((S, T, nt, V), np.float32)
        if move_probs:
            out["move_probs"] = np.empty((S, T, 3, K), np.float32)
        if next_body:
            out["next_body"] = np.empty((S, T, 2), np.float32)
        opt = lambda a, ct: None if a is None else _p(a, ct)   # noqa: E731
        _check(lib().vmc_forward_uni(self.handle, S, T, _p(img, C.c_uint8), _p(drv, C.c_float), _p(act, C.c_float),
                                     opt(es, C.c_uint8), opt(out.get("values"), C.c_float),
                                     opt(out.get("move_probs"), C.c_float), opt(out.get("next_body"), C.c_float)),
               "forward_uni")
        return out

    def forward_uni_step(self, images, drives, actions, ep_start, t, mode=False):
        """forward_uni's values of step t only, values_t [S, ntok, V], and next_body [S, T, 2] (task 00029). mode: 0
        (False) each value's expectation, 1 (True) its most probable bin's centre, 2 ("pair", task 00038) the most
        probable bin and its more probable neighbour, weighted by their probabilities."""
        img = np.ascontiguousarray(images, np.uint8)
        S, T = img.shape[:2]
        c = self.config
        nt, V = (c["img"] // c["patch"]) ** 2 + 2, c["patch"] ** 2 * (1 if c.get("gray") else 3)
        vt, nb = np.empty((S, nt, V), np.float32), np.empty((S, T, 2), np.float32)
        es = None if ep_start is None else np.ascontiguousarray(ep_start, np.uint8)
        _check(lib().vmc_forward_uni_step(self.handle, S, T, _p(img, C.c_uint8),
                                          _p(np.ascontiguousarray(drives, np.float32), C.c_float),
                                          _p(np.ascontiguousarray(actions, np.float32), C.c_float),
                                          None if es is None else _p(es, C.c_uint8), t, int(mode), _p(vt, C.c_float),
                                          _p(nb, C.c_float)), "forward_uni_step")
        return dict(values_t=vt, next_body=nb)

    @property
    def uni_candidates(self):
        """The instinct's candidates, right after forward_uni of S windows: cand [S, Mc, 3] -> predicted next hunger
        [S, Mc] (only the move token is computed; the forward pass's keys and values are reused). Absent (an
        AttributeError) with an older libvmcuda.so, so callers fall back to full passes."""
        if not hasattr(lib(), "vmc_uni_candidates"):
            raise AttributeError("uni_candidates")

        def run(cand):
            c = np.ascontiguousarray(cand, np.float32)
            out = np.empty(c.shape[:2], np.float32)
            _check(lib().vmc_uni_candidates(self.handle, c.shape[1], _p(c, C.c_float), _p(out, C.c_float)),
                   "uni_candidates")
            return out
        return run

    def uni_token_ce(self, S0):
        """The universal model's last update: each token's cross-entropy at the last step of the batch's first S0
        sequences [S0, ntok] (token k's prediction of token k + 1; the move token's of the next body)."""
        c = self.config
        out = np.empty((S0, (c["img"] // c["patch"]) ** 2 + 2), np.float32)
        _check(lib().vmc_uni_token_ce(self.handle, S0, _p(out, C.c_float)), "uni_token_ce")
        return out

    def stats(self):
        st = VMCStats()
        _check(lib().vmc_stats(self.handle, C.byref(st)), "stats")
        return {n: getattr(st, n) for n, _ in VMCStats._fields_}

    def params(self):
        out = np.empty(self.n_params, np.float32)
        _check(lib().vmc_params(self.handle, _p(out, C.c_float)), "params")
        return out

    def grads(self):
        out = np.empty(self.n_params, np.float32)
        _check(lib().vmc_grads(self.handle, _p(out, C.c_float)), "grads")
        return out

    def set_params(self, params):
        p = np.ascontiguousarray(params, np.float32)
        _check(lib().vmc_set_params(self.handle, _p(p, C.c_float)), "set_params")

    def sync(self):
        _check(lib().vmc_sync(self.handle), "sync")

    def close(self):
        if self.handle:
            lib().vmc_free(self.handle)
            self.handle = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

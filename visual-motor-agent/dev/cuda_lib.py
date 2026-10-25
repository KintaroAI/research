"""ctypes wrapper of libvmcuda.so: the basic agent's update on the GPU (task 00011, src/vm_cuda.cu).

The model is created from a C++ agent (agent_lib.Agent): the same config and flat parameter layout, copied to the
device. A batch is S sequences of ctx steps as [S, ctx, ...] arrays (as tests/test_model_reference.seq_batch builds
them); prev_reward is the command slot.

    m = CudaModel(agent)
    m.set_batch(images, drives, prev_action, prev_reward, ep_start, action, next_small, next_drives)
    stats = m.update(ppo_cfg)          # forward, backward, clip, AdamW on the device
    params = m.params()                # back to the host (agent.params()[:] = params; agent.params_changed())

online.py (online.device=cuda) fills a cuda_lib.Rollout like the C++ one, updates with update_rollout() and copies
the weights back to the C++ agent, which keeps acting on the CPU. The optimizer state stays on the device.
"""
import ctypes as C
import os

import numpy as np

import agent_lib
from agent_lib import PPOConfig, VMConfig, DEFAULT_PPO, _struct

LIB_PATH = os.environ.get("VM_CUDA_LIB") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "libvmcuda.so")


class VMCStats(C.Structure):
    _fields_ = [(n, C.c_float) for n in ("loss", "pred_mse", "drive_mse", "move_nll", "click_p", "pred_changed_mse",
                                         "grad_norm", "gn_backbone", "gn_policy", "gn_pred")]


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
            "vmc_loss_and_grads": (i32, [vp, C.POINTER(PPOConfig)]),
            "vmc_update": (i32, [vp, C.POINTER(PPOConfig)]),
            "vmc_stats": (i32, [vp, C.POINTER(VMCStats)]),
            "vmc_params": (i32, [vp, fp]),
            "vmc_grads": (i32, [vp, fp]),
            "vmc_set_params": (i32, [vp, fp]),
            "vmc_sync": (i32, [vp]),
        }
        for name, (res, args) in sig.items():
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

    def __init__(self, T, N, img):
        self.T, self.N = T, N
        self.img = np.zeros((T, N, img, img, 3), np.uint8)
        self.nimg = np.zeros((T, N, img, img, 3), np.uint8)
        self.drv, self.ndrv = np.zeros((T, N, 2), np.float32), np.zeros((T, N, 2), np.float32)
        self.pa, self.act = np.zeros((T, N, 3), np.float32), np.zeros((T, N, 3), np.float32)
        self.pr = np.zeros((T, N), np.float32)
        self.es = np.zeros((T, N), np.uint8)

    def store(self, t, images, drives, prev_action, prev_reward, ep_start, action):
        self.img[t], self.drv[t], self.pa[t], self.pr[t], self.es[t], self.act[t] = (
            images, drives, prev_action, prev_reward, ep_start, action)

    def outcome(self, t, reward, done, next_images, next_drives):
        self.nimg[t], self.ndrv[t] = next_images, next_drives

    def batch(self):
        return [np.ascontiguousarray(a.swapaxes(0, 1)) for a in (self.img, self.drv, self.pa, self.pr, self.es,
                                                                 self.act, self.nimg, self.ndrv)]


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

    def set_batch(self, images, drives, prev_action, prev_reward, ep_start, action, next_small, next_drives):
        """next_small: [S, ctx, pred_img, pred_img, 3], or the full-size next frames (downsampled on the GPU)."""
        a = [np.ascontiguousarray(x, t) for x, t in ((images, np.uint8), (drives, np.float32),
                                                     (prev_action, np.float32), (prev_reward, np.float32),
                                                     (ep_start, np.uint8), (action, np.float32),
                                                     (next_small, np.uint8), (next_drives, np.float32))]
        S, T = a[0].shape[:2]
        if T != self.ctx:
            raise ValueError("sequences must be ctx = %d steps" % self.ctx)
        cts = (C.c_uint8, C.c_float, C.c_float, C.c_float, C.c_uint8, C.c_float, C.c_uint8, C.c_float)
        self._batch = a   # keep alive until copied
        full = int(a[6].shape[2] == a[0].shape[2])
        _check(lib().vmc_set_batch(self.handle, S, *[_p(x, ct) for x, ct in zip(a, cts)], full), "set_batch")

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

    def update_rollout(self, ro, ppo_cfg, agent):
        """One update from a filled Rollout; the weights are copied back to the C++ agent. Returns the stats under
        agent.update()'s names (inverse and latent dynamics: 0, as the C++ model reports them when absent)."""
        self.set_batch(*ro.batch())
        st = self.update(ppo_cfg)
        p = agent.params()
        p[:] = self.params()
        agent.params_changed()
        off = {name: o for name, o, _ in agent.param_layout()}["logstd"]
        st.update(inv_mse=0.0, inv_r2=0.0, dyn_mse=0.0, dyn_hunger_mse=0.0, dyn_recon_mse=0.0,
                  std_x=float(np.exp(np.clip(p[off], -5.0, 1.0))))
        return st

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

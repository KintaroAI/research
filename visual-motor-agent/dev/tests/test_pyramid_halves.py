"""Halves that share the summaries (task 00055, model.attn_split with model.uni_pyr): a patch attends to the patches of
its own band and to every summary, the body and the move; a summary is in every band its block spans. The table (C++
and reference), where information flows, C++ and GPU against the PyTorch reference, invalid configs.

Requires libvmagent.so and torch (CPU) in the venv; the GPU tests libvmcuda.so and a GPU.
"""
import os
import sys

import numpy as np
import pytest

torch = pytest.importorskip("torch")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import agent_lib  # noqa: E402
import cuda_lib  # noqa: E402
from reference import torch_reference as ref  # noqa: E402
from test_model_reference import SMALL, compare_grads, seq_batch  # noqa: E402
from test_patch_tokens import store_rollout  # noqa: E402

HAVE_GPU = cuda_lib.available()
UNI = dict(SMALL, patch=8, universal=1, bins=8, gray=1, in_bins=1)   # 32 x 32 frames: a 4 x 4 grid
HALVES = dict(UNI, uni_pyr=2, attn_split=2, attn_cols=1)


@pytest.mark.parametrize("cols", [1, 0])
def test_table(cols):
    cfg = dict(UNI, img=64, uni_pyr=3, attn_split=2, attn_cols=cols)    # 8 x 8 grid, B1-B3; B3 spans both halves
    a, t = ref.attn_allow(cfg), ref.uni_slot_table(cfg)
    kind, lev = t["kind"], t["level"]
    pos = t["c"] if cols else t["r"]
    half = lambda k: pos[k] * (2 ** lev[k] if kind[k] == 2 else 1) // 4   # noqa: E731
    for q in range(len(kind)):
        for k in range(len(kind)):
            if kind[k] != 1:
                want = True                                             # every summary, the body, the move
            elif kind[q] == 1:
                want = half(q) == half(k)                               # patches: within a half
            elif kind[q] == 2:
                want = lev[q] == 3 or half(q) == half(k)                # B3 spans both, B1, B2 one half
            else:
                want = True
            assert a[q, k] == want, (q, k)
    lay = agent_lib.uni_layout(cfg)
    assert lay["nt"] - 2 == lay["summary"][-1] and lev[lay["nt"] - 2] == 3   # the move's predictor reads all


def run(model, cfg, img, drv, act, es):
    return model.forward_uni(img, drv, act, es, values=True, move_probs=False)["values"]


def batch(seed=0, S=2, T=3):
    rng = np.random.default_rng(seed)
    img = rng.integers(0, 256, (S, T, 32, 32, 3), dtype=np.uint8)
    return img, rng.random((S, T, 2)).astype(np.float32), rng.uniform(-1, 1, (S, T, 3)).astype(np.float32), \
        np.zeros((S, T), np.uint8)


@pytest.mark.parametrize("dev", ["cpu", "gpu"])
def test_flow(dev):
    """Changing right patch (0, 2) of the last step: the left patches of row 1 come before any summary that reads it,
    so with the halves they stay the same (and change without them); B1(0, 1), after patch (1, 3), changes."""
    if dev == "gpu" and not HAVE_GPU:
        pytest.skip("needs libvmcuda.so and a GPU")
    img, drv, act, es = batch()
    far = img.copy()
    far[:, -1, 0:8, 16:24] = 255 - far[:, -1, 0:8, 16:24]
    for cfg, same in ((HALVES, True), (dict(HALVES, attn_split=0, attn_cols=0), False)):
        agent = agent_lib.Agent(cfg, seed=3)
        m = agent if dev == "cpu" else cuda_lib.CudaModel(agent)
        lay, t = agent_lib.uni_layout(cfg), ref.uni_slot_table(cfg)
        left1 = [k for k in lay["patch"] if t["r"][k] == 1 and t["c"][k] < 2]
        b1 = [k for k in lay["summary"] if t["level"][k] == 1 and t["r"][k] == 0 and t["c"][k] == 1]
        a, b = run(m, cfg, img, drv, act, es)[:, -1], run(m, cfg, far, drv, act, es)[:, -1]
        d = np.abs(a - b).max(axis=(0, 2))                               # per slot
        assert (d[left1].max() < 1e-7) == same, d[left1]
        assert d[b1].max() > 1e-6
        if dev == "gpu":
            m.close()
        agent.close()


@pytest.mark.parametrize("cfg", [HALVES, dict(HALVES, attn_cols=0), dict(HALVES, uni_pyr=1, sum_thumb=1),
                                 dict(HALVES, attn_split=4)], ids=["cols_L2", "rows_L2", "cols_L1_thumbs", "quarters"])
def test_cpu_loss_and_grads_match_reference(cfg):
    agent_lib.set_threads(4)
    agent = agent_lib.Agent(cfg, seed=7)
    T, N = 16, 3
    ro, data = store_rollout(agent, T=T, N=N, seed=3, p_done=0.2)
    ids = [0, 2, 4, 5]
    pc = dict(agent_lib.DEFAULT_PPO, use_ppo=0)
    loss_cpp, _ = agent.debug_grads(ro, np.zeros(N, np.float32), pc, ids)
    c = agent.config
    P = ref.params_from_flat(agent.params(), agent.param_layout(), c)
    z = np.zeros((T, N))
    loss_ref, _ = ref.ppo_loss(P, c, pc, seq_batch(data, ids, c["ctx"], N, z, z, c["img"] // c["pred_img"]))
    loss_ref.backward()
    assert abs(loss_cpp - loss_ref.item()) < 1e-4 * max(1.0, abs(loss_ref.item())), (loss_cpp, loss_ref.item())
    compare_grads(agent, P)
    agent.close()


@pytest.mark.skipif(not HAVE_GPU, reason="needs libvmcuda.so and a GPU")
@pytest.mark.parametrize("cfg", [HALVES, dict(HALVES, uni_pyr=1, sum_thumb=1)], ids=["cols_L2", "cols_L1_thumbs"])
def test_gpu_loss_and_grads_match_reference(cfg):
    from test_cuda_model import PC, _ref_flat_grads, compare, cuda_batch
    agent = agent_lib.Agent(cfg, seed=4)
    c = agent.config
    N, T = 4, 2 * c["ctx"]
    ro, data = store_rollout(agent, T=T, N=N, seed=2, p_done=0.12)
    ids = [0, 2, 3, 5, 6]
    m = cuda_lib.CudaModel(agent)
    m.set_batch(*cuda_batch(agent, data, ids, N))
    st, g = m.loss_and_grads(PC), m.grads().copy()
    z = np.zeros((T, N))
    want_l, want_g = _ref_flat_grads(agent, seq_batch(data, ids, c["ctx"], N, z, z, c["img"] // c["pred_img"]))
    assert abs(st["loss"] - want_l) < 1e-4 * max(1.0, abs(want_l)), (st["loss"], want_l)
    compare(agent, want_g, g, 1e-4)
    m.close()
    agent.close()


@pytest.mark.parametrize("bad", [dict(attn_split=3), dict(attn_overlap=1), dict(attn_checker=1),
                                 dict(attn_pred_global=1), dict(attn_win=1), dict(attn_split=8)],
                         ids=["not_dividing", "overlap", "checker", "pred_global", "window", "past_grid"])
def test_invalid_configs(bad):
    with pytest.raises(ValueError):
        agent_lib.Agent(dict(HALVES, **bad), seed=1)

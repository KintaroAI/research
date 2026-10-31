"""The age key (task 00062, model.age_key): position by age, read at attention, instead of a step embedding in the
tokens. Forward, loss and gradients against the PyTorch reference (C++ and GPU), the candidates against full passes,
a token's K and V independent of the window, checkpoints, invalid configs.

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
UNI = dict(SMALL, patch=8, universal=1, bins=8, gray=1, in_bins=1, age_key=1)
PYR = dict(UNI, uni_pyr=2, sum_thumb=1)
BAND = dict(PYR, band_n=2, band_k=1, band_cols=1)
CFGS = [UNI, PYR, BAND]
IDS = ["plain", "pyramid", "band"]


def batch(seed=0, S=2, T=3, img=32):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, (S, T, img, img, 3), dtype=np.uint8), rng.random((S, T, 2)).astype(np.float32), \
        rng.uniform(-1, 1, (S, T, 3)).astype(np.float32), np.zeros((S, T), np.uint8)


def ref_values(agent, img, drv, act, es):
    c = agent.config
    P = ref.params_from_flat(agent.params(), agent.param_layout(), c, dtype=torch.float32)
    return ref.forward_uni(P, c, img, drv, act, es)


@pytest.mark.parametrize("cfg", CFGS, ids=IDS)
def test_cpu_forward_matches_reference(cfg):
    img, drv, act, es = batch()
    agent = agent_lib.Agent(cfg, seed=2)
    out = agent.forward_uni(img, drv, act, es, values=True, move_probs=True)
    r = ref_values(agent, img, drv, act, es)
    S, T = img.shape[:2]
    nt, V, K = ref.uni_dims(agent.config)
    p = r["logits"].reshape(S, T * nt, V, K).softmax(-1)
    centres = torch.arange(K, dtype=p.dtype) / (K - 1)
    np.testing.assert_allclose(out["values"].reshape(S, T * nt, V), (p * centres).sum(-1).detach().numpy(), atol=2e-5)
    agent.close()


def test_the_age_key_matters():
    """Different keys change the outputs, and without age_key the step embedding is back."""
    img, drv, act, es = batch()
    a = agent_lib.Agent(UNI, seed=2)
    o0 = a.forward_uni(img, drv, act, es, values=True)["values"]
    off = {n: (o, s) for n, o, s in a.param_layout()}["wpe"]
    a.params()[off[0]:off[0] + off[1]] = np.random.default_rng(0).normal(0, 5.0, off[1])
    a.params_changed()
    o1 = a.forward_uni(img, drv, act, es, values=True)["values"]
    assert np.abs(o0 - o1).max() > 1e-4
    a.close()


def test_a_later_key_row_does_not_reach_earlier_tokens():
    """wpe[age] with age >= T is unused in a window of T steps."""
    img, drv, act, es = batch()
    a = agent_lib.Agent(UNI, seed=2)
    o0 = a.forward_uni(img, drv, act, es, values=True)["values"]
    o, s = {n: (o, s) for n, o, s in a.param_layout()}["wpe"]
    d = UNI["d_model"]
    a.params()[o + 3 * d:o + s] = 1.0
    a.params_changed()
    np.testing.assert_array_equal(a.forward_uni(img, drv, act, es, values=True)["values"], o0)
    a.close()


@pytest.mark.parametrize("cfg", CFGS, ids=IDS)
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
@pytest.mark.parametrize("cfg", CFGS, ids=IDS)
def test_gpu_forward_matches_cpu(cfg):
    img, drv, act, es = batch(seed=5)
    agent = agent_lib.Agent(cfg, seed=2)
    m = cuda_lib.CudaModel(agent)
    a = agent.forward_uni(img, drv, act, es, values=True, move_probs=True)
    b = m.forward_uni(img, drv, act, es, values=True, move_probs=True)
    np.testing.assert_allclose(b["values"], a["values"], atol=2e-5)
    np.testing.assert_allclose(b["move_probs"], a["move_probs"], atol=2e-5)
    m.close()
    agent.close()


@pytest.mark.skipif(not HAVE_GPU, reason="needs libvmcuda.so and a GPU")
@pytest.mark.parametrize("cfg", CFGS, ids=IDS)
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


@pytest.mark.skipif(not HAVE_GPU, reason="needs libvmcuda.so and a GPU")
@pytest.mark.parametrize("cfg", CFGS, ids=IDS)
def test_candidates_match_full_passes(cfg):
    img, drv, act, es = batch(seed=6, S=3, T=4)
    act[:, -1] = 0.0
    agent = agent_lib.Agent(cfg, seed=3)
    m = cuda_lib.CudaModel(agent)
    cand = np.random.default_rng(1).uniform(-1, 1, (3, 5, 3)).astype(np.float32)
    cand[..., 2] = (cand[..., 2] > 0).astype(np.float32)
    m.forward_uni(img, drv, act, es, values=False, move_probs=True)
    fast = m.uni_candidates(cand)
    for i in range(5):
        a2 = act.copy()
        a2[:, -1] = cand[:, i]
        nb = m.forward_uni(img, drv, a2, es, values=False, move_probs=False, next_body=True)["next_body"]
        np.testing.assert_allclose(fast[:, i], nb[:, -1, 0], atol=3e-5)
    m.close()
    agent.close()


def test_checkpoint_keeps_it(tmp_path):
    a = agent_lib.Agent(UNI, seed=1)
    a.save(str(tmp_path / "c.bin"))
    assert agent_lib.Agent.load(str(tmp_path / "c.bin")).config["age_key"] == 1


@pytest.mark.parametrize("bad", [dict(age_key=2), dict(age_key=-1), dict(universal=0, patch=0)],
                         ids=["two", "negative", "not_universal"])
def test_invalid(bad):
    with pytest.raises(ValueError):
        agent_lib.Agent(dict(UNI, **bad), seed=1)

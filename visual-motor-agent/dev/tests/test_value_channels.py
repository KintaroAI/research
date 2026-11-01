"""A channel per value (task 00075, model.uni_chan): W_in and W_out get the body's 2 and the move's 3 value positions after
a patch's V, so that no position has two meanings. Table sizes, hunger and pixel 0 reaching the model only through
their own rows, forward, loss and gradients against the PyTorch reference (C++ and GPU), candidates against full passes,
the ring against a full pass, checkpoints, invalid configs. uni_chan = 0 is every other test of the universal model.

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
gpu = pytest.mark.skipif(not HAVE_GPU, reason="needs libvmcuda.so and a GPU")
UNI = dict(SMALL, patch=8, universal=1, bins=8, gray=1, in_bins=1, uni_chan=1)
AGE = dict(UNI, age_key=1)
PYR = dict(AGE, uni_pyr=2, sum_thumb=1)
BAND = dict(PYR, band_n=4, band_k=2, band_cols=2)
CFGS = [UNI, AGE, PYR, BAND, dict(UNI, gray=0)]
IDS = ["plain", "age", "pyramid", "piece", "colour"]


def batch(seed=0, S=2, T=3, img=32):
    rng = np.random.default_rng(seed)
    a = rng.uniform(-1, 1, (S, T, 3)).astype(np.float32)
    a[..., 2] = (a[..., 2] > 0).astype(np.float32)
    return rng.integers(0, 256, (S, T, img, img, 3), dtype=np.uint8), rng.random((S, T, 2)).astype(np.float32), a, \
        np.zeros((S, T), np.uint8)


def table(agent, name):
    o, s = {n: (o, s) for n, o, s in agent.param_layout()}[name]
    return agent.params()[o:o + s]


def test_table_sizes():
    """Five more value positions, in both tables and the bias, and nothing else."""
    on, off = agent_lib.Agent(UNI, seed=1), agent_lib.Agent(dict(UNI, uni_chan=0), seed=1)
    d, K = UNI["d_model"], UNI["bins"]
    assert len(table(on, "uni_in_w")) - len(table(off, "uni_in_w")) == 5 * K * d
    assert len(table(on, "uni_out_w")) - len(table(off, "uni_out_w")) == 5 * K * d
    assert len(table(on, "uni_out_b")) - len(table(off, "uni_out_b")) == 5 * K
    assert on.num_params - off.num_params == 2 * 5 * K * d + 5 * K
    on.close()
    off.close()


def rows(agent, first, n):
    """Set the input rows of value positions first..first + n - 1 to zero (W_in is d x VC * K, by row of d)."""
    d, K = agent.config["d_model"], agent.config["bins"]
    w = table(agent, "uni_in_w").reshape(d, -1)
    w[:, first * K:(first + n) * K] = 0.0
    agent.params_changed()


def out(agent, img, drv, act, es):
    return agent.forward_uni(img, drv, act, es, values=True, move_probs=False)["values"]


@pytest.mark.parametrize("name,first,n,change", [
    ("hunger", 64, 2, "drv"), ("move", 66, 3, "act"), ("pixel 0", 0, 1, "px")], ids=["hunger", "move", "pixel"])
def test_a_value_reaches_the_model_through_its_own_rows_only(name, first, n, change):
    """Zero a value's rows and it cannot be seen; zero the rows of what used to share them and it still can."""
    img, drv, act, es = batch()
    img2, drv2, act2 = img.copy(), drv.copy(), act.copy()
    if change == "drv":
        drv2 = 1 - drv
    elif change == "act":
        act2[:, 0] = -act[:, 0]           # step 0's move: before the last token
    else:
        img2[:, :, 0, 0] = 255 - img[:, :, 0, 0]
    a = agent_lib.Agent(UNI, seed=2)
    assert np.abs(out(a, img, drv, act, es) - out(a, img2, drv2, act2, es)).max() > 1e-5
    rows(a, first, n)
    np.testing.assert_array_equal(out(a, img, drv, act, es), out(a, img2, drv2, act2, es))
    a.close()
    if change != "px":                    # pixel 0's rows no longer carry the body or the move
        b = agent_lib.Agent(UNI, seed=2)
        rows(b, 0, n)
        assert np.abs(out(b, img, drv, act, es) - out(b, img2, drv2, act2, es)).max() > 1e-5
        b.close()


@pytest.mark.parametrize("cfg", CFGS, ids=IDS)
def test_cpu_forward_matches_reference(cfg):
    img, drv, act, es = batch()
    agent = agent_lib.Agent(cfg, seed=2)
    o = agent.forward_uni(img, drv, act, es, values=True, move_probs=True, next_body=True)
    c = agent.config
    P = ref.params_from_flat(agent.params(), agent.param_layout(), c, dtype=torch.float32)
    r = ref.forward_uni(P, c, img, drv, act, es)
    S, T = img.shape[:2]
    nt, V, K = ref.uni_dims(c)
    p = r["logits"].reshape(S, T, nt, V, K).softmax(-1).detach()
    want = (p * (torch.arange(K, dtype=p.dtype) / (K - 1))).sum(-1).numpy()
    np.testing.assert_allclose(o["values"], want, atol=2e-5)
    np.testing.assert_allclose(o["move_probs"], p[:, :, nt - 2, :3].numpy(), atol=2e-5)
    np.testing.assert_allclose(o["next_body"], want[:, :, nt - 1, :2], atol=2e-5)
    agent.close()


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


def test_every_new_row_learns():
    """The body's and the move's rows of W_out and W_in have gradients, and the patch rows too."""
    agent_lib.set_threads(4)
    agent = agent_lib.Agent(UNI, seed=7)
    ro, _ = store_rollout(agent, T=16, N=3, seed=3, p_done=0.2)
    agent.debug_grads(ro, np.zeros(3, np.float32), dict(agent_lib.DEFAULT_PPO, use_ppo=0), [0, 2, 4, 5])
    g, lay = agent.grads(), {n: (o, s) for n, o, s in agent.param_layout()}
    d, K = UNI["d_model"], UNI["bins"]
    o, s = lay["uni_out_w"]
    gw = g[o:o + s].reshape(-1, d)
    for first, n in ((0, 64), (64, 2), (66, 3)):
        assert np.abs(gw[first * K:(first + n) * K]).max() > 0, first
    o, s = lay["uni_in_w"]
    gi = g[o:o + s].reshape(d, -1)
    for first, n in ((0, 64), (64, 2), (66, 3)):
        assert np.abs(gi[:, first * K:(first + n) * K]).max() > 0, first
    agent.close()


@gpu
@pytest.mark.parametrize("cfg", CFGS, ids=IDS)
def test_gpu_forward_matches_cpu(cfg):
    img, drv, act, es = batch(seed=5, T=cfg["ctx"])
    es[1, 3] = 1
    agent = agent_lib.Agent(cfg, seed=2)
    m = cuda_lib.CudaModel(agent)
    a = agent.forward_uni(img, drv, act, es, values=True, move_probs=True, next_body=True)
    b = m.forward_uni(img, drv, act, es, values=True, move_probs=True, next_body=True)
    c = m.forward_uni(img, drv, act, es, values=False, move_probs=True, next_body=True)   # the last two rows only
    for k in ("values", "move_probs", "next_body"):
        np.testing.assert_allclose(b[k], a[k], atol=2e-5)
    np.testing.assert_allclose(c["move_probs"], a["move_probs"], atol=2e-5)
    np.testing.assert_allclose(c["next_body"], a["next_body"], atol=2e-5)
    m.close()
    agent.close()


@gpu
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


@gpu
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


@gpu
@pytest.mark.parametrize("cfg", [AGE, PYR, BAND], ids=["age", "pyramid", "piece"])
def test_ring_matches_a_full_pass(cfg):
    N, T = 3, cfg["ctx"]
    x, d, a, es = batch(seed=8, S=N, T=T)
    agent = agent_lib.Agent(cfg, seed=2)
    m = cuda_lib.CudaModel(agent)
    m.uni_ring_reset(N)
    cand = np.random.default_rng(1).uniform(-1, 1, (N, 4, 3)).astype(np.float32)
    for t in range(T):
        full_a = a[:, :t + 1].copy()
        full_a[:, -1] = 0.0
        p = m.forward_uni(x[:, :t + 1], d[:, :t + 1], full_a, es[:, :t + 1], values=False, move_probs=True)["move_probs"]
        hf = m.uni_candidates(cand)
        pr = m.uni_ring_step(x[:, t], d[:, t], es[:, t])
        hr = m.uni_candidates(cand)
        np.testing.assert_allclose(pr, p[:, -1], atol=3e-5, err_msg="tick %d" % t)
        np.testing.assert_allclose(hr, hf, atol=3e-5, err_msg="tick %d" % t)
        m.uni_ring_move(a[:, t])
    m.close()
    agent.close()


def test_checkpoint_keeps_it(tmp_path):
    a = agent_lib.Agent(UNI, seed=1)
    a.save(str(tmp_path / "c.bin"))
    b = agent_lib.Agent.load(str(tmp_path / "c.bin"))
    assert b.config["uni_chan"] == 1
    np.testing.assert_array_equal(a.params(), b.params())
    a.close()
    b.close()


@pytest.mark.parametrize("bad", [dict(uni_chan=2), dict(uni_chan=-1), dict(in_bins=0), dict(universal=0, patch=0)],
                         ids=["two", "negative", "no bins", "not universal"])
def test_invalid_configs(bad):
    with pytest.raises(Exception):
        agent_lib.Agent(dict(UNI, **bad), seed=1)

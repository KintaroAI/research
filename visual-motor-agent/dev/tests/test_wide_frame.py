"""Frames wider than high (task 00064, model.img_w): the universal model's layout, thumbnails and pieces on a 16 x 8 grid;
C++ and GPU against the PyTorch reference; every block inside one half; checkpoints; invalid configs.

Requires libvmagent.so and torch (CPU); the GPU tests libvmcuda.so and a GPU.
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
WIDE = dict(SMALL, img=32, img_w=64, patch=8, pred_img=16, universal=1, bins=8, gray=1, in_bins=1, uni_pyr=2, sum_thumb=1)   # 8 x 4
LEFT = dict(WIDE, band_n=2, band_k=0, band_cols=1)
RIGHT = dict(LEFT, band_k=1)
AGE = dict(RIGHT, age_key=1)
CFGS = [WIDE, LEFT, RIGHT, AGE]
IDS = ["whole", "left", "right", "age"]


def batch(seed=0, S=2, T=8):
    rng = np.random.default_rng(seed)
    a = rng.uniform(-1, 1, (S, T, 3)).astype(np.float32)
    return rng.integers(0, 256, (S, T, 32, 64, 3), dtype=np.uint8), rng.random((S, T, 2)).astype(np.float32), a, \
        np.zeros((S, T), np.uint8)


def test_layout():
    t = ref.uni_slot_table(WIDE)
    assert (t["kind"] == 1).sum() == 32 and (t["kind"] == 2).sum() == 8 + 2
    assert agent_lib.uni_layout(WIDE)["nt"] == 44
    lv = t["level"][t["kind"] == 2]
    assert ((lv == 1).sum(), (lv == 2).sum()) == (8, 2)
    l, r = ref.uni_slot_table(LEFT), ref.uni_slot_table(RIGHT)
    assert (l["c"][l["kind"] == 1] < 4).all() and (r["c"][r["kind"] == 1] >= 4).all()
    assert len(l["kind"]) == len(r["kind"]) == 16 + 10 + 2


def test_no_block_spans_the_two_halves():
    t = ref.uni_slot_table(WIDE)
    s = t["kind"] == 2
    b = 2 ** t["level"][s]
    assert ((t["c"][s] * b < 4) == ((t["c"][s] + 1) * b <= 4)).all()


def test_thumbnails_keep_the_halves_apart():
    img = batch()[0]
    other = img.copy()
    other[:, :, :, 32:] = 0
    a, b = ref.uni_thumbs(WIDE, img), ref.uni_thumbs(WIDE, other)
    t = ref.uni_slot_table(WIDE)
    k = t["kind"] == 2
    left = (t["c"][k] * 2 ** t["level"][k]) < 4
    np.testing.assert_array_equal(a[:, :, left], b[:, :, left])


@pytest.mark.parametrize("cfg", CFGS, ids=IDS)
def test_cpu_forward_matches_reference(cfg):
    img, drv, act, es = batch()
    agent = agent_lib.Agent(cfg, seed=2)
    out = agent.forward_uni(img, drv, act, es, values=True)["values"]
    c = agent.config
    P = ref.params_from_flat(agent.params(), agent.param_layout(), c, dtype=torch.float32)
    r = ref.forward_uni(P, c, img, drv, act, es)
    S, T = img.shape[:2]
    nt, V, K = ref.uni_dims(c)
    p = r["logits"].reshape(S, T * nt, V, K).softmax(-1)
    want = (p * (torch.arange(K, dtype=p.dtype) / (K - 1))).sum(-1).detach().numpy()
    np.testing.assert_allclose(out.reshape(S, T * nt, V), want, atol=2e-5)
    agent.close()


@pytest.mark.skipif(not HAVE_GPU, reason="needs libvmcuda.so and a GPU")
@pytest.mark.parametrize("cfg", CFGS, ids=IDS)
def test_gpu_forward_matches_cpu(cfg):
    img, drv, act, es = batch(seed=5)
    agent = agent_lib.Agent(cfg, seed=2)
    m = cuda_lib.CudaModel(agent)
    a = agent.forward_uni(img, drv, act, es, values=True, move_probs=True)
    b = m.forward_uni(img, drv, act, es, values=True, move_probs=True)
    np.testing.assert_allclose(b["values"], a["values"], atol=3e-5)
    np.testing.assert_allclose(b["move_probs"], a["move_probs"], atol=3e-5)
    m.close()
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
def test_ring_on_a_wide_frame_equals_a_full_pass():
    img, drv, act, es = batch(seed=7, S=2)
    agent = agent_lib.Agent(AGE, seed=4)
    m = cuda_lib.CudaModel(agent)
    m.uni_ring_reset(2)
    for t in range(img.shape[1]):
        w = act[:, :t + 1].copy()
        w[:, -1] = 0.0
        p = m.forward_uni(img[:, :t + 1], drv[:, :t + 1], w, es[:, :t + 1], values=False, move_probs=True)["move_probs"]
        np.testing.assert_allclose(m.uni_ring_step(img[:, t], drv[:, t], es[:, t]), p[:, -1], atol=3e-5)
        m.uni_ring_move(act[:, t])
    m.close()
    agent.close()


def test_checkpoint_keeps_it(tmp_path):
    a = agent_lib.Agent(WIDE, seed=1)
    a.save(str(tmp_path / "c.bin"))
    assert agent_lib.Agent.load(str(tmp_path / "c.bin")).config["img_w"] == 64


@pytest.mark.parametrize("bad", [dict(img_w=60), dict(img_w=-8), dict(universal=0, patch=0), dict(img_w=56),
                                 dict(band_n=4, band_cols=2), dict(attn_win=1)],
                         ids=["patch", "negative", "not_universal", "pyramid", "blocks", "window"])
def test_invalid(bad):
    with pytest.raises(ValueError):
        agent_lib.Agent(dict(WIDE, **bad), seed=1)

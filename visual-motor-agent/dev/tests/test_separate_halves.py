"""Separate transformers that see every summary (task 00057, model.band_n / band_k / band_cols): a model's patch tokens
are only those of its band; every summary, the body and the move stay in their places. The layout (C++ and reference);
the other bands reach a band model only through the thumbnails; C++ and GPU against the PyTorch reference;
checkpoints; invalid configs.

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
FULL = dict(SMALL, patch=8, universal=1, bins=8, gray=1, in_bins=1, uni_pyr=2, sum_thumb=1)   # 32 x 32: a 4 x 4 grid
LEFT = dict(FULL, band_n=2, band_k=0, band_cols=1)
RIGHT = dict(LEFT, band_k=1)


@pytest.mark.parametrize("n,k,cols", [(2, 0, 1), (2, 1, 1), (2, 0, 0), (2, 1, 0), (4, 2, 1)])
def test_layout_is_the_full_one_without_the_other_bands_patches(n, k, cols):
    cfg = dict(FULL, img=64, uni_pyr=3, band_n=n, band_k=k, band_cols=cols)   # 8 x 8 grid, B1-B3
    full, t = ref.uni_slot_table(dict(FULL, img=64, uni_pyr=3)), ref.uni_slot_table(cfg)
    pos = full["c"] if cols else full["r"]
    keep = (full["kind"] != 1) | (pos // (8 // n) == k)
    for f in ("kind", "index", "level", "r", "c"):
        assert np.array_equal(t[f], full[f][keep]), f
    lay = agent_lib.uni_layout(cfg)
    assert lay["nt"] == len(t["kind"]) == 64 // n + 2 + 64 // 4 + 4 + 1
    assert np.array_equal(lay["patch"], np.flatnonzero(t["kind"] == 1))
    agent = agent_lib.Agent(cfg, seed=1)                                       # C++: the same number of slots
    S, T = 1, 2
    out = agent.forward_uni(np.zeros((S, T, 64, 64, 3), np.uint8), np.zeros((S, T, 2), np.float32),
                            np.zeros((S, T, 3), np.float32), np.zeros((S, T), np.uint8), values=True)["values"]
    assert out.shape[2] == lay["nt"]
    agent.close()


def batch(seed=0, S=2, T=3):
    rng = np.random.default_rng(seed)
    img = rng.integers(0, 256, (S, T, 32, 32, 3), dtype=np.uint8)
    return img, rng.random((S, T, 2)).astype(np.float32), rng.uniform(-1, 1, (S, T, 3)).astype(np.float32), \
        np.zeros((S, T), np.uint8)


def shuffle_keeping_thumbnails(img, x0, x1):
    """Columns x0..x1 - 1 with the pixels of each aligned 2 x 2 square rotated: every box average over 2^l x 2^l
    squares (the thumbnails) stays, the patches change."""
    out = img.copy()
    a = img[:, :, :, x0:x1]
    out[:, :, 0::2, x0:x1][..., 0::2, :] = a[:, :, 1::2][..., 0::2, :]
    out[:, :, 1::2, x0:x1][..., 0::2, :] = a[:, :, 1::2][..., 1::2, :]
    out[:, :, 1::2, x0:x1][..., 1::2, :] = a[:, :, 0::2][..., 1::2, :]
    out[:, :, 0::2, x0:x1][..., 1::2, :] = a[:, :, 0::2][..., 0::2, :]
    return out


def test_shuffle_keeps_the_thumbnails():
    img = batch()[0]
    sh = shuffle_keeping_thumbnails(img, 16, 32)
    assert not np.array_equal(sh, img)
    np.testing.assert_allclose(ref.uni_thumbs(FULL, sh), ref.uni_thumbs(FULL, img), atol=1e-12)


@pytest.mark.parametrize("dev", ["cpu", "gpu"])
def test_the_other_half_reaches_a_band_only_through_the_thumbnails(dev):
    """The right half's pixels shuffled within 2 x 2 squares: the left model's outputs stay the same to the bit, the
    full model's and the right model's change."""
    if dev == "gpu" and not HAVE_GPU:
        pytest.skip("needs libvmcuda.so and a GPU")
    img, drv, act, es = batch()
    sh = shuffle_keeping_thumbnails(img, 16, 32)
    for cfg, same in ((LEFT, True), (RIGHT, False), (FULL, False)):
        agent = agent_lib.Agent(cfg, seed=3)
        m = agent if dev == "cpu" else cuda_lib.CudaModel(agent)
        a = m.forward_uni(img, drv, act, es, values=True)["values"]
        b = m.forward_uni(sh, drv, act, es, values=True)["values"]
        assert np.array_equal(a, b) == same, np.abs(a - b).max()
        if dev == "gpu":
            m.close()
        agent.close()


@pytest.mark.skipif(not HAVE_GPU, reason="needs libvmcuda.so and a GPU")
@pytest.mark.parametrize("cfg", [LEFT, dict(RIGHT, band_cols=0)], ids=["left", "bottom"])
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


@pytest.mark.parametrize("cfg", [LEFT, RIGHT, dict(LEFT, band_cols=0), dict(LEFT, band_n=4, band_k=3)],
                         ids=["left", "right", "top", "quarter"])
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
@pytest.mark.parametrize("cfg", [RIGHT, dict(LEFT, band_cols=0)], ids=["right", "top"])
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


def test_checkpoint_keeps_the_band(tmp_path):
    a = agent_lib.Agent(dict(RIGHT, band_cols=0), seed=1)
    a.save(str(tmp_path / "c.bin"))
    c = agent_lib.Agent.load(str(tmp_path / "c.bin")).config
    assert (c["band_n"], c["band_k"], c["band_cols"]) == (2, 1, 0)


@pytest.mark.parametrize("bad", [dict(band_n=1), dict(band_n=3), dict(band_k=2), dict(band_k=-1), dict(band_cols=2),
                                 dict(sum_thumb=0), dict(uni_pyr=0, sum_thumb=0), dict(sum_only=1),
                                 dict(attn_split=2), dict(attn_win=1), dict(band_n=0, band_k=1)],
                         ids=["one", "not_dividing", "past_bands", "negative", "cols_two", "no_thumbs", "no_pyramid",
                              "sum_only", "attn_split", "window", "k_without_n"])
def test_invalid_configs(bad):
    with pytest.raises(ValueError):
        agent_lib.Agent(dict(LEFT, **bad), seed=1)

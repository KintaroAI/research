"""Separate transformers that see only nearby summaries (task 00058, model.band_win; band_cols = 2, square blocks): a
piece keeps a summary only within band_win cells of itself at the summary's level. The layout (C++ and reference,
hand counts); a dropped summary changes nothing; C++ and GPU against the PyTorch reference; checkpoints; invalid
configs.

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
FULL = dict(SMALL, img=64, patch=8, universal=1, bins=8, gray=1, in_bins=1, uni_pyr=3, sum_thumb=1)   # 8 x 8 grid
BLOCK = dict(FULL, band_n=16, band_cols=2, band_k=0, band_win=1)   # 16 blocks of 2 x 2 patches; block 0 top left
TEXT2X = dict(FULL, img=128, uni_pyr=4)                              # the 2x text world's grid: 16 x 16, B1-B4


def nsum(cfg):
    return int((ref.uni_slot_table(cfg)["kind"] == 2).sum())


@pytest.mark.parametrize("k,want", [(0, 37), (1, 49), (2, 49), (3, 37)])
def test_quarters_count(k, want):
    """Four horizontal quarters of the 16 x 16 grid, w = 1: B1 4 (edge 3) rows of 8, B2 3 (2) rows of 4, B3 4, B4 1."""
    assert nsum(dict(TEXT2X, band_n=4, band_k=k, band_win=1)) == want
    assert nsum(dict(TEXT2X, band_n=4, band_k=k)) == 85                     # without the window: every summary


@pytest.mark.parametrize("k,want", [(0, 18), (5, 30), (6, 30), (15, 18), (1, 23)])
def test_blocks_count(k, want):
    """16 blocks of 4 x 4 patches, w = 1: interior B1 4 x 4, B2 3 x 3, B3 2 x 2, B4 1 = 30; a corner 9 + 4 + 4 + 1;
    block 1 (top edge) 3 x 4 + 2 x 3 + 4 + 1."""
    assert nsum(dict(TEXT2X, band_n=16, band_cols=2, band_k=k, band_win=1)) == want


@pytest.mark.parametrize("n,k,cols,w", [(4, 1, 0, 1), (4, 2, 1, 1), (16, 5, 2, 1), (16, 0, 2, 2), (4, 3, 2, 1),
                                        (16, 9, 2, 0)])
def test_layout_is_the_band_one_without_the_far_summaries(n, k, cols, w):
    cfg = dict(TEXT2X, band_n=n, band_k=k, band_cols=cols, band_win=w)
    full = ref.uni_slot_table(dict(TEXT2X))
    g = 16
    s = int(round(np.sqrt(n)))
    if cols == 2:   # the piece's patches, by enumeration
        piece = {(r, c) for r in range(g) for c in range(g) if r // (g // s) == k // s and c // (g // s) == k % s}
    else:
        piece = {(r, c) for r in range(g) for c in range(g) if (c if cols else r) // (g // n) == k}
    keep = []
    for kind, lv, r, c in zip(full["kind"], full["level"], full["r"], full["c"]):
        if kind == 1:
            keep.append((r, c) in piece)
        elif kind == 2 and w > 0:   # some cell of the piece at this level within w rows and w columns
            cells = {(pr >> lv, pc >> lv) for pr, pc in piece}
            rows, cs = {a for a, _ in cells}, {b for _, b in cells}
            keep.append(min(abs(r - a) for a in rows) <= w and min(abs(c - b) for b in cs) <= w)
        else:
            keep.append(True)
    keep = np.array(keep)
    t = ref.uni_slot_table(cfg)
    for f in ("kind", "level", "r", "c"):
        assert np.array_equal(t[f], full[f][keep]), f
    lay = agent_lib.uni_layout(cfg)
    assert lay["nt"] == len(t["kind"]) and np.array_equal(lay["kind"], t["kind"])
    assert (t["kind"] == 1).sum() == g * g // n


def test_cpp_has_the_same_slots():
    agent = agent_lib.Agent(BLOCK, seed=1)
    out = agent.forward_uni(np.zeros((1, 2, 64, 64, 3), np.uint8), np.zeros((1, 2, 2), np.float32),
                            np.zeros((1, 2, 3), np.float32), np.zeros((1, 2), np.uint8), values=True)["values"]
    assert out.shape[2] == agent_lib.uni_layout(BLOCK)["nt"] == 4 + 4 + 4 + 1 + 2   # 4 patches, B1 2x2, B2 all, B3
    agent.close()


def batch(seed=0, S=2, T=3):
    rng = np.random.default_rng(seed)
    img = rng.integers(0, 256, (S, T, 64, 64, 3), dtype=np.uint8)
    return img, rng.random((S, T, 2)).astype(np.float32), rng.uniform(-1, 1, (S, T, 3)).astype(np.float32), \
        np.zeros((S, T), np.uint8)


def swap_far_b1(img):
    """In B1 cell (3, 3) (pixels 48-63, a 2 x 2 patch block) the four 2 x 2-pixel boxes of every aligned 4 x 4 square
    rotated: every 4 x 4 box mean stays, so the B2 and B3 thumbnails (a 32- or 64-pixel block scaled to 8 x 8) stay;
    the B1 thumbnail (2 x 2 boxes) and the patches change. Block 0 with w = 1 reads B1 rows and columns 0-1 only."""
    out = img.copy()
    a = img[:, :, 48:64, 48:64].reshape(img.shape[:2] + (4, 2, 2, 4, 2, 2, 3))   # (sq y, box y, y, sq x, box x, x)
    b = a.copy()
    b[:, :, :, 0, :, :, 0], b[:, :, :, 0, :, :, 1] = a[:, :, :, 1, :, :, 0], a[:, :, :, 0, :, :, 0]
    b[:, :, :, 1, :, :, 1], b[:, :, :, 1, :, :, 0] = a[:, :, :, 0, :, :, 1], a[:, :, :, 1, :, :, 1]
    out[:, :, 48:64, 48:64] = b.reshape(img.shape[:2] + (16, 16, 3))
    return out


@pytest.mark.parametrize("dev", ["cpu", "gpu"])
def test_a_dropped_summary_changes_nothing(dev):
    if dev == "gpu" and not HAVE_GPU:
        pytest.skip("needs libvmcuda.so and a GPU")
    img, drv, act, es = batch()
    sw = swap_far_b1(img)
    t0, t1 = ref.uni_thumbs(FULL, img), ref.uni_thumbs(FULL, sw)
    lev = ref.uni_slot_table(FULL)["level"][ref.uni_slot_table(FULL)["kind"] == 2]
    assert np.allclose(t0[:, :, lev >= 2], t1[:, :, lev >= 2]) and not np.allclose(t0, t1)
    for cfg, same in ((BLOCK, True), (dict(BLOCK, band_win=0), False), (dict(BLOCK, band_k=15), False)):
        agent = agent_lib.Agent(cfg, seed=3)
        m = agent if dev == "cpu" else cuda_lib.CudaModel(agent)
        a = m.forward_uni(img, drv, act, es, values=True)["values"]
        b = m.forward_uni(sw, drv, act, es, values=True)["values"]
        assert np.array_equal(a, b) == same, (cfg["band_k"], cfg["band_win"], np.abs(a - b).max())
        if dev == "gpu":
            m.close()
        agent.close()


@pytest.mark.skipif(not HAVE_GPU, reason="needs libvmcuda.so and a GPU")
@pytest.mark.parametrize("cfg", [dict(BLOCK, band_k=5), dict(FULL, band_n=4, band_k=2, band_win=1)],
                         ids=["block", "quarter"])
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


@pytest.mark.parametrize("cfg", [dict(BLOCK, band_k=5), dict(BLOCK, band_k=12),
                                 dict(FULL, band_n=4, band_k=2, band_win=1)], ids=["block5", "block12", "quarter"])
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
@pytest.mark.parametrize("cfg", [dict(BLOCK, band_k=6), dict(FULL, band_n=4, band_k=0, band_win=1)],
                         ids=["block", "quarter"])
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


def test_checkpoint_keeps_the_window(tmp_path):
    a = agent_lib.Agent(dict(BLOCK, band_k=7, band_win=2), seed=1)
    a.save(str(tmp_path / "c.bin"))
    c = agent_lib.Agent.load(str(tmp_path / "c.bin")).config
    assert (c["band_n"], c["band_k"], c["band_cols"], c["band_win"]) == (16, 7, 2, 2)


@pytest.mark.parametrize("bad", [dict(band_win=-1), dict(band_n=0, band_k=0, band_cols=0, band_win=1),
                                 dict(band_cols=3), dict(band_n=8), dict(band_n=9), dict(band_k=16)],
                         ids=["negative", "without_band_n", "cols_three", "not_square", "side_not_dividing",
                              "past_blocks"])
def test_invalid_configs(bad):
    with pytest.raises(ValueError):
        agent_lib.Agent(dict(BLOCK, **bad), seed=1)

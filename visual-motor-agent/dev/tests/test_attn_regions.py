"""Attention in regions (task 00044, model.attn_split / attn_overlap): the allow table, what a region cannot see,
C++ and GPU against the PyTorch reference. GPU against C++ for every update and forward path: test_cuda_model.py's
*_split models.

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

UNI = dict(SMALL, patch=8, universal=1, bins=8)   # 32x32 frames: a 4x4 patch grid, 18 tokens per step


def rows_allowed(allow, g):
    """allow [nt, nt] -> the patch-row pairs that meet [g, g] (asserting every patch of a row behaves alike)."""
    nt = g * g + 2
    assert allow[0].all() and allow[:, 0].all() and allow[nt - 1].all() and allow[:, nt - 1].all()   # body, move
    p = allow[1:nt - 1, 1:nt - 1].reshape(g, g, g, g)
    r = p[:, 0, :, 0]
    assert (p == r[:, None, :, None]).all()
    return r


def test_allow_table():
    cfg = dict(universal=1, img=64, patch=8, attn_split=2, attn_overlap=1)
    r = rows_allowed(ref.attn_allow(cfg), 8)
    top, bottom = set(range(0, 5)), set(range(3, 8))     # one row of overlap each way: rows 3 and 4 in both
    want = np.array([[(a in top and b in top) or (a in bottom and b in bottom) for b in range(8)] for a in range(8)])
    np.testing.assert_array_equal(r, want)
    r0 = rows_allowed(ref.attn_allow(dict(cfg, attn_overlap=0)), 8)
    np.testing.assert_array_equal(r0, np.kron(np.eye(2, dtype=bool), np.ones((4, 4), bool)))
    r4 = rows_allowed(ref.attn_allow(dict(cfg, attn_split=4, attn_overlap=1)), 8)
    bands = [range(0, 3), range(1, 5), range(3, 7), range(5, 8)]       # rows 0-1, 2-3, 4-5, 6-7, each reaching one row
    want4 = np.array([[any(a in b and c in b for b in bands) for c in range(8)] for a in range(8)])
    np.testing.assert_array_equal(r4, want4)
    assert r4[2, 4] and not r4[2, 5] and not r4[0, 3]
    assert ref.attn_allow(dict(cfg, attn_split=1)) is None and ref.attn_allow(dict(cfg, attn_split=0)) is None
    # attn_pred_global: the last patch (it predicts the move) meets every token; the other patches are as before
    a, ag = ref.attn_allow(cfg), ref.attn_allow(dict(cfg, attn_pred_global=1))
    assert ag[64].all() and ag[:, 64].all() and not a[64].all()
    np.testing.assert_array_equal(ag[:64, :64], a[:64, :64])
    # attn_cols: the same bands over the columns (the transposed grid)
    ac = ref.attn_allow(dict(cfg, attn_cols=1))
    p = ac[1:65, 1:65].reshape(8, 8, 8, 8)                            # [row q, col q, row k, col k]
    assert (p == p[:1, :, :1, :]).all()                               # rows do not matter
    np.testing.assert_array_equal(p[0, :, 0, :], want)


def test_checker_allow_table():
    """attn_checker (task 00046): patch (r, c) in region (r + c) % S; two patches meet only in the same region; the
    body and the move meet every token, with attn_pred_global also the last patch."""
    cfg = dict(universal=1, img=64, patch=8, attn_split=2, attn_checker=1)
    a = ref.attn_allow(cfg)
    assert a[0].all() and a[:, 0].all() and a[65].all() and a[:, 65].all()
    p = a[1:65, 1:65].reshape(8, 8, 8, 8)                             # [row q, col q, row k, col k]
    r, c = np.meshgrid(np.arange(8), np.arange(8), indexing="ij")
    par = (r + c) % 2
    np.testing.assert_array_equal(p, par[:, :, None, None] == par[None, None])
    assert p[0, 0, 1, 1] and not p[0, 0, 0, 1] and not p[0, 1, 1, 1] and p[0, 1, 1, 0]
    assert a[64].sum() == 32 + 2                                    # the last patch (7, 7): its 32 and the globals
    ag = ref.attn_allow(dict(cfg, attn_pred_global=1))
    assert ag[64].all() and ag[:, 64].all()
    np.testing.assert_array_equal(ag[:64, :64], a[:64, :64])
    a3 = ref.attn_allow(dict(cfg, attn_split=3))[1:65, 1:65].reshape(8, 8, 8, 8)
    np.testing.assert_array_equal(a3, ((r + c) % 3)[:, :, None, None] == ((r + c) % 3)[None, None])


@pytest.mark.parametrize("bad", [dict(attn_split=3), dict(attn_split=2, attn_overlap=2), dict(attn_overlap=1),
                                 dict(attn_split=-1), dict(attn_split=2, attn_overlap=-1), dict(attn_pred_global=1),
                                 dict(attn_split=2, attn_pred_global=2), dict(attn_cols=1),
                                 dict(attn_split=2, attn_cols=2), dict(attn_checker=1),
                                 dict(attn_split=2, attn_checker=2), dict(attn_split=2, attn_overlap=1, attn_checker=1),
                                 dict(attn_split=2, attn_cols=1, attn_checker=1), dict(attn_split=5, attn_checker=1)])
def test_invalid_configs(bad):
    with pytest.raises(ValueError):
        agent_lib.Agent(dict(UNI, **bad), seed=1)          # a 4x4 grid: halves of 2 rows, overlap < 2
    with pytest.raises(ValueError):
        agent_lib.Agent(dict(SMALL, patch=8, pred_residual=1, attn_split=2), seed=1)   # not the universal model


def test_a_region_does_not_see_the_other_directly():
    """One layer: a bottom-only patch's output does not change when a top-only patch's input changes (in any earlier
    step), and does when an overlap patch's or the body's does; without the split it changes for all."""
    rng = np.random.default_rng(0)
    S, T = 2, 3
    img = rng.integers(0, 256, (S, T, 32, 32, 3), dtype=np.uint8)
    drv = rng.random((S, T, 2)).astype(np.float32)
    act = rng.uniform(-1, 1, (S, T, 3)).astype(np.float32)
    es = np.zeros((S, T), bool)
    g, nt = 4, 18

    def out(cfg, im, dv):
        agent = agent_lib.Agent(cfg, seed=3)
        P = ref.params_from_flat(agent.params(), agent.param_layout(), cfg)
        agent.close()
        with torch.no_grad():
            return ref.forward_uni(P, cfg, im, dv, act, es)["h"].numpy().reshape(S, T, nt, -1)

    def changed(cfg, im, dv, q=1 + 3 * g + 1):   # default: a bottom-only patch (row 3) of the last step
        a, b = out(cfg, img, drv), out(cfg, im, dv)
        return np.abs(a[:, -1, q] - b[:, -1, q]).max() > 1e-9

    top = img.copy()
    top[:, 0, 0:8, 8:16] = 255 - top[:, 0, 0:8, 8:16]           # step 0, row 0 (top only), column 1
    ovl = img.copy()
    ovl[:, 1, 8:16, 0:8] = 255 - ovl[:, 1, 8:16, 0:8]          # step 1, row 1 (in both halves with overlap 1)
    body = drv.copy()
    body[:, 1, 0] = 1 - body[:, 1, 0]
    for in_bins in (0, 1):
        split = dict(UNI, n_layer=1, in_bins=in_bins, attn_split=2, attn_overlap=1)
        assert not changed(split, top, drv)
        assert changed(split, ovl, drv) and changed(split, img, body)
        assert changed(dict(split, attn_split=0, attn_overlap=0), top, drv)
        # the last patch (it predicts the move) sees the top only with attn_pred_global
        assert not changed(split, top, drv, q=nt - 2) and changed(dict(split, attn_pred_global=1), top, drv, q=nt - 2)
        # the checkerboard: a patch sees an earlier patch of its own parity (row + column), not of the other
        ck = dict(UNI, n_layer=1, in_bins=in_bins, attn_split=2, attn_checker=1)
        same, other = img.copy(), img.copy()
        same[:, 0, 0:8, 8:16] = 255 - same[:, 0, 0:8, 8:16]             # step 0, patch (0, 1): odd
        other[:, 0, 0:8, 0:8] = 255 - other[:, 0, 0:8, 0:8]             # step 0, patch (0, 0): even
        q30 = 1 + 3 * g + 0                                              # patch (3, 0) of the last step: odd
        assert changed(ck, same, drv, q=q30) and not changed(ck, other, drv, q=q30)
        assert changed(ck, img, body, q=q30)
        q33 = 1 + 3 * g + 3                                              # patch (3, 3): even
        assert changed(ck, other, drv, q=q33) and not changed(ck, same, drv, q=q33)


@pytest.fixture(scope="module", params=[(8, 0, 0, 2, 1, 0), (8, 0, 1, 2, 0, 0), (4, 1, 1, 2, 1, 0), (4, 1, 1, 4, 1, 0),
                                        (8, 0, 1, 2, 1, 1), (4, 1, 1, 4, 1, 1), (4, 1, 1, 2, 1, 2), (8, 0, 1, 2, 0, 4),
                                        (4, 1, 1, 2, 0, 5), (4, 1, 1, 3, 0, 4)],
                ids=["p8_S2_O1", "p8_inbins_S2_O0", "p4_gray_inbins_S2_O1", "p4_gray_inbins_S4_O1", "p8_inbins_S2_O1_pg",
                     "p4_gray_inbins_S4_O1_pg", "p4_gray_inbins_S2_O1_cols", "p8_inbins_checker",
                     "p4_gray_inbins_checker_pg", "p4_gray_inbins_checker_S3"])
def split_agent(request):
    agent_lib.set_threads(4)
    patch, gray, in_bins, S, O, pg = request.param      # pg bits: 1 pred_global, 2 columns (attn_cols), 4 checker
    return agent_lib.Agent(dict(UNI, patch=patch, gray=gray, in_bins=in_bins, bins=6 if gray else 8, attn_split=S,
                                attn_overlap=O, attn_pred_global=pg % 2, attn_cols=pg // 2 % 2,
                                attn_checker=pg // 4), seed=7)


def test_cpu_loss_and_grads_match_reference(split_agent):
    T, N = 16, 3
    ro, data = store_rollout(split_agent, T=T, N=N, seed=3, p_done=0.2)   # episodes also start inside windows
    ids = [0, 2, 4, 5]
    pc = dict(agent_lib.DEFAULT_PPO, use_ppo=0)
    loss_cpp, _ = split_agent.debug_grads(ro, np.zeros(N, np.float32), pc, ids)
    c = split_agent.config
    P = ref.params_from_flat(split_agent.params(), split_agent.param_layout(), c)
    z = np.zeros((T, N))
    batch = seq_batch(data, ids, c["ctx"], N, z, z, c["img"] // c["pred_img"])
    loss_ref, _ = ref.ppo_loss(P, c, pc, batch)
    loss_ref.backward()
    assert abs(loss_cpp - loss_ref.item()) < 1e-4 * max(1.0, abs(loss_ref.item())), (loss_cpp, loss_ref.item())
    worst = compare_grads(split_agent, P)[0][0]
    P0 = ref.params_from_flat(split_agent.params(), split_agent.param_layout(), c)   # the split is applied: the
    loss_full, _ = ref.ppo_loss(P0, dict(c, attn_split=0, attn_overlap=0), pc, batch)  # C++ gradients are far from
    loss_full.backward()                                                                # the unsplit model's
    with pytest.raises(AssertionError):
        compare_grads(split_agent, P0, tol=max(10 * worst, 1e-5))


@pytest.mark.skipif(not cuda_lib.available(), reason="libvmcuda.so not built")
@pytest.mark.parametrize("cfg", [dict(UNI, in_bins=1, attn_split=2, attn_overlap=1),
                                 dict(UNI, img=16, pred_img=16, ctx=4, attn_split=2),          # short windows
                                 dict(UNI, img=64, pred_img=32, patch=8, gray=1, in_bins=1, bins=32, attn_split=2,
                                      attn_overlap=1),
                                 dict(UNI, in_bins=1, attn_split=2, attn_overlap=1, attn_pred_global=1),
                                 dict(UNI, img=16, pred_img=16, ctx=4, attn_split=2, attn_pred_global=1),
                                 dict(UNI, in_bins=1, attn_split=2, attn_overlap=1, attn_cols=1),
                                 dict(UNI, in_bins=1, attn_split=2, attn_checker=1),
                                 dict(UNI, img=16, pred_img=16, ctx=4, attn_split=2, attn_checker=1),
                                 dict(UNI, img=64, pred_img=32, patch=8, gray=1, in_bins=1, bins=32, attn_split=2,
                                      attn_checker=1, attn_pred_global=1)],
                         ids=["small_S2_O1", "tiny_short_windows", "gray_inbins_64", "small_pred_global",
                              "tiny_pred_global", "small_cols", "small_checker", "tiny_checker",
                              "gray_inbins_64_checker_pg"])
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


def test_checkpoint_keeps_the_split(tmp_path):
    agent = agent_lib.Agent(dict(UNI, attn_split=2, attn_overlap=1, attn_pred_global=1, attn_cols=1), seed=2)
    path = str(tmp_path / "a.bin")
    agent.save(path)
    b = agent_lib.Agent(path=path)
    assert b.config["attn_split"] == 2 and b.config["attn_overlap"] == 1 and b.config["attn_pred_global"] == 1
    assert b.config["attn_cols"] == 1
    agent.close()
    b.close()
    agent = agent_lib.Agent(dict(UNI, attn_split=2, attn_checker=1), seed=2)
    agent.save(path)
    b = agent_lib.Agent(path=path)
    assert b.config["attn_split"] == 2 and b.config["attn_checker"] == 1
    agent.close()
    b.close()

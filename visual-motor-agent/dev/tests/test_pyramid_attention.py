"""A pyramid of summary tokens (task 00048, model.uni_pyr with model.attn_win): the slot layout, the allow table, the
cost's growth, what a patch can see, C++ and GPU against the PyTorch reference, checkpoints. The candidates, lookahead,
self-forcing and dream tests run pyramid models too (test_cuda_model.py, test_lookahead.py, test_self_forcing.py,
test_eval_dream.py).

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

UNI = dict(SMALL, patch=8, universal=1, bins=8)   # 32x32 frames: a 4x4 patch grid


@pytest.mark.parametrize("img,patch,L", [(32, 8, 1), (32, 8, 2), (64, 8, 1), (64, 8, 2), (64, 8, 3)])
def test_layout(img, patch, L):
    cfg = dict(universal=1, img=img, patch=patch, uni_pyr=L, gray=1, bins=4)
    g = img // patch
    t = ref.uni_slot_table(cfg)
    lay = agent_lib.uni_layout(cfg)
    nsum = sum((g >> lv) ** 2 for lv in range(1, L + 1))
    assert lay["nt"] == len(t["kind"]) == g * g + 2 + nsum == ref.uni_dims(cfg)[0]
    np.testing.assert_array_equal(lay["kind"], t["kind"])
    np.testing.assert_array_equal(t["index"][lay["patch"]], np.arange(g * g))
    S = lay["summary"]
    assert len(S) == nsum and t["kind"][-1] == 3
    for k in S:   # right after its block's last patch, behind any finer summary of the same patch
        lv, r, c = t["level"][k], t["r"][k], t["c"][k]
        last = (r + 1) * 2 ** lv - 1, (c + 1) * 2 ** lv - 1
        j = k - 1
        while t["kind"][j] == 2:
            assert t["level"][j] < lv
            j -= 1
        assert t["kind"][j] == 1 and (t["r"][j], t["c"][j]) == last
    if 2 ** L == g:   # the top level: one global summary, the move's predictor
        assert t["level"][-2] == L and lay["nt"] - 2 == S[-1]


def test_allow_table():
    cfg = dict(universal=1, img=64, patch=8, attn_win=1, uni_pyr=3)
    a = ref.attn_allow(cfg)
    t = ref.uni_slot_table(cfg)
    kind, lev, r, c = t["kind"], t["level"], t["r"], t["c"]
    nt = len(kind)
    assert a.shape == (nt, nt) and nt == 64 + 21 + 2
    glob = (kind == 0) | (kind == 3)
    assert a[glob].all() and a[:, glob].all()
    P, S = np.flatnonzero(kind == 1), np.flatnonzero(kind == 2)
    want = (np.abs(r[P][:, None] - r[P][None]) <= 1) & (np.abs(c[P][:, None] - c[P][None]) <= 1)
    np.testing.assert_array_equal(a[np.ix_(P, P)], want)
    for q in P:   # a patch: at most 9 summaries per level, the nearest ones
        for lv in (1, 2, 3):
            Sl = S[lev[S] == lv]
            got = a[q, Sl]
            assert got.sum() <= 9 and got.any()
            near = (np.abs(r[q] // 2 ** lv - r[Sl]) <= 1) & (np.abs(c[q] // 2 ** lv - c[Sl]) <= 1)
            np.testing.assert_array_equal(got, near)
    for q in S:   # a summary: exactly its block's patches, no finer summary
        blk = (r[P] // 2 ** lev[q] == r[q]) & (c[P] // 2 ** lev[q] == c[q])
        np.testing.assert_array_equal(a[q, P], blk)
        assert blk.sum() == 4 ** lev[q]
        assert not a[q, S[lev[S] < lev[q]]].any()
    assert ref.attn_allow(dict(cfg, attn_win=0)) is None          # registers: no table


def scores(cfg):
    a = ref.attn_allow(cfg)
    nt = len(ref.uni_slot_table(cfg)["kind"])
    return a.sum() if a is not None else nt * nt


def test_cost_grows_like_n_log_n():
    """Allowed scores within one step: the pyramid's grow about 5x per 4x more patches (N log N), full attention's
    16x."""
    pyr = [scores(dict(universal=1, img=8 * g, patch=8, attn_win=1, uni_pyr=int(np.log2(g)))) for g in (8, 16, 32)]
    full = [scores(dict(universal=1, img=8 * g, patch=8)) for g in (8, 16, 32)]
    for i in (1, 2):
        assert pyr[i] / pyr[i - 1] < 6 and full[i] / full[i - 1] > 15   # N log N: 4 x 8/6 = 5.3, 4 x 10/8 = 5
    assert pyr[2] < 0.1 * full[2]


@pytest.mark.parametrize("bad", [dict(uni_pyr=3), dict(uni_pyr=-1), dict(uni_pyr=1, uni_sum=1),
                                 dict(uni_pyr=1, attn_split=2)])
def test_invalid_configs(bad):
    with pytest.raises(ValueError):
        agent_lib.Agent(dict(UNI, **bad), seed=1)              # a 4x4 grid: at most 2 levels
    with pytest.raises(ValueError):
        agent_lib.Agent(dict(SMALL, patch=8, pred_residual=1, uni_pyr=1), seed=1)


def test_what_a_patch_sees():
    """4x4 grid, two levels, W = 1. One layer: patch (3, 3) does not see patch (0, 0) of this step. Two layers: without
    the pyramid it still does not (distance 3 > 2W); with it, (0, 0) reaches it through B1(0, 0), complete after patch
    (1, 1) and within one block of (3, 3)'s."""
    rng = np.random.default_rng(0)
    S, T = 2, 3
    img = rng.integers(0, 256, (S, T, 32, 32, 3), dtype=np.uint8)
    drv = rng.random((S, T, 2)).astype(np.float32)
    act = rng.uniform(-1, 1, (S, T, 3)).astype(np.float32)
    es = np.zeros((S, T), bool)
    far = img.copy()
    far[:, 2, 0:8, 0:8] = 255 - far[:, 2, 0:8, 0:8]                       # this step, patch (0, 0)

    def changed(cfg):
        agent = agent_lib.Agent(cfg, seed=3)
        P = ref.params_from_flat(agent.params(), agent.param_layout(), cfg)
        agent.close()
        nt = ref.uni_dims(cfg)[0]
        q = agent_lib.uni_layout(cfg)["patch"][15]
        with torch.no_grad():
            o = [ref.forward_uni(P, cfg, im, drv, act, es)["h"].numpy().reshape(S, T, nt, -1)[:, -1, q] for im in (img, far)]
        return np.abs(o[0] - o[1]).max() > 1e-9

    one = dict(UNI, n_layer=1, in_bins=1, attn_win=1, uni_pyr=2)
    assert not changed(one)
    assert not changed(dict(one, n_layer=2, uni_pyr=0))
    assert changed(dict(one, n_layer=2))


@pytest.fixture(scope="module", params=[(8, 0, 1, 1, 2), (8, 0, 0, 0, 2), (4, 1, 1, 1, 3), (4, 1, 1, 2, 1)],
                ids=["p8_inbins_W1_L2", "p8_L2_registers", "p4_gray_inbins_W1_L3", "p4_gray_inbins_W2_L1"])
def pyr_agent(request):
    agent_lib.set_threads(4)
    patch, gray, in_bins, Wn, L = request.param
    return agent_lib.Agent(dict(UNI, patch=patch, gray=gray, in_bins=in_bins, bins=6 if gray else 8, attn_win=Wn,
                                uni_pyr=L), seed=7)


def test_cpu_loss_and_grads_match_reference(pyr_agent):
    T, N = 16, 3
    ro, data = store_rollout(pyr_agent, T=T, N=N, seed=3, p_done=0.2)
    ids = [0, 2, 4, 5]
    pc = dict(agent_lib.DEFAULT_PPO, use_ppo=0)
    loss_cpp, _ = pyr_agent.debug_grads(ro, np.zeros(N, np.float32), pc, ids)
    c = pyr_agent.config
    P = ref.params_from_flat(pyr_agent.params(), pyr_agent.param_layout(), c)
    z = np.zeros((T, N))
    batch = seq_batch(data, ids, c["ctx"], N, z, z, c["img"] // c["pred_img"])
    loss_ref, _ = ref.ppo_loss(P, c, pc, batch)
    loss_ref.backward()
    assert abs(loss_cpp - loss_ref.item()) < 1e-4 * max(1.0, abs(loss_ref.item())), (loss_cpp, loss_ref.item())
    compare_grads(pyr_agent, P)


@pytest.mark.skipif(not cuda_lib.available(), reason="libvmcuda.so not built")
@pytest.mark.parametrize("cfg", [dict(UNI, in_bins=1, attn_win=1, uni_pyr=2),
                                 dict(UNI, img=16, pred_img=16, ctx=4, attn_win=1, uni_pyr=1),   # short windows
                                 dict(UNI, uni_pyr=1),
                                 dict(UNI, img=64, pred_img=32, patch=8, gray=1, in_bins=1, bins=32, attn_win=1,
                                      uni_pyr=3)],
                         ids=["small_W1_L2", "tiny_short_windows", "small_registers", "gray_inbins_64_W1_L3"])
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


def test_checkpoint_keeps_the_flag(tmp_path):
    agent = agent_lib.Agent(dict(UNI, attn_win=1, uni_pyr=2), seed=2)
    path = str(tmp_path / "a.bin")
    agent.save(path)
    b = agent_lib.Agent(path=path)
    assert b.config["uni_pyr"] == 2 and b.num_params == agent.num_params
    agent.close()
    b.close()

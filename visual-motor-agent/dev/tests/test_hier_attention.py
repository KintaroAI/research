"""Local attention with summary tokens (task 00047, model.attn_win / model.uni_sum): the slot layout, the allow table,
what a patch can see, C++ and GPU against the PyTorch reference, checkpoints. GPU against C++ for the forward paths and
the instinct's candidates: test_cuda_model.py's *_hier models.

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


@pytest.mark.parametrize("img,patch,R", [(32, 8, 0), (32, 8, 1), (32, 8, 2), (32, 8, 4), (64, 8, 1), (64, 8, 2)])
def test_layout(img, patch, R):
    cfg = dict(universal=1, img=img, patch=patch, uni_sum=R, gray=1, bins=4)
    g = img // patch
    lay = agent_lib.uni_layout(cfg)
    kind, index = ref.uni_slots(cfg)
    nsum = g // R if R else 0
    assert lay["nt"] == len(kind) == g * g + 2 + nsum == ref.uni_dims(cfg)[0]
    np.testing.assert_array_equal(lay["kind"], kind)
    assert kind[0] == 0 and kind[-1] == 3 and lay["move"] == len(kind) - 1
    np.testing.assert_array_equal(index[lay["patch"]], np.arange(g * g))     # the patches in order
    np.testing.assert_array_equal(index[lay["summary"]], np.arange(nsum))
    if R:   # a summary right after every R rows: R * g patches, then one summary
        np.testing.assert_array_equal(lay["summary"], [(j + 1) * (R * g + 1) for j in range(nsum)])
    assert len(lay["summary"]) == nsum


def test_allow_table():
    cfg = dict(universal=1, img=64, patch=8, attn_win=1, uni_sum=2)
    a = ref.attn_allow(cfg)
    kind, index = ref.uni_slots(cfg)
    nt = len(kind)
    assert a.shape == (nt, nt) and nt == 64 + 4 + 2
    glob = (kind == 0) | (kind == 3)
    assert a[glob].all() and a[:, glob].all()                              # body and move: to and from everything
    P, Sm = np.flatnonzero(kind == 1), np.flatnonzero(kind == 2)
    rc = np.stack([index[P] // 8, index[P] % 8], 1)
    want = (np.abs(rc[:, None] - rc[None]) <= 1).all(-1)
    np.testing.assert_array_equal(a[np.ix_(P, P)], want)                  # a 3x3 window
    assert a[np.ix_(P, Sm)].all() and a[np.ix_(Sm, Sm)].all()             # a patch reads every summary
    band = rc[:, 0] // 2
    np.testing.assert_array_equal(a[np.ix_(Sm, P)], np.arange(4)[:, None] == band[None])   # a summary its band
    assert a[np.ix_(P, P)].sum(1).min() == 4 and a[np.ix_(P, P)].sum(1).max() == 9          # corners, inside
    # the window alone (no summaries), and summaries alone (registers: no table)
    aw = ref.attn_allow(dict(cfg, uni_sum=0))
    assert aw.shape == (66, 66)
    np.testing.assert_array_equal(aw[1:65, 1:65], want)
    assert ref.attn_allow(dict(cfg, attn_win=0)) is None


@pytest.mark.parametrize("bad", [dict(uni_sum=3), dict(uni_sum=-1), dict(attn_win=-1),
                                 dict(attn_win=1, attn_split=2), dict(uni_sum=1, attn_split=2)])
def test_invalid_configs(bad):
    with pytest.raises(ValueError):
        agent_lib.Agent(dict(UNI, **bad), seed=1)              # a 4x4 grid: uni_sum must divide 4
    with pytest.raises(ValueError):
        agent_lib.Agent(dict(SMALL, patch=8, pred_residual=1, attn_win=1), seed=1)   # not the universal model
    with pytest.raises(ValueError):
        agent_lib.Agent(dict(SMALL, patch=8, pred_residual=1, uni_sum=1), seed=1)


def test_what_a_patch_sees():
    """With one layer a patch sees only its window (a summary's input carries nothing, nor does the body's or the
    move's). With two layers and no summaries its view within the step is 2W wide, so this step's far corner stays
    invisible (an earlier step's reaches it through the body and the move, which are global); with summaries it
    reaches it through the summary of the row above."""
    rng = np.random.default_rng(0)
    S, T = 2, 3
    img = rng.integers(0, 256, (S, T, 32, 32, 3), dtype=np.uint8)
    drv = rng.random((S, T, 2)).astype(np.float32)
    act = rng.uniform(-1, 1, (S, T, 3)).astype(np.float32)
    es = np.zeros((S, T), bool)

    def out(cfg, im):
        agent = agent_lib.Agent(cfg, seed=3)
        P = ref.params_from_flat(agent.params(), agent.param_layout(), cfg)
        agent.close()
        nt = ref.uni_dims(cfg)[0]
        with torch.no_grad():
            return ref.forward_uni(P, cfg, im, drv, act, es)["h"].numpy().reshape(S, T, nt, -1)

    def changed(cfg, im, p):   # the last step's patch p
        q = agent_lib.uni_layout(cfg)["patch"][p]
        return np.abs(out(cfg, img)[:, -1, q] - out(cfg, im)[:, -1, q]).max() > 1e-9

    far_prev = img.copy()
    far_prev[:, 1, 0:8, 0:8] = 255 - far_prev[:, 1, 0:8, 0:8]               # step 1, patch (0, 0)
    far_now = img.copy()
    far_now[:, 2, 0:8, 0:8] = 255 - far_now[:, 2, 0:8, 0:8]                 # this step, patch (0, 0)
    near = img.copy()
    near[:, 1, 16:24, 16:24] = 255 - near[:, 1, 16:24, 16:24]               # step 1, patch (2, 2)
    p33 = 15                                                                 # patch (3, 3): far from (0, 0)
    for in_bins in (0, 1):
        one = dict(UNI, n_layer=1, in_bins=in_bins, attn_win=1, uni_sum=1)
        assert not changed(one, far_prev, p33) and not changed(one, far_now, p33) and changed(one, near, p33)
        two = dict(one, n_layer=2, uni_sum=0)
        assert not changed(two, far_now, p33) and changed(two, far_prev, p33)
        hier = dict(two, uni_sum=1)
        assert changed(hier, far_now, p33)                                   # through the row-0 summary
        assert changed(dict(one, attn_win=0, uni_sum=0), far_prev, p33)      # full attention: directly


def test_summary_tokens_carry_no_values_and_no_target():
    """A summary's input row is all zeros, and the token before it has no target: the loss is the mean over the
    tokens that have one."""
    cfg = dict(UNI, uni_sum=2, in_bins=1)
    rng = np.random.default_rng(1)
    S, T = 2, 3
    img = rng.integers(0, 256, (S, T, 32, 32, 3), dtype=np.uint8)
    v, used = ref.uni_values(cfg, img, rng.random((S, T, 2)), rng.uniform(-1, 1, (S, T, 3)))
    lay = agent_lib.uni_layout(cfg)
    assert (used[lay["summary"]] == 0).all() and (v[:, :, lay["summary"]] == 0).all()
    assert (used[lay["patch"]] == cfg["patch"] ** 2 * 3).all()


@pytest.fixture(scope="module", params=[(8, 0, 1, 1, 1), (8, 0, 0, 0, 2), (8, 0, 1, 1, 0), (4, 1, 1, 1, 2),
                                        (4, 1, 1, 2, 4)],
                ids=["p8_inbins_W1_R1", "p8_R2", "p8_inbins_W1", "p4_gray_inbins_W1_R2", "p4_gray_inbins_W2_R4"])
def hier_agent(request):
    agent_lib.set_threads(4)
    patch, gray, in_bins, Wn, R = request.param
    return agent_lib.Agent(dict(UNI, patch=patch, gray=gray, in_bins=in_bins, bins=6 if gray else 8, attn_win=Wn,
                                uni_sum=R), seed=7)


def test_cpu_loss_and_grads_match_reference(hier_agent):
    T, N = 16, 3
    ro, data = store_rollout(hier_agent, T=T, N=N, seed=3, p_done=0.2)
    ids = [0, 2, 4, 5]
    pc = dict(agent_lib.DEFAULT_PPO, use_ppo=0)
    loss_cpp, _ = hier_agent.debug_grads(ro, np.zeros(N, np.float32), pc, ids)
    c = hier_agent.config
    P = ref.params_from_flat(hier_agent.params(), hier_agent.param_layout(), c)
    z = np.zeros((T, N))
    batch = seq_batch(data, ids, c["ctx"], N, z, z, c["img"] // c["pred_img"])
    loss_ref, _ = ref.ppo_loss(P, c, pc, batch)
    loss_ref.backward()
    assert abs(loss_cpp - loss_ref.item()) < 1e-4 * max(1.0, abs(loss_ref.item())), (loss_cpp, loss_ref.item())
    compare_grads(hier_agent, P)


@pytest.mark.skipif(not cuda_lib.available(), reason="libvmcuda.so not built")
@pytest.mark.parametrize("cfg", [dict(UNI, in_bins=1, attn_win=1, uni_sum=1),
                                 dict(UNI, img=16, pred_img=16, ctx=4, attn_win=1, uni_sum=1),   # short windows
                                 dict(UNI, uni_sum=2),
                                 dict(UNI, in_bins=1, attn_win=1),
                                 dict(UNI, img=64, pred_img=32, patch=8, gray=1, in_bins=1, bins=32, attn_win=1,
                                      uni_sum=1)],
                         ids=["small_W1_R1", "tiny_short_windows", "small_R2", "small_W1", "gray_inbins_64_W1_R1"])
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


def test_checkpoint_keeps_the_flags(tmp_path):
    agent = agent_lib.Agent(dict(UNI, attn_win=1, uni_sum=2), seed=2)
    path = str(tmp_path / "a.bin")
    agent.save(path)
    b = agent_lib.Agent(path=path)
    assert b.config["attn_win"] == 1 and b.config["uni_sum"] == 2 and b.num_params == agent.num_params
    agent.close()
    b.close()

"""Summary tokens with thumbnails (task 00049, model.sum_thumb): each summary's values are its block box-averaged to
one patch. Thumbnails against a direct box average, no leakage, C++ and GPU against the reference, text_dream's own
forward and its thumbnails, checkpoints. The candidates, lookahead, self-forcing and dream tests run thumbnail models
too.

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


def box_thumb(img, y0, x0, fy, fx, ps, gray):
    """One thumbnail by loops: [ps*ps*ch] values (y, x, c), the mean of each fy x fx box of pixels / 255."""
    ch = 1 if gray else 3
    out = np.zeros((ps, ps, ch))
    for y in range(ps):
        for x in range(ps):
            for c in range(ch):
                out[y, x, c] = img[y0 + y * fy:y0 + (y + 1) * fy, x0 + x * fx:x0 + (x + 1) * fx, c].mean() / 255.0
    return out.reshape(-1)


@pytest.mark.parametrize("extra", [dict(uni_pyr=1), dict(uni_pyr=3), dict(uni_sum=1), dict(uni_sum=2)],
                         ids=["pyr1", "pyr3", "rows1", "rows2"])
@pytest.mark.parametrize("gray", [0, 1])
def test_thumbnails_are_box_averages(extra, gray):
    cfg = dict(universal=1, img=64, patch=8, bins=8, gray=gray, sum_thumb=1, **extra)
    rng = np.random.default_rng(1)
    img = rng.integers(0, 256, (2, 3, 64, 64, 3), dtype=np.uint8)
    v, used = ref.uni_values(cfg, img, rng.random((2, 3, 2)), rng.uniform(-1, 1, (2, 3, 3)))
    t = ref.uni_slot_table(cfg)
    V = 64 * (1 if gray else 3)
    for k in np.flatnonzero(t["kind"] == 2):
        assert used[k] == V
        if "uni_sum" in extra:
            R = extra["uni_sum"]
            geo = (t["r"][k] * R * 8, 0, R, 8)
        else:
            b = 2 ** t["level"][k]
            geo = (t["r"][k] * b * 8, t["c"][k] * b * 8, b, b)
        for s in range(2):
            for st in range(3):
                np.testing.assert_allclose(v[s, st, k], box_thumb(img[s, st], *geo, 8, gray), atol=1e-12)
    if "uni_pyr" in extra and extra["uni_pyr"] == 3:   # the global summary: the whole frame at 8x8
        np.testing.assert_allclose(v[0, 0, -2], box_thumb(img[0, 0], 0, 0, 8, 8, 8, gray), atol=1e-12)


def test_no_leakage():
    """A summary's values depend only on its block, which ends before it in the sequence."""
    cfg = dict(universal=1, img=64, patch=8, bins=8, gray=1, sum_thumb=1, uni_pyr=3)
    rng = np.random.default_rng(2)
    img = rng.integers(0, 256, (1, 1, 64, 64, 3), dtype=np.uint8)
    t = ref.uni_slot_table(cfg)
    lay = agent_lib.uni_layout(cfg)
    v0, _ = ref.uni_values(cfg, img, np.zeros((1, 1, 2)), np.zeros((1, 1, 3)))
    for p in (0, 27, 63):   # change one patch: only the summaries whose block holds it change, all after it
        im = img.copy()
        r, c = divmod(p, 8)
        im[0, 0, r * 8:(r + 1) * 8, c * 8:(c + 1) * 8] = 255 - im[0, 0, r * 8:(r + 1) * 8, c * 8:(c + 1) * 8]
        v1, _ = ref.uni_values(cfg, im, np.zeros((1, 1, 2)), np.zeros((1, 1, 3)))
        for k in lay["summary"]:
            b = 2 ** t["level"][k]
            inside = r // b == t["r"][k] and c // b == t["c"][k]
            assert (np.abs(v1[0, 0, k] - v0[0, 0, k]).max() > 0) == inside
            if inside:
                assert k > lay["patch"][p]


def test_invalid_without_summaries():
    with pytest.raises(ValueError):
        agent_lib.Agent(dict(UNI, sum_thumb=1), seed=1)
    with pytest.raises(ValueError):
        agent_lib.Agent(dict(UNI, sum_thumb=2, uni_pyr=1), seed=1)


@pytest.fixture(scope="module", params=[dict(attn_win=1, uni_pyr=2), dict(uni_pyr=2), dict(attn_win=1, uni_sum=1),
                                        dict(patch=4, gray=1, bins=6, attn_win=1, uni_pyr=3)],
                ids=["W1_pyr2", "pyr2_full", "W1_rows", "p4_gray_W1_pyr3"])
def thumb_agent(request):
    agent_lib.set_threads(4)
    return agent_lib.Agent(dict(UNI, in_bins=1, sum_thumb=1, **request.param), seed=7)


def test_cpu_loss_and_grads_match_reference(thumb_agent):
    T, N = 16, 3
    ro, data = store_rollout(thumb_agent, T=T, N=N, seed=3, p_done=0.2)
    ids = [0, 2, 4, 5]
    pc = dict(agent_lib.DEFAULT_PPO, use_ppo=0)
    loss_cpp, _ = thumb_agent.debug_grads(ro, np.zeros(N, np.float32), pc, ids)
    c = thumb_agent.config
    P = ref.params_from_flat(thumb_agent.params(), thumb_agent.param_layout(), c)
    z = np.zeros((T, N))
    batch = seq_batch(data, ids, c["ctx"], N, z, z, c["img"] // c["pred_img"])
    loss_ref, _ = ref.ppo_loss(P, c, pc, batch)
    loss_ref.backward()
    assert abs(loss_cpp - loss_ref.item()) < 1e-4 * max(1.0, abs(loss_ref.item())), (loss_cpp, loss_ref.item())
    compare_grads(thumb_agent, P)


@pytest.mark.skipif(not cuda_lib.available(), reason="libvmcuda.so not built")
@pytest.mark.parametrize("cfg", [dict(UNI, in_bins=1, attn_win=1, uni_pyr=2, sum_thumb=1),
                                 dict(UNI, img=16, pred_img=16, ctx=4, attn_win=1, uni_pyr=1, sum_thumb=1),
                                 dict(UNI, uni_sum=2, sum_thumb=1),
                                 dict(UNI, img=64, pred_img=32, patch=8, gray=1, in_bins=1, bins=32, attn_win=1,
                                      uni_pyr=3, sum_thumb=1),
                                 dict(UNI, img=16, pred_img=8, patch=2, gray=1, in_bins=1, bins=16, attn_win=1,
                                      uni_pyr=3, sum_thumb=1)],
                         ids=["small_W1_pyr2", "tiny_short_windows", "small_rows", "gray_inbins_64_pyr3", "text_like"])
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


def test_text_dream_thumbnails_and_forward():
    """text_dream's Uni: fill_thumbs equals the reference's thumbnails of the same page, and its forward the
    reference's (a text-like gray model, 2x2 letters)."""
    from reference import text_dream
    cfg = dict(UNI, img=16, pred_img=8, patch=2, gray=1, in_bins=1, bins=16, attn_win=1, uni_pyr=3, sum_thumb=1, ctx=4)
    agent = agent_lib.Agent(cfg, seed=5)
    c = agent.config
    uni = text_dream.Uni(agent, "cpu")
    rng = np.random.default_rng(3)
    lev = rng.integers(0, 16, (2, 4, 8, 8)) * 17                              # a letter's grey per 2x2 block
    img = np.repeat(np.repeat(lev, 2, -1), 2, -2)[..., None].repeat(3, -1).astype(np.uint8)
    act = rng.uniform(-1, 1, (2, 4, 3)).astype(np.float32)
    v, _ = ref.uni_values(c, img, np.zeros((2, 4, 2)), act)
    vt = torch.tensor(v, dtype=torch.float32)
    w = vt.clone()
    w[:, :, uni.ss] = 0
    uni.fill_thumbs(w)
    torch.testing.assert_close(w, vt, atol=1e-6, rtol=0)
    w[:, -1, uni.ss] = 0
    uni.fill_thumbs(w, -1)                                                     # the last step only
    torch.testing.assert_close(w, vt, atol=1e-6, rtol=0)
    P = ref.params_from_flat(agent.params(), agent.param_layout(), c)
    want = ref.forward_uni(P, c, img, np.zeros((2, 4, 2), np.float32), act, np.eye(1, 4, dtype=np.uint8).repeat(2, 0))
    got = uni.logits(vt).double()
    assert (got - want["logits"].detach()).abs().max().item() < 1e-3
    agent.close()


def test_checkpoint_keeps_the_flag(tmp_path):
    agent = agent_lib.Agent(dict(UNI, attn_win=1, uni_pyr=2, sum_thumb=1), seed=2)
    path = str(tmp_path / "a.bin")
    agent.save(path)
    b = agent_lib.Agent(path=path)
    assert b.config["sum_thumb"] == 1 and b.config["uni_pyr"] == 2
    agent.close()
    b.close()

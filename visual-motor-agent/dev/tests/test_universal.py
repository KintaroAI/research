"""The universal model (model.universal=1, task 00023): C++ against the PyTorch reference (loss, every gradient, the
forward outputs), the targets and two-hot encoding by hand, causality, checkpoints, and the update.

Requires libvmagent.so and torch (CPU) in the venv.
"""
import os
import sys

import numpy as np
import pytest

torch = pytest.importorskip("torch")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import agent_lib  # noqa: E402
from reference import torch_reference as ref  # noqa: E402
from test_model_reference import SMALL, compare_grads, seq_batch  # noqa: E402
from test_patch_tokens import store_rollout  # noqa: E402

# 32x32 frames: 16 patches of 8x8 (V = 192 values each), 18 tokens per step, 144 per context of 8 steps; 8 bins
UNI = dict(SMALL, patch=8, universal=1, bins=8)


@pytest.fixture(scope="module", params=[(8, 8, 0, 0), (16, 5, 0, 0), (4, 8, 0, 0), (8, 8, 1, 0), (4, 6, 1, 0),
                                        (8, 8, 0, 1), (16, 5, 0, 1), (4, 6, 1, 1)],
                ids=["p8_bins8", "p16_bins5", "p4_bins8", "p8_gray", "p4_gray", "p8_bins8_inbins", "p16_bins5_inbins",
                     "p4_gray_inbins"])
def uni_agent(request):
    agent_lib.set_threads(4)
    patch, bins, gray, in_bins = request.param
    return agent_lib.Agent(dict(UNI, patch=patch, bins=bins, gray=gray, in_bins=in_bins), seed=7)


def ref_batch(agent, data, ids, N):
    cfg = agent.config
    T = data["images"].shape[0]
    z = np.zeros((T, N))
    return seq_batch(data, ids, cfg["ctx"], N, z, z, cfg["img"] // cfg["pred_img"])


@pytest.mark.parametrize("threads", [1, 4, 16])
def test_loss_and_grads_match_reference(uni_agent, threads):
    agent_lib.set_threads(threads)
    T, N = 16, 3
    ro, data = store_rollout(uni_agent, T=T, N=N, seed=3, p_done=0.2)   # episodes also start inside windows
    ids = [0, 2, 4, 5]
    pc = dict(agent_lib.DEFAULT_PPO, use_ppo=0)
    loss_cpp, st = uni_agent.debug_grads(ro, np.zeros(N, np.float32), pc, ids)
    P = ref.params_from_flat(uni_agent.params(), uni_agent.param_layout(), uni_agent.config)
    loss_ref, terms = ref.ppo_loss(P, uni_agent.config, pc, ref_batch(uni_agent, data, ids, N))
    loss_ref.backward()
    assert abs(loss_cpp - loss_ref.item()) < 1e-4 * max(1.0, abs(loss_ref.item())), (loss_cpp, loss_ref.item())
    pt = terms["per_token"].detach().numpy()                       # [S, T, nt]
    nt = pt.shape[2]
    for name, sl in (("body", nt - 1), ("move", nt - 2)):
        assert abs(st["uni_ce_" + name] - pt[:, :, sl].mean()) < 1e-4, name
    assert abs(st["uni_ce_patch"] - pt[:, :, :nt - 2].mean()) < 1e-4
    compare_grads(uni_agent, P)


def test_forward_matches_reference(uni_agent):
    cfg = uni_agent.config
    rng = np.random.default_rng(4)
    S, T = 2, cfg["ctx"]
    img = rng.integers(0, 256, (S, T, cfg["img"], cfg["img"], 3), dtype=np.uint8)
    drv = rng.random((S, T, 2)).astype(np.float32)
    act = np.concatenate([rng.uniform(-1.3, 1.3, (S, T, 2)), rng.random((S, T, 1)) < 0.4], -1).astype(np.float32)
    es = np.zeros((S, T), np.uint8)
    es[:, 0] = es[:, 5] = 1
    out = uni_agent.forward_uni(img, drv, act, es)
    P = ref.params_from_flat(uni_agent.params(), uni_agent.param_layout(), cfg, dtype=torch.float64)
    with torch.no_grad():
        r = ref.forward_uni(P, cfg, img, drv, act, es)
    nt, V, K = ref.uni_dims(cfg)
    prob = r["logits"].softmax(-1).numpy()                          # [S, T*nt, V, K]
    ev = (prob * (np.arange(K) / (K - 1))).sum(-1).reshape(S, T, nt, V)
    np.testing.assert_allclose(out["values"], ev, atol=2e-5)
    np.testing.assert_allclose(out["move_probs"], prob.reshape(S, T, nt, V, K)[:, :, nt - 2, :3], atol=2e-5)
    nb = uni_agent.forward_uni(img, drv, act, es, values=False, move_probs=False, next_body=True)["next_body"]
    np.testing.assert_allclose(nb, ev[:, :, nt - 1, :2], atol=2e-5)


def test_causality(uni_agent):
    """A token's prediction depends only on itself and the earlier tokens of its context: changing a later patch, a
    later step or the move of a step changes nothing before it."""
    cfg = uni_agent.config
    rng = np.random.default_rng(5)
    T = cfg["ctx"]
    nt, V, K = ref.uni_dims(cfg)
    img = rng.integers(0, 256, (1, T, cfg["img"], cfg["img"], 3), dtype=np.uint8)
    drv = rng.random((1, T, 2)).astype(np.float32)
    act = rng.uniform(-1, 1, (1, T, 3)).astype(np.float32)
    es = np.zeros((1, T), np.uint8)
    es[0, 0] = 1
    f = lambda **kw: uni_agent.forward_uni(kw.get("img", img), drv, kw.get("act", act), es)["values"]
    base = f()
    a2 = act.copy()
    a2[0, 3] = -a2[0, 3]                                             # the move of step 3: only its own token and later
    o = f(act=a2)
    np.testing.assert_array_equal(o[0, :3], base[0, :3])
    np.testing.assert_array_equal(o[0, 3, :nt - 1], base[0, 3, :nt - 1])
    assert not np.array_equal(o[0, 3, nt - 1], base[0, 3, nt - 1])   # the move's own prediction (the next drives)
    ps = cfg["patch"]
    im2 = img.copy()
    im2[0, 4, -ps:, -ps:] = 255 - im2[0, 4, -ps:, -ps:]              # the last patch of step 4
    o = f(img=im2)
    np.testing.assert_array_equal(o[0, :4], base[0, :4])
    np.testing.assert_array_equal(o[0, 4, :nt - 2], base[0, 4, :nt - 2])
    assert not np.array_equal(o[0, 4, nt - 2], base[0, 4, nt - 2])   # its own prediction (the move) changes


def test_context_restarts_at_episode_starts(uni_agent):
    cfg = uni_agent.config
    rng = np.random.default_rng(6)
    T = cfg["ctx"]
    img = rng.integers(0, 256, (1, T, cfg["img"], cfg["img"], 3), dtype=np.uint8)
    drv = rng.random((1, T, 2)).astype(np.float32)
    act = rng.uniform(-1, 1, (1, T, 3)).astype(np.float32)
    es = np.zeros((1, T), np.uint8)
    es[0, 0] = es[0, 4] = 1
    base = uni_agent.forward_uni(img, drv, act, es)["values"]
    im2 = img.copy()
    im2[0, :4] = 255 - im2[0, :4]                                    # before the episode start at step 4
    o = uni_agent.forward_uni(im2, drv, act, es)["values"]
    np.testing.assert_array_equal(o[0, 4:], base[0, 4:])


def test_two_hot_and_targets_by_hand():
    t = ref.two_hot(np.array([0.0, 1.0, 0.5, 0.3]), 5)               # centres 0, .25, .5, .75, 1
    np.testing.assert_allclose(t, [[1, 0, 0, 0, 0], [0, 0, 0, 0, 1], [0, 0, 1, 0, 0], [0, 0.8, 0.2, 0, 0]])
    cfg = dict(UNI, patch=16)                                        # 4 patches of 16x16
    img = np.zeros((1, 1, 32, 32, 3), np.uint8)
    img[0, 0, 0, 16] = (255, 0, 51)                                  # the first pixel of patch 1 (top right)
    v, used = ref.uni_values(cfg, img, np.array([[[0.3, 0.7]]]), np.array([[[0.5, -2.0, 1.0]]]))
    np.testing.assert_allclose(v[0, 0, 0, :2], [0.3, 0.7])
    np.testing.assert_allclose(v[0, 0, 2, :3], [1.0, 0.0, 0.2])
    np.testing.assert_allclose(v[0, 0, 5, :3], [0.75, 0.0, 1.0])     # (dx+1)/2, clipped dy, click
    assert list(used) == [2, 768, 768, 768, 768, 3]


def test_checkpoint_roundtrip_and_layout(uni_agent, tmp_path):
    path = str(tmp_path / "ckpt.bin")
    uni_agent.save(path, with_optimizer=True)
    b = agent_lib.Agent.load(path)
    assert b.config == uni_agent.config and b.config["universal"] == 1
    np.testing.assert_array_equal(b.params(), uni_agent.params())
    b.close()
    sizes = {n: s for n, _, s in uni_agent.param_layout()}
    for n in ("pi_w", "pr1w", "drw", "pt_w", "tok_type", "drv_w", "enc_c1w", "in_w"):
        assert sizes[n] == 0, n
    assert sizes["uni_out_w"] > 0 and sizes["wpe"] > 0


def test_update_lowers_the_loss():
    agent_lib.set_threads(4)
    a = agent_lib.Agent(UNI, seed=2)
    ro, _ = store_rollout(a, T=16, N=4, seed=8)
    pc = dict(agent_lib.DEFAULT_PPO, use_ppo=0, epochs=1, minibatches=1, lr=3e-3)
    first = a.update(ro, np.zeros(4, np.float32), pc)
    for _ in range(30):
        last = a.update(ro, np.zeros(4, np.float32), pc)
    ce = lambda st: st["uni_ce_body"] + st["uni_ce_patch"] + st["uni_ce_move"]
    assert ce(last) < 0.9 * ce(first)


@pytest.mark.parametrize("bad", [dict(universal=1, patch=0), dict(universal=1, patch=8, bins=1),
                                 dict(universal=1, patch=8, act_token=1, pred_residual=1), dict(patch=8, gray=1),
                                 dict(universal=1, patch=1, gray=1), dict(patch=8, in_bins=1),
                                 dict(universal=1, patch=8, in_bins=2)])
def test_invalid_configs(bad):
    with pytest.raises(Exception):
        agent_lib.Agent(dict(SMALL, **bad), seed=1)


def test_gray_reads_the_first_channel():
    cfg = dict(UNI, patch=16, gray=1)
    img = np.zeros((1, 1, 32, 32, 3), np.uint8)
    img[0, 0, 1, 2] = (102, 7, 9)
    v, used = ref.uni_values(cfg, img, np.zeros((1, 1, 2)), np.zeros((1, 1, 3)))
    assert v.shape[-1] == 256 and list(used) == [2, 256, 256, 256, 256, 3]
    np.testing.assert_allclose(v[0, 0, 1, 1 * 16 + 2], 0.4)            # patch 0, (y 1, x 2): 102 / 255



def test_bin_embedding_input_is_the_sum_of_bin_embeddings():
    """in_bins (task 00036): a token's input is b + the sum over its used values of (1 - w) E[v, lo] + w E[v, lo + 1],
    E[v, k] = W_in's column v * K + k; a value on a bin centre picks one embedding (a lookup); init at 0.02."""
    torch = pytest.importorskip("torch")
    agent = agent_lib.Agent(dict(UNI, patch=8, bins=8, gray=1, in_bins=1), seed=3)
    cfg = agent.config
    nt, V, K = ref.uni_dims(cfg)
    lay = {n: (o, k) for n, o, k in agent.param_layout()}
    o, k = lay["uni_in_w"]
    assert k == cfg["d_model"] * V * K
    W = np.asarray(agent.params()[o:o + k]).reshape(cfg["d_model"], V, K)
    assert 0.015 < W.std() < 0.025
    rng = np.random.default_rng(0)
    v = rng.random((2, 1, nt, V))
    v[..., 5] = 3 / 7                                                   # a bin centre
    _, used = ref.uni_values(cfg, np.zeros((2, 1, 32, 32, 3), np.uint8), np.zeros((2, 1, 2)), np.zeros((2, 1, 3)))
    m = np.arange(V)[None] < used[:, None]
    x = ref.two_hot(v, K) * m[..., None]
    got = x.reshape(2, nt, V * K) @ W.reshape(cfg["d_model"], V * K).T
    want = np.zeros((2, nt, cfg["d_model"]))
    for s in range(2):
        for t in range(nt):
            for j in range(used[t]):
                p = v[s, 0, t, j] * (K - 1)
                lo = min(int(np.floor(p)), K - 2)
                w = p - lo
                want[s, t] += (1 - w) * W[:, j, lo] + w * W[:, j, lo + 1]
    assert np.allclose(got, want)
    assert np.allclose(x[0, 0, 1, 5], np.eye(K)[3])                     # on a centre: one-hot
    agent.close()

"""Patch tokens (model.patch, vm-00010): C++ vs the PyTorch reference (forward, losses, every gradient), block-causal
attention, the attention weights, checkpoints.

Requires libvmagent.so and torch (CPU) in the venv.
"""
import os
import sys

import numpy as np
import pytest

torch = pytest.importorskip("torch")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import agent_lib  # noqa: E402
from reference import torch_reference as ref  # noqa: E402
from test_model_reference import SMALL, compare_grads, seq_batch  # noqa: E402

# 32x32 frames: a 4x4 grid of 8x8 patches, each predicting a 4x4 cell of the 16x16 next frame; 19 tokens per step
PATCH = dict(SMALL, patch=8)
NT = (PATCH["img"] // PATCH["patch"]) ** 2 + 3


def store_rollout(agent, T=16, N=3, seed=0, p_done=0.15):
    """Fill a rollout with random experience through store() (patch tokens do not act with the KV cache)."""
    rng = np.random.default_rng(seed)
    S = agent.config["img"]
    ro = agent.new_rollout(T, N)
    keys = ("images", "drives", "prev_action", "prev_reward", "ep_start", "action", "logp", "value", "reward", "done",
            "next_images", "next_drives")
    data = {k: [] for k in keys}
    es = np.ones(N, np.uint8)
    pa = np.zeros((N, 3), np.float32)
    pr = np.zeros(N, np.float32)
    img = rng.integers(0, 256, (N, S, S, 3), dtype=np.uint8)
    drv = rng.random((N, 2), dtype=np.float32)
    for t in range(T):
        a = np.concatenate([rng.normal(0, 0.7, (N, 2)), rng.random((N, 1)) < 0.3], 1).astype(np.float32)
        ro.store(t, img, drv, pa, pr, es, a)
        reward = (rng.random(N) < 0.2).astype(np.float32)
        done = (rng.random(N) < p_done).astype(np.uint8)
        nimg = rng.integers(0, 256, (N, S, S, 3), dtype=np.uint8)
        ndrv = rng.random((N, 2), dtype=np.float32)
        ro.outcome(t, reward, done, nimg, ndrv)
        zero = np.zeros(N, np.float32)
        for k, v in zip(keys, (img, drv, pa, pr, es, a, zero, zero, reward, done, nimg, ndrv)):
            data[k].append(np.array(v))
        pa = np.stack([np.clip(a[:, 0], -1, 1), np.clip(a[:, 1], -1, 1), a[:, 2]], 1).astype(np.float32)
        pa[done.astype(bool)] = 0
        pr = rng.random(N).astype(np.float32)          # the scalar slot (the hindsight command in online.py)
        es = done.copy()
        img = np.where(done[:, None, None, None].astype(bool), rng.integers(0, 256, nimg.shape, dtype=np.uint8), nimg)
        drv = ndrv
    return ro, {k: np.stack(v) for k, v in data.items()}


def random_inputs(cfg, S=2, seed=1, start=5):
    rng = np.random.default_rng(seed)
    shp = (S, cfg["ctx"])
    es = np.zeros(shp, np.uint8)
    es[:, 0] = 1
    es[:, start] = 1
    return dict(images=rng.integers(0, 256, shp + (cfg["img"], cfg["img"], 3), dtype=np.uint8),
                drives=rng.random(shp + (2,), dtype=np.float32),
                prev_action=rng.uniform(-1, 1, shp + (3,)).astype(np.float32),
                prev_reward=rng.random(shp).astype(np.float32), ep_start=es,
                pred_actions=rng.uniform(-1, 1, shp + (3,)).astype(np.float32))


@pytest.fixture(scope="module", params=[(1, 0, 0), (0, 0, 0), (1, 5, 0), (1, 0, 1)],
                ids=["residual", "sigmoid", "policy_hidden", "one_frame"])
def patch_agent(request):
    agent_lib.set_threads(4)
    residual, pih, one = request.param
    return agent_lib.Agent(dict(PATCH, pred_residual=residual, pi_hidden=pih, one_frame=one), seed=5)


@pytest.mark.parametrize("pc_over", [
    dict(use_ppo=0, pred_coef=1.0, drive_coef=0.5, move_coef=1.5),                       # next state + next move
    dict(use_ppo=0, pred_coef=2.0, drive_coef=1.0, pred_changed_weight=9.0),             # changed-pixel weighting
    dict(use_ppo=0, pred_coef=1.0, drive_coef=0.5, move_coef=1.0, pi_detach=1),          # policy head detached
    dict(use_ppo=0, pred_coef=1.0, drive_coef=0.5, move_coef=1.0, pred_backbone_scale=0.3),
])
@pytest.mark.parametrize("threads", [4, 16])
def test_patch_loss_and_grads_match_reference(patch_agent, pc_over, threads):
    agent_lib.set_threads(threads)
    cfg = patch_agent.config
    T, N = 16, 3
    ro, data = store_rollout(patch_agent, T=T, N=N, seed=7)
    pc = dict(agent_lib.DEFAULT_PPO, **pc_over)
    ids = [0, 2, 4, 5]
    loss_cpp, _ = patch_agent.debug_grads(ro, np.zeros(N, np.float32), pc, ids)
    zeros = np.zeros((T, N))
    batch = seq_batch(data, ids, cfg["ctx"], N, zeros, zeros, cfg["img"] // cfg["pred_img"])
    P = ref.params_from_flat(patch_agent.params(), patch_agent.param_layout(), cfg)
    loss_ref, _ = ref.ppo_loss(P, cfg, pc, batch)
    loss_ref.backward()
    assert abs(loss_cpp - loss_ref.item()) < 1e-4 * max(1.0, abs(loss_ref.item())), (loss_cpp, loss_ref.item())
    compare_grads(patch_agent, P)


def test_patch_forward_matches_reference(patch_agent):
    cfg = patch_agent.config
    x = random_inputs(cfg)
    out = patch_agent.forward(x["images"], x["drives"], x["prev_action"], x["prev_reward"], x["ep_start"],
                              pred_actions=x["pred_actions"])
    P = ref.params_from_flat(patch_agent.params(), patch_agent.param_layout(), cfg)
    with torch.no_grad():
        r = ref.forward(P, cfg, x["images"], x["drives"], x["prev_action"], x["prev_reward"], x["ep_start"],
                        pred_actions=x["pred_actions"])
    pi = r["pi"].numpy()
    np.testing.assert_allclose(out["z"], r["z"].numpy(), atol=2e-5)
    np.testing.assert_allclose(out["mu"], np.tanh(pi[..., :2]), atol=2e-5)
    np.testing.assert_allclose(out["click_p"], 1 / (1 + np.exp(-pi[..., 2])), atol=2e-5)
    np.testing.assert_allclose(out["pred_frames"], r["pred_frames"].numpy().transpose(0, 1, 3, 4, 2), atol=2e-5)
    np.testing.assert_allclose(out["pred_drives"], r["pred_drives"].numpy(), atol=2e-5)


# Other patch sizes (vm-00014): a fine 8x8 grid of 4x4 patches (67 tokens per step, 536 per context of 8 steps, as
# 8x8 patches of 64x64 frames) and the whole frame as one patch (4 tokens per step)
@pytest.fixture(scope="module", params=[4, 32], ids=["fine", "whole_frame"])
def sized_agent(request):
    agent_lib.set_threads(4)
    return agent_lib.Agent(dict(PATCH, patch=request.param, pred_residual=1, pi_hidden=5), seed=6)


@pytest.mark.parametrize("pc_over", [dict(use_ppo=0, pred_coef=1.0, drive_coef=0.5, move_coef=1.5),
                                     dict(use_ppo=0, pred_coef=2.0, drive_coef=1.0, pred_changed_weight=9.0,
                                          pi_detach=1, drive_changed_weight=4.0, drive_change_eps=0.3)])
def test_patch_sizes_loss_and_grads_match_reference(sized_agent, pc_over):
    cfg = sized_agent.config
    T, N = 16, 3
    ro, data = store_rollout(sized_agent, T=T, N=N, seed=8)
    pc = dict(agent_lib.DEFAULT_PPO, **pc_over)
    ids = [0, 2, 4, 5]
    loss_cpp, _ = sized_agent.debug_grads(ro, np.zeros(N, np.float32), pc, ids)
    zeros = np.zeros((T, N))
    batch = seq_batch(data, ids, cfg["ctx"], N, zeros, zeros, cfg["img"] // cfg["pred_img"])
    P = ref.params_from_flat(sized_agent.params(), sized_agent.param_layout(), cfg)
    loss_ref, _ = ref.ppo_loss(P, cfg, pc, batch)
    loss_ref.backward()
    assert abs(loss_cpp - loss_ref.item()) < 1e-4 * max(1.0, abs(loss_ref.item())), (loss_cpp, loss_ref.item())
    compare_grads(sized_agent, P)


def test_patch_sizes_forward_matches_reference(sized_agent):
    test_patch_forward_matches_reference(sized_agent)


# The act token (task 00020): a learned query per step that the policy reads, trained by its own predictions of the
# next drives and the coarse next frame
@pytest.fixture(scope="module", params=[0, 5], ids=["act_linear_head", "act_hidden_head"])
def act_agent(request):
    agent_lib.set_threads(4)
    return agent_lib.Agent(dict(PATCH, pred_residual=1, pi_hidden=request.param, act_token=1), seed=9)


@pytest.mark.parametrize("pc_over", [
    dict(use_ppo=0, pred_coef=1.0, drive_coef=0.5, move_coef=1.5, act_coef=0.7),
    dict(use_ppo=0, pred_coef=1.0, drive_coef=0.5, move_coef=1.0, pi_detach=1, act_coef=2.0,
         drive_changed_weight=4.0, drive_change_eps=0.3),
    dict(use_ppo=0, pred_coef=1.0, drive_coef=0.5, move_coef=1.0, pi_detach=1, act_coef=1.0, pred_backbone_scale=0.3),
    dict(use_ppo=0, pred_coef=1.0, drive_coef=0.5, move_coef=1.0, act_coef=0.0),
])
@pytest.mark.parametrize("threads", [4, 16])
def test_act_token_loss_and_grads_match_reference(act_agent, pc_over, threads):
    agent_lib.set_threads(threads)
    cfg = act_agent.config
    T, N = 16, 3
    ro, data = store_rollout(act_agent, T=T, N=N, seed=11)
    pc = dict(agent_lib.DEFAULT_PPO, **pc_over)
    ids = [0, 2, 4, 5]
    loss_cpp, st = act_agent.debug_grads(ro, np.zeros(N, np.float32), pc, ids)
    zeros = np.zeros((T, N))
    batch = seq_batch(data, ids, cfg["ctx"], N, zeros, zeros, cfg["img"] // cfg["pred_img"])
    P = ref.params_from_flat(act_agent.params(), act_agent.param_layout(), cfg)
    loss_ref, terms = ref.ppo_loss(P, cfg, pc, batch)
    loss_ref.backward()
    assert abs(loss_cpp - loss_ref.item()) < 1e-4 * max(1.0, abs(loss_ref.item())), (loss_cpp, loss_ref.item())
    assert abs(st["act_frame_mse"] - terms["act_frame"]) < 1e-5 and abs(st["act_drive_mse"] - terms["act_drive"]) < 1e-5
    compare_grads(act_agent, P)


def test_act_token_forward_and_causality(act_agent):
    test_patch_forward_matches_reference(act_agent)
    cfg = act_agent.config
    x = random_inputs(cfg)
    f = lambda im: act_agent.forward(im, x["drives"], x["prev_action"], x["prev_reward"], x["ep_start"],
                                     pred_actions=x["pred_actions"])
    base = f(x["images"])
    im = x["images"].copy()
    im[:, 6:] = 255 - im[:, 6:]
    o = f(im)
    np.testing.assert_array_equal(o["z"][:, :6], base["z"][:, :6])
    im = x["images"].copy()
    im[:, 3, :8, :8] = 255 - im[:, 3, :8, :8]     # one patch of step 3: the act token of step 3 sees it
    o = f(im)
    assert not np.array_equal(o["z"][:, 3], base["z"][:, 3])
    np.testing.assert_array_equal(o["z"][:, :3], base["z"][:, :3])


def test_act_token_detached_move_loss_trains_the_head_only(act_agent):
    agent_lib.set_threads(4)
    ro, _ = store_rollout(act_agent, T=16, N=3, seed=5)
    pc = dict(agent_lib.DEFAULT_PPO, use_ppo=0, pred_coef=0.0, drive_coef=0.0, move_coef=1.0, pi_detach=1)
    act_agent.debug_grads(ro, np.zeros(3, np.float32), pc, [0, 2, 4, 5])
    g = act_agent.grads()
    for name, off, size in act_agent.param_layout():
        nz = int(np.count_nonzero(g[off:off + size]))
        if name in ("pi_w", "pi_b", "pi_h_w", "pi_h_b"):
            assert nz > 0 or size == 0, name
        elif name != "logstd":
            assert nz == 0, (name, nz)


def test_act_token_aux_loss_trains_the_transformer(act_agent):
    """With only the act token's own losses, gradients reach its embedding, the Transformer and the patch embedding,
    but not the patch prediction heads or the policy."""
    agent_lib.set_threads(4)
    ro, _ = store_rollout(act_agent, T=16, N=3, seed=6)
    pc = dict(agent_lib.DEFAULT_PPO, use_ppo=0, pred_coef=1e-30, drive_coef=0.0, move_coef=0.0, act_coef=1.0)
    act_agent.debug_grads(ro, np.zeros(3, np.float32), pc, [0, 2, 4, 5])
    g = act_agent.grads()
    lay = {name: (off, size) for name, off, size in act_agent.param_layout()}
    nz = lambda n: np.abs(g[lay[n][0]:lay[n][0] + lay[n][1]]).max() if lay[n][1] else 0.0
    for n in ("act_e", "ax1w", "axdw", "axfw", "qkvw", "pt_w"):
        assert nz(n) > 0, n
    for n in ("pi_w", "drw"):
        assert nz(n) == 0, n


def test_act_token_checkpoint_roundtrip(act_agent, tmp_path):
    path = str(tmp_path / "ckpt.bin")
    act_agent.save(path, with_optimizer=True)
    b = agent_lib.Agent.load(path)
    assert b.config == act_agent.config and b.config["act_token"] == 1
    np.testing.assert_array_equal(b.params(), act_agent.params())
    b.close()


def test_patch_causality(patch_agent):
    """A step's outputs depend on its own step and the earlier steps of its context block only."""
    cfg = patch_agent.config
    x = random_inputs(cfg)
    f = lambda **kw: patch_agent.forward(*(kw.get(k, x[k]) for k in ("images", "drives", "prev_action",
                                                                    "prev_reward", "ep_start")),
                                         pred_actions=kw.get("pred_actions", x["pred_actions"]))
    base = f()
    im = x["images"].copy()
    im[:, 6:] = 255 - im[:, 6:]
    o = f(images=im)
    np.testing.assert_array_equal(o["z"][:, :6], base["z"][:, :6])
    np.testing.assert_array_equal(o["pred_frames"][:, :6], base["pred_frames"][:, :6])
    im = x["images"].copy()
    im[:, :5] = 255 - im[:, :5]               # before the context start at step 5
    o = f(images=im)
    np.testing.assert_array_equal(o["z"][:, 5:], base["z"][:, 5:])
    acts = x["pred_actions"].copy()
    acts[:, 3] = -acts[:, 3]                  # the move at t does not see a_t; the prediction does
    o = f(pred_actions=acts)
    np.testing.assert_array_equal(o["z"], base["z"])
    assert not np.array_equal(o["pred_frames"][:, 3], base["pred_frames"][:, 3])
    im = x["images"].copy()
    im[:, 3, :8, :8] = 255 - im[:, 3, :8, :8]  # one patch of step 3: the move token of step 3 sees it
    o = f(images=im)
    assert not np.array_equal(o["z"][:, 3], base["z"][:, 3])
    np.testing.assert_array_equal(o["z"][:, :3], base["z"][:, :3])


def test_patch_attention_weights(patch_agent):
    cfg = patch_agent.config
    T = cfg["ctx"]
    x = random_inputs(cfg, S=1)
    att = patch_agent.attention(x["images"][0], x["drives"][0], x["prev_action"][0], x["prev_reward"][0],
                                x["ep_start"][0])
    assert att.shape == (cfg["n_layer"], cfg["n_head"], T * NT, T * NT)
    np.testing.assert_allclose(att.sum(-1), 1.0, atol=1e-5)
    step = np.repeat(np.arange(T), NT)
    first = step.copy() if cfg["one_frame"] else np.where(step >= 5, 5, 0)
    allowed = (step[None, :] <= step[:, None]) & (step[None, :] >= first[:, None])
    assert np.all(att[..., ~allowed] == 0)
    assert np.all(att[..., allowed] > 0)


def test_patch_detached_move_loss_trains_the_head_only(patch_agent):
    agent_lib.set_threads(4)
    ro, _ = store_rollout(patch_agent, T=16, N=3, seed=5)
    pc = dict(agent_lib.DEFAULT_PPO, use_ppo=0, pred_coef=0.0, drive_coef=0.0, move_coef=1.0, pi_detach=1)
    patch_agent.debug_grads(ro, np.zeros(3, np.float32), pc, [0, 2, 4, 5])
    g = patch_agent.grads()
    for name, off, size in patch_agent.param_layout():
        nz = int(np.count_nonzero(g[off:off + size]))
        if name in ("pi_w", "pi_b", "pi_h_w", "pi_h_b"):
            assert nz > 0 or size == 0, name
        elif name != "logstd":
            assert nz == 0, (name, nz)


def test_patch_checkpoint_roundtrip(patch_agent, tmp_path):
    path = str(tmp_path / "ckpt.bin")
    patch_agent.save(path, with_optimizer=True)
    b = agent_lib.Agent.load(path)
    assert b.config == patch_agent.config and b.config["patch"] == PATCH["patch"]
    np.testing.assert_array_equal(b.params(), patch_agent.params())
    x = random_inputs(patch_agent.config)
    args = [x[k] for k in ("images", "drives", "prev_action", "prev_reward", "ep_start")]
    np.testing.assert_array_equal(b.forward(*args)["mu"], patch_agent.forward(*args)["mu"])
    b.close()


def test_patch_model_has_no_cnn_paths(patch_agent):
    with pytest.raises(ValueError):
        patch_agent.encode(np.zeros((1, 32, 32, 3), np.uint8))
    ctx = patch_agent.new_context(1, seed=0)
    with pytest.raises(ValueError):
        patch_agent.act(ctx, np.zeros((1, 32, 32, 3), np.uint8), np.zeros((1, 2), np.float32),
                        np.zeros((1, 3), np.float32), np.zeros(1, np.float32))
    names = {n: s for n, _, s in patch_agent.param_layout()}
    assert names["enc_c1w"] == 0 and names["in_w"] == 0 and names["dc1w"] == 0 and names["pt_w"] > 0


@pytest.mark.parametrize("bad", [dict(patch=7), dict(patch=8, inv_hidden=4), dict(patch=8, dyn_hidden=4),
                                 dict(patch=16, pred_img=12)])
def test_patch_invalid_configs(bad):
    with pytest.raises(Exception):
        agent_lib.Agent(dict(PATCH, **bad), seed=1)

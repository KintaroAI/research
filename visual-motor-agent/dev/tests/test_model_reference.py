"""C++ agent vs PyTorch reference: forward outputs, losses and all gradients; acting == training.

Requires libvmagent.so (make libvmagent.so) and torch (CPU) in the venv.
"""
import os
import sys

import numpy as np
import pytest

torch = pytest.importorskip("torch")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import agent_lib  # noqa: E402
from reference import torch_reference as ref  # noqa: E402

# Small but structurally complete model (2 layers, 4 heads, full decoder) so the test is fast.
SMALL = dict(img=32, c1=4, c2=8, c3=8, feat=16, d_model=16, n_layer=2, n_head=4, d_ff=32, ctx=8,
             pred_img=16, pc1=4, pc2=8, pred_hidden=16, use_prev_reward=1, logstd_init=-0.3)


def make_rollout(agent, T=16, N=3, seed=0, p_done=0.15):
    """Fill a C++ rollout by acting on random images; returns the Python-side copies of everything."""
    rng = np.random.default_rng(seed)
    cfg = agent.config
    S = cfg["img"]
    ctx = agent.new_context(N, seed=seed + 1)
    ro = agent.new_rollout(T, N)
    data = {k: [] for k in ("images", "drives", "prev_action", "prev_reward", "ep_start", "action", "logp", "value",
                            "reward", "done", "next_images", "next_drives", "mu")}
    ep_start = np.ones(N, np.uint8)
    prev_a = np.zeros((N, 3), np.float32)
    prev_r = np.zeros(N, np.float32)
    img = rng.integers(0, 256, (N, S, S, 3), dtype=np.uint8)
    drv = rng.random((N, 2), dtype=np.float32)
    for t in range(T):
        out = agent.act(ctx, img, drv, prev_a, prev_r, ep_start=ep_start, rollout=ro, t=t)
        for k, v in (("images", img), ("drives", drv), ("prev_action", prev_a), ("prev_reward", prev_r),
                     ("ep_start", ep_start), ("action", out["action"]), ("logp", out["logp"]),
                     ("value", out["value"]), ("mu", out["mu"])):
            data[k].append(np.array(v))
        reward = (rng.random(N) < 0.2).astype(np.float32) * rng.normal(size=N).astype(np.float32)
        done = (rng.random(N) < p_done).astype(np.uint8)
        nimg = rng.integers(0, 256, (N, S, S, 3), dtype=np.uint8)
        ndrv = rng.random((N, 2), dtype=np.float32)
        ro.outcome(t, reward, done, nimg, ndrv)
        data["reward"].append(reward); data["done"].append(done)
        data["next_images"].append(nimg); data["next_drives"].append(ndrv)
        a = out["action"]
        prev_a = np.stack([np.clip(a[:, 0], -1, 1), np.clip(a[:, 1], -1, 1), a[:, 2]], 1).astype(np.float32)
        prev_a[done.astype(bool)] = 0
        prev_r = np.where(done.astype(bool), 0, reward).astype(np.float32)
        ep_start = done.copy()
        # next obs: a fresh frame after reset, otherwise the next frame
        img = np.where(done[:, None, None, None].astype(bool), rng.integers(0, 256, nimg.shape, dtype=np.uint8), nimg)
        drv = ndrv
    last_value = agent.act(ctx, img, drv, prev_a, prev_r, ep_start=ep_start, peek=True)["value"]
    data = {k: np.stack(v) for k, v in data.items()}  # [T, N, ...]
    return ro, data, last_value


def seq_batch(data, ids, K, N, adv, ret, f):
    """Gather sequences (C++ numbering: s -> env s % N, block s // N) into [S, K, ...] arrays."""
    keys = ("images", "drives", "prev_action", "prev_reward", "ep_start", "action", "logp", "next_drives")
    out = {k: [] for k in keys + ("adv", "ret", "next_small")}
    for s in ids:
        n, b = s % N, s // N
        sl = slice(b * K, (b + 1) * K)
        for k in keys:
            out[k].append(data[k][sl, n])
        out["adv"].append(adv[sl, n]); out["ret"].append(ret[sl, n])
        out["next_small"].append(ref.downsample_u8(data["next_images"][sl, n], f))
    out = {k: np.stack(v) for k, v in out.items()}
    out["old_logp"] = out.pop("logp")
    return out


def compare_grads(agent, P, tol=3e-3):
    g_cpp = agent.grads()
    worst = []
    for name, off, size in agent.param_layout():
        g_ref = P[name].grad
        g_ref = np.zeros(size) if g_ref is None else g_ref.detach().numpy().reshape(-1)
        gc = g_cpp[off:off + size].astype(np.float64)
        denom = np.linalg.norm(g_ref) + 1e-9
        rel = np.linalg.norm(gc - g_ref) / denom
        if np.linalg.norm(g_ref) < 1e-10 and np.linalg.norm(gc) < 1e-10:
            rel = 0.0
        worst.append((rel, name))
    worst.sort(reverse=True)
    assert worst[0][0] < tol, "gradient mismatch: %s" % worst[:5]
    return worst


@pytest.fixture(scope="module", params=[(0, 0, 0, 0, 0), (1, 0, 0, 0, 0), (1, 1, 0, 0, 0), (1, 1, 1, 0, 0),
                                        (0, 0, 1, 0, 0), (1, 1, 1, 6, 0), (1, 1, 1, 6, 8)],
                ids=["plain", "coord", "coord_residual", "coord_residual_skip", "skip", "skip_inverse",
                     "skip_inverse_dyn"])
def agent(request):
    agent_lib.set_threads(4)
    coord, residual, skip, inv, dyn = request.param
    return agent_lib.Agent(dict(SMALL, coord=coord, pb=4, pred_residual=residual, pred_skip=skip, inv_hidden=inv,
                                dyn_hidden=dyn), seed=3)


@pytest.mark.parametrize("pc_over", [
    dict(pred_coef=1.0, drive_coef=0.5),                                   # PPO + prediction
    dict(pred_coef=0.0, drive_coef=0.0, clip_eps=0.2),                     # PPO only
    dict(use_ppo=0, pred_coef=2.0, drive_coef=1.0),                        # prediction only
    dict(pred_coef=1.0, drive_coef=0.5, pred_backbone_scale=0.3),          # scaled prediction gradient
    dict(pred_coef=1.0, drive_coef=0.5, pred_changed_weight=9.0),          # changed-pixel weighted loss
    dict(use_ppo=0, pred_coef=1.0, drive_coef=0.5, inv_coef=2.0),          # prediction + inverse dynamics
    dict(use_ppo=0, pred_coef=1.0, drive_coef=0.5, inv_coef=1.0, dyn_coef=1.0, dyn_h_coef=2.0, dyn_x_coef=1.5,
         dyn_horizon=3),                                                   # + latent dynamics (short chains)
    dict(pred_coef=1.0, drive_coef=0.5, dyn_coef=0.7, dyn_h_coef=1.0, dyn_x_coef=0.5, dyn_horizon=8),  # PPO + dynamics
    dict(use_ppo=0, pred_coef=1.0, drive_coef=0.5, inv_coef=1.0, dyn_coef=1.0, dyn_h_coef=2.0, dyn_x_coef=1.0,
         dyn_horizon=4, dyn_h_horizon=1),                                   # hunger head on real latents only
])
def test_loss_and_grads_match_reference(agent, pc_over):
    cfg = agent.config
    T, N = 16, 3
    ro, data, last_value = make_rollout(agent, T=T, N=N, seed=7)
    pc = dict(agent_lib.DEFAULT_PPO, **pc_over)
    ids = [0, 2, 4, 5]
    loss_cpp, st = agent.debug_grads(ro, last_value, pc, ids)
    adv, ret = ref.gae(data["reward"].astype(np.float64), data["value"].astype(np.float64),
                       data["done"].astype(np.float64), last_value.astype(np.float64),
                       pc["gamma"], pc["lam"], pc["reward_scale"])
    batch = seq_batch(data, ids, cfg["ctx"], N, adv, ret, cfg["img"] // cfg["pred_img"])
    P = ref.params_from_flat(agent.params(), agent.param_layout(), cfg)
    loss_ref, terms = ref.ppo_loss(P, cfg, pc, batch)
    loss_ref.backward()
    assert abs(loss_cpp - loss_ref.item()) < 1e-4 * max(1.0, abs(loss_ref.item())), (loss_cpp, loss_ref.item())
    if pc["use_ppo"]:
        # fresh rollout: ratio == 1 exactly, so the KL estimate and clip fraction are zero
        assert abs(st["approx_kl"]) < 1e-6 and st["clipfrac"] == 0.0
    compare_grads(agent, P)


def test_perturbed_policy_ratio_clipping_matches(agent):
    """After perturbing weights, ratios != 1 and clipping is active; gradients must still match."""
    cfg = agent.config
    T, N = 16, 3
    ro, data, last_value = make_rollout(agent, T=T, N=N, seed=11)
    p = agent.params()
    saved = p.copy()
    rng = np.random.default_rng(0)
    p += rng.normal(scale=0.05, size=p.shape).astype(np.float32)
    agent.params_changed()
    try:
        pc = dict(agent_lib.DEFAULT_PPO, pred_coef=1.0, drive_coef=0.5, clip_eps=0.1)
        ids = list(range(N * (T // cfg["ctx"])))
        loss_cpp, st = agent.debug_grads(ro, last_value, pc, ids)
        assert st["clipfrac"] > 0.0
        adv, ret = ref.gae(data["reward"].astype(np.float64), data["value"].astype(np.float64),
                           data["done"].astype(np.float64), last_value.astype(np.float64),
                           pc["gamma"], pc["lam"], pc["reward_scale"])
        batch = seq_batch(data, ids, cfg["ctx"], N, adv, ret, cfg["img"] // cfg["pred_img"])
        P = ref.params_from_flat(agent.params(), agent.param_layout(), cfg)
        loss_ref, _ = ref.ppo_loss(P, cfg, pc, batch)
        loss_ref.backward()
        assert abs(loss_cpp - loss_ref.item()) < 1e-4 * max(1.0, abs(loss_ref.item()))
        compare_grads(agent, P)
    finally:
        p[:] = saved
        agent.params_changed()


def test_acting_equals_training_forward(agent):
    """KV-cached acting and the batched sequence forward give bitwise-identical values and means."""
    cfg = agent.config
    T, N, K = 16, 3, agent.config["ctx"]
    ro, data, _ = make_rollout(agent, T=T, N=N, seed=5, p_done=0.2)
    S = N * (T // K)
    seqs = [(s % N, s // N) for s in range(S)]
    get = lambda k: np.stack([data[k][b * K:(b + 1) * K, n] for n, b in seqs])
    out = agent.forward(get("images"), get("drives"), get("prev_action"), get("prev_reward"), get("ep_start"),
                        pred_actions=get("action"))
    np.testing.assert_array_equal(out["value"], get("value"))
    np.testing.assert_array_equal(out["mu"], get("mu"))
    # and the forward pass matches the reference
    P = ref.params_from_flat(agent.params(), agent.param_layout(), cfg)
    r = ref.forward(P, cfg, get("images"), get("drives"), get("prev_action"), get("prev_reward"), get("ep_start"),
                    pred_actions=get("action"))
    np.testing.assert_allclose(out["value"], r["value"].detach().numpy(), rtol=1e-4, atol=1e-5)
    pf = r["pred_frames"].detach().numpy().transpose(0, 1, 3, 4, 2)
    np.testing.assert_allclose(out["pred_frames"], pf, rtol=1e-4, atol=1e-5)


def test_causality(agent):
    """Outputs at position t never depend on inputs after t, nor on anything before a context start."""
    cfg = agent.config
    K, S = cfg["ctx"], 2
    rng = np.random.default_rng(1)
    shp = (S, K)
    images = rng.integers(0, 256, shp + (cfg["img"], cfg["img"], 3), dtype=np.uint8)
    drives = rng.random(shp + (2,), dtype=np.float32)
    pa = rng.uniform(-1, 1, shp + (3,)).astype(np.float32)
    pr = rng.normal(size=shp).astype(np.float32)
    es = np.zeros(shp, np.uint8)
    es[:, 5] = 1
    acts = rng.uniform(-1, 1, shp + (3,)).astype(np.float32)
    base = agent.forward(images, drives, pa, pr, es, pred_actions=acts)
    # perturb the future (t >= 6): outputs at t <= 5 unchanged
    im2, pa2, acts2 = images.copy(), pa.copy(), acts.copy()
    im2[:, 6:] = 255 - im2[:, 6:]
    pa2[:, 6:] *= -1
    acts2[:, 6:] *= -1
    o2 = agent.forward(im2, drives, pa2, pr, es, pred_actions=acts2)
    np.testing.assert_array_equal(o2["z"][:, :6], base["z"][:, :6])
    np.testing.assert_array_equal(o2["pred_frames"][:, :6], base["pred_frames"][:, :6])
    # perturb the past before the context start at t=5: outputs at t >= 5 unchanged
    im3 = images.copy()
    im3[:, :5] = 255 - im3[:, :5]
    o3 = agent.forward(im3, drives, pa, pr, es, pred_actions=acts)
    np.testing.assert_array_equal(o3["z"][:, 5:], base["z"][:, 5:])
    # the policy at t does not see a_t: changing the prediction-conditioning action leaves z/value unchanged
    acts4 = acts.copy()
    acts4[:, 3] = -acts4[:, 3]
    o4 = agent.forward(images, drives, pa, pr, es, pred_actions=acts4)
    np.testing.assert_array_equal(o4["z"], base["z"])
    assert not np.array_equal(o4["pred_frames"][:, 3], base["pred_frames"][:, 3])


def test_checkpoint_roundtrip(agent, tmp_path):
    path = str(tmp_path / "ckpt.bin")
    agent.save(path, with_optimizer=True)
    b = agent_lib.Agent.load(path)
    assert b.config == agent.config
    np.testing.assert_array_equal(b.params(), agent.params())


def test_imagine_and_decode_match_reference():
    """Free-running imagination (predicted hunger fed back) and the coarse decoder vs the PyTorch reference."""
    agent_lib.set_threads(4)
    agent = agent_lib.Agent(dict(SMALL, coord=1, pb=4, pred_residual=1, pred_skip=1, inv_hidden=6, dyn_hidden=8),
                            seed=11)
    cfg = agent.config
    rng = np.random.default_rng(0)
    n, H = 70, 5                                   # n > 64 exercises two chunks
    s0 = rng.random((n, cfg["feat"])).astype(np.float32)
    h0 = rng.random(n).astype(np.float32)
    act = rng.normal(0, 1, (n, H, 3)).astype(np.float32)
    act[..., 2] = rng.random((n, H))
    out = agent.imagine(s0, h0, act, latents=True, frames=True)
    P = ref.params_from_flat(agent.params(), agent.param_layout(), cfg)
    s = torch.tensor(s0, dtype=torch.float64)
    h = torch.tensor(h0, dtype=torch.float64)
    ex = torch.tensor(ref.executed(act))
    r = cfg["pred_img"] // 2
    with torch.no_grad():
        for k in range(H):
            s, h = ref.dyn_step(P, s, h, ex[:, k])
            np.testing.assert_allclose(out["hunger"][:, k], h.numpy(), rtol=1e-4, atol=1e-5)
            np.testing.assert_allclose(out["latents"][:, k], s.numpy(), rtol=1e-4, atol=1e-4)
            np.testing.assert_allclose(out["frames"][:, k].reshape(n, -1), ref.dyn_decode(P, s).numpy(),
                                       rtol=1e-4, atol=1e-4)
        np.testing.assert_allclose(agent.decode(s0).reshape(n, -1),
                                   ref.dyn_decode(P, torch.tensor(s0, dtype=torch.float64)).numpy(),
                                   rtol=1e-4, atol=1e-4)
    assert out["frames"].shape == (n, H, r, r, 3)


def test_rollout_set_action():
    """Replacing the stored actions (planner as actor) by the same actions changes nothing; other actions change
    the action-conditioned losses."""
    agent_lib.set_threads(4)
    agent = agent_lib.Agent(dict(SMALL, coord=1, pb=4, pred_residual=1, pred_skip=1, inv_hidden=6, dyn_hidden=8),
                            seed=12)
    pc = dict(use_ppo=0, pred_coef=1.0, drive_coef=1.0, inv_coef=1.0, dyn_coef=1.0, dyn_h_coef=1.0,
              dyn_x_coef=1.0, dyn_horizon=3)
    seqs = np.arange(6)
    ro, data, lv = make_rollout(agent)
    base, st0 = agent.debug_grads(ro, lv, pc, seqs)
    ro_same, data2, _ = make_rollout(agent)
    for t in range(data2["action"].shape[0]):
        ro_same.set_action(t, data2["action"][t])
    same, st1 = agent.debug_grads(ro_same, lv, pc, seqs)
    assert same == base and st1 == st0
    ro_other, data3, _ = make_rollout(agent)
    for t in range(data3["action"].shape[0]):
        a = data3["action"][t].copy()
        a[:, :2] *= -1
        ro_other.set_action(t, a)
    other, st2 = agent.debug_grads(ro_other, lv, pc, seqs)
    assert other != base and st2["inv_mse"] != st0["inv_mse"]

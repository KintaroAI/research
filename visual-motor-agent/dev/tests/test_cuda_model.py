"""The GPU update (libvmcuda.so, task 00011) against the C++ model: loss, every gradient tensor and one AdamW step.

Skipped without libvmcuda.so (make libvmcuda.so) or a GPU.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import agent_lib  # noqa: E402
import cuda_lib  # noqa: E402

pytestmark = pytest.mark.skipif(not cuda_lib.available(), reason="libvmcuda.so not built")

from test_model_reference import make_rollout, seq_batch, SMALL  # noqa: E402

BASIC = dict(coord=1, pb=32, pred_residual=1, pred_skip=1, ctx=8, use_prev_reward=1)   # configs/AHR_gated.json
PC = dict(use_ppo=0, pred_coef=1.0, drive_coef=1.0, pred_changed_weight=10.0, move_coef=1.0, epochs=1,
          minibatches=1, lr=3e-4, pi_detach=1)


def cuda_batch(agent, data, ids, N):
    K = agent.config["ctx"]
    T = data["images"].shape[0]
    z = np.zeros((T, N))
    b = seq_batch(data, ids, K, N, z, z, agent.config["img"] // agent.config["pred_img"])
    return [b[k] for k in ("images", "drives", "prev_action", "prev_reward", "ep_start", "action", "next_small",
                           "next_drives")]


def compare(agent, g_cpu, g_gpu, tol):
    worst = []
    for name, off, size in agent.param_layout():
        a, b = g_cpu[off:off + size].astype(np.float64), g_gpu[off:off + size].astype(np.float64)
        na = np.linalg.norm(a)
        rel = 0.0 if na < 1e-12 and np.linalg.norm(b) < 1e-12 else np.linalg.norm(a - b) / (na + 1e-12)
        worst.append((rel, name))
    worst.sort(reverse=True)
    assert worst[0][0] < tol, "gradient mismatch: %s" % worst[:5]
    return worst


MODELS = {"basic": dict(BASIC), "basic_hidden_head": dict(BASIC, pi_hidden=128),
          "small": dict(SMALL, coord=1, pb=4, pred_residual=1, pred_skip=1),
          "small_one_frame": dict(SMALL, coord=1, pb=4, pred_residual=1, pred_skip=1, one_frame=1, pi_hidden=5),
          "dmi": dict(BASIC, pi_hidden=128, inv_hidden=128),                                  # vm-00008 DMI
          "small_inverse_one_frame": dict(SMALL, coord=1, pb=4, pred_residual=1, pred_skip=1, one_frame=1,
                                          inv_hidden=6),
          "planner": dict(BASIC, inv_hidden=128, dyn_hidden=256, one_frame=1),               # configs/L_gated.json
          "small_dynamics": dict(SMALL, coord=1, pb=4, pred_residual=1, pred_skip=1, inv_hidden=6, dyn_hidden=7),
          "patch": dict(BASIC, patch=16),                                                       # vm-00010 P0
          "small_patch_hidden": dict(SMALL, patch=8, pred_residual=1, pi_hidden=5),
          "patch8": dict(BASIC, patch=8, pi_hidden=128),             # vm-00014 P8: 536 tokens per context (> 256)
          "patch_whole_frame": dict(BASIC, patch=64, pi_hidden=128),  # vm-00014 P64: the frame as one token
          "patch8_act": dict(BASIC, patch=8, pi_hidden=128, act_token=1),          # task 00020: the act token
          "small_patch_act": dict(SMALL, patch=8, pred_residual=1, pi_hidden=0, act_token=1),
          "universal": dict(BASIC, patch=8, universal=1, bins=32),                 # task 00023: the universal model
          "small_universal": dict(SMALL, patch=8, universal=1, bins=8),
          "small_universal_p16": dict(SMALL, patch=16, universal=1, bins=5),
          "universal_gray": dict(BASIC, patch=8, universal=1, bins=32, gray=1),           # task 00024
          "small_universal_gray": dict(SMALL, patch=4, universal=1, bins=6, gray=1),
          "universal_gray_inbins": dict(BASIC, patch=8, universal=1, bins=32, gray=1, in_bins=1),   # task 00036
          "small_universal_inbins": dict(SMALL, patch=8, universal=1, bins=8, in_bins=1),
          # task 00044: attention in regions (halves of the patch rows, one row of overlap; the tiny one: short windows)
          "small_universal_split": dict(SMALL, patch=8, universal=1, bins=8, in_bins=1, attn_split=2, attn_overlap=1),
          "tiny_universal_split": dict(SMALL, img=16, pred_img=16, ctx=4, patch=8, universal=1, bins=8, attn_split=2),
          "universal_gray_inbins_split": dict(BASIC, patch=8, universal=1, bins=32, gray=1, in_bins=1, attn_split=2,
                                              attn_overlap=1),
          "universal_gray_inbins_split_pg": dict(BASIC, patch=8, universal=1, bins=32, gray=1, in_bins=1, attn_split=2,
                                                 attn_overlap=1, attn_pred_global=1),
          # task 00046: checkerboard regions
          "universal_gray_inbins_checker": dict(BASIC, patch=8, universal=1, bins=32, gray=1, in_bins=1, attn_split=2,
                                                attn_checker=1)}
LOSSES = {"detached": dict(PC), "move_into_features": dict(PC, pi_detach=0),
          "scaled_prediction": dict(PC, pred_backbone_scale=0.3, drive_coef=0.5, move_coef=1.5),
          "no_move": dict(PC, move_coef=0.0),
          "inverse": dict(PC, inv_coef=0.1), "inverse_strong_scaled": dict(PC, inv_coef=2.0, pred_backbone_scale=0.3),
          "planner": dict(PC, move_coef=0.0, inv_coef=0.1, dyn_coef=1.0, dyn_h_coef=10.0, dyn_x_coef=1.0, dyn_horizon=7),
          "dynamics_short_hunger": dict(PC, inv_coef=0.1, dyn_coef=0.7, dyn_h_coef=2.0, dyn_x_coef=0.5, dyn_horizon=4,
                                        dyn_h_horizon=1),
          "drive_weighted": dict(PC, inv_coef=0.1, dyn_coef=1.0, dyn_h_coef=10.0, dyn_x_coef=1.0, dyn_horizon=5,
                                 drive_changed_weight=9.0, drive_change_eps=0.3),
          "act": dict(PC, act_coef=0.7, drive_changed_weight=3.0, drive_change_eps=0.3),
          "act_move_through_scaled": dict(PC, pi_detach=0, act_coef=1.5, pred_backbone_scale=0.4)}


@pytest.fixture(scope="module", params=list(MODELS), ids=list(MODELS))
def setup(request):
    agent_lib.set_threads(8)
    agent = agent_lib.Agent(MODELS[request.param], seed=4)
    N, T = 4, 2 * agent.config["ctx"]
    if agent.config["patch"]:   # patch tokens do not act with the KV cache: fill the rollout through store()
        from test_patch_tokens import store_rollout
        ro, data = store_rollout(agent, T=T, N=N, seed=2, p_done=0.12)
        lv = np.zeros(N, np.float32)
    else:
        ro, data, lv = make_rollout(agent, T=T, N=N, seed=2, p_done=0.12)   # episodes also start inside blocks
    return agent, ro, data, lv, N


@pytest.mark.parametrize("loss", list(LOSSES), ids=list(LOSSES))
def test_loss_and_grads_match_cpp(setup, loss):
    agent, ro, data, lv, N = setup
    pc = LOSSES[loss]
    ids = [0, 2, 3, 5, 6]
    loss_cpu, st_cpu = agent.debug_grads(ro, lv, pc, ids)
    g_cpu = agent.grads().copy()
    m = cuda_lib.CudaModel(agent)
    m.set_batch(*cuda_batch(agent, data, ids, N))
    st = m.loss_and_grads(pc)
    g_gpu = m.grads()
    m.close()
    assert abs(st["loss"] - loss_cpu) < 1e-4 * max(1.0, abs(loss_cpu)), (st["loss"], loss_cpu)
    for k in ("pred_mse", "pred_changed_mse", "drive_mse", "move_nll", "inv_mse", "inv_r2", "dyn_mse",
              "dyn_hunger_mse", "dyn_recon_mse", "act_drive_mse", "act_frame_mse", "uni_ce_body", "uni_ce_patch",
              "uni_ce_move", "uni_mse_body", "uni_mse_patch", "uni_mse_move"):
        assert abs(st[k] - st_cpu[k]) < 1e-5 * max(1.0, abs(st_cpu[k])), (k, st[k], st_cpu[k])
    compare(agent, g_cpu, g_gpu, 1e-3)


def test_update_matches_cpp(setup):
    agent, ro, data, lv, N = setup
    pc = dict(PC, max_grad_norm=0.5, pi_own_clip=1, inv_coef=0.1, dyn_coef=1.0, dyn_h_coef=10.0, dyn_x_coef=1.0,
              dyn_horizon=5)
    p0 = agent.params().copy()
    a2 = agent_lib.Agent(agent.config, seed=4)          # a copy with fresh optimizer state
    a2.params()[:] = p0
    a2.params_changed()
    m = cuda_lib.CudaModel(a2)
    nseq = N * (data["images"].shape[0] // agent.config["ctx"])
    m.set_batch(*cuda_batch(a2, data, list(range(nseq)), N))
    st_gpu = m.update(pc)
    p_gpu = m.params()
    m.close()
    st_cpu = a2.update(ro, lv, pc)
    dp_cpu, dp_gpu = a2.params() - p0, p_gpu - p0
    assert abs(st_gpu["grad_norm"] - st_cpu["grad_norm"]) < 1e-3 * st_cpu["grad_norm"]
    # Adam's first step is about lr * sign(g): parameters whose gradient is at float noise level may differ by
    # up to 2 lr, but the update as a whole must agree
    assert np.abs(dp_cpu - dp_gpu).max() <= 2.0 * pc["lr"] + 1e-7
    assert np.linalg.norm(dp_cpu - dp_gpu) < 1e-2 * np.linalg.norm(dp_cpu)
    a2.close()


def test_refuses_unsupported():
    agent = agent_lib.Agent(dict(BASIC, patch=16, pred_residual=0), seed=1)
    with pytest.raises(ValueError):
        cuda_lib.CudaModel(agent)
    agent.close()


def test_rollout_matches_cpp_rollout(setup):
    """cuda_lib.Rollout, filled through the same store / outcome calls as online.py makes, gives the same batch as
    the C++ rollout: the same loss and gradients."""
    agent, ro, data, lv, N = setup
    K = agent.config["ctx"]
    pc = LOSSES["detached"]
    r = cuda_lib.Rollout(K, N, agent.config["img"])
    for t in range(K):
        r.store(t, data["images"][t], data["drives"][t], data["prev_action"][t], data["prev_reward"][t],
                data["ep_start"][t], data["action"][t])
        r.outcome(t, data["reward"][t], data["done"][t], data["next_images"][t], data["next_drives"][t])
    loss_cpu, _ = agent.debug_grads(ro, lv, pc, list(range(N)))     # sequences 0..N-1: steps 0..K-1 of every env
    g_cpu = agent.grads().copy()
    m = cuda_lib.CudaModel(agent)
    m.set_batch(*r.batch())
    st = m.loss_and_grads(pc)
    g_gpu = m.grads()
    m.close()
    assert abs(st["loss"] - loss_cpu) < 1e-4 * max(1.0, abs(loss_cpu))
    compare(agent, g_cpu, g_gpu, 1e-3)


@pytest.mark.parametrize("T", [1, 5, 8])
@pytest.mark.parametrize("pred", [False, True], ids=["act", "with_prediction"])
def test_forward_matches_cpp(setup, T, pred):
    """The GPU forward against vm_forward: windows of T <= ctx steps, episodes starting inside them."""
    agent, ro, data, lv, N = setup
    T = min(T, agent.config["ctx"])
    sl = slice(3, 3 + T)                       # windows that do not start at a block boundary
    x = [np.ascontiguousarray(data[k][sl].swapaxes(0, 1)) for k in ("images", "drives", "prev_action", "prev_reward",
                                                                    "ep_start")]
    x[4] = x[4].copy()
    x[4][:, 0] = 1
    pa = np.ascontiguousarray(data["action"][sl].swapaxes(0, 1)) if pred else None
    m = cuda_lib.CudaModel(agent)
    if agent.config["universal"]:   # its own forward: the moves made are inputs
        act = np.ascontiguousarray(data["action"][sl].swapaxes(0, 1))
        ref = agent.forward_uni(x[0], x[1], act, x[4], next_body=True)
        out = m.forward_uni(x[0], x[1], act, x[4], next_body=True)
        lean = m.forward_uni(x[0], x[1], act, x[4], values=False, next_body=True)   # the head on two rows per step
        for k in ("move_probs", "next_body"):
            np.testing.assert_allclose(lean[k], ref[k], rtol=1e-4, atol=2e-5, err_msg=k)
    else:
        ref, out = agent.forward(*x, pred_actions=pa), m.forward(*x, pred_actions=pa)
    m.close()
    assert set(out) == set(ref)
    for k in ref:
        np.testing.assert_allclose(out[k], ref[k], rtol=1e-4, atol=2e-5, err_msg=k)


def test_forward_follows_updates(setup):
    """After a GPU update, the GPU forward equals the C++ forward of the synced weights."""
    agent, ro, data, lv, N = setup
    a2 = agent_lib.Agent(agent.config, seed=4)
    a2.params()[:] = agent.params()
    a2.params_changed()
    m = cuda_lib.CudaModel(a2)
    nseq = N * (data["images"].shape[0] // agent.config["ctx"])
    for _ in range(3):
        m.set_batch(*cuda_batch(a2, data, list(range(nseq)), N))
        m.update(dict(PC, max_grad_norm=0.5), stats=False)
    m.sync_to(a2)
    x = [np.ascontiguousarray(data[k][:4].swapaxes(0, 1)) for k in ("images", "drives", "prev_action", "prev_reward",
                                                                   "ep_start")]
    if agent.config["universal"]:
        act = np.ascontiguousarray(data["action"][:4].swapaxes(0, 1))
        ref, out = a2.forward_uni(x[0], x[1], act, x[4]), m.forward_uni(x[0], x[1], act, x[4])
    else:
        ref, out = a2.forward(*x), m.forward(*x)
    m.close()
    a2.close()
    for k in ref:
        np.testing.assert_allclose(out[k], ref[k], rtol=1e-4, atol=2e-5, err_msg=k)


def test_planner_calls_match_cpp(setup):
    """encode, imagine (hunger, latents, frames) and decode against the C++ model (latent dynamics only)."""
    agent, ro, data, lv, N = setup
    if not agent.config["dyn_hidden"]:
        pytest.skip("no latent dynamics")
    rng = np.random.default_rng(3)
    img = data["images"][:3].reshape(-1, *data["images"].shape[2:])
    m = cuda_lib.CudaModel(agent)
    vf_c, vf_g = agent.encode(img)["vf"], m.encode(img)["vf"]
    np.testing.assert_allclose(vf_g, vf_c, rtol=1e-4, atol=1e-5)
    n, H = 70, 6                                                    # more rows than the C++ chunk of 64
    s0 = np.repeat(vf_c, (n + len(vf_c) - 1) // len(vf_c), 0)[:n]
    h0 = rng.random(n).astype(np.float32)
    act = np.concatenate([rng.uniform(-1.3, 1.3, (n, H, 2)), rng.random((n, H, 1))], -1).astype(np.float32)
    rc = agent.imagine(s0, h0, act, latents=True, frames=True)
    rg = m.imagine(s0, h0, act, latents=True, frames=True)
    for k in rc:
        np.testing.assert_allclose(rg[k], rc[k], rtol=1e-4, atol=2e-5, err_msg=k)
    np.testing.assert_allclose(m.decode(rc["latents"][:, -1]), agent.decode(rc["latents"][:, -1]), rtol=1e-4, atol=2e-5)
    m.close()


@pytest.mark.parametrize("model", ["universal_gray", "small_universal", "small_universal_p16", "universal_gray_inbins",
                                   "small_universal_inbins", "small_universal_split", "universal_gray_inbins_split",
                                   "universal_gray_inbins_split_pg", "universal_gray_inbins_checker"])
def test_uni_candidates_match_full_pass(model):
    """The instinct's fast path (task 00026): each candidate's predicted next hunger from the move token alone, against
    the full forward pass with that candidate as the window's last move."""
    agent = agent_lib.Agent(MODELS[model], seed=5)
    c = agent.config
    rng = np.random.default_rng(1)
    S, T, Mc = 3, c["ctx"], 5
    img = rng.integers(0, 256, (S, T, c["img"], c["img"], 3), dtype=np.uint8)
    drv = rng.random((S, T, 2)).astype(np.float32)
    act = np.concatenate([rng.uniform(-1, 1, (S, T, 2)), rng.random((S, T, 1)) < 0.5], -1).astype(np.float32)
    es = np.zeros((S, T), np.uint8)
    es[:, 0] = 1
    es[1, 4] = 1                                                      # an episode start inside one window
    cand = np.concatenate([rng.uniform(-1.2, 1.2, (S, Mc, 2)), rng.random((S, Mc, 1)) < 0.5], -1).astype(np.float32)
    m = cuda_lib.CudaModel(agent)
    m.forward_uni(img, drv, act, es, values=False, move_probs=True)
    fast = m.uni_candidates(cand)
    full = np.empty((S, Mc), np.float32)
    for k in range(Mc):
        a2 = act.copy()
        a2[:, -1] = cand[:, k]
        full[:, k] = m.forward_uni(img, drv, a2, es, values=False, move_probs=False, next_body=True)["next_body"][:, -1, 0]
    ref = np.empty((S, Mc), np.float32)
    for k in range(Mc):
        a2 = act.copy()
        a2[:, -1] = cand[:, k]
        ref[:, k] = agent.forward_uni(img, drv, a2, es, values=False, move_probs=False, next_body=True)["next_body"][:, -1, 0]
    m.close()
    agent.close()
    np.testing.assert_allclose(fast, full, atol=1e-5)
    np.testing.assert_allclose(fast, ref, atol=2e-5)


def test_universal_batch_without_next_frames():
    """The universal model needs no next frames (task 00026): a batch without them gives the same loss and gradients."""
    agent = agent_lib.Agent(MODELS["small_universal"], seed=4)
    N, T = 4, 2 * agent.config["ctx"]
    from test_patch_tokens import store_rollout
    ro, data = store_rollout(agent, T=T, N=N, seed=2, p_done=0.12)
    b = cuda_batch(agent, data, [0, 2, 3, 5, 6], N)
    m = cuda_lib.CudaModel(agent)
    m.set_batch(*b)
    st, g = m.loss_and_grads(PC), m.grads().copy()
    b[6] = None
    m.set_batch(*b)
    st2, g2 = m.loss_and_grads(PC), m.grads()
    m.close()
    assert abs(st["loss"] - st2["loss"]) < 1e-6
    np.testing.assert_allclose(g2, g, rtol=1e-5, atol=1e-8)



@pytest.mark.parametrize("model", ["small_universal", "small_universal_gray"])
def test_uni_token_ce_matches_reference(model):
    """The gate's surprise (task 00028): each token's cross-entropy at the last step of the update's first sequences,
    against the PyTorch reference's per-token loss."""
    pytest.importorskip("torch")
    from reference import torch_reference as ref
    from test_patch_tokens import store_rollout
    if not cuda_lib.has("vmc_uni_token_ce"):
        pytest.skip("libvmcuda.so without vmc_uni_token_ce")
    agent = agent_lib.Agent(MODELS[model], seed=4)
    c = agent.config
    N, T = 4, 2 * c["ctx"]
    ro, data = store_rollout(agent, T=T, N=N, seed=2, p_done=0.12)
    ids = [0, 2, 3, 5, 6]
    m = cuda_lib.CudaModel(agent)
    m.set_batch(*cuda_batch(agent, data, ids, N))
    m.loss_and_grads(PC)
    ce = m.uni_token_ce(3)
    z = np.zeros((T, N))
    P = ref.params_from_flat(agent.params(), agent.param_layout(), c)
    _, terms = ref.ppo_loss(P, c, dict(agent_lib.DEFAULT_PPO, use_ppo=0),
                            seq_batch(data, ids, c["ctx"], N, z, z, c["img"] // c["pred_img"]))
    pt = terms["per_token"].detach().numpy()                          # [S, T, ntok]
    np.testing.assert_allclose(ce, pt[:3, -1], rtol=1e-4, atol=1e-5)
    m.forward_uni(data["images"][:2, :2].transpose(1, 0, 2, 3, 4), np.zeros((2, 2, 2), np.float32),
                  np.zeros((2, 2, 3), np.float32))                    # acting in between leaves the last update's
    np.testing.assert_array_equal(m.uni_token_ce(3), ce)
    with pytest.raises(RuntimeError):
        m.uni_token_ce(len(ids) + 1)
    m.close()
    agent.close()


def _ref_flat_grads(agent, batch):
    """The PyTorch reference's universal loss and its gradient in the agent's flat layout."""
    from reference import torch_reference as ref
    c = agent.config
    P = ref.params_from_flat(agent.params(), agent.param_layout(), c)
    loss, _ = ref.uni_loss(P, c, batch)
    loss.backward()
    g = np.zeros(agent.num_params)
    for name, off, size in agent.param_layout():
        if P[name].grad is not None:
            g[off:off + size] = P[name].grad.numpy().ravel()
    return loss.item(), g


@pytest.mark.parametrize("model", ["small_universal", "small_universal_gray", "small_universal_inbins"])
def test_uni_inputs_match_reference(model):
    """Self-forcing (task 00039): with other token inputs (frames and bodies) the GPU's loss and gradients equal the
    reference's, whose targets stay the batch's; without them the batch's own inputs are used (and the next batch
    clears them)."""
    pytest.importorskip("torch")
    from test_patch_tokens import store_rollout
    if not cuda_lib.has("vmc_set_uni_inputs"):
        pytest.skip("libvmcuda.so without vmc_set_uni_inputs")
    agent = agent_lib.Agent(MODELS[model], seed=4)
    c = agent.config
    N, T = 4, 2 * c["ctx"]
    ro, data = store_rollout(agent, T=T, N=N, seed=2, p_done=0.0)
    ids = [0, 2, 3, 5, 6]
    b = cuda_batch(agent, data, ids, N)
    z = np.zeros((T, N))
    batch = seq_batch(data, ids, c["ctx"], N, z, z, c["img"] // c["pred_img"])
    rng = np.random.default_rng(7)
    iimg, idrv = b[0].copy(), b[1].copy()
    iimg[:, -3:] = rng.integers(0, 256, iimg[:, -3:].shape, dtype=np.uint8)   # the last 3 steps "dreamed"
    idrv[:, -3:] = rng.random(idrv[:, -3:].shape).astype(np.float32)
    m = cuda_lib.CudaModel(agent)
    m.set_batch(*b)
    st0, g0 = m.loss_and_grads(PC), m.grads().copy()
    m.set_uni_inputs(iimg, idrv)
    st1, g1 = m.loss_and_grads(PC), m.grads().copy()
    want_l, want_g = _ref_flat_grads(agent, dict(batch, input_images=iimg, input_drives=idrv))
    assert abs(st1["loss"] - want_l) < 1e-4 * max(1.0, abs(want_l))
    compare(agent, want_g, g1, 1e-4)
    l0, _ = _ref_flat_grads(agent, batch)
    assert abs(st0["loss"] - l0) < 1e-4 * max(1.0, abs(l0))
    assert np.linalg.norm(g1 - g0) > 0.01 * np.linalg.norm(g0)      # the inputs mattered
    m.set_batch(*b)                                                  # a new batch: its own inputs again
    st2 = m.loss_and_grads(PC)
    assert abs(st2["loss"] - st0["loss"]) < 1e-6
    with pytest.raises(RuntimeError):
        m.set_uni_inputs(iimg[:2], idrv[:2])
    m.close()
    agent.close()


def test_uni_inputs_targets_stay_real():
    """The reference: inputs equal to the batch change nothing; other inputs leave every target as the batch's."""
    pytest.importorskip("torch")
    from reference import torch_reference as ref
    from test_patch_tokens import store_rollout
    agent = agent_lib.Agent(MODELS["small_universal_gray"], seed=4)
    c = agent.config
    N, T = 2, c["ctx"]
    ro, data = store_rollout(agent, T=T, N=N, seed=3, p_done=0.0)
    z = np.zeros((T, N))
    batch = seq_batch(data, [0, 1], c["ctx"], N, z, z, c["img"] // c["pred_img"])
    P = ref.params_from_flat(agent.params(), agent.param_layout(), c)
    l0, t0 = ref.uni_loss(P, c, batch)
    l1, t1 = ref.uni_loss(P, c, dict(batch, input_images=batch["images"].copy(), input_drives=batch["drives"].copy()))
    assert float(l0) == float(l1)
    other = dict(batch, input_images=255 - batch["images"], input_drives=1 - batch["drives"])
    l2, t2 = ref.uni_loss(P, c, other)
    np.testing.assert_array_equal(t2["tgt"], t0["tgt"])
    assert float(l2) != float(l0)
    agent.close()

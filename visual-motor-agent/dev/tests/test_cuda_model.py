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
          "small_one_frame": dict(SMALL, coord=1, pb=4, pred_residual=1, pred_skip=1, one_frame=1, pi_hidden=5)}
LOSSES = {"detached": dict(PC), "move_into_features": dict(PC, pi_detach=0),
          "scaled_prediction": dict(PC, pred_backbone_scale=0.3, drive_coef=0.5, move_coef=1.5),
          "no_move": dict(PC, move_coef=0.0)}


@pytest.fixture(scope="module", params=list(MODELS), ids=list(MODELS))
def setup(request):
    agent_lib.set_threads(8)
    agent = agent_lib.Agent(MODELS[request.param], seed=4)
    N, T = 4, 2 * agent.config["ctx"]
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
    for k in ("pred_mse", "pred_changed_mse", "drive_mse", "move_nll"):
        assert abs(st[k] - st_cpu[k]) < 1e-5 * max(1.0, abs(st_cpu[k])), (k, st[k], st_cpu[k])
    compare(agent, g_cpu, g_gpu, 1e-3)


def test_update_matches_cpp(setup):
    agent, ro, data, lv, N = setup
    pc = dict(PC, max_grad_norm=0.5, pi_own_clip=1)
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
    agent = agent_lib.Agent(dict(BASIC, inv_hidden=16), seed=1)
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

"""The label as a body value (task 00109, model.body_label): the body carries a third value with its own channel (one more
value position in W_in, W_out and the bias, the move's after it). Its input is blank (0.5); the move token predicts it, its
target from the frame's prev_reward slot. Forward, loss and gradients against the PyTorch reference (C++ and GPU), the GPU
against the C++ model, the instinct's candidates reading the label against full passes, the ring, checkpoints, invalid
configs, and the label's target reaching the loss only through the label.

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
from test_model_reference import compare_grads, seq_batch  # noqa: E402
from test_patch_tokens import store_rollout  # noqa: E402
from test_value_channels import AGE, BAND, UNI, batch, table  # noqa: E402

HAVE_GPU = cuda_lib.available()
gpu = pytest.mark.skipif(not HAVE_GPU, reason="needs libvmcuda.so and a GPU")
L_UNI, L_AGE, L_BAND = (dict(c, body_label=1) for c in (UNI, AGE, BAND))
CFGS = [L_UNI, L_AGE, L_BAND, dict(L_UNI, gray=0)]
IDS = ["plain", "age", "piece", "colour"]


def test_table_sizes():
    """One more value position than the per-value channels alone, in both tables and the bias, and nothing else."""
    on, off = agent_lib.Agent(L_UNI, seed=1), agent_lib.Agent(UNI, seed=1)
    d, K = UNI["d_model"], UNI["bins"]
    assert len(table(on, "uni_in_w")) - len(table(off, "uni_in_w")) == K * d
    assert len(table(on, "uni_out_w")) - len(table(off, "uni_out_w")) == K * d
    assert len(table(on, "uni_out_b")) - len(table(off, "uni_out_b")) == K
    assert on.num_params - off.num_params == 2 * K * d + K
    on.close()
    off.close()


@pytest.mark.parametrize("cfg", CFGS, ids=IDS)
def test_cpu_forward_matches_reference(cfg):
    img, drv, act, es = batch()
    agent = agent_lib.Agent(cfg, seed=2)
    o = agent.forward_uni(img, drv, act, es, values=True, move_probs=True, next_body=True)
    c = agent.config
    P = ref.params_from_flat(agent.params(), agent.param_layout(), c, dtype=torch.float32)
    r = ref.forward_uni(P, c, img, drv, act, es)
    S, T = img.shape[:2]
    nt, V, K = ref.uni_dims(c)
    p = r["logits"].reshape(S, T, nt, V, K).softmax(-1).detach()
    want = (p * (torch.arange(K, dtype=p.dtype) / (K - 1))).sum(-1).numpy()
    np.testing.assert_allclose(o["values"], want, atol=2e-5)
    np.testing.assert_allclose(o["move_probs"], p[:, :, nt - 2, :3].numpy(), atol=2e-5)
    assert o["next_body"].shape == (S, T, 3)
    np.testing.assert_allclose(o["next_body"], want[:, :, nt - 1, :3], atol=2e-5)
    agent.close()


@pytest.mark.parametrize("cfg", CFGS, ids=IDS)
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


def slot_rollout(agent, flip, T=16, N=3, seed=3):
    """Random experience as store_rollout's, with the prev_reward slot (the label's target) as drawn or flipped (1 - x)."""
    rng = np.random.default_rng(seed)
    S = agent.config["img"]
    ro = agent.new_rollout(T, N)
    es, pa = np.ones(N, np.uint8), np.zeros((N, 3), np.float32)
    for t in range(T):
        img = rng.integers(0, 256, (N, S, S, 3), dtype=np.uint8)
        drv = rng.random((N, 2), dtype=np.float32)
        pr = rng.random(N).astype(np.float32)
        a = np.concatenate([rng.normal(0, 0.7, (N, 2)), rng.random((N, 1)) < 0.3], 1).astype(np.float32)
        ro.store(t, img, drv, pa, 1.0 - pr if flip else pr, es, a)
        ro.outcome(t, np.zeros(N, np.float32), np.zeros(N, np.uint8), img, rng.random((N, 2), dtype=np.float32))
        es = np.zeros(N, np.uint8)
    return ro


def test_the_label_target_is_the_prev_reward_slot():
    """With the body label, the frame's prev_reward slot is the label's target: changing it changes the loss, and only the
    label's output rows get a different gradient. Without it the universal model ignores the slot."""
    agent_lib.set_threads(4)
    pc = dict(agent_lib.DEFAULT_PPO, use_ppo=0)
    for cfg, matters in ((L_UNI, True), (UNI, False)):
        agent = agent_lib.Agent(cfg, seed=7)
        l1, _ = agent.debug_grads(slot_rollout(agent, False), np.zeros(3, np.float32), pc, [0, 1])
        g1 = agent.grads().copy()
        l2, _ = agent.debug_grads(slot_rollout(agent, True), np.zeros(3, np.float32), pc, [0, 1])
        g2 = agent.grads().copy()
        assert (abs(l1 - l2) > 1e-6) == matters, (cfg, l1, l2)
        if matters:   # the label's own output rows (position V + 2) differ; the hunger's (V, V + 1) do not
            lay = {n: (o, s) for n, o, s in agent.param_layout()}
            o, s = lay["uni_out_b"]
            K, V = cfg["bins"], cfg["patch"] ** 2
            db = np.abs(g1[o:o + s] - g2[o:o + s]).reshape(-1, K).max(1)
            assert db[V + 2] > 0 and db[V] == 0 and db[V + 1] == 0
        agent.close()


@gpu
@pytest.mark.parametrize("cfg", CFGS, ids=IDS)
def test_gpu_forward_matches_cpu(cfg):
    img, drv, act, es = batch(seed=5, T=cfg["ctx"])
    es[1, 3] = 1
    agent = agent_lib.Agent(cfg, seed=2)
    m = cuda_lib.CudaModel(agent)
    a = agent.forward_uni(img, drv, act, es, values=True, move_probs=True, next_body=True)
    b = m.forward_uni(img, drv, act, es, values=True, move_probs=True, next_body=True)
    c = m.forward_uni(img, drv, act, es, values=False, move_probs=True, next_body=True)   # the last two rows only
    for k in ("values", "move_probs", "next_body"):
        np.testing.assert_allclose(b[k], a[k], atol=2e-5)
    np.testing.assert_allclose(c["move_probs"], a["move_probs"], atol=2e-5)
    np.testing.assert_allclose(c["next_body"], a["next_body"], atol=2e-5)
    m.close()
    agent.close()


@gpu
@pytest.mark.parametrize("cfg", CFGS, ids=IDS)
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


@gpu
@pytest.mark.parametrize("cfg", CFGS, ids=IDS)
def test_candidates_read_the_label(cfg):
    """The instinct's one-token candidates return the predicted label (the move token's third value), as full passes do."""
    img, drv, act, es = batch(seed=6, S=3, T=4)
    act[:, -1] = 0.0
    agent = agent_lib.Agent(cfg, seed=3)
    m = cuda_lib.CudaModel(agent)
    cand = np.random.default_rng(1).uniform(-1, 1, (3, 5, 3)).astype(np.float32)
    cand[..., 2] = (cand[..., 2] > 0).astype(np.float32)
    m.forward_uni(img, drv, act, es, values=False, move_probs=True)
    fast = m.uni_candidates(cand)
    for i in range(5):
        a2 = act.copy()
        a2[:, -1] = cand[:, i]
        nb = m.forward_uni(img, drv, a2, es, values=False, move_probs=False, next_body=True)["next_body"]
        np.testing.assert_allclose(fast[:, i], nb[:, -1, 2], atol=3e-5)
    m.close()
    agent.close()


@gpu
@pytest.mark.parametrize("cfg", [L_AGE, L_BAND], ids=["age", "piece"])
def test_ring_matches_a_full_pass(cfg):
    N, T = 3, cfg["ctx"]
    x, d, a, es = batch(seed=8, S=N, T=T)
    agent = agent_lib.Agent(cfg, seed=2)
    m = cuda_lib.CudaModel(agent)
    m.uni_ring_reset(N)
    cand = np.random.default_rng(1).uniform(-1, 1, (N, 4, 3)).astype(np.float32)
    for t in range(T):
        full_a = a[:, :t + 1].copy()
        full_a[:, -1] = 0.0
        p = m.forward_uni(x[:, :t + 1], d[:, :t + 1], full_a, es[:, :t + 1], values=False, move_probs=True)["move_probs"]
        hf = m.uni_candidates(cand)
        pr = m.uni_ring_step(x[:, t], d[:, t], es[:, t])
        hr = m.uni_candidates(cand)
        np.testing.assert_allclose(pr, p[:, -1], atol=3e-5, err_msg="tick %d" % t)
        np.testing.assert_allclose(hr, hf, atol=3e-5, err_msg="tick %d" % t)
        m.uni_ring_move(a[:, t])
    m.close()
    agent.close()


def test_checkpoint_keeps_it(tmp_path):
    agent = agent_lib.Agent(L_BAND, seed=4)
    p = str(tmp_path / "c.bin")
    agent.save(p)
    b = agent_lib.Agent.load(p)
    assert b.config["body_label"] == 1 and b.num_params == agent.num_params
    np.testing.assert_array_equal(b.params(), agent.params())
    agent.close()
    b.close()


@pytest.mark.parametrize("bad", [dict(body_label=2), dict(body_label=-1), dict(body_label=1, uni_chan=0),
                                 dict(body_label=1, state=1)], ids=["two", "negative", "no_channels", "state"])
def test_invalid_configs(bad):
    with pytest.raises(ValueError):
        agent_lib.Agent(dict(UNI, **bad), seed=1)

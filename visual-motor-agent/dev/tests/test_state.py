"""A state carried across windows (task 00104, model.state): a write and a read token after the body of every step. The
write's expected values are the state; the read's values are the state written ctx steps earlier, given from outside.

The layout and the parameters it adds, checkpoints, invalid configs; the C++ model with a blank state (0.5) against the
PyTorch reference; on the GPU, forward passes with given states (values and writes) against the reference, the blank
state against the C++ model, an update on pairs of windows (the second reading the first's writes, with and without the
gradient through them) against the reference's pair loss, the ring with its queue of writes against full passes, and a
copy task the gradient through the carry learns.

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

HAVE_GPU = cuda_lib.available()
gpu = pytest.mark.skipif(not HAVE_GPU, reason="needs libvmcuda.so and a GPU")
UNI = dict(SMALL, patch=8, universal=1, bins=8, gray=1, in_bins=1, uni_chan=1, state=1)
AGE = dict(UNI, age_key=1)
PYR = dict(AGE, uni_pyr=2, sum_thumb=1, band_n=4, band_k=1, band_cols=2, band_win=1)   # as the runs: pieces, summaries
CFGS = [UNI, AGE, PYR, dict(UNI, uni_chan=0), dict(UNI, in_bins=0, uni_chan=0)]
IDS = ["plain", "age", "piece", "no channels", "no bins"]
PC = dict(use_ppo=0, epochs=1, minibatches=1, lr=3e-4)


def batch(seed=0, S=2, T=3, img=32):
    rng = np.random.default_rng(seed)
    a = rng.uniform(-1, 1, (S, T, 3)).astype(np.float32)
    a[..., 2] = (a[..., 2] > 0).astype(np.float32)
    return rng.integers(0, 256, (S, T, img, img, 3), dtype=np.uint8), rng.random((S, T, 2)).astype(np.float32), a, \
        np.zeros((S, T), np.uint8)


def states(seed, S, T, V):
    return np.random.default_rng(seed).random((S, T, V)).astype(np.float32)


def ref_values(c, logits, S, T):
    nt, V, K = ref.uni_dims(c)
    p = logits.reshape(S, T, nt, V, K).softmax(-1).detach()
    return (p * (torch.arange(K, dtype=p.dtype) / (K - 1))).sum(-1).numpy()


def test_layout_and_parameters():
    """The write and the read follow the body; two more slot embeddings and nothing else."""
    on, off = agent_lib.Agent(UNI, seed=1), agent_lib.Agent(dict(UNI, state=0), seed=1)
    lay, lay0 = agent_lib.uni_layout(on.config), agent_lib.uni_layout(off.config)
    assert lay["nt"] == lay0["nt"] + 2
    assert list(lay["kind"][:3]) == [0, 5, 4] and lay["kind"][-1] == 3
    assert ref.uni_slot_table(on.config)["kind"].tolist() == lay["kind"].tolist()
    assert on.num_params - off.num_params == 2 * UNI["d_model"]
    on.close()
    off.close()


def test_checkpoint_keeps_it(tmp_path):
    a = agent_lib.Agent(UNI, seed=1)
    a.save(str(tmp_path / "c.bin"))
    b = agent_lib.Agent.load(str(tmp_path / "c.bin"))
    assert b.config["state"] == 1
    np.testing.assert_array_equal(a.params(), b.params())
    a.close()
    b.close()


@pytest.mark.parametrize("bad", [dict(state=2), dict(state=-1), dict(attn_split=2), dict(attn_win=2),
                                 dict(universal=0, patch=0)], ids=["two", "negative", "split", "window", "not universal"])
def test_invalid_configs(bad):
    with pytest.raises(Exception):
        agent_lib.Agent(dict(UNI, **bad), seed=1)


@pytest.mark.parametrize("cfg", CFGS, ids=IDS)
def test_cpu_blank_matches_reference(cfg):
    """The C++ model reads a blank state (0.5): its forward pass, loss and gradients are the reference's with state None."""
    img, drv, act, es = batch()
    agent = agent_lib.Agent(cfg, seed=2)
    o = agent.forward_uni(img, drv, act, es, values=True, move_probs=True, next_body=True)
    c = agent.config
    P = ref.params_from_flat(agent.params(), agent.param_layout(), c, dtype=torch.float32)
    S, T = img.shape[:2]
    np.testing.assert_allclose(o["values"], ref_values(c, ref.forward_uni(P, c, img, drv, act, es)["logits"], S, T), atol=2e-5)
    agent_lib.set_threads(4)
    T2, N = 16, 3
    ro, data = store_rollout(agent, T=T2, N=N, seed=3, p_done=0.2)
    ids = [0, 2, 4, 5]
    pc = dict(agent_lib.DEFAULT_PPO, use_ppo=0)
    loss_cpp, _ = agent.debug_grads(ro, np.zeros(N, np.float32), pc, ids)
    P = ref.params_from_flat(agent.params(), agent.param_layout(), c)
    z = np.zeros((T2, N))
    loss_ref, _ = ref.ppo_loss(P, c, pc, seq_batch(data, ids, c["ctx"], N, z, z, c["img"] // c["pred_img"]))
    loss_ref.backward()
    assert abs(loss_cpp - loss_ref.item()) < 1e-4 * max(1.0, abs(loss_ref.item())), (loss_cpp, loss_ref.item())
    compare_grads(agent, P)
    agent.close()


@gpu
@pytest.mark.parametrize("cfg", CFGS, ids=IDS)
def test_gpu_forward_with_state_matches_reference(cfg):
    """Given reads: every token's values and the writes, against the reference; no reads: the C++ model's blank state."""
    img, drv, act, es = batch(seed=5, T=cfg["ctx"])
    es[1, 3] = 1
    agent = agent_lib.Agent(cfg, seed=2)
    c = agent.config
    S, T = img.shape[:2]
    V = ref.uni_dims(c)[1]
    st = states(1, S, T, V)
    m = cuda_lib.CudaModel(agent)
    o = m.forward_uni(img, drv, act, es, values=True, move_probs=True, next_body=True, state=st, writes=True)
    P = ref.params_from_flat(agent.params(), agent.param_layout(), c, dtype=torch.float32)
    r = ref.forward_uni(P, c, img, drv, act, es, state=st)
    np.testing.assert_allclose(o["values"], ref_values(c, r["logits"], S, T), atol=3e-5)
    np.testing.assert_allclose(o["writes"], ref.uni_writes(c, r["logits"]).detach().numpy(), atol=3e-5)
    o2 = m.forward_uni(img, drv, act, es, values=False, move_probs=True, next_body=True, state=st, writes=True)
    np.testing.assert_allclose(o2["writes"], o["writes"], atol=1e-6)    # the last two rows only: the same writes
    np.testing.assert_allclose(o2["move_probs"], o["move_probs"], atol=2e-5)
    blank = m.forward_uni(img, drv, act, es, values=True, move_probs=True, next_body=True)   # no reads given
    cpu = agent.forward_uni(img, drv, act, es, values=True, move_probs=True, next_body=True)
    for k in ("values", "move_probs", "next_body"):
        np.testing.assert_allclose(blank[k], cpu[k], atol=2e-5)
    assert np.abs(blank["values"] - o["values"]).max() > 1e-4            # the reads reach the model
    m.close()
    agent.close()


class Ro:
    """A rollout of S windows of ctx steps for CudaModel.set_batch: batch() as cuda_lib.Rollout's."""

    def __init__(self, img, drv, act, es, ndrv):
        S, T = img.shape[:2]
        self.a = [img, drv, np.zeros((S, T, 3), np.float32), np.zeros((S, T), np.float32), es, act, None, ndrv]

    def batch(self):
        return list(self.a)


def pair(seed, S, c):
    """Two consecutive windows of S bodies: (window 1, window 2) as Ro and as reference batches."""
    T = c["ctx"]
    img, drv, act, es = batch(seed, S, 2 * T, c["img"])
    es[:, 0] = 1
    ndrv = np.random.default_rng(seed + 1).random((S, 2 * T, 2)).astype(np.float32)
    out = []
    for sl in (slice(0, T), slice(T, 2 * T)):
        w = [np.ascontiguousarray(x[:, sl]) for x in (img, drv, act, es, ndrv)]
        out.append((Ro(*w), dict(images=w[0], drives=w[1], action=w[2], ep_start=w[3], next_drives=w[4])))
    return out


@gpu
@pytest.mark.parametrize("grad", [True, False], ids=["carry", "detached"])
@pytest.mark.parametrize("cfg", CFGS, ids=IDS)
def test_pair_update_matches_reference(cfg, grad):
    """The GPU's two passes (the second window reading the first's writes) give the reference pair loss and its gradient,
    with the gradient through the writes (carry) or without (detached)."""
    agent = agent_lib.Agent(cfg, seed=4)
    c = agent.config
    S, V = 3, ref.uni_dims(c)[1]
    (ro1, b1), (ro2, b2) = pair(7, S, c)
    reads1 = states(2, S, c["ctx"], V)
    m = cuda_lib.CudaModel(agent)
    st = m.update_pairs(ro1, ro2, reads1, PC, grad=grad, apply=False)
    g = m.grads().copy()
    P = ref.params_from_flat(agent.params(), agent.param_layout(), c)
    loss, _ = ref.uni_pair_loss(P, c, b1, b2, reads1, grad)
    loss.backward()
    want = np.zeros(agent.num_params)
    for name, off, size in agent.param_layout():
        if P[name].grad is not None:
            want[off:off + size] = P[name].grad.numpy().ravel()
    assert abs(st["loss"] - loss.item()) < 1e-4 * max(1.0, abs(loss.item())), (st["loss"], loss.item())
    from test_cuda_model import compare
    compare(agent, want, g, 2e-4)
    m.close()
    agent.close()


@gpu
def test_the_carry_changes_the_gradient():
    """With the gradient through the writes, the first window's weights get more than its own loss gives them."""
    agent = agent_lib.Agent(UNI, seed=4)
    c = agent.config
    (ro1, _), (ro2, _) = pair(7, 3, c)
    reads1 = states(2, 3, c["ctx"], ref.uni_dims(c)[1])
    m = cuda_lib.CudaModel(agent)
    m.update_pairs(ro1, ro2, reads1, PC, grad=True, apply=False)
    a = m.grads().copy()
    m.update_pairs(ro1, ro2, reads1, PC, grad=False, apply=False)
    b = m.grads().copy()
    assert np.abs(a - b).max() > 1e-6
    m.close()
    agent.close()


@gpu
@pytest.mark.parametrize("cfg", [AGE, PYR], ids=["age", "piece"])
def test_ring_reads_and_writes_match_full_passes(cfg):
    """Ring steps read the queue's oldest writes and write in their place: each step's move prediction and write are a
    full pass's over the steps so far with the same reads."""
    N, T = 3, cfg["ctx"]
    x, d, a, es = batch(seed=8, S=N, T=T)
    agent = agent_lib.Agent(cfg, seed=2)
    V = ref.uni_dims(agent.config)[1]
    m = cuda_lib.CudaModel(agent)
    m.uni_ring_reset(N)
    m.st_q[:] = np.random.default_rng(3).random(m.st_q.shape)   # writes as if from earlier ticks
    queue = m.st_q.copy()
    reads = np.zeros((N, T, V), np.float32)
    for t in range(T):
        reads[:, t] = queue[t]
        full_a = a[:, :t + 1].copy()
        full_a[:, -1] = 0.0
        f = m.forward_uni(x[:, :t + 1], d[:, :t + 1], full_a, es[:, :t + 1], values=False, move_probs=True,
                          state=reads[:, :t + 1], writes=True)
        p = m.uni_ring_step(x[:, t], d[:, t], es[:, t])
        np.testing.assert_allclose(m.last_read, queue[t], atol=0)
        np.testing.assert_allclose(p, f["move_probs"][:, -1], atol=3e-5, err_msg="tick %d" % t)
        np.testing.assert_allclose(m.st_q[t], f["writes"][:, -1], atol=3e-5, err_msg="tick %d" % t)
        m.uni_ring_move(a[:, t])
    assert m.st_i == 0   # the queue went round once
    m.state_blank = True
    m.uni_ring_step(x[:, 0], d[:, 0], es[:, 0])
    assert (m.last_read == 0.5).all()
    m.close()
    agent.close()


@gpu
@pytest.mark.parametrize("seed", [11, 12])
def test_copy_task_learned_through_the_carry(seed):
    """A value shown in the first window (the body's hunger at every step) is the target of the second window's move
    tokens, whose own bodies and frames are noise. With the gradient through the carry the writes learn to keep it: with
    32 bins (as the runs) by update 750 in 3 of 3 seeds, the error falling suddenly to about 0.001; with 8 bins it sat on
    a plateau, and learned within 3,000 updates in only half the runs.
    Detached, the writes are never trained, yet the reader can learn to decode what they happen to carry: in one run part
    of it (error .04, against .083 for a guess), in another all of it (4e-5). So the detached arm is not asserted to fail;
    this tests the carry."""
    cfg = dict(UNI, d_model=64, n_head=4, d_ff=128, bins=32)
    S, T = 32, cfg["ctx"]
    err = {}
    for grad in (True,):
        agent = agent_lib.Agent(cfg, seed=seed)
        m = cuda_lib.CudaModel(agent)
        V = m.V
        rng = np.random.default_rng(5)

        def make(n):
            h0 = rng.random(n).astype(np.float32)
            img = rng.integers(0, 256, (n, 2 * T, 32, 32, 3), dtype=np.uint8)
            drv = rng.random((n, 2 * T, 2)).astype(np.float32)
            drv[:, :T, 0] = h0[:, None]
            act = np.zeros((n, 2 * T, 3), np.float32)
            es = np.zeros((n, 2 * T), np.uint8)
            es[:, 0] = 1
            ndrv = np.zeros((n, 2 * T, 2), np.float32)
            ndrv[:, :, 0] = h0[:, None]
            return h0, [np.ascontiguousarray(z[:, :T]) for z in (img, drv, act, es, ndrv)], \
                [np.ascontiguousarray(z[:, T:]) for z in (img, drv, act, es, ndrv)]

        blank = np.full((S, T, V), 0.5, np.float32)
        for _ in range(1500):
            _, w1, w2 = make(S)
            m.update_pairs(Ro(*w1), Ro(*w2), blank, dict(PC, lr=3e-3), grad=grad, stats=False)
        h0, w1, w2 = make(64)
        wr = m.forward_uni(w1[0], w1[1], w1[2], w1[3], values=False, move_probs=False, state=np.full((64, T, V), 0.5),
                           writes=True)["writes"]
        nb = m.forward_uni(w2[0], w2[1], w2[2], w2[3], values=False, move_probs=False, next_body=True, state=wr)["next_body"]
        err[grad] = float(((nb[:, :, 0] - h0[:, None]) ** 2).mean())
        m.close()
        agent.close()
    assert err[True] < 0.01, err

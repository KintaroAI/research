"""Task 00089: transformers switched off. A training batch's step mask (CudaModel.set_step_mask): a missed step is no key
for the seen steps and carries no loss. The ring's skipped ticks (uni_ring_skip): their slots stay unseen, and a step
before them keeps its age in ticks. Acting with gaps equals training with the same mask.

Requires libvmcuda.so, a GPU and torch (CPU).
"""
import math
import os
import sys

import numpy as np
import pytest

torch = pytest.importorskip("torch")
F = torch.nn.functional
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import agent_lib  # noqa: E402
import cuda_lib  # noqa: E402
from reference import torch_reference as ref  # noqa: E402
from test_model_reference import SMALL, seq_batch  # noqa: E402

pytestmark = pytest.mark.skipif(not cuda_lib.available() or not cuda_lib.has("vmc_set_step_mask"),
                                reason="needs libvmcuda.so with vmc_set_step_mask and a GPU")
UNI = dict(SMALL, patch=8, universal=1, bins=8, gray=1, in_bins=1, age_key=1)
BAND = dict(UNI, uni_pyr=2, sum_thumb=1, band_n=4, band_k=2, band_cols=2)
CHAN = dict(BAND, uni_chan=1, band_win=1)
CFGS, IDS = [UNI, BAND, CHAN], ["plain", "piece", "piece_channels_window"]
PC = dict(use_ppo=0, pred_coef=1.0, drive_coef=1.0, pred_changed_weight=10.0, move_coef=1.0, epochs=1, minibatches=1,
          lr=3e-4, pi_detach=1)


def batches(agent, seed=2):
    """A GPU batch (set_batch's arrays) and the same as the reference's dict, 5 sequences of ctx steps."""
    from test_cuda_model import cuda_batch
    from test_patch_tokens import store_rollout
    c = agent.config
    N, T = 4, 2 * c["ctx"]
    ro, data = store_rollout(agent, T=T, N=N, seed=seed, p_done=0.0)
    ids = [0, 2, 3, 5, 6]
    z = np.zeros((T, N))
    return cuda_batch(agent, data, ids, N), seq_batch(data, ids, c["ctx"], N, z, z, c["img"] // c["pred_img"])


def a_mask(S, T, seed=5):
    rng = np.random.default_rng(seed)
    m = (rng.random((S, T)) < 0.7).astype(np.uint8)
    m[0] = 1                    # a sequence fully seen
    m[1] = 0                    # one fully missed
    m[2, -1] = 0                # the last step missed
    return m


def gpu_loss(agent, b, mask=None, images=None):
    m = cuda_lib.CudaModel(agent)
    bb = list(b)
    if images is not None:
        bb[0] = images
    m.set_batch(*bb)
    if mask is not None:
        m.set_step_mask(mask)
    st, g = m.loss_and_grads(PC), m.grads().copy()
    m.close()
    return st, g


@pytest.mark.parametrize("cfg", CFGS, ids=IDS)
def test_the_step_mask_matches_the_reference(cfg):
    from test_cuda_model import _ref_flat_grads, compare
    agent = agent_lib.Agent(cfg, seed=4)
    b, batch = batches(agent)
    mask = a_mask(*b[0].shape[:2])
    st, g = gpu_loss(agent, b, mask)
    want_l, want_g = _ref_flat_grads(agent, dict(batch, step_mask=mask.astype(bool)))
    assert abs(st["loss"] - want_l) < 1e-4 * max(1.0, abs(want_l))
    compare(agent, want_g, g, 1e-4)
    l0, _ = _ref_flat_grads(agent, batch)
    assert abs(want_l - l0) > 1e-3 * abs(l0)                       # the mask mattered
    agent.close()


@pytest.mark.parametrize("cfg", [BAND], ids=["piece"])
def test_every_step_seen_is_no_mask(cfg):
    agent = agent_lib.Agent(cfg, seed=4)
    b, _ = batches(agent)
    st0, g0 = gpu_loss(agent, b)
    st1, g1 = gpu_loss(agent, b, np.ones(b[0].shape[:2], np.uint8))
    stn, gn = gpu_loss(agent, b)                                     # the GPU's own noise: atomics sum in any order
    assert st0["loss"] == st1["loss"] == stn["loss"]
    assert np.abs(g1 - g0).max() <= max(2 * np.abs(gn - g0).max(), 1e-12) + 1e-9
    agent.close()


@pytest.mark.parametrize("cfg", [BAND, CHAN], ids=["piece", "piece_channels_window"])
def test_a_missed_steps_frames_change_nothing(cfg):
    """Its frames are neither keys nor targets of a seen step (the move token's target is the next body, which stays)."""
    agent = agent_lib.Agent(cfg, seed=4)
    b, _ = batches(agent)
    mask = a_mask(*b[0].shape[:2])
    other = b[0].copy()
    gone = mask == 0
    other[gone] = np.random.default_rng(9).integers(0, 256, other[gone].shape, dtype=np.uint8)
    st0, g0 = gpu_loss(agent, b, mask)
    st1, g1 = gpu_loss(agent, b, mask, images=other)
    assert abs(st0["loss"] - st1["loss"]) < 1e-6 * abs(st0["loss"])
    np.testing.assert_allclose(g1, g0, rtol=1e-5, atol=1e-7)
    st2, g2 = gpu_loss(agent, b, None, images=other)                # without the mask they do change it
    assert abs(st2["loss"] - st0["loss"]) > 1e-4
    agent.close()


def test_the_next_batch_clears_the_mask():
    agent = agent_lib.Agent(BAND, seed=4)
    b, _ = batches(agent)
    m = cuda_lib.CudaModel(agent)
    m.set_batch(*b)
    l0 = m.loss_and_grads(PC)["loss"]
    m.set_batch(*b)
    m.set_step_mask(a_mask(*b[0].shape[:2]))
    l1 = m.loss_and_grads(PC)["loss"]
    m.set_batch(*b)
    l2 = m.loss_and_grads(PC)["loss"]
    assert l1 != l0 and l2 == l0
    with pytest.raises(RuntimeError):
        m.set_step_mask(np.ones((2, BAND["ctx"]), np.uint8))       # not the batch's shape
    m.close()
    agent.close()


def ticks(n, N, seed=0, img=32):
    rng = np.random.default_rng(seed)
    x = rng.integers(0, 256, (N, n, img, img, 3), dtype=np.uint8)
    d = rng.random((N, n, 2)).astype(np.float32)
    a = rng.uniform(-1, 1, (N, n, 3)).astype(np.float32)
    a[..., 2] = (a[..., 2] > 0).astype(np.float32)
    return x, d, a


class TickStream:
    """The ring in torch, one body, by ticks: each kept step's K and V per layer with its tick; a new step at tick t reads
    the kept steps of the last ctx ticks, each at age t - its tick."""

    def __init__(self, agent):
        self.c = agent.config
        self.P = ref.params_from_flat(agent.params(), agent.param_layout(), self.c)
        self.steps = []                                                 # (tick, [(k, v) per layer])

    def step(self, t, img, drv, act, keep):
        """The move's probabilities [3, K] and the predicted next hunger (the move token's first value)."""
        c, P = self.c, self.P
        nt, V, K = ref.uni_dims(c)
        d, L, nh = c["d_model"], c["n_layer"], c["n_head"]
        hs = d // nh
        v, used = ref.uni_values(c, img[None, None], drv[None, None], act[None, None])
        m = (np.arange(V)[None, :] < used[:, None]).astype(np.float64)
        x = torch.tensor(ref.two_hot(v, K) * m[..., None], dtype=torch.float64).reshape(1, nt, V * K)
        h = F.linear(x, P["uni_in_w"], P["uni_in_b"]) + P["uni_slot"][None]
        old = [s for s in self.steps if t - s[0] < c["ctx"]]
        new = []
        for l in range(L):
            y = F.layer_norm(h, (d,), P["ln1w"][l], P["ln1b"][l], eps=1e-5)
            q, k, val = F.linear(y, P["qkvw"][l], P["qkvb"][l]).split(d, dim=-1)
            kk = torch.cat([s[1][l][0] for s in old] + [k], 1)
            vv = torch.cat([s[1][l][1] for s in old] + [val], 1)
            ages = torch.tensor([t - s[0] for s in old for _ in range(nt)] + [0] * nt)
            e = P["wpe"][ages]
            sc = ((q.reshape(nt, 1, nh, hs) * (kk + e[None]).reshape(1, -1, nh, hs)).sum(-1) / math.sqrt(hs))
            mask = torch.arange(kk.shape[1])[None] <= len(old) * nt + torch.arange(nt)[:, None]
            att = sc.masked_fill(~mask[..., None], float("-inf")).softmax(1)
            o = (att[..., None] * vv.reshape(1, -1, nh, hs)).sum(1).reshape(1, nt, d)
            h = h + F.linear(o, P["attprojw"][l], P["attprojb"][l])
            y = F.layer_norm(h, (d,), P["ln2w"][l], P["ln2b"][l], eps=1e-5)
            y = F.gelu(F.linear(y, P["fcw"][l], P["fcb"][l]), approximate="tanh")
            h = h + F.linear(y, P["fcprojw"][l], P["fcprojb"][l])
            new.append((k.detach(), val.detach()))
        if keep:
            self.steps.append((t, new))
        z = F.layer_norm(h, (d,), P["lnfw"], P["lnfb"], eps=1e-5)
        lg = F.linear(z, P["uni_out_w"], P["uni_out_b"])[0]
        move = lg[nt - 2].reshape(V, K)[:3].softmax(-1).detach().numpy()
        p = lg[nt - 1].reshape(V, K)[0].softmax(-1).detach().numpy()
        return move, float((p * np.arange(K) / (K - 1)).sum())


# ticks the model is off: a gap inside the window, one past it, a single tick, and a gap across the buffer's wrap
OFF = set(range(5, 8)) | set(range(15, 31)) | {33} | set(range(37, 41))


@pytest.mark.parametrize("cfg", [UNI, BAND], ids=["plain", "piece"])
def test_a_ring_with_gaps_equals_the_tick_stream(cfg):
    N, n = 2, 48
    x, d, a = ticks(n, N, seed=3)
    agent = agent_lib.Agent(cfg, seed=2)
    m = cuda_lib.CudaModel(agent)
    m.uni_ring_reset(N)
    st = [TickStream(agent) for _ in range(N)]
    zero = np.zeros(3, np.float32)
    cand = np.random.default_rng(1).uniform(-1, 1, (N, 4, 3)).astype(np.float32)
    es = np.zeros((N, n), np.uint8)
    es[:, 0] = 1
    missed = 0
    for t in range(n):
        if t in OFF:
            missed += 1
            continue
        if missed:
            m.uni_ring_skip(missed)
            missed = 0
        pr = m.uni_ring_step(x[:, t], d[:, t], es[:, t])
        hc = m.uni_candidates(cand)
        m.uni_ring_move(a[:, t])
        for b in range(N):
            want, _ = st[b].step(t, x[b, t], d[b, t], zero, False)
            np.testing.assert_allclose(pr[b], want, atol=5e-5, err_msg="tick %d body %d" % (t, b))
            if cfg is BAND:
                for j in range(cand.shape[1]):
                    _, h = st[b].step(t, x[b, t], d[b, t], cand[b, j], False)
                    assert abs(hc[b, j] - h) < 5e-5, "tick %d body %d candidate %d" % (t, b, j)
            st[b].step(t, x[b, t], d[b, t], a[b, t], True)
    m.close()
    agent.close()


@pytest.mark.parametrize("cfg", [UNI, BAND], ids=["plain", "piece"])
def test_acting_with_gaps_equals_training_with_the_mask(cfg):
    """Within the first ctx ticks the ring is the full pass: with gaps, the reference's forward with the same step mask."""
    N, T = 2, cfg["ctx"]
    x, d, a = ticks(T, N, seed=6)
    seen = np.array([1, 1, 0, 0, 1, 0, 1, 1], bool)
    agent = agent_lib.Agent(cfg, seed=2)
    P = ref.params_from_flat(agent.params(), agent.param_layout(), agent.config)
    nt, V, K = ref.uni_dims(agent.config)
    m = cuda_lib.CudaModel(agent)
    m.uni_ring_reset(N)
    es = np.zeros((N, T), np.uint8)
    es[:, 0] = 1
    missed = 0
    for t in range(T):
        if not seen[t]:
            missed += 1
            continue
        if missed:
            m.uni_ring_skip(missed)
            missed = 0
        pr = m.uni_ring_step(x[:, t], d[:, t], es[:, t])
        full_a = a[:, :t + 1].copy()
        full_a[:, -1] = 0.0
        out = ref.forward_uni(P, agent.config, x[:, :t + 1], d[:, :t + 1], full_a, es[:, :t + 1],
                              np.broadcast_to(seen[None, :t + 1], (N, t + 1)))
        want = out["logits"].reshape(N, t + 1, nt, V, K)[:, -1, nt - 2, :3].softmax(-1).detach().numpy()
        np.testing.assert_allclose(pr, want, atol=5e-5, err_msg="tick %d" % t)
        m.uni_ring_move(a[:, t])
    m.close()
    agent.close()


def test_skipping_without_gaps_is_the_old_ring():
    """uni_ring_skip(0) does nothing: a ring that never skips is the ring as before (test_ring.py)."""
    N, n = 2, 20
    x, d, a = ticks(n, N, seed=8)
    agent = agent_lib.Agent(BAND, seed=2)
    out = []
    for skip in (False, True):
        m = cuda_lib.CudaModel(agent)
        m.uni_ring_reset(N)
        ps = []
        for t in range(n):
            if skip:
                m.uni_ring_skip(0)
            ps.append(m.uni_ring_step(x[:, t], d[:, t]))
            m.uni_ring_move(a[:, t])
        out.append(np.stack(ps))
        m.close()
    np.testing.assert_array_equal(out[0], out[1])
    agent.close()


def test_a_life_with_an_outage(tmp_path):
    """16 transformers off a quarter of the time: each piece on about 75% of the ticks, every one learns, and the life
    refuses an outage without the ring."""
    import json
    import subprocess
    dev = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
    out = str(tmp_path / "life")
    cmd = [sys.executable, "lifetime.py", "--config", "configs/U_default.json", "--seed", "3", "-o", out,
           "--set", "model.band_n=16", "--set", "online.future_k=4", "--set", "online.off=0.25", "--set", "online.steps=400",
           "--set", "online.log_every=200", "--set", "online.save_every=400", "--set", "online.off_span=8"]
    r = subprocess.run(cmd, cwd=dev, capture_output=True, text=True, timeout=900)
    assert r.returncode == 0, r.stderr[-3000:]
    rows = [json.loads(line) for line in open(os.path.join(out, "metrics.jsonl"))]
    on = np.array([[r_["on_p%d" % k] for k in range(16)] for r_ in rows])
    assert 0.15 < rows[-1]["off_share"] < 0.35 and (on > 0.3).all() and (on < 1).any()
    assert all(r_["uni_ce_body_p%d" % k] > 0 for r_ in rows[1:] for k in range(16))
    bad = cmd[:-2] + ["--set", "online.ring=0"]
    r = subprocess.run(bad, cwd=dev, capture_output=True, text=True, timeout=300)
    assert r.returncode != 0 and "online.off needs" in r.stderr

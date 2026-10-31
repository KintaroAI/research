"""The key and value ring (task 00062, CudaModel.uni_ring_*): only a new step's tokens are computed. Within the first ctx
ticks it is the full pass, exactly; past them a streaming reference in torch, across the buffer's wrap; episode starts;
candidates; the move written back.

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
from test_model_reference import SMALL  # noqa: E402

pytestmark = pytest.mark.skipif(not cuda_lib.available(), reason="needs libvmcuda.so and a GPU")
UNI = dict(SMALL, patch=8, universal=1, bins=8, gray=1, in_bins=1, age_key=1)
PYR = dict(UNI, uni_pyr=2, sum_thumb=1)
BAND = dict(PYR, band_n=4, band_k=2, band_cols=2)
CFGS = [UNI, PYR, BAND]
IDS = ["plain", "pyramid", "piece"]


def ticks(n, N, seed=0, img=32):
    rng = np.random.default_rng(seed)
    x = rng.integers(0, 256, (N, n, img, img, 3), dtype=np.uint8)
    d = rng.random((N, n, 2)).astype(np.float32)
    a = rng.uniform(-1, 1, (N, n, 3)).astype(np.float32)
    a[..., 2] = (a[..., 2] > 0).astype(np.float32)
    return x, d, a


@pytest.mark.parametrize("cfg", CFGS, ids=IDS)
def test_first_ticks_equal_a_full_pass(cfg):
    N, T = 3, cfg["ctx"]
    x, d, a = ticks(T, N)
    agent = agent_lib.Agent(cfg, seed=2)
    m = cuda_lib.CudaModel(agent)
    m.uni_ring_reset(N)
    cand = np.random.default_rng(1).uniform(-1, 1, (N, 4, 3)).astype(np.float32)
    es = np.zeros((N, T), np.uint8)
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


class Stream:
    """The ring in torch, one body: K and V of the last ctx steps per layer; a new step reads them, by age."""

    def __init__(self, agent):
        self.c = agent.config
        self.P = ref.params_from_flat(agent.params(), agent.param_layout(), self.c)
        self.steps = [[] for _ in range(self.c["n_layer"])]             # per layer: (k, v) of each kept step

    def step(self, img, drv, act, keep):
        """The new step's tokens: the move's bins [3, K] if not keep; with keep (the move made) its K and V stay."""
        c, P = self.c, self.P
        S = 1
        nt, V, K = ref.uni_dims(c)
        d, L, nh = c["d_model"], c["n_layer"], c["n_head"]
        hs = d // nh
        v, used = ref.uni_values(c, img[None, None], drv[None, None], act[None, None])
        m = (np.arange(V)[None, :] < used[:, None]).astype(np.float64)
        x = torch.tensor(ref.two_hot(v, K) * m[..., None], dtype=torch.float64).reshape(S, nt, V * K)
        h = F.linear(x, P["uni_in_w"], P["uni_in_b"]) + P["uni_slot"][None]
        for l in range(L):
            y = F.layer_norm(h, (d,), P["ln1w"][l], P["ln1b"][l], eps=1e-5)
            q, k, val = F.linear(y, P["qkvw"][l], P["qkvb"][l]).split(d, dim=-1)
            old = self.steps[l][-(c["ctx"] - 1):]
            kk = torch.cat([o[0] for o in old] + [k], 1)
            vv = torch.cat([o[1] for o in old] + [val], 1)
            ages = torch.tensor([len(old) - i for i in range(len(old)) for _ in range(nt)] + [0] * nt)
            e = P["wpe"][ages]
            s = ((q.reshape(nt, 1, nh, hs) * (kk + e[None]).reshape(1, -1, nh, hs)).sum(-1) / math.sqrt(hs))   # [nt, n, nh]
            mask = torch.arange(kk.shape[1])[None] <= len(old) * nt + torch.arange(nt)[:, None]
            att = s.masked_fill(~mask[..., None], float("-inf")).softmax(1)
            o = (att[..., None] * vv.reshape(1, -1, nh, hs)).sum(1).reshape(S, nt, d)
            h = h + F.linear(o, P["attprojw"][l], P["attprojb"][l])
            y = F.layer_norm(h, (d,), P["ln2w"][l], P["ln2b"][l], eps=1e-5)
            y = F.gelu(F.linear(y, P["fcw"][l], P["fcb"][l]), approximate="tanh")
            h = h + F.linear(y, P["fcprojw"][l], P["fcprojb"][l])
            if keep:
                self.steps[l].append((k.detach(), val.detach()))
        z = F.layer_norm(h, (d,), P["lnfw"], P["lnfb"], eps=1e-5)
        lg = F.linear(z[:, nt - 2], P["uni_out_w"], P["uni_out_b"]).reshape(V, K)
        return lg[:3].softmax(-1).detach().numpy()

    def reset(self):
        self.steps = [[] for _ in range(self.c["n_layer"])]


@pytest.mark.parametrize("cfg", [UNI, BAND], ids=["plain", "piece"])
def test_past_the_window_equals_the_streaming_reference(cfg):
    N, n = 2, 44                                                        # past the wrap (2 ctx) several times
    x, d, a = ticks(n, N, seed=3)
    agent = agent_lib.Agent(cfg, seed=2)
    m = cuda_lib.CudaModel(agent)
    m.uni_ring_reset(N)
    st = [Stream(agent) for _ in range(N)]
    zero = np.zeros(3, np.float32)
    es = np.zeros((N, n), np.uint8)
    es[1, 20] = 1                                                       # body 1 starts anew at tick 20
    for t in range(n):
        pr = m.uni_ring_step(x[:, t], d[:, t], es[:, t])
        m.uni_ring_move(a[:, t])
        for b in range(N):
            if es[b, t]:
                st[b].reset()
            want = st[b].step(x[b, t], d[b, t], zero, False)
            st[b].step(x[b, t], d[b, t], a[b, t], True)
            np.testing.assert_allclose(pr[b], want, atol=5e-5, err_msg="tick %d body %d" % (t, b))
    m.close()
    agent.close()


def test_it_is_not_a_full_pass_past_the_window():
    """The accepted difference: a cached token keeps what it read when new."""
    N, n = 1, 12
    x, d, a = ticks(n, N, seed=4)
    agent = agent_lib.Agent(PYR, seed=2)
    for o, s in [(o, s) for n_, o, s in agent.param_layout() if n_ in ("qkvw", "attprojw")]:
        agent.params()[o:o + s] *= 5.0                                  # an untrained model barely attends
    agent.params_changed()
    m = cuda_lib.CudaModel(agent)
    m.uni_ring_reset(N)
    T = PYR["ctx"]
    for t in range(n):
        pr = m.uni_ring_step(x[:, t], d[:, t])
        m.uni_ring_move(a[:, t])
    w = a[:, n - T:n].copy()
    w[:, -1] = 0.0
    p = m.forward_uni(x[:, n - T:n], d[:, n - T:n], w, np.zeros((N, T), np.uint8), values=False,
                      move_probs=True)["move_probs"][:, -1]
    assert np.abs(pr - p).max() > 1e-4
    m.close()
    agent.close()


def test_refused_without_the_age_key():
    agent = agent_lib.Agent(dict(UNI, age_key=0), seed=1)
    m = cuda_lib.CudaModel(agent)
    with pytest.raises(Exception):
        m.uni_ring_reset(2)
    m.close()
    agent.close()

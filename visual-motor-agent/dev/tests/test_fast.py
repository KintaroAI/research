"""The flexible layer (task 00105, online.fast): the parameters hold W = S + F; F <- lam F - k lr clamp(g / sqrt(v_hat), -1, 1)
from the gradient g of the newest window, v_hat the slow update's Adam second moment; the parameters change by F's change.

On the GPU: the step against numpy, F decaying as lam^t with no gradient, S = P - F untouched by fast steps, F zero before
the first step. And eval_hunger.MemoryScore (the chain's memory measure, also scored in life) on a hand-made sequence.

Requires libvmagent.so; the GPU tests libvmcuda.so and a GPU.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import agent_lib  # noqa: E402
import cuda_lib  # noqa: E402
from test_model_reference import SMALL  # noqa: E402

HAVE_GPU = cuda_lib.available()
gpu = pytest.mark.skipif(not HAVE_GPU, reason="needs libvmcuda.so and a GPU")
UNI = dict(SMALL, patch=8, universal=1, bins=8, gray=1, in_bins=1, uni_chan=1, age_key=1)
PC = dict(agent_lib.DEFAULT_PPO, use_ppo=0, epochs=1, minibatches=1, lr=3e-4)


def batch(seed, S=3, T=8):
    rng = np.random.default_rng(seed)
    a = rng.uniform(-1, 1, (S, T, 3)).astype(np.float32)
    a[..., 2] = (a[..., 2] > 0).astype(np.float32)
    img = rng.integers(0, 256, (S, T, 32, 32, 3), dtype=np.uint8)
    drv = rng.random((S, T, 2)).astype(np.float32)
    z3, z1 = np.zeros((S, T, 3), np.float32), np.zeros((S, T), np.float32)
    return img, drv, z3, z1, np.zeros((S, T), np.uint8), a, None, rng.random((S, T, 2)).astype(np.float32)


def rule(F, G, v, step, lam, k):
    bc2 = 1.0 - PC["beta2"] ** step
    r = G.astype(np.float64) / (np.sqrt(v.astype(np.float64) / bc2) + PC["adam_eps"])
    return lam * F - k * PC["lr"] * np.clip(r, -1.0, 1.0)


@gpu
def test_step_matches_numpy_and_keeps_s():
    agent = agent_lib.Agent(UNI, seed=3)
    m = cuda_lib.CudaModel(agent)
    m.set_batch(*batch(1))
    m.update(PC)                                    # the slow update: Adam's moments, step 1
    assert not m.fast(0).any()                     # F is zero before the first fast step
    lam, k = 0.99, 10.0
    F = np.zeros(agent.num_params)
    for i in range(3):
        m.set_batch(*batch(10 + i, S=1))           # the newest window's gradient
        m.loss_and_grads(PC)
        G, v, P0 = m.grads(), m.fast(1), m.params().astype(np.float64)
        m.fast_step(lam, k, PC)
        want = rule(F, G, v, 1, lam, k)
        np.testing.assert_allclose(m.fast(0), want, rtol=1e-5, atol=1e-7)   # float32 against float64; steps are ~3e-3
        np.testing.assert_allclose(m.params() - m.fast(0), P0 - F, rtol=0, atol=2e-6)   # S = P - F untouched
        F = want
    assert np.abs(F).max() <= 3 * k * PC["lr"] + 1e-9   # each step at most k lr per parameter
    m.close()
    agent.close()


@gpu
def test_decay_without_gradient():
    agent = agent_lib.Agent(UNI, seed=4)
    m = cuda_lib.CudaModel(agent)
    m.set_batch(*batch(2))
    m.update(PC)
    m.set_batch(*batch(3, S=1))
    m.loss_and_grads(PC)
    m.fast_step(0.9, 10.0, PC)
    F0, S0 = m.fast(0).astype(np.float64), (m.params() - m.fast(0)).astype(np.float64)
    for _ in range(5):
        m.fast_step(0.9, 10.0, PC, use_grad=False)
    np.testing.assert_allclose(m.fast(0), F0 * 0.9 ** 5, rtol=1e-5, atol=1e-10)
    np.testing.assert_allclose(m.params() - m.fast(0), S0, rtol=0, atol=2e-6)
    m.close()
    agent.close()


class World:
    """A stub world for MemoryScore: which bodies see food this tick."""

    def __init__(self, inv):
        self.inv = np.asarray(inv, bool)

    def foods_in_view(self):
        return self.inv[:, None]


def test_memory_score_bins_and_meals():
    """Two bodies over 12 ticks: body 0 loses sight at tick 2 and stays out of view; body 1 eats at tick 3 (its food then out
    of view) and stays out. Ticks since the last sight: 1-8 then 9-16; ticks after body 1's meal count only as all."""
    from eval_hunger import MemoryScore, SINCE_NAMES
    ms = MemoryScore(2)
    for t in range(12):
        inv = [t < 2, t < 4]
        ms.see(World(inv), np.array([1.0, 0.5]))
        ms.ate([False, t == 3])
    assert SINCE_NAMES[:2] == ("1-8", "9-16")
    # body 0: out of view at ticks 2..11, ticks since 1..10: 8 in 1-8 and 2 in 9-16; body 1: ticks 4..11, since 1..8
    np.testing.assert_array_equal(ms.out_n[:2], [8 + 8, 2])
    np.testing.assert_allclose(ms.out_aim[:2], [8 * 1.0 + 8 * 0.5, 2 * 1.0])
    # without meals: body 1 ate at tick 3, when food was still in view; the meal is cleared at tick 3's see (in view), then
    # set by ate at tick 3, and from tick 4 on its food is out of view with a meal since: none of its ticks count
    np.testing.assert_array_equal(ms.nm_n[:2], [8, 2])
    out = ms.take()
    assert out["out_n"][:2] == [16, 2] and not ms.out_n.any()

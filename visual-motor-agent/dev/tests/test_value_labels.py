"""The bootstrapped value labels (task 00096, online.value_gamma): the direct sum, an episode end, and the labels a gated
memory hands to training (newest window, chunks written to long-term memory) against the same sum on the stream."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from online import DEFAULT_ONLINE, GatedMemory, future_labels, goal_labels, value_labels  # noqa: E402


def discounted(h, gamma, tail=0.5):
    """V_t = (1 - gamma) h_t + gamma V_{t+1}, from V_T = tail backwards: the exact exponentially weighted mean."""
    v = np.zeros(len(h) + 1)
    v[-1] = tail
    for t in range(len(h) - 1, -1, -1):
        v[t] = (1 - gamma) * h[t] + gamma * v[t + 1]
    return v[:-1]


def test_direct_sum():
    rng = np.random.default_rng(0)
    h = rng.random(300).astype(np.float32)
    for gamma, n in ((0.9, 4), (0.97, 4), (0.9, 1)):
        v = discounted(h.astype(np.float64), gamma)
        lab = value_labels(h, v.astype(np.float32), np.zeros(300, np.uint8), gamma, n)
        assert np.abs(lab[:300 - n] - v[:300 - n]).max() < 1e-6
        assert lab.min() >= 0 and lab.max() <= 1


def test_episode_end_and_stretch_end():
    h = np.array([0.2, 0.4, 0.6, 0.8, 1.0, 0.1], np.float32)
    vh = np.full(6, 0.9, np.float32)
    done = np.array([0, 0, 1, 0, 0, 0], np.uint8)          # step 2 ends the episode
    g, n = 0.5, 2
    lab = value_labels(h, vh, done, g, n)
    # step 0: two real ticks (0, 1), then the tail at step 2, which ends the episode but is still in it
    assert np.isclose(lab[0], (1 - g) * (h[0] + g * h[1]) + g ** 2 * vh[2])
    # step 1: real ticks 1, 2; the tail would be step 3, the next episode: real part only, renormalized
    assert np.isclose(lab[1], (h[1] + g * h[2]) / (1 + g))
    # step 2: its episode ends at once
    assert np.isclose(lab[2], h[2])
    # step 5: the stretch ends: its own tick only
    assert np.isclose(lab[5], h[5])


def test_future_mode_unchanged():
    rng = np.random.default_rng(1)

    class M:
        ndrv = rng.random((20, 2)).astype(np.float32)
        done = np.zeros(20, np.uint8)
    idx = np.arange(20)
    assert np.array_equal(goal_labels(M, idx, 4, "future"), future_labels(M.ndrv[:, 0], M.done, 4))


def test_gated_memory_labels():
    """A stream through a gated memory in value mode: the newest window's labels (lag n) and every stored chunk's labels
    equal value_labels on the whole stream."""
    L, n, gamma, T = 8, 4, 0.9, 400
    on = dict(DEFAULT_ONLINE, memory="gated", goal_mode="value", value_gamma=gamma, stm=256, replay=1600,
              store_warmup=10_000)
    gm = GatedMemory(on, L, 8, n + 1, seed=1)
    rng = np.random.default_rng(2)
    h, vh = rng.random(T).astype(np.float32), rng.random(T).astype(np.float32)
    full = value_labels(h, vh, np.zeros(T, np.uint8), gamma, n)
    z3 = np.zeros(3, np.float32)
    for t in range(T):
        gm.stm.add(img=np.zeros((8, 8, 3), np.uint8), drv=np.array([h[t - 1] if t else 0, 0], np.float32),
                   ndrv=np.array([h[t], 0], np.float32), pa=z3, pr=0.0, es=0, act=z3, rew=0.0, done=0, hit=0, cmd=0.0,
                   vh=vh[t])
        gm.note(np.zeros(3))
        gm.decide(t + 1)
        if t + 1 >= L + n:
            new = gm.newest(n)["pr"][0]
            first = t - n - L + 1
            assert np.allclose(new, full[first:first + L], atol=1e-6)
    st = gm.store
    assert st.n > 10
    for c in range(st.n):
        end = st.end[c]                       # the step (counted from 1) the chunk ended
        assert np.allclose(st.ltm["lbl"][c], full[end - L:end], atol=1e-6)

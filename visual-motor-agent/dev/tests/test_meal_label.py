"""The meal label (task 00108, online.value_target = meal): its parts read from the hunger alone, the mapping onto the
model's scale, the recursion it encodes, the windows a gated memory hands out with it, the per-body tolerance of
online.need, and a short lifetime with each on the GPU."""
import json
import os
import subprocess
import sys

import numpy as np
import pytest

DEV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, DEV)

import agent_lib  # noqa: E402
import cuda_lib  # noqa: E402
import online  # noqa: E402
from online import DEFAULT_ONLINE, GatedMemory, meal_parts, meal_target  # noqa: E402

HAVE_GPU = cuda_lib.available()
SMALL_UNI = dict(img=32, c1=4, c2=8, c3=8, feat=16, d_model=16, n_layer=2, n_head=4, d_ff=32, ctx=8, pred_img=16,
                 pc1=4, pc2=8, pred_hidden=16, use_prev_reward=1, logstd_init=-0.3, patch=8, universal=1, bins=8)


def hunger_series(T, seed):
    """Hunger as the world makes it: +0.001 a tick, a meal resets it to 0, a poison adds 0.5 (capped at 1)."""
    rng = np.random.default_rng(seed)
    h = [0.3]
    for _ in range(T):
        e, x = rng.random(), h[-1] + 0.001
        x = 0.0 if e < 0.04 else x + 0.5 if e < 0.06 else x
        h.append(min(1.0, x))
    return np.array(h)


def exact(h0, h1, gamma):
    """V[t] = r[t] + gamma (1 - meal[t]) V[t + 1], backwards from 0 after the last step, each V clipped as the label is."""
    meal = h1 < h0
    r = np.where(meal, 1.0, -(h1 - h0))
    v = np.zeros(len(h0) + 1)
    for t in range(len(h0) - 1, -1, -1):
        v[t] = np.clip(r[t] + (0.0 if meal[t] else gamma * v[t + 1]), -1.0, 1.0)
    return v[:-1]


def test_parts_from_hunger():
    h0 = np.array([0.200, 0.201, 0.202, 0.000, 0.001, 0.502])
    h1 = np.array([0.201, 0.202, 0.000, 0.001, 0.502, 0.503])
    real, tw = meal_parts(h0, h1, np.zeros(6, bool), 0.95)
    np.testing.assert_allclose(real, [-0.001, -0.001, 1.0, -0.001, -0.501, -0.001], atol=1e-6)
    np.testing.assert_allclose(tw, [0.95, 0.95, 0.0, 0.95, 0.95, 0.0])   # no tail after a meal, nor past the stretch
    _, tw = meal_parts(h0, h1, np.array([0, 1, 0, 0, 0, 0], bool), 0.95)
    assert tw[1] == 0.0                                                  # nor at an episode's end


def test_the_models_scale():
    assert meal_target(1.0, 0.0, 0.7) == pytest.approx(0.0)              # a meal now: T = 0
    assert meal_target(-0.001, 0.95, 0.0) == pytest.approx((1 - 0.949) / 2)
    assert meal_target(0.0, 0.95, 0.5) == pytest.approx(0.5)             # nothing in sight stays at .5
    assert meal_target(-0.9, 0.95, 1.0) == pytest.approx(1.0)            # clipped at V = -1


def test_the_label_is_the_recursion():
    """With exact tails each step's label is its exact value: a meal worth 1, carried back by gamma a tick, the hunger
    added on the way subtracted."""
    gamma = 0.95
    h = hunger_series(400, 0)
    h0, h1 = h[:-1], h[1:]
    v = exact(h0, h1, gamma)
    real, tw = meal_parts(h0, h1, np.zeros(len(h0), bool), gamma)
    got = meal_target(real[:-1], tw[:-1], (1 - v[1:]) / 2)
    np.testing.assert_allclose(got, (1 - v[:-1]) / 2, atol=1e-6)
    k = np.flatnonzero(h1 < h0)[3]
    assert v[k] == 1.0
    if not (h1[k - 1] < h0[k - 1] or h1[k - 1] - h0[k - 1] > 0.1):      # a plain tick before the meal: 0.95 less its drift
        assert v[k - 1] == pytest.approx(0.95 - (h1[k - 1] - h0[k - 1]), abs=1e-9)


def stream(on, h, L, seed=2):
    """A gated memory fed the hunger series h (before step t: h[t], after it: h[t + 1]); act[0] is the step's index / 1000
    and vh a random stored tail."""
    gm = GatedMemory(on, L, 8, 2, seed=1)
    T = len(h) - 1
    vh = np.random.default_rng(seed).random(T).astype(np.float32)
    for t in range(T):
        gm.stm.add(img=np.full((8, 8, 3), t % 256, np.uint8), drv=np.array([h[t], 0], np.float32),
                   ndrv=np.array([h[t + 1], 0], np.float32), pa=np.zeros(3, np.float32), pr=0.0, es=0,
                   act=np.array([t / 1000, 0, 0], np.float32), rew=0.0, done=0, hit=0, cmd=0.0, vh=vh[t])
        gm.note(np.zeros(3))
        gm.decide(t + 1)
    return gm, vh


def test_memory_windows_carry_the_meal_label():
    """With value_target = meal, every window (the newest and the replayed) carries the meal label's real part and tail
    weight for its steps, and its bootstrap window one step later; pr is the label with the stored tails."""
    L, gamma = 8, 0.95
    on = dict(DEFAULT_ONLINE, memory="gated", goal_mode="value", value_gamma=gamma, value_refresh=1, value_target="meal",
              stm=256, replay=1600, store_warmup=10_000)
    h = hunger_series(400, 1).astype(np.float32)
    gm, vh = stream(on, h, L)
    real, tw = meal_parts(h[:-1], h[1:], np.zeros(len(h) - 1, bool), gamma)
    assert gm.store.ltm["act"].shape[1] == L + 1 and gm.store.n > 10
    rows = gm.rows(16, np.random.default_rng(4), 1)
    steps = np.rint(rows["act"][..., 0] * 1000).astype(int)
    bsteps = np.rint(rows["b_act"][..., 0] * 1000).astype(int)
    assert np.array_equal(bsteps, steps + 1)
    np.testing.assert_allclose(rows["pr_real"], real[steps], atol=1e-6)
    np.testing.assert_allclose(rows["tw"], tw[steps], atol=1e-6)
    np.testing.assert_allclose(rows["pr"], meal_target(real[steps], tw[steps], vh[bsteps]), atol=1e-6)


def test_need_weighs_the_tolerance():
    """online.need gives each body its own tolerance: a sated body (a huge one) keeps its own move, a hungry one (none)
    takes the best candidate."""
    agent_lib.set_threads(4)
    agent = agent_lib.Agent(SMALL_UNI, seed=3)
    rng = np.random.default_rng(1)
    img = rng.integers(0, 256, (2, 8, 32, 32, 3), dtype=np.uint8)
    drv = rng.random((2, 8, 2)).astype(np.float32)
    act = rng.uniform(-1, 1, (2, 8, 3)).astype(np.float32)
    es = np.zeros((2, 8), np.uint8)
    es[:, 0] = 1
    move = lambda tol: online.uni_act(agent, img, drv, act, es, 8, np.random.default_rng(0), 6, tol)[0]   # noqa: E731
    none, huge, mixed = move(0.0), move(1e9), move(np.array([0.0, 1e9], np.float32))
    np.testing.assert_array_equal(mixed[0], none[0])
    np.testing.assert_array_equal(mixed[1], huge[1])


def chain(N):
    """A run of N ticks ending in a meal (the owner's check, 2026-11-17): hunger 0.2 rising 0.001 a tick, the last tick's
    move eats. Before and after each tick."""
    h0 = 0.2 + 0.001 * np.arange(N)
    h1 = h0 + 0.001
    h1[-1] = 0.0
    return h0, h1


@pytest.mark.parametrize("start", [0.5, 0.0], ids=["nothing_in_sight", "all_optimistic"])
def test_the_label_propagates_back_and_heals(start):
    """A table in place of the model, trained round after round on its own labels (lr 1: the prediction becomes its
    target). The meal's value travels back one tick per round: after r rounds the r ticks nearest the meal are exact,
    farther ones not yet. From any start, even every prediction a meal next tick (T = 0, the owner's worry: "predict 1 all
    the time"), the error shrinks by gamma per tick it travels, so it heals: within 1e-3 after 200 rounds."""
    N, gamma = 40, 0.95
    h0, h1 = chain(N)
    real, tw = meal_parts(h0, h1, np.zeros(N, bool), gamma)
    tw[-1] = 0.0
    want = (1 - exact(h0, h1, gamma)) / 2
    t_hat = np.full(N + 1, start)                                         # T predicted for each tick's move; one past the end
    err = []
    for r in range(1, 201):
        t_hat[:N] = meal_target(real, tw, t_hat[1:])                      # every tick's label from the predictions as they were
        err.append(np.abs(t_hat[:N] - want))
        if r in (5, 20):
            near = err[-1][N - r:]                                        # the r ticks nearest the meal: exact
            assert near.max() < 1e-6
            if start == 0.5:
                assert err[-1][:N - r].min() > 1e-3                      # farther: not yet reached
    assert err[-1].max() < 1e-3
    if start == 0.0:                                                      # the optimism shrinks round by round
        first = err[0][0]
        assert err[50][0] < first * gamma ** 40 + 1e-6


def test_optimism_heals_without_any_meal():
    """The owner's worry with no meal to anchor it: an endless cycle of plain ticks, every prediction starting at a meal
    next tick (T = 0, V = 1). Each round's labels are -0.001 + 0.95 x the predictions, so the excess over the truth (the
    drift's cost forever, -0.001 / (1 - 0.95) = -0.02) shrinks by exactly gamma per round."""
    N, gamma = 30, 0.95
    h0 = 0.2 + 0.001 * np.arange(N)
    real, tw = meal_parts(h0, h0 + 0.001, np.zeros(N, bool), gamma)
    tw[-1] = gamma                                                        # a cycle: the last tick's tail is the first tick
    truth = (1 - (-0.001 / (1 - gamma))) / 2
    t_hat = np.zeros(N)
    e0 = truth - t_hat[0]
    for r in range(1, 101):
        t_hat = meal_target(real, tw, np.roll(t_hat, -1))
        assert np.allclose(truth - t_hat, e0 * gamma ** r, atol=1e-4)   # optimistic still, by gamma^r of the start
    assert (truth - t_hat).max() < 0.01


def test_the_monitor_scores_predictions_by_ticks_to_the_meal():
    """MealMonitor: when the meal comes, each waiting tick's real value is known; predicted and real V add up by ticks to
    the meal (0: the move that ate, 1, 2-3, ...)."""
    gamma = 0.95
    mon = online.MealMonitor(1, gamma)
    h0, h1 = chain(4)                                                     # three plain ticks, then the meal
    pred_t = np.array([0.30, 0.20, 0.10, 0.05])                           # the predicted label (T) of each tick's move
    for t in range(4):
        mon.add(pred_t[t:t + 1], h0[t:t + 1], h1[t:t + 1])
    out = mon.take()
    v = exact(h0, h1, gamma)
    assert out["n"][:4] == [1, 1, 2, 0]                                   # k 0, 1, 2-3
    np.testing.assert_allclose(out["real"][:3], [v[3], v[2], v[1] + v[0]], atol=1e-9)
    np.testing.assert_allclose(out["pred"][:3], [1 - 2 * 0.05, 1 - 2 * 0.10, (1 - 0.40) + (1 - 0.60)], atol=1e-9)
    assert mon.take()["n"] == [0] * 7                                     # take() starts afresh


@pytest.mark.skipif(not HAVE_GPU, reason="needs libvmcuda.so and a GPU")
@pytest.mark.parametrize("need", [0, 1])
def test_a_short_life(tmp_path, need):
    out = str(tmp_path / "run")
    sets = ["online.steps=60", "online.log_every=30", "online.save_every=60", "online.future_k=4", "online.value_gamma=0.95",
            "online.value_n=1", "online.value_refresh=1", "online.value_target=meal", "online.need=%d" % need]
    cmd = [sys.executable, os.path.join(DEV, "lifetime.py"), "--config", os.path.join(DEV, "configs", "U_view.json"),
           "--seed", "2", "-o", out] + sum([["--set", s] for s in sets], [])
    env = dict(os.environ)
    env.pop("CUDA_MPS_PIPE_DIRECTORY", None)
    subprocess.run(cmd, check=True, cwd=DEV, capture_output=True, env=env)
    rows = [json.loads(line) for line in open(os.path.join(out, "metrics.jsonl"))]
    assert rows[-1]["step"] == 60 and "value_boot" in rows[-1] and 0.0 <= rows[-1]["value_boot"] <= 1.0
    assert all(len(rows[-1]["meal"][k]) == len(online.MEAL_BIN_NAMES) for k in ("pred", "real", "n"))


def test_meal_needs_its_settings(tmp_path):
    cmd = [sys.executable, os.path.join(DEV, "lifetime.py"), "--config", os.path.join(DEV, "configs", "U_view.json"),
           "-o", str(tmp_path / "run"), "--set", "online.steps=10", "--set", "online.value_target=meal"]
    r = subprocess.run(cmd, cwd=DEV, capture_output=True, text=True)
    assert r.returncode != 0 and "value_target" in r.stderr

"""The miss autopsy's scoring (task 00129, vm-00116): the landing point, the ticks to it and the hit test, computed from
the court's state by analyze_misses.py, agree with what the Breakout world then does."""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import analyze_misses as am  # noqa: E402
from world import make_config, make_world  # noqa: E402


def court(n=8, seed=3):
    env = json.load(open(os.path.join(os.path.dirname(__file__), "..", "configs", "U_breakout.json")))["env"]
    return make_world(n, make_config(env, auto_reset=True), seed=seed)


def moves(w, rng, kind):
    a = np.zeros((w.num_envs, 3))
    if kind == "random":
        a[:, 0] = rng.uniform(-1, 1, w.num_envs)
    else:   # the tracker with a wobble: returns most balls, so flights that go up and come down again are covered
        c = w.config
        a[:, 0] = np.clip(((w.ball[:, 0] + c.ball / 2) - (w.paddle + c.paddle_w / 2)) / c.paddle_speed + rng.normal(0, 0.5, w.num_envs), -1, 1)
    return a


def test_landing_and_hit_agree_with_the_world():
    for kind in ("random", "tracker"):
        w, rng = court(), np.random.default_rng(0)
        half = (w.config.paddle_w + w.config.ball) / 2.0
        landed = returned = 0
        for _ in range(3000):
            ttl, x = am.landing(w)
            a = moves(w, rng, kind)
            need, d = am.need_after(w, np.where(ttl > 0, x, 0.0), a[:, 0])
            _, _, _, info = w.step(a)
            hit = info["gt_paddle_hit"]
            now = ttl == 1
            assert (ttl[hit] == 1).all()                       # a ball is returned only on the tick the landing was due
            assert (hit[now] == (np.abs(d[now]) < half)).all()   # and exactly when the move left the paddle under it
            assert (need[now & hit] == 0).all() and (need[now & ~hit] >= 0).all()
            landed += int(now.sum())
            returned += int(hit.sum())
        assert landed > 100
        assert abs(int(w.hits.sum()) - returned) == 0
        assert abs(int(w.misses.sum()) - (landed - returned)) <= w.num_envs   # a miss is counted a few ticks after its landing
        if kind == "tracker":
            assert returned > 0.8 * landed


def test_countdown_and_fixed_landing_point():
    w, rng = court(n=4, seed=5), np.random.default_rng(1)
    prev_ttl, prev_x = am.landing(w)
    for _ in range(1500):
        _, _, _, info = w.step(moves(w, rng, "tracker"))
        ttl, x = am.landing(w)
        same = (prev_ttl > 1) & (ttl > 0)                      # the flight goes on: no landing between the two readings
        assert (ttl[same] == prev_ttl[same] - 1).all()
        assert np.allclose(x[same], prev_x[same])
        prev_ttl, prev_x = ttl, x


def test_need_after_and_concordance():
    w = court(n=2, seed=1)
    c = w.config
    w.paddle[:] = 20.0
    x = np.array([40.0, 22.0])                                 # the first ball lands right of the paddle, the second on it
    need, d = am.need_after(w, x, np.zeros(2))
    assert np.isclose(d[0], (40 + c.ball / 2) - (20 + c.paddle_w / 2)) and need[1] == 0
    assert np.isclose(need[0], abs(d[0]) - (c.paddle_w + c.ball) / 2)
    cand = np.array([[-1.0, 0.0, 1.0], [-1.0, 0.0, 1.0]])
    n2 = am.need_after(w, x, cand)[0]
    assert n2.shape == (2, 3) and n2[0, 2] < n2[0, 1] < n2[0, 0]   # moving right leaves less to travel
    good = np.array([[0.9, 0.5, 0.1], [0.3, 0.3, 0.3]])        # a label that is lower the nearer the candidate ends
    conc, pairs = am.concordance(n2, good)
    assert conc[0] == 1.0 and pairs[0] == 3
    assert am.concordance(n2, good[:, ::-1])[0][0] == 0.0
    assert np.isnan(conc[1]) or pairs[1] == 0 or conc[1] == 0.5   # the second paddle is already under its ball

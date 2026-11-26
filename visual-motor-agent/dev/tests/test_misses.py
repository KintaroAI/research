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


def test_need_after_a_held_move_is_where_the_world_puts_the_paddle():
    """Task 00131: a move made `hold` ticks running (action repeat). The formula against the court itself."""
    w, rng = court(n=16, seed=2), np.random.default_rng(3)
    x = np.full(16, 30.0)                                      # any landing point: only the paddle is under test
    checked = 0
    for _ in range(200):
        dx, h = rng.uniform(-1.3, 1.3, 16), rng.integers(1, 6, 16)
        want = am.need_after(w, x, dx, h)
        start = w.paddle.copy()
        a = np.zeros((16, 3))
        a[:, 0] = dx
        for k in range(1, 6):
            ok = np.ones(16, bool) if k == 1 else ok
            _, _, done, _ = w.step(a)
            ok &= ~np.asarray(done, bool)
            now = ok & (h == k)                                # the bodies whose hold ends on this tick, with no new court between
            got = am.need_after(w, x, np.zeros(16))
            assert np.allclose(want[0][now], got[0][now]) and np.allclose(want[1][now], got[1][now])
            checked += int(now.sum())
        assert (np.abs(w.paddle - start) > 0).any()
    assert checked > 1500
    one = am.need_after(w, x, dx)
    assert np.array_equal(one[0], am.need_after(w, x, dx, 1)[0]) and np.array_equal(one[1], am.need_after(w, x, dx, np.ones(16))[1])
    cand = rng.uniform(-1, 1, (16, 5))                         # candidates [N, M] with a hold per body
    n4 = am.need_after(w, x, cand, h)[0]
    for j in range(5):
        assert np.array_equal(n4[:, j], am.need_after(w, x, cand[:, j], h)[0])


def test_a_held_move_can_overshoot_what_one_tick_of_it_would_not():
    """The label's order is scored against where the candidates end: a full move toward a near ball ends under it after
    one tick and beyond it after four, so the two scorings can disagree about the same label."""
    w = court(n=1, seed=1)
    c = w.config
    w.paddle[:] = 20.0
    x = np.array([20.0 + c.paddle_w + 1.0])                    # the ball lands 1 px beyond the catching zone, to the right
    assert np.isclose(am.need_after(w, x, np.zeros(1))[0][0], 1.0)
    cand = np.array([[1.0, 0.1]])                              # a full move and a tenth of one, both toward the ball
    one, four = am.need_after(w, x, cand)[0], am.need_after(w, x, cand, 4)[0]
    assert one[0, 0] == 0 and one[0, 1] > 0                    # one tick: the full move is under the ball, the small one not yet
    assert np.isclose(four[0, 0], 4 * c.paddle_speed - 1.0 - (c.paddle_w + c.ball)) and four[0, 1] == 0   # held: the full move is past it
    full = np.array([[0.1, 0.2]])                              # a label (lower is better) that prefers the full move
    assert am.concordance(one, full)[0][0] == 1.0 and am.concordance(four, full)[0][0] == 0.0


# ------------------------------------------------------------------ one recording's measures (task 00131, vm-00118)
class Rec(dict):
    """A recording as np.load gives it: arrays by name and the list of names."""
    files = property(lambda self: list(self.keys()))


def flights():
    """Two bodies, 20 ticks, half a zone of 7 px: a flight of 16 ticks, then one of 4. Decisions on ticks 0, 4, 8, ..."""
    T = 20
    ttl = np.concatenate([np.arange(16, 0, -1), np.arange(4, 0, -1)])[:, None] * np.ones((1, 2), int)
    d = np.zeros((T, 2))
    d[[0, 4, 8, 12, 16], 0] = 20.0, 10.0, -9.0, -5.0, 10.0   # far; short of it; past it; under it; (the next ball) 3 px off
    d[[0, 4, 8, 12, 16], 1] = 3.0, 9.0, 30.0, 30.0, 0.0      # under it; out of the zone; further; no nearer; (the next ball) under
    hit = np.zeros((T, 2), bool)
    hit[15, 0] = hit[19, 1] = True                           # the tick with 1 to go: the ball is returned
    z = Rec(ttl=ttl, d=d, need=np.maximum(0.0, np.abs(d) - 7.0), hit=hit, miss=np.zeros((T, 2), bool),
            decided=(np.arange(T) % 4 == 0)[:, None] & np.ones((1, 2), bool), act_every=np.array(4))
    return z


def test_spans_sort_what_four_ticks_do_to_the_paddle():
    z = flights()
    for rec in (z, Rec({k: v for k, v in z.items() if k not in ("decided", "act_every")})):   # held, and deciding every tick
        s = am.spans(rec, 4)
        assert s["spans"] == 6 and np.isclose(s["away"], 5 / 6)       # starts 16, 12 and 8 ticks before the landing; not 4
        assert s["left"] == 1.0                                       # the one span that starts under the ball ends out of the zone
        assert s["all"] == 5 and [s["all_" + k] for k in ("under", "past", "short", "worse")] == [.2, .2, .2, .4]
        assert s["near"] == 3 and np.allclose([s["near_" + k] for k in ("under", "past", "short", "worse")], [1 / 3, 1 / 3, 0, 1 / 3])
        assert s["far"] == 2 and [s["far_" + k] for k in ("under", "past", "short", "worse")] == [0, 0, .5, .5]
        assert np.isclose(s["travel"], (10 + 19 + 4 + 6 + 21 + 0) / 6)


def test_endgame_reads_the_paddle_four_ticks_before_the_landing():
    e = am.endgame(flights(), 4)
    assert e["balls"] == 4 and e["under"] == .5 and e["near"] == .25
    assert e["ret_under"] == 1.0 and e["ret_near"] == 0.0 and e["ret_far"] == 0.0


def test_last_ticks_takes_the_deciding_ticks_near_the_landing():
    T = 10
    ttl = np.arange(T, 0, -1)[:, None]
    dec = (np.arange(T) % 4 == 0)[:, None]                    # ticks 0, 4, 8: 10, 6 and 2 ticks before the landing
    col = lambda *v: np.array(v, float).reshape(T, 1, 1)      # noqa: E731
    z = Rec(ttl=ttl, decided=dec, act_every=np.array(4), need=np.full((T, 1), 5.0), d=np.full((T, 1), 12.0),
            dx=np.array([1, 1, 1, 1, 1, 1, 1, 1, -1, -1], float)[:, None], hit=(ttl == 1), miss=np.zeros((T, 1), bool),
            conc=col(.1, 0, 0, 0, .6, 0, 0, 0, .8, 0), conc_hold=col(.1, 0, 0, 0, .7, 0, 0, 0, .9, 0), clear=col(*[.5] * T),
            clear_hold=col(*[.5] * T), spread=col(9, 9, 9, 9, .02, 9, 9, 9, .04, 9), grid_toward=col(*[1] * T),
            r_need=col(*[.2] * T), r_need_hold=col(*[.3] * T))
    m = am.last_ticks(z, 8)
    assert np.isclose(m["order"], .7) and np.isclose(m["order_hold"], .8) and m["toward"] == .5 and m["returned"] == 1.0
    assert np.isclose(m["spread"], .03) and np.isclose(m["r_need_hold"], .3)
    every = Rec({k: v for k, v in z.items() if k not in ("decided", "act_every", "conc_hold", "clear_hold", "r_need_hold")})
    m = am.last_ticks(every, 8)                               # a run that decides every tick: all 8 ticks, and one scoring
    assert np.isclose(m["order"], 1.4 / 8) and m["order_hold"] == m["order"] and m["toward"] == 6 / 8


def test_moves_reads_the_walls_and_how_a_decision_follows_the_last():
    T, speed = 12, 4.5
    dx = np.array([1, 1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1], float)
    d = np.array([30, 25.5, 21, 21, 21, 16.5, 12, 7.5, 3, 7.5, 12, 16.5])    # the moves of ticks 2 and 3 are cut by a wall
    ttl = np.arange(T, 0, -1)[:, None]
    dec = (np.arange(T) % 4 == 0)[:, None]
    cand = lambda a, b: np.where(dec, a, b)[:, :, None] * np.ones((1, 1, 2))   # noqa: E731
    z = Rec(ttl=ttl, d=d[:, None], need=np.maximum(0.0, np.abs(d) - 7.0)[:, None], dx=dx[:, None], speed=np.array(speed), decided=dec,
            act_every=np.array(4), cand_copy=cand(.5, .9), cand_distinct=cand(6.0, 3.0), cand_sd=cand(.4, .2))
    m = am.moves(z, 4)
    assert np.isclose(m["wall"], 2 / 11) and m["size"] == 1.0
    assert m["same"] == .5                                    # the decision of tick 4 repeats that of tick 0; that of tick 8 turns
    assert m["toward_far"] == 1.0 and np.isnan(m["toward_near"])   # ticks 0 and 4, 23 and 14 px off; tick 8 is 4 ticks before the landing
    assert np.allclose((m["cand_copy"], m["cand_copy_held"], m["cand_distinct"], m["cand_distinct_held"]), (.5, .9, 6.0, 3.0))
    every = am.moves(Rec({k: v for k, v in z.items() if k not in ("decided", "act_every")}), 4)
    assert every["same"] == .5 and np.isclose(every["same1"], 10 / 11) and "cand_copy_held" in every and np.isnan(every["cand_copy_held"])


def test_variety_counts_the_copies_of_the_last_move():
    sd, mean, distinct, copy = am.variety(np.array([[.5, .5, .5, -.5], [0, .1, .2, .3]]), np.array([.5, np.nan]))
    assert np.isclose(sd[0], np.std([.5, .5, .5, -.5])) and np.isclose(mean[1], .15)
    assert list(distinct) == [2, 4] and copy[0] == .75 and np.isnan(copy[1])

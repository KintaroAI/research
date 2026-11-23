"""Task 00111 (vm-00103): the Breakout court. Bounces, hits and misses, the body's events (a hit a meal, a miss a poison
with hunger_cut), the paddle at the walls, the pause after a miss, determinism by seed, and the default agent's config
unchanged in U_breakout.json."""
import json
import os
import sys

import numpy as np
import pytest

DEV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, DEV)

from world import make_config, make_world  # noqa: E402
from world.breakout_world import BreakoutConfig, BreakoutWorld  # noqa: E402


def world(n=1, seed=0, **kw):
    return make_world(n, make_config(dict({"kind": "breakout"}, **kw)), seed=seed)


def act(dx):
    a = np.zeros((len(np.atleast_1d(dx)), 3))
    a[:, 0] = dx
    return a


def place(w, b, x, y, vx, vy):
    w.ball[b] = (x, y)
    w.vel[b] = (vx, vy)
    w.live[b] = True


def track(w):
    """The tracker walker's move: toward the ball's x."""
    c = w.config
    return act(np.clip(((w.ball[:, 0] + c.ball / 2) - (w.paddle + c.paddle_w / 2)) / c.paddle_speed, -1, 1))


def test_factory_and_frame():
    w = world(2)
    assert isinstance(w, BreakoutWorld)
    o = w.observe()
    assert o["image"].shape == (2, 64, 64, 3) and o["image"].dtype == np.uint8
    assert (o["image"][..., 0] == o["image"][..., 1]).all() and (o["image"][..., 0] == o["image"][..., 2]).all()   # grey
    assert o["drives"].shape == (2, 2)
    img = o["image"][0, ..., 0]
    c = w.config
    assert (img[:, :c.wall] == c.wall_grey).all() and (img[:c.wall, :] == c.wall_grey).all()
    assert (img[c.paddle_y, :] == c.paddle_grey).sum() == c.paddle_w
    assert (img == c.ball_grey).sum() == c.ball ** 2


def test_validate():
    with pytest.raises(ValueError):
        BreakoutConfig(paddle_w=1).validate()
    with pytest.raises(ValueError):
        BreakoutConfig(ball_speed=5.0).validate()   # could pass through the paddle in one tick
    with pytest.raises(ValueError):
        BreakoutConfig(kind="circle").validate()


def test_walls_and_top_bounce():
    w = world()
    c = w.config
    place(w, 0, c.wall + 0.5, 30.0, -1.0, 0.0)
    w.step(act(0))
    assert w.vel[0, 0] > 0 and w.ball[0, 0] >= c.wall
    place(w, 0, c.size - c.wall - c.ball - 0.5, 30.0, 1.0, 0.0)
    w.step(act(0))
    assert w.vel[0, 0] < 0 and w.ball[0, 0] <= c.size - c.wall - c.ball
    place(w, 0, 30.0, c.wall + 0.5, 0.0, -1.0)
    w.step(act(0))
    assert w.vel[0, 1] > 0 and w.ball[0, 1] >= c.wall


def test_hit_is_a_meal_and_aims():
    w = world()
    c = w.config
    w.hunger[:] = 0.7
    w.paddle[:] = 20.0
    centre = 20.0 + c.paddle_w / 2 - c.ball / 2
    place(w, 0, centre, c.paddle_y - c.ball - 0.5, 0.0, 1.0)
    _, _, _, info = w.step(act(0))
    assert info["success"][0] and not info["gt_hazard_click"][0]
    assert w.hunger[0] == 0.0 and info["hunger_cut"][0] == 0.0
    assert w.vel[0, 1] < 0 and abs(w.vel[0, 0]) < 1e-9                 # straight up off the centre
    assert np.isclose(np.hypot(*w.vel[0]), c.ball_speed)
    place(w, 0, 20.0 + c.paddle_w - 1.0, c.paddle_y - c.ball - 0.5, 0.0, 1.0)   # near the right end: leaves to the right
    w.step(act(0))
    assert w.vel[0, 0] > 0 and w.vel[0, 1] < 0


def test_miss_is_a_poison_with_the_cut_then_a_pause():
    w = world(pause=5)
    c = w.config
    w.hunger[:] = 0.8
    w.paddle[:] = c.wall
    place(w, 0, 50.0, c.size - 1.0, 0.0, 1.0)
    _, _, _, info = w.step(act(0))
    assert info["gt_hazard_click"][0] and not info["success"][0]
    assert w.hunger[0] == 1.0
    assert np.isclose(info["hunger_cut"][0], 0.8 + c.h_rate + c.miss_add - 1.0)
    assert not w.live[0] and (info["terminal_image"][0] == c.ball_grey).sum() == 0
    for _ in range(c.pause):
        assert not w.live[0]
        w.step(act(0))
    assert w.live[0] and w.vel[0, 1] > 0 and w.ball[0, 1] < c.wall + 2   # served from the top, falling


def test_paddle_moves_and_stops_at_the_walls():
    w = world()
    c = w.config
    x0 = w.paddle[0]
    w.step(act(1.0))
    assert np.isclose(w.paddle[0], x0 + c.paddle_speed)
    w.step(act(0.0))
    assert np.isclose(w.paddle[0], x0 + c.paddle_speed)
    for _ in range(40):
        w.step(act(1.0))
    assert w.paddle[0] == c.size - c.wall - c.paddle_w
    for _ in range(40):
        w.step(act(-1.0))
    assert w.paddle[0] == c.wall
    a = act(1.0)
    a[:, 1:] = [1.0, 1.0]                                             # dy and click do nothing
    w2 = world(seed=0)
    w3 = world(seed=0)
    for _ in range(30):
        w2.step(a)
        w3.step(act(1.0))
    assert np.array_equal(w2.observe()["image"], w3.observe()["image"])


def test_hunger_drift_and_restlessness():
    w = world()
    c = w.config
    w.hunger[:] = 0.2
    w.ball[:] = (30.0, 20.0)
    w.vel[:] = (0.0, 0.1)
    r0 = w.restlessness[0]
    w.step(act(0))
    assert np.isclose(w.hunger[0], 0.2 + c.h_rate)
    assert np.isclose(w.restlessness[0], min(1.0, r0 + c.b_rate))
    r1 = w.restlessness[0]
    w.step(act(1.0))
    assert np.isclose(w.restlessness[0], np.clip(r1 + c.b_rate - c.b_relief * (1 - c.paddle_speed / c.move_budget_px), 0, 1))


def test_determinism_by_seed():
    def run(seed):
        w = world(3, seed=seed)
        rng = np.random.default_rng(7)
        out = []
        for _ in range(400):
            o, _, _, i = w.step(act(rng.uniform(-1, 1, 3)))
            out.append((o["image"].copy(), i["success"].copy(), i["gt_hazard_click"].copy(), o["drives"].copy()))
        return out
    a, b, c = run(5), run(5), run(6)
    assert all(np.array_equal(x, y) for p, q in zip(a, b) for x, y in zip(p, q))
    assert not all(np.array_equal(p[0], q[0]) for p, q in zip(a, c))


def test_tracker_never_misses_and_a_still_paddle_does():
    w = world(4, seed=2)
    for _ in range(5000):
        w.step(track(w))
    assert w.misses.sum() == 0 and w.hits.min() > 30
    s = world(4, seed=2)
    for _ in range(5000):
        s.step(act(np.zeros(4)))
    assert s.misses.sum() > s.hits.sum()


def test_u_breakout_is_the_default_agent():
    a = json.load(open(os.path.join(DEV, "configs", "U_agent.json")))
    b = json.load(open(os.path.join(DEV, "configs", "U_breakout.json")))
    for k in ("model", "online", "ppo"):
        assert a[k] == b[k], k
    assert b["env"]["kind"] == "breakout"
    make_config(b["env"])

"""Task 00111 (vm-00103): the Breakout court. Bounces, hits and misses, the body's events (a hit a meal, a miss a poison
with hunger_cut), the paddle at the walls, the pause after a miss, determinism by seed, and the default agent's config
unchanged in U_breakout.json. Task 00117 (vm-00104): stage 2, a paddle hit without food, bricks of three looks and foods,
the refill in which no slot keeps its type, the serve below the bricks, U_breakout2.json."""
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


def track(w, off=0.0):
    """The tracker walker's move: toward the ball's x (plus off px: where on the paddle it aims to meet the ball)."""
    c = w.config
    return act(np.clip(((w.ball[:, 0] + c.ball / 2) - (w.paddle + c.paddle_w / 2 + off)) / c.paddle_speed, -1, 1))


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
    assert w.hunger[0] == 0.0 and np.isclose(info["hunger_cut"][0], 0.7 + c.h_rate - 1.0)   # signed (task 00117): the floor's cut
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


# ---------------------------------------------------------------------- stage 2 (task 00117)
def world2(n=1, seed=0, **kw):
    env = json.load(open(os.path.join(DEV, "configs", "U_breakout2.json")))["env"]
    return make_world(n, make_config(dict(env, **kw)), seed=seed)


def brick_xy(w, k):
    return float(w.bx[k]), float(w.by[k])


def test_stage2_layout_and_looks():
    w = world2(3, seed=4)
    c = w.config
    assert c.ball_speed == 2.25 and c.paddle_relief == 0.0 and w.bw == 9
    for e in range(3):
        assert np.bincount(w.layout[e], minlength=3).tolist() == [8, 8, 8]
    img = w.observe()["image"][0, ..., 0]
    for k in range(24):   # each brick drawn with its look's tile
        x, y = int(w.bx[k]), int(w.by[k])
        assert (img[y:y + c.brick_h, x:x + w.bw] == w.tiles[(w.layout[0, k] + c.brick_shift) % 3]).all()
    looks = [set(np.unique(t).tolist()) for t in w.tiles]
    assert looks[0] != looks[1] and looks[1] != looks[2] and looks[0] != looks[2]
    s = world2(3, seed=4, brick_shift=1)   # the shift changes the looks, not the layout
    assert (s.layout == w.layout).all() and not (s.observe()["image"] == w.observe()["image"]).all()
    assert w.ball[0, 1] > c.brick_top + 4 * (c.brick_h + 1)   # the serve comes from below the bricks


def test_brick_hit_from_below_feeds_by_rank_and_bounces():
    w = world2()
    c = w.config
    k = 20   # bottom row
    x, y = brick_xy(w, k)
    rank = int(w.layout[0, k])
    w.hunger[:] = 0.6
    place(w, 0, x + 3, y + c.brick_h + 0.5, 0.0, -c.ball_speed)
    _, _, _, info = w.step(act(0))
    assert not w.alive[0, k] and w.vel[0, 1] > 0
    assert info["success"][0] and info["gt_brick_hit"][0] == rank
    assert np.isclose(w.hunger[0], 0.6 + c.h_rate - c.brick_food[rank])
    assert info["hunger_cut"][0] == 0.0 and info["gt_bricks"][0, k] == -1
    st = w.stats()
    assert st["bo_bricks"] == 1 and st["bo_brick_t%d" % rank] == 1 and np.isclose(st["bo_food"], c.brick_food[rank])
    assert np.isclose(st["bo_pref_hit"], rank) and np.isclose(st["bo_pref_stand"], 1.0)


def test_brick_at_the_floor_reports_the_cut_and_side_hits_bounce_sideways():
    w = world2()
    c = w.config
    k = 21
    x, y = brick_xy(w, k)
    rank = int(w.layout[0, k])
    w.hunger[:] = 0.0
    w.alive[0, k - 1] = False                            # its left neighbour gone: room for the ball beside it
    place(w, 0, x - c.ball - 0.5, y + 0.5, 1.0, 0.0)   # from the left, level with the brick
    _, _, _, info = w.step(act(0))
    assert not w.alive[0, k] and w.vel[0, 0] < 0 and w.vel[0, 1] == 0.0
    assert w.hunger[0] == 0.0 and np.isclose(info["hunger_cut"][0], c.h_rate - c.brick_food[rank])


def test_paddle_hit_is_no_meal_in_stage2():
    w = world2()
    c = w.config
    w.hunger[:] = 0.5
    w.paddle[:] = 20.0
    place(w, 0, 20.0 + c.paddle_w / 2 - 1, c.paddle_y - c.ball - 0.5, 0.0, 1.0)
    _, _, _, info = w.step(act(0))
    assert info["gt_paddle_hit"][0] and not info["success"][0] and w.vel[0, 1] < 0
    assert np.isclose(w.hunger[0], 0.5 + c.h_rate) and w.stats()["bo_hits"] == 1


def test_refill_keeps_no_slot_type_and_the_counts():
    for seed in range(20):
        w = world2(seed=seed)
        c = w.config
        old = w.layout[0].copy()
        w.alive[0, :] = False
        w.alive[0, 20] = True
        x, y = brick_xy(w, 20)
        place(w, 0, x + 3, y + c.brick_h + 0.5, 0.0, -c.ball_speed)
        w.step(act(0))
        assert w.alive[0].all() and w.clears[0] == 1
        assert np.bincount(w.layout[0], minlength=3).tolist() == [8, 8, 8]
        assert (w.layout[0] != old).all()


def test_stage2_tracker_never_misses_and_clears():
    w = world2(4, seed=3)
    rng = np.random.default_rng(1)
    off = rng.uniform(-4, 4, 4)
    for _ in range(8000):
        info = w.step(track(w, off))[3]
        hit = info["gt_paddle_hit"]
        off[hit] = rng.uniform(-4, 4, hit.sum())   # a new aim after each paddle hit, as breakout_walkers' tracker
    st = w.stats()
    assert st["bo_misses"] == 0 and st["bo_bricks"] > 50
    assert np.isclose(st["bo_food"], sum(st["bo_brick_t%d" % r] * f for r, f in enumerate(w.config.brick_food)))


def test_stage2_determinism():
    a, b = world2(2, seed=9), world2(2, seed=9)
    rng = np.random.default_rng(0)
    for _ in range(600):
        m = act(rng.uniform(-1, 1, 2))
        oa, ob = a.step(m), b.step(m)
        assert (oa[0]["image"] == ob[0]["image"]).all() and (oa[3]["hunger_cut"] == ob[3]["hunger_cut"]).all()


def test_u_breakout2_is_the_default_agent():
    a = json.load(open(os.path.join(DEV, "configs", "U_agent.json")))
    b = json.load(open(os.path.join(DEV, "configs", "U_breakout2.json")))
    for k in ("model", "online", "ppo"):
        assert a[k] == b[k], k
    make_config(b["env"])

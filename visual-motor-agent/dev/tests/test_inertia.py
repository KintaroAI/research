"""The inertia world (task 00106, env.inertia = k): the body keeps its velocity and the move pushes it, vel += move *
max_step_px / k, at most max_step_px per axis; with no push it keeps moving. inertia = 0 is the old world.

Ramp and coast, reversal in 2k ticks, the per-axis cap, drag, a wall stopping the blocked axis, switching inertia on
mid-life (the body keeps moving as it last moved), state round trips, and inertia = 0 moving as before.
"""
import dataclasses
import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from world import make_config, make_world  # noqa: E402

DEV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
VIEW = json.load(open(os.path.join(DEV, "configs", "U_view.json")))["env"]


def world(n=2, **kw):
    env = make_world(n, make_config(dict(VIEW, **kw), auto_reset=True), seed=3)
    env.observe()
    return env


def push(env, dx, dy):
    a = np.zeros((env.num_envs, 3), np.float32)
    a[:, 0], a[:, 1] = dx, dy
    return env.step(a)[3]


def test_ramp_coast_and_reverse():
    env = world(inertia=4)
    m = env.config.max_step_px
    v = [push(env, 1, 0)["gt_vel"][0, 0] for _ in range(6)]
    np.testing.assert_allclose(v, [m / 4, m / 2, 3 * m / 4, m, m, m])
    x0 = env.cursor[0, 0]
    for _ in range(5):                                   # no push: it keeps moving at full speed
        np.testing.assert_allclose(push(env, 0, 0)["gt_vel"][0], [m, 0])
    np.testing.assert_allclose((env.cursor[0, 0] - x0) % env.config.size, 5 * m)
    v = [push(env, -1, 0)["gt_vel"][0, 0] for _ in range(8)]   # a reversal takes 2k ticks
    np.testing.assert_allclose(v, [m * (1 - (i + 1) / 4) for i in range(8)])


def test_cap_per_axis_and_drag():
    env = world(inertia=2)
    m = env.config.max_step_px
    for _ in range(4):
        inf = push(env, 1, -1)
    np.testing.assert_allclose(inf["gt_vel"][0], [m, -m])         # each axis capped at max_step_px
    env.reconfigure(inertia_drag=0.5)
    np.testing.assert_allclose(push(env, 0, 0)["gt_vel"][0], [m / 2, -m / 2])   # drag .5
    np.testing.assert_allclose(push(env, 0, 0)["gt_vel"][0], [m / 4, -m / 4])


def test_wall_stops_the_blocked_axis():
    cfg = make_config(dict(VIEW, egocentric=False, view=0, wrap=False, foods=1, no_overlap=False, inertia=4), auto_reset=True)
    env = make_world(1, cfg, seed=3)
    env.observe()
    env.set_state(dict(cursor=np.array([[cfg.size - 2.0, 10.0]])))
    for _ in range(4):
        push(env, 1, 1)
    assert env.cursor[0, 0] == pytest.approx(cfg.size - 1e-4, abs=1e-3)
    inf = push(env, 0, 0)
    assert inf["gt_vel"][0, 0] == 0.0 and inf["gt_vel"][0, 1] > 0   # the wall took the x velocity; y keeps going


def test_switching_on_mid_life_keeps_the_motion():
    env = world(inertia=0)
    m = env.config.max_step_px
    for _ in range(3):
        push(env, 1, 0.5)
    env.reconfigure(inertia=4)                            # as online.env_sched does
    np.testing.assert_allclose(push(env, 0, 0)["gt_vel"][0], [m, m / 2])   # it keeps moving as it last moved
    env.reconfigure(inertia=0)
    assert "gt_vel" not in push(env, 0, 0)


def test_state_round_trip():
    env = world(inertia=4)
    for _ in range(3):
        push(env, 1, -1)
    st = env.get_state()
    assert "vel" in st
    a = push(env, 0.3, 0.2)
    env.set_state(st)
    b = push(env, 0.3, 0.2)
    np.testing.assert_array_equal(a["gt_vel"], b["gt_vel"])
    np.testing.assert_array_equal(a["gt_cursor"], b["gt_cursor"])


def test_no_inertia_moves_as_before():
    env, ref = world(inertia=0), world()
    rng = np.random.default_rng(0)
    m = env.config.max_step_px
    for _ in range(20):
        a = rng.uniform(-1, 1, (2, 3)).astype(np.float32)
        x0 = env.cursor.copy()
        o1, _, _, i1 = env.step(a)
        o2, _, _, i2 = ref.step(a)
        np.testing.assert_array_equal(o1["image"], o2["image"])
        np.testing.assert_allclose(np.mod(x0 + np.clip(a[:, :2], -1, 1) * m - i1["gt_cursor"] + 64, 128) - 64, 0, atol=1e-9)
    assert "vel" not in env.get_state()
    assert dataclasses.asdict(env.config)["inertia"] == 0.0


@pytest.mark.parametrize("k", [0, 4])
def test_scoring_helpers_follow_the_step(k):
    """push_of, eval_hunger.would_touch and in_reach (the frozen reach test, the grab baseline) against stepping copies."""
    from eval_hunger import in_reach, would_touch

    n = 256
    env = world(n=n, inertia=k, foods=1, inertia_drag=0.1 if k else 0.0)
    rng = np.random.default_rng(1)
    for _ in range(5):                                     # some velocity
        push(env, *rng.uniform(-1, 1, 2))
    near = env.target + rng.uniform(-12, 12, (n, 2))       # many bodies near their food
    env.set_state(dict(cursor=np.mod(near, env.config.size)))
    st = env.get_state()
    cand = rng.uniform(-1.2, 1.2, (n, 6, 3)).astype(np.float32)
    food, _ = would_touch(env, cand)
    land = np.mod(env.cursor[:, None] + env.push_of(cand), env.config.size)
    for j in range(cand.shape[1]):
        env.set_state(st)
        inf = env.step(cand[:, j])[3]
        np.testing.assert_allclose(inf["gt_cursor"], land[:, j], atol=1e-9)
        np.testing.assert_array_equal(food[:, j], inf["success"])
    env.set_state(st)
    ok, mv = in_reach(env)
    a = np.zeros((n, 3), np.float32)
    a[:, :2] = mv
    inf = env.step(a)[3]
    assert ok.sum() > 20 and inf["success"][ok].all()
    grid = np.stack(np.meshgrid(np.linspace(-1, 1, 9), np.linspace(-1, 1, 9)), -1).reshape(1, -1, 2)
    env.set_state(st)
    food, _ = would_touch(env, np.concatenate([np.repeat(grid, n, 0), np.zeros((n, grid.shape[1], 1))], -1))
    r0 = env.radius.copy()
    env.radius = r0 / 0.98 * (1 + 1e-9)                    # in_reach keeps a 2% margin; at the full radius it covers every touch
    ok_full, _ = in_reach(env)
    env.radius = r0
    assert food.any(1).sum() > 20 and not food[~ok_full].any()   # out of reach: no move touches it


@pytest.mark.parametrize("bad", [dict(inertia=-1), dict(inertia=2, inertia_drag=1.0), dict(inertia_drag=-0.1)])
def test_invalid(bad):
    with pytest.raises(ValueError):
        make_config(dict(VIEW, **bad), auto_reset=True).validate()

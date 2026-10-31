"""No two circles overlap (task 00060, env.no_overlap): at any tick of a random walk, on the torus; food stays put;
bounces keep the speeds; new circles are placed clear; food and poison never on one tick; off, the world repeats.
"""
import dataclasses
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from world import CircleWorld, EnvConfig  # noqa: E402
from world.circle_world import wrap_delta  # noqa: E402

BASE = EnvConfig(size=64, wrap=True, egocentric=True, touch=True, respawn=True, end_on_success=False, max_steps=10 ** 9,
                 hazards=3, hazard_click_add=0.5, drives=True, gray=True, h_rate=0.001, bonuses=False, radius_min=8.0,
                 radius_max=11.0, hazard_pattern="checker", hazard_speed=2.0, auto_reset=True)
FREE = dataclasses.replace(BASE, no_overlap=True)


def gaps(env):
    """The smallest clearance [N] over all pairs of circles."""
    size = float(env.config.size)
    pos = np.concatenate((env.hz_pos, env.target[:, None]), 1)
    rad = np.concatenate((env.hz_rad, env.radius[:, None]), 1)
    out = np.full(env.num_envs, np.inf)
    for a in range(pos.shape[1]):
        for b in range(a + 1, pos.shape[1]):
            d = np.hypot(wrap_delta(pos[:, a, 0], pos[:, b, 0], size), wrap_delta(pos[:, a, 1], pos[:, b, 1], size))
            out = np.minimum(out, d - rad[:, a] - rad[:, b])
    return out


def walk(env, ticks, seed=0):
    rng = np.random.default_rng(seed)
    n = env.num_envs
    for _ in range(ticks):
        a = np.concatenate([rng.uniform(-1, 1, (n, 2)), np.zeros((n, 1))], 1).astype(np.float32)
        yield env.step(a)


def test_never_overlap_and_never_food_and_poison_together():
    env = CircleWorld(8, FREE, seed=1)
    assert (gaps(env) >= -1e-6).all()
    touched = 0
    for _, _, _, inf in walk(env, 4000):
        assert (gaps(env) >= -1e-6).all()
        assert not (inf["success"] & inf["gt_hazard_click"]).any()
        touched += int(inf["gt_hazard_click"].sum()) + int(inf["success"].sum())
    assert touched > 100


def test_overlap_happens_without_the_rule():
    env = CircleWorld(8, BASE, seed=1)
    assert any((gaps(env) < -1.0).any() for _ in walk(env, 500))


def test_bounces_keep_speed_and_food_stays():
    still = CircleWorld(4, dataclasses.replace(FREE, touch=True), seed=3)
    for _ in range(300):
        t0, hit = still.target.copy(), None
        v0 = np.sqrt((still.hz_vel ** 2).sum(-1)).copy()
        still.cursor[:] = 0.0                                    # keep the body away: nothing respawns by chance
        _, _, _, inf = still.step(np.zeros((4, 3), np.float32))
        ok = ~inf["success"]
        np.testing.assert_array_equal(still.target[ok], t0[ok])
        hit = inf["gt_hazard_click"]
        np.testing.assert_allclose(((still.hz_vel ** 2).sum((1, 2)))[~hit], (v0 ** 2).sum(1)[~hit], rtol=1e-9)


def test_reflects_off_food():
    env = CircleWorld(1, dataclasses.replace(FREE, hazards=1), seed=4)
    env.cursor[:] = (5.0, 5.0)
    env.target[0], env.radius[0] = (40.0, 32.0), 8.0
    env.hz_pos[0, 0], env.hz_rad[0, 0], env.hz_vel[0, 0] = (30.5, 32.0), 6.0, (2.0, 0.0)
    env.step(np.zeros((1, 3), np.float32))
    np.testing.assert_allclose(env.hz_vel[0, 0], (-2.0, 0.0), atol=1e-9)
    assert gaps(env)[0] >= -1e-6 and env.target[0, 0] == 40.0


def test_two_hazards_swap():
    env = CircleWorld(1, dataclasses.replace(FREE, hazards=2), seed=5)
    env.cursor[:] = (2.0, 2.0)
    env.target[0], env.radius[0] = (10.0, 50.0), 8.0
    env.hz_pos[0], env.hz_rad[0] = ((29.0, 32.0), (40.0, 32.0)), (6.0, 6.0)
    env.hz_vel[0] = ((1.0, 0.0), (-1.0, 0.0))
    env.step(np.zeros((1, 3), np.float32))
    np.testing.assert_allclose(env.hz_vel[0], ((-1.0, 0.0), (1.0, 0.0)), atol=1e-9)


def test_off_repeats_the_old_world():
    a, b = CircleWorld(4, BASE, seed=7), CircleWorld(4, dataclasses.replace(BASE), seed=7)
    for x, y in zip(walk(a, 200), walk(b, 200)):
        np.testing.assert_array_equal(x[0]["image"], y[0]["image"])
    assert not BASE.no_overlap


@pytest.mark.parametrize("bad", [dict(wrap=False, egocentric=False), dict(hazard_shape="square")], ids=["walls", "square"])
def test_invalid(bad):
    with pytest.raises(ValueError):
        CircleWorld(1, dataclasses.replace(FREE, **bad), seed=1)

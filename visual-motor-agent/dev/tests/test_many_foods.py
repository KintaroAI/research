"""Several foods (task 00083, env.foods): every food is drawn, slot 0 is the nearest, touching any food feeds and replaces that food only, no circles overlap,
the state repeats, and foods above 1 refuses a world that cannot hold it.
"""
import dataclasses
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from world import CircleWorld, EnvConfig  # noqa: E402
from world.circle_world import wrap_delta  # noqa: E402

BIG = EnvConfig(size=128, wrap=True, egocentric=True, touch=True, respawn=True, end_on_success=False, max_steps=10 ** 9, hazards=8, foods=4, distractors=0,
                hazard_click_add=0.5, drives=True, gray=True, h_rate=0.001, bonuses=False, radius_min=8.0, radius_max=11.0, hazard_pattern="checker",
                hazard_checker_cell=2, auto_reset=True, no_overlap=True)


def foods(env):
    return np.concatenate((env.target[:, None], env.fx_pos), 1), np.concatenate((env.radius[:, None], env.fx_rad), 1)


def gaps_to(env, xy):
    size = float(env.config.size)
    pos, rad = foods(env)
    return np.hypot(wrap_delta(pos[..., 0], xy[:, None, 0], size), wrap_delta(pos[..., 1], xy[:, None, 1], size)) - rad


def clearance(env):
    size = float(env.config.size)
    fp, fr = foods(env)
    pos, rad = np.concatenate((env.hz_pos, fp), 1), np.concatenate((env.hz_rad, fr), 1)
    d = np.hypot(wrap_delta(pos[:, None, :, 0], pos[:, :, None, 0], size), wrap_delta(pos[:, None, :, 1], pos[:, :, None, 1], size))
    out = d - rad[:, None, :] - rad[:, :, None]
    n = pos.shape[1]
    out[:, np.arange(n), np.arange(n)] = np.inf
    return out.min((1, 2))


def walk(env, ticks, seed=0):
    rng = np.random.default_rng(seed)
    n = env.num_envs
    for _ in range(ticks):
        yield env.step(np.concatenate([rng.uniform(-1, 1, (n, 2)), np.zeros((n, 1))], 1).astype(np.float32))


def test_one_food_has_no_others():
    env = CircleWorld(2, dataclasses.replace(BIG, foods=1, size=64, hazards=2), seed=1)
    assert env.fx_pos.shape == (2, 0, 2) and env.fx_rad.shape == (2, 0) and not env._fx_rngs


def test_every_food_is_drawn():
    env = CircleWorld(1, dataclasses.replace(BIG, hazards=0, egocentric=False), seed=3)
    img = env.observe()["image"][0, ..., 0]
    bg = np.array(env.config.bg_color).max()
    pos, rad = foods(env)
    yy, xx = np.mgrid[:128, :128] + 0.5
    drawn = 0
    for (x, y), r in zip(pos[0], rad[0]):
        d = np.hypot(wrap_delta(xx, x, 128.0), wrap_delta(yy, y, 128.0))
        ring = d < r - 3
        ring[np.abs(xx - env.cursor[0, 0]) < 6] = False                # the cursor's sprite
        drawn += ring.sum() and np.mean(img[ring] != bg) > 0.9
    assert drawn == 4


def test_slot_0_is_the_nearest():
    env = CircleWorld(8, BIG, seed=4)
    assert np.all(gaps_to(env, env.cursor).argmin(1) == 0)
    for _ in walk(env, 300):
        assert np.all(gaps_to(env, env.cursor).argmin(1) == 0)


def test_touching_any_food_feeds_and_replaces_it_only():
    env = CircleWorld(1, BIG, seed=5)
    env.set_state(dict(hunger=np.array([0.5])))
    far = env.fx_pos[0, 2].copy()                                        # not the nearest
    left = [tuple(p) for k, p in enumerate(env.fx_pos[0]) if k != 2] + [tuple(env.target[0])]
    env.set_state(dict(cursor=(far - np.array([4.0, 0.0]))[None] % 128.0))
    obs, _, _, inf = env.step(np.array([[1.0, 0.0, 0.0]], np.float32))
    assert inf["success"][0] and env.hunger[0] < 0.01
    pos = [tuple(p) for p in foods(env)[0][0]]
    assert sum(p == tuple(far) for p in pos) == 0
    assert sum(p in pos for p in left) == 3


def test_many_ticks_no_overlap_and_food_counts():
    env = CircleWorld(4, BIG, seed=6)
    assert clearance(env).min() > 0
    eaten = 0
    for _, _, _, inf in walk(env, 1500):
        eaten += int(inf["success"].sum())
        assert clearance(env).min() > -1e-6
    assert eaten > 10


def test_the_state_repeats():
    a = CircleWorld(2, BIG, seed=7)
    list(walk(a, 50))
    st = a.get_state()
    x = [o["image"].copy() for o, *_ in walk(a, 50, seed=9)]
    b = CircleWorld(2, BIG, seed=99)
    b.set_state(st)
    y = [o["image"].copy() for o, *_ in walk(b, 50, seed=9)]
    assert all(np.array_equal(p, q) for p, q in zip(x, y))


@pytest.mark.parametrize("change", [dict(wrap=False, no_overlap=False), dict(respawn=False, touch=False), dict(target_speed=1.0)])
def test_foods_above_1_refuses_a_world_that_cannot_hold_them(change):
    with pytest.raises(ValueError):
        dataclasses.replace(BIG, **change).validate()

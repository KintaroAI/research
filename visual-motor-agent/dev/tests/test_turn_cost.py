"""Turning costs hunger (task 00061, env.turn_cost): the formula, a straight run free, a reversal, the reset, the
saved state, the world unchanged at 0, invalid values.
"""
import dataclasses
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from world import CircleWorld, EnvConfig  # noqa: E402

BASE = EnvConfig(size=64, wrap=True, drives=True, bonuses=False, respawn=True, end_on_success=False, max_steps=10 ** 9,
                 h_rate=0.0, hunger_cost=0.0, auto_reset=True, h_init_min=0.5, h_init_max=0.5, radius_min=2.0,
                 radius_max=2.0)
C = 0.04


def step(env, dx, dy):
    a = np.full((env.num_envs, 3), 0.0, np.float32)
    a[:, 0], a[:, 1] = dx, dy
    env.target[:] = env.cursor + 20.0                                        # keep food away
    h0 = env.hunger.copy()
    env.step(a)
    return env.hunger - h0


def world(**kw):
    return CircleWorld(2, dataclasses.replace(BASE, turn_cost=C, **kw), seed=1)


def test_the_formula():
    env = world()
    np.testing.assert_allclose(step(env, 0.5, 0.0), C * 0.25 / 8)            # from rest: the first move
    np.testing.assert_allclose(step(env, 0.5, 1.0), C * 1.0 / 8)             # dy changed by 1
    np.testing.assert_allclose(step(env, -0.5, 1.0), C * 1.0 / 8)            # dx by 1
    np.testing.assert_allclose(step(env, 0.5, -1.0), C * 5.0 / 8)


def test_a_full_reversal_costs_the_whole_cost():
    env = world()
    step(env, 1.0, 1.0)
    np.testing.assert_allclose(step(env, -1.0, -1.0), C)


def test_a_steady_move_is_free():
    env = world()
    step(env, 0.7, -0.3)
    for _ in range(5):
        np.testing.assert_allclose(step(env, 0.7, -0.3), 0.0, atol=1e-12)


def test_clipped_moves_count_as_made():
    env = world()
    step(env, 1.0, 0.0)
    np.testing.assert_allclose(step(env, 5.0, 0.0), 0.0, atol=1e-12)


def test_reset_clears_the_previous_move():
    env = world()
    step(env, 1.0, 1.0)
    env.reset()
    np.testing.assert_allclose(env.last_move, 0.0)


def test_state_keeps_the_previous_move():
    env = world()
    step(env, 1.0, -1.0)
    s = env.get_state()
    a = step(env, -1.0, 1.0)
    env.set_state(s)
    np.testing.assert_allclose(env.last_move, [[1.0, -1.0]] * 2)
    np.testing.assert_allclose(step(env, -1.0, 1.0), a)


def test_zero_changes_nothing():
    a, b = CircleWorld(2, BASE, seed=3), CircleWorld(2, dataclasses.replace(BASE, turn_cost=0.0), seed=3)
    rng = np.random.default_rng(0)
    for _ in range(50):
        m = rng.uniform(-1, 1, (2, 3)).astype(np.float32)
        np.testing.assert_array_equal(a.step(m)[0]["image"], b.step(m)[0]["image"])
    np.testing.assert_array_equal(a.hunger, b.hunger)


def test_invalid():
    with pytest.raises(ValueError):
        CircleWorld(1, dataclasses.replace(BASE, turn_cost=-0.1), seed=1)

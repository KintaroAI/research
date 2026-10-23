"""The observation carries rendered pixels and drives only (plan section 2).

The target position, cursor coordinates, distances and the environment's
counters must reach the agent only through the image, at pixel resolution.
"""

import numpy as np
import pytest

from world import CircleWorld, EnvConfig

CONFIGS = [EnvConfig(), EnvConfig(bonuses=True, drives=True, hunger_cost=0.01)]


def circle_mask(target, radius, size=64):
    """Reference set of pixels covered by the circle (spec formula)."""
    c = np.arange(size) + 0.5
    dx = c[None, :] - target[0]
    dy = c[:, None] - target[1]
    return dx * dx + dy * dy <= radius * radius


@pytest.mark.parametrize("cfg", CONFIGS)
def test_observation_keys_are_exactly_image_and_drives(cfg):
    env = CircleWorld(4, cfg, seed=0)
    assert set(env.reset()) == {"image", "drives"}
    assert set(env.observe()) == {"image", "drives"}
    rng = np.random.default_rng(0)
    for _ in range(300):
        obs, _, _, info = env.step(rng.uniform(-1, 1, (4, 3)))
        assert set(obs) == {"image", "drives"}


@pytest.mark.parametrize("cfg", CONFIGS)
def test_ground_truth_keys_only_in_info(cfg):
    env = CircleWorld(4, cfg, seed=0)
    obs, _, _, info = env.step(np.zeros((4, 3)))
    gt = [k for k in info if k.startswith("gt_")]
    assert set(gt) == {"gt_cursor", "gt_target", "gt_radius", "gt_target_alive", "gt_inside",
                       "gt_drives"}
    assert not any(k.startswith("gt_") for k in obs)
    for k in gt:
        assert all(info[k] is not v for v in obs.values())


def test_drives_stay_in_unit_interval():
    cfg = EnvConfig(bonuses=True, drives=True, hunger_cost=0.01, h_init_max=1.0, b_init_max=1.0,
                    h_rate=0.05, b_rate=0.08, max_steps=60)
    env = CircleWorld(64, cfg, seed=0)
    rng = np.random.default_rng(0)
    obs = env.reset()
    for _ in range(600):
        assert np.all((obs["drives"] >= 0) & (obs["drives"] <= 1))
        a = rng.uniform(-1, 1, (64, 3))
        a[:, :2] *= rng.random((64, 1)) < 0.5     # half the envs mostly sit still
        obs, _, _, info = env.step(a)
        assert np.all((info["terminal_drives"] >= 0) & (info["terminal_drives"] <= 1))
    assert obs["drives"].max() == 1.0              # the clamp is actually exercised


def test_image_is_a_function_of_cursor_pixel_and_circle_pixels():
    """Sub-pixel geometry and counters never change the observation."""
    rng = np.random.default_rng(0)
    cfg = EnvConfig(drives=True, bonuses=True)
    a, b = CircleWorld(1, cfg, seed=0), CircleWorld(1, cfg, seed=1)
    compared = 0
    for _ in range(400):
        cursor = rng.uniform(0, 63.9, 2)
        target = rng.uniform(16, 48, 2)
        radius = rng.uniform(4, 16)
        alive = bool(rng.random() < 0.8)
        cursor_b = np.floor(cursor) + rng.uniform(0, 0.9999, 2)        # same hotspot pixel
        target_b = target + rng.uniform(-1e-3, 1e-3, 2)                 # tiny shifts
        radius_b = radius + rng.uniform(-1e-3, 1e-3)
        if not np.array_equal(circle_mask(target, radius), circle_mask(target_b, radius_b)):
            continue                                                    # a pixel flipped: skip
        drives = {"hunger": [0.3], "restlessness": [0.6]}
        a.set_state({"cursor": [cursor], "target": [target], "radius": [radius],
                     "target_alive": [alive], "t": [3], "cum_move": [12.0], "ep_return": [0.5],
                     "clicks": [2], "path_length": [12.0], **drives})
        b.set_state({"cursor": [cursor_b], "target": [target_b], "radius": [radius_b],
                     "target_alive": [alive], "t": [150], "cum_move": [190.0], "ep_return": [-3.0],
                     "clicks": [40], "path_length": [500.0], "init_dist": [17.0], **drives})
        oa, ob = a.observe(), b.observe()
        np.testing.assert_array_equal(oa["image"], ob["image"])
        np.testing.assert_array_equal(oa["drives"], ob["drives"])
        compared += 1
    assert compared > 300


def test_reward_settings_do_not_change_pixels():
    plain = CircleWorld(8, EnvConfig(), seed=4)
    rich = CircleWorld(8, EnvConfig(bonuses=True, drives=True, hunger_cost=0.05), seed=4)
    np.testing.assert_array_equal(plain.observe()["image"], rich.observe()["image"])


def test_drives_uncorrelated_with_geometry_over_5000_resets():
    env = CircleWorld(5000, EnvConfig(drives=True), seed=123)
    drives = env.reset()["drives"].astype(np.float64)
    geometry = {"target_x": env.target[:, 0], "target_y": env.target[:, 1], "radius": env.radius,
                "cursor_x": env.cursor[:, 0], "cursor_y": env.cursor[:, 1]}
    for i, name in enumerate(("hunger", "restlessness")):
        assert drives[:, i].std() > 0.1
        for gname, g in geometry.items():
            r = np.corrcoef(drives[:, i], g)[0, 1]
            assert abs(r) < 0.1, f"corr({name}, {gname}) = {r:.3f}"

"""A world larger than the view (task 00102, env.view): the frame is the center of the full egocentric frame, view 0 is the old
frame bit for bit, only the frame changes (the state runs the same), foods_in_view agrees with the food's pixels, and the
validation refuses windows that do not fit.
"""
import dataclasses
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from world import CircleWorld, EnvConfig  # noqa: E402

BASE = EnvConfig(size=128, wrap=True, egocentric=True, touch=True, respawn=True, end_on_success=False, max_steps=10 ** 9, hazards=8, foods=2,
                 distractors=32, hazard_click_add=0.5, drives=True, gray=True, h_rate=0.001, bonuses=False, radius_min=8.0, radius_max=11.0,
                 hazard_pattern="checker", hazard_checker_cell=2, auto_reset=True, no_overlap=True)
VIEW = dataclasses.replace(BASE, view=64)


def moves(rng, n):
    return np.concatenate((rng.uniform(-1, 1, (n, 2)), np.zeros((n, 1))), 1).astype(np.float32)


def pair(cfg_full, cfg_view, n=6, seed=3):
    return CircleWorld(n, cfg_full, seed=seed), CircleWorld(n, cfg_view, seed=seed)


@pytest.mark.parametrize("view", [64, 96, 32])
def test_frame_is_the_center_of_the_full_frame(view):
    full, win = pair(BASE, dataclasses.replace(BASE, view=view))
    rng = np.random.default_rng(0)
    m = (128 - view) // 2
    for t in range(200):
        a, b = full.observe()["image"], win.observe()["image"]
        assert b.shape == (6, view, view, 3)
        np.testing.assert_array_equal(b, a[:, m:m + view, m:m + view])
        act = moves(rng, 6)
        full.step(act)
        win.step(act)
        np.testing.assert_array_equal(full.cursor, win.cursor)          # only the frame changes
        np.testing.assert_array_equal(full.hunger, win.hunger)


def test_view_zero_and_full_size_are_the_old_frame():
    old, zero = pair(BASE, dataclasses.replace(BASE, view=0))
    whole = CircleWorld(6, dataclasses.replace(BASE, view=128), seed=3)
    rng = np.random.default_rng(1)
    for t in range(50):
        f = old.observe()["image"]
        np.testing.assert_array_equal(f, zero.observe()["image"])
        np.testing.assert_array_equal(f, whole.observe()["image"])
        act = moves(rng, 6)
        for e in (old, zero, whole):
            e.step(act)


def test_noise_is_drawn_on_the_window():
    env = CircleWorld(4, dataclasses.replace(VIEW, noise=0.3), seed=2)
    img = env.observe()["image"]
    assert img.shape == (4, 64, 64, 3)
    clean = env.clean_frame()
    assert clean.shape == img.shape and (img != clean).any()


def test_foods_in_view_agrees_with_the_food_pixels():
    cfg = dataclasses.replace(VIEW, gray=False, distractors=0)          # in colour, so food and poison differ by more than pattern
    env = CircleWorld(16, cfg, seed=5)
    rng = np.random.default_rng(2)
    food = np.array(cfg.target_color, np.uint8)
    seen = hidden = 0
    for t in range(400):
        img = env.observe()["image"]
        has = (img == food).all(-1)
        has[:, 30:35, 30:35] = False                                    # under the cursor's sprite
        inv = env.foods_in_view()
        assert inv.shape == (16, 2)
        any_pixel = has.reshape(16, -1).any(1)
        ok = any_pixel == inv.any(1)
        # a food whose only pixels in the window are under the cursor's sprite is in view but not seen here
        assert ok.mean() > 0.99, (t, np.flatnonzero(~ok))
        seen += inv.any(1).sum()
        hidden += (~inv.any(1)).sum()
        env.step(moves(rng, 16))
    assert seen > 0 and hidden > 0


def test_foods_in_view_without_a_window_is_every_live_food():
    env = CircleWorld(4, BASE, seed=1)
    np.testing.assert_array_equal(env.foods_in_view(), np.concatenate((env.drawn()[:, None], np.ones((4, 1), bool)), 1))


@pytest.mark.parametrize("bad", [dict(view=63), dict(view=130), dict(view=-2), dict(view=64, egocentric=False),
                                 dict(view=62), dict(view=60, hazard_checker_cell=4)])
def test_bad_windows_are_refused(bad):
    with pytest.raises(ValueError):
        dataclasses.replace(BASE, **bad).validate()


def test_good_windows_pass():
    for v in (0, 8, 32, 64, 96, 120, 128):
        dataclasses.replace(BASE, view=v).validate()

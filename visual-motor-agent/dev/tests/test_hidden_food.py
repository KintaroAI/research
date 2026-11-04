"""Task 00092: hidden food. food_show = 0 is the old world; with food_show = v the food is drawn in exactly its first v
frames after each appearance, and everything but the food's pixels is the always-seen world's."""
import json
import os
import sys

import numpy as np
import pytest

DEV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, DEV)

from world import make_config, make_world  # noqa: E402

ENV = json.load(open(os.path.join(DEV, "configs", "U_default.json")))["env"]


def moves(n, rng, toward=None):
    a = np.concatenate([rng.uniform(-1, 1, (n, 2)), np.zeros((n, 1))], 1).astype(np.float32)
    return a


def chase(env, rng):
    """Moves straight at the food most of the time, so that it gets eaten and respawns often."""
    from eval_pieces import relative
    d = relative(env, env.target)
    a = np.clip(d / env.config.max_step_px, -1, 1)
    a = np.where(rng.random((len(a), 1)) < 0.8, a, rng.uniform(-1, 1, a.shape))
    return np.concatenate([a, np.zeros((len(a), 1))], 1).astype(np.float32)


@pytest.mark.parametrize("v", [1, 4])
def test_the_food_is_drawn_in_its_first_frames_only(v):
    seen_w = make_world(4, make_config(ENV, auto_reset=True), seed=3)
    hid_w = make_world(4, make_config(dict(ENV, food_show=v), auto_reset=True), seed=3)
    rng = np.random.default_rng(0)
    age = np.zeros(4, int)
    eaten = 0
    for t in range(600):
        a = chase(seen_w, rng)
        o1, _, _, i1 = seen_w.step(a)
        o2, _, _, i2 = hid_w.step(a)
        age = np.where(i1["success"], 0, age + 1)
        eaten += int(i1["success"].sum())
        np.testing.assert_array_equal(i2["gt_food_seen"], age < v)
        for k in ("success", "gt_hazard_click"):
            np.testing.assert_array_equal(i1[k], i2[k])
        np.testing.assert_array_equal(o1["drives"], o2["drives"])
        np.testing.assert_array_equal(seen_w.cursor, hid_w.cursor)
        same = (o1["image"] == o2["image"]).all(axis=(1, 2, 3))
        np.testing.assert_array_equal(same[age < v], True)                    # drawn: the same frame
        assert not same[age >= v].any()                                       # hidden: the food's pixels differ
    assert eaten > 20


def test_food_show_zero_is_the_old_world():
    a_w = make_world(3, make_config(ENV, auto_reset=True), seed=7)
    b_w = make_world(3, make_config(dict(ENV, food_show=0), auto_reset=True), seed=7)
    rng = np.random.default_rng(1)
    for t in range(300):
        a = chase(a_w, rng)
        oa, _, _, ia = a_w.step(a)
        ob, _, _, ib = b_w.step(a)
        np.testing.assert_array_equal(oa["image"], ob["image"])
        assert "gt_food_seen" not in ib


@pytest.mark.parametrize("bad", [-1, 1.5])
def test_bad_food_show_is_refused(bad):
    with pytest.raises(ValueError):
        make_world(1, make_config(dict(ENV, food_show=bad)))

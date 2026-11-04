"""Task 00094: online.env_sched, world settings changed mid-life. A world switched to food_show = v at some step is the
always-seen world before it and the world born with food_show = v after it, frame for frame; the schedule is refused
for keys a world cannot change, bad ticks and bad values."""
import json
import os
import sys

import numpy as np
import pytest

DEV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, DEV)

from lifetime import env_schedule  # noqa: E402
from world import make_config, make_world  # noqa: E402

ENV = json.load(open(os.path.join(DEV, "configs", "U_default.json")))["env"]


def chase(env, rng):
    from eval_pieces import relative
    a = np.clip(relative(env, env.target) / env.config.max_step_px, -1, 1)
    a = np.where(rng.random((len(a), 1)) < 0.8, a, rng.uniform(-1, 1, a.shape))
    return np.concatenate([a, np.zeros((len(a), 1))], 1).astype(np.float32)


@pytest.mark.parametrize("v", [1, 4, 16])
def test_switched_world_is_the_seen_world_then_the_hidden_one(v):
    cfg = make_config(ENV, auto_reset=True)
    sw = make_world(4, cfg, seed=5)                                              # switched at step 200
    seen = make_world(4, cfg, seed=5)
    hid = make_world(4, make_config(dict(ENV, food_show=v), auto_reset=True), seed=5)
    rng = np.random.default_rng(0)
    eaten = 0
    for t in range(1, 501):
        if t == 200:
            sw.reconfigure(food_show=v)
        a = chase(seen, rng)
        o, _, _, i = sw.step(a)
        o_s, _, _, i_s = seen.step(a)
        o_h, _, _, i_h = hid.step(a)
        eaten += int(i_s["success"].sum())
        ref = o_s if t < 200 else o_h
        np.testing.assert_array_equal(o["image"], ref["image"])
        np.testing.assert_array_equal(o["drives"], o_s["drives"])
        np.testing.assert_array_equal(i["success"], i_s["success"])
        if t >= 200:
            np.testing.assert_array_equal(i["gt_food_seen"], i_h["gt_food_seen"])
        else:
            assert "gt_food_seen" not in i
    assert sw.config.food_show == v and eaten > 20


def test_reconfigure_refuses_other_settings():
    w = make_world(1, make_config(ENV, auto_reset=True), seed=1)
    with pytest.raises(ValueError):
        w.reconfigure(foods=2)
    with pytest.raises(ValueError):
        w.reconfigure(food_show=-1)


def test_schedule_parsed_and_refused():
    cfg = make_config(ENV, auto_reset=True)
    w = make_world(1, cfg, seed=1)
    assert env_schedule(w, cfg, None, 100) == {}
    s = env_schedule(w, cfg, {"food_show": [[10, 64], [20, 4]]}, 100)
    assert s == {10: {"food_show": 64}, 20: {"food_show": 4}}
    for bad in ({"foods": [[10, 2]]},                       # a key the world cannot change
                {"food_show": [[20, 4], [10, 64]]},         # ticks not increasing
                {"food_show": [[10, 4], [10, 2]]},          # a tick twice
                {"food_show": [[0, 4]]},                    # ticks count from 1
                {"food_show": [[101, 4]]},                  # past the life
                {"food_show": [[10, 2.5]]},                 # a value the config refuses
                {"food_show": []}):
        with pytest.raises(ValueError):
            env_schedule(w, cfg, bad, 100)


def test_a_wrapper_passes_it_to_the_world_inside():
    """Delayed copies forward attributes to the world inside, which does the drawing: the change reaches it."""
    cfg = make_config(dict(ENV, delays=[0, 1, 2, 3]), auto_reset=True)
    w = make_world(1, cfg, seed=1)
    assert env_schedule(w, cfg, {"food_show": [[10, 4]]}, 100) == {10: {"food_show": 4}}
    w.reconfigure(food_show=4)
    assert w.world.config.food_show == 4


def test_refused_on_a_world_that_cannot_change():
    class Fixed:
        pass
    cfg = make_config(ENV, auto_reset=True)
    with pytest.raises(ValueError):
        env_schedule(Fixed(), cfg, {"food_show": [[10, 4]]}, 100)

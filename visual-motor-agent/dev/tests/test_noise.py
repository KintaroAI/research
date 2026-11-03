"""White noise in the world (task 00087): noise = 0 is the world as it was, the share of pixels is the level, nothing of the world is left at 1, fresh every tick, and only the frame changes."""
import json
import os

import numpy as np

from world import make_config, make_world

CFG = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "configs", "U_default.json")))["env"]


def world(q, n=4, seed=3):
    return make_world(n, make_config(dict(CFG, noise=q), auto_reset=True), seed=seed)


def moves(n, rng):
    return np.concatenate((rng.uniform(-1, 1, (n, 2)), np.zeros((n, 1))), 1).astype(np.float32)


def test_no_noise_is_the_clean_frame():
    e = world(0.0)
    assert np.array_equal(e.observe()["image"], e.clean_frame())


def test_the_share_is_the_level():
    e = world(0.5)
    seen, clean = e.observe()["image"], e.clean_frame()
    share = (seen != clean).any(3).mean()
    assert 0.47 < share < 0.53   # a pixel can draw its own brightness: 0.5 x 255 / 256 at most


def test_at_one_nothing_of_the_world_is_left():
    e = world(1.0, n=64)
    seen, clean = e.observe()["image"][..., 0].astype(float), e.clean_frame()[..., 0].astype(float)
    assert abs(np.corrcoef(seen.ravel(), clean.ravel())[0, 1]) < 0.01
    assert abs(seen.mean() - 127.5) < 2


def test_the_noise_is_fresh_each_tick():
    e = world(0.5)
    a = e.observe()["image"]
    b = e.step(np.zeros((4, 3), np.float32))[0]["image"]
    assert (a != b).mean() > 0.3


def test_the_noise_stays_gray():
    seen = world(0.5).observe()["image"]
    assert np.array_equal(seen[..., 0], seen[..., 1]) and np.array_equal(seen[..., 0], seen[..., 2])


def test_only_the_frame_changes():
    a, b = world(0.0), world(0.9)
    rng = np.random.default_rng(1)
    for _ in range(300):
        act = moves(4, rng)
        oa, _, _, ia = a.step(act)
        ob, _, _, ib = b.step(act)
        assert np.array_equal(oa["drives"], ob["drives"]) and np.array_equal(ia["success"], ib["success"])
        assert np.array_equal(a.clean_frame(), oa["image"]) and np.array_equal(b.clean_frame(), a.clean_frame())

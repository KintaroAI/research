"""The delayed-cue world (task 00065, env.cue): the phases, what a frame shows in each, the food looking like the sample, what ends a
trial, the stats; ThoughtWorld over it; a world without env.cue draws as before.
"""
import json
import os
import sys

import numpy as np
import pytest

DEV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, DEV)

from world import make_config, make_world  # noqa: E402
from world.circle_world import _checker  # noqa: E402

ENV = json.load(open(os.path.join(DEV, "configs", "U_cue.json")))["env"]
STILL = np.zeros((1, 3), np.float32)
BG = 21                                                             # the background's grey


def world(n=1, seed=0, thought=False, **cue):
    env = {k: v for k, v in ENV.items() if thought or k != "thought"}
    env["distractors"] = 0
    env["cue"] = cue
    return make_world(n, make_config(env, auto_reset=True), seed=seed)


def to_phase(w, p, act=STILL):
    for _ in range(100):
        if w.phase[0] == p:
            return
        w.step(act)
    raise AssertionError("never reached phase %d" % p)


def side_pixels(w):
    """The frame without the cursor's middle: (food's pixels, poison's pixels), by the true centres."""
    im = w.observe()["image"][0, :, :, 0].astype(int)
    shift = 32.0 - w.cursor[0]
    out = []
    for c, r in ((w.target[0], w.radius[0]), (w.hz_pos[0, 0], w.hz_rad[0, 0])):
        v = np.mod(c + shift, 64.0)
        yy, xx = np.mgrid[:64, :64]
        m = np.hypot(xx + 0.5 - v[0], yy + 0.5 - v[1]) < r - 1
        out.append(im[m])
    return out


def test_food_and_poison_have_one_grey():
    w = world()
    to_phase(w, 2)
    f, p = side_pixels(w)
    assert set(np.unique(np.concatenate([f, p]))) <= {BG, int(np.max(f))} and np.max(f) == np.max(p)


def test_phases_and_lengths():
    w = world(delay_min=5, delay_max=5, flash=3, timeout=7)
    seen = []
    for _ in range(3 + 5 + 7 + 2):
        seen.append(int(w.phase[0]))
        w.step(STILL)
    assert seen == [0] * 3 + [1] * 5 + [2] * 7 + [0] * 2


def test_delays_cover_the_range():
    w = world(delay_min=2, delay_max=4, timeout=1, flash=1)
    lens = []
    run = 0
    for _ in range(600):
        if w.phase[0] == 1:
            run += 1
        elif run:
            lens.append(run)
            run = 0
        w.step(STILL)
    assert set(lens) == {2, 3, 4}


@pytest.mark.parametrize("sample", [False, True])
def test_the_block_shows_only_in_the_flash(sample):
    for seed in range(40):
        w = world(seed=seed)
        if w.sample[0] == sample:
            break
    im = w.observe()["image"][0]
    b = im[:16, :16, 0]
    assert (im[..., 0] == im[..., 1]).all()
    v = b.max()
    assert v > BG
    if sample:
        np.testing.assert_array_equal(b, np.where(_checker(64, 2)[:16, :16], v, BG))
    else:
        assert (b == v).all()
    to_phase(w, 1)
    assert w.observe()["image"][0, :16, :16].max() == BG
    assert (w.observe()["image"][0, 40:] == BG).all()               # no circle, no trace of the sample


def test_the_block_keeps_its_pattern_in_a_b1_thumbnail():
    for seed in range(40):
        w = world(seed=seed)
        if w.sample[0]:
            break
    b = w.observe()["image"][0, :16, :16, 0].astype(float)
    thumb = b.reshape(8, 2, 8, 2).mean((1, 3))                      # 2 x 2 box average: one B1 cell scaled to a patch
    assert thumb.max() == b.max() and thumb.min() == BG and thumb.mean() == pytest.approx((b.max() + BG) / 2)


def test_the_food_looks_like_the_sample():
    seen = set()
    for seed in range(30):
        w = world(seed=seed)
        s = bool(w.sample[0])
        to_phase(w, 2)
        f, p = side_pixels(w)
        solid, checker = (p, f) if s else (f, p)
        assert (solid > BG).all()
        assert 0.3 < (checker > BG).mean() < 0.7
        seen.add(s)
    assert seen == {False, True}


def test_no_touch_before_the_choice_and_a_touch_ends_it():
    w = world(delay_min=3, delay_max=3)
    for _ in range(4):
        assert w.phase[0] != 2
        _, _, _, inf = w.step(STILL)
        assert not inf["success"][0] and not inf["gt_hazard_click"][0]
    to_phase(w, 2)
    assert w.target_alive[0] and w.hz_rad[0, 0] > 0
    d = (w.target[0] - w.cursor[0] + 32.0) % 64.0 - 32.0
    h1 = w.hunger.copy()
    a = np.array([[*(d / np.abs(d).max()), 0.0]], np.float32)
    for _ in range(12):
        _, _, _, inf = w.step(a)
        if inf["success"][0]:
            break
    assert inf["success"][0] and w.phase[0] == 0 and not w.target_alive[0] and w.hz_rad[0, 0] == 0
    assert w.hunger[0] <= h1[0] + 12 * w.config.h_rate
    st = w.stats()
    assert st["cue_trials"] >= 1 and st["cue_correct"] == 1.0 and st["cue_touch"] > 0


def test_touching_the_poison_counts_against():
    w = world(delay_min=1, delay_max=1)
    to_phase(w, 2)
    d = (w.hz_pos[0, 0] - w.cursor[0] + 32.0) % 64.0 - 32.0
    h = w.hunger[0]
    a = np.array([[*(d / np.abs(d).max()), 0.0]], np.float32)
    for _ in range(12):
        _, _, _, inf = w.step(a)
        if inf["gt_hazard_click"][0]:
            break
    assert inf["gt_hazard_click"][0] and w.hunger[0] > h and w.phase[0] == 0
    assert w.stats()["cue_correct"] == 0.0


def test_a_timeout_ends_the_trial_without_a_choice():
    w = world(delay_min=1, delay_max=1, timeout=3)
    to_phase(w, 2)
    w.stats()
    for _ in range(3):
        w.step(STILL)
    st = w.stats()
    assert w.phase[0] == 0 and st["cue_trials"] == 1 and st["cue_correct"] is None and st["cue_touch"] == 0


def test_hold_keeps_the_block_on():
    w = world(hold=True, delay_min=2, delay_max=2)
    b = w.observe()["image"][0, :16, :16].copy()
    for p in (1, 2):
        to_phase(w, p)
        np.testing.assert_array_equal(w.observe()["image"][0, :16, :16], b)


def test_step_returns_the_new_phases_frame():
    w = world(flash=1, delay_min=1, delay_max=1)
    obs, _, _, inf = w.step(STILL)
    np.testing.assert_array_equal(obs["image"], w.observe()["image"])
    np.testing.assert_array_equal(inf["terminal_image"], w.observe()["image"])
    assert obs["image"][0, :16, :16].max() == BG


def test_thought_world_over_it():
    w = world(thought=True)
    assert w.observe()["image"].shape == (1, 64, 128, 3)
    w.step(STILL)
    st = w.stats()
    assert "cue_trials" in st and "thought_extreme" in st


def test_world_invalid():
    with pytest.raises(ValueError):
        world(dist=5.0)
    with pytest.raises(ValueError):
        world(delay_min=9, delay_max=3)
    env = dict(ENV, hazards=2, cue={})
    env.pop("thought")
    with pytest.raises(ValueError):
        make_world(1, make_config(env, auto_reset=True))

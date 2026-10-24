"""Distractor dots (env.distractors, vm-00010): they move on their own, are drawn under the circle and the cursor,
and never change the task. With distractors=0 the world is unchanged."""
import dataclasses
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from world.circle_world import CircleWorld, EnvConfig  # noqa: E402

N, STEPS = 16, 300


def run_pair(seed=3, k=8, **kw):
    """The same seeds and actions in a world with k dots and one without."""
    cfg0 = EnvConfig(drives=True, **kw)
    cfg1 = dataclasses.replace(cfg0, distractors=k)
    w0, w1 = CircleWorld(N, cfg0, seed=seed), CircleWorld(N, cfg1, seed=seed)
    rng = np.random.default_rng(seed)
    out = []
    o0, o1 = w0.observe(), w1.observe()
    for _ in range(STEPS):
        a = np.concatenate([rng.uniform(-1, 1, (N, 2)), rng.random((N, 1)) < 0.3], 1)
        s0, s1 = w0.step(a), w1.step(a)
        out.append((o0, o1, s0, s1))
        o0, o1 = s0[0], s1[0]
    return w0, w1, out


def test_dots_never_change_the_task():
    w0, w1, out = run_pair()
    for _, _, (ob0, r0, d0, i0), (ob1, r1, d1, i1) in out:
        np.testing.assert_array_equal(r0, r1)
        np.testing.assert_array_equal(d0, d1)
        np.testing.assert_array_equal(ob0["drives"], ob1["drives"])
        for key in ("success", "click", "gt_cursor", "gt_target", "gt_radius", "gt_inside", "gt_drives"):
            np.testing.assert_array_equal(i0[key], i1[key])
    assert sum(int(s0[3]["success"].sum()) for _, _, s0, _ in out) > 0     # episodes ended with hits along the way


def test_dots_are_drawn_on_background_only():
    w0, w1, out = run_pair()
    bg = np.array(EnvConfig().bg_color, np.uint8)
    lo, hi = EnvConfig().dot_brightness_min, EnvConfig().dot_brightness_max
    n_dot_px = 0
    for o0, o1, _, _ in out:
        diff = np.any(o0["image"] != o1["image"], axis=-1)                # pixels the dots changed
        assert np.all(o0["image"][diff] == bg)                            # ... were background without dots
        g = o1["image"][diff].astype(int)
        assert np.all(g[:, 0] == g[:, 1]) and np.all(g[:, 1] == g[:, 2])  # grey
        assert np.all((g[:, 0] >= np.floor(lo)) & (g[:, 0] <= np.ceil(hi)))
        n_dot_px += int(diff.sum())
    assert n_dot_px > 0


def test_dots_move_bounce_and_keep_their_speed():
    cfg = EnvConfig(distractors=8)
    w = CircleWorld(N, dataclasses.replace(cfg, auto_reset=False, max_steps=10 ** 6), seed=5)
    speed0 = np.linalg.norm(w.dot_vel, axis=-1)
    assert np.all((speed0 >= cfg.dot_speed_min - 1e-9) & (speed0 <= cfg.dot_speed_max + 1e-9))
    flips = 0
    for _ in range(STEPS):
        v = w.dot_vel.copy()
        w.step(np.zeros((N, 3)))
        assert np.all((w.dot_pos >= 0) & (w.dot_pos <= cfg.size - cfg.dot_size))
        flips += int(np.sum(np.sign(v) != np.sign(w.dot_vel)))
        np.testing.assert_allclose(np.linalg.norm(w.dot_vel, axis=-1), speed0)
    assert flips > 0


def test_dots_are_deterministic_and_redrawn_at_reset():
    cfg = EnvConfig(distractors=8)
    a, b, c = CircleWorld(N, cfg, seed=11), CircleWorld(N, cfg, seed=11), CircleWorld(N, cfg, seed=12)
    np.testing.assert_array_equal(a.dot_pos, b.dot_pos)
    assert not np.array_equal(a.dot_pos, c.dot_pos)
    before = a.dot_pos.copy()
    a.reset(np.array([0, 3]))
    assert not np.array_equal(a.dot_pos[[0, 3]], before[[0, 3]])
    np.testing.assert_array_equal(a.dot_pos[1], before[1])


def test_state_roundtrip_with_dots():
    cfg = EnvConfig(distractors=8)
    a = CircleWorld(N, cfg, seed=2)
    rng = np.random.default_rng(0)
    acts = [np.concatenate([rng.uniform(-1, 1, (N, 2)), rng.random((N, 1)) < 0.3], 1) for _ in range(400)]
    for x in acts[:100]:
        a.step(x)
    st = a.get_state()
    b = CircleWorld(N, cfg, seed=99)
    b.set_state(st)
    for x in acts[100:]:
        oa, ob = a.step(x)[0], b.step(x)[0]
        np.testing.assert_array_equal(oa["image"], ob["image"])


def test_defaults_unchanged():
    cfg = EnvConfig()
    assert cfg.distractors == 0
    w = CircleWorld(4, cfg, seed=0)
    obs, _, _, info = w.step(np.zeros((4, 3)))
    assert set(obs) == {"image", "drives"} and "gt_dots" not in info and "dot_pos" not in w.get_state()
    d = EnvConfig(distractors=8).to_dict()
    assert EnvConfig.from_dict(d).distractors == 8

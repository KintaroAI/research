"""ThoughtWorld (task 00064): the world's half and body are the wrapped world's; a dream becomes the next thought; waves go by
and settle, overlap, and start only when thought is quiet; a short lifetime.
"""
import json
import os
import subprocess
import sys

import numpy as np
import pytest

DEV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, DEV)

import cuda_lib  # noqa: E402
from world import make_config, make_world  # noqa: E402
from world.thought_world import ThoughtConfig, ThoughtWorld, smooth_noise  # noqa: E402

ENV = dict(size=64, wrap=True, egocentric=True, touch=True, respawn=True, end_on_success=False, max_steps=10 ** 9, hazards=2,
           hazard_click_add=0.5, drives=True, gray=True, h_rate=0.001, bonuses=False, radius_min=8.0, radius_max=11.0,
           no_overlap=True)


def worlds(thought, n=2, seed=1):
    return make_world(n, make_config(dict(ENV, thought=thought), auto_reset=True), seed=seed), \
        make_world(n, make_config(ENV, auto_reset=True), seed=seed)


def test_the_world_half_is_the_world():
    t, w = worlds(dict(mode="frozen"))
    rng = np.random.default_rng(0)
    a0, b0 = t.observe(), w.observe()
    assert a0["image"].shape == (2, 64, 128, 3)
    np.testing.assert_array_equal(a0["image"][:, :, :64], b0["image"])
    f0 = a0["image"][:, :, 64:]
    for _ in range(40):
        m = rng.uniform(-1, 1, (2, 3)).astype(np.float32)
        a, _, _, _ = t.step(m)
        b, _, _, _ = w.step(m)
        img = t.think(None)
        np.testing.assert_array_equal(a["image"][:, :, :64], b["image"])
        np.testing.assert_array_equal(a["drives"], b["drives"])
        np.testing.assert_array_equal(img[:, :, :64], b["image"])
        np.testing.assert_array_equal(img[:, :, 64:], f0)                    # frozen: thought never changes


def test_a_dream_becomes_the_next_thought():
    t, _ = worlds(dict(p_wave=0.0, p_area=0.0))
    d = np.full((2, 64, 64, 3), 77, np.uint8)
    t.step(np.zeros((2, 3), np.float32))
    img = t.think(d)
    np.testing.assert_array_equal(img[:, :, 64:], d)


def test_birth_is_grey_and_smooth():
    x = smooth_noise(np.random.default_rng(0), 2, 64, 8)
    assert x.shape == (2, 64, 64) and x.min() >= 0 and x.max() <= 1
    assert np.abs(np.diff(x, axis=1)).max() < 0.2 and x.std() > 0.1


def wave_world(**kw):
    c = dict(p_wave=1.0, p_area=0.0, max_waves=1, settle=0.0, speed=2.0)
    c.update(kw)
    t, _ = worlds(c, n=1)
    t.T[:] = 0.5
    return t


def test_a_wave_passes_and_leaves_things_as_they_were():
    t = wave_world()
    t.step(np.zeros((1, 3), np.float32))
    t.cfg.max_waves = 1
    t.think(None)
    assert t.st["waves"] == 1
    t.cfg.p_wave = 0.0
    y, x = int(t.waves[0][0][0]), int(t.waves[0][0][1])
    seen = []
    for _ in range(80):
        t.think(None)
        seen.append(t.T[0, min(y + 10, 63) if y < 30 else y - 10, x])
    seen = np.array(seen)
    assert seen.max() > 0.52 and seen.min() < 0.48                           # up and down as it goes by
    assert not t.waves[0]
    assert np.abs(t.T - 0.5).mean() < 0.02                                   # and back


def test_waves_overlap():
    a, b = wave_world(max_waves=2), wave_world(max_waves=2)
    y, x = 32.0, 32.0
    one = a.wave(y, x, 5) + 0.0
    two = one + a.wave(y, x, 5)
    assert np.abs(two).max() == pytest.approx(2 * np.abs(one).max())


def test_waves_start_only_when_quiet():
    t, _ = worlds(dict(p_wave=1.0, p_area=1.0))
    t.step(np.zeros((2, 3), np.float32))
    busy = np.clip(np.rint(t.T * 255) + 40, 0, 255).astype(np.uint8)[..., None].repeat(3, -1)
    t.think(busy)
    assert t.st["waves"] == 0 and t.st["areas"] == 0
    t.think(None)
    assert t.st["waves"] == 2 and t.st["areas"] == 2


def test_invalid_mode():
    with pytest.raises(ValueError):
        ThoughtWorld(1, make_world(1, make_config(ENV, auto_reset=True)), ThoughtConfig(mode="x"))


@pytest.mark.skipif(not cuda_lib.available(), reason="needs libvmcuda.so and a GPU")
@pytest.mark.parametrize("mode,ring", [("dream", 1), ("frozen", 0)], ids=["dream_ring", "frozen"])
def test_a_lifetime(tmp_path, mode, ring):
    out = str(tmp_path / "r")
    cmd = [sys.executable, "lifetime.py", "--config", "configs/U_thought.json", "-o", out, "--seed", "2",
           "--set", "online.steps=40", "--set", "online.log_every=20", "--set", "online.save_every=40",
           "--set", "online.store_warmup=8", "--set", "env.thought.mode=" + mode, "--set", "online.ring=%d" % ring,
           "--set", "online.instinct_m=4", "--set", "online.ring=%d" % ring]
    r = subprocess.run(cmd, cwd=DEV, capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stderr[-2000:]
    row = [json.loads(line) for line in open(os.path.join(out, "metrics.jsonl"))][-1]
    assert "thought_change" in row
    if mode == "frozen":
        assert row["thought_change"] == 0.0
    else:
        assert row["thought_change"] > 0.0

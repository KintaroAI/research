"""CarWorld (task 00067): a car in MuJoCo with CircleWorld's interface. Frames; a seed repeats; the moves move it; touching food and
poison; the torus; a short lifetime of the default model through lifetime.py; invalid configs.

Needs mujoco in the venv (a no-AVX build here, see CLAUDE.md) and EGL; the lifetime also libvmcuda.so and a GPU.
"""
import json
import os
import subprocess
import sys

import numpy as np
import pytest

os.environ.setdefault("MUJOCO_GL", "egl")   # before the first import of mujoco
pytest.importorskip("mujoco")
DEV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, DEV)

import cuda_lib  # noqa: E402
from world import make_config, make_world  # noqa: E402
from world.car_world import CarConfig, CarWorld  # noqa: E402

HAVE_GPU = cuda_lib.available()


def world(n=1, seed=1, **kw):
    return CarWorld(n, CarConfig(**kw), seed=seed)


def drive(w, move, ticks):
    obs = None
    for _ in range(ticks):
        obs = w.step(np.tile(np.array(move, np.float32), (w.num_envs, 1)))
    return obs


def clear(w):
    """Every object far from the car, so that nothing is touched."""
    w.obj[:] = (w._xy(0) + w.config.side / 2) % w.config.side + np.random.default_rng(0).uniform(-.1, .1, w.obj.shape)


def test_frames_are_grey_and_the_interface_fits():
    w = world(2)
    obs = w.observe()
    assert obs["image"].shape == (2, 64, 64, 3) and obs["image"].dtype == np.uint8
    assert (obs["image"][..., 0] == obs["image"][..., 1]).all() and (obs["image"][..., 0] == obs["image"][..., 2]).all()
    assert obs["drives"].shape == (2, 2) and obs["drives"].dtype == np.float32
    o, r, done, inf = w.step(np.zeros((2, 3), np.float32))
    assert r.shape == (2,) and not done.any()
    for k in ("terminal_image", "terminal_drives", "success", "timeout", "click", "ep_length", "ep_return", "ep_success", "ep_clicks",
              "gt_hazard_click", "gt_drives"):
        assert k in inf, k
    with pytest.raises(ValueError):
        w.step(np.zeros((1, 3), np.float32))
    w.close()


def test_the_first_frame_has_its_horizon_level():
    """Bright floor below, black sky above: the camera is set before the first step."""
    w = world()
    a = w.observe()["image"][0, ..., 0]
    assert a[:20].mean() < 20 and a[-20:].mean() > 50
    w.close()


def test_a_seed_repeats():
    a, b, c = world(seed=3), world(seed=3), world(seed=4)
    assert np.array_equal(a.obj, b.obj) and not np.array_equal(a.obj, c.obj)
    assert np.array_equal(drive(a, [.5, 1, 0], 20)[0]["image"], drive(b, [.5, 1, 0], 20)[0]["image"])
    for w in (a, b, c):
        w.close()


def test_the_moves_move_it():
    w = world()
    clear(w)
    drive(w, [0, 1, 0], 40)
    assert 1.5 < w._xy(0)[0] < 2.5 and abs(w._xy(0)[1]) < 0.2                # about 1 m/s, along +x
    drive(w, [0, 0, 0], 20)
    p = w._xy(0)
    drive(w, [0, 0, 0], 40)
    assert np.hypot(*(w._xy(0) - p)) < 0.05                                   # no move, no motion
    y0 = w.yaw(0)
    drive(w, [1, 0, 0], 10)
    assert 0.8 < (w.yaw(0) - y0) % (2 * np.pi) < 2.0                          # turn: about 3 rad/s, on the spot
    assert np.hypot(*(w._xy(0) - p)) < 0.1
    click = world()
    clear(click)
    drive(click, [0, 1, 1], 5)
    plain = world()
    clear(plain)
    drive(plain, [0, 1, 0], 5)
    np.testing.assert_allclose(click._xy(0), plain._xy(0), atol=1e-9)         # the click does nothing
    for x in (w, click, plain):
        x.close()


@pytest.mark.parametrize("k,food", [(0, True), (4, False)])
def test_a_touch(k, food):
    w = world()
    w.hunger[:] = 0.4
    clear(w)
    w.obj[0, k] = w._xy(0) + [0.1, 0.0]
    _, _, _, inf = w.step(np.zeros((1, 3), np.float32))
    assert bool(inf["success"][0]) == food and bool(inf["gt_hazard_click"][0]) != food
    want = 0.0 if food else 0.901
    assert abs(float(w.hunger[0]) - want) < 1e-6
    assert np.hypot(*w.near(0, w._xy(0))[k]) >= w.config.spawn_min            # replaced, away from the car
    w.close()


def test_hunger_rises_and_stops_at_one():
    w = world()
    clear(w)
    drive(w, [0, 0, 0], 3)
    assert abs(float(w.hunger[0]) - 0.503) < 1e-9
    w.hunger[:] = 0.9995
    drive(w, [0, 0, 0], 2)
    assert float(w.hunger[0]) == 1.0
    w.close()


def test_the_torus_has_no_edge():
    """Far from the origin, the objects are still around the car, drawn at their nearest images."""
    w = world()
    clear(w)
    drive(w, [0, 1, 0], 200)
    xy = w._xy(0)
    assert xy[0] > w.config.side                                              # past one side of the torus
    w._show(0)
    rel = w.datas[0].mocap_pos[:, :2] - xy
    assert (np.abs(rel) <= w.config.side / 2 + 1e-9).all()
    w.close()


def test_make_world():
    w = make_world(1, make_config({"kind": "car", "foods": 2, "poisons": 1}, auto_reset=True), seed=1)
    assert isinstance(w, CarWorld) and w.K == 3
    w.close()


@pytest.mark.parametrize("bad", [dict(foods=0), dict(poisons=-1), dict(size=4), dict(touch=2.0), dict(side=2.0), dict(kind="x")],
                         ids=["no_food", "poisons", "size", "touch", "side", "kind"])
def test_invalid(bad):
    with pytest.raises(ValueError):
        CarConfig(**bad).validate()


@pytest.mark.skipif(not HAVE_GPU, reason="needs libvmcuda.so and a GPU")
def test_a_short_lifetime(tmp_path):
    out = str(tmp_path / "car")
    r = subprocess.run([sys.executable, "lifetime.py", "--config", "configs/U_car.json", "--seed", "3", "-o", out, "--set", "online.steps=60",
                        "--set", "online.save_every=60", "--set", "online.log_every=30", "--set", "online.store_warmup=16"],
                       cwd=DEV, capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stderr[-3000:]
    rows = [json.loads(line) for line in open(os.path.join(out, "metrics.jsonl"))]
    assert rows[-1]["step"] == 60 and "uni_ce_patch" in rows[-1]
    assert json.load(open(os.path.join(out, "info.json")))["pieces"] == 4

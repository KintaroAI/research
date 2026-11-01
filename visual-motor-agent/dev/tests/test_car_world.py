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


@pytest.mark.parametrize("hold", [1, 5], ids=["every_tick", "held_5"])
def test_harsh_driving_never_tips_it(hold):
    """The first car flipped under moves that change sharply, as a model's do, and stayed on its back (vm-00059)."""
    w = world()
    rng = np.random.default_rng(0)
    worst = 0.0
    m = np.zeros((1, 3), np.float32)
    for t in range(1500):
        if t % hold == 0:
            m = rng.choice([-1.0, 1.0], (1, 3)).astype(np.float32)
        w.step(m)
        worst = max(worst, w.tilt(0))
        assert w.datas[0].cam_xpos[0, 2] > 0.05                               # the camera stays above the floor
    assert worst < 5.0 and w.stats()["car_max_tilt"] == worst
    assert w.stats()["car_max_tilt"] == 0.0                                   # since the last read
    w.close()


SCHED = dict(dense=8, dense_hold=100, dense_end=400)


def test_the_schedule():
    w = world(**SCHED)
    assert w.K == 48 and w.F == 24
    assert w.counts(0) == w.counts(100) == (24, 24)
    assert w.counts(200) == (12, 12) and w.counts(300) == (6, 6)              # three halvings over dense_hold .. dense_end
    assert w.counts(399) == (3, 3) and w.counts(400) == w.counts(10 ** 6) == (3, 3)
    last = 24
    for t in range(100, 401):
        now = w.counts(t)[0]
        assert now <= last
        last = now
    assert w.stats() == dict(car_foods=24, car_poisons=24, car_max_tilt=0.0)
    w.close()


def test_no_schedule_by_default():
    w = world()
    assert w.counts(0) == w.counts(10 ** 6) == (3, 3) and w.alive().all() and w.K == 6
    w.close()


def test_a_schedule_with_one_changes_nothing():
    a, b = world(seed=5), world(seed=5, dense=1, dense_hold=7, dense_end=9)
    np.testing.assert_array_equal(a.obj, b.obj)
    np.testing.assert_array_equal(drive(a, [.3, 1, 0], 30)[0]["image"], drive(b, [.3, 1, 0], 30)[0]["image"])
    a.close()
    b.close()


def test_a_removed_box_is_neither_drawn_nor_touched():
    w = world(**SCHED)
    w.t = 400
    alive = w.alive()
    assert alive.sum() == 6 and alive[:3].all() and not alive[3:24].any() and alive[24:27].all() and not alive[27:].any()
    clear(w)
    gone = 10                                                                  # a removed food box
    w.obj[0, gone] = w._xy(0) + [0.1, 0.0]
    w.hunger[:] = 0.4
    _, _, _, inf = w.step(np.zeros((1, 3), np.float32))
    assert not inf["success"][0] and abs(float(w.hunger[0]) - 0.401) < 1e-6
    assert w.datas[0].mocap_pos[gone, 2] == -5.0 and w.datas[0].mocap_pos[0, 2] == 0.0
    w.obj[0, gone] = w._xy(0) + [1.0, 0.0]                                     # in front of the camera, and not in the frame
    w._frame = None
    empty = w.observe()["image"]
    w.t = 0                                                                    # alive again: drawn
    w._frame = None
    assert not np.array_equal(empty, w.observe()["image"])
    w.close()


def test_a_touched_box_is_replaced_while_the_count_stands():
    w = world(**SCHED)
    w.hunger[:] = 0.4
    clear(w)
    w.obj[0, 30] = w._xy(0) + [0.1, 0.0]                                       # a poison box above the final count
    _, _, _, inf = w.step(np.zeros((1, 3), np.float32))
    assert inf["gt_hazard_click"][0] and not inf["success"][0]
    assert np.hypot(*w.near(0, w._xy(0))[30]) >= w.config.spawn_min and w.alive()[30]
    w.close()


def test_a_dense_world_starts_spread_out():
    w = world(**SCHED)
    rel = w.obj[0][:, None] - w.obj[0][None]
    d = np.hypot(*(((rel + 2.5) % 5.0 - 2.5).transpose(2, 0, 1)))
    assert d[~np.eye(48, dtype=bool)].min() >= 0.5
    assert np.hypot(*w.near(0, w._xy(0)).T).min() >= 1.0
    w.close()


def test_make_world():
    w = make_world(1, make_config({"kind": "car", "foods": 2, "poisons": 1}, auto_reset=True), seed=1)
    assert isinstance(w, CarWorld) and w.K == 3
    w.close()


@pytest.mark.parametrize("bad", [dict(foods=0), dict(poisons=-1), dict(size=4), dict(touch=2.0), dict(side=2.0), dict(kind="x"), dict(dense=0),
                                 dict(dense=2), dict(dense=2, dense_hold=5, dense_end=5)],
                         ids=["no_food", "poisons", "size", "touch", "side", "kind", "dense", "no_end", "hold_at_end"])
def test_invalid(bad):
    with pytest.raises(ValueError):
        CarConfig(**bad).validate()


@pytest.mark.skipif(not HAVE_GPU, reason="needs libvmcuda.so and a GPU")
@pytest.mark.parametrize("cfg,want", [("U_car", 3), ("U_car_dense", 3)], ids=["plain", "dense"])
def test_a_short_lifetime(tmp_path, cfg, want):
    out = str(tmp_path / "car")
    extra = ["--set", "env.dense_hold=20", "--set", "env.dense_end=50"] if cfg == "U_car_dense" else []
    r = subprocess.run([sys.executable, "lifetime.py", "--config", "configs/%s.json" % cfg, "--seed", "3", "-o", out, "--set", "online.steps=60",
                        "--set", "online.save_every=60", "--set", "online.log_every=30", "--set", "online.store_warmup=16"] + extra,
                       cwd=DEV, capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stderr[-3000:]
    rows = [json.loads(line) for line in open(os.path.join(out, "metrics.jsonl"))]
    assert rows[-1]["step"] == 60 and "uni_ce_patch" in rows[-1] and rows[-1]["car_foods"] == want
    if cfg == "U_car_dense":
        assert rows[0]["car_foods"] > want
    assert json.load(open(os.path.join(out, "info.json")))["pieces"] == 4

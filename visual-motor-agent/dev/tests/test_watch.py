"""Task 00132: worlds to watch, and a model that only watches.

- WatchWorld (world/watch_world.py): food and poison discs that keep their speeds, never overlap and mirror off each
  other; the looks; nothing left to chance after birth; the move does nothing.
- BreakoutWorld's script (env.auto_lag): the paddle follows the ball as it was d ticks ago, the move given is ignored,
  and with the script off the court plays as before.
- online.watch: the null move every tick, in a short life (requires libvmcuda.so and a GPU); the refusals.
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
from online import DEFAULT_ONLINE  # noqa: E402
from world import make_config, make_world  # noqa: E402
from world.circle_world import wrap_delta  # noqa: E402
from world.watch_world import WatchConfig, WatchWorld  # noqa: E402

gpu = pytest.mark.skipif(not cuda_lib.available(), reason="needs libvmcuda.so and a GPU")


def watch(n=1, seed=0, **kw):
    return make_world(n, make_config(dict({"kind": "watch"}, **kw)), seed=seed)


def court(n=1, seed=0, **kw):
    return make_world(n, make_config(dict({"kind": "breakout"}, **kw)), seed=seed)


def none(n):
    return np.zeros((n, 3))


def gaps(w):
    """The gap between every two discs' edges, per world (negative: an overlap)."""
    d = wrap_delta(w.pos[:, :, None, :], w.pos[:, None, :, :], float(w.config.size))
    g = np.hypot(d[..., 0], d[..., 1]) - (w.rad[:, :, None] + w.rad[:, None, :])
    i = np.triu_indices(w.rad.shape[1], 1)
    return g[:, i[0], i[1]]


# ---------------------------------------------------------------------- the discs
def test_factory_frame_and_looks():
    w = watch(2, dots=0)
    assert isinstance(w, WatchWorld)
    o = w.observe()
    c = w.config
    assert o["image"].shape == (2, 64, 64, 3) and o["image"].dtype == np.uint8
    assert (o["image"][..., 0] == o["image"][..., 1]).all() and (o["image"][..., 0] == o["image"][..., 2]).all()   # grey
    assert o["drives"].shape == (2, 2) and not o["drives"].any()             # no body: constants
    img = o["image"][0, ..., 0]
    assert set(np.unique(img)) == {c.bg, c.grey}
    px = np.arange(64) + 0.5
    for j in range(w.rad.shape[1]):
        dx, dy = wrap_delta(px[None, :], w.pos[0, j, 0], 64.0), wrap_delta(px[:, None], w.pos[0, j, 1], 64.0)
        disc = img[dx * dx + dy * dy <= w.rad[0, j] ** 2]
        if w.kind[j] == 0:
            assert (disc == c.grey).all()                                    # food: solid
        else:
            assert 0.3 < (disc == c.grey).mean() < 0.7                       # poison: the view's checker, about half lit
    assert (w.kind == [0, 0, 1, 1]).all()                                    # foods first


def test_dots_lie_under_the_discs():
    w = watch(1, seed=2)
    img = w.observe()["image"][0, ..., 0]
    c = w.config
    assert len(set(np.unique(img)) - {c.bg, c.grey}) >= 1                    # dots show
    px = np.arange(64) + 0.5
    for j in np.flatnonzero(w.kind == 0):                                    # never on a food disc
        dx, dy = wrap_delta(px[None, :], w.pos[0, j, 0], 64.0), wrap_delta(px[:, None], w.pos[0, j, 1], 64.0)
        assert (img[dx * dx + dy * dy <= w.rad[0, j] ** 2] == c.grey).all()


def test_speeds_are_kept_and_nothing_overlaps():
    w = watch(3, seed=1)
    assert gaps(w).min() > 0
    low = np.inf
    for _ in range(3000):
        _, _, _, info = w.step(none(3))
        low = min(low, gaps(w).min())
    assert low > -1e-6
    sp = np.hypot(w.vel[..., 0], w.vel[..., 1])
    assert np.allclose(sp[:, w.kind == 0], w.config.food_speed) and np.allclose(sp[:, w.kind == 1], w.config.poison_speed)
    assert w.stats()["wa_bounces"] > 100                                     # they do meet
    assert (info["gt_kind"] == w.kind).all() and info["gt_pos"].shape == (3, 4, 2)


def test_a_bounce_mirrors_each_disc_that_was_closing_in():
    w = watch(1, foods=1, poisons=1, dots=0)
    w.rad[0] = (8.0, 6.0)
    w.pos[0] = ((20.0, 32.0), (36.0, 32.0))                                  # a gap of 2 px, on one line
    w.vel[0] = ((1.5, 0.0), (-0.75, 0.0))                                    # head on
    _, _, _, info = w.step(none(1))
    assert info["gt_bounce"][0].all()
    assert np.allclose(w.vel[0], ((-1.5, 0.0), (0.75, 0.0)))
    assert gaps(w).min() >= -1e-9
    w.pos[0] = ((20.0, 32.0), (34.5, 32.0))                                  # the food catches a poison that runs from it
    w.vel[0] = ((1.5, 0.0), (0.75, 0.0))
    _, _, _, info = w.step(none(1))
    assert (info["gt_bounce"][0] == [True, False]).all()
    assert np.allclose(w.vel[0], ((-1.5, 0.0), (0.75, 0.0)))
    w.pos[0] = ((20.0, 20.0), (40.0, 44.0))                                  # apart: nothing turns
    _, _, _, info = w.step(none(1))
    assert not info["gt_bounce"].any()


def test_the_torus():
    w = watch(1, foods=1, poisons=0, dots=0)
    w.pos[0, 0] = (63.5, 0.25)
    w.vel[0, 0] = (1.5, -0.5)
    w.step(none(1))
    assert np.allclose(w.pos[0, 0], (1.0, 63.75))
    assert (w.observe()["image"][0, 0, :, 0] == w.config.grey).any()         # drawn across the edge


def test_nothing_is_left_to_chance_and_the_move_does_nothing():
    a, b = watch(2, seed=5), watch(2, seed=5)
    rng = np.random.default_rng(0)
    for _ in range(300):
        oa = a.step(none(2))[0]
        ob = b.step(rng.uniform(-1, 1, (2, 3)))[0]
    assert (oa["image"] == ob["image"]).all() and (a.pos == b.pos).all()
    assert not (watch(2, seed=6).observe()["image"] == a.reset()["image"]).all()      # another seed, another world
    c = watch(2, seed=9)                                                     # the state is the whole world
    c.set_state(a.get_state())
    for _ in range(200):
        oa, oc = a.step(none(2))[0], c.step(none(2))[0]
    assert (oa["image"] == oc["image"]).all() and c.t == a.t


def test_validate():
    for bad in (dict(food_speed=9.0), dict(foods=0, poisons=0), dict(foods=8, poisons=8), dict(food_radius=(5.0, 3.0)),
                dict(size=8), dict(bg=94)):
        with pytest.raises(ValueError):
            WatchConfig(**bad).validate()
    with pytest.raises(ValueError):
        watch(1).step(np.zeros((2, 3)))


# ---------------------------------------------------------------------- the court's script
def test_the_script_is_off_by_default_and_the_court_unchanged():
    a, b = court(2, seed=3), court(2, seed=3, auto_lag=0)
    rng = np.random.default_rng(1)
    assert a.config.auto_lag == 0
    for _ in range(400):
        m = rng.uniform(-1, 1, (2, 3))
        oa, ob = a.step(m)[0], b.step(m)[0]
    assert (oa["image"] == ob["image"]).all() and (a.paddle == b.paddle).all() and a.paddle.std() > 0


def test_the_script_ignores_the_move():
    a, b = court(2, seed=4, auto_lag=6), court(2, seed=4, auto_lag=6)
    rng = np.random.default_rng(2)
    for _ in range(600):
        oa, ob = a.step(none(2))[0], b.step(rng.uniform(-1, 1, (2, 3)))[0]
    assert (oa["image"] == ob["image"]).all() and (a.hunger == b.hunger).all()


@pytest.mark.parametrize("lag", [1, 6])
def test_the_paddle_goes_where_the_ball_was(lag):
    w = court(1, seed=7, auto_lag=lag)
    c = w.config
    seen = []                                                                # (the ball's centre x, in play) before each step
    for t in range(1500):
        seen.append((float(w.ball[0, 0] + c.ball / 2.0), bool(w.live[0])))
        before = float(w.paddle[0])
        w.step(none(1))
        x, live = seen[t - lag] if t >= lag else (0.0, False)
        if not live:
            assert w.paddle[0] == before                                     # no ball then: the paddle stays
            continue
        want = np.clip(before + np.clip(x - (before + c.paddle_w / 2.0), -c.paddle_speed, c.paddle_speed), w.lo, w.hi - c.paddle_w)
        assert abs(w.paddle[0] - want) < 1e-9
    assert w.hits[0] > 5


def test_the_script_plays_a_varied_game_with_hits_and_misses():
    w = court(4, seed=1, auto_lag=6)
    for _ in range(10000):
        w.step(none(4))
    share = w.hits.sum() / (w.hits.sum() + w.misses.sum())
    assert 0.65 < share < 0.95 and w.misses.sum() > 20                       # about .80: both events happen
    with pytest.raises(ValueError):
        court(1, auto_lag=-1)


# ---------------------------------------------------------------------- the model only watches
def test_the_default_acts():
    assert DEFAULT_ONLINE["watch"] == 0


def life(tmp_path, config, *sets):
    cmd = [sys.executable, os.path.join(DEV, "lifetime.py"), "--config", os.path.join(DEV, "configs", config),
           "--set", "online.steps=40", "--set", "online.log_every=20", "--set", "online.save_every=100000",
           "--set", "online.frame_record=40", "--seed", "1", "-o", str(tmp_path / "run")]
    for s in sets:
        cmd += ["--set", s]
    return subprocess.run(cmd, cwd=DEV, capture_output=True, text=True)


@gpu
def test_a_watching_life_makes_no_move(tmp_path):
    r = life(tmp_path, "W_breakout.json", "env.auto_lag=0")                  # the script off: only the model could move the paddle
    assert r.returncode == 0, r.stderr[-2000:]
    f = np.load(tmp_path / "run" / "frames.npy")
    c = make_config(json.load(open(os.path.join(DEV, "configs", "W_breakout.json")))["env"])
    assert f.shape[0] == 40 and (f[:, c.paddle_y] == f[0, c.paddle_y]).all()             # the paddle's row never changes
    assert len([json.loads(line) for line in open(tmp_path / "run" / "metrics.jsonl")]) == 2


@gpu
@pytest.mark.parametrize("config", ["W_breakout.json", "W_discs.json"])
def test_the_watching_configs_live(tmp_path, config):
    r = life(tmp_path, config)
    assert r.returncode == 0, r.stderr[-2000:]
    f = np.load(tmp_path / "run" / "frames.npy")
    assert f.shape[0] == 40 and (f[1:] != f[:-1]).any((1, 2)).all()                      # the world moves on its own, every tick


@gpu
@pytest.mark.parametrize("bad", [["online.instinct_m=16"], ["online.act_every=4"]], ids=["instinct", "repeat"])
def test_the_refusals(tmp_path, bad):
    r = life(tmp_path, "W_breakout.json", *bad)
    assert r.returncode != 0 and ("watch" in r.stderr or "act_every" in r.stderr)

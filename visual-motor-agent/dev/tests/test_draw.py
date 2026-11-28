"""Task 00138: a dream that rolls dice (the decoding `draw`).

The rule on given chances: the device draws as the numpy reference says; a value with one hump is `pair`'s whatever the
dice; two humps are drawn by their weights, as they are and raised to 1 / temperature, a hump weighed by its chance or
by its tallest bin (`pair`'s ranking, so that without dice it is `pair`). A dream with dice: the same dice give the
same dream and other dice another, the decoding settles and does not depend on its first guess, a scoring names its
temperature and dice. And the court's balls in a long dream, counted on frames made by hand."""
import json
import os
import subprocess
import sys

import numpy as np
import pytest

DEV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, DEV)

import cuda_lib  # noqa: E402
import wm_measures as W  # noqa: E402

gpu = pytest.mark.skipif(not cuda_lib.available(), reason="needs libvmcuda.so and a GPU")
K = 32


def pair_value(p):
    """`pair` (the device's mode 2) in numpy: the most probable bin and its more probable neighbour, weighted."""
    p = np.asarray(p, np.float32)
    best = p.argmax(1)
    nb = np.where(best == 0, 1, np.where(best == K - 1, K - 2, np.where(p[np.arange(len(p)), np.minimum(best + 1, K - 1)]
                                                                       > p[np.arange(len(p)), np.maximum(best - 1, 0)], best + 1, best - 1)))
    a, b = p[np.arange(len(p)), best], p[np.arange(len(p)), nb]
    return (a * best + b * nb) / ((a + b) * (K - 1))


@gpu
@pytest.mark.parametrize("peak", [False, True], ids=["by chance", "by the tallest bin"])
def test_the_device_draws_as_the_rule_says(peak):
    rng = np.random.default_rng(1)
    x = rng.random((3000, K), dtype=np.float32) ** 6                       # a few bins carry most: several humps a value
    p = (x / x.sum(1, keepdims=True)).astype(np.float32)
    u = rng.random(3000, dtype=np.float32)
    for temp in (1.0, 0.7, 0.5, 2.0, 0.0):
        a, b = cuda_lib.draw_values(p, u, temp, peak), W.draw_values(p, u, temp, peak)
        assert np.abs(a - b).max() < 1e-6, (temp, peak)
        assert a.min() >= 0.0 and a.max() <= 1.0 + 1e-6                      # (the weighted centre may round a bit past 1, as pair's)
    assert len(np.unique(cuda_lib.draw_values(p, u, 1.0, peak))) > 100       # and the dice do choose
    if peak:                                                                 # without dice, by the tallest bin: `pair`
        assert np.abs(cuda_lib.draw_values(p, u, 0.0, True) - pair_value(p)).max() < 1e-6
    with pytest.raises(ValueError):
        cuda_lib.draw_values(p, u[:10])
    with pytest.raises(RuntimeError):
        cuda_lib.draw_values(np.ones((4, 65), np.float32) / 65, u[:4])       # more bins than the device's 64


@gpu
def test_a_value_with_one_hump_is_pairs_whatever_the_dice():
    rng = np.random.default_rng(2)
    v = rng.random(2000) * (K - 1)
    lo = np.floor(v).astype(int).clip(0, K - 2)
    f = (v - lo).astype(np.float32)
    two_hot = np.zeros((2000, K), np.float32)                                # a sure grey level between two bins
    two_hot[np.arange(2000), lo], two_hot[np.arange(2000), lo + 1] = 1 - f, f
    b = np.arange(K, dtype=np.float32)
    bell = np.exp(-0.5 * ((b[None] - v[:, None].astype(np.float32)) / 0.8) ** 2)   # one hump with tails
    bell = (bell / bell.sum(1, keepdims=True)).astype(np.float32)
    for p, exact in ((two_hot, True), (bell, False)):
        want = pair_value(p)
        for temp in (1.0, 0.5, 3.0, 0.0):
            for peak in (False, True):
                got = cuda_lib.draw_values(p, rng.random(2000, dtype=np.float32), temp, peak)
                assert np.abs(got - want).max() < 1e-6, (temp, peak)
        if exact:
            assert np.abs(want - v / (K - 1)).max() < 1e-6                   # and a two-hot value decodes exactly


@gpu
def test_two_humps_are_drawn_by_their_weights():
    rng = np.random.default_rng(3)
    n = 40000
    q = np.zeros((n, K), np.float32)
    q[:, 2], q[:, 3], q[:, 28], q[:, 29] = .60, .30, .06, .04               # dark: chance .9, tallest bin .6; bright: .1, .06
    dark, bright = (.6 * 2 + .3 * 3) / (.9 * 31), (.06 * 28 + .04 * 29) / (.1 * 31)
    for peak, a, c in ((False, .9, .1), (True, .6, .06)):
        for temp in (1.0, 0.5, 2.0):
            got = cuda_lib.draw_values(q, rng.random(n, dtype=np.float32), temp, peak)
            assert np.abs(np.where(got > .5, got - bright, got - dark)).max() < 1e-6      # one of the two sharp values
            want = c ** (1 / temp) / (c ** (1 / temp) + a ** (1 / temp))
            assert abs((got > .5).mean() - want) < 4 * np.sqrt(want * (1 - want) / n) + 1e-4, (peak, temp, (got > .5).mean(), want)
        assert (cuda_lib.draw_values(q, rng.random(n, dtype=np.float32), 0.0, peak) < .5).all()   # no dice: the heavier
    # `pair` favours a grey level that sits on a bin: dark between two bins (.30 and .25: chance .55), bright on one (.45)
    q = np.zeros((n, K), np.float32)
    q[:, 2], q[:, 3], q[:, 31] = .30, .25, .45
    u = rng.random(n, dtype=np.float32)
    assert (cuda_lib.draw_values(q, u, 0.0, False) < .5).all()               # the likelier value is the dark one
    assert (cuda_lib.draw_values(q, u, 0.0, True) > 1 - 1e-6).all() and (pair_value(q) > 1 - 1e-6).all()   # `pair` draws the bright
    share = (cuda_lib.draw_values(q, u, 1.0, False) > .5).mean()
    assert abs(share - .45) < .012 and abs((cuda_lib.draw_values(q, u, 1.0, True) > .5).mean() - .45 / .75) < .012


# ---------------------------------------------------------------------------------------------- a dream with dice
@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    out = {}
    for name in ("W1_breakout.json", "W_breakout.json"):
        d = tmp_path_factory.mktemp("draw") / name[:-5]
        r = subprocess.run([sys.executable, os.path.join(DEV, "lifetime.py"), "--config", os.path.join(DEV, "configs", name),
                            "--set", "online.steps=8", "--set", "online.log_every=8", "--set", "online.save_every=100000",
                            "--seed", "1", "-o", str(d)], cwd=DEV, capture_output=True, text=True)
        assert r.returncode == 0, r.stderr[-2000:]
        out[name] = str(d)
    return out


@pytest.fixture(scope="module")
def court():
    env = json.load(open(os.path.join(DEV, "configs", "W_breakout.json")))["env"]
    return W.Exam(env, seed=5, copies=6, ticks=900, starts=(100, 180, 260, 340))


def dream(run, decode, court, dice=0, guess=None, H=3):
    m = W.Pieces(run, 0, decode, dice=dice)
    e, s = court.regular()
    e, s = e[:6], s[:6]
    null = np.zeros((6, 3), np.float32)
    m.reset(6)
    for j in range(9, -1, -1):
        m.observe(court.frames[e, s - j], court.body[e, s - j], null)
    m.guess = guess
    f, b = m.dream(H)
    again = m.dream(H)[0]
    stats = (m.unsettled, max(m.passes), m.max_passes)
    m.close()
    return f, b, again, stats


@gpu
def test_a_dream_with_dice(runs, court):
    run = runs["W1_breakout.json"]
    fp, bp, again, _ = dream(run, "pair", court)
    assert (again == fp).all()                                               # without dice a dream repeats
    f0, b0, _, st = dream(run, "drawp0", court)                              # by the tallest bin and no dice: `pair`
    assert (f0 == fp).all() and np.allclose(b0, bp, atol=1e-6) and st[0] == 0
    f1, b1, again, st = dream(run, "draw", court)
    assert f1.shape == (6, 3, 64, 64) and f1.dtype == np.uint8 and st[0] == 0 and st[1] <= st[2]   # it settles
    assert (f1 != fp).any() and (again != f1).any()                          # the dice choose, and a new dream is a new throw
    assert np.allclose(b1[:, 0], bp[:, 0], atol=1e-6)                        # the first body is the expected one, dice or none
    f2 = dream(run, "draw", court)[0]
    assert (f2 == f1).all()                                                  # the same dice, the same calls: the same dream
    assert (dream(run, "draw", court, dice=1)[0] != f1).any()                # other dice, another dream
    guess = np.random.default_rng(0).integers(0, 256, (6, 64, 64)).astype(np.uint8)
    assert (dream(run, "draw", court, guess=guess)[0] == f1).all()           # drawn token by token, not a guess kept
    for name, temp, peak in (("draw", 1.0, False), ("draw0.7", 0.7, False), ("drawp", 1.0, True), ("drawp0.5", 0.5, True)):
        m = W.Pieces(run, 0, name)
        assert (m.temp, m.peak) == (temp, peak)
        m.close()
    m = W.Pieces(run, 0, "pair")
    assert m.temp is None
    m.close()
    with pytest.raises(ValueError):
        W.Pieces(runs["W_breakout.json"], 0, "draw")                         # one transformer only


@gpu
def test_scored_with_dice(runs, court, tmp_path, monkeypatch):
    monkeypatch.setattr(W, "FREE_TICKS", 12)
    out = str(tmp_path / "other" / "wm")
    r = W.score(runs["W1_breakout.json"], 0, "draw0.7", "work", parts=("free",), exam=court, out_dir=out, name="free", dice=3)
    p = os.path.join(out, "step_00000000_draw0.7_free.json")
    assert os.path.exists(p) and not os.path.exists(os.path.join(runs["W1_breakout.json"], "wm"))   # nothing beside the run
    s = json.load(open(p))
    assert (s["decode"], s["temp"], s["dice"], s["peak"]) == ("draw0.7", 0.7, 3, False) and s["unsettled"] == {"free": 0}
    b = s["free"]["balls"]
    assert len(b["ball"]) == 12 and len(b["last"]) == s["free"]["starts"] == 12 and set(b) == set(s["free"]["balls_real"])
    assert r["free"]["balls_real"]["back"] == 0
    s = W.score(runs["W1_breakout.json"], 0, "drawp", "work", parts=("free",), exam=court, out_dir=out, name="free")
    assert (s["temp"], s["dice"], s["peak"]) == (1.0, 0, True)
    s = W.score(runs["W1_breakout.json"], 0, "pair", "work", parts=("free",), exam=court, out_dir=out, name="free")
    assert "temp" not in s and "balls" in s["free"]


# ---------------------------------------------------------------------------------------------- the court's balls
def test_the_courts_balls(court):
    c = court.cfg
    mask, lv = {n: (m, lv) for n, m, lv in court.zones()}["court"]
    base = court.frames[0, 100].copy()
    base[(base > lv) & mask] = c.bg                                          # the court with no ball
    f = np.repeat(base[None, None], 40, 1).repeat(2, 0)

    def ball(k, t, y, x, cells=((0, 0), (0, 1), (1, 0), (1, 1))):
        for dy, dx in cells:
            f[k, t, y + dy, x + dx] = c.ball_grey

    for t in range(10):
        ball(0, t, 30, 20 + t)                                               # a ball in play
    for t in (13, 14, 15):
        ball(0, t, 4 + (t - 13), 30)                                         # back at the top after 3 ticks without: served
    for t in range(20, 25):
        ball(0, t, 40, 12, cells=((0, 0), (0, 1), (1, 0)))                   # back elsewhere after 4, a pixel short: a ball
    ball(0, 26, 40, 12)                                                      # after 1 tick without: no return
    for t in range(30, 40):
        f[0, t, 10, 10] = c.ball_grey                                        # a stray pixel in the serving rows
    for t in (5, 6):
        ball(1, t, 4, 40)                                                    # at the top for 2 ticks only: not served
    for t in range(20, 40):
        ball(1, t, 4 + (t - 20), 25)                                         # served, and it stays
    r = W.court_balls(court, f)
    want = np.zeros((2, 40), bool)
    want[0, :10] = want[0, 13:16] = want[0, 20:25] = want[0, 26] = True
    want[1, 5:7] = want[1, 20:] = True
    assert np.allclose(r["ball"], want.mean(0))
    whole = want.copy()
    whole[0, 20:25] = False                                                  # a pixel short is a ball, not a whole one
    assert np.allclose(r["whole"], whole.mean(0))
    anyb = want.copy()
    anyb[0, 30:] = True
    assert np.allclose(r["any"], anyb.mean(0))
    assert (r["served"], r["back"], r["served_by_dream"], r["last"]) == (2, 1, [1, 1], [27, 40])
    assert r["specks"] == pytest.approx(10 / 80) and r["specks_top"] == pytest.approx(10 / (21 + 18))   # 39 frames without a ball
    # the world itself: its balls are whole, none comes back but at the top, and it has next to no stray pixel
    w = W.court_balls(court, court.frames[:, 100:400])
    assert w["back"] == 0 and w["served"] >= 1 and w["specks"] < .1
    assert all(a >= b - 1e-9 for a, b in zip(w["any"], w["ball"])) and all(a >= b - 1e-9 for a, b in zip(w["ball"], w["whole"]))

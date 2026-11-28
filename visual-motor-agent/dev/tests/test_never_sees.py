"""Task 00139: a life that never sees (online.blind over the whole frame) and the model's picture of every frame
before it learns from it (online.pred_score, pred_record, pred_every; pred_show.py).

With the whole frame blind no pixel of a real frame reaches the model: two worlds that differ in their frames give the
same pictures from the same weights with learning off (and other pictures once the model is shown them). The errors
in the metrics are those of the kept frames. The exam shown as a blind life saw it is black. The frozen weights on the
life's own world are lined up with the life and paint what the life painted."""
import json
import os
import subprocess
import sys

import numpy as np
import pytest

DEV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, DEV)

import cuda_lib  # noqa: E402
import pred_show as P  # noqa: E402
import wm_measures as W  # noqa: E402

pytestmark = pytest.mark.skipif(not cuda_lib.available(), reason="needs libvmcuda.so and a GPU")
BLACK = ["online.blind=[0,0,64,64]"]
SCORE = ["online.pred_score=1", "online.pred_record=48", "online.log_every=16", "online.save_every=48", "online.steps=48"]


def life(out, sets, config="W1_discs.json", seed=1, init=None, ok=True):
    cmd = [sys.executable, os.path.join(DEV, "lifetime.py"), "--config", os.path.join(DEV, "configs", config), "--seed", str(seed),
           "-o", str(out)] + (["--init", init] if init else []) + sum([["--set", s] for s in sets], [])
    r = subprocess.run(cmd, cwd=DEV, capture_output=True, text=True)
    assert (r.returncode == 0) == ok, r.stderr[-1500:]
    return r


@pytest.fixture(scope="module")
def lives(tmp_path_factory):
    """One set of weights (a life's first checkpoint) in two disc worlds of other seeds, learning off: blind, and seeing;
    and a blind life that learns."""
    d = tmp_path_factory.mktemp("never")
    life(d / "first", ["online.steps=8", "online.log_every=8", "online.save_every=100000"])
    init = str(d / "first" / "ckpt_00000000.bin")              # one transformer: the file itself
    for name, seed, sets in (("blind2", 2, BLACK), ("blind3", 3, BLACK), ("sees2", 2, []), ("sees3", 3, [])):
        life(d / name, SCORE + sets + ["ppo.lr=0"], seed=seed, init=init)
    life(d / "learns", SCORE + BLACK)
    return {n: str(d / n) for n in ("first", "blind2", "blind3", "sees2", "sees3", "learns")}


def kept(run):
    z = np.load(os.path.join(run, "pred_frames.npz"))
    return z["tick"], z["real"], z["pred"]


def test_no_pixel_of_a_real_frame_reaches_the_model(lives):
    t2, real2, pred2 = kept(lives["blind2"])
    t3, real3, pred3 = kept(lives["blind3"])
    assert t2.tolist() == t3.tolist() == list(range(1, 49))
    assert (real2 != real3).any()                                  # two worlds
    assert (pred2 == pred3).all()                                  # one picture: nothing of the frames went in
    assert (pred2[8:] == pred2[-1]).all()                          # and, with learning off, the same at every tick with a whole window
    s2, s3 = kept(lives["sees2"])[2], kept(lives["sees3"])[2]
    assert (s2 != s3).any() and (s2[8:] != pred2[8:]).any()        # shown its frames the same model paints other pictures
    assert (kept(lives["learns"])[2][-1] != kept(lives["learns"])[2][8]).any()   # and with learning on the blind picture changes


def test_the_errors_in_the_metrics_are_the_kept_frames(lives):
    for name in ("learns", "sees2"):
        run = lives[name]
        tick, real, pred = kept(run)
        r64 = real.astype(np.float64)
        err = ((pred.astype(np.float64) - r64) ** 2).sum((1, 2))
        copy = np.array([np.nan] + [((r64[i - 1] - r64[i]) ** 2).sum() for i in range(1, len(r64))])
        mean = np.array([np.nan] + [((np.rint(r64[:i].mean(0)) - r64[i]) ** 2).sum() for i in range(1, len(r64))])
        rows = P.metrics(run)
        assert [r["step"] for r in rows] == [16, 32, 48]
        for r in rows:
            m = (tick > r["step"] - 16) & (tick <= r["step"]) & (tick > 1)       # the first tick has no frame before it
            assert r["pred_err"] == pytest.approx(err[m].mean()) and r["pred_copy"] == pytest.approx(copy[m].mean())
            assert r["pred_mean"] == pytest.approx(mean[m].mean())
    assert "pred_err" not in json.loads(open(os.path.join(lives["first"], "metrics.jsonl")).readline())   # off: not there


def test_the_frames_kept(tmp_path):
    life(tmp_path / "run", ["online.steps=60", "online.log_every=30", "online.save_every=60", "online.pred_score=1", "online.pred_record=4",
                            "online.pred_every=25"] + BLACK, config="W1_breakout.json")
    tick, real, pred = kept(str(tmp_path / "run"))
    assert tick.tolist() == [1, 2, 3, 4, 25, 50, 57, 58, 59, 60] and real.shape == pred.shape == (10, 64, 64) and pred.dtype == np.uint8
    assert P.pick(tick) == [1, 59, 60] and P.pick(list(range(1, 65)) + [100, 1000, 59999, 60000]) == [1, 100, 1000, 59999, 60000]


@pytest.mark.parametrize("sets, config", [(["online.pred_score=1"], "W_discs.json"), (["online.pred_score=1", "online.ring=0"], "W1_discs.json"),
                                          (["online.pred_every=10", "online.pred_record=4"], "W1_discs.json"),
                                          (["online.pred_score=1", "online.mix=0.5"], "W1_discs.json"),
                                          (["online.pred_score=1", "online.sf_windows=2"], "W1_discs.json")],
                         ids=["four transformers", "no ring", "pred_every alone", "the mix", "self-forcing"])
def test_what_it_is_refused_with(tmp_path, sets, config):
    r = life(tmp_path / "run", ["online.steps=8", "online.log_every=8"] + sets, config=config, ok=False)
    assert "ValueError" in r.stderr


def test_the_exam_as_a_blind_life_saw_it(lives):
    frame = np.full((2, 64, 64), 77, np.uint8)
    body, move = np.zeros((2, 2), np.float32), np.zeros((2, 3), np.float32)
    m = W.Pieces(lives["learns"], 48, "pair", lived=True)
    assert m.blind == [0, 0, 64, 64]
    m.reset(2)
    m.observe(frame, body, move)
    assert (m.fr[-1] == 0).all() and (frame == 77).all()            # black in what the model is given; the caller's frame untouched
    m.close()
    m = W.Pieces(lives["learns"], 48, "pair")                        # without it the exam's frames are shown whole
    m.reset(2)
    m.observe(frame, body, move)
    assert (m.fr[-1] == 77).all() and m.blind == []
    m.close()
    with pytest.raises(ValueError):
        W.Pieces(lives["first"], 0, "pair", lived=True)              # a plain life: nothing to show as lived


def test_the_tool(lives, tmp_path):
    run = lives["blind2"]                                            # learning off: the frozen weights are the life's
    a = P.after(run, ticks=40)
    t, e = np.array(a["tick"]), np.array(a["err"])
    assert t.min() <= -30 and t.max() >= 40 and a["back"] == -t.min() and os.path.exists(os.path.join(run, "wm", "after.json"))
    tick, real, pred = kept(run)
    own = ((pred.astype(np.float64) - real) ** 2).sum((1, 2))
    for back in (0, 5, 20):                                          # lined up with the life: the same picture, the same error
        assert e[t == -back][0] == pytest.approx(own[tick == 48 - back][0])
    text = P.table([run, lives["learns"]], at=(16, 48))
    w = P.whole(run, after=16)                                       # the life after its first rows as one
    rows = [r for r in P.metrics(run) if r["step"] > 16]
    assert w["rows"] == 2 and w["pred_err"] == pytest.approx(np.mean([r["pred_err"] for r in rows])) and w["beats"] in (0, 1, 2)
    assert P.whole(run, after=10 ** 6) is None and "| blind2 | all after 1k |" not in text   # a life of 48 ticks has no row after 1k
    assert "| blind2 | 0k |" in text and "### After the life" in text and "| blind2 | 1 to 8 |" in text
    assert P.title_of(run) == "blind2: the discs, with replay"
    for x in text.split("\n"):
        if x.startswith("| blind2 | 0k"):
            assert x.count("|") == 8
    assert P.rms(4096.0 * 9, 4096) == 3.0 and P.skill(1.0, 4.0) == 0.75 and P.f2s(-3.412) == "−3.41" and P.f2s(0.5) == ".50"
    from PIL import Image
    png = str(tmp_path / "strip.png")
    assert P.strip(run, png) == [1, 47, 48] and Image.open(png).width <= 960
    png = str(tmp_path / "curves.png")
    P.curves([run, lives["learns"]], png)
    assert Image.open(png).width <= 960 and os.path.getsize(png) <= 200 * 1024
    mp4 = str(tmp_path / "v.mp4")
    assert P.video([run, lives["learns"]], mp4) == 48 + 5 * 9 and os.path.getsize(mp4) > 1000


def test_the_courts_paddle_and_ball_in_the_pictures(lives, tmp_path):
    """Pictures made by hand from a court's real frames: the paddle 3 px to the right and no ball; then no paddle."""
    from world import make_config, make_world
    cfg = json.load(open(os.path.join(DEV, "configs", "W1_breakout.json")))
    c = make_config(cfg["env"])
    w = make_world(1, c, seed=3)
    real = []
    for _ in range(1300):
        real.append(w.step(np.zeros((1, 3)))[0]["image"][0, ..., 0].copy())
    real = np.array(real)
    pred = real.copy()
    pred[:, :c.paddle_y] = np.where(pred[:, :c.paddle_y] > 200, c.bg, pred[:, :c.paddle_y])     # the ball painted out
    pred[:, c.paddle_y:c.paddle_y + c.paddle_h, c.wall + 3:64 - c.wall] = real[:, c.paddle_y:c.paddle_y + c.paddle_h, c.wall:64 - c.wall - 3]
    pred[:, c.paddle_y:c.paddle_y + c.paddle_h, c.wall:c.wall + 3] = c.bg
    run = tmp_path / "run"
    os.makedirs(run)
    json.dump(dict(config=cfg, seed=3), open(run / "info.json", "w"))
    open(run / "metrics.jsonl", "w").close()                           # a life with no row: the table has the court's block alone
    np.savez(run / "pred_frames.npz", tick=np.arange(1, 1301), real=real, pred=pred)
    t = P.court_things(str(run))
    moved = 64 - 2 * c.wall - c.paddle_w - 3                           # a paddle at the right wall cannot move 3 px: it is cut
    assert t["n"] == 301 and t["paddle"] == 1.0 and t["ball"] == 0.0 and t["within6"] == 1.0
    assert 1.5 <= t["off"] <= 3.0 and t["still"] > t["off"] and moved > 0
    np.savez(run / "pred_frames.npz", tick=np.arange(1, 1301), real=real, pred=np.where(real > 150, c.bg, real).astype(np.uint8))
    t = P.court_things(str(run))
    assert t["paddle"] == 0.0 and t["off"] != t["off"] and t["ball"] == 0.0     # no paddle painted: no place to be off
    np.savez(run / "pred_frames.npz", tick=np.arange(1, 1301), real=real, pred=real)
    t = P.court_things(str(run))
    assert t["off"] == 0.0 and t["within2"] == 1.0 and t["ball"] > 0.9              # the real frames as pictures
    assert P.court_things(lives["learns"]) is None                                 # the disc world: not the court
    assert P.title_of(str(run)) == "run: the court, with replay"
    cfg["online"]["batch"], cfg["ppo"]["lr"] = 1, 0.003
    json.dump(dict(config=cfg, seed=3), open(run / "info.json", "w"))
    assert P.title_of(str(run)) == "run: the court, no replay, learning rate x10"
    assert "### The court: the paddle and the ball" in P.table([str(run)]) and P.f3(0.4567) == ".457" and P.f3(1.0) == "1.000"

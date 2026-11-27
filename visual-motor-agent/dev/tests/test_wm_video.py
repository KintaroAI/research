"""Task 00134: the video of a watching model's dreams beside the world (wm_video.py). Predictors made by hand stand
in for a model, so no card is needed: the world itself must show the world, the frames must come in order (the watched
ticks, then the dream), and the file must be written."""
import json
import os
import sys

import numpy as np
import pytest

DEV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, DEV)

import wm_measures as W  # noqa: E402
import wm_video as V  # noqa: E402


def env_of(name):
    return json.load(open(os.path.join(DEV, "configs", name)))["env"]


@pytest.mark.parametrize("name", ["W_discs.json", "W_breakout.json"])
def test_the_frames_of_a_video(name, tmp_path):
    ex = W.Exam(env_of(name), seed=5, copies=4, ticks=420, starts=(100, 180, 260, 340))
    e, s = V.stretches(ex, 3)
    assert len(set(zip(e.tolist(), s.tolist()))) == 3 and set(s.tolist()) <= set(ex.starts)   # fixed by the number asked for
    world, copy = W.dreams(W.TheWorld(), ex, e, s, 6)[0], W.dreams(W.Copy(), ex, e, s, 6)[0]
    dreamed = [("dream: the world", world), ("dream: copy", copy)]
    plain = list(V.frames(ex, e, s, dreamed, real=4, scale=2, hold=2, rims=False))
    assert len(plain) == 4 + 2 + 6 + 2 and all(f.shape == plain[0].shape and f.dtype == np.uint8 for f in plain)
    h, w, _ = plain[0].shape
    assert h % 8 == 0 and w % 8 == 0                                    # the encoder's block
    side, x = ex.S * 2, [V.PAD + c * (ex.S * 2 + V.GAP) for c in range(3)]
    y = V.PAD + V.HEAD + V.CAPTION
    panel = lambda f, c: f[y:y + side, x[c]:x[c] + side, 0]             # the first stretch's panel of column c  # noqa: E731
    up = lambda a: np.kron(np.asarray(a, np.uint8), np.ones((2, 2), np.uint8))   # noqa: E731
    for i in range(4):                                                  # watching: every column shows the real tick
        t = s[0] - 3 + i
        assert all((panel(plain[i], c) == up(ex.frames[e[0], t])).all() for c in range(3))
    last = plain[4 + 2 + 5]                                             # dreamed tick 6
    assert (panel(last, 0) == up(ex.frames[e[0], s[0] + 6])).all()      # the world's own continuation
    assert (panel(last, 1) == panel(last, 0)).all()                     # a dream that is the world
    assert (panel(last, 2) == up(ex.frames[e[0], s[0]])).all()          # the copy still shows the last real frame
    rims = list(V.frames(ex, e, s, dreamed, real=4, scale=2, hold=2))    # with the true places outlined on the dreams
    assert (rims[0] == plain[0]).all() and (rims[-1][y:y + side, x[0]:x[0] + side] == plain[-1][y:y + side, x[0]:x[0] + side]).all()
    assert (rims[-1][y:y + side, x[1]:x[1] + side] != plain[-1][y:y + side, x[1]:x[1] + side]).any()
    out = tmp_path / "v" / "test.mp4"
    assert V.write_mp4(str(out), rims, 6) == len(rims) and os.path.getsize(out) > 1000
    # task 00135, a long dream: the true places outlined on the first ticks only, and the frames taken one at a time
    some = list(V.frames(ex, e, s, dreamed, real=4, scale=2, hold=2, rims=2))
    box = lambda f, c: f[y:y + side, x[c]:x[c] + side]                  # noqa: E731
    at = lambda t: 4 + 2 + t - 1                                        # the frame of dreamed tick t  # noqa: E731
    assert all((box(some[at(t)], c) == box(rims[at(t)], c)).all() for t in (1, 2) for c in range(3))      # outlined
    assert all((box(some[at(t)], c) == box(plain[at(t)], c)).all() for t in (3, 6) for c in range(3))     # no longer
    assert (some[at(3)] != plain[at(3)]).any()                          # the caption says up to which tick
    none = list(V.frames(ex, e, s, dreamed, real=4, scale=2, hold=2, rims=0))
    assert all((a == b).all() for a, b in zip(none, plain))
    one_by_one = V.frames(ex, e, s, dreamed, real=4, scale=2, hold=2, rims=2)
    assert V.write_mp4(str(tmp_path / "v" / "long.mp4"), one_by_one, 12) == len(some) and next(one_by_one, None) is None


@pytest.mark.parametrize("name", ["W_discs.json", "W_breakout.json"])
def test_a_longer_exam_is_the_same_exam_continued(name):
    """A dream longer than the exam's needs more ticks of the world: the first ticks must stay the exam's own."""
    short = W.Exam(env_of(name), seed=5, copies=3, ticks=260, starts=(100, 180))
    long_ = W.Exam(env_of(name), seed=5, copies=3, ticks=420, starts=(100, 180))
    assert (long_.frames[:, :260] == short.frames).all() and (long_.body[:, :260] == short.body).all()
    assert all((long_.state[k][:, :260] == short.state[k]).all() for k in short.state)
    assert (long_.frames[:, 260:] != long_.frames[:, 259:260]).any()     # and the world goes on

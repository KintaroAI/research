"""Task 00138: the tables and the picture of a dream that rolls dice (wm_dice.py). Summaries written by hand stand in
for scored checkpoints, so no card is needed: a cell must be the right number of the right life and decoding, the
world's own row comes from the same file, and what is not scored is a dash."""
import json
import os
import sys

from PIL import Image

DEV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, DEV)

import wm_dice as D  # noqa: E402


def put(out, arm, decode, which, r, step=60000):
    d = os.path.join(str(out), "%s_s1" % arm, "wm")
    os.makedirs(d, exist_ok=True)
    json.dump(r, open(os.path.join(d, "step_%08d_%s_%s.json" % (step, decode, which)), "w"))


def free(served, at200, back, specks, last, real_served=12):
    ball = [1.0] * 100 + [at200] * 100
    return dict(world="breakout", free=dict(starts=32, balls=dict(ball=ball, served=served, back=back, specks=specks, last=last),
                                            balls_real=dict(ball=[1.0] * 200, served=real_served, back=0, specks=0.01, last=[200] * 32)))


def work(horizon, held8, skill):
    return dict(world="breakout", regular=dict(next=dict(skill=skill), hold=dict(ball=dict(held=[held8] * 15 + [held8 / 2] * 17, horizon=horizon),
                                                                                 paddle=dict(held=[1.0] * 32, horizon=30))))


def test_the_names():
    assert D.decode_name("pair") == "today's (`pair`)" and D.decode_name("draw0") == "the likeliest value, no dice"
    assert D.decode_name("draw") == "dice by chance, T 1" and D.decode_name("draw0.7") == "dice by chance, T .7"
    assert D.decode_name("drawp0.5") == "dice by the tallest bin, T .5" and D.decode_name("drawp") == "dice by the tallest bin, T 1"
    assert D.decode_name("mean") == "mean"
    assert [D.short_name(d) for d in ("pair", "draw0", "draw", "draw0.7", "drawp", "drawp0.5")] == ["today", "likeliest", "chance T 1", "chance T .7", "dice T 1", "dice T .5"]
    assert D.rolling([0, 0, 3, 0, 0], 3).tolist() == [0, 1, 1, 1, 0] and D.rolling([1, 2], 5).tolist() == [1.5, 1.5]


def test_the_tables_and_the_picture(tmp_path):
    out = tmp_path / "exp"
    put(out, "OB", "pair", "free", free(0, .25, 3, .04, [50, 150, 200]))
    put(out, "OB", "drawp0.5", "free", free(9, .875, 12, .11, [200, 200, 180]))
    put(out, "M7B", "pair", "free", free(0, .3125, 5, .04, [116]))
    put(out, "OB", "pair", "work", work(9, .61, .75))
    put(out, "OB", "drawp0.5", "work", work(8, .55, .74))
    text = D.tables(str(out), ["OB", "M7B", "MB"], ["pair", "draw", "drawp0.5"])
    lines = text.split("\n")
    assert lines[0] == "### The long dream: 200 ticks from 32 stretches of the exam"
    assert lines[2] == "| Measure | Decoding | OB: no self-forcing | M7B: .7 its own prediction | MB: its own prediction mixed in (.5) |"
    assert "| Balls served in the long dreams | today's (`pair`) | 0 | 0 | – |" in lines
    assert "| Balls served in the long dreams | dice by the tallest bin, T .5 | 9 | – | – |" in lines
    assert "| Balls served in the long dreams | the world itself | 12 | 12 | – |" in lines      # from the same files
    assert "| Dreams with a ball at tick 200 | dice by the tallest bin, T .5 | .875 | – | – |" in lines
    assert "| Dreams with a ball at tick 100 | today's (`pair`) | 1.000 | 1.000 | – |" in lines
    assert "| Stray bright pixels a frame | dice by the tallest bin, T .5 | 0.11 | – | – |" in lines
    assert "| The last ball, median tick | today's (`pair`) | 150 | 116 | – |" in lines
    assert not any("dice by chance" in x for x in lines)                                        # a decoding that is not scored: no row
    assert "### The short dreams: the regular part" in lines
    assert "| The ball held, dreamed ticks | dice by the tallest bin, T .5 | 8 | – | – |" in lines
    assert "| The ball held at tick 16 | today's (`pair`) | .305 | – | – |" in lines               # tick 16 is index 15
    assert "| The ball held at tick 8 | today's (`pair`) | .610 | – | – |" in lines
    assert "| Next-frame skill | today's (`pair`) | .750 | – | – |" in lines
    for i, x in enumerate(lines):                                                               # every row as wide as its header
        if x.startswith("| Measure"):
            j = i
            while j < len(lines) and lines[j].startswith("|"):
                assert lines[j].count("|") == x.count("|"), lines[j]
                j += 1
    assert D.tables(str(out), ["XX"], ["pair"]) == ""
    png = str(tmp_path / "images" / "balls.png")
    assert D.figure(str(out), ["OB", "M7B"], ["pair", "drawp0.5"], png) == png
    im = Image.open(png)
    assert im.width <= 960 and os.path.getsize(png) <= 200 * 1024
    try:
        D.figure(str(out), ["OB"], ["pair", "draw", "draw0.7", "drawp"], png)
        raise AssertionError("four decodings in one picture")
    except ValueError:
        pass

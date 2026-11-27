"""Task 00135: a life's numbers by its ticks (wm_curves.py). Summaries written by hand stand in for scored lives, so no
card is needed: a cell must be the seeds' mean of the right number at the right tick, a tick or a set that is not
scored must not be invented, and the tables must stay tables."""
import json
import os
import sys

import pytest
from PIL import Image

DEV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, DEV)

import wm_curves as C  # noqa: E402


def summary(world, skill, held, horizon, travel=None):
    """A scored checkpoint with the keys the curves read: `held` at every dreamed tick but tick 16, which is half."""
    thing, other, straight = ("ball", "paddle", "ball") if world == "breakout" else ("food", "poison", "poison")
    r = dict(world=world, regular=dict(next=dict(skill=skill), curve=dict(skill_copy=[skill / 2] * 32),
                                       hold={thing: dict(held=[held] * 15 + [held / 2] + [0.0] * 16, horizon=horizon),
                                             other: dict(held=[held] * 32, horizon=horizon + 2)}))
    if travel is not None:
        r["course"] = {straight: dict(travel=[travel] * 16)}
    return r


def put(out, arm, seed, step, which, r):
    d = os.path.join(str(out), "%s_s%d" % (arm, seed), "wm")
    os.makedirs(d, exist_ok=True)
    json.dump(r, open(os.path.join(d, "step_%08d_pair_%s.json" % (step, which)), "w"))


@pytest.fixture
def lives(tmp_path):
    """A court with two arms and two seeds, the second seed scored at 20k only; a disc world with one arm, scored at
    home at 40k only; nothing at 60k."""
    out = tmp_path / "exp"
    put(out, "OB", 1, 20000, "work", summary("breakout", .60, .40, 6, .90))
    put(out, "OB", 2, 20000, "work", summary("breakout", .70, .60, 8, .80))
    put(out, "OB", 1, 40000, "work", summary("breakout", .75, .50, 9, .95))
    put(out, "SB3", 1, 20000, "work", summary("breakout", .62, .70, 10, .92))
    put(out, "SB3", 1, 40000, "work", summary("breakout", .72, .90, 15))          # no course part
    put(out, "OD", 1, 20000, "work", summary("watch", .30, .50, 4, .70))
    put(out, "OD", 1, 40000, "work", summary("watch", .35, .60, 5, .75))
    put(out, "OD", 1, 40000, "home64", summary("watch", .65, .80, 7, .85))
    return str(out)


STEPS = [20000, 40000, 60000]
NEXT, HELD16, WAY = ("regular", "next", "skill"), ("regular", "hold", "ball", "held", 15), ("course", "ball", "travel", 7)


def test_a_number_by_tick(lives):
    v = C.series(lives, "OB", [1, 2], STEPS, "work", NEXT)
    assert v[0] == pytest.approx(.65) and v[1] == pytest.approx(.75) and v[2] != v[2]     # two seeds, one, none
    assert C.series(lives, "OB", [1], STEPS, "work", HELD16)[:2] == pytest.approx([.20, .25])   # tick 16 is index 15
    w = C.series(lives, "SB3", [1], STEPS, "work", WAY)
    assert w[0] == pytest.approx(.92) and w[1] != w[1]                    # a part that was not scored is no number
    assert all(x != x for x in C.series(lives, "OB", [1], STEPS, "home64", NEXT))         # the court has no home set
    assert C.series(lives, "OD", [1], STEPS, "home64", NEXT)[1] == pytest.approx(.65)
    assert C.worlds(lives, ["OD", "OB", "SB3", "XX"], [1, 2], STEPS) == dict(watch=["OD"], breakout=["OB", "SB3"])


def test_the_names_and_the_signs():
    assert C.name_of("OD") == "no self-forcing" and C.name_of("SB3") == "up to 3 dreamed steps" and C.name_of("WD") == "WD"
    assert C.signed(C.f3, .0916) == "+.092" and C.signed(C.f3, -.016) == "−.016" and C.signed(C.f3, .0001) == ".000"
    assert C.signed(C.f1, 6.0) == "+6.0" and C.signed(C.f1, 0.0) == "0.0" and C.signed(C.f3, float("nan")) == "–"


def test_the_tables(lives):
    text = C.tables(lives, ["OD", "OB", "SB3"], [1, 2], STEPS)
    lines = text.split("\n")
    heads = [x for x in lines if x.startswith("### ")]
    assert heads == ["### The moving discs, the exam", "### The moving discs, the life's own world, after its end", "### The court, the exam"]
    court = lines[lines.index(heads[2]):]
    assert court[2] == "| Measure | Arm | 20k | 40k |"                        # 60k is not scored: no column for it
    assert "| Next-frame skill | OB: no self-forcing | .650 | .750 |" in court
    assert "| Next-frame skill | SB3: up to 3 dreamed steps | .620 | .720 |" in court
    assert "| Next-frame skill | SB3 − OB | −.030 | −.030 |" in court         # the second arm named less the first
    assert "| Horizon of the ball, ticks | SB3 − OB | +3.0 | +6.0 |" in court
    assert "| The ball's way gone at tick 8 | SB3: up to 3 dreamed steps | .920 | – |" in court
    assert "| The ball's way gone at tick 8 | SB3 − OB | +.070 | – |" in court
    home = lines[lines.index(heads[1]):lines.index(heads[2])]
    assert home[2] == "| Measure | Arm | 40k |" and "| The food held at tick 4 | OD: no self-forcing | .800 |" in home
    assert not any("−" in x and "OD" in x for x in home)                      # one arm: no difference row
    for i, x in enumerate(lines):                                             # every row as wide as its header
        if x.startswith("| Measure"):
            j = i
            while j < len(lines) and lines[j].startswith("|"):
                assert lines[j].count("|") == x.count("|"), lines[j]
                j += 1
    assert C.tables(lives, ["XX"], [1], STEPS) == ""


def test_the_picture_and_the_command(lives, tmp_path, monkeypatch, capsys):
    json.dump(dict(copy=dict(hold_seen={"8": dict(ball=dict(horizon=11, held=[.5] * 32))})), open(os.path.join(lives, "anchors_breakout_work.json"), "w"))
    png = str(tmp_path / "images" / "curves.png")
    assert C.figure(lives, ["OD", "OB", "SB3"], [1, 2], STEPS, png) == png
    im = Image.open(png)
    assert im.width <= 960 and os.path.getsize(png) <= 200 * 1024 and im.height > 400      # a row a world: two rows
    one = str(tmp_path / "images" / "court.png")
    C.figure(lives, ["OB", "SB3"], [1], STEPS, one)
    assert Image.open(one).height < im.height
    monkeypatch.setattr(sys, "argv", ["wm_curves.py", lives, "--arms", "OB", "SB3", "--seeds", "1", "--steps", "20000", "40000"])
    C.main()
    said = capsys.readouterr().out
    assert "### The court, the exam" in said and "| Next-frame skill | OB: no self-forcing | .600 | .750 |" in said and "moving discs" not in said

"""Action repeat (task 00131, online.act_every): a move is decided every act_every ticks and made again on the ticks
between. lifetime.hold_move by hand; a short life logs the share of held moves, (k - 1) / k; the default (1) logs
nothing of it; the refusals.

The lives require libvmcuda.so and a GPU.
"""
import json
import os
import subprocess
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import cuda_lib  # noqa: E402
from lifetime import hold_move  # noqa: E402
from online import DEFAULT_ONLINE  # noqa: E402

gpu = pytest.mark.skipif(not cuda_lib.available(), reason="needs libvmcuda.so and a GPU")
DEV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")


def test_the_default_decides_every_tick():
    assert DEFAULT_ONLINE["act_every"] == 1


def test_hold_move():
    rng = np.random.default_rng(0)
    new = lambda: (rng.uniform(-1, 1, (3, 3)).astype(np.float32), rng.integers(0, 4, (3, 2)))   # noqa: E731
    held, made = None, []
    for t in range(9):
        a, won = new()
        m, w, held, decided = hold_move(a, won, held, t % 4 == 0)
        assert decided == (t % 4 == 0)
        if decided:
            assert m is a and w is won                                   # the move just chosen, untouched
            a[:] = 9.0                                                   # what is kept is a copy
            assert not (held[0] == 9.0).any()
        else:
            assert (m == made[-1][0]).all() and (w == made[-1][1]).all()   # the kept move again, not the one chosen now
            assert m is not held[0]                                      # and a copy: the caller may change it
        made.append((m.copy() if not decided else held[0].copy(), w.copy()))
    assert all((made[t][0] == made[t - t % 4][0]).all() for t in range(9))
    a, won = new()
    m, w, held, decided = hold_move(a[:2], won[:2], held, False)         # another number of bodies: this tick decides
    assert decided and m.shape == (2, 3)
    assert hold_move(a, won, None, False)[3]                             # nothing kept yet: this tick decides


def life(tmp_path, *sets, world="U_breakout.json"):
    cmd = [sys.executable, os.path.join(DEV, "lifetime.py"), "--config", os.path.join(DEV, "configs", world),
           "--set", "online.steps=40", "--set", "online.log_every=20", "--set", "online.save_every=100000",
           "--seed", "1", "-o", str(tmp_path / "run")]
    for s in sets:
        cmd += ["--set", s]
    return subprocess.run(cmd, cwd=DEV, capture_output=True, text=True)


@gpu
@pytest.mark.parametrize("world", ["U_breakout.json", "U_agent.json"])
def test_a_life_holds_its_moves(tmp_path, world):
    r = life(tmp_path, "online.act_every=4", world=world)
    assert r.returncode == 0, r.stderr[-2000:]
    rows = [json.loads(line) for line in open(tmp_path / "run" / "metrics.jsonl")]
    assert len(rows) == 2
    assert all(row["held"] == 0.75 for row in rows)                      # ticks 1, 5, 9, ... decide, the three between hold


@gpu
def test_the_default_life_logs_no_hold(tmp_path):
    r = life(tmp_path)
    assert r.returncode == 0, r.stderr[-2000:]
    assert all("held" not in json.loads(line) for line in open(tmp_path / "run" / "metrics.jsonl"))


@gpu
@pytest.mark.parametrize("bad", [["online.act_every=0"], ["online.act_every=4", "online.actor_pg=1", "online.piece_rule=random"],
                                 ["online.act_every=4", "online.instinct_h=4", "online.ring=0"],
                                 ["online.act_every=4", "online.label=", "model.body_label=0", "online.full_dose=0", "online.label_refresh=0"]],
                         ids=["zero", "actor", "horizon", "no_label"])
def test_the_refusals(tmp_path, bad):
    r = life(tmp_path, *bad)
    assert r.returncode != 0 and "act_every" in r.stderr

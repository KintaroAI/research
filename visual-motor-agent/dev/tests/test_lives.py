"""The helpers of analyze_lives.py (tasks 00128, 00130, 00131): windows of ticks from the logged rows. CPU only."""
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import analyze_lives as L  # noqa: E402


def write(path, rows):
    os.makedirs(path, exist_ok=True)
    with open(os.path.join(path, "metrics.jsonl"), "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return str(path)


def court(path, per_k):
    """A Breakout run: per_k[i] = (hits, misses) of the i-th 1,000 ticks; the logged counters are cumulative."""
    h = m = 0
    rows = []
    for i, (dh, dm) in enumerate(per_k):
        h, m = h + dh, m + dm
        rows.append(dict(step=(i + 1) * 1000, bo_hits=h, bo_misses=m))
    return write(path, rows)


def test_share_is_taken_from_the_cumulative_counters(tmp_path):
    p = court(tmp_path / "a", [(1, 9), (2, 8), (5, 5), (9, 1)])
    assert L.share(p, 0, 1000) == 0.1
    assert L.share(p, 1000, 3000) == 7 / 20          # ticks 1001 .. 3000
    assert L.share(p, 0, 4000) == 17 / 40
    assert L.share(p, 3000, 4000) == 0.9


def test_share_of_a_window_without_balls_is_not_a_number(tmp_path):
    p = court(tmp_path / "a", [(0, 0), (3, 1)])
    assert math.isnan(L.share(p, 0, 1000))
    assert L.share(p, 0, 2000) == 0.75


def test_food_and_poison_are_per_1000_body_ticks(tmp_path):
    p = write(tmp_path / "a", [dict(step=100 * (i + 1), step_hits=i, poison=1, bodies=1) for i in range(20)])
    assert L.food(p, 0, 2000) == 1000 * sum(range(20)) / 2000
    assert L.food(p, 1000, 2000) == 1000 * sum(range(10, 20)) / 1000
    assert L.poison(p, 0, 2000) == 10.0
    q = write(tmp_path / "b", [dict(step=1000, step_hits=8, poison=4, bodies=4)])
    assert L.food(q, 0, 1000) == 2.0 and L.poison(q, 0, 1000) == 1.0


def test_first_window_is_the_first_5k_window_at_the_level(tmp_path):
    p = court(tmp_path / "a", [(2, 8)] * 7 + [(8, 2)] * 5)                # .2 for 7k ticks, then .8
    value = lambda a, b: L.share(p, a, b)                                 # noqa: E731
    assert L.first_window(value, .5, 12000) == 10000                      # ticks 5001 .. 10000: 2 of 5 low, 3 high = .56
    assert L.first_window(value, .8, 12000) == 12000
    assert L.first_window(value, .9, 12000) is None
    assert L.first_window(value, .2, 12000) == 5000


def test_the_formats():
    assert L.f3(.678) == ".678" and L.f3(-.092) == "−.092" and L.f3(1.0) == "1.000" and L.f3(-.0001) == ".000"
    assert L.f1(71.64) == "71.6" and L.f1(-2.35) in ("−2.4", "−2.3") and L.f1(-.01) == "0.0"
    assert L.f2(-.004) == "0.00" and L.f2(-.13) == "−0.13"
    assert L.signed(L.f3, -.3941, .0752) == "−.394 ± .075" and L.signed(L.f1, 19.3, 5.3) == "+19.3 ± 5.3"
    assert L.win(40000, 60000) == "40–60k"
    assert L.ticks([32000, None, 55000, 41000], 4) == "32–55k (median 41k), 3 of 4"
    assert L.ticks([None, None], 2) == "none"


def test_diff_is_the_paired_mean_with_its_95_interval_and_leaves_out_missing_values():
    a, b = [.5, .6, .7, float("nan")], [.4, .4, .4, .4]
    cell, seeds = L.diff(a, b)
    d = np.array([.1, .2, .3])
    half = 4.303 * d.std(ddof=1) / np.sqrt(3)
    assert cell == L.signed(L.f3, d.mean(), half) and seeds == "3 of 3 above"


def test_keep_seeds_names_the_unfinished(tmp_path, capsys):
    for s, n in ((1, 60), (2, 60), (3, 30)):
        court(tmp_path / ("X_s%d" % s), [(1, 1)] * n)
    good = L.keep_seeds([1, 2, 3, 4], [(str(tmp_path / "X_s%d"), 60000)])
    assert good == [1, 2]
    assert "3, 4" in capsys.readouterr().out

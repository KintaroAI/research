"""Task 00117 (vm-00104): online.meal_size, a meal worth its size. Off, the labels are exactly the old ones; on, a meal is a
step whose full change (rise + the signed cut) is negative, worth min(1, relief / size), in meal_parts, stored_labels and
MealMonitor alike; poison steps are unchanged."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from online import MealMonitor, meal_parts, meal_target, stored_labels  # noqa: E402

G = 0.95
#             drift   meal .25 at .6   meal .5 at .1 (floor cut .399)   poison .5 at .8 (cap cut .301)   drift
H0 = np.array([0.300, 0.600, 0.100, 0.800, 0.0])
H1 = np.array([0.301, 0.351, 0.000, 1.000, 0.001])
CUT = np.array([0.0, 0.0, -0.399, 0.301, 0.0])
DONE = np.zeros(5, bool)


def test_off_is_the_old_label():
    a = meal_parts(H0, H1, DONE, G, drift=False, cut=CUT)
    b = meal_parts(H0, H1, DONE, G, drift=False, cut=CUT, size=0.0)
    for x, y in zip(a, b):
        assert (x == y).all()
    assert a[0][1] == 1.0 and a[0][2] == 1.0 and np.isclose(a[0][3], -0.501)


def test_sized_parts():
    real, tw = meal_parts(H0, H1, DONE, G, drift=False, cut=CUT, size=0.5)
    assert np.isclose(real[1], 0.249 / 0.5) and tw[1] == 0.0
    assert np.isclose(real[2], 0.499 / 0.5) and tw[2] == 0.0         # the floor's cut counted: the full .5 (less the drift)
    assert np.isclose(real[3], -0.501) and tw[3] == np.float32(G)      # the poison as before
    assert real[0] == 0.0 and real[4] == 0.0                           # the drift costs nothing (drift=False)
    big = meal_parts(np.array([0.9, 0.9]), np.array([0.0, 0.0]), np.zeros(2, bool), G, drift=False, cut=np.zeros(2), size=0.5)[0]
    assert big[0] == 1.0                                               # capped at 1


def test_a_meal_the_floor_hid_is_still_a_meal():
    # hunger 0 before and after (fed the tick before), food .25: the plain label sees no meal, the sized one does
    real = meal_parts(np.array([0.0, 0.0]), np.array([0.0, 0.0]), np.zeros(2, bool), G, drift=False, cut=np.array([0.001 - 0.25, 0.0]), size=0.5)[0]
    assert np.isclose(real[0], 0.249 / 0.5) and real[1] == 0.0


def test_stored_labels_match_the_parts():
    nxt = np.array([0.3, 0.7])
    for i in range(5):
        real, tw = meal_parts(H0[i:i + 2], H1[i:i + 2], DONE[i:i + 2], G, drift=False, cut=CUT[i:i + 2], size=0.5)
        want = meal_target(np.full(2, real[0]), np.full(2, tw[0] if i < 4 else G), nxt)
        got = stored_labels(H0[i], H1[i], False, nxt, G, CUT[i], size=0.5)
        assert np.allclose(got, want), i
    assert (stored_labels(0.6, 0.351, False, nxt, G, 0.0) == 0).all()   # off: a meal is T = 0 (V = 1)


def test_monitor_counts_the_size():
    m = MealMonitor(1, G, drift=False, size=0.5)
    m.add(np.array([0.5]), np.array([0.3]), np.array([0.301]), np.array([0.0]))     # one tick before the meal
    m.add(np.array([0.5]), np.array([0.6]), np.array([0.351]), np.array([0.0]))     # a .25 meal
    out = m.take()
    assert np.isclose(out["real"][0], 0.498) and np.isclose(out["real"][1], G * 0.498)

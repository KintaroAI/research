"""Task 00118 (vm-00105): the blind spot (online.blind) and eval_blind's measures."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from eval_blind import background, measure, runs_since  # noqa: E402
from online import DEFAULT_ONLINE, blind_frames  # noqa: E402

GAP = (38, 16, 12, 12)


def test_defaults_off():
    assert DEFAULT_ONLINE["blind"] == [] and DEFAULT_ONLINE["pred_record"] == 0


def test_blind_frames_is_a_copy_with_the_box_black():
    f = np.full((2, 3, 64, 64, 3), 77, np.uint8)
    g = blind_frames(f, GAP)
    assert (f == 77).all()                                        # the original (memory's frame) untouched
    assert (g[..., 16:28, 38:50, :] == 0).all()
    g[..., 16:28, 38:50, :] = 77
    assert (g == 77).all()                                        # nothing else changed


def test_runs_since_and_background():
    assert runs_since(np.array([0, 1, 1, 0, 1], bool)).tolist() == [0, 1, 2, 0, 1]
    f = np.full((1, 8, 8), 20, np.uint8)
    f[0, :2, :2] = 200
    assert background(f).tolist() == [20.0]


def test_measure_perfect_and_blank(tmp_path):
    rng = np.random.default_rng(0)
    y = np.full((60, 64, 64), 20, np.uint8)
    for t in range(30, 50):                                       # an object in the gap for 20 ticks
        y[t, 20:24, 40:44] = 200
    y[:, 0, :] = rng.integers(0, 255, (60, 64))                   # something outside the gap
    for name, p in (("perfect", y), ("blank", np.full_like(y, 20))):
        d = tmp_path / name
        d.mkdir()
        np.save(d / "pred_frames.npy", p)
        np.save(d / "pred_truth.npy", y)
    a, b = measure(str(tmp_path / "perfect"), GAP, skip=0), measure(str(tmp_path / "blank"), GAP, skip=0)
    assert a["gap"] == 0.0 and a["hit"] == 1.0 and a["fa"] == 0.0
    assert np.isclose(b["gap"], b["gap_bg"]) and b["hit"] == 0.0
    assert a["since"]["16+"][1] == 5 and a["since"]["1"][1] == 1


def test_gap_teacher_fills_a_still_scene_exactly():
    """A still world seen through a moving 64 x 64 window: once the body has moved, the gap's target is the world."""
    from online import GapTeacher
    S, V = 128, 64
    rng = np.random.default_rng(1)
    world = rng.integers(0, 256, (S, S), dtype=np.uint8)
    t = GapTeacher(1, S, V, GAP)

    def view(c):
        cy, cx = t.cells(np.array([c], np.float64))
        return world[cy[0], cx[0]][None]

    c = np.array([40.3, 70.6])
    first = t.target(view(c), [c])
    assert (first[0][~t.gap] == view(c)[0][~t.gap]).all()        # the visible part kept as seen
    for step in ([3.0, 0.0], [3.0, 0.0], [3.0, 0.0], [3.0, 0.0], [0.0, -2.5], [0.0, -2.5], [0.0, -2.5], [0.0, -2.5], [0.0, -2.5], [0.0, -2.5]):
        c = c + step
        out = t.target(view(c), [c])
    assert (out == view(c)).all()                                 # the gap filled from what was seen before
    assert t.take()["teach_filled"] > 0


def test_gap_teacher_never_reads_the_gap():
    from online import GapTeacher
    t = GapTeacher(1, 128, 64, GAP)
    f = np.full((1, 64, 64), 20, np.uint8)
    f[0, 16:28, 38:50] = 255                                      # what is in the gap must not reach the canvas
    out = t.target(f, [[64.0, 64.0]])
    assert (out[0, 16:28, 38:50] == 20).all() and not (t.canvas == 255).any()

"""Clogged channels (task 00085): the damage of online.clog: its share, long and brief masks, its two kinds, a copy with the original untouched, each draw its own."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from online import ClogStream, clog_frames, clog_masks, damage  # noqa: E402


def frames(n=4, T=8, H=64):
    return np.full((n, T, H, H, 3), 40, np.uint8)


def patches(a, ps=8):
    """[n, T, g, g] a patch differs from the plain frame."""
    n, T, H = a.shape[:3]
    g = H // ps
    return (a[..., 0] != 40).reshape(n, T, g, ps, g, ps).any((3, 5))


def test_nothing_at_p_0():
    f = frames()
    assert np.array_equal(clog_frames(f, np.random.default_rng(0), 8, 0.0), f)


def test_the_share_of_patches():
    hit, _ = clog_masks(np.random.default_rng(1), 400, 8, 8, 8, 0.3, 0.5)
    assert abs(hit.mean() - 0.3) < 0.015


def test_long_masks_last_the_whole_window_and_brief_ones_do_not():
    rng = np.random.default_rng(2)
    a, _ = clog_masks(rng, 200, 8, 8, 8, 0.2, 1.0)
    assert np.all(a == a[:, :1])
    b, _ = clog_masks(rng, 200, 8, 8, 8, 0.2, 0.0)
    assert not np.all(b == b[:, :1]) and abs(b.mean() - 0.2) < 0.02


def test_the_original_stays_clean():
    f, g = frames(), frames()
    out = clog_frames(f, np.random.default_rng(3), 8, 0.5)
    assert np.array_equal(f, g) and not np.array_equal(out, f)


def test_black_and_speckle():
    f = frames(1, 1)
    hit = np.ones((1, 1, 8, 8), bool)
    rng = np.random.default_rng(4)
    assert (damage(f, hit, np.ones((1, 1, 8, 8), bool), 8, rng) == 0).all()
    out = damage(f, hit, np.zeros((1, 1, 8, 8), bool), 8, rng)
    assert 0.4 < (out[..., 0] != 40).mean() < 0.6
    assert np.array_equal(out[..., 0], out[..., 2])                      # grey stays grey


def test_damage_stays_in_its_patches():
    f = frames(1, 1)
    hit = np.zeros((1, 1, 8, 8), bool)
    hit[0, 0, 2, 3] = True
    out = damage(f, hit, np.ones((1, 1, 8, 8), bool), 8, np.random.default_rng(5))
    want = np.zeros((1, 1, 8, 8), bool)
    want[0, 0, 2, 3] = True
    assert np.array_equal(patches(out), want)


def test_two_transformers_get_different_damage():
    f, rng = frames(), np.random.default_rng(6)
    assert not np.array_equal(clog_frames(f, rng, 8, 0.3), clog_frames(f, rng, 8, 0.3))


def test_the_stream_keeps_a_long_mask_for_its_span():
    s = ClogStream(np.random.default_rng(7), 8, 64, 64, 8, 0.4, long_share=1.0, span=4)
    f = np.full((8, 64, 64, 3), 40, np.uint8)
    a = [patches(s(f)[:, None])[:, 0] for _ in range(5)]
    assert all(np.array_equal(a[0], x) for x in a[:4]) and not np.array_equal(a[0], a[4])

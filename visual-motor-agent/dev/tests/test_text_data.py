"""Task 00034: the classical reference on a lifetime's batches, and the universal model trained offline."""
import os
import sys

import numpy as np
import pytest

DEV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, DEV)


def test_lifetime_batches_are_a_lifetimes_windows():
    pytest.importorskip("torch")
    from reference.char_gpt import LifetimeBatches
    n, page, L, B = 100003, 64, 8, 8
    lb = LifetimeBatches(n, page, L, B, seed=1)
    p0 = lb.pos0
    for _ in range(400):
        t = lb.t
        st = lb.next()
        assert len(st) == B
        assert st[0] == (p0 + (t - L) * page) % n                         # the newest window ends at the stream
        pages = ((st[1:] - p0) % n) // page                               # (no wrap within 400 ticks here)
        assert ((st[1:] - p0) % n % page == 0).all()
        chunks = (t - 11) // L
        if chunks > 0:
            assert (pages % L == 0).all() and (pages // L < chunks).all()  # whole chunks, decided 11 pages ago
        else:
            assert (pages >= max(0, t - 256)).all() and (pages <= t - L).all()
    assert lb.t == 408


def test_offline_windows_are_consecutive_pages(tmp_path):
    from train_text_offline import windows
    from world import TextConfig, TextWorld
    p = tmp_path / "c.txt"
    p.write_text("abcdefghij klmnop qrstuvwxyz.\n" * 300)
    w = TextWorld(1, TextConfig(corpus=str(p), size=16, cell=2), seed=0)
    off = np.array([5, 1000, 7000])
    r = windows(w, off, 4)
    assert r["img"].shape == (3, 4, 16, 16, 3) and (r["es"][:, 0] == 1).all() and (r["es"][:, 1:] == 0).all()
    for b in range(3):
        for k in range(4):
            want = "".join(w.alphabet[c] for c in w.page_codes(np.array([off[b] + k * 64]))[0])
            assert w.read(r["img"][b, k][None])[0] == want
            assert (r["nimg"][b, k] == (r["img"][b, k + 1] if k < 3 else w.render(w.page_codes(
                np.array([off[b] + 4 * 64])))[0])).all()

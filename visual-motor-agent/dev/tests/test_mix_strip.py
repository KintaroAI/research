"""Task 00136: the picture of what a life with online.mix was shown (mix_strip.py), from a record written by hand."""
import os
import sys

import numpy as np
from PIL import Image

DEV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, DEV)

import mix_strip  # noqa: E402


def test_the_first_ticks_and_the_last(tmp_path):
    rng = np.random.default_rng(0)
    ticks = [1, 2, 3, 4, 97, 98, 99, 100]                              # online.mix_record = 4 in a life of 100 ticks
    real = rng.integers(0, 256, (8, 64, 64), dtype=np.uint8)
    pred = real.copy()
    pred[:4] = 255 - pred[:4]                                          # a newborn's prediction: wrong everywhere
    pred[4:, 10, 10:13] = 255 - real[4:, 10, 10:13]                    # late in life: three pixels off
    mask = np.zeros((8, 64, 64), bool)
    mask[:, :, ::2] = True                                             # every other column from the prediction
    seen = np.where(mask, pred, real)
    np.savez_compressed(tmp_path / "mix_frames.npz", tick=np.array(ticks), real=real, pred=pred, mask=mask, seen=seen)
    png = str(tmp_path / "images" / "seen.png")
    got = mix_strip.strip(str(tmp_path), png, ticks=3)
    assert [t for t, _ in got] == [1, 2, 3, 98, 99, 100]               # three of the first, the last three
    assert [n for _, n in got] == [2048] * 3 + [2] * 3                 # of columns 10, 11, 12 two are from the prediction
    im = Image.open(png)
    assert im.width <= 960 and os.path.getsize(png) <= 200 * 1024
    assert [t for t, _ in mix_strip.strip(str(tmp_path), png, ticks=9)] == ticks   # no more than there are

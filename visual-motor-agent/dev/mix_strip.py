"""Task 00136: what a life with online.mix was shown, as a picture (RUN/mix_frames.npz, written with
online.mix_record). For a few of the life's first ticks and a few of its last: the real frame, the model's prediction
of it, the frame as seen (each pixel from the prediction or from the real frame), and the pixels of the frame as seen
that are not the real ones.

    ./venv/bin/python mix_strip.py RUN -o PNG [--ticks 3] [--scale 2]
"""
import argparse
import os

import numpy as np
from PIL import Image, ImageDraw, ImageFont

BG, INK, MUTED, MARK = (252, 252, 251), (11, 11, 11), (82, 81, 78), (227, 73, 72)
PAD, GAP, LEFT, HEAD, FOOT = 8, 6, 118, 20, 18
ROWS = ("the real frame", "its prediction", "the frame as seen", "seen, not real")


def strip(run, path, ticks=3, scale=2):
    """Writes the picture; returns [(tick, pixels of the frame as seen that are not the real ones)] of the ticks shown."""
    from readme_images import FONT, fit
    z = np.load(os.path.join(run, "mix_frames.npz"))
    tick, real, pred, seen = z["tick"], z["real"], z["pred"], z["seen"]
    late = np.flatnonzero(np.diff(tick) > 1)                       # where the first ticks end and the last ones begin
    cut = int(late[0]) + 1 if len(late) else len(tick)
    show = list(range(min(ticks, cut))) + list(range(max(cut, len(tick) - ticks), len(tick)))
    S = real.shape[1] * scale
    w = 2 * PAD + LEFT + len(show) * S + (len(show) - 1) * GAP
    h = 2 * PAD + HEAD + len(ROWS) * S + (len(ROWS) - 1) * GAP + FOOT
    im = Image.new("RGB", (w, h), BG)
    d = ImageDraw.Draw(im)
    f = ImageFont.truetype(FONT, 12)
    out = []
    for r, name in enumerate(ROWS):
        d.text((PAD, PAD + HEAD + r * (S + GAP) + S // 2 - 7), name, font=f, fill=INK)
    for c, i in enumerate(show):
        x = PAD + LEFT + c * (S + GAP)
        d.text((x, PAD), "tick {:,}".format(int(tick[i])), font=f, fill=MUTED)
        off = seen[i] != real[i]
        out.append((int(tick[i]), int(off.sum())))
        marks = np.full(real[i].shape + (3,), 255, np.uint8)       # white, the pixels that are not the real ones marked
        marks[off] = MARK
        for r, a in enumerate((real[i], pred[i], seen[i], marks)):
            tile = Image.fromarray(a if a.ndim == 3 else np.repeat(a[..., None], 3, -1)).resize((S, S), Image.NEAREST)
            y = PAD + HEAD + r * (S + GAP)
            im.paste(tile, (x, y))
            d.rectangle((x - 1, y - 1, x + S, y + S), outline=(230, 229, 225))
        d.text((x, h - PAD - FOOT + 4), "{:,} px".format(int(off.sum())), font=f, fill=MUTED)
    fit(im, path)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("-o", "--path", required=True)
    ap.add_argument("--ticks", type=int, default=3, help="of the first ticks and of the last")
    ap.add_argument("--scale", type=int, default=2)
    a = ap.parse_args()
    for t, n in strip(a.run, a.path, a.ticks, a.scale):
        print("tick %6d: %4d pixels of the frame as seen are not the real ones" % (t, n))
    print("wrote", a.path)


if __name__ == "__main__":
    main()

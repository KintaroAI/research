"""Task 00115: a video of forecast-panel lives. Each run's panel_frames.npy (online.panel_record, body 0's composites as
the model saw them) is shown tick by tick beside the real frames its forecasts aimed at:

  as seen:  [P1 sensed           | P2 forecast of P1 now ]    real:  [P1 sensed  | P1 now ]
            [P3 forecast of +1   | P4 forecast of +2     ]           [P1 at +1   | P1 at +2]

(panel 2 at tick t is transformer 0's forecast from t - 1 of the frame now in panel 1; panel 3 transformer 1's of the
next one; panel 4 transformer 2's of the one after.) The last 2 ticks, whose real frames are not recorded, are left out.

    ./venv/bin/python panel_video.py RUN:LABEL [RUN:LABEL ...] -o OUT.mp4 [--scale 3] [--fps 8]
"""
import argparse
import os

import numpy as np
from PIL import Image, ImageDraw



def write_mp4(path, frames, size, fps):
    import imageio_ffmpeg
    gen = imageio_ffmpeg.write_frames(path, size, fps=fps, codec="libx264", quality=None, bitrate="4M", macro_block_size=8)
    gen.send(None)
    for im in frames:
        gen.send(np.ascontiguousarray(im).tobytes())
    gen.close()


def truth(seq):
    """seq [T, 2h, 2w, 3] composites -> [T - 2, 2h, 2w, 3]: the real frames panels 2-4 aimed at."""
    h, w = seq.shape[1] // 2, seq.shape[2] // 2
    p1 = seq[:, :h, :w]
    T = len(seq) - 2
    top = np.concatenate([p1[:T], p1[:T]], 2)
    bottom = np.concatenate([p1[1:T + 1], p1[2:T + 2]], 2)
    return np.concatenate([top, bottom], 1)


def tag(img, xy, text):
    d = ImageDraw.Draw(img)
    x, y = xy
    d.rectangle([x, y, x + 6 * len(text) + 4, y + 11], fill=(0, 0, 0))
    d.text((x + 2, y), text, fill=(255, 220, 120))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("runs", nargs="+", help="RUN_DIR:LABEL")
    ap.add_argument("-o", "--out", required=True)
    ap.add_argument("--scale", type=int, default=3)
    ap.add_argument("--fps", type=int, default=8)
    args = ap.parse_args()
    runs = []
    for r in args.runs:
        path, _, label = r.partition(":")
        seq = np.load(os.path.join(path, "panel_frames.npy"))
        runs.append((label or os.path.basename(path), seq[:len(seq) - 2], truth(seq)))
    T = min(len(s) for _, s, _ in runs)
    S = args.scale
    H, W = runs[0][1].shape[1:3]
    ph, pw = H * S // 2, W * S // 2
    head, gap = 18, 8
    row_h = head + H * S
    width, height = 2 * W * S + gap, len(runs) * row_h
    width, height = -(-width // 8) * 8, -(-height // 8) * 8

    def frame(t):
        img = Image.new("RGB", (width, height), (40, 40, 40))
        for i, (label, seen, real) in enumerate(runs):
            y = i * row_h
            d = ImageDraw.Draw(img)
            d.text((4, y + 3), "%s   tick %d   left: as the model sees it   right: the real frames the forecasts aim at" % (label, t + 1),
                   fill=(230, 230, 230))
            for j, a in enumerate((seen[t], real[t])):
                x = j * (W * S + gap)
                img.paste(Image.fromarray(a).resize((W * S, H * S), Image.NEAREST), (x, y + head))
                names = (("sensed", "forecast: now", "forecast: +1", "forecast: +2") if j == 0 else
                         ("sensed", "real: now", "real: +1", "real: +2"))
                for k, n in enumerate(names):
                    tag(img, (x + (k % 2) * pw + 3, y + head + (k // 2) * ph + 3), n)
        return np.asarray(img)

    write_mp4(args.out, (frame(t) for t in range(T)), size=(width, height), fps=args.fps)
    Image.fromarray(frame(T // 2)).save(os.path.splitext(args.out)[0] + "_still.png")
    print("wrote %s (%d ticks, %dx%d)" % (args.out, T, width, height))


if __name__ == "__main__":
    main()

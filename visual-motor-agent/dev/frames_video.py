"""Task 00111: a video of lives in any world from their frames.npy (online.frame_record: body 0's frames as the model saw
them), runs side by side, each scaled up, with a label and the tick.

    ./venv/bin/python frames_video.py RUN:LABEL [RUN:LABEL ...] -o OUT.mp4 [--scale 4] [--fps 15]
"""
import argparse
import os

import numpy as np
from PIL import Image, ImageDraw

from panel_video import write_mp4


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("runs", nargs="+", help="RUN_DIR:LABEL")
    ap.add_argument("-o", "--out", required=True)
    ap.add_argument("--scale", type=int, default=4)
    ap.add_argument("--fps", type=int, default=15)
    args = ap.parse_args()
    runs = []
    for r in args.runs:
        path, _, label = r.partition(":")
        runs.append((label or os.path.basename(path), np.load(os.path.join(path, "frames.npy"))))
    T = min(len(s) for _, s in runs)
    H, W = runs[0][1].shape[1:3]
    S, head, gap = args.scale, 18, 8
    width, height = len(runs) * (W * S + gap) - gap, head + H * S
    width, height = -(-width // 8) * 8, -(-height // 8) * 8

    def frame(t):
        img = Image.new("RGB", (width, height), (40, 40, 40))
        d = ImageDraw.Draw(img)
        for i, (label, seq) in enumerate(runs):
            x = i * (W * S + gap)
            d.text((x + 4, 3), "%s   tick %d" % (label, t + 1), fill=(230, 230, 230))
            img.paste(Image.fromarray(seq[t]).resize((W * S, H * S), Image.NEAREST), (x, head))
        return np.asarray(img)

    write_mp4(args.out, (frame(t) for t in range(T)), size=(width, height), fps=args.fps)
    Image.fromarray(frame(T // 2)).save(os.path.splitext(args.out)[0] + "_still.png")
    print("wrote %s (%d ticks, %dx%d)" % (args.out, T, width, height))


if __name__ == "__main__":
    main()

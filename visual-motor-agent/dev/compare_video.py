"""Frozen agents side by side (task 00059 on): each run, as RUN:LABEL or RUN:LABEL:RULE (RULE overrides the run's
online.piece_rule), forages from the same start for --steps ticks; writes compare.mp4, a grid of their views with path,
touches and counts, and ARM.mp4, forage_video.py's video of each.

Usage (from visual-motor-agent/dev):
    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 ./venv/bin/python compare_video.py RUN:QG RUN:QN --step 60000 -o OUT
"""
import argparse
import os
import sys
import types

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from forage_video import BG, FOOD, MUTED, POISON, TEXT, frame, rings, run, trail, view, write_mp4  # noqa: E402
from make_video import font  # noqa: E402

SIZE = (1280, 816)
SC, COLS = 5, 3                       # the 64 x 64 view at 320 x 320


def grid_size(cols):
    return max(SIZE[0], cols * 64 * SC + (cols + 1) * 24), SIZE[1]


def grid(recs, labels, metas, t, cols=COLS):
    size = grid_size(cols)
    im = Image.new("RGB", size, BG)
    d = ImageDraw.Draw(im)
    gap = (size[0] - cols * 64 * SC) // (cols + 1)
    d.text((gap, 8), "step %d / %d" % (t + 1, len(recs[0])), font=font(18, True), fill=TEXT)
    for i, (rec, label, meta) in enumerate(zip(recs, labels, metas)):
        x0 = gap + (i % cols) * (64 * SC + gap)
        y0 = 62 + (i // cols) * (64 * SC + 64)
        im.paste(view(rec[t]["img"]).resize((64 * SC, 64 * SC), Image.NEAREST), (x0, y0))
        ref = rec[t]["cursor"] if meta["ego"] else None
        trail(d, rec, t, x0, y0, SC, ref)
        rings(d, rec, t, x0, y0, SC, ref)
        if meta.get("piece_side") and rec[t]["won"] >= 0:   # the piece that made this move
            s, k = meta["piece_side"], rec[t]["won"]
            side = 64 * SC // s
            x, y = x0 + (k % s) * side, y0 + (k // s) * side
            d.rectangle([x + 1, y + 1, x + side - 2, y + side - 2], outline=(120, 170, 255), width=2)
        d.text((x0, y0 - 26), label, font=font(18, True), fill=TEXT)
        r = rec[t]
        d.text((x0, y0 + 64 * SC + 6), "food %d" % r["food"], font=font(18, True), fill=FOOD)
        d.text((x0 + 110, y0 + 64 * SC + 6), "poison %d" % r["poison"], font=font(18, True), fill=POISON)
        d.text((x0 + 230, y0 + 64 * SC + 8), "hunger %.2f" % r["hunger"], font=font(15), fill=MUTED)
    return im


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("runs", nargs="+", help="RUN:LABEL or RUN:LABEL:RULE")
    ap.add_argument("--step", type=int, default=60000)
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--worlds", type=int, default=8)
    ap.add_argument("--cols", type=int, default=COLS)
    ap.add_argument("-o", "--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    recs, labels, metas = [], [], []
    for r in a.runs:
        path, label, *rule = r.split(":")
        args = types.SimpleNamespace(step=a.step, steps=a.steps, worlds=a.worlds, horizon=0, decode="", rule=rule[0] if rule else "")
        rec, food, poison, meta = run(args, path)
        print("%-40s world 0: food %d poison %d; all %d worlds: food %.1f poison %.1f per %d ticks" % (
            label, food[0], poison[0], len(food), food.mean(), poison.mean(), a.steps), flush=True)
        recs.append(rec); labels.append(label); metas.append(meta)   # noqa: E702
        name = label.split()[0]
        write_mp4(os.path.join(a.out, name + ".mp4"), (frame(rec, t, label, meta) for t in range(len(rec))))
    write_mp4(os.path.join(a.out, "compare.mp4"), (grid(recs, labels, metas, t, a.cols) for t in range(a.steps)), grid_size(a.cols), "8M")


if __name__ == "__main__":
    main()

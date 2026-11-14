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
SC, COLS = 5, 3                       # a world pixel is 5 screen pixels: 64 x 64 at 320 x 320, 128 x 128 at 640 x 640


def grid_size(cols, side=64 * SC, rows=None, view_side=None):
    """rows (task 00102, a window and its map): the height fits the rows, instead of the fixed one."""
    h = SIZE[1] if rows is None else -(-(62 + rows * (view_side + 64)) // 8) * 8
    return max(SIZE[0], cols * side + (cols + 1) * 24), h


def size_of(metas, cols, n):
    vs = metas[0].get("size", 64) * SC
    return grid_size(cols, block_side(metas[0]), -(-n // cols) if "full" in metas[0] else None, vs)


MAP_GAP = 12


def block_side(meta):
    """Screen width of one run: its view, and beside a window on a larger world (task 00102) the whole world."""
    v = meta.get("size", 64) * SC
    return 2 * v + MAP_GAP if "full" in meta else v


def grid(recs, labels, metas, t, cols=COLS):
    vs = metas[0].get("size", 64)    # the view's side in world px
    view_side = vs * SC
    block = block_side(metas[0])
    size = size_of(metas, cols, len(recs))
    im = Image.new("RGB", size, BG)
    d = ImageDraw.Draw(im)
    gap = (size[0] - cols * block) // (cols + 1)
    d.text((gap, 8), "step %d / %d" % (t + 1, len(recs[0])), font=font(18, True), fill=TEXT)
    for i, (rec, label, meta) in enumerate(zip(recs, labels, metas)):
        x0 = gap + (i % cols) * (block + gap)
        y0 = 62 + (i // cols) * (view_side + 64)
        im.paste(view(rec[t]["img"]).resize((view_side, view_side), Image.NEAREST), (x0, y0))
        ref = rec[t]["cursor"] if meta["ego"] else None
        world = meta.get("world")
        trail(d, rec, t, x0, y0, SC, ref, size=vs, world=world)
        rings(d, rec, t, x0, y0, SC, ref, size=vs, world=world)
        if "full" in meta:   # the whole world, centered on the cursor like the view, the window outlined
            mx, msc = x0 + view_side + MAP_GAP, view_side / world
            im.paste(view(rec[t]["full"]).resize((view_side, view_side), Image.NEAREST), (mx, y0))
            trail(d, rec, t, mx, y0, msc, ref, size=world)
            m0, m1 = (world - vs) / 2 * msc, (world + vs) / 2 * msc
            d.rectangle([mx + m0, y0 + m0, mx + m1, y0 + m1], outline=TEXT, width=1)
            d.text((mx, y0 - 22), "whole world, unseen outside the box", font=font(15), fill=MUTED)
        if meta.get("piece_side") and rec[t]["won"] >= 0:   # the piece that made this move
            s, k = meta["piece_side"], rec[t]["won"]
            side = view_side // s
            x, y = x0 + (k % s) * side, y0 + (k // s) * side
            d.rectangle([x + 1, y + 1, x + side - 2, y + side - 2], outline=(120, 170, 255), width=2)
        d.text((x0, y0 - 26), label, font=font(18, True), fill=TEXT)
        r = rec[t]
        d.text((x0, y0 + view_side + 6), "food %d" % r["food"], font=font(18, True), fill=FOOD)
        d.text((x0 + 110, y0 + view_side + 6), "poison %d" % r["poison"], font=font(18, True), fill=POISON)
        d.text((x0 + 230, y0 + view_side + 8), "hunger %.2f" % r["hunger"], font=font(15), fill=MUTED)
    return im


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("runs", nargs="+", help="RUN:LABEL or RUN:LABEL:RULE")
    ap.add_argument("--step", type=int, default=60000)
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--worlds", type=int, default=8)
    ap.add_argument("--cols", type=int, default=COLS)
    ap.add_argument("--noise", type=float, help="env.noise for every run (task 00087); default: each run's own")
    ap.add_argument("-o", "--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    recs, labels, metas = [], [], []
    for r in a.runs:
        path, label, *rule = r.split(":")
        args = types.SimpleNamespace(step=a.step, steps=a.steps, worlds=a.worlds, horizon=0, decode="", rule=rule[0] if rule else "", noise=a.noise)
        rec, food, poison, meta = run(args, path)
        print("%-40s world 0: food %d poison %d; all %d worlds: food %.1f poison %.1f per %d ticks" % (
            label, food[0], poison[0], len(food), food.mean(), poison.mean(), a.steps), flush=True)
        if "full" in rec[0]:
            meta["full"] = True
        recs.append(rec); labels.append(label); metas.append(meta)   # noqa: E702
        name = label.split()[0]
        write_mp4(os.path.join(a.out, name + ".mp4"), (frame(rec, t, label, meta) for t in range(len(rec))))
    write_mp4(os.path.join(a.out, "compare.mp4"), (grid(recs, labels, metas, t, a.cols) for t in range(a.steps)), size_of(metas, a.cols, len(recs)), "8M")


if __name__ == "__main__":
    main()

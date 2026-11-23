"""Task 00118 (vm-00105, vm-00106): how well a model restores its blind spot, from frozen lives recorded with
online.pred_record (pred_frames.npy: body 0's predicted current frame, each piece its own block, decoded pair, from the
frame as the model saw it; pred_truth.npy: the true frame).

Per run, over the ticks after --skip, in grey levels (0-255):
- gap: the mean absolute error of the prediction on the gap's pixels; its baselines: bg (the gap predicted as background,
  the frame's most common value) and prev (the true gap of the tick before, which the model never sees with the gap);
- obj: the same over ticks with an object in the gap (at least 4 true pixels off the background by more than 25), and
  by ticks since it entered (1, 2-3, 4-7, 8-15, 16+);
- seen: whether the predicted gap shows an object (the same rule on the prediction): hits, false alarms, d';
- edge: the error on the visible pixels within 4 px of the gap; rest: the error on the rest of the gap's quarter, and on
  the other quarters.

    ./venv/bin/python eval_blind.py RUN_DIR [...] --gap 38 16 12 12 [--skip 20] [--out rows.jsonl] [--label NAME]
    ./venv/bin/python eval_blind.py --video OUT.mp4 RUN_DIR --gap 38 16 12 12 [--ticks 600]
"""
import argparse
import json
import os

from statistics import NormalDist

import numpy as np

SINCE = (1, 2, 4, 8, 16)
SINCE_NAMES = ("1", "2-3", "4-7", "8-15", "16+")
OFF, NPIX = 25, 4


def background(frames):
    """Each frame's most common grey [T]."""
    return np.array([np.bincount(f.ravel().astype(np.int64), minlength=256).argmax() for f in frames], np.float64)


def runs_since(flag):
    """Ticks since the current stretch of True began (1 on its first tick), 0 where False."""
    out = np.zeros(len(flag), np.int64)
    for t, f in enumerate(flag):
        out[t] = (out[t - 1] + 1 if t else 1) if f else 0
    return out


def measure(run, gap, skip=20):
    p = np.load(os.path.join(run, "pred_frames.npy")).astype(np.float64)
    y = np.load(os.path.join(run, "pred_truth.npy")).astype(np.float64)
    x0, y0, w, h = gap
    m = np.zeros(y.shape[1:], bool)
    m[y0:y0 + h, x0:x0 + w] = True
    edge = np.zeros_like(m)
    edge[max(y0 - 4, 0):y0 + h + 4, max(x0 - 4, 0):x0 + w + 4] = True
    edge &= ~m
    H, W = m.shape
    qr, qc = y0 // (H // 2), x0 // (W // 2)
    quarter = np.zeros_like(m)
    quarter[qr * H // 2:(qr + 1) * H // 2, qc * W // 2:(qc + 1) * W // 2] = True
    bg = background(y)
    err = np.abs(p - y)
    g_t = err[:, m].mean(1)
    bg_t = np.abs(y[:, m] - bg[:, None]).mean(1)
    prev_t = np.r_[np.nan, np.abs(y[1:, m] - y[:-1, m]).mean(1)]
    obj = (np.abs(y[:, m] - bg[:, None]) > OFF).sum(1) >= NPIX
    seen = (np.abs(p[:, m] - bg[:, None]) > OFF).sum(1) >= NPIX
    since = runs_since(obj)
    s = slice(skip, None)
    obj_s, seen_s, g_s, since_s = obj[s], seen[s], g_t[s], since[s]
    hit = float(seen_s[obj_s].mean()) if obj_s.any() else float("nan")
    fa = float(seen_s[~obj_s].mean()) if (~obj_s).any() else float("nan")
    n_o, n_n = max(obj_s.sum(), 1), max((~obj_s).sum(), 1)
    z = NormalDist().inv_cdf
    dp = float(z(float(np.clip(hit, 0.5 / n_o, 1 - 0.5 / n_o))) - z(float(np.clip(fa, 0.5 / n_n, 1 - 0.5 / n_n)))) if obj_s.any() and (~obj_s).any() else float("nan")
    by = {}
    edges = SINCE + (10 ** 9,)
    for name, lo, hi in zip(SINCE_NAMES, edges[:-1], edges[1:]):
        k = obj_s & (since_s >= lo) & (since_s < hi)
        by[name] = [float(g_s[k].mean()) if k.any() else None, int(k.sum())]
    return dict(run=run, ticks=int(len(g_s)), gap=float(g_s.mean()), gap_bg=float(bg_t[s].mean()), gap_prev=float(np.nanmean(prev_t[s])),
                obj_share=float(obj_s.mean()), gap_obj=float(g_s[obj_s].mean()) if obj_s.any() else None,
                gap_obj_bg=float(bg_t[s][obj_s].mean()) if obj_s.any() else None,
                gap_empty=float(g_s[~obj_s].mean()) if (~obj_s).any() else None, since=by,
                hit=hit, fa=fa, dprime=dp, edge=float(err[s][:, edge].mean()),
                quarter_rest=float(err[s][:, quarter & ~m & ~edge].mean()), other=float(err[s][:, ~quarter].mean()))


def video(out, run, gap, ticks, scale=4):
    from PIL import Image, ImageDraw
    from panel_video import write_mp4
    p = np.load(os.path.join(run, "pred_frames.npy"))
    y = np.load(os.path.join(run, "pred_truth.npy"))
    x0, y0, w, h = gap
    seen = y.copy()
    seen[:, y0:y0 + h, x0:x0 + w] = 0
    T, H, W = min(ticks, len(y)), y.shape[1], y.shape[2]
    head, gap_px = 18, 8
    width, height = 3 * (W * scale + gap_px) - gap_px, head + H * scale
    width, height = -(-width // 8) * 8, -(-height // 8) * 8
    names = ("what it sees", "its prediction", "the truth")

    def frame(t):
        img = Image.new("RGB", (width, height), (40, 40, 40))
        d = ImageDraw.Draw(img)
        for i, a in enumerate((seen[t], p[t], y[t])):
            x = i * (W * scale + gap_px)
            d.text((x + 4, 3), "%s   tick %d" % (names[i], t + 1), fill=(230, 230, 230))
            tile = Image.fromarray(a).convert("RGB").resize((W * scale, H * scale), Image.NEAREST)
            ImageDraw.Draw(tile).rectangle([x0 * scale, y0 * scale, (x0 + w) * scale - 1, (y0 + h) * scale - 1], outline=(220, 60, 60))
            img.paste(tile, (x, head))
        return np.asarray(img)

    write_mp4(out, (frame(t) for t in range(T)), size=(width, height), fps=15)
    Image.fromarray(frame(T // 2)).save(os.path.splitext(out)[0] + "_still.png")
    print("wrote %s (%d ticks)" % (out, T))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--gap", type=int, nargs=4, required=True, metavar=("X", "Y", "W", "H"))
    ap.add_argument("--skip", type=int, default=20)
    ap.add_argument("--out")
    ap.add_argument("--label", default="")
    ap.add_argument("--video")
    ap.add_argument("--ticks", type=int, default=600)
    a = ap.parse_args()
    if a.video:
        video(a.video, a.runs[0], a.gap, a.ticks)
        return
    rows = []
    for r in a.runs:
        row = dict(measure(r, a.gap, a.skip), label=a.label)
        rows.append(row)
        print("%s  gap %.2f (bg %.2f, prev %.2f)  obj %.2f (bg %.2f, share %.2f)  hit %.2f fa %.2f d' %.2f  edge %.2f other %.2f" % (
            os.path.basename(r), row["gap"], row["gap_bg"], row["gap_prev"], row["gap_obj"] or np.nan, row["gap_obj_bg"] or np.nan,
            row["obj_share"], row["hit"], row["fa"], row["dprime"], row["edge"], row["other"]))
    if a.out:
        with open(a.out, "a") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()

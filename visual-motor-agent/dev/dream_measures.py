"""Measures of a dream video's frames (dream_video.py writes them to dream_<decode>.npz; see ANALYSIS.md): per dreamed
step, the cursor's pixels (white) and the objects' pixels (circles and dots: neither background, white nor the cursor's
black outline), dreamed and real; when the dream loses the cursor and when it empties.

Usage (from visual-motor-agent/dev):
    ./venv/bin/python dream_measures.py ~/data/research/visual-motor-agent/exp_00036/video_ESF4_s3/dream_pair.npz
"""
import argparse

import numpy as np

WHITE, DARK = 250, 40       # gray levels: the cursor's white (>=) and its black outline (<=)


def counts(frames, bg):
    """frames [T, H, W, C] uint8 -> (cursor pixels [T], object pixels [T]) from the first channel."""
    a = frames[..., 0].astype(int).reshape(len(frames), -1)
    return (a >= WHITE).sum(1), ((a != bg) & (a > DARK) & (a < WHITE)).sum(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz", nargs="+")
    ap.add_argument("--real", type=int, default=None, help="real steps before the dream (default: from the npz, "
                                                           "else 200)")
    ap.add_argument("--at", type=int, nargs="+", default=[1, 2, 3, 10, 20, 50, 100, 200], help="dreamed steps shown")
    ap.add_argument("--cursor-min", type=int, default=10, help="the cursor counts as drawn from this many white pixels")
    ap.add_argument("--empty", type=int, default=30, help="the dream counts as empty below this many object pixels")
    args = ap.parse_args()
    for path in args.npz:
        z = np.load(path)
        R = args.real if args.real is not None else int(z["n_real"]) if "n_real" in z else 200
        pred, real = z["pred"][R:], z["real"][R:]
        bg = np.bincount(z["real"][0][..., 0].ravel()).argmax()
        pc, po = counts(pred, bg)
        rc, ro = counts(real, bg)
        drawn = np.flatnonzero(pc >= args.cursor_min)
        gone = np.flatnonzero(po < args.empty)
        print(path)
        print("  dreamed steps %d (after %d real); background %d" % (len(pred), R, bg))
        at = [h for h in args.at if h <= len(pred)]
        print("  step            " + " ".join("%5d" % h for h in at))
        print("  cursor  dream   " + " ".join("%5d" % pc[h - 1] for h in at))
        print("          real    " + " ".join("%5d" % rc[h - 1] for h in at))
        print("  objects dream   " + " ".join("%5d" % po[h - 1] for h in at))
        print("          real    " + " ".join("%5d" % ro[h - 1] for h in at))
        print("  cursor drawn (>= %d px) in %d of %d dreamed steps; last at step %s" % (
            args.cursor_min, len(drawn), len(pred), drawn[-1] + 1 if len(drawn) else "-"))
        print("  first dreamed step with < %d object px: %s (real frames: mean %d object px)" % (
            args.empty, gone[0] + 1 if len(gone) else "never", ro.mean()))


if __name__ == "__main__":
    main()

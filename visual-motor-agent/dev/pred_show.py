"""Task 00139 (a life that never sees): what a model paints for the frame before it learns from it.

A life run with online.pred_score has, in its metrics, the summed squared error a frame of its picture of the current
frame (pred_err), of the last real frame (pred_copy) and of the mean of all frames so far (pred_mean); with
online.pred_record it keeps the real frame and the picture at its first and last ticks and at every pred_every-th
(RUN/pred_frames.npz: tick, real, pred).

    ./venv/bin/python pred_show.py table RUN [RUN ...]            the errors by ticks of life, and after it (Markdown)
    ./venv/bin/python pred_show.py after RUN [--ticks 2000]       (GPU) the last checkpoint frozen on the life's own world
                                                                  from 64 ticks before its end on: RUN/wm/after.json
    ./venv/bin/python pred_show.py curves RUN [RUN ...] -o PNG    the errors over the life, a panel a life
    ./venv/bin/python pred_show.py strip RUN -o PNG               the real frame and the picture at a few ticks
    ./venv/bin/python pred_show.py video RUN [RUN ...] -o MP4     the kept ticks as a time-lapse, a life a pair of panels

An error is shown as the root of its mean a pixel, in grey levels (rms); skill is 1 - the picture's summed squared
error / the other's, so 0 is level with it and below 0 worse.
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

BACK = 64                       # `after` starts this many ticks before the life's end, where the kept frames are
LR = 0.0003                     # the watching configs' learning rate (a title names a larger one)
WINDOWS = ((1, 8), (9, 64), (65, 512), (513, 2000))   # ticks after the life's end, in the tables


def name_of(run):
    return os.path.basename(run.rstrip("/")).rsplit("_s", 1)[0]


def title_of(run):
    """EB: the court, with replay; ED1: the discs, no replay (from the life's own settings)."""
    c = json.load(open(os.path.join(run, "info.json")))["config"]
    world = dict(breakout="the court", watch="the discs").get(c["env"].get("kind", ""), c["env"].get("kind", "the world"))
    lr = float(c.get("ppo", {}).get("lr", LR))
    fast = ", learning rate x%g" % round(lr / LR, 2) if lr > LR * 1.01 else ""
    return "%s: %s, %s%s" % (name_of(run), world, "no replay" if int(c["online"]["batch"]) == 1 else "with replay", fast)


def metrics(run):
    """The life's rows that carry the picture's errors."""
    rows = [json.loads(x) for x in open(os.path.join(run, "metrics.jsonl"))]
    return [r for r in rows if "pred_err" in r]


def whole(run, after=1000):
    """The life after its first `after` ticks as one: the mean errors a frame (pred_err, pred_copy, pred_mean; every row
    is a mean over as many ticks), the number of its rows and of those in which the picture beats copy."""
    rows = [r for r in metrics(run) if r["step"] > after]
    if not rows:
        return None
    return dict(pred_err=float(np.mean([r["pred_err"] for r in rows])), pred_copy=float(np.mean([r["pred_copy"] for r in rows])),
                pred_mean=float(np.mean([r["pred_mean"] for r in rows])), rows=len(rows),
                beats=int(sum(r["pred_err"] < r["pred_copy"] for r in rows)), first=min(r["step"] for r in rows), last=max(r["step"] for r in rows))


def rms(err, pixels):
    """The root of the mean squared error a pixel, in grey levels."""
    return float(np.sqrt(err / pixels))


def skill(err, other):
    return 1.0 - err / other if other > 0 else float("nan")


def pixels_of(run):
    c = json.load(open(os.path.join(run, "info.json")))["config"]["model"]
    return int(c["img"]) * int(c.get("img_w") or c["img"])


def f1(x):
    return "–" if x != x else "%.1f" % x


def f3(x):
    """A share: .456, 1.000."""
    if x != x:
        return "–"
    t = "%.3f" % x
    return t[1:] if t.startswith("0.") else t


def f2s(x):
    """A skill: +.12, −3.41 (a skill far below 0 is no share: two decimals)."""
    if x != x:
        return "–"
    t = "%.2f" % abs(x)
    t = t[1:] if t.startswith("0.") else t
    return ("−" if x < 0 else "") + t


def table(runs, at=(1000, 5000, 20000, 40000, 60000)):
    """Markdown: the picture's error in the life at some ticks (the 1,000 ticks before each), then after the life."""
    text = ["### In the life: the picture of a frame before the model learns from it\n",
            "| Life | Ticks | The picture, rms | Copy, rms | The mean frame, rms | Skill against copy | Skill against the mean frame |",
            "|---|---|---|---|---|---|---|"]
    for run in runs:
        px, rows = pixels_of(run), {r["step"]: r for r in metrics(run)}
        for st in at:
            if st in rows:
                r = rows[st]
                text.append("| %s | %dk | %s | %s | %s | %s | %s |" % (
                    name_of(run), st // 1000, f1(rms(r["pred_err"], px)), f1(rms(r["pred_copy"], px)), f1(rms(r["pred_mean"], px)),
                    f2s(skill(r["pred_err"], r["pred_copy"])), f2s(skill(r["pred_err"], r["pred_mean"]))))
        r = whole(run)
        if r:   # the life after its first 1,000 ticks as one
            text.append("| %s | all after 1k | %s | %s | %s | %s | %s |" % (
                name_of(run), f1(rms(r["pred_err"], px)), f1(rms(r["pred_copy"], px)), f1(rms(r["pred_mean"], px)),
                f2s(skill(r["pred_err"], r["pred_copy"])), f2s(skill(r["pred_err"], r["pred_mean"]))))
    court = [(run, court_things(run)) for run in runs if os.path.exists(os.path.join(run, "pred_frames.npz"))]
    court = [(run, t) for run, t in court if t]
    if court:
        text += ["", "### The court: the paddle and the ball in the pictures kept from tick 1,000 on\n",
                 "| Life | Pictures | With a paddle | The paddle's place off, median px | Within 2 px | Within 6 px | A paddle standing at its mean place: off, median px | With a ball pixel |",
                 "|---|---|---|---|---|---|---|---|"]
        for run, t in court:
            text.append("| %s | %d | %s | %s | %s | %s | %s | %s |" % (name_of(run), t["n"], f3(t["paddle"]), f1(t["off"]), f3(t["within2"]),
                                                                    f3(t["within6"]), f1(t["still"]), f3(t["ball"])))
    got = [(run, json.load(open(os.path.join(run, "wm", "after.json")))) for run in runs if os.path.exists(os.path.join(run, "wm", "after.json"))]
    if got:
        text += ["", "### After the life: the last weights frozen, on the life's own world as it goes on\n",
                 "| Life | Ticks after the end | The frozen picture, rms | Copy, rms | Skill against copy |", "|---|---|---|---|---|"]
        for run, a in got:
            px, t, e, c = pixels_of(run), np.array(a["tick"]), np.array(a["err"]), np.array(a["copy"])
            for lo, hi in ((-a["back"], 0),) + WINDOWS:
                m = (t >= lo) & (t <= hi)
                if m.any():
                    label = "the last %d of the life" % int(m.sum()) if hi == 0 else "%d to %s" % (lo, "{:,}".format(hi))
                    text.append("| %s | %s | %s | %s | %s |" % (name_of(run), label, f1(rms(e[m].mean(), px)), f1(rms(c[m].mean(), px)),
                                                                 f2s(skill(e[m].sum(), c[m].sum()))))
    return "\n".join(text) + "\n"


def court_things(run, first=1000):
    """The court: is the paddle in the kept pictures from tick `first` on, and where; is the ball? The paddle is read in
    its first row as the pixels above the level between the background and the paddle's grey, its place their mean
    column. n: the pictures read; paddle: the share with a paddle; off: the median px between the painted paddle's place
    and the real one's (within2, within6: the shares that near); still: the same for a paddle that stood at the real
    paddle's mean place (a picture that knows the scenery and not the present); ball: the share of the pictures with
    any pixel of the ball's zone above the level between the background and the ball's grey. None: not the court."""
    from world import make_config
    c = make_config(json.load(open(os.path.join(run, "info.json")))["config"]["env"])
    if getattr(c, "kind", "") != "breakout":
        return None
    z = np.load(os.path.join(run, "pred_frames.npz"))
    keep = z["tick"] >= first
    real, pred = z["real"][keep], z["pred"][keep]
    S = real.shape[1]
    row, level = int(c.paddle_y), (c.bg + c.paddle_grey) / 2.0

    def place(f):
        bright = f[:, row, c.wall:S - c.wall] > level                                  # [n, columns inside the walls]
        n = bright.sum(1)
        return np.where(n > 0, (bright * np.arange(c.wall, S - c.wall)[None]).sum(1) / np.maximum(n, 1), np.nan)

    pr, pp = place(real), place(pred)
    both = (pr == pr) & (pp == pp)
    d = np.abs(pp[both] - pr[both])
    still = np.abs(pr[pr == pr] - np.nanmean(pr))
    zone = np.zeros((S, S), bool)
    zone[c.wall:row, c.wall:S - c.wall] = True
    zone[row + int(c.paddle_h):, c.wall:S - c.wall] = True
    return dict(n=int(keep.sum()), paddle=float((pp == pp).mean()), off=float(np.median(d)) if len(d) else float("nan"),
                within2=float((d <= 2).mean()) if len(d) else float("nan"), within6=float((d <= 6).mean()) if len(d) else float("nan"),
                still=float(np.median(still)), ball=float(((pred > (c.bg + c.ball_grey) / 2.0) & zone[None]).any((1, 2)).mean()))


def after(run, ticks=2000, step=None, chunk=64, decode="pair", whole=False):
    """The last checkpoint with learning off on the life's own world, from BACK ticks before the life's end on: at
    every tick its picture of the frame, made as the life made it (the window as the life saw it, the real bodies),
    against the real frame; and copy. Writes RUN/wm/after.json: tick (0: the life's last tick), err, copy (summed
    squared errors a frame). The world is lined up by the life's own last kept frame. decode 'mean': the picture as
    the expected value of every pixel, which shows a faint trace that the sharp decoding rounds away (after_mean.json).
    whole: a blind life's model shown the real frames for the first time, the window's and the earlier patches of the
    frame it paints (after_whole.json)."""
    import agent_lib
    import cuda_lib
    import wm_measures as W
    from lifetime import ckpt_name
    from online import DECODE, blind_frames, uni_frame
    info = json.load(open(os.path.join(run, "info.json")))
    on, steps = info["config"]["online"], int(info["config"]["online"]["steps"])
    step = steps if step is None else int(step)
    back = min(BACK, step - 8)                       # a short life: as far back as it goes
    exam = W.Exam(info["config"]["env"], seed=info["seed"], copies=1, skip=step - back, ticks=back + ticks + 8)
    z = np.load(os.path.join(run, "pred_frames.npz"))
    last = z["real"][list(z["tick"]).index(step)]
    hits = [i for i in range(max(0, back - 8), back + 8) if (exam.frames[0, i] == last).all()]
    if len(hits) != 1:
        raise ValueError("the life's frame of tick %d is found %d times in its own world near there" % (step, len(hits)))
    j = hits[0]                                      # the exam's index of the life's tick `step`
    agent = agent_lib.Agent.load(os.path.join(run, ckpt_name(step, 0, 1)))
    model = cuda_lib.CudaModel(agent)
    c = agent.config
    L, lay = c["ctx"], agent_lib.uni_layout(c)["patch"] - 1
    idx = np.arange(L - 1, exam.T)                   # every tick with a whole window before it
    img = np.repeat(exam.frames[0][..., None], 3, -1)
    if on.get("blind") and not whole:
        img = blind_frames(img, on["blind"])
    err, cop = [], []
    for i in range(0, len(idx), chunk):
        t = idx[i:i + chunk]
        win = t[:, None] + np.arange(-(L - 1), 1)[None]                       # [n, L]: each tick's window
        o = model.forward_uni_step(img[win], exam.body[0][win], np.zeros(win.shape + (3,), np.float32), np.zeros(win.shape, np.uint8),
                                   L - 1, mode=DECODE[decode])
        pf = uni_frame(c, np.clip(o["values_t"][:, lay], 0.0, 1.0))[..., 0].astype(np.float64)
        real = exam.frames[0][t].astype(np.float64)
        err += ((pf - real) ** 2).sum((1, 2)).tolist()
        cop += ((exam.frames[0][t - 1].astype(np.float64) - real) ** 2).sum((1, 2)).tolist()
    model.close()
    agent.close()
    res = dict(run=os.path.basename(run.rstrip("/")), step=step, decode=decode, whole=bool(whole), back=int(-(idx - j).min()),
               tick=(idx - j).tolist(), err=err, copy=cop)
    os.makedirs(os.path.join(run, "wm"), exist_ok=True)
    p = os.path.join(run, "wm", "after%s%s.json" % ("_whole" if whole else "", "" if decode == "pair" else "_" + decode))
    json.dump(res, open(p + ".tmp", "w"))
    os.replace(p + ".tmp", p)
    return res


def curves(runs, path):
    """A panel a life: the error over the life, the picture (a colour) beside copy and the mean frame (grey, each with
    its label at the line's end)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from readme_images import COLORS, GRID, INK, MUTED_INK, SURFACE, save
    from wm_curves import spread
    cols = min(3, len(runs))
    nrow = -(-len(runs) // cols)
    fig, axs = plt.subplots(nrow, cols, figsize=(9.4, 3.1 * nrow), dpi=100, facecolor=SURFACE, squeeze=False)
    for ax in axs.ravel()[len(runs):]:
        ax.set_visible(False)
    for ax, run in zip(axs.ravel(), runs):
        px, rows = pixels_of(run), metrics(run)
        x = np.array([r["step"] for r in rows]) / 1000.0
        ends = []
        for key, label, style in (("pred_copy", "copy", dict(color=MUTED_INK, lw=1.2, ls=(0, (4, 2)))),
                                  ("pred_mean", "mean frame", dict(color=MUTED_INK, lw=1.2, ls=(0, (1, 2)))),
                                  ("pred_err", "its picture", dict(color=COLORS[0], lw=2, solid_capstyle="round"))):
            y = np.array([rms(r[key], px) for r in rows])
            ax.plot(x, y, **style)
            ends.append((x[-1], y[-1], label))
        ax.set_xlim(0, x.max() * 1.42)
        ax.set_ylim(bottom=0)
        ax.set_title(title_of(run).replace(": ", "\n", 1), fontsize=9.5, color=INK, loc="left")
        ax.set_facecolor(SURFACE)
        ax.grid(axis="y", color=GRID, lw=1)
        ax.set_axisbelow(True)
        ax.set_xlabel("ticks of life, thousands", fontsize=9, color=MUTED_INK)
        ax.tick_params(colors=MUTED_INK, labelsize=8.5, length=0)
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax._ends = ends
    for r in range(nrow):
        axs[r][0].set_ylabel("error a pixel, grey levels (rms)", fontsize=9, color=MUTED_INK)
    fig.tight_layout()
    fig.canvas.draw()
    for ax in axs.ravel()[:len(runs)]:
        pts = [ax.transData.transform((xe, ye))[1] * 72.0 / fig.dpi for xe, ye, _ in ax._ends]
        for (xe, ye, text), y0, y1 in zip(ax._ends, pts, spread(pts, 9.0)):
            ax.annotate(text, xy=(xe, ye), xytext=(5, y1 - y0), textcoords="offset points", va="center", ha="left", fontsize=8, color=MUTED_INK)
    save(fig, path)
    return path


def pick(tick, want=(1, 100, 1000, 10000, 30000)):
    """The kept ticks shown in a strip: those wanted that are there, and the life's last two."""
    have = [int(t) for t in tick]
    show = [t for t in want if t in have and t < have[-2]]
    return show + have[-2:]


def strip(run, path, scale=2):
    """The real frame over the model's picture of it, at a few ticks of the life. Returns the ticks shown."""
    from PIL import Image, ImageDraw, ImageFont
    from readme_images import FONT, fit
    BG, INK, MUTED = (252, 252, 251), (11, 11, 11), (82, 81, 78)
    PAD, GAP, LEFT, HEAD = 8, 6, 104, 20
    z = np.load(os.path.join(run, "pred_frames.npz"))
    tick, real, pred = z["tick"], z["real"], z["pred"]
    show = pick(tick)
    S = real.shape[1] * scale
    names = ("the real frame", "its picture")
    im = Image.new("RGB", (2 * PAD + LEFT + len(show) * S + (len(show) - 1) * GAP, 2 * PAD + HEAD + 2 * S + GAP), BG)
    d = ImageDraw.Draw(im)
    f = ImageFont.truetype(FONT, 12)
    for r, name in enumerate(names):
        d.text((PAD, PAD + HEAD + r * (S + GAP) + S // 2 - 7), name, font=f, fill=INK)
    for col, t in enumerate(show):
        i = list(tick).index(t)
        x = PAD + LEFT + col * (S + GAP)
        d.text((x, PAD), "tick {:,}".format(t), font=f, fill=MUTED)
        for r, a in enumerate((real[i], pred[i])):
            y = PAD + HEAD + r * (S + GAP)
            im.paste(Image.fromarray(np.repeat(a[..., None], 3, -1)).resize((S, S), Image.NEAREST), (x, y))
            d.rectangle((x - 1, y - 1, x + S, y + S), outline=(230, 229, 225))
    fit(im, path)
    return show


def video(runs, path, scale=3, fps=12, labels=None, cols=3):
    """The kept ticks as a time-lapse: for every life the real frame beside the model's picture, `cols` lives a row.
    Returns the number of frames. The lives must have kept the same ticks."""
    from PIL import Image, ImageDraw
    import wm_video as V
    zs = [np.load(os.path.join(r, "pred_frames.npz")) for r in runs]
    tick = zs[0]["tick"]
    if any(len(z["tick"]) != len(tick) or (z["tick"] != tick).any() for z in zs):
        raise ValueError("the lives kept different ticks")
    S = zs[0]["real"].shape[1] * scale
    PAD, GAP, TOP, HEAD = 10, 8, 26, 40
    labels = labels or [title_of(r).split(": ", 1)[1] for r in runs]
    cols = min(cols, len(runs))
    nrow = -(-len(runs) // cols)
    block = 2 * S + GAP                                   # a life: the world and its picture
    w, h = 2 * PAD + cols * block + (cols - 1) * 3 * GAP, 2 * PAD + TOP + nrow * (HEAD + S) + (nrow - 1) * GAP
    w, h = w + (-w) % 8, h + (-h) % 8                     # the encoder's blocks
    f, small = V.font(13), V.font(12)

    def one(i):
        im = Image.new("RGB", (w, h), (17, 17, 17))
        d = ImageDraw.Draw(im)
        d.text((PAD, PAD - 2), "tick {:,}".format(int(tick[i])), font=f, fill=(240, 170, 80))
        for k, z in enumerate(zs):
            x, y = PAD + (k % cols) * (block + 3 * GAP), PAD + TOP + (k // cols) * (HEAD + S + GAP)
            d.text((x, y), labels[k], font=small, fill=(235, 235, 235))
            d.text((x, y + 18), "the world", font=small, fill=(150, 150, 150))
            d.text((x + S + GAP, y + 18), "its picture", font=small, fill=(150, 150, 150))
            for c, a in enumerate((z["real"][i], z["pred"][i])):
                im.paste(Image.fromarray(np.repeat(a[..., None], 3, -1)).resize((S, S), Image.NEAREST), (x + c * (S + GAP), y + HEAD))
        return np.asarray(im)

    hold = lambda i: 6 if (i < 8 or i >= len(tick) - 1) else 1          # noqa: E731   the first ticks and the last one stay a while
    return V.write_mp4(path, (one(i) for i in range(len(tick)) for _ in range(hold(i))), fps)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for cmd in ("table", "curves", "video"):
        a = sub.add_parser(cmd)
        a.add_argument("runs", nargs="+")
        if cmd != "table":
            a.add_argument("-o", "--path", required=True)
    a = sub.add_parser("strip")
    a.add_argument("run")
    a.add_argument("-o", "--path", required=True)
    a = sub.add_parser("after")
    a.add_argument("run")
    a.add_argument("--ticks", type=int, default=2000)
    a.add_argument("--decode", default="pair", choices=["pair", "mean"])
    a.add_argument("--whole", action="store_true", help="a blind life's model shown the real frames for the first time")
    args = ap.parse_args()
    if args.cmd == "table":
        print(table(args.runs))
    elif args.cmd == "curves":
        print("wrote", curves(args.runs, args.path))
    elif args.cmd == "strip":
        print("wrote", args.path, "ticks", strip(args.run, args.path))
    elif args.cmd == "video":
        print("wrote %s: %d frames" % (args.path, video(args.runs, args.path)))
    else:
        r = after(args.run, args.ticks, decode=args.decode, whole=args.whole)
        t, e, c = np.array(r["tick"]), np.array(r["err"]), np.array(r["copy"])
        print("%s: the frozen picture after the life's end: " % r["run"] + "; ".join(
            "ticks %d..%d skill against copy %.2f" % (lo, hi, skill(e[(t >= lo) & (t <= hi)].sum(), c[(t >= lo) & (t <= hi)].sum())) for lo, hi in WINDOWS))


if __name__ == "__main__":
    main()

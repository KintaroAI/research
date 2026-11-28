"""Task 00138: a dream that rolls dice, read from the scored files.

One checkpoint of several lives, each dreamed with several decodings (wm_measures.py score --decode pair draw0.7 ...
--out OUT/<arm>_s1/wm): the long dream's balls (the free part: step_<N>_<decode>_free.json) and what the dice cost in
the short dreams (the regular part: step_<N>_<decode>_work.json). Prints Markdown tables, a row a measure and decoding,
a column a life; -o writes a picture instead: a panel a life, the share of the long dreams with a ball by dreamed tick.

    ./venv/bin/python wm_dice.py OUT --arms OB SB3 MB M7B MNB --decodes pair draw0 draw drawp0.5 [--step 60000]
                                 [--seed 1] [-o PNG --show pair draw0.7 drawp0.7]
"""
import argparse
import json
import os

import numpy as np

import wm_measures as W
from analyze_lives import f1, f2, f3
from wm_curves import name_of, spread


def decode_name(d):
    """pair: today's; draw0: the likeliest value, no dice; draw0.7: dice by chance at temperature .7; drawp0.5: dice by
    the tallest bin (pair's ranking) at .5."""
    if d == "pair":
        return "today's (`pair`)"
    if d.startswith("draw"):
        peak = d.startswith("drawp")
        rest = d[5 if peak else 4:]
        t = float(rest) if rest else 1.0
        if t <= 0:
            return "`pair` as dice by the tallest bin at 0" if peak else "the likeliest value, no dice"
        return "dice by %s, T %s" % ("the tallest bin" if peak else "chance", ("%g" % t).lstrip("0") if t < 1 else "%g" % t)
    return d


def short_name(d):
    """The label at a curve's end: today, likeliest, dice T .7 (by the tallest bin), chance T 1."""
    if d == "pair":
        return "today"
    if d.startswith("draw"):
        peak = d.startswith("drawp")
        rest = d[5 if peak else 4:]
        t = float(rest) if rest else 1.0
        if t <= 0:
            return "today" if peak else "likeliest"
        return "%s T %s" % ("dice" if peak else "chance", ("%g" % t).lstrip("0") if t < 1 else "%g" % t)
    return d


def rolling(y, k):
    """The mean over k ticks around each tick (fewer at the ends)."""
    y = np.asarray(y, float)
    h = k // 2
    return np.array([y[max(0, i - h):i + h + 1].mean() for i in range(len(y))])


def get(out, arm, step, decode, which, seed, *path):
    """One number of a scored file, or nan."""
    r = W.load(out, arm, step, decode, which, [seed])
    x = W._get(r[seed], *path) if r else float("nan")
    return float(x) if isinstance(x, (int, float)) else float("nan")


def whole(x):
    return "–" if x != x else "%d" % round(x)


def two(x):
    return "–" if x != x else f2(x)


def median_last(out, arm, step, decode, seed):
    r = W.load(out, arm, step, decode, "free", [seed])
    last = W._get(r[seed], "free", "balls", "last") if r else None
    return float(np.median(last)) if isinstance(last, list) and last else float("nan")


# (title, set, the path in the summary, the number's format); a tick is an index: tick 200 is 199
LONG = [("Balls served in the long dreams", ("free", "balls", "served"), whole),
        ("Dreams with a ball at tick 50", ("free", "balls", "ball", 49), f3),
        ("Dreams with a ball at tick 100", ("free", "balls", "ball", 99), f3),
        ("Dreams with a ball at tick 200", ("free", "balls", "ball", 199), f3),
        ("Balls back elsewhere after 3 ticks or more without one", ("free", "balls", "back"), whole),
        ("Stray bright pixels a frame", ("free", "balls", "specks"), two)]
SHORT = [("The ball held, dreamed ticks", ("regular", "hold", "ball", "horizon"), whole),
         ("The ball held at tick 8", ("regular", "hold", "ball", "held", 7), f3),
         ("The ball held at tick 16", ("regular", "hold", "ball", "held", 15), f3),
         ("The paddle held, dreamed ticks", ("regular", "hold", "paddle", "horizon"), whole),
         ("Next-frame skill", ("regular", "next", "skill"), f3)]


def tables(out, arms, decodes, step=60000, seed=1):
    """The Markdown text: the long dream, then the short dreams; a row a measure and decoding, a column a life."""
    head = "| Measure | Decoding | " + " | ".join("%s: %s" % (a, name_of(a)) for a in arms) + " |"
    rule = "|---|---|" + "---|" * len(arms)
    text = []
    scored = [d for d in decodes if any(W.load(out, a, step, d, "free", [seed]) for a in arms)]
    if scored:
        starts = [get(out, a, step, d, "free", seed, "free", "starts") for a in arms for d in scored]
        n = whole(np.nanmax(starts)) if any(x == x for x in starts) else "?"
        text += ["### The long dream: %d ticks from %s stretches of the exam\n" % (W.FREE_TICKS, n), head, rule]
        for title, path, fmt in LONG:
            for d in scored:
                text.append("| %s | %s | " % (title, decode_name(d)) + " | ".join(fmt(get(out, a, step, d, "free", seed, *path)) for a in arms) + " |")
            if path[-1] in ("served", "back") or path[2] == "ball":   # the world's own continuation of the same stretches
                real = ("free", "balls_real") + path[2:]
                text.append("| %s | the world itself | " % title + " | ".join(fmt(get(out, a, step, scored[0], "free", seed, *real)) for a in arms) + " |")
        for d in scored:
            text.append("| The last ball, median tick | %s | " % decode_name(d) + " | ".join(whole(median_last(out, a, step, d, seed)) for a in arms) + " |")
        text.append("")
    scored = [d for d in decodes if any(W.load(out, a, step, d, "work", [seed]) for a in arms)]
    if scored:
        text += ["### The short dreams: the regular part\n", head, rule]
        for title, path, fmt in SHORT:
            for d in scored:
                text.append("| %s | %s | " % (title, decode_name(d)) + " | ".join(fmt(get(out, a, step, d, "work", seed, *path)) for a in arms) + " |")
        text.append("")
    return "\n".join(text)


def figure(out, arms, show, path, step=60000, seed=1, smooth=5):
    """A panel a life: the share of the long dreams with a ball in the court by dreamed tick (a mean over `smooth`
    ticks), for up to three decodings (a colour and a label at the line's end each) and for the world itself (grey,
    dotted)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from readme_images import COLORS, GRID, INK, MUTED_INK, SURFACE, save
    if not 1 <= len(show) <= 3:
        raise ValueError("the picture shows one to three decodings (three colours can be told apart by every reader)")
    cols = 3
    rows = -(-(len(arms) + 1) // cols)       # one more panel's room for the legend
    fig, axs = plt.subplots(rows, cols, figsize=(9.4, 2.9 * rows), dpi=100, facecolor=SURFACE, squeeze=False)
    x = np.arange(1, W.FREE_TICKS + 1)
    handles = None
    for k, ax in enumerate(axs.ravel()):
        ax.set_facecolor(SURFACE)
        for sp in ax.spines.values():
            sp.set_visible(False)
        if k >= len(arms):
            ax.set_xticks([]), ax.set_yticks([])
            continue
        a, ends, lines = arms[k], [], []
        real = None
        for i, d in enumerate(show):
            r = W.load(out, a, step, d, "free", [seed])
            if not r:
                continue
            y = rolling(W._get(r[seed], "free", "balls", "ball"), smooth)
            real = rolling(W._get(r[seed], "free", "balls_real", "ball"), smooth)
            lines.append(ax.plot(x[:len(y)], y, color=COLORS[i], lw=2, solid_capstyle="round", label=decode_name(d).replace("`", ""))[0])
            ends.append((len(y), float(y[-1]), short_name(d)))
        if real is not None:
            lines.append(ax.plot(x[:len(real)], real, color=MUTED_INK, lw=1.2, ls=(0, (1, 2)), label="the world itself")[0])
        handles = handles or lines
        ax.set_ylim(-0.03, 1.03)
        ax.set_xlim(0, W.FREE_TICKS * 1.24)
        ax.set_xticks([0, 50, 100, 150, 200])
        ax.set_title("%s\n%s" % (a, name_of(a)), fontsize=9.5, color=INK, loc="left")
        ax.grid(axis="y", color=GRID, lw=1)
        ax.set_axisbelow(True)
        ax.set_xlabel("dreamed tick", fontsize=9, color=MUTED_INK)
        if k % cols == 0:
            ax.set_ylabel("dreams with a ball", fontsize=9, color=MUTED_INK)
        ax.tick_params(colors=MUTED_INK, labelsize=8.5, length=0)
        ax._ends = ends
    fig.tight_layout()
    if handles:
        spare = axs.ravel()[len(arms)]
        spare.legend(handles=handles, loc="center", frameon=False, fontsize=9, labelcolor=MUTED_INK,
                     title="Dreams with a ball in the court\n(a mean over %d ticks)" % smooth, title_fontsize=9.5)
        spare.get_legend().get_title().set_color(INK)
    fig.canvas.draw()
    for ax in axs.ravel()[:len(arms)]:      # the labels at the lines' ends, no two closer than a line of text
        pts = [ax.transData.transform((xe, ye))[1] * 72.0 / fig.dpi for xe, ye, _ in ax._ends]
        for (xe, ye, text), y0, y1 in zip(ax._ends, pts, spread(pts, 9.0)):
            ax.annotate(text, xy=(xe, ye), xytext=(5, y1 - y0), textcoords="offset points", va="center", ha="left", fontsize=8,
                        color=MUTED_INK)
    save(fig, path)
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--arms", nargs="+", required=True)
    ap.add_argument("--decodes", nargs="+", default=["pair"])
    ap.add_argument("--step", type=int, default=60000)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("-o", "--path", default=None)
    ap.add_argument("--show", nargs="+", default=None, help="-o: the decodings drawn (three at most; default: the first three of --decodes)")
    a = ap.parse_args()
    if a.path:
        print("wrote", figure(a.out, a.arms, a.show or a.decodes[:3], a.path, a.step, a.seed))
    else:
        print(tables(a.out, a.arms, a.decodes, a.step, a.seed))


if __name__ == "__main__":
    main()

"""Task 00135: a life's numbers by its ticks (the owner: "to see if it improves over time").

Every checkpoint that `wm_measures.py score` has scored, on the exam (step_<N>_pair_work.json) and, where it is there,
on the life's own world after its end (step_<N>_pair_home64.json). Prints Markdown tables, one a world and set: a row a
measure and arm, a column a tick, so a row reads as a curve; under two arms their difference (the second named less the
first). With several seeds a cell is their mean. -o writes a picture of the main curves instead.

    ./venv/bin/python wm_curves.py OUT [--arms OD SD3 OB SB3] [--seeds 1] [--steps 20000 40000 ...] [-o PNG]
"""
import argparse
import json
import os

import numpy as np

import wm_measures as W
from analyze_lives import f1, f3

STEPS = tuple(range(20000, 200001, 20000))
G = ("regular",)
MEASURES = dict(
    watch=[("Next-frame skill", G + ("next", "skill"), f3), ("The food held at tick 4", G + ("hold", "food", "held", 3), f3),
           ("The food held at tick 8", G + ("hold", "food", "held", 7), f3), ("Horizon of the food, ticks", G + ("hold", "food", "horizon"), f1),
           ("Horizon of the poison, ticks", G + ("hold", "poison", "horizon"), f1), ("Skill against the copy at tick 8", G + ("curve", "skill_copy", 7), f3),
           ("The poison's way gone at tick 4", ("course", "poison", "travel", 3), f3)],
    breakout=[("Next-frame skill", G + ("next", "skill"), f3), ("The ball held at tick 8", G + ("hold", "ball", "held", 7), f3),
              ("The ball held at tick 16", G + ("hold", "ball", "held", 15), f3), ("Horizon of the ball, ticks", G + ("hold", "ball", "horizon"), f1),
              ("Horizon of the paddle, ticks", G + ("hold", "paddle", "horizon"), f1), ("Skill against the copy at tick 8", G + ("curve", "skill_copy", 7), f3),
              ("The ball's way gone at tick 8", ("course", "ball", "travel", 7), f3)])
WORLD = dict(watch="The moving discs", breakout="The court")
SETS = (("work", "the exam"), ("home64", "the life's own world, after its end"))


def name_of(arm):
    """OD, OB: no self-forcing; SD3, SB7: up to 3, up to 7 dreamed steps."""
    return "no self-forcing" if arm.startswith("O") else "up to %s dreamed steps" % arm[2:] if arm.startswith("S") and arm[2:].isdigit() else arm


def series(out, arm, seeds, steps, which, path):
    """One number by tick: the seeds' mean at each step (nan where no seed is scored)."""
    v = []
    for st in steps:
        r = W.load(out, arm, st, "pair", which, seeds)
        x = [W._get(r[k], *path) for k in r]
        x = [a for a in x if isinstance(a, (int, float)) and a == a]
        v.append(float(np.mean(x)) if x else float("nan"))
    return v


def worlds(out, arms, seeds, steps):
    """world -> its arms, in the order given (an arm's world from its first scored checkpoint)."""
    res = {}
    for a in arms:
        for st in steps:
            r = W.load(out, a, st, "pair", "work", seeds)
            if r:
                res.setdefault(next(iter(r.values()))["world"], []).append(a)
                break
    return res


def signed(fmt, x):
    """A difference: +.092, −.016, .000."""
    t = fmt(x)
    return ("+" if x > 0 and t.strip("0.") else "") + t


def tables(out, arms, seeds, steps):
    """The Markdown text: a table a world and set, with the ticks that are scored as its columns."""
    text = []
    for world, its in worlds(out, arms, seeds, steps).items():
        for which, label in SETS:
            got = {a: {title: series(out, a, seeds, steps, which, path) for title, path, _ in MEASURES[world]} for a in its}
            cols = [i for i in range(len(steps)) if any(v[i] == v[i] for m in got.values() for v in m.values())]
            if not cols:
                continue
            text += ["### %s, %s\n" % (WORLD[world], label),
                     "| Measure | Arm | " + " | ".join("%dk" % (steps[i] // 1000) for i in cols) + " |", "|---|---|" + "---|" * len(cols)]
            for title, _, fmt in MEASURES[world]:
                rows = [("%s: %s" % (a, name_of(a)), [fmt(got[a][title][i]) for i in cols]) for a in its if any(x == x for x in got[a][title])]
                if len(its) == 2 and len(rows) == 2:
                    a, b = its
                    rows.append(("%s − %s" % (b, a), [signed(fmt, got[b][title][i] - got[a][title][i]) for i in cols]))
                text += ["| %s | %s | " % (title, name) + " | ".join(cells) + " |" for name, cells in rows]
            text.append("")
    return "\n".join(text)


def figure(out, arms, seeds, steps, path):
    """A picture of the curves: a row a world; the exam solid, the life's own world dashed; the marks dotted."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from readme_images import COLORS, GRID, INK, MUTED_INK, SURFACE, save
    ws = worlds(out, arms, seeds, steps)
    panels = dict(breakout=[("Horizon of the ball, ticks", G + ("hold", "ball", "horizon"), ("hold_seen", "8", "ball", "horizon")),
                            ("The ball held at tick 16", G + ("hold", "ball", "held", 15), ("hold_seen", "8", "ball", "held", 15)),
                            ("Next-frame skill", G + ("next", "skill"), None)],
                  watch=[("The food held at tick 4", G + ("hold", "food", "held", 3), ("hold_seen", "2", "food", "held", 3)),
                         ("The poison's way gone at tick 4", ("course", "poison", "travel", 3), None),
                         ("Next-frame skill", G + ("next", "skill"), None)])
    order = [w for w in ("breakout", "watch") if w in ws]
    fig, axs = plt.subplots(len(order), 3, figsize=(9.4, 3.1 * len(order)), dpi=100, facecolor=SURFACE, squeeze=False)
    x = np.array(steps) / 1000.0
    for row, world in zip(axs, order):
        anchors = None
        p = os.path.join(out, "anchors_%s_work.json" % world)
        if os.path.exists(p):
            anchors = json.load(open(p))["copy"]
        for ax, (title, pth, mark) in zip(row, panels[world]):
            for a in ws[world]:
                col = COLORS[1] if a.startswith("O") else COLORS[0]
                for which, ls in (("work", "-"), ("home64", (0, (4, 2)))):
                    v = np.array(series(out, a, seeds, steps, which, pth))
                    ok = v == v
                    if ok.any():
                        ax.plot(x[ok], v[ok], color=col, lw=2, ls=ls, marker="o", ms=3.5, solid_capstyle="round")
            if mark and anchors is not None:
                m = W._get(anchors, *mark)
                if m == m:      # named in the legend only: a label on the line sits on the curves that cross it
                    ax.axhline(m, color=MUTED_INK, lw=1.2, ls=(0, (1, 2)))
            ax.set_title("%s\n%s" % (WORLD[world], title), fontsize=9.5, color=INK, loc="left")   # two lines: one would run into the next panel
            ax.set_facecolor(SURFACE)
            ax.grid(axis="y", color=GRID, lw=1)
            ax.set_axisbelow(True)
            ax.set_xlabel("ticks of life, thousands", fontsize=9, color=MUTED_INK)
            ax.tick_params(colors=MUTED_INK, labelsize=8.5, length=0)
            for sp in ax.spines.values():
                sp.set_visible(False)
    from matplotlib.lines import Line2D
    used = [a for w in order for a in ws[w]]
    handles = []
    if any(a.startswith("S") for a in used):
        handles.append(Line2D([], [], color=COLORS[0], lw=2, label=name_of(next(a for a in used if a.startswith("S")))))
    if any(a.startswith("O") for a in used):
        handles.append(Line2D([], [], color=COLORS[1], lw=2, label="no self-forcing"))
    handles += [Line2D([], [], color=MUTED_INK, lw=2, label="the exam"), Line2D([], [], color=MUTED_INK, lw=2, ls=(0, (4, 2)), label="the life's own world"),
                Line2D([], [], color=MUTED_INK, lw=1.2, ls=(0, (1, 2)), label="a straight line through the frames as seen")]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), frameon=False, fontsize=8.5, labelcolor=MUTED_INK)
    fig.tight_layout(rect=(0, 0.07 if len(order) == 1 else 0.045, 1, 1))
    save(fig, path)
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--arms", nargs="+", default=["OD", "SD3", "OB", "SB3"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[1])
    ap.add_argument("--steps", type=int, nargs="+", default=list(STEPS))
    ap.add_argument("-o", "--path", default=None)
    a = ap.parse_args()
    if a.path:
        print("wrote", figure(a.out, a.arms, a.seeds, a.steps, a.path))
    else:
        print(tables(a.out, a.arms, a.seeds, a.steps))


if __name__ == "__main__":
    main()

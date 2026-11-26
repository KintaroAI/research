"""Lived lifetimes by windows of ticks, paired by seed: the tables of vm-00117 (task 00130, two worlds, one set of
weights) and vm-00118 (task 00131, action repeat), printed as Markdown.

Subcommands:
    two-worlds [--seeds 1 2 ...]
        — vm-00117. Into Breakout: TB (from TT's weights) against BD (from scratch), the hit share by 10k ticks, the
          paired differences and the first 5k window at a share. Into the food world: BF (from BD's weights) against TT.
          Forgetting: the old world with learning off after 10k, 30k and 60k ticks of the new one, against the first
          life's own frozen test (vm-00114's check/TT_ring; vm-00116's ring arm) and an untrained model's (the floor),
          as the share kept = (after - floor) / (before - floor) on the seeds' means, with the paired distance of
          `after` from the halfway point between floor and before. Savings: the third life (TBF, BFB) against the
          first life's first 30k ticks, and the ticks to half of the first life's 50-60k level.
    repeat [--seeds 1 2 ...]
        — vm-00118. BA (online.act_every = 4) against BD: the hit share by 10k ticks, the paired differences, the
          first 5k window at a share, learning off (BA's frozen60k against BD's ring arm of vm-00116, the same courts)
          and the ticks per second. Then the miss autopsy's recordings by seed (analyze_misses.py: BA's in
          exp_00118/autopsy, BD's in exp_00118/autopsy_BD or, without them, vm-00116's): the last 8 ticks before a
          landing, what 4 ticks do to the paddle, how it moves and what its candidates are, the paddle 4 ticks before
          the landing; and the test in which each transformer picks from an even grid of dx (grid_BA, grid_BD).
    pg [--seeds 1 2 ...]
        — vm-00115 (task 00128, the move token as a policy-gradient actor). PB against BD (Breakout) and PG against TT
          (the food world) by 10k ticks with the paired differences, and by 10k ticks the move token's entropy, the
          share of positive weights and the size of the advantage. Seeds default to 1-6 (the runs were cut to them).
    check
        — the helpers against numbers already published (vm-00101, vm-00103, vm-00110).
Seeds default to 1-12; a seed with a run missing is left out of every table and named.

A food-world row counts the food (`step_hits`) and the poison of its window; Breakout's hit and miss counters are
cumulative. A window a-b is ticks a + 1 .. b.
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from analyze_misses import D, paired  # noqa: E402

K = 1000
TEN = [(a * K, (a + 10) * K) for a in range(0, 60, 10)]
_rows = {}


# ------------------------------------------------------------------------------------------------- reading the runs
def rows(path):
    if path not in _rows:
        _rows[path] = [json.loads(line) for line in open(os.path.join(path, "metrics.jsonl"))]
    return _rows[path]


def has(path, end=None):
    """The run exists and (with `end`) has reached that tick."""
    f = os.path.join(path, "metrics.jsonl")
    return os.path.exists(f) and os.path.getsize(f) > 0 and (end is None or rows(path)[-1]["step"] >= end)


def share(path, a, b):
    """Breakout: the hit share over ticks a + 1 .. b."""
    def at(t):
        r = [x for x in rows(path) if x["step"] <= t]
        return (r[-1]["bo_hits"], r[-1]["bo_misses"]) if r else (0, 0)
    (h1, m1), (h0, m0) = at(b), at(a)
    n = h1 - h0 + m1 - m0
    return (h1 - h0) / n if n else float("nan")


def per1k(path, key, a, b):
    """The food world: the count `key` (step_hits: food; poison) per 1,000 body-ticks over ticks a + 1 .. b."""
    r = [x for x in rows(path) if a < x["step"] <= b]
    return 1000 * sum(x[key] / x.get("bodies", 1) for x in r) / (b - a)


def food(path, a, b):
    return per1k(path, "step_hits", a, b)


def poison(path, a, b):
    return per1k(path, "poison", a, b)


def mean_of(path, key):
    """The mean of a logged value over a run's rows (a frozen test's losses: the weights do not change)."""
    return float(np.mean([x[key] for x in rows(path)]))


def first_window(value, level, end, width=5 * K, step=K):
    """The end tick of the first `width`-tick window (moved by `step`) whose value is at or above `level`; None if none."""
    for t in range(width, end + 1, step):
        v = value(t - width, t)
        if v == v and v >= level:
            return t
    return None


# ------------------------------------------------------------------------------------------------- printing
def f3(x):
    """A share: .678, −.092."""
    if x != x:
        return "–"
    s = "%.3f" % abs(x)
    return ("−" if x < 0 and float(s) else "") + (s[1:] if s.startswith("0") else s)


def f1(x):
    if x != x:
        return "–"
    s = "%.1f" % abs(x)
    return ("−" if x < 0 and float(s) else "") + s


def f2(x):
    s = "%.2f" % abs(x)
    return ("−" if x < 0 and float(s) else "") + s


def signed(fmt, mean, half):
    return ("−" if mean < 0 else "+") + fmt(abs(mean)) + " ± " + fmt(half)


def diff(a, b, fmt=f3):
    """Cells of the paired difference a - b: 'mean ± 95% half-width' and 'k of n above'."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = ~(np.isnan(a) | np.isnan(b))
    mean, se, half, up, down = paired(a[ok], b[ok])
    return [signed(fmt, mean, half), "%d of %d above" % (up, ok.sum())]


def table(head, body):
    print("| " + " | ".join(head) + " |")
    print("|" + "---|" * len(head))
    for r in body:
        print("| " + " | ".join(str(c) for c in r) + " |")
    print()


def win(a, b):
    return "%d–%dk" % (a // K, b // K)


def ticks(ts, n):
    """Per-seed first ticks (None: never): 'lo–hik (median mk), k of n'."""
    got = sorted(t for t in ts if t is not None)
    if not got:
        return "none"
    med = np.median(got) / K
    lo, hi = got[0] // K, got[-1] // K
    span = "%dk" % lo if lo == hi else "%d–%dk" % (lo, hi)
    return span + " (median %sk)" % ("%d" % med if med == int(med) else "%.1f" % med) + ("" if len(got) == n else ", %d of %d" % (len(got), n))


def keep_seeds(seeds, needs):
    """The seeds whose every needed run exists (and has reached its end); the others are named."""
    good = [s for s in seeds if all(has(p % s, end) for p, end in needs)]
    gone = [s for s in seeds if s not in good]
    if gone:
        print("**Seeds left out (a run missing or unfinished): %s.**\n" % ", ".join(map(str, gone)))
    if len(good) < 2:
        raise SystemExit("fewer than 2 complete seeds")
    print("Seeds: %s (%d).\n" % (", ".join(map(str, good)), len(good)))
    return good


def court_tables(name, new, base, bname, seeds, end, pre, levels=(.5, .7)):
    """A Breakout arm against its baseline: by 10k, the paired differences over `pre` windows, the first 5k window."""
    wins = [w for w in TEN if w[1] <= end]
    A = {w: [share(new % s, *w) for s in seeds] for w in wins + pre}
    B = {w: [share(base % s, *w) for s in seeds] for w in wins + pre}
    table(["Ticks", "%s (%d)" % (bname, len(seeds)), "%s (%d)" % (name, len(seeds)), "%s − %s (95%%)" % (name, bname), "Seeds"],
          [[win(*w), f3(np.mean(B[w])), f3(np.mean(A[w]))] + diff(A[w], B[w]) for w in wins])
    table(["Comparison", "Ticks", bname, name, "Mean difference (95%)", "Seeds"],
          [["%s − %s" % (name, bname), win(*w), f3(np.mean(B[w])), f3(np.mean(A[w]))] + diff(A[w], B[w]) for w in pre])
    body = []
    for lv in levels:
        ta = [first_window(lambda a, b: share(new % s, a, b), lv, end) for s in seeds]
        tb = [first_window(lambda a, b: share(base % s, a, b), lv, end) for s in seeds]
        both = [(x, y) for x, y in zip(ta, tb) if x is not None and y is not None]
        cell = "–"
        if len(both) > 1:
            m, se, half, up, down = paired([x for x, _ in both], [y for _, y in both])
            cell = "%s%.1fk ± %.1fk (%d earlier, %d later of %d)" % ("−" if m < 0 else "+", abs(m) / K, half / K, down, up, len(both))
        body.append([f3(lv), ticks(ta, len(seeds)), ticks(tb, len(seeds)), cell])
    table(["First 5k window at or above", name, bname, "%s − %s, seeds reaching it in both" % (name, bname)], body)
    return A, B


# ------------------------------------------------------------------------------------------------- vm-00117
def two_worlds(seeds):
    O = D + "/exp_00117"
    TT, BD = D + "/exp_00101/TT_s%d", D + "/exp_00103/BD_s%d"
    TB, BF, TBF, BFB = O + "/TB_s%d", O + "/BF_s%d", O + "/TBF_s%d", O + "/BFB_s%d"
    before_f, before_c = D + "/exp_00114/check/TT_ring/s%d", D + "/exp_00116/ring/s%d"
    old = {(w, t): O + "/old_%s_%dk/s%%d" % (w, t) for w in ("TB", "BF") for t in (10, 30, 60)}
    floor_f, floor_c = O + "/floor_food/s%d", O + "/floor_court/s%d"
    seeds = keep_seeds(seeds, [(TT, 60 * K), (BD, 60 * K), (TB, 60 * K), (BF, 60 * K),
                               (before_f, 2 * K), (before_c, 6 * K), (floor_f, 2 * K), (floor_c, 6 * K)]
                       + [(old[k], 2 * K if k[0] == "TB" else 6 * K) for k in old])
    n = len(seeds)
    third = [s for s in seeds if has(TBF % s, 30 * K) and has(BFB % s, 30 * K)]      # sections 4 and 5 only
    pre = [(0, 10 * K), (10 * K, 30 * K), (40 * K, 60 * K), (50 * K, 60 * K)]

    print("### 1. Into Breakout: TB (from TT's 60k weights) against BD (from scratch), hit share\n")
    court_tables("TB", TB, BD, "BD", seeds, 60 * K, pre)

    print("### 2. Into the food world: BF (from BD's 60k weights) against TT (from scratch), food and poison per 1k\n")
    wins = TEN
    F = {a: {w: [food(p % s, *w) for s in seeds] for w in wins + pre} for a, p in (("BF", BF), ("TT", TT))}
    P = {a: {w: [poison(p % s, *w) for s in seeds] for w in wins + pre} for a, p in (("BF", BF), ("TT", TT))}
    table(["Ticks", "TT food", "BF food", "BF − TT food (95%)", "Seeds", "TT poison", "BF poison", "BF − TT poison (95%)", "Seeds"],
          [[win(*w), f1(np.mean(F["TT"][w])), f1(np.mean(F["BF"][w]))] + diff(F["BF"][w], F["TT"][w], f1)
           + [f1(np.mean(P["TT"][w])), f1(np.mean(P["BF"][w]))] + diff(P["BF"][w], P["TT"][w], f1) for w in wins])
    table(["Comparison", "Ticks", "Food per 1k (95%)", "Seeds", "Poison per 1k (95%)", "Seeds"],
          [["BF − TT", win(*w)] + diff(F["BF"][w], F["TT"][w], f1) + diff(P["BF"][w], P["TT"][w], f1) for w in pre])

    print("### 3. Forgetting: the old world with learning off, after 10k, 30k and 60k ticks of the new world\n")

    def kept(after, before, floor):
        """The share kept on the seeds' means, and the paired distance of `after` from the halfway point."""
        a, b, f = (np.asarray(x, float) for x in (after, before, floor))
        return (a.mean() - f.mean()) / (b.mean() - f.mean()), a - (b + f) / 2

    def forget(title, value, fmt, before, floor, olds, end):
        b = [value(before % s, 0, end) for s in seeds]
        f = [value(floor % s, 0, end) for s in seeds]
        body = [["before (the first life's 60k weights)", fmt(np.mean(b)), "1", "", "", "", ""]]
        for t in (10, 30, 60):
            a = [value(olds[t] % s, 0, end) for s in seeds]
            k, half_d = kept(a, b, f)
            m, se, half, up, down = paired(half_d, np.zeros(n))
            body.append(["after %dk ticks of the new world" % t, fmt(np.mean(a)), f2(k)] + diff(a, f, fmt)
                        + [signed(fmt, m, half), "%d above, %d below" % (up, down)])
        body.append(["floor (an untrained model)", fmt(np.mean(f)), "0", "", "", "", ""])
        print("**%s**\n" % title)
        table(["Weights", "Mean of %d seeds" % n, "Share kept", "After − floor (95%)", "Seeds", "After − halfway point (95%)", "Seeds"], body)
        m, se, half, up, down = paired(b, f)
        print("Before − floor: %s (%d of %d seeds above).\n" % (signed(fmt, m, half), up, n))

    TBo = {t: old[("TB", t)] for t in (10, 30, 60)}
    BFo = {t: old[("BF", t)] for t in (10, 30, 60)}
    forget("TB's weights in the food world (2,000 ticks): food per 1k", food, f1, before_f, floor_f, TBo, 2 * K)
    forget("TB's weights in the food world (2,000 ticks): poison per 1k", poison, f1, before_f, floor_f, TBo, 2 * K)
    forget("BF's weights in Breakout (6,000 ticks): hit share", share, f3, before_c, floor_c, BFo, 6 * K)

    print("**The old world's prediction, learning off: cross-entropy of the frame's patches, the body and the move** "
          "(the mean over the test's rows; not among the readings set beforehand)\n")
    body = []
    for name, before, floor, olds in (("food world, TB's weights", before_f, floor_f, TBo), ("Breakout, BF's weights", before_c, floor_c, BFo)):
        for lab, p in [("before", before)] + [("after %dk" % t, olds[t]) for t in (10, 30, 60)] + [("floor", floor)]:
            body.append([name, lab] + ["%.2f" % np.mean([mean_of(p % s, k) for s in seeds]) for k in ("uni_ce_patch", "uni_ce_body", "uni_ce_move")])
    table(["Old world", "Weights", "Patches", "Body", "Move"], body)

    print("### 4. Savings: the third life, back in the old world, against the first life's first 30k ticks\n")

    def savings(ss):
        w3 = [w for w in TEN if w[1] <= 30 * K] + [(10 * K, 30 * K)]
        Ff = {a: {w: [food(p % s, *w) for s in ss] for w in w3} for a, p in (("TBF", TBF), ("TT", TT))}
        Pf = {a: {w: [poison(p % s, *w) for s in ss] for w in w3} for a, p in (("TBF", TBF), ("TT", TT))}
        table(["Ticks", "TT food", "TBF food", "TBF − TT food (95%)", "Seeds", "TT poison", "TBF poison", "TBF − TT poison (95%)", "Seeds"],
              [[win(*w), f1(np.mean(Ff["TT"][w])), f1(np.mean(Ff["TBF"][w]))] + diff(Ff["TBF"][w], Ff["TT"][w], f1)
               + [f1(np.mean(Pf["TT"][w])), f1(np.mean(Pf["TBF"][w]))] + diff(Pf["TBF"][w], Pf["TT"][w], f1) for w in w3])
        Sc = {a: {w: [share(p % s, *w) for s in ss] for w in w3} for a, p in (("BFB", BFB), ("BD", BD))}
        table(["Ticks", "BD", "BFB", "BFB − BD (95%)", "Seeds"],
              [[win(*w), f3(np.mean(Sc["BD"][w])), f3(np.mean(Sc["BFB"][w]))] + diff(Sc["BFB"][w], Sc["BD"][w]) for w in w3])

        print("**The third life against the second life of the other order** (both enter the world with the other world's "
              "habits; only the third has lived in it before; not among the readings set beforehand)\n")
        Fb = {w: [food(BF % s, *w) for s in ss] for w in w3}
        Pb = {w: [poison(BF % s, *w) for s in ss] for w in w3}
        table(["Ticks", "BF food", "TBF food", "TBF − BF food (95%)", "Seeds", "BF poison", "TBF poison", "TBF − BF poison (95%)", "Seeds"],
              [[win(*w), f1(np.mean(Fb[w])), f1(np.mean(Ff["TBF"][w]))] + diff(Ff["TBF"][w], Fb[w], f1)
               + [f1(np.mean(Pb[w])), f1(np.mean(Pf["TBF"][w]))] + diff(Pf["TBF"][w], Pb[w], f1) for w in w3])
        St = {w: [share(TB % s, *w) for s in ss] for w in w3}
        table(["Ticks", "TB", "BFB", "BFB − TB (95%)", "Seeds"],
              [[win(*w), f3(np.mean(St[w])), f3(np.mean(Sc["BFB"][w]))] + diff(Sc["BFB"][w], St[w]) for w in w3])

        last = (50 * K, 60 * K)
        body = []
        for name, back, first, value, fmt in (("food per 1k: TBF, TT", TBF, TT, food, f1), ("hit share: BFB, BD", BFB, BD, share, f3)):
            lvl = [value(first % s, *last) for s in ss]
            t3 = [first_window(lambda a, b: value(back % s, a, b), l / 2, 30 * K) for s, l in zip(ss, lvl)]
            t1 = [first_window(lambda a, b: value(first % s, a, b), l / 2, 60 * K) for s, l in zip(ss, lvl)]
            both = [(x, y) for x, y in zip(t3, t1) if x is not None and y is not None]
            cell = "–"
            if len(both) > 1:
                m, se, half, up, down = paired([x for x, _ in both], [y for _, y in both])
                cell = "%s%.1fk ± %.1fk (%d earlier, %d later of %d)" % ("−" if m < 0 else "+", abs(m) / K, half / K, down, up, len(both))
            end3 = [value(back % s, 20 * K, 30 * K) for s in ss]
            body.append([name, fmt(np.mean(lvl)), ticks(t3, len(ss)), ticks(t1, len(ss)), cell, fmt(np.mean(end3))] + diff(end3, lvl, fmt))
        table(["Measure: third life, first life", "First life at 50–60k", "Third life: first 5k window at half of it", "First life: the same",
               "Third − first", "Third life at 20–30k", "Third at 20–30k − first at 50–60k (95%)", "Seeds"], body)

    if len(third) < 2:
        print("Not run yet.\n")
    else:
        if third != seeds:
            print("**Seeds with a third life: %s (%d).**\n" % (", ".join(map(str, third)), len(third)))
        savings(third)

    print("### 5. How the lives move (logged in life; not among the readings set beforehand)\n")
    print("Smoothness is the mean cosine between one tick's move and the next; capped is the share of ticks at the hunger cap.\n")

    def logged(p, key, a, b):
        return float(np.mean([x[key] for x in rows(p) if a < x["step"] <= b]))

    for world, arms in (("The food world", (("TT", TT, 60), ("BF", BF, 60), ("TBF", TBF, 30))), ("Breakout", (("BD", BD, 60), ("TB", TB, 60), ("BFB", BFB, 30)))):
        body = []
        for key, lab in (("smooth", "smoothness"), ("hunger_capped", "capped")):
            for name, p, end in arms:
                ss = third if end == 30 else seeds
                if len(ss) < 2:
                    continue
                note = "" if ss == seeds else " (%d seeds)" % len(ss)
                body.append(["%s%s, %s" % (name, note, lab)] + [f2(np.mean([logged(p % s, key, *w) for s in ss])) if w[1] <= end * K else "" for w in TEN])
        print("**%s**\n" % world)
        table(["Arm"] + [win(*w) for w in TEN], body)


# ------------------------------------------------------------------------------------------------- vm-00118
def repeat(seeds):
    O = D + "/exp_00118"
    BA, BD = O + "/BA_s%d", D + "/exp_00103/BD_s%d"
    fz, ring = O + "/frozen60k/s%d", D + "/exp_00116/ring/s%d"
    seeds = keep_seeds(seeds, [(BA, 60 * K), (BD, 60 * K)])
    n = len(seeds)
    print("### 1. Lived: BA (a move decided every 4 ticks and held) against BD, hit share\n")
    court_tables("BA", BA, BD, "BD", seeds, 60 * K, [(10 * K, 30 * K), (20 * K, 40 * K), (40 * K, 60 * K), (50 * K, 60 * K)], levels=(.5, .7, .9))
    held = [mean_of(BA % s, "held") for s in seeds]
    print("Held moves: %.3f of body-ticks (%.3f to %.3f across seeds).\n" % (np.mean(held), min(held), max(held)))
    best = {name: [max(share(p % s, t - 5 * K, t) for t in range(5 * K, 60 * K + 1, K)) for s in seeds] for name, p in (("BA", BA), ("BD", BD))}
    print("The best 5k window of each life: BA %s (mean %s); BD %s (mean %s).\n" % (
        ", ".join(f3(x) for x in best["BA"]), f3(np.mean(best["BA"])), ", ".join(f3(x) for x in best["BD"]), f3(np.mean(best["BD"]))))
    table(["Seed"] + [str(s) for s in seeds],
          [["BD, 40–60k"] + [f3(share(BD % s, 40 * K, 60 * K)) for s in seeds], ["BA, 40–60k"] + [f3(share(BA % s, 40 * K, 60 * K)) for s in seeds]])

    print("### 2. Learning off at 60k: 6,000 ticks in new courts (seed 100 + s)\n")
    fs = [s for s in seeds if has(fz % s, 6 * K) and has(ring % s, 6 * K)]
    if len(fs) < 2:
        print("Not run yet.\n")
    else:
        if fs != seeds:
            print("Seeds with the test: %s.\n" % ", ".join(map(str, fs)))
        a = [share(fz % s, 0, 6 * K) for s in fs]
        b = [share(ring % s, 0, 6 * K) for s in fs]
        lived_a = [share(BA % s, 50 * K, 60 * K) for s in fs]
        lived_b = [share(BD % s, 50 * K, 60 * K) for s in fs]
        table(["", "BD (%d)" % len(fs), "BA (%d)" % len(fs), "BA − BD (95%)", "Seeds"],
              [["learning off, 60k weights", f3(np.mean(b)), f3(np.mean(a))] + diff(a, b),
               ["lived, 50–60k", f3(np.mean(lived_b)), f3(np.mean(lived_a))] + diff(lived_a, lived_b),
               ["learning off − lived (95%)", diff(b, lived_b)[0], diff(a, lived_a)[0], "", ""]])
        table(["Seed"] + [str(s) for s in fs], [["BD"] + [f3(x) for x in b], ["BA"] + [f3(x) for x in a]])
        balls = [sum(rows(p % s)[-1][k] for k in ("bo_hits", "bo_misses")) for p in (fz, ring) for s in fs]
        print("Balls per test: %d to %d.\n" % (min(balls), max(balls)))

    print("### 3. How the lives move, and their speed (logged in life)\n")

    def logged(p, key, a, b):
        return float(np.mean([x[key] for x in rows(p) if a < x["step"] <= b]))

    body = []
    for key, lab, fmt in (("smooth", "cosine between one move and the next", "%.2f"), ("instinct_override", "ticks on which the instinct overrules the model's own move", "%.2f"),
                          ("uni_ce_move", "the move token's cross-entropy", "%.2f"), ("hunger_capped", "share of ticks at the hunger cap", "%.2f")):
        for name, p in (("BD", BD), ("BA", BA)):
            body.append(["%s, %s" % (name, lab)] + [fmt % np.mean([logged(p % s, key, *w) for s in seeds]) for w in TEN])
    table(["Arm, measure"] + [win(*w) for w in TEN], body)

    def tps(p):
        r = rows(p)
        return r[-1]["step"] / r[-1]["time"]
    body = []                                                  # BD's seeds 1-6 ran on the RTX 3090 (vm-00103); all else on the RTX 4090
    for name, cards, grp in (("1–6", "BD on the RTX 3090, BA on the RTX 4090", [s for s in seeds if s <= 6]),
                             ("7–12", "both on the RTX 4090", [s for s in seeds if s > 6])):
        if grp:
            body.append([name, "%.2f" % np.mean([tps(BD % s) for s in grp]), "%.2f" % np.mean([tps(BA % s) for s in grp]), cards])
    table(["Seeds", "BD, ticks per second", "BA, ticks per second", "Cards (six lives at a time)"], body)

    print("### 4. The miss autopsy by seed: BA's recordings against BD's (vm-00116), the same courts\n")
    ra, rb = O + "/autopsy/s%d.npz", O + "/autopsy_BD/s%d.npz"   # BD recorded again here with its candidates kept (make autopsy-bd)
    if not all(os.path.exists(rb % s) for s in seeds):
        rb = D + "/exp_00116/autopsy/BD_60000/s%d.npz"            # vm-00116's own recordings: the same play, no candidates
    rs = [s for s in seeds if os.path.exists(ra % s) and os.path.exists(rb % s)]
    if len(rs) < 2:
        print("Not run yet.\n")
        return
    if rs != seeds:
        print("Seeds with both recordings: %s.\n" % ", ".join(map(str, rs)))
    from analyze_misses import endgame, last_ticks, moves, spans
    ZA, ZB = [np.load(ra % s) for s in rs], [np.load(rb % s) for s in rs]
    k = int(ZA[0]["act_every"])
    head = ["BD (%d)" % len(rs), "BA (%d)" % len(rs), "BA − BD (95%)", "Seeds"]

    def lines(fn, names):
        A, B = [fn(z) for z in ZA], [fn(z) for z in ZB]
        body = []
        for key, lab, fmt in names:
            a, b = [x[key] for x in A], [x[key] for x in B]
            body.append([lab, fmt(np.nanmean(b)), fmt(np.nanmean(a))] + diff(a, b, fmt))
        return body

    f4 = lambda x: ("−" if x < 0 else "") + ("%.4f" % abs(x))[1:]         # noqa: E731
    print("The last 8 ticks before a landing (BA: its deciding ticks; BD decides every tick, so for BD the order after the hold is the order by one tick):\n")
    table(["Measure"] + head, lines(last_ticks, (
        ("returned", "Balls returned, the whole recording", f3), ("toward", "Move made goes toward the ball", f3),
        ("order", "Label's order of the candidates, by one tick of each", f3), ("order_hold", "The same, by where the hold leaves the paddle", f3),
        ("clear", "Label's order of pairs ending 4+ px apart, by one tick", f3), ("clear_hold", "The same, by where the hold leaves the paddle", f3),
        ("grid_toward", "The grid's best dx goes toward the ball", f3),
        ("r_need", "Label's correlation with the px left across the candidates, by one tick", f3), ("r_need_hold", "The same, after the hold", f3),
        ("spread", "Label's spread over the candidates (median)", f4))))
    for upto in (16,):
        print("The same over the last %d ticks:\n" % upto)
        table(["Measure"] + head, lines(lambda z: last_ticks(z, upto), (
            ("toward", "Move made goes toward the ball", f3), ("order", "Label's order of the candidates, by one tick of each", f3),
            ("order_hold", "The same, by where the hold leaves the paddle", f3))))
    print("What %d ticks do to the paddle: spans of %d ticks from a deciding tick (BD: every %dth tick), starting %d to 16 ticks before a landing "
          "(the catching zone is 14 px wide, a full move held for %d ticks 18 px):\n" % (k, k, k, k + 1, k))
    sp = lambda z: spans(z, k)                                           # noqa: E731
    table(["Spans of %d ticks" % k] + head, lines(sp, (
        ("travel", "The paddle's net travel over the span, px", f1), ("away", "Spans that start with the paddle not under the landing point", f3),
        ("left", "Start under it: end outside the catching zone", f3),
        ("all_under", "Start not under it: end under it", f3), ("all_past", "Start not under it: end past it, on the other side", f3),
        ("all_short", "Start not under it: end nearer, still short", f3), ("all_worse", "Start not under it: end no nearer", f3),
        ("near_under", "Start within 8 px of the zone: end under it", f3), ("near_past", "Start within 8 px: end past it", f3),
        ("near_short", "Start within 8 px: end nearer, still short", f3), ("near_worse", "Start within 8 px: end no nearer", f3),
        ("far_under", "Start further off: end under it", f3), ("far_past", "Start further off: end past it", f3),
        ("far_short", "Start further off: end nearer, still short", f3), ("far_worse", "Start further off: end no nearer", f3))))
    n = [x["spans"] for z in (ZA, ZB) for x in map(sp, z)]
    print("Spans per recording: %d to %d.\n" % (min(n), max(n)))
    print("How the paddle moves, and what the instinct is given to choose from (deciding ticks with a ball in play):\n")
    mv = lambda z: moves(z, k)                                           # noqa: E731
    names = [("wall", "Ticks on which the move is cut by a wall (all ticks)", f3), ("size", "Size of the move, abs dx (1 is full speed)", f3),
             ("same", "A decision goes the same way as the one before (BD: as the move %d ticks before)" % k, f3),
             ("toward_near", "Move goes toward the ball, %d to 16 ticks before a landing, paddle within 8 px of the zone" % (k + 1), f3),
             ("toward_far", "The same, paddle further off", f3)]
    if all("cand_copy" in z.files for z in ZA + ZB):
        names += [("cand_copy", "Of a transformer's 16 candidates: the share that repeat the move made on the tick before", f3),
                  ("cand_distinct", "Of its 16 candidates: different values of dx", f1), ("cand_sd", "Standard deviation of its candidates' dx", f3)]
    table(["Measure"] + head, lines(mv, names))
    A = [mv(z) for z in ZA]
    if "cand_copy_held" in A[0]:
        print("BA on its held ticks, where the pick is not made: %s of the 16 candidates repeat the move made on the tick before, %s different values of dx, "
              "standard deviation %s. BD's moves go the same way as on the tick before on %s of ticks.\n" % (
                  f3(np.mean([x["cand_copy_held"] for x in A])), f1(np.mean([x["cand_distinct_held"] for x in A])),
                  f3(np.mean([x["cand_sd_held"] for x in A])), f3(np.mean([mv(z)["same1"] for z in ZB]))))
    print("The paddle %d ticks before the landing, and the ball's outcome:\n" % k)
    table(["Per ball"] + head, lines(lambda z: endgame(z, k), (
        ("under", "Balls with the paddle under the landing point %d ticks before" % k, f3), ("near", "Balls with it within 8 px of the zone", f3),
        ("ret_under", "Returned, of the balls it was under", f3), ("ret_near", "Returned, of the balls within 8 px", f3),
        ("ret_far", "Returned, of the balls further off", f3))))

    print("### 5. The same weights, each transformer picking the best dx of an even grid of 16 by its own label (a test added after the results)\n")
    ga, gb = O + "/grid_BA/s%d.npz", O + "/grid_BD/s%d.npz"
    gs = [s for s in rs if os.path.exists(ga % s) and os.path.exists(gb % s)]
    if len(gs) < 2:
        print("Not run yet.\n")
        return
    if gs != rs:
        print("Seeds with the test: %s.\n" % ", ".join(map(str, gs)))
    own = {"BA": [np.load(ra % s) for s in gs], "BD": [np.load(rb % s) for s in gs]}
    grd = {"BA": [np.load(ga % s) for s in gs], "BD": [np.load(gb % s) for s in gs]}
    R = {(a, m): [last_ticks(z)["returned"] for z in Z[a]] for m, Z in (("own", own), ("grid", grd)) for a in ("BA", "BD")}
    gain = {a: np.array(R[a, "grid"]) - np.array(R[a, "own"]) for a in ("BA", "BD")}
    body = [[a, f3(np.mean(R[a, "own"])), f3(np.mean(R[a, "grid"]))] + diff(R[a, "grid"], R[a, "own"]) for a in ("BD", "BA")]
    body.append(["BA − BD (95%)", diff(R["BA", "own"], R["BD", "own"])[0], diff(R["BA", "grid"], R["BD", "grid"])[0]] + diff(gain["BA"], gain["BD"]))
    table(["Balls returned", "Its own candidates", "The even grid", "Grid − own (95%)", "Seeds"], body)
    table(["Seed"] + [str(s) for s in gs], [["%s, %s" % (a, m)] + [f3(x) for x in R[a, m]] for a in ("BD", "BA") for m in ("own", "grid")])
    body = []
    for key, lab, fn in (("toward", "Move made goes toward the ball, the last 8 ticks", last_ticks), ("same", "A decision goes the same way as the one before (BD: as the move %d ticks before)" % k, mv),
                         ("wall", "Ticks on which the move is cut by a wall", mv), ("size", "Size of the move, abs dx", mv),
                         ("left", "Spans of %d ticks that start under the landing point and end outside the zone" % k, sp),
                         ("ret_under", "Returned, of the balls the paddle was under %d ticks before" % k, lambda z: endgame(z, k))):
        row = [lab]
        for a in ("BD", "BA"):
            x, y = [fn(z)[key] for z in own[a]], [fn(z)[key] for z in grd[a]]
            row += [f3(np.mean(x)), f3(np.mean(y)), diff(y, x)[0]]
        body.append(row)
    table(["Measure", "BD, own", "BD, grid", "BD, grid − own (95%)", "BA, own", "BA, grid", "BA, grid − own (95%)"], body)


# ------------------------------------------------------------------------------------------------- vm-00115
def pg(seeds):
    O = D + "/exp_00115"
    PB, PG, TT, BD = O + "/PB_s%d", O + "/PG_s%d", D + "/exp_00101/TT_s%d", D + "/exp_00103/BD_s%d"
    seeds = keep_seeds(seeds, [(PB, 60 * K), (PG, 60 * K), (TT, 60 * K), (BD, 60 * K)])
    pre = [(10 * K, 30 * K), (40 * K, 60 * K), (50 * K, 60 * K)]
    print("### 1. Breakout: PB (the policy-gradient actor) against BD (the instinct), hit share\n")
    court_tables("PB", PB, BD, "BD", seeds, 60 * K, pre, levels=(.5,))
    best = [max(share(PB % s, t - 5 * K, t) for t in range(5 * K, 60 * K + 1, K)) for s in seeds]
    print("PB's best 5k window of the life, per seed: %s.\n" % ", ".join(f3(x) for x in best))

    print("### 2. The food world: PG against TT, food and poison per 1k\n")
    F = {a: {w: [food(p % s, *w) for s in seeds] for w in TEN + pre} for a, p in (("PG", PG), ("TT", TT))}
    P = {a: {w: [poison(p % s, *w) for s in seeds] for w in TEN + pre} for a, p in (("PG", PG), ("TT", TT))}
    table(["Ticks", "TT food", "PG food", "PG − TT food (95%)", "Seeds", "TT poison", "PG poison", "PG − TT poison (95%)", "Seeds"],
          [[win(*w), f1(np.mean(F["TT"][w])), f1(np.mean(F["PG"][w]))] + diff(F["PG"][w], F["TT"][w], f1)
           + [f1(np.mean(P["TT"][w])), f1(np.mean(P["PG"][w]))] + diff(P["PG"][w], P["TT"][w], f1) for w in TEN])
    table(["Comparison", "Ticks", "Food per 1k (95%)", "Seeds", "Poison per 1k (95%)", "Seeds"],
          [["PG − TT", win(*w)] + diff(F["PG"][w], F["TT"][w], f1) + diff(P["PG"][w], P["TT"][w], f1) for w in pre])

    print("### 3. The actor over the life (means of the logged rows and of the seeds; the entropy's range across seeds in brackets)\n")

    def logged(p, key, a, b):
        return float(np.mean([x[key] for x in rows(p) if a < x["step"] <= b]))

    def line(name, p, key, lab, fmt, span=False):
        cells = []
        for w in TEN:
            v = [logged(p % s, key, *w) for s in seeds]
            cells.append(fmt % np.mean(v) + (" (%.2f–%.2f)" % (min(v), max(v)) if span else ""))
        return ["%s, %s" % (name, lab)] + cells

    body = []
    for name, p, base, bp in (("PB", PB, "BD", BD), ("PG", PG, "TT", TT)):
        body.append(line(name, p, "move_ent", "the move token's entropy, nats", "%.2f", True))
        body.append(line(name, p, "pg_pos", "share of positive weights", "%.2f"))
        body.append(line(name, p, "pg_abs", "mean size of the advantage", "%.4f"))
        for key, lab, fmt in (("uni_ce_move", "the move token's cross-entropy on the batch's moves", "%.1f"),
                              ("hunger_capped", "share of ticks at the hunger cap", "%.2f"),
                              ("smooth", "cosine between one move and the next", "%.2f")):
            body.append(line(name, p, key, lab, fmt))
            body.append(line(base, bp, key, lab, "%.2f" if key == "uni_ce_move" else fmt))
    table(["Arm, measure"] + [win(*w) for w in TEN], body)
    for name, p in (("PB", PB), ("PG", PG)):
        r = [rows(p % s) for s in seeds]
        print("%s: the move token's entropy in the rows at ticks 1k, 2k, 3k, 5k and 10k, mean of seeds: %s; ticks per second %.2f.\n" % (
            name, ", ".join("%.2f" % np.mean([x[i]["move_ent"] for x in r]) for i in (0, 1, 2, 4, 9)),
            np.mean([x[-1]["step"] / x[-1]["time"] for x in r])))


# ------------------------------------------------------------------------------------------------- the helpers' check
def check():
    seeds = list(range(1, 13))
    TT, BD, RB = D + "/exp_00101/TT_s%d", D + "/exp_00103/BD_s%d", D + "/exp_00110/RB_s%d"
    print("TT at 40-60k: food %.1f, poison %.1f per 1k (vm-00101: 71.6, 8.3)" % (
        np.mean([food(TT % s, 40 * K, 60 * K) for s in seeds]), np.mean([poison(TT % s, 40 * K, 60 * K) for s in seeds])))
    print("BD by 10k: " + " ".join(f3(np.mean([share(BD % s, *w) for s in seeds])) for w in TEN) + "  (vm-00110: .226 .251 .335 .442 .602 .678)")
    t = [first_window(lambda a, b: share(BD % s, a, b), .5, 60 * K) for s in seeds]
    print("BD's first 5k window at .5: " + ticks(t, 12) + "  (vm-00110: 32-55k, median 40.5k)")
    if all(has(RB % s) for s in seeds):
        w = (40 * K, 60 * K)
        print("RB - BD at 40-60k: " + " ".join(diff([share(RB % s, *w) for s in seeds], [share(BD % s, *w) for s in seeds])) + "  (vm-00110: +.092 ± .060, 10 of 12)")
        t = [first_window(lambda a, b: share(RB % s, a, b), .5, 60 * K) for s in seeds]
        print("RB's first 5k window at .5: " + ticks(t, 12) + "  (vm-00110: 17-27k, median 21.5k)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["two-worlds", "repeat", "pg", "check"])
    ap.add_argument("--seeds", type=int, nargs="+")
    args = ap.parse_args()
    seeds = args.seeds or list(range(1, 7 if args.cmd == "pg" else 13))
    {"two-worlds": lambda: two_worlds(seeds), "repeat": lambda: repeat(seeds), "pg": lambda: pg(seeds), "check": check}[args.cmd]()

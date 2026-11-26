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
          and the ticks per second.
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
    table(["Comparison", "Ticks", "Mean difference (95%)", "Seeds"],
          [["%s − %s" % (name, bname), win(*w)] + diff(A[w], B[w]) for w in pre])
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

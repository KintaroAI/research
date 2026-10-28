"""Markdown tables of universal-model lifetimes (see ANALYSIS.md): the frozen forage tests, life by segments, the dream
curve. Runs are grouped into arms by their config name (info.json); arms keep the order they are given in.

Usage (from visual-motor-agent/dev):
    ./venv/bin/python tables.py ~/data/research/visual-motor-agent/exp_00035/00?_T*_s? [--step 100000]
"""
import argparse
import glob
import json
import os
import re

import numpy as np

# the frozen forage tests (online.py --eval-tests forage): file tag -> column name, in this order
TESTS = [("_instinct0", "instinct off"), ("_h1", "1-step"), ("_h4_mean", "4-step mean"), ("_h4_pair", "4-step pair")]


def load_runs(paths):
    runs = []
    for p in paths:
        p = p.rstrip("/")
        if not os.path.exists(os.path.join(p, "info.json")):
            continue
        info = json.load(open(os.path.join(p, "info.json")))
        runs.append(dict(path=p, name=os.path.basename(p), arm=info["config"].get("name") or os.path.basename(p),
                         seed=info.get("seed"), steps=info["config"]["online"]["steps"]))
    return runs


def last_step(run):
    """The newest checkpoint step that has forage test results."""
    steps = [int(m.group(1)) for f in glob.glob(os.path.join(run["path"], "eval", "ckpt_*forage*.json"))
             if (m := re.search(r"ckpt_(\d{8})", f))]
    return max(steps) if steps else None


def forage(run, step, tag):
    f = os.path.join(run["path"], "eval", "ckpt_%08d%s_forage16.json" % (step, tag))
    if not os.path.exists(f):
        cands = glob.glob(os.path.join(run["path"], "eval", "ckpt_%08d%s_forage*.json" % (step, tag)))
        if not cands:
            return None
        f = cands[0]
    j = json.load(open(f))
    return j["forage_hits_per_1k"], j.get("forage_poison_per_1k", float("nan"))


def cell(fp):
    if fp is None:
        return "–"
    f, p = fp
    return "%.1f / %.1f (%.2f)" % (f, p, f / p) if p > 0 else "%.1f / %.1f" % (f, p)


def frozen_tables(runs, step):
    print("### Frozen at %s: food / poison per 1k (food : poison)\n" % ("each run's last tested checkpoint"
                                                                       if step is None else "%dk" % (step // 1000)))
    print("| run | " + " | ".join(c for _, c in TESTS) + " |")
    print("|---" * (len(TESTS) + 1) + "|")
    by_arm = {}
    for r in runs:
        st = step or last_step(r)
        vals = [forage(r, st, t) if st else None for t, _ in TESTS]
        print("| %s s%s | %s |" % (r["arm"], r["seed"], " | ".join(cell(v) for v in vals)))
        by_arm.setdefault(r["arm"], []).append(vals)
    print("\nMeans per arm (food / poison per 1k):\n")
    print("| arm (seeds) | " + " | ".join(c for _, c in TESTS) + " |")
    print("|---" * (len(TESTS) + 1) + "|")
    for arm, rows in by_arm.items():
        cells = []
        for k in range(len(TESTS)):
            v = [r[k] for r in rows if r[k] is not None]
            cells.append("%.1f / %.1f" % tuple(np.mean(v, 0)) if v else "–")
        print("| %s (%d) | %s |" % (arm, len(rows), " | ".join(cells)))


def life_table(runs, segments):
    print("\n### In life: food / poison per 1k, by %d segments of the life\n" % segments)
    print("| run | " + " | ".join("seg %d" % (i + 1) for i in range(segments)) + " | ticks |")
    print("|---" * (segments + 2) + "|")
    for r in runs:
        f = os.path.join(r["path"], "metrics.jsonl")
        m = [json.loads(line) for line in open(f)] if os.path.exists(f) else []
        m = [x for x in m if "step_hits" in x]
        cells = []
        for i in range(segments):
            lo, hi = i * r["steps"] / segments, (i + 1) * r["steps"] / segments
            seg = [x for x in m if lo < x["step"] <= hi]
            cells.append("%.1f / %.1f" % (np.mean([x["step_hits"] for x in seg]), np.mean([x.get("poison", np.nan)
                                                                                         for x in seg]))
                         if seg else "–")
        free = sum(1 for x in m if x["step_hits"] == 0)
        print("| %s s%s | %s | %dk%s |" % (r["arm"], r["seed"], " | ".join(cells), (m[-1]["step"] // 1000) if m else 0,
                                           ", %d food-free 1k windows" % free if free else ""))


def dream_table(runs, step, hs):
    print("\n### The dream (eval_dream.py, the real moves): pixels exactly right after h dreamed steps; at the last h "
          "the hunger error and the cursor's distance (px) and lost share\n")
    print("| run | " + " | ".join("exact h%d" % h for h in hs) + " | hunger err | cursor dist | cursor lost |")
    print("|---" * (len(hs) + 4) + "|")
    for r in runs:
        st = step or last_step(r)
        f = os.path.join(r["path"], "eval", "ckpt_%08d_dream.json" % st) if st else ""
        if not os.path.exists(f):
            continue
        j = json.load(open(f))
        H = min(max(hs), len(j["exact"]))
        print("| %s s%s | %s | %.3f | %.1f | %.2f |" % (
            r["arm"], r["seed"], " | ".join("%.3f" % j["exact"][h - 1] if h <= len(j["exact"]) else "–" for h in hs),
            j["hunger_err"][H - 1], j["cursor_dist"][H - 1], j["cursor_lost"][H - 1]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", help="run directories (each with info.json)")
    ap.add_argument("--step", type=int, default=None, help="the checkpoint of the tests (default: each run's newest)")
    ap.add_argument("--segments", type=int, default=4, help="life split into this many equal segments")
    ap.add_argument("--dream-h", type=int, nargs="+", default=[1, 6, 11, 16])
    args = ap.parse_args()
    runs = load_runs(args.runs)
    if not runs:
        raise SystemExit("no runs (directories with info.json) given")
    frozen_tables(runs, args.step)
    life_table(runs, args.segments)
    dream_table(runs, args.step, args.dream_h)


if __name__ == "__main__":
    main()

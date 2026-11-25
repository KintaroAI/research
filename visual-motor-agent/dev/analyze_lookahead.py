"""The instinct's dream lookahead, frozen (task 00127, vm-00114).

Subcommands:
    table DIR                 — the frozen check: food and poison per 1k (food world) and hit share (Breakout) of each
                                mode (DIR/{TT,BD}_{full,h2,h4}/s1..12; BD's ring is exp_00103/frozen60k), paired by seed
    record OUT.npz ARGS...    — a lifetime.py life (ARGS as lifetime.py's) that saves every dream_return call's input:
                                the candidates' hunger paths [calls, 16, H] (index 0 the real hunger) and H-th labels
    read DIR                  — reads DIR/{TT,BD}_s{1,2,3}.npz: the dreamed meal chance with and without a real meal in
                                the next 4 ticks, and the spread of the score across the 16 candidates
Each tick makes one call per quarter piece (4 with `pieces = 1`).
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import online  # noqa: E402

D = os.path.expanduser("~/data/research/visual-motor-agent")
PIECES = 4


def totals(path):
    rows = [json.loads(line) for line in open(os.path.join(path, "metrics.jsonl"))]
    n, last = rows[-1]["step"], rows[-1]
    hits, miss = last.get("bo_hits", 0), last.get("bo_misses", 0)      # Breakout's counters are cumulative
    return dict(food=1000 * sum(r["step_hits"] for r in rows) / n, poison=1000 * sum(r.get("poison", 0) for r in rows) / n,
                share=hits / max(hits + miss, 1),
                override=np.mean([r["instinct_override"] for r in rows]), gain=np.mean([r["instinct_gain"] for r in rows]))


def table(out):
    def path(w, m, s):
        return f"{D}/exp_00103/frozen60k/BD_s{s}" if (w, m) == ("BD", "ring") else f"{out}/{w}_{m}/s{s}"
    for w, keys in (("TT", ("food", "poison", "override", "gain")), ("BD", ("share", "override", "gain"))):
        R = {m: [totals(path(w, m, s)) for s in range(1, 13)] for m in ("ring", "full", "h2", "h4")}
        for k in keys:
            a = {m: np.array([r[k] for r in R[m]]) for m in R}
            line = f"{w} {k:8s} " + "  ".join(f"{m} {a[m].mean():7.3f}" for m in a)
            for x in ("h2", "h4"):
                d = a[x] - a["full"]
                line += f"   {x}-full {d.mean():+.3f} +- {d.std(ddof=1) / np.sqrt(len(d)):.3f} ({(d > 0).sum()} up, {(d < 0).sum()} down)"
            print(line)


def record(out, args):
    H, L = [], []
    orig = online.dream_return

    def probe(hun, lab, gamma):
        H.append(np.asarray(hun, np.float32).copy())
        L.append(np.asarray(lab, np.float32).copy())
        return orig(hun, lab, gamma)

    online.dream_return = probe
    import lifetime
    sys.argv = ["lifetime.py"] + args
    try:
        lifetime.main()
    finally:
        np.savez(out, hun=np.stack(H), lab=np.stack(L))


def read(src, gamma=0.95):
    for w in ("TT", "BD"):
        A = {k: [] for k in ("qmax", "pmax", "spread_ret", "spread_lab", "meal", "h0")}
        for s in (1, 2, 3):
            z = np.load(f"{src}/{w}_s{s}.npz")
            hun, lab = z["hun"].astype(np.float64), z["lab"]
            real = hun[::PIECES, 0, 0]                                  # the real hunger, one row per tick
            for t in range(len(hun) - 5 * PIECES):
                h = hun[t]
                rise = np.diff(h, axis=1)
                q = np.where(rise < 0, np.minimum(1, -rise / np.maximum(h[:, :-1], 1e-6)), 0)
                p = np.where(rise > online.POISON_RISE, rise, 0)
                A["qmax"].append(q.max()); A["pmax"].append(p.max())
                A["spread_ret"].append(np.ptp(1 - 2 * online.dream_return(h, lab[t], gamma)))
                A["spread_lab"].append(np.ptp(1 - 2 * lab[t]))
                k = t // PIECES
                A["meal"].append(np.any(np.diff(real[k:k + 5]) < -0.005)); A["h0"].append(real[k])
        A = {k: np.array(v) for k, v in A.items()}
        print(f"{w}: {len(A['meal'])} calls; a real meal in the next 4 ticks at {A['meal'].mean():.3f}; "
              f"real hunger median {np.median(A['h0']):.3f}")
        for name, m in (("no real meal", ~A["meal"]), ("real meal", A["meal"])):
            print(f"  {name:13s} best candidate's dreamed meal chance {A['qmax'][m].mean():.3f} "
                  f"(above .1 at {(A['qmax'][m] > .1).mean():.2f}); largest dreamed poison cost {A['pmax'][m].mean():.4f}")
        print(f"  spread across the 16 candidates: the return {A['spread_ret'].mean():.3f}, "
              f"the 4th move's label alone {A['spread_lab'].mean():.3f}")


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "table":
        table(sys.argv[2])
    elif cmd == "record":
        record(sys.argv[2], sys.argv[3:])
    elif cmd == "read":
        read(sys.argv[2])
    else:
        raise SystemExit(__doc__)

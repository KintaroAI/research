"""Tables of stacked_local.py runs (task 00045): each run's best and final val nats per letter, the bottom's own val
(read as logits), means per arm, and the dtp / dni diagnostics over training.

Usage (from visual-motor-agent/dev):
    ./venv/bin/python reference/stacked_summary.py ~/data/research/visual-motor-agent/exp_00038/0??_*
"""
import argparse
import json
import os

import numpy as np


def load(d):
    info = json.load(open(os.path.join(d, "info.json")))
    m = [json.loads(line) for line in open(os.path.join(d, "metrics.jsonl"))]
    arm = info["arm"] + ("+tie" if info.get("tie") else "") + ("+norm" if info.get("cut_norm") else "")
    arm += "" if info.get("beta", 1.0) == 1.0 or info["arm"] != "dtp" else " beta %g" % info["beta"]
    return dict(name=os.path.basename(d.rstrip("/")), arm=arm,
                seed=info["seed"], m=m)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--every", type=int, default=2, help="diagnostics: every n-th evaluation")
    args = ap.parse_args()
    runs = [load(d) for d in args.runs if os.path.exists(os.path.join(d, "metrics.jsonl"))]
    print("| run | best val (update) | final val | bottom's own val |")
    print("|---|---|---|---|")
    arms = {}
    for r in runs:
        b = min(r["m"], key=lambda x: x["val"])
        f = r["m"][-1]
        print("| %s s%d | %.4f (%dk) | %.4f (%dk) | %.3f |" % (r["arm"], r["seed"], b["val"], b["update"] // 1000,
                                                             f["val"], f["update"] // 1000, f["val_bottom"]))
        arms.setdefault(r["arm"], []).append((b["val"], f["val"]))
    print("\n| arm (seeds) | best val, mean | final val, mean |")
    print("|---|---|---|")
    for a, v in arms.items():
        v = np.array(v)
        print("| %s (%d) | %.3f | %.3f |" % (a, len(v), v[:, 0].mean(), v[:, 1].mean()))
    print("\nDiagnostics (update: val / cosine of the bottom's step with the true descent / inverse error or "
          "predictor mse):")
    for r in runs:
        k = "cos_tp" if r["arm"].startswith("dtp") else "cos_sg" if r["arm"].startswith("dni") else None
        if k:
            print(r["name"], " ".join("%dk: %.3f / %s / %.2g" % (x["update"] // 1000, x["val"],
                                                                "%.3f" % x[k] if k in x else "-",
                                                                x.get("inv_rec", x.get("sg_mse", 0)))
                                      for x in r["m"][::args.every]))


if __name__ == "__main__":
    main()

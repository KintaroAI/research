"""Learning curves for one or more runs (metrics.jsonl / eval.jsonl written by train.py).

Usage:
    ./venv/bin/python plot_runs.py RUN_DIR [RUN_DIR ...] -o curves.png [--labels A C D]
Runs sharing a label (e.g. several seeds) are drawn in the same color.
"""
import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def read_jsonl(path):
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def series(rows, key):
    xs = [r["steps"] for r in rows if r.get(key) is not None and np.isfinite(r.get(key, np.nan))]
    ys = [r[key] for r in rows if r.get(key) is not None and np.isfinite(r.get(key, np.nan))]
    return np.array(xs), np.array(ys)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("-o", "--out", required=True)
    ap.add_argument("--labels", nargs="*", default=None)
    ap.add_argument("--title", default="")
    args = ap.parse_args()
    labels = args.labels or [os.path.basename(r.rstrip("/")) for r in args.runs]
    uniq = list(dict.fromkeys(labels))
    colors = {l: plt.cm.tab10(i % 10) for i, l in enumerate(uniq)}
    panels = [("eval", "success", "eval success (stochastic policy)"),
              ("eval", "det_success", "eval success (deterministic policy)"),
              ("eval", "steps_to_success_median", "eval median steps to success"),
              ("eval", "clicks_per_success", "eval clicks per success"),
              ("train", "entropy", "policy entropy"),
              ("train", "pred_changed_mse", "next-frame MSE on changed pixels (dashed: copy last frame)")]
    fig, axes = plt.subplots(2, 3, figsize=(16, 8.5))
    seen = set()
    for run, lab in zip(args.runs, labels):
        ev = read_jsonl(os.path.join(run, "eval.jsonl"))
        tr = read_jsonl(os.path.join(run, "metrics.jsonl"))
        for ax, (src, key, title) in zip(axes.flat, panels):
            rows = ev if src == "eval" else tr
            x, y = series(rows, key)
            if key == "pred_changed_mse":
                ok = y > 0
                x, y = x[ok], y[ok]
                xc, yc = series(tr, "copy_changed_mse")
                okc = yc > 0
                if okc.any():
                    ax.plot(xc[okc], yc[okc], "--", color=colors[lab], alpha=0.6, lw=1)
            if len(x):
                ax.plot(x, y, color=colors[lab], lw=1.6, alpha=0.9, label=lab if lab not in seen else None)
            ax.set_title(title, fontsize=10)
            ax.set_xlabel("environment steps")
            ax.grid(alpha=0.3)
        seen.add(lab)
    axes.flat[0].legend(fontsize=9)
    if args.title:
        fig.suptitle(args.title)
    fig.tight_layout()
    fig.savefig(args.out, dpi=110)
    print("wrote", args.out)


if __name__ == "__main__":
    main()

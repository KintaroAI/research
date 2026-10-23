"""Compare reward-free dynamics metrics across runs (reads RUN/analysis/dynamics_over_training.json).

Usage:
    ./venv/bin/python compare_dynamics.py RUN1 RUN2 ... --labels L1 L2 ... -o fig.png [--untrained DIR]
"""
import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

PANELS = [
    ("outcome_explained_acc", "Correct action best explains the outcome\n(chance 1/9)", (0.08, None)),
    ("disp_corr", "corr(predicted, true cursor displacement)\n(mean of x, y)", (-0.2, 1.0)),
    ("vanish_sel", "Click contingency: vanish(inside) − vanish(outside)\n(truth 1.0)", (-0.1, 1.0)),
    ("hunger_drop_inside", "Predicted hunger relief from an inside click\n(truth ≈ 0.25)", (-0.02, 0.45)),
    ("probe_vf_r2_cursor", "Cursor position decodable from visual feature\n(linear probe R², mean x/y)", (-0.05, 1.02)),
]


def derived(r, key):
    if key == "disp_corr":
        return np.nanmean([r.get("disp_corr_x", np.nan), r.get("disp_corr_y", np.nan)])
    if key == "vanish_sel":
        return r["vanish_inside"] - r["vanish_outside"]
    if key == "probe_vf_r2_cursor":
        return np.mean([r.get("probe_vf_r2_cursor_x", np.nan), r.get("probe_vf_r2_cursor_y", np.nan)])
    return r.get(key, np.nan)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--labels", nargs="*")
    ap.add_argument("-o", "--out", required=True)
    args = ap.parse_args()
    labels = args.labels or [os.path.basename(r.rstrip("/")) for r in args.runs]
    fig, axes = plt.subplots(1, len(PANELS), figsize=(4.2 * len(PANELS), 4.2))
    for run, lab in zip(args.runs, labels):
        path = os.path.join(run, "analysis", "dynamics_over_training.json")
        if not os.path.exists(path):
            print("missing", path)
            continue
        rows = json.load(open(path))
        x = np.array([r["steps"] for r in rows], float)
        x[x == 0] = 3e3                                   # show the initialization on the log axis
        for ax, (key, title, ylim) in zip(axes, PANELS):
            ax.plot(x, [derived(r, key) for r in rows], "o-", label=lab, lw=1.6)
    for ax, (key, title, ylim) in zip(axes, PANELS):
        ax.set_xscale("log")
        ax.set_title(title, fontsize=9.5)
        ax.set_xlabel("environment steps (init shown at 3e3)")
        ax.grid(alpha=0.3)
        if ylim[1] is not None:
            ax.set_ylim(*ylim)
        if key == "outcome_explained_acc":
            ax.axhline(1 / 9, ls="--", c="gray", lw=1)
    axes[0].legend(fontsize=8)
    fig.suptitle("Reward-free babbling: what the model learns about its own actions", fontsize=11)
    fig.tight_layout()
    fig.savefig(args.out, dpi=110)
    print("wrote", args.out)


if __name__ == "__main__":
    main()

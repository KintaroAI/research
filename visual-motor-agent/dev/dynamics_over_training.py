"""Run the counterfactual dynamics tests on every milestone checkpoint of a run and plot them.

Usage:
    ./venv/bin/python dynamics_over_training.py RUN_DIR [--states 192] [-o fig.png]
Writes RUN_DIR/analysis/dynamics_over_training.{json,png} by default (re-uses existing per-checkpoint
results in RUN_DIR/analysis/ckpt_*/dynamics.json).
"""
import argparse
import glob
import json
import os
import re
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import agent_lib  # noqa: E402
from analyze_dynamics import analyze  # noqa: E402
from record_episodes import env_config_from_info  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--states", type=int, default=192)
    ap.add_argument("-o", "--out", default=None)
    ap.add_argument("--redo", action="store_true")
    args = ap.parse_args()
    with open(os.path.join(args.run_dir, "info.json")) as f:
        info = json.load(f)
    env_cfg = env_config_from_info(info)
    ckpts = sorted(glob.glob(os.path.join(args.run_dir, "ckpt_[0-9]*.bin")))
    rows = []
    for ck in ckpts:
        m = re.search(r"ckpt_(\d+)\.bin$", ck)
        if not m:
            continue
        steps = int(m.group(1))
        tag = os.path.splitext(os.path.basename(ck))[0]
        res_path = os.path.join(args.run_dir, "analysis", tag, "dynamics.json")
        if os.path.exists(res_path) and not args.redo:
            with open(res_path) as f:
                res = json.load(f)
        else:
            agent = agent_lib.Agent.load(ck)
            res = analyze(agent, env_cfg, args.states, out_dir=os.path.dirname(res_path))
            agent.close()
        res["steps"] = steps
        rows.append(res)
        print("%9d  retrieval %.3f  explained %.3f  margin %.3f  disp_corr %.2f/%.2f  vanish in/out %.2f/%.2f  "
              "hunger_drop %.3f  probe z cursor %.2f  vf cursor %.2f target %.2f" % (
            steps, res["action_retrieval_acc"], res.get("outcome_explained_acc", float("nan")),
            res.get("outcome_margin_rel", float("nan")), res.get("disp_corr_x", float("nan")),
            res.get("disp_corr_y", float("nan")), res["vanish_inside"], res["vanish_outside"], res["hunger_drop_inside"],
            res["probe_r2_cursor_x"], res.get("probe_vf_r2_cursor_x", float("nan")),
            res.get("probe_vf_r2_target_x", float("nan"))), flush=True)
    out_json = os.path.join(args.run_dir, "analysis", "dynamics_over_training.json")
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(rows, f, indent=1)
    x = np.array([r["steps"] for r in rows]) + 1
    fig, axes = plt.subplots(1, 4, figsize=(19, 4.2))
    ax = axes[0]
    ax.plot(x, [r["action_retrieval_acc"] for r in rows], "o-", label="action retrieval acc (strict)")
    ax.plot(x, [r.get("outcome_explained_acc", np.nan) for r in rows], "o-", label="outcome explained acc")
    ax.axhline(1 / 9, ls="--", c="gray", label="chance (1/9)")
    ax.set_title("Does the prediction depend on the action?")
    ax.legend(fontsize=8)
    ax = axes[1]
    ax.plot(x, [r.get("disp_corr_x", np.nan) for r in rows], "o-", label="corr(pred, true) dx")
    ax.plot(x, [r.get("disp_corr_y", np.nan) for r in rows], "o-", label="corr(pred, true) dy")
    ax.set_title("Predicted cursor displacement vs truth")
    ax.legend(fontsize=8)
    ax = axes[2]
    ax.plot(x, [r["vanish_inside"] for r in rows], "o-", label="click, cursor INSIDE (truth 1)")
    ax.plot(x, [r["vanish_outside"] for r in rows], "o-", label="click, cursor outside (truth 0)")
    ax.plot(x, [r["hunger_drop_inside"] for r in rows], "o-", label="predicted hunger drop, inside click")
    ax.set_title("Click consequences (circle vanishes / hunger relief)")
    ax.legend(fontsize=8)
    ax = axes[3]
    for k, lab in (("probe_r2_cursor_x", "z: cursor x"), ("probe_r2_target_x", "z: target x"),
                   ("probe_vf_r2_cursor_x", "visual feature: cursor x"), ("probe_vf_r2_target_x", "visual feature: target x")):
        ax.plot(x, [r.get(k, np.nan) for r in rows], "o-", label=lab)
    ax.set_ylim(-0.1, 1.05)
    ax.set_title("Linear probe R² (position decodable?)")
    ax.legend(fontsize=8)
    for ax in axes:
        ax.set_xscale("log")
        ax.set_xlabel("environment steps (+1)")
        ax.grid(alpha=0.3)
    fig.suptitle(os.path.basename(args.run_dir.rstrip("/")))
    fig.tight_layout()
    out_png = args.out or os.path.join(args.run_dir, "analysis", "dynamics_over_training.png")
    fig.savefig(out_png, dpi=110)
    print("wrote", out_png)


if __name__ == "__main__":
    main()

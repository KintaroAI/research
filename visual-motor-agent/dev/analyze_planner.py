"""Where and why does the planner fail? The CEM planner of plan.py on the test episodes, traced.

Per episode (evaluation only; the true positions are used for scoring, never by the planner):
  start gap        distance from the cursor to the circle's edge at the start (px)
  min / final gap  closest approach to the edge and the gap at the end
  steps inside     steps with the cursor inside the circle
  clicks outside   clicks with the cursor outside the circle
  imagined relief  the largest hunger drop the planner's best plan imagined during the episode, relative to the
                   true hunger (about 1 = it expected the hunger to vanish)
Writes RUN_DIR/plan/<ckpt>_diag.json and prints success by start gap and the failed episodes.

Usage:
    ./venv/bin/python analyze_planner.py RUN_DIR [--ckpt ckpt_final.bin] [--episodes 128]
"""
import argparse
import dataclasses
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import agent_lib  # noqa: E402
from plan import EVAL_SEED, CEMPlanner  # noqa: E402
from record_episodes import env_config_from_info  # noqa: E402
from world import CircleWorld  # noqa: E402

GAP_BINS = (0, 8, 16, 24, 32, 48, 1000)


def trace(agent, cfg, n, seed=EVAL_SEED, **planner_kw):
    env = CircleWorld(n, cfg, seed=seed)
    st = env.get_state()
    tgt, rad = st["target"].copy(), st["radius"].copy()
    planner = CEMPlanner(agent, n, **planner_kw)
    obs = env.observe()
    active, success = np.ones(n, bool), np.zeros(n, bool)
    length = np.zeros(n, int)
    gaps, inside = [], np.zeros(n, int)
    clicks_out, relief = np.zeros(n, int), np.zeros(n)
    for _ in range(cfg.max_steps):
        a, best_h = planner.act(obs["image"], obs["drives"][:, 0], active)
        h = obs["drives"][:, 0]
        expected = h[:, None] + cfg.h_rate * np.arange(1, planner.H + 1)[None]       # hunger without relief
        drop = np.nan_to_num(((expected - best_h) / expected).max(1))
        relief = np.where(active, np.maximum(relief, drop), relief)
        g = np.hypot(*(env.get_state()["cursor"] - tgt).T) - rad
        gaps.append(np.where(active, g, np.nan))
        inside += active & (g <= 0)
        clicks_out += active & (g > 0) & (a[:, 2] > 0.5)
        obs, _, done, inf = env.step(np.where(active[:, None], a, 0))
        length += active
        success |= active & inf["success"]
        active &= ~done
        if not active.any():
            break
    gaps = np.array(gaps)
    with np.errstate(all="ignore"):
        return dict(success=success, length=length, start_gap=gaps[0], min_gap=np.nanmin(gaps, 0),
                    final_gap=np.array([gaps[np.isfinite(gaps[:, i]), i][-1] for i in range(n)]),
                    steps_inside=inside, clicks_outside=clicks_out, imagined_relief=relief)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir")
    ap.add_argument("--ckpt", default="ckpt_final.bin")
    ap.add_argument("--episodes", type=int, default=128)
    ap.add_argument("--threads", type=int, default=32)
    args = ap.parse_args()
    agent_lib.set_threads(args.threads)
    with open(os.path.join(args.run_dir, "info.json")) as f:
        info = json.load(f)
    cfg = dataclasses.replace(env_config_from_info(info), bonuses=False, auto_reset=False, drives=True)
    agent = agent_lib.Agent.load(os.path.join(args.run_dir, args.ckpt))
    t = trace(agent, cfg, args.episodes)
    s, gap = t["success"], t["start_gap"]
    bins = []
    for lo, hi in zip(GAP_BINS[:-1], GAP_BINS[1:]):
        m = (gap >= lo) & (gap < hi)
        bins.append(dict(gap=[lo, hi], episodes=int(m.sum()), hits=int(s[m].sum())))
    fail = np.flatnonzero(~s)
    res = dict(run=args.run_dir, ckpt=args.ckpt, success=float(s.mean()), by_start_gap=bins,
               failures=[dict(episode=int(i), **{k: float(t[k][i]) for k in ("start_gap", "min_gap", "final_gap",
                                                                            "steps_inside", "clicks_outside",
                                                                            "imagined_relief")}) for i in fail],
               imagined_relief_median_hit=float(np.median(t["imagined_relief"][s])) if s.any() else float("nan"),
               imagined_relief_median_miss=float(np.median(t["imagined_relief"][~s])) if fail.size else float("nan"),
               clicks_outside_median_hit=float(np.median(t["clicks_outside"][s])) if s.any() else float("nan"))
    out = os.path.join(args.run_dir, "plan")
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, os.path.splitext(args.ckpt)[0] + "_diag.json"), "w") as f:
        json.dump(res, f, indent=1)
    print("success %.3f" % res["success"])
    print("by start gap: " + "  ".join("%d-%d px %d/%d" % (b["gap"][0], b["gap"][1], b["hits"], b["episodes"])
                                       for b in bins))
    print("failures: episode, start gap, min gap, final gap, steps inside, clicks outside, imagined relief")
    for r in res["failures"]:
        print("  %4d %6.1f %6.1f %6.1f %4d %4d %6.2f" % (r["episode"], r["start_gap"], r["min_gap"], r["final_gap"],
                                                         r["steps_inside"], r["clicks_outside"],
                                                         r["imagined_relief"]))
    print("imagined relief, median: hits %.2f, misses %.2f; clicks outside, median in hits: %.0f" % (
        res["imagined_relief_median_hit"], res["imagined_relief_median_miss"], res["clicks_outside_median_hit"]))


if __name__ == "__main__":
    main()

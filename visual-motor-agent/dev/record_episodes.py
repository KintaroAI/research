"""Record a trained agent's episodes: GIFs, browser-replay JSON, and the model's own predictions.

For each recorded env, writes into OUT_DIR:
  episode_<i>.json       replay file for world/browser_demo/index.html (re-rendered by the JS world)
  episode_<i>.gif        what the agent saw
  predict_<i>.gif        (models with a prediction head) per step: [current frame | predicted next
                         frame for the action the agent chose | actual next frame], 32x32 frames
                         upscaled; predictions use a single-frame context.

Usage:
    ./venv/bin/python record_episodes.py RUN_DIR [--ckpt ckpt_final.bin] [-n 4] [--seed 5] [--out DIR]
"""
import argparse
import dataclasses
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import agent_lib  # noqa: E402
from analyze_dynamics import downsample  # noqa: E402
from evaluate import executed_action  # noqa: E402
from world import CircleWorld, EnvConfig  # noqa: E402
from world.record import EpisodeRecorder, save_gif  # noqa: E402


def env_config_from_info(info):
    ec = dict(info["env_config"])
    for k in ("bg_color", "target_color", "cursor_color", "cursor_outline"):
        ec[k] = tuple(ec[k])
    return EnvConfig(**ec)


def side_by_side(frames, scale=4, pad=2):
    """frames: list of HxWx3 (float or u8, possibly different sizes) -> one u8 image row."""
    h = max(f.shape[0] for f in frames)
    out = []
    for f in frames:
        f = np.asarray(f)
        if f.dtype != np.uint8:
            f = (np.clip(f, 0, 1) * 255 + 0.5).astype(np.uint8)
        s = h // f.shape[0]
        out.append(f.repeat(s * scale, 0).repeat(s * scale, 1))
        out.append(np.full((h * scale, pad, 3), 128, np.uint8))
    return np.concatenate(out[:-1], axis=1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir")
    ap.add_argument("--ckpt", default="ckpt_final.bin")
    ap.add_argument("-n", type=int, default=4)
    ap.add_argument("--seed", type=int, default=5)
    ap.add_argument("--deterministic", action="store_true")
    ap.add_argument("--radius", type=float, nargs=2, default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    with open(os.path.join(args.run_dir, "info.json")) as f:
        info = json.load(f)
    agent = agent_lib.Agent.load(os.path.join(args.run_dir, args.ckpt))
    cfg = dataclasses.replace(env_config_from_info(info), bonuses=False, auto_reset=False)
    if args.radius:
        cfg = dataclasses.replace(cfg, radius_min=args.radius[0], radius_max=args.radius[1])
    out_dir = args.out or os.path.join(args.run_dir, "episodes", os.path.splitext(args.ckpt)[0])
    os.makedirs(out_dir, exist_ok=True)
    n = args.n
    env = CircleWorld(n, cfg, seed=args.seed)
    obs = env.observe()
    recs = [EpisodeRecorder(cfg, i, meta=dict(run=args.run_dir, ckpt=args.ckpt)) for i in range(n)]
    for r in recs:
        r.begin(env)
    ctx = agent.new_context(n, seed=args.seed)
    ep_start = np.ones(n, np.uint8)
    prev_a = np.zeros((n, 3), np.float32)
    prev_r = np.zeros(n, np.float32)
    seen = [[obs["image"][i]] for i in range(n)]
    pred_rows = [[] for _ in range(n)]
    done_all = np.zeros(n, bool)
    f = cfg.size // agent.config["pred_img"]
    for _ in range(cfg.max_steps):
        a = agent.act(ctx, obs["image"], obs["drives"], prev_a, prev_r, ep_start=ep_start,
                      deterministic=args.deterministic)["action"]
        ea = executed_action(a)
        pred = agent.forward(obs["image"][:, None], obs["drives"][:, None], np.zeros((n, 1, 3), np.float32),
                             np.zeros((n, 1), np.float32), np.ones((n, 1), np.uint8), pred_actions=ea[:, None])
        a_env = np.where(done_all[:, None], 0.0, a)      # finished envs idle (auto_reset off)
        nobs, rew, done, inf = env.step(a_env)
        for i in range(n):
            if done_all[i]:
                continue
            recs[i].record(a_env, rew, done, inf)
            seen[i].append(inf["terminal_image"][i])
            pred_rows[i].append(side_by_side([downsample(obs["image"][i], f), pred["pred_frames"][i, 0],
                                              downsample(inf["terminal_image"][i], f)]))
        done_all |= done
        prev_a = ea
        prev_r = (rew * 0.1).astype(np.float32)
        ep_start = np.zeros(n, np.uint8)
        obs = nobs
        if done_all.all():
            break
    summary = []
    for i in range(n):
        recs[i].save_json(os.path.join(out_dir, "episode_%d.json" % i))
        save_gif(seen[i], os.path.join(out_dir, "episode_%d.gif" % i), scale=4, fps=10)
        if pred_rows[i]:
            save_gif(pred_rows[i], os.path.join(out_dir, "predict_%d.gif" % i), scale=1, fps=6)
        summary.append(recs[i].summary())
    with open(os.path.join(out_dir, "summary.json"), "w") as fp:
        json.dump(summary, fp, indent=2, default=float)
    print(json.dumps(summary, default=float))
    print("wrote", out_dir)


if __name__ == "__main__":
    main()

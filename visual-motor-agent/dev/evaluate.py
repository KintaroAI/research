"""Evaluation of a trained (or untrained) agent: task metrics and learned-dynamics analysis.

Library use (train.py): run_episodes(agent, env_cfg, n, seed) -> task metrics on fixed seeds.

CLI:
    ./venv/bin/python evaluate.py RUN_DIR [--ckpt ckpt_final.bin] [--episodes 512] [--dynamics] [--record 4]

Ground-truth geometry (`info["gt_*"]`, env state) is used here only to *score* the agent, never as
model input.
"""
import argparse
import dataclasses
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from world import CircleWorld, EnvConfig  # noqa: E402


def executed_action(a):
    """The action as the world applies it: clipped (dx, dy) and binary click."""
    return np.stack([np.clip(a[:, 0], -1, 1), np.clip(a[:, 1], -1, 1), (a[:, 2] > 0.5).astype(np.float32)],
                    axis=1).astype(np.float32)


def run_episodes(agent, env_cfg, n, seed, deterministic=False, reward_scale=0.1, policy=None):
    """Run one episode in each of n parallel envs (fixed seed) and return task metrics.

    policy: optional callable(obs, ctx_state) -> actions [n,3] replacing the agent (random baselines).
    """
    env = CircleWorld(n, env_cfg, seed=seed)
    obs = env.reset()
    ctx = agent.new_context(n, seed=seed + 7) if agent is not None else None
    ep_start = np.ones(n, np.uint8)
    prev_a = np.zeros((n, 3), np.float32)
    prev_r = np.zeros(n, np.float32)
    finished = np.zeros(n, bool)
    res = {k: np.zeros(n) for k in ("success", "length", "clicks", "empty_clicks", "ret", "path", "init_dist")}
    for _ in range(env_cfg.max_steps):
        if policy is None:
            a = agent.act(ctx, obs["image"], obs["drives"], prev_a, prev_r, ep_start=ep_start,
                          deterministic=deterministic)["action"]
        else:
            a = policy(obs)
        obs, rew, done, inf = env.step(a)
        new = done & ~finished
        if new.any():
            res["success"][new] = inf["ep_success"][new]
            res["length"][new] = inf["ep_length"][new]
            res["clicks"][new] = inf["ep_clicks"][new]
            res["empty_clicks"][new] = inf["ep_empty_clicks"][new]
            res["ret"][new] = inf["ep_return"][new]
            res["path"][new] = inf["ep_path_length"][new]
            res["init_dist"][new] = inf["ep_init_dist"][new]
        finished |= done
        if finished.all():
            break
        first = inf["first"].astype(bool)
        prev_a = executed_action(a)
        prev_a[first] = 0.0
        prev_r = (rew * reward_scale).astype(np.float32)
        prev_r[first] = 0.0
        ep_start = first.astype(np.uint8)
    if ctx is not None:
        ctx.close()
    return summarize(res)


def summarize(res):
    succ = res["success"].astype(bool)
    out = dict(episodes=int(len(succ)), success=float(succ.mean()), len_mean=float(res["length"].mean()),
               return_mean=float(res["ret"].mean()), clicks_per_ep=float(res["clicks"].mean()),
               empty_clicks_per_ep=float(res["empty_clicks"].mean()))
    if succ.any():
        L = res["length"][succ]
        out["steps_to_success_median"] = float(np.median(L))
        out["steps_to_success_mean"] = float(L.mean())
        out["clicks_per_success"] = float(res["clicks"].sum() / succ.sum())
        ok = succ & (res["path"] > 0)
        if ok.any():
            # straight-line distance to the circle edge / path actually travelled (1 = perfectly direct)
            out["path_efficiency_median"] = float(np.median(np.minimum(1.0, res["init_dist"][ok] / res["path"][ok])))
    return out


def load_run(run_dir, ckpt="ckpt_final.bin"):
    import agent_lib
    with open(os.path.join(run_dir, "info.json")) as f:
        info = json.load(f)
    agent = agent_lib.Agent.load(os.path.join(run_dir, ckpt))
    return agent, info


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir")
    ap.add_argument("--ckpt", default="ckpt_final.bin")
    ap.add_argument("--episodes", type=int, default=512)
    ap.add_argument("--seed", type=int, default=999_000)
    ap.add_argument("--radius", type=float, nargs=2, default=None, help="evaluate on radius range (min max)")
    args = ap.parse_args()
    agent, info = load_run(args.run_dir, args.ckpt)
    env_cfg = EnvConfig(**info["env_config"])
    env_cfg = dataclasses.replace(env_cfg, bonuses=False)
    if args.radius:
        env_cfg = dataclasses.replace(env_cfg, radius_min=args.radius[0], radius_max=args.radius[1])
    for det in (False, True):
        m = run_episodes(agent, env_cfg, args.episodes, args.seed, deterministic=det)
        print(("deterministic" if det else "stochastic"), json.dumps(m))


if __name__ == "__main__":
    main()

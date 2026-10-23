"""Random-policy baselines for CircleWorld (plan section 10, milestone 2).

Evaluates memoryless random policies on a fixed number of episodes and
reports success rate, steps to success (successful episodes only), median
episode length including timeouts, clicks per success and mean return.

Protocol: N parallel envs share the episode quota, and env i runs exactly
quota_i complete episodes (the quotas sum to --episodes).  Fixing the quota
per env avoids the bias toward short episodes that you get by keeping the
first K finishers.  Every policy uses the same env seed, so env i's k-th
episode starts from the same configuration for every policy (common random
numbers).

Usage (from dev/):
    ./venv/bin/python random_baseline.py                                  # radius [12, 16]
    ./venv/bin/python random_baseline.py --radius-min 4 --radius-max 8
    ./venv/bin/python random_baseline.py --json /path/to/results.json
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import time
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

from world import CircleWorld, EnvConfig

Policy = Callable[[np.random.Generator, int], np.ndarray]


def _actions(move: np.ndarray, click: np.ndarray) -> np.ndarray:
    return np.concatenate([move, click[:, None].astype(np.float64)], axis=1)


#: (key, description, policy).  A policy maps (rng, n) to (n, 3) actions.
POLICIES: List[tuple] = [
    ("a", "dx,dy ~ U[-1,1], click p=0.5",
     lambda rng, n: _actions(rng.uniform(-1, 1, (n, 2)), rng.random(n) < 0.5)),
    ("b", "dx,dy ~ U[-1,1], click p=0.1",
     lambda rng, n: _actions(rng.uniform(-1, 1, (n, 2)), rng.random(n) < 0.1)),
    ("c", "dx,dy ~ N(0,1) clipped, click p=0.5",
     lambda rng, n: _actions(np.clip(rng.normal(0, 1.0, (n, 2)), -1, 1), rng.random(n) < 0.5)),
    ("d", "dx,dy ~ N(0,0.5²), click p=0.5",
     lambda rng, n: _actions(np.clip(rng.normal(0, 0.5, (n, 2)), -1, 1), rng.random(n) < 0.5)),
    ("e", "no movement, click p=0.5",
     lambda rng, n: _actions(np.zeros((n, 2)), rng.random(n) < 0.5)),
]


def evaluate_policy(policy: Policy, cfg: EnvConfig, episodes: int, num_envs: int,
                    seed: int, policy_seed: Sequence[int]) -> Dict[str, Optional[float]]:
    """Run ``episodes`` complete episodes and return summary statistics."""
    env = CircleWorld(num_envs, cfg, seed=seed)
    env.reset()
    quota = np.full(num_envs, episodes // num_envs)
    quota[: episodes % num_envs] += 1
    finished = np.zeros(num_envs, dtype=np.int64)
    rng = np.random.default_rng(list(policy_seed))
    keys = ("length", "return", "success", "clicks")
    parts: Dict[str, List[np.ndarray]] = {k: [] for k in keys}
    env_steps = 0
    t0 = time.perf_counter()
    while (finished < quota).any():
        obs, reward, done, info = env.step(policy(rng, num_envs))
        env_steps += num_envs
        if done.any():
            take = done & (finished < quota)
            for k in keys:
                parts[k].append(info["ep_" + k][take])
            finished += done
    ep = {k: np.concatenate(v) for k, v in parts.items()}
    assert len(ep["length"]) == episodes

    success = ep["success"] > 0.5
    n_success = int(success.sum())
    p = n_success / episodes
    lengths_ok = ep["length"][success]
    return {
        "episodes": episodes,
        "successes": n_success,
        "success_rate": p,
        "success_rate_se": math.sqrt(p * (1 - p) / episodes),
        "median_steps_to_success": float(np.median(lengths_ok)) if n_success else None,
        "mean_steps_to_success": float(lengths_ok.mean()) if n_success else None,
        "median_episode_length": float(np.median(ep["length"])),
        "clicks_per_success": float(ep["clicks"].sum() / n_success) if n_success else None,
        "mean_return": float(ep["return"].mean()),
        "env_steps": env_steps,
        "seconds": time.perf_counter() - t0,
    }


def _fmt(x: Optional[float], digits: int, none: str = "–") -> str:
    return none if x is None else f"{x:.{digits}f}"


def markdown_table(results: List[Dict], cfg: EnvConfig, args: argparse.Namespace) -> str:
    lines = [
        f"Random-policy baselines, target radius [{cfg.radius_min:g}, {cfg.radius_max:g}]: "
        f"{args.episodes} episodes per policy ({args.num_envs} parallel envs, env seed {args.seed}), "
        f"size {cfg.size}, max_steps {cfg.max_steps}, max_step_px {cfg.max_step_px:g}, "
        f"bonuses {'on' if cfg.bonuses else 'off'}, drives {'on' if cfg.drives else 'off'}.",
        "",
        "| Policy | Success rate | Median steps to success | Mean steps to success "
        "| Median episode length | Clicks per success | Mean return |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in results:
        rate = f"{100 * r['success_rate']:.1f}% ± {100 * r['success_rate_se']:.1f}"
        lines.append(
            f"| ({r['key']}) {r['policy']} | {rate} | {_fmt(r['median_steps_to_success'], 0)} "
            f"| {_fmt(r['mean_steps_to_success'], 1)} | {_fmt(r['median_episode_length'], 0)} "
            f"| {_fmt(r['clicks_per_success'], 1, none='∞')} | {r['mean_return']:.3f} |")
    lines.append("")
    lines.append("Success rate ± one standard error.  Steps to success count successful "
                 "episodes only; episode length includes timeouts at max_steps.")
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Random-policy baselines for CircleWorld.")
    parser.add_argument("--episodes", type=int, default=2000)
    parser.add_argument("--num-envs", type=int, default=128)
    parser.add_argument("--seed", type=int, default=12345, help="env seed (shared by all policies)")
    parser.add_argument("--radius-min", type=float, default=EnvConfig.radius_min)
    parser.add_argument("--radius-max", type=float, default=EnvConfig.radius_max)
    parser.add_argument("--policies", default="abcde", help="subset of policy keys to run")
    parser.add_argument("--json", default=None, help="write results to this JSON file")
    args = parser.parse_args(argv)

    cfg = dataclasses.replace(EnvConfig(), radius_min=args.radius_min,
                              radius_max=args.radius_max).validate()
    results = []
    for index, (key, name, policy) in enumerate(POLICIES):
        if key not in args.policies:
            continue
        stats = evaluate_policy(policy, cfg, args.episodes, args.num_envs, args.seed,
                                policy_seed=(args.seed, index))
        results.append({"key": key, "policy": name, **stats})

    print(markdown_table(results, cfg, args))
    if args.json:
        with open(args.json, "w") as f:
            json.dump({"config": cfg.to_dict(), "episodes": args.episodes,
                       "num_envs": args.num_envs, "seed": args.seed, "results": results},
                      f, indent=2, allow_nan=False)
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()

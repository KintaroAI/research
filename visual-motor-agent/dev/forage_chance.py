"""Chance foraging (task 00042): blind random walks in a continual world (respawn), food and poison events per 1000
steps over the forage test's worlds (online.py's EVAL_SEED), no model. In the touch world the ratio food : poison of
a blind walk is set by how it covers the torus (food stands still, the poison drifts), so an agent's ratio above every
walk's is steering.

Usage:
    ./venv/bin/python forage_chance.py --config configs/U_touch.json [--worlds 16 --steps 1000]
"""
import argparse
import dataclasses
import json

import numpy as np

from online import EVAL_SEED
from world import CircleWorld, EnvConfig


def walk(cfg, n, steps, policy):
    """Food and poison per 1k of the moves policy(rng, t, prev) -> [n, 2]."""
    env = CircleWorld(n, dataclasses.replace(cfg, auto_reset=False, max_steps=10 ** 9), seed=EVAL_SEED)
    rng = np.random.default_rng(0)
    food = poison = 0
    m = None
    for t in range(steps):
        m = policy(rng, t, m)
        _, _, _, inf = env.step(np.concatenate([m, np.zeros((n, 1))], 1))
        food += inf["success"].sum()
        poison += inf.get("gt_hazard_click", np.zeros(n, bool)).sum()
    return food / n * 1000 / steps, poison / n * 1000 / steps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--worlds", type=int, default=16)
    ap.add_argument("--steps", type=int, default=1000)
    args = ap.parse_args()
    cfg = EnvConfig(**json.load(open(args.config))["env"])
    n = args.worlds
    corner = lambda r: r.choice([-1.0, 1.0], (n, 2))   # noqa: E731  a full-speed diagonal
    walks = [("uniform moves in [-1, 1]^2", lambda r, t, m: r.uniform(-1, 1, (n, 2))),
             ("full-speed diagonals, new each step", lambda r, t, m: corner(r))]
    walks += [("full-speed diagonals, held %d steps" % k, lambda r, t, m, k=k: m if m is not None and t % k else corner(r))
              for k in (5, 20, 100)]
    for name, pol in walks:
        f, p = walk(cfg, n, args.steps, pol)
        print("%-38s food/1k %5.1f  poison/1k %5.1f  ratio %.2f" % (name, f, p, f / max(p, 1e-9)))


if __name__ == "__main__":
    main()

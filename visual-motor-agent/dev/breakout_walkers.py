"""Task 00111 (vm-00103): the Breakout court's walkers, for scale. No model; the same world and seeds as the lives.

- random: a random push each 6 ticks, held (as readme_images' world strip);
- still: the paddle never moves;
- tracker: pushes toward the ball's x (a script reading gt state; the best simple play).

Per walker and seed: hits and misses per 1k ticks, the hit share (hits over hits and misses) and mean hunger, over
--steps ticks after --skip. Rows go to --out as JSON, a table to stdout.

    ./venv/bin/python breakout_walkers.py --config configs/U_breakout.json --seeds 1 2 3 --steps 20000 [--skip 200] [--out FILE]
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from world import make_config, make_world  # noqa: E402

WALKERS = ("random", "still", "tracker")


def move(kind, w, t, rng, held):
    c = w.config
    a = np.zeros((w.num_envs, 3))
    if kind == "random":
        if t % 6 == 0:
            held[:] = rng.uniform(-1, 1, w.num_envs)
        a[:, 0] = held
    elif kind == "tracker":
        a[:, 0] = np.clip(((w.ball[:, 0] + c.ball / 2) - (w.paddle + c.paddle_w / 2)) / c.paddle_speed, -1, 1)
    return a


def walk(kind, env, seed, steps, skip):
    w = make_world(1, make_config(env, auto_reset=True), seed=seed)
    rng = np.random.default_rng([seed, 0xB0])
    held = np.zeros(1)
    hits = misses = 0
    hunger = []
    for t in range(skip + steps):
        _, _, _, info = w.step(move(kind, w, t, rng, held))
        if t >= skip:
            hits += int(info["success"][0])
            misses += int(info["gt_hazard_click"][0])
            hunger.append(float(info["terminal_drives"][0, 0]))
    return dict(walker=kind, seed=seed, hits_per_1k=1000.0 * hits / steps, misses_per_1k=1000.0 * misses / steps,
                hit_share=hits / max(hits + misses, 1), hunger=float(np.mean(hunger)))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--config", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", required=True)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--skip", type=int, default=200)
    ap.add_argument("--out")
    args = ap.parse_args()
    env = json.load(open(args.config))["env"]
    rows = [walk(k, env, s, args.steps, args.skip) for k in WALKERS for s in args.seeds]
    print("| Walker | Hits per 1k | Misses per 1k | Hit share | Mean hunger |\n|---|---|---|---|---|")
    for k in WALKERS:
        r = [x for x in rows if x["walker"] == k]
        print("| %s | %.2f | %.2f | %.3f | %.3f |" % (k, *(np.mean([x[f] for x in r]) for f in ("hits_per_1k", "misses_per_1k", "hit_share", "hunger"))))
    if args.out:
        json.dump(rows, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()

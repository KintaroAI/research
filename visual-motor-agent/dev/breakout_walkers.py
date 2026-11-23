"""Task 00111 (vm-00103): the Breakout court's walkers, for scale. No model; the same world and seeds as the lives.

- random: a random push each 6 ticks, held (as readme_images' world strip);
- still: the paddle never moves;
- tracker: pushes toward the ball's x (a script reading gt state; the best simple play);
- aimer (task 00117): the tracker meeting the ball off the paddle's centre by an offset drawn anew (within 4 px) after
  each paddle hit, so the ball leaves at varied angles; it aims at no brick: the chance level of the brick preference.

With bricks (task 00117) also: bricks and food per 1k, the paddle hit share (paddle hits over paddle hits and misses),
clears, and the preference (the food rank of each brick hit minus the mean rank of those standing, over brick hits).

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

WALKERS = ("random", "still", "tracker", "aimer")


def move(kind, w, t, rng, held):
    c = w.config
    a = np.zeros((w.num_envs, 3))
    if kind == "random":
        if t % 6 == 0:
            held[:] = rng.uniform(-1, 1, w.num_envs)
        a[:, 0] = held
    elif kind in ("tracker", "aimer"):
        off = held if kind == "aimer" else 0.0
        a[:, 0] = np.clip(((w.ball[:, 0] + c.ball / 2) - (w.paddle + c.paddle_w / 2 + off)) / c.paddle_speed, -1, 1)
    return a


def walk(kind, env, seed, steps, skip):
    w = make_world(1, make_config(env, auto_reset=True), seed=seed)
    rng = np.random.default_rng([seed, 0xB0])
    held = np.zeros(1)
    if kind == "aimer":
        held[:] = rng.uniform(-4, 4)
    hits = misses = paddle = 0
    hunger = []
    for t in range(skip + steps):
        if t == skip:
            s0 = w.stats()
        _, _, _, info = w.step(move(kind, w, t, rng, held))
        if kind == "aimer" and info["gt_paddle_hit"][0]:
            held[:] = rng.uniform(-4, 4)
        if t >= skip:
            hits += int(info["success"][0])
            misses += int(info["gt_hazard_click"][0])
            paddle += int(info["gt_paddle_hit"][0])
            hunger.append(float(info["terminal_drives"][0, 0]))
    r = dict(walker=kind, seed=seed, hits_per_1k=1000.0 * hits / steps, misses_per_1k=1000.0 * misses / steps,
             hit_share=hits / max(hits + misses, 1), hunger=float(np.mean(hunger)))
    if w.config.bricks:
        s1 = w.stats()
        d = {k: s1[k] - s0[k] for k in s1}
        r.update(paddle_share=paddle / max(paddle + misses, 1), bricks_per_1k=1000.0 * d["bo_bricks"] / steps,
                 food_per_1k=1000.0 * d["bo_food"] / steps, clears=d["bo_clears"],
                 pref=(d["bo_pref_hit"] - d["bo_pref_stand"]) / max(d["bo_bricks"], 1),
                 **{"rank%d" % k: d["bo_brick_t%d" % k] / max(d["bo_bricks"], 1) for k in range(3)})
    return r


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
    if "pref" in rows[0]:
        f = ("paddle_share", "bricks_per_1k", "food_per_1k", "clears", "pref", "rank0", "rank1", "rank2")
        print("\n| Walker | Paddle hit share | Bricks per 1k | Food per 1k | Clears | Preference | Small | Middle | Big |\n|---|---|---|---|---|---|---|---|---|")
        for k in WALKERS:
            r = [x for x in rows if x["walker"] == k]
            print("| %s | %.3f | %.2f | %.3f | %.1f | %+.3f | %.3f | %.3f | %.3f |" % (k, *(np.mean([x[q] for x in r]) for q in f)))
    if args.out:
        json.dump(rows, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()

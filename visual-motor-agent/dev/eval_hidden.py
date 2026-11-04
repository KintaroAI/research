"""Hidden food (task 00092): how long does the agent remember where the food is? Rows of JSON on stdout.

    aim   frozen: the cosine between the move and the direction to the food, by the time since the food vanished
          (seen, 1, 2, 3-4, 5-8, 9-16, 17-32, over 32 ticks), and food and poison per 1,000 ticks. --show v tests the
          model at food_show = v whatever it lived with (0: the run's own)
    walk  walkers, no model, in a world with food_show = v: blind (a heading redrawn every 32 ticks or so); forgetful
          (chases while the food is seen, walks blind after); perfect (chases always, as if it remembered exactly)
    show  tables

    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 ./venv/bin/python eval_hidden.py aim RUN [RUN ...] > aim.jsonl
    ./venv/bin/python eval_hidden.py walk configs/U_default.json --shows 16 4 1 > walk.jsonl
"""
import argparse
import dataclasses
import json
import os

import numpy as np

BINS = (0, 1, 2, 4, 8, 16, 32, 10 ** 9)                 # ticks since the food vanished: (lo, hi] per bin after "seen"
NAMES = ("seen", "1", "2", "3-4", "5-8", "9-16", "17-32", ">32")
SKIP = 500


def since(env):
    """(N,) ticks since the food vanished, 0 while it is drawn."""
    v = env.config.food_show
    return np.where(env.drawn(), 0, env.food_age - v + 1) if v else np.zeros(env.num_envs, int)


def aim(path, ticks, bodies, seed, show):
    from eval_hunger import Life
    from eval_pieces import relative
    from lifetime import cosine
    from world import make_world

    life = Life(path, bodies, seed)
    if show:
        life.cfg = dataclasses.replace(life.cfg, food_show=show)
    env = make_world(bodies, life.cfg, seed=seed)
    cs, n = np.zeros(len(NAMES)), np.zeros(len(NAMES))
    food = poison = 0
    obs = env.observe()
    for t in range(ticks):
        a = life.act(obs)
        g = np.digitize(since(env), BINS, right=True)
        np.add.at(cs, g, cosine(a[:, :2].astype(np.float64), relative(env, env.target)))
        np.add.at(n, g, 1)
        obs, _, done, inf = env.step(a)
        life.es = done.astype(np.uint8)
        food += int(inf["success"].sum())
        poison += int(inf["gt_hazard_click"].sum())
    life.close()
    k = 1000.0 / (ticks * bodies)
    return dict(mode="aim", run=os.path.basename(path), show=env.config.food_show, food=food * k, poison=poison * k,
                aim=(cs / np.maximum(n, 1)).tolist(), share=(n / n.sum()).tolist())


def walk(cfg_env, show, bodies, ticks, seed, who):
    from eval_food_gap import gaps
    from world import make_config, make_world

    env = make_world(bodies, make_config(dict(cfg_env, food_show=show), auto_reset=True), seed=seed)
    rng = np.random.default_rng([seed, show, len(who)])
    heading = rng.uniform(0, 2 * np.pi, bodies)
    food, poison = np.zeros(bodies), np.zeros(bodies)
    for t in range(ticks):
        dx, dy, _ = gaps(env)
        heading = np.where(rng.random(bodies) < 1 / 32, rng.uniform(0, 2 * np.pi, bodies), heading)
        move = np.stack((np.cos(heading), np.sin(heading)), 1)
        chase = np.clip(np.stack((dx[:, 0], dy[:, 0]), 1) / env.config.max_step_px, -1, 1)
        knows = np.zeros(bodies, bool) if who == "blind" else env.drawn() if who == "forgetful" else np.ones(bodies, bool)
        move = np.where(knows[:, None], chase, move)
        _, _, _, inf = env.step(np.concatenate((move, np.zeros((bodies, 1))), 1).astype(np.float32))
        if t >= SKIP:
            food += inf["success"]
            poison += inf["gt_hazard_click"]
    m = ticks - SKIP
    rate, prate = food / m * 1000, poison / m * 1000
    return dict(mode="walk", who=who, show=show, food=float(rate.mean()), food_half=float(1.96 * rate.std(ddof=1) / np.sqrt(bodies)),
                poison=float(prate.mean()))


def show_tables(paths):
    from eval_hunger import interval

    rows = [json.loads(line) for p in paths for line in open(p)]
    W = [r for r in rows if r["mode"] == "walk"]
    if W:
        print("walkers: food per 1k (95%), poison per 1k")
        for r in sorted(W, key=lambda r: (-r["show"], r["who"])):
            print("  show %2d  %-9s food %6.1f +- %4.1f  poison %5.1f" % (r["show"], r["who"], r["food"], r["food_half"], r["poison"]))
    A = {}
    for r in rows:
        if r["mode"] == "aim":
            arm, s = r["run"].split("_s")
            A.setdefault((arm, r["show"]), {})[int(s)] = r
    if A:
        print("frozen: food, poison per 1k; aim by ticks since the food vanished (mean of the seeds, +- 95%)")
        print("  %-14s %6s %6s  " % ("arm, show", "food", "poison") + " ".join("%13s" % x for x in NAMES))
        for (arm, v), R in sorted(A.items()):
            al = np.array([r["aim"] for r in R.values()])
            cells = []
            for j in range(len(NAMES)):
                x = al[:, j]
                cells.append("%6.3f+-%5.3f" % (x.mean(), interval(x) if len(x) > 1 else float("nan")))
            print("  %-14s %6.1f %6.2f  " % ("%s, %d" % (arm, v), np.mean([r["food"] for r in R.values()]), np.mean([r["poison"] for r in R.values()]))
                  + " ".join(cells))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("what", choices=("aim", "walk", "show"))
    ap.add_argument("runs", nargs="+", help="runs (aim), a config (walk) or row files (show)")
    ap.add_argument("--ticks", type=int, default=1500)
    ap.add_argument("--bodies", type=int, default=16)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--show", type=int, default=0)
    ap.add_argument("--shows", type=int, nargs="+", default=[16, 4, 1])
    a = ap.parse_args()
    if a.what == "show":
        show_tables(a.runs)
    elif a.what == "walk":
        cfg_env = json.load(open(a.runs[0]))["env"]
        for v in a.shows:
            for who in ("blind", "forgetful", "perfect"):
                print(json.dumps(walk(cfg_env, v, 32, 6000, 1, who)), flush=True)
    else:
        for run in a.runs:
            print(json.dumps(aim(run, a.ticks, a.bodies, a.seed, a.show)), flush=True)


if __name__ == "__main__":
    main()

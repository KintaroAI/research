"""How far is the food, whatever the model does? Scripted walkers in the worlds of a set of configs, no model, no GPU (vm-00072).

The small and the big world have the same density, so they look the same to a bite. They need not look the same to a walk: this measures what the
world itself asks for. A walker goes straight on a heading it redraws every 32 ticks or so, and turns to the nearest food once the gap to its edge is
within reach (0: never, a blind walk; 1000: always, a perfect chase). It moves a full step on each axis, the fastest the world allows, and ignores poison.

    python eval_food_gap.py configs/U_default.json configs/U_big.json [--bodies 32] [--ticks 6000] [--seed 1]

Prints a row per world and reach: food per 1,000 ticks with its 95% interval across bodies, the gap to the nearest food right after a bite, and the share
of ticks with food within 16 and within 24 pixels.
"""
import argparse
import json

import numpy as np

from world import make_config, make_world
from world.circle_world import wrap_delta

REACHES = (0, 16, 24, 1000)
SKIP = 500  # ticks left out at the start, while the walkers leave where they were put


def gaps(env):
    """(N, F) dx, dy to each food and the gap to its edge, from the cursor."""
    size = float(env.config.size)
    xy = np.concatenate((env.target[:, None], env.fx_pos), 1)
    rad = np.concatenate((env.radius[:, None], env.fx_rad), 1)
    dx = wrap_delta(xy[..., 0], env.cursor[:, None, 0], size)
    dy = wrap_delta(xy[..., 1], env.cursor[:, None, 1], size)
    return dx, dy, np.hypot(dx, dy) - rad


def walk(cfg_env, reach, bodies, ticks, seed):
    env = make_world(bodies, make_config(cfg_env, auto_reset=True), seed=seed)
    rng = np.random.default_rng([seed, reach])
    heading = rng.uniform(0, 2 * np.pi, bodies)
    food = np.zeros(bodies)
    after, near16, near24 = [], 0, 0
    for t in range(ticks):
        dx, dy, gap = gaps(env)
        best = gap.argmin(1)
        i = np.arange(bodies)
        g = gap[i, best]
        redraw = rng.random(bodies) < 1 / 32
        heading = np.where(redraw, rng.uniform(0, 2 * np.pi, bodies), heading)
        move = np.stack((np.cos(heading), np.sin(heading)), 1)
        chase = np.stack((dx[i, best], dy[i, best]), 1)
        move = np.where((g <= reach)[:, None], np.clip(chase / env.config.max_step_px, -1, 1), move)
        _, _, _, inf = env.step(np.concatenate((move, np.zeros((bodies, 1))), 1).astype(np.float32))
        if t < SKIP:
            continue
        hit = np.asarray(inf["success"], bool)
        food += hit
        near16 += (g <= 16).mean()
        near24 += (g <= 24).mean()
        if hit.any():
            after.append(gaps(env)[2].min(1)[hit])
    n = ticks - SKIP
    rate = food / n * 1000
    after = np.concatenate(after) if after else np.zeros(1)
    half = 1.96 * rate.std(ddof=1) / np.sqrt(bodies)
    return dict(reach=reach, food=float(rate.mean()), half=float(half), after=float(after.mean()), after_far=float((after > 24).mean()),
                near16=near16 / n, near24=near24 / n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("configs", nargs="+")
    ap.add_argument("--bodies", type=int, default=32)
    ap.add_argument("--ticks", type=int, default=6000)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()
    print("world reach | food per 1k | gap after a bite, share beyond 24 | ticks with food within 16, 24")
    for path in args.configs:
        env = json.load(open(path))["env"]
        for reach in REACHES:
            r = walk(env, reach, args.bodies, args.ticks, args.seed)
            print(f"{env.get('size', 64):4d} {reach:5d} | {r['food']:6.1f} +- {r['half']:4.1f} | {r['after']:5.1f}, {r['after_far']:.2f} | {r['near16']:.2f}, {r['near24']:.2f}", flush=True)


if __name__ == "__main__":
    main()

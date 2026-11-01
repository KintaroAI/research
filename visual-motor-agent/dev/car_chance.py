"""Task 00067: what chance gives in the car world: blind drivers, food and poison per 1k ticks.

    MUJOCO_EGL_DEVICE_ID=0 ./venv/bin/python car_chance.py --ticks 20000 --seeds 3
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from world.car_world import CarConfig, CarWorld  # noqa: E402


def walk(rng, kind, hold, last, t, reverse):
    if kind == "straight":
        return np.array([0.0, 1.0, 0.0], np.float32)
    if kind == "random":      # a new random move every tick: it hardly goes anywhere
        return rng.uniform(-1, 1, 3).astype(np.float32)
    if t % hold == 0:         # a held random move: drives around
        return np.array([rng.uniform(-1, 1), rng.uniform(0, 1) if reverse else rng.uniform(-1, 1), 0.0], np.float32)   # speed uniform in 0..1
    return last


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--ticks", type=int, default=20000)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--hold", type=int, default=20)
    ap.add_argument("--foods", type=int, default=3)
    ap.add_argument("--poisons", type=int, default=3)
    ap.add_argument("--frame-skip", type=int, default=10)
    ap.add_argument("--reverse", action="store_true", help="the car of vm-00059..61")
    a = ap.parse_args()
    for kind in ("random", "held", "straight"):
        out = []
        for s in range(a.seeds):
            w = CarWorld(1, CarConfig(foods=a.foods, poisons=a.poisons, frame_skip=a.frame_skip, reverse=a.reverse), seed=s)
            rng = np.random.default_rng(s)
            m = np.zeros(3, np.float32)
            for t in range(a.ticks):
                m = walk(rng, kind, a.hold, m, t, a.reverse)
                w.step(m[None])
            out.append(w.touched * 1000.0 / a.ticks)
            w.close()
        out = np.array(out)
        print("%-9s food %5.1f  poison %5.1f   (per 1k ticks; seeds: %s)" % (kind, out[:, 0].mean(), out[:, 1].mean(),
                                                                          " ".join("%.1f/%.1f" % tuple(r) for r in out)), flush=True)


if __name__ == "__main__":
    main()

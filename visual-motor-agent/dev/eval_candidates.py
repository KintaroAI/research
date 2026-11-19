"""How different the instinct's candidates are (vm-00098, after the write-up): frozen, each run in its own world.

The instinct chooses among m candidate moves that uni_act samples from the model's own move prediction; when that
prediction is sure of one move, the candidates are all that move and the instinct has nothing to choose from. Per run,
over pieces, bodies and ticks: the mean number of distinct candidate moves (dx, dy), the share of ticks where all are
the same move, the median spread of the predicted label (max - min over the candidates), the food per 1k ticks and the
mean hunger. One JSON line per run (mode "candidates").

    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 ./venv/bin/python eval_candidates.py RUN [RUN ...] [--ticks 300]
"""
import argparse
import json
import os

import numpy as np


def candidates(path, ticks, bodies, seed):
    from eval_hunger import Life
    from world import make_world

    life = Life(path, bodies, seed)
    env = make_world(bodies, life.cfg, seed=seed)
    obs = env.observe()
    nd, spr, hunger, food = [], [], [], 0
    for _ in range(ticks):
        a = life.act(obs)
        for m in life.models:
            c = np.round(m.cand[..., :2], 4)                                    # [bodies, m, 2]
            nd.extend(len({tuple(x) for x in cb}) for cb in c)
            spr.extend((m.pred.max(1) - m.pred.min(1)).tolist())
        hunger.extend(obs["drives"][:, 0].tolist())
        obs, _, done, inf = env.step(a)
        life.es = done.astype(np.uint8)
        food += int(np.sum(inf["success"]))
    m = int(life.models[0].cand.shape[1])
    life.close()
    nd = np.array(nd)
    return dict(mode="candidates", run=os.path.basename(path.rstrip("/")), ticks=ticks, bodies=bodies, m=m,
                distinct=float(nd.mean()), all_same=float((nd == 1).mean()), spread=float(np.median(spr)),
                food=1000.0 * food / (ticks * bodies), hunger=float(np.mean(hunger)))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--ticks", type=int, default=300)
    ap.add_argument("--bodies", type=int, default=16)
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    for r in a.runs:
        print(json.dumps(candidates(r, a.ticks, a.bodies, a.seed)), flush=True)


if __name__ == "__main__":
    main()

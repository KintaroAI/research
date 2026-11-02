"""With the eyes closed, who steers, and do the moves follow the words (vm-00070, added after its readings)? Frozen, as lived, on a run of
2 pieces in the sense world. Ticks are split by how long the eyes have been closed: within the context (1-8 ticks: the last views are
still in the window) and past it (9 on). `cos`: the move made against the direction of the nearest thing, by the kind it really is (a
piece's own proposal: cos0, cos1); `won1`: how often piece 1, the text half, made the move.

    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 ./venv/bin/python eval_steer.py RUN [RUN ...] --ticks 1500
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import agent_lib  # noqa: E402
import cuda_lib  # noqa: E402
from lifetime import ckpt_name, pieces_move  # noqa: E402
from online import DEFAULT_ONLINE, uni_act  # noqa: E402
from world import make_config, make_world  # noqa: E402


def one(path, ticks, bodies, seed):
    info = json.load(open(os.path.join(path, "info.json")))
    on = dict(DEFAULT_ONLINE, **info["config"]["online"])
    agents = [agent_lib.Agent.load(os.path.join(path, ckpt_name("final", k, 2))) for k in range(2)]
    models = [cuda_lib.CudaModel(a) for a in agents]
    c = agents[0].config
    L, bins = c["ctx"], c["bins"]
    env = make_world(bodies, make_config(info["config"]["env"], auto_reset=True), seed=seed)
    rng = np.random.default_rng(seed)
    obs = env.observe()
    hist, es = [], np.ones(bodies, np.uint8)
    closed = np.zeros(bodies, int)
    keys = ("seen", "early", "late")
    r = {k: dict(n=0, food=0, poison=0, won1=0, cos_food=[], cos_poison=[], cos1_food=[], cos1_poison=[], cos0_food=[], cos0_poison=[]) for k in keys}
    for t in range(ticks):
        cur = dict(img=obs["image"], drv=obs["drives"], act=np.zeros((bodies, 3), np.float32), es=es)
        win = hist + [cur]
        st = lambda k: np.stack([w[k] for w in win], 1)   # noqa: E731
        props = [uni_act(m, st("img"), st("drv"), st("act"), st("es"), bins, rng, on["instinct_m"], on["instinct_tol"], on["instinct_h"],
                         on["instinct_jacobi"], on["instinct_decode"], per_body=True, first="sample", ring=False) for m in models]
        a, won = pieces_move(models, props, "gain", rng, prev=None)
        d, _ = env.nearest_xy()
        kind = env.said[:, 0]                      # the truth, in both arms
        closed = np.where(env.eyes > 0, closed + 1, 0)
        group = np.where(closed == 0, 0, np.where(closed <= 8, 1, 2))
        cur = dict(cur, act=a)
        hist = (hist + [cur])[-(L - 1):]
        obs, _, done, inf = env.step(a)
        es = done.astype(np.uint8)
        for b in range(bodies):
            g = r[keys[group[b]]]
            g["n"] += 1
            g["food"] += int(inf["success"][b])
            g["poison"] += int(inf["gt_hazard_click"][b])
            g["won1"] += float(np.mean(np.asarray(won[b]) == 1))
            nd = np.linalg.norm(d[b])
            if nd < 1e-6:
                continue
            name = "poison" if kind[b] else "food"
            for tag, mv in (("cos", a[b, :2]), ("cos0", props[0][0][b, :2]), ("cos1", props[1][0][b, :2])):
                nm = np.linalg.norm(mv)
                if nm > 1e-6:
                    g[f"{tag}_{name}"].append(float(mv @ d[b] / nm / nd))
    for m in models:
        m.close()
    return {k: dict(n=g["n"], food=1000.0 * g["food"] / g["n"], poison=1000.0 * g["poison"] / g["n"], won1=g["won1"] / g["n"],
                    **{q: float(np.mean(g[q])) if g[q] else None for q in g if q.startswith("cos")}) for k, g in r.items()}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--ticks", type=int, default=1500)
    ap.add_argument("--bodies", type=int, default=16)
    ap.add_argument("--seed", type=int, default=1234)
    a = ap.parse_args()
    for p in a.runs:
        print(json.dumps(dict(run=os.path.basename(p), **one(p, a.ticks, a.bodies, a.seed))), flush=True)


if __name__ == "__main__":
    main()

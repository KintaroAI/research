"""Frozen lifetimes of one-body runs (task 00059): the checkpoints of a run (one model, or the pieces of online.pieces)
act for --ticks ticks in --bodies copies of its world, without learning. Prints food and poison per 1k ticks; the size
of the moves made; how well a move points at the nearest food (cosine, ego view: the cursor at the centre); with
pieces, each piece's share of the wins by the quadrant the food is in, and with --rule random the same pieces with
the winner drawn at random, so that the selection's own part shows.

Ground truth (gt_*, the world's cursor and circles) is for scoring only and never reaches the model.

    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 ./venv/bin/python eval_pieces.py RUN [RUN ...] --rule gain
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import agent_lib  # noqa: E402
import cuda_lib  # noqa: E402
from lifetime import body_move, cosine, ckpt_name  # noqa: E402
from online import DEFAULT_ONLINE, uni_act  # noqa: E402
from world import make_config, make_world  # noqa: E402


def relative(env, xy):
    """Where the points xy [N, 2] are from the cursor on the torus: (dx, dy) in [-size / 2, size / 2)."""
    size = env.config.size
    return (xy - env.cursor + size / 2) % size - size / 2


def run(path, ticks, bodies, rule, seed):
    info = json.load(open(os.path.join(path, "info.json")))
    on = dict(DEFAULT_ONLINE, **info["config"]["online"])
    S = int(info.get("pieces", 1))
    agents = [agent_lib.Agent.load(os.path.join(path, ckpt_name("final", k, S))) for k in range(S)]
    models = [cuda_lib.CudaModel(a) for a in agents]
    L, bins = agents[0].config["ctx"], agents[0].config["bins"]
    env = make_world(bodies, make_config(info["config"]["env"], auto_reset=True), seed=seed)
    rng = np.random.default_rng(seed)
    use = rule if S > 1 else "mean"
    obs = env.observe()
    hist, es = [], np.ones(bodies, np.uint8)
    food = poison = 0
    size, cos_near, cos_all = [], [], []
    won = np.zeros((4, max(S, 4)))
    near = 16.0
    for t in range(ticks):
        cur = dict(img=obs["image"], drv=obs["drives"], act=np.zeros((bodies, 3), np.float32), es=es)
        win = hist + [cur]
        st = lambda k: np.stack([w[k] for w in win], 1)   # noqa: E731
        props = [uni_act(m, st("img"), st("drv"), st("act"), st("es"), bins, rng, on["instinct_m"], on["instinct_tol"],
                         on["instinct_h"], on["instinct_jacobi"], on["instinct_decode"], per_body=True)
                 for m in models]
        a, w = body_move([p[0] for p in props], use, np.stack([p[3] for p in props]), rng)
        d = relative(env, env.target)
        dist = np.linalg.norm(d, axis=1)
        q = (d[:, 1] >= 0) * 2 + (d[:, 0] >= 0)                 # the food's quadrant in the ego view, as band_k
        c = cosine(a[:, :2].astype(np.float64), d)
        size.append(np.abs(a[:, :2]).mean())
        cos_all.append(c.mean())
        if (dist <= near).any():
            cos_near.append(c[dist <= near].mean())
        if S > 1 and use != "mean":
            np.add.at(won, (q[dist <= near], w[dist <= near, 0]), 1)
        cur = dict(cur, act=a)
        hist = (hist + [cur])[-(L - 1):]
        obs, _, done, inf = env.step(a)
        food += int(inf["success"].sum())
        poison += int(inf["gt_hazard_click"].sum()) if "gt_hazard_click" in inf else 0
        es = done.astype(np.uint8)
    n = ticks * bodies / 1000.0
    print("%-14s %-6s food %5.1f poison %4.1f share %.2f  move size %.3f  cos to food %.3f (within %d px %.3f)" % (
        os.path.basename(path), use, food / n, poison / n, food / max(food + poison, 1), np.mean(size), np.mean(cos_all),
        near, np.mean(cos_near)), flush=True)
    if S > 1 and use != "mean":
        print("  wins by food quadrant, food within %d px (rows TL TR BL BR, columns piece):" % near)
        print(np.array2string(won[:, :S] / np.maximum(won[:, :S].sum(1, keepdims=True), 1), precision=2, floatmode="fixed"))
    for m in models:
        m.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--ticks", type=int, default=1500)
    ap.add_argument("--bodies", type=int, default=16)
    ap.add_argument("--rule", default="gain", help="pieces: the run's rule to use: mean, max, gain, or random")
    ap.add_argument("--seed", type=int, default=1234)
    a = ap.parse_args()
    for r in a.runs:
        run(r, a.ticks, a.bodies, a.rule, a.seed)


if __name__ == "__main__":
    main()

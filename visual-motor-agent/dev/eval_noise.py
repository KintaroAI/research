"""White noise (task 00087): each final model tested at each level of noise, whatever it was raised on. Rows of JSON on stdout.

    chain  foraging at a test level: eval_hunger.py's chain, with food and poison beside a blind walk
    error  the model's prediction of its own patches against the clean frame, on all pixels and on those that hold something: can it find the world under the noise?
    show   tables of train level by test level, from any rows of the two

    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 ./venv/bin/python eval_noise.py chain RUN [RUN ...] --levels 0 .5 .9 1 > chain.jsonl
    ./venv/bin/python eval_noise.py show chain.jsonl error.jsonl
"""
import argparse
import dataclasses
import json
import os

import numpy as np


def error(path, level, ticks, bodies, seed, every):
    import agent_lib
    from eval_hunger import Life
    from online import DECODE
    from world import make_world

    life = Life(path, bodies, seed)
    life.cfg = dataclasses.replace(life.cfg, noise=level)
    env = make_world(bodies, life.cfg, seed=seed)
    cfgs = [m.model.config for m in life.models]
    c = cfgs[0]
    L, ps = c["ctx"], c["patch"]
    own = [agent_lib.band_piece(q) for q in cfgs]
    slots = [agent_lib.uni_layout(q)["patch"] - 1 for q in cfgs]
    bg = life.cfg.bg_color[0] / 255.0
    err, n = np.zeros(2), np.zeros(2)
    obs, hist = env.observe(), []
    for t in range(ticks):
        es = life.es.copy()
        real = env.clean_frame()[..., 0] / 255.0
        a = life.act(obs)
        hist = (hist + [dict(img=obs["image"], drv=obs["drives"], act=a.astype(np.float32), es=es)])[-L:]
        if len(hist) == L and t % every == 0:
            st = lambda f: np.stack([w[f] for w in hist], 1)                                         # noqa: E731
            thing = np.abs(real - bg) > 3.0 / 255
            for m, (r0, r1, c0, c1), sl in zip(life.models, own, slots):
                v = m.model.forward_uni_step(st("img"), st("drv"), st("act"), st("es"), L - 1, mode=DECODE["pair"])["values_t"][:, sl]
                rows, cols = r1 - r0 + 1, c1 - c0 + 1
                v = v.reshape(bodies, rows, cols, ps, ps).transpose(0, 1, 3, 2, 4).reshape(bodies, rows * ps, cols * ps)
                e = np.abs(v - real[:, r0 * ps:(r1 + 1) * ps, c0 * ps:(c1 + 1) * ps])
                on = thing[:, r0 * ps:(r1 + 1) * ps, c0 * ps:(c1 + 1) * ps]
                err += (e.sum(), e[on].sum())
                n += (e.size, on.sum())
        obs, _, done, _ = env.step(a)
        life.es = done.astype(np.uint8)
    life.close()
    return dict(mode="error", run=os.path.basename(path), noise=level, error=(err / np.maximum(n, 1)).tolist())


def show(paths):
    rows = [json.loads(line) for p in paths for line in open(p)]
    arms = sorted({r["run"].split("_s")[0] for r in rows}, key=lambda a: (a != "K4", int("".join(c for c in a if c.isdigit()) or 0)))
    levels = sorted({r["noise"] for r in rows})
    def table(title, mode, value, fmt):
        got = [r for r in rows if r["mode"] == mode]
        if not got:
            return
        print(title)
        print("  raised on  " + "".join("%8g" % q for q in levels))
        for arm in arms:
            cells = [[value(r) for r in got if r["run"].split("_s")[0] == arm and r["noise"] == q] for q in levels]
            print("  %-10s " % arm + "".join((fmt % np.mean(c)) if c else "       ." for c in cells))
    table("food", "chain", lambda r: r["food"]["seen"], "%8.1f")
    table("food above a blind walk", "chain", lambda r: r["food"]["seen"] - r["food"]["blind"], "%8.1f")
    table("poison", "chain", lambda r: r["poison"]["seen"], "%8.2f")
    table("poison below a blind walk", "chain", lambda r: r["poison"]["blind"] - r["poison"]["seen"], "%8.2f")
    table("error against the clean frame, on things", "error", lambda r: r["error"][1], "%8.3f")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("what", choices=("chain", "error", "show"))
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--levels", type=float, nargs="+", default=[0.0])
    ap.add_argument("--ticks", type=int, default=1500)
    ap.add_argument("--bodies", type=int, default=16)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--every", type=int, default=10)
    args = ap.parse_args()
    if args.what == "show":
        show(args.runs)
    else:
        from eval_hunger import chain
        for run in args.runs:
            for q in args.levels:
                if args.what == "error":
                    print(json.dumps(error(run, q, args.ticks, args.bodies, args.seed, args.every), default=float), flush=True)
                else:
                    for row in chain(run, args.ticks, args.bodies, args.seed, noise=q):
                        print(json.dumps(row, default=float), flush=True)

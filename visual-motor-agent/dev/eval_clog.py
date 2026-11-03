"""Can a frozen model restore what does not arrive (task 00085)? It acts on clean frames; every so often the same window is shown to each piece three ways, and the patches
it predicts at the window's last step are scored against the clean frame, on its own patches that the damage chose (30%, half black, half speckle, drawn per piece):

    clean  nothing damaged: the floor
    brief  those patches damaged in the step before only: rebuilt from older steps and neighbours
    long   those patches damaged in every step: only neighbours and summaries can give them
    copy   the baseline: the patch of the damaged frame before, shifted by the last move (copy_brief, copy_long)

Error is the mean absolute difference in [0, 1], on all pixels of those patches and on those that hold something (not background). Rows of JSON on stdout.

    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 ./venv/bin/python eval_clog.py score RUN [RUN ...] > clog.jsonl
    ./venv/bin/python eval_clog.py show clog.jsonl --base K4
"""
import argparse
import json
import os

import numpy as np

CASES = ("clean", "brief", "long", "copy_brief", "copy_long")


def shifted(frame, move, step):
    """frames [n, H, W] as the egocentric view will show them after move [n, 2]: the world goes the other way."""
    out = np.empty_like(frame)
    for b in range(len(frame)):
        out[b] = np.roll(frame[b], (-int(round(move[b, 1] * step)), -int(round(move[b, 0] * step))), (0, 1))
    return out


def score(path, ticks, bodies, seed, p, every):
    import agent_lib
    from eval_hunger import Life
    from online import DECODE, damage
    from world import make_world

    life = Life(path, bodies, seed)
    env = make_world(bodies, life.cfg, seed=seed)
    cfgs = [m.model.config for m in life.models]
    c = cfgs[0]
    L, ps, H, g = c["ctx"], c["patch"], c["img"], c["img"] // c["patch"]
    own = [agent_lib.band_piece(q) for q in cfgs]
    slots = [agent_lib.uni_layout(q)["patch"] - 1 for q in cfgs]
    rng = np.random.default_rng(seed + 1)
    bg, step = life.cfg.bg_color[0] / 255.0, life.cfg.max_step_px
    err = {k: np.zeros(2) for k in CASES}
    n = {k: np.zeros(2) for k in CASES}
    obs = env.observe()
    hist, food, poison = [], 0, 0
    for t in range(ticks):
        es = life.es.copy()
        a = life.act(obs)
        hist = (hist + [dict(img=obs["image"], drv=obs["drives"], act=a.astype(np.float32), es=es)])[-L:]
        if len(hist) == L and t % every == 0:
            st = lambda f: np.stack([w[f] for w in hist], 1)                                         # noqa: E731
            frames, drv, act, ep = st("img"), st("drv"), st("act"), st("es")
            real = frames[:, -1, ..., 0] / 255.0
            for m, (r0, r1, c0, c1), sl in zip(life.models, own, slots):
                chosen = rng.random((bodies, 1, g, g)) < p
                black = rng.random((bodies, 1, g, g)) < 0.5
                inside = np.zeros((g, g), bool)
                inside[r0:r1 + 1, c0:c1 + 1] = True
                last = np.zeros((1, L, 1, 1), bool)
                last[:, L - 2] = True
                brief = damage(frames, chosen & last, black, ps, rng)
                long_ = damage(frames, np.broadcast_to(chosen, (bodies, L, g, g)), black, ps, rng)
                wanted = (chosen[:, 0] & inside)[..., None, None]
                on = np.repeat(np.repeat(wanted[..., 0, 0], ps, 1), ps, 2)
                thing = np.abs(real - bg) > 3.0 / 255
                for name, inp in (("clean", frames), ("brief", brief), ("long", long_)):
                    v = m.model.forward_uni_step(inp, drv, act, ep, L - 1, mode=DECODE["pair"])["values_t"][:, sl]
                    v = v.reshape(bodies, r1 - r0 + 1, c1 - c0 + 1, ps, ps).transpose(0, 1, 3, 2, 4).reshape(bodies, (r1 - r0 + 1) * ps, (c1 - c0 + 1) * ps)
                    full = np.zeros((bodies, H, H), np.float32)
                    full[:, r0 * ps:(r1 + 1) * ps, c0 * ps:(c1 + 1) * ps] = v
                    e = np.abs(full - real)
                    err[name] += (e[on].sum(), e[on & thing].sum())
                    n[name] += (on.sum(), (on & thing).sum())
                    if name != "clean":
                        e = np.abs(shifted(inp[:, -2, ..., 0] / 255.0, hist[-2]["act"], step) - real)
                        err["copy_" + name] += (e[on].sum(), e[on & thing].sum())
                        n["copy_" + name] += (on.sum(), (on & thing).sum())
        obs, _, done, inf = env.step(a)
        life.es = done.astype(np.uint8)
        food += int(inf["success"].sum())
        poison += int(inf["gt_hazard_click"].sum())
    life.close()
    return dict(run=os.path.basename(path), food=food * 1000.0 / (ticks * bodies), poison=poison * 1000.0 / (ticks * bodies),
                **{k: (err[k] / np.maximum(n[k], 1)).tolist() for k in CASES})


def show(paths, base):
    from eval_hunger import interval

    A = {}
    for p in paths:
        for line in open(p):
            r = json.loads(line)
            arm, s = r["run"].split("_s")
            A.setdefault(arm, {})[int(s)] = r
    print("error on all pixels / on things, mean of the seeds")
    for arm, R in sorted(A.items()):
        print("  %-4s %d seeds, food %.1f poison %.2f  " % (arm, len(R), np.mean([r["food"] for r in R.values()]), np.mean([r["poison"] for r in R.values()]))
              + "  ".join("%s %.3f/%.3f" % (k, np.mean([r[k][0] for r in R.values()]), np.mean([r[k][1] for r in R.values()])) for k in CASES))
    if base:
        for arm in sorted(set(A) - {base}):
            for k in ("clean", "brief", "long"):
                seeds = sorted(set(A[arm]) & set(A[base]))
                d = np.array([A[arm][s][k][1] - A[base][s][k][1] for s in seeds])
                print("  %s - %s, %-6s on things %+.4f +- %.4f, seeds %s" % (arm, base, k, d.mean(), interval(d), np.round(d, 3).tolist()))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("what", choices=("score", "show"))
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--ticks", type=int, default=1500)
    ap.add_argument("--bodies", type=int, default=16)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--p", type=float, default=0.3)
    ap.add_argument("--every", type=int, default=10)
    ap.add_argument("--base")
    args = ap.parse_args()
    if args.what == "show":
        show(args.runs, args.base)
    else:
        for run in args.runs:
            print(json.dumps(score(run, args.ticks, args.bodies, args.seed, args.p, args.every), default=float), flush=True)

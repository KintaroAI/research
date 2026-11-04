"""Outage (task 00089): frozen models with their transformers switched off at random at test time, and how well the
live ones could restore an off one's region from what they already predict. Rows of JSON on stdout.

    chain    foraging at a test off share: eval_hunger.py's chain (food and poison beside a blind walk, aim)
    restore  each piece's region as the transformers predict it, every piece on, scored against the real thumbnail of
             that region (its block box-averaged to one patch), on all pixels and on those that hold something:
               owner      the piece's own patches, predicted, then box-averaged
               own_sum    the piece's own prediction of its block's summary thumbnail
               neighbour  the other pieces holding that summary (band_win), their predictions of it, averaged over them
               copy       the frame before, shifted by the last move, box-averaged
             The input works whatever is off, so a neighbour's prediction does not depend on the owner being off: this
             measures what there is to restore with. Nobody but the owner predicts its patches.
    alone    each transformer foraging by itself, every other one off, and all of them together: food and poison per 1,000
             ticks, the same worlds for each
    show     tables: by arm and test off share, and an arm against a base, seed by seed

    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 ./venv/bin/python eval_outage.py chain RUN [RUN ...] --offs 0 .25 .5 > chain.jsonl
    ./venv/bin/python eval_outage.py show chain.jsonl restore.jsonl --base R0
"""
import argparse
import json
import os

import numpy as np

CASES = ("owner", "own_sum", "neighbour", "copy")


def summary_slots(cfg):
    """{(level, row, column): slot} of the pyramid summaries this model holds, as agent_lib.uni_layout places them."""
    import agent_lib
    if cfg.get("uni_sum", 0):
        raise ValueError("row summaries are not handled")
    g, L, gw = cfg["img"] // cfg["patch"], cfg.get("uni_pyr", 0), agent_lib.grid_w(cfg)
    out, slot = {}, 1
    for r in range(g):
        for c in range(gw):
            slot += 1 if agent_lib.has_patch(cfg, r, c) else 0
            for lv in range(1, L + 1):
                if (r + 1) % (1 << lv) == 0 and (c + 1) % (1 << lv) == 0 and agent_lib.has_sum(cfg, lv, r >> lv, c >> lv):
                    out[(lv, r >> lv, c >> lv)] = slot
                    slot += 1
    assert sorted(out.values()) == agent_lib.uni_layout(cfg)["summary"].tolist()
    return out


def box(x, f):
    """[n, H, W] averaged over f x f blocks."""
    n, H, W = x.shape
    return x.reshape(n, H // f, f, W // f, f).mean((2, 4))


def restore(path, ticks, bodies, seed, every):
    import agent_lib
    from eval_clog import shifted
    from eval_hunger import Life
    from online import DECODE
    from world import make_world

    life = Life(path, bodies, seed)
    env = make_world(bodies, life.cfg, seed=seed)
    cfgs = [m.model.config for m in life.models]
    c = cfgs[0]
    L, ps, S = c["ctx"], c["patch"], len(cfgs)
    own = [agent_lib.band_piece(q) for q in cfgs]
    lays = [agent_lib.uni_layout(q) for q in cfgs]
    sums = [summary_slots(q) for q in cfgs]
    h = own[0][1] - own[0][0] + 1                          # a piece's side in patches
    lv = int(round(np.log2(h)))
    if (1 << lv) != h or own[0][3] - own[0][2] + 1 != h:
        raise ValueError("a piece must be one square summary block")
    key = [(lv, r0 >> lv, c0 >> lv) for r0, _, c0, _ in own]
    holders = [[j for j in range(S) if key[k] in sums[j]] for k in range(S)]
    bg, step = life.cfg.bg_color[0] / 255.0, life.cfg.max_step_px
    err = {q: np.zeros(2) for q in CASES}
    n = {q: np.zeros(2) for q in CASES}
    obs, hist = env.observe(), []
    for t in range(ticks):
        es = life.es.copy()
        a = life.act(obs)
        hist = (hist + [dict(img=obs["image"], drv=obs["drives"], act=a.astype(np.float32), es=es)])[-L:]
        if len(hist) == L and t % every == 0:
            st = lambda f: np.stack([w[f] for w in hist], 1)                                   # noqa: E731
            frames = st("img")
            vals = [m.model.forward_uni_step(frames, st("drv"), st("act"), st("es"), L - 1, mode=DECODE["pair"])["values_t"]
                    for m in life.models]
            real = frames[:, -1, ..., 0] / 255.0
            before = shifted(frames[:, -2, ..., 0] / 255.0, hist[-2]["act"], step)
            for k, (r0, r1, c0, c1) in enumerate(own):
                ys, xs = slice(r0 * ps, (r1 + 1) * ps), slice(c0 * ps, (c1 + 1) * ps)
                truth = box(real[:, ys, xs], h)
                thing = np.abs(truth - bg) > 3.0 / 255
                pv = vals[k][:, lays[k]["patch"] - 1].reshape(bodies, h, h, ps, ps).transpose(0, 1, 3, 2, 4).reshape(bodies, h * ps, h * ps)
                got = dict(owner=box(pv, h), own_sum=vals[k][:, sums[k][key[k]] - 1].reshape(bodies, ps, ps),
                           copy=box(before[:, ys, xs], h))
                nb = [vals[j][:, sums[j][key[k]] - 1].reshape(bodies, ps, ps) for j in holders[k] if j != k]
                if nb:
                    got["neighbour"] = np.mean(np.abs(np.stack(nb) - truth[None]), 0)
                for q, v in got.items():
                    e = v if q == "neighbour" else np.abs(v - truth)
                    err[q] += (e.sum(), e[thing].sum())
                    n[q] += (e.size, thing.sum())
        obs, _, done, inf = env.step(a)
        life.es = done.astype(np.uint8)
    life.close()
    return dict(mode="restore", run=os.path.basename(path), level=lv, holders=float(np.mean([len(x) - 1 for x in holders])),
                **{q: (err[q] / np.maximum(n[q], 1)).tolist() for q in CASES})


def alone(path, ticks, bodies, seed):
    from eval_hunger import Life
    from world import make_world

    out = []
    for k in [-1] + list(range(Life.pieces(path))):
        life = Life(path, bodies, seed)
        if k >= 0:   # off = 0: the chain never runs, so the pieces stay as set here
            life.is_off = np.ones(life.S, bool)
            life.is_off[k] = False
        env = make_world(bodies, life.cfg, seed=seed)
        obs, food, poison = env.observe(), 0, 0
        for t in range(ticks):
            a = life.act(obs)
            obs, _, done, inf = env.step(a)
            life.es = done.astype(np.uint8)
            food += int(inf["success"].sum())
            poison += int(inf["gt_hazard_click"].sum())
        life.close()
        out.append((food * 1000.0 / (ticks * bodies), poison * 1000.0 / (ticks * bodies)))
    return dict(mode="alone", run=os.path.basename(path), all=out[0], alone=out[1:])


def show(paths, base):
    from eval_hunger import interval

    rows = [json.loads(line) for p in paths for line in open(p)]
    far = lambda r: float(np.dot(r["cos_seen"][4:6], r["share"][4:6]) / np.sum(r["share"][4:6]))      # noqa: E731
    M = dict(food=lambda r: r["food"]["seen"], lead=lambda r: r["food"]["seen"] - r["food"]["blind"],
             poison=lambda r: r["poison"]["seen"], aim=far)
    C, R, A1 = {}, {}, {}
    for r in rows:
        arm, s = r["run"].split("_s")
        if r["mode"] == "chain":
            C.setdefault((arm, r.get("off", 0.0)), {})[int(s)] = r
        elif r["mode"] == "restore":
            R.setdefault(arm, {})[int(s)] = r
        elif r["mode"] == "alone":
            A1.setdefault(arm, {})[int(s)] = r
    if C:
        print("frozen, mean of the seeds: food, lead over a blind walk, poison, aim at 12-24 px")
        for (arm, q), A in sorted(C.items()):
            print("  %-4s off %.2f  %d seeds  " % (arm, q, len(A)) + "  ".join("%s %.2f" % (k, np.mean([f(r) for r in A.values()])) for k, f in M.items()))
        if base:
            for (arm, q), A in sorted(C.items()):
                if arm == base or (base, q) not in C:
                    continue
                B = C[base, q]
                seeds = sorted(set(A) & set(B))
                print("  %s - %s at off %.2f, seeds %s: " % (arm, base, q, seeds) + "  ".join(
                    "%s %+.2f +- %.2f" % (k, np.mean(d), interval(d) if len(d) > 1 else float("nan")) for k, f in M.items()
                    for d in [np.array([f(A[s]) - f(B[s]) for s in seeds])]))
    if R:
        print("restoration: error against the real thumbnail, all pixels / things")
        for arm, A in sorted(R.items()):
            print("  %-4s %d seeds, %.1f holders  " % (arm, len(A), np.mean([r["holders"] for r in A.values()])) + "  ".join(
                "%s %.3f/%.3f" % (q, np.mean([r[q][0] for r in A.values()]), np.mean([r[q][1] for r in A.values()])) for q in CASES))
    if A1:
        print("alone: food (poison) of all together, and of each transformer by itself, mean of the seeds; its share of all")
        for arm, A in sorted(A1.items()):
            al = np.array([r["all"] for r in A.values()])
            one = np.array([r["alone"] for r in A.values()])                  # [seeds, pieces, 2]
            share = one[..., 0] / al[:, None, 0]
            print("  %-4s %d seeds  all %.1f (%.2f)  alone mean %.1f (%.2f), range %.1f to %.1f  share %.2f, range %.2f to %.2f" % (
                arm, len(A), al[:, 0].mean(), al[:, 1].mean(), one[..., 0].mean(), one[..., 1].mean(), one[..., 0].mean(0).min(),
                one[..., 0].mean(0).max(), share.mean(), share.min(), share.max()))

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("what", choices=("chain", "restore", "alone", "show"))
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--offs", type=float, nargs="+", default=[0.0, 0.25, 0.5])
    ap.add_argument("--ticks", type=int, default=1500)
    ap.add_argument("--bodies", type=int, default=16)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--every", type=int, default=10)
    ap.add_argument("--alone-ticks", type=int, default=600)
    ap.add_argument("--alone-bodies", type=int, default=8)
    ap.add_argument("--base")
    a = ap.parse_args()
    if a.what == "show":
        show(a.runs, a.base)
        return
    from eval_hunger import chain
    for run in a.runs:
        if a.what == "chain":
            for q in a.offs:
                for row in chain(run, a.ticks, a.bodies, a.seed, off=q):
                    print(json.dumps(row), flush=True)
        elif a.what == "alone":
            print(json.dumps(alone(run, a.alone_ticks, a.alone_bodies, a.seed)), flush=True)
        else:
            print(json.dumps(restore(run, a.ticks, a.bodies, a.seed, a.every), default=float), flush=True)


if __name__ == "__main__":
    main()

"""Delayed copies (task 00091): frozen models whose frame is a composite of the view now and in the past. Rows of JSON on stdout.

    error  each transformer's prediction of its own patches against the present, by the copy it sits in (its delay), on
           all pixels and on those that hold something; beside it, the copy's own stale input against the present. A
           patch is predicted from the tokens before it, so its own input of this tick is not yet seen: the stale input
           is a reference to beat, not a fair rival (and 0 for the present copy)
    chain  foraging (eval_hunger.py's chain) as seen, and with the delayed copies blanked to black for every transformer
    show   tables by arm

    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 ./venv/bin/python eval_delay.py error RUN [RUN ...] > error.jsonl
"""
import argparse
import json
import os

import numpy as np


def quarter_of(piece, H, W, ps):
    """Which copy (0 top-left .. 3) a piece's block (r0, r1, c0, c1) in patches lies in."""
    r0, _, c0, _ = piece
    return 2 * int(r0 * ps >= H // 2) + int(c0 * ps >= W // 2)


def blank(frame):
    """The composite with every copy but the present (top-left) black."""
    out = frame.copy()
    h, w = frame.shape[1] // 2, frame.shape[2] // 2
    out[:, :, w:] = 0
    out[:, h:, :] = 0
    return out


def error(path, ticks, bodies, seed, every):
    import agent_lib
    from eval_hunger import Life
    from online import DECODE, delay_targets
    from world import make_world

    life = Life(path, bodies, seed)
    env = make_world(bodies, life.cfg, seed=seed)
    cfgs = [m.model.config for m in life.models]
    c = cfgs[0]
    L, ps, H = c["ctx"], c["patch"], c["img"]
    W = c.get("img_w") or H
    own = [agent_lib.band_piece(q) for q in cfgs]
    slots = [agent_lib.uni_layout(q)["patch"] - 1 for q in cfgs]
    quarter = [quarter_of(p, H, W, ps) for p in own]
    bg = life.cfg.bg_color[0] / 255.0
    err = {k: np.zeros((4, 2)) for k in ("model", "input")}
    n = np.zeros((4, 2))
    obs, hist = env.observe(), []
    for t in range(ticks):
        es = life.es.copy()
        a = life.act(obs)
        hist = (hist + [dict(img=obs["image"], drv=obs["drives"], act=a.astype(np.float32), es=es)])[-L:]
        if len(hist) == L and t % every == 0:
            st = lambda f: np.stack([w[f] for w in hist], 1)                                   # noqa: E731
            frames = st("img")
            seen = frames[:, -1, ..., 0] / 255.0
            now = delay_targets(frames[:, -1])[..., 0] / 255.0
            for m, (r0, r1, c0, c1), sl, q in zip(life.models, own, slots, quarter):
                v = m.model.forward_uni_step(frames, st("drv"), st("act"), st("es"), L - 1, mode=DECODE["pair"])["values_t"][:, sl]
                h, w = r1 - r0 + 1, c1 - c0 + 1
                v = v.reshape(bodies, h, w, ps, ps).transpose(0, 1, 3, 2, 4).reshape(bodies, h * ps, w * ps)
                ys, xs = slice(r0 * ps, (r1 + 1) * ps), slice(c0 * ps, (c1 + 1) * ps)
                truth = now[:, ys, xs]
                thing = np.abs(truth - bg) > 3.0 / 255
                for k, got in (("model", v), ("input", seen[:, ys, xs])):
                    e = np.abs(got - truth)
                    err[k][q] += (e.sum(), e[thing].sum())
                n[q] += (truth.size, thing.sum())
        obs, _, done, inf = env.step(a)
        life.es = done.astype(np.uint8)
    life.close()
    return dict(mode="error", run=os.path.basename(path), delays=list(life.cfg.delays),
                **{k: (v / np.maximum(n, 1)).tolist() for k, v in err.items()})


def show(paths):
    from eval_hunger import interval

    rows = [json.loads(line) for p in paths for line in open(p)]
    E, C = {}, {}
    for r in rows:
        arm, s = r["run"].split("_s")
        if r["mode"] == "error":
            E.setdefault(arm, {})[int(s)] = r
        elif r["mode"] == "chain":
            C.setdefault((arm, r.get("blank", False)), {})[int(s)] = r
    if E:
        print("prediction of the present on things, by copy: the model, and its stale input (mean of the seeds; the model - input, seed by seed)")
        for arm, A in sorted(E.items()):
            d = next(iter(A.values()))["delays"]
            parts = []
            for q in range(4):
                m = np.array([r["model"][q][1] for r in A.values()])
                i = np.array([r["input"][q][1] for r in A.values()])
                parts.append("delay %d: %.3f / %.3f (%+.3f +- %.3f)" % (d[q], m.mean(), i.mean(), (m - i).mean(), interval(m - i) if len(m) > 1 else float("nan")))
            print("  %-3s %d seeds  " % (arm, len(A)) + "   ".join(parts))
    if C:
        print("frozen foraging: food, poison (mean of the seeds); blanked - seen, seed by seed")
        for (arm, b), A in sorted(C.items()):
            print("  %-3s %-7s %d seeds  food %.1f  poison %.2f" % (arm, "blanked" if b else "seen", len(A), np.mean([r["food"]["seen"] for r in A.values()]),
                                                                   np.mean([r["poison"]["seen"] for r in A.values()])))
        for (arm, b), A in sorted(C.items()):
            if b and (arm, False) in C:
                S = C[arm, False]
                seeds = sorted(set(A) & set(S))
                for k in ("food", "poison"):
                    d = np.array([A[s][k]["seen"] - S[s][k]["seen"] for s in seeds])
                    print("  %s blanked - seen, %s %+.2f +- %.2f" % (arm, k, d.mean(), interval(d) if len(d) > 1 else float("nan")))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("what", choices=("error", "chain", "show"))
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--ticks", type=int, default=1500)
    ap.add_argument("--bodies", type=int, default=16)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--every", type=int, default=10)
    a = ap.parse_args()
    if a.what == "show":
        show(a.runs)
        return
    from eval_hunger import chain
    for run in a.runs:
        if a.what == "error":
            print(json.dumps(error(run, a.ticks, a.bodies, a.seed, a.every), default=float), flush=True)
        else:
            for b in (False, True):
                for row in chain(run, a.ticks, a.bodies, a.seed, view=blank if b else None):
                    print(json.dumps(dict(row, blank=b)), flush=True)


if __name__ == "__main__":
    main()

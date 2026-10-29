"""The dream curve (task 00039): how long the universal model's own predictions stay true.

The frozen agent forages (its instinct on, as online.py's forage test) in n fresh worlds; from several points of each
life the model dreams H steps on its own predictions with the moves that were really made next, so every dreamed
step has a true one. Each dreamed step: the body is the previous move token's prediction, the frame is decoded patch
by patch (Jacobi iterations until it stops changing: exactly sequential decoding), values decoded `--decode` (pair by
default); the window slides (the last ctx steps, real ones first). Scored per dream step h: pixels exactly right, the
mean absolute error (grey levels), hunger's absolute error, and the cursor (its white pixels, a scoring rule of this
world): the share of the real cursor's white pixels the dream has white ("cursor_hit"), lost (fewer than 3 white
pixels in the dream), else the distance between the centroids on the torus.

Usage:
    ./venv/bin/python eval_dream.py RUN --ckpts 100000 [--worlds 16 --starts 100 300 500 700 --horizon 16]
Writes RUN/eval/ckpt_<step>_dream[_<decode>].json.
"""
import argparse
import dataclasses
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import agent_lib  # noqa: E402
import cuda_lib  # noqa: E402
from evaluate import executed_action  # noqa: E402
from online import DECODE, DEFAULT_ONLINE, EVAL_SEED, _GpuEval, eval_moves, uni_frame  # noqa: E402
from world import CircleWorld, EnvConfig  # noqa: E402

REPORT = (1, 2, 4, 8, 16)


def record(ev, env_cfg, inst, n, steps, rng):
    """The frozen agent's life in n worlds: frames [n, steps, ...], bodies, moves made."""
    env = CircleWorld(n, dataclasses.replace(env_cfg, auto_reset=False, bonuses=False, max_steps=10 ** 9),
                      seed=EVAL_SEED)
    L = ev.config["ctx"]
    hist, obs = [], env.observe()
    prev_a, es = np.zeros((n, 3), np.float32), np.ones(n, np.uint8)
    img, drv, act = [], [], []
    for _ in range(steps):
        cur = dict(img=obs["image"], drv=obs["drives"], pa=prev_a, cmd=np.zeros(n, np.float32), es=es,
                   act=np.zeros((n, 3), np.float32))
        a = eval_moves(ev, hist + [cur], rng, inst)
        cur["act"] = a
        img.append(obs["image"]), drv.append(obs["drives"]), act.append(a)
        obs, _, _, _ = env.step(a)
        hist = (hist + [cur])[-(L - 1):]
        prev_a, es = executed_action(a), np.zeros(n, np.uint8)
    return np.stack(img, 1), np.stack(drv, 1).astype(np.float32), np.stack(act, 1).astype(np.float32)


def dream(gpu, img, drv, act, H, decode):
    """Windows [N, ctx, ...] ending at the dream's start (act: the real moves of the window and of the H steps after
    it, [N, ctx + H, 3]) -> dreamed frames [N, H, ...] and bodies [N, H, 2]."""
    c = gpu.config
    P = (c["img"] // c["patch"]) ** 2
    lay = agent_lib.uni_layout(c)
    pp, mv = lay["patch"] - 1, lay["move"]   # the tokens that predict the patches; the move token (task 00047)
    N, L = img.shape[:2]
    wi, wd = img.copy(), drv.copy()
    wa = np.ascontiguousarray(act[:, :L])
    we = np.zeros((N, L), np.uint8)
    mode = DECODE[decode]
    frames, bodies = [], []
    for h in range(H):
        o = gpu.forward_uni_step(wi, wd, wa, we, L - 1, mode=mode)          # the next body, from the last move token
        body = np.clip(o["values_t"][:, mv, :2], 0.0, 1.0)
        wi = np.concatenate([wi[:, 1:], wi[:, -1:]], 1)
        wd = np.concatenate([wd[:, 1:], body[:, None]], 1)
        wa = np.ascontiguousarray(np.concatenate([wa[:, 1:], act[:, L + h:L + h + 1]], 1))
        for _ in range(P + 1):                                              # until the frame stops changing
            o = gpu.forward_uni_step(wi, wd, wa, we, L - 1, mode=mode)
            fr = uni_frame(c, o["values_t"][:, pp])
            done = np.array_equal(fr, wi[:, -1])
            wi[:, -1] = fr
            if done:
                break
        frames.append(wi[:, -1].copy()), bodies.append(body)
    return np.stack(frames, 1), np.stack(bodies, 1)


def cursor(f):
    """The white pixels' centroid on the torus [2] (x, y), or None below 3 of them."""
    ys, xs = np.nonzero(f[..., 0] == 255)
    if len(xs) < 3:
        return None
    S = f.shape[0]
    m = lambda v: (np.angle(np.exp(2j * np.pi * v / S).mean()) / (2 * np.pi) * S) % S   # noqa: E731
    return np.array([m(xs), m(ys)])


def score(fd, bd, fr, br):
    """Dreamed against real, frames [N, H, ...] and bodies [N, H, 2] -> per dream step lists."""
    N, H = fd.shape[:2]
    S = fd.shape[2]
    out = dict(exact=[], mae=[], hunger_err=[], cursor_hit=[], cursor_lost=[], cursor_dist=[])
    for h in range(H):
        e = np.abs(fd[:, h, ..., 0].astype(int) - fr[:, h, ..., 0].astype(int))
        out["exact"].append(float((e == 0).mean()))
        out["mae"].append(float(e.mean()))
        out["hunger_err"].append(float(np.abs(bd[:, h, 0] - br[:, h, 0]).mean()))
        white = fr[:, h, ..., 0] == 255
        out["cursor_hit"].append(float((fd[:, h, ..., 0][white] == 255).mean()) if white.any() else float("nan"))
        lost, dist = [], []
        for i in range(N):
            cr = cursor(fr[i, h])
            if cr is None:
                continue
            cd = cursor(fd[i, h])
            lost.append(cd is None)
            if cd is not None:
                d = np.abs(cd - cr)
                dist.append(float(np.hypot(*np.minimum(d, S - d))))
        out["cursor_lost"].append(float(np.mean(lost)) if lost else float("nan"))
        out["cursor_dist"].append(float(np.median(dist)) if dist else float("nan"))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--ckpts", type=int, nargs="+", required=True)
    ap.add_argument("--worlds", type=int, default=16)
    ap.add_argument("--starts", type=int, nargs="+", default=[100, 300, 500, 700])
    ap.add_argument("--horizon", type=int, default=16)
    ap.add_argument("--decode", default="pair", choices=list(DECODE))
    args = ap.parse_args()
    info = json.load(open(os.path.join(args.run, "info.json")))
    ec = {k: tuple(v) if isinstance(v, list) else v for k, v in info["env_config"].items()}
    ion = dict(DEFAULT_ONLINE, **info["config"]["online"])
    inst = (ion["instinct_m"], ion["instinct_tol"], ion["instinct_h"], ion["instinct_jacobi"], ion["instinct_decode"])
    os.makedirs(os.path.join(args.run, "eval"), exist_ok=True)
    H = args.horizon
    for st in args.ckpts:
        agent = agent_lib.Agent.load(os.path.join(args.run, "ckpt_%08d.bin" % st))
        gpu = cuda_lib.CudaModel(agent)
        ev = _GpuEval(agent, gpu)
        L = agent.config["ctx"]
        if min(args.starts) < L - 1:
            raise ValueError("starts must leave ctx real steps")
        img, drv, act = record(ev, EnvConfig(**ec), inst, args.worlds, max(args.starts) + H + 1,
                               np.random.default_rng(0))
        sl = lambda a, s, n: a[:, s + 1 - L:s + 1 - L + n]   # noqa: E731  n steps from the window's first
        wi = np.concatenate([sl(img, s, L) for s in args.starts])
        wd = np.concatenate([sl(drv, s, L) for s in args.starts])
        wa = np.concatenate([sl(act, s, L + H) for s in args.starts])
        fr = np.concatenate([img[:, s + 1:s + 1 + H] for s in args.starts])
        br = np.concatenate([drv[:, s + 1:s + 1 + H] for s in args.starts])
        fd, bd = dream(gpu, wi, wd, wa, H, args.decode)
        res = score(fd, bd, fr, br)
        res.update(run=args.run, step=st, worlds=args.worlds, starts=args.starts, horizon=H, decode=args.decode)
        tag = "" if args.decode == "pair" else "_" + args.decode
        with open(os.path.join(args.run, "eval", "ckpt_%08d_dream%s.json" % (st, tag)), "w") as f:
            json.dump(res, f)
        name = os.path.basename(args.run.rstrip("/"))
        for h in REPORT:
            if h <= H:
                print("%-14s %7d  h %2d  exact %.3f  mae %5.2f  hunger err %.3f  cursor hit %.2f lost %.2f dist %4.1f" % (
                    name, st, h, res["exact"][h - 1], res["mae"][h - 1], res["hunger_err"][h - 1],
                    res["cursor_hit"][h - 1], res["cursor_lost"][h - 1], res["cursor_dist"][h - 1]))
        gpu.close()
        agent.close()


if __name__ == "__main__":
    main()

"""How does the self actor (online.actor=self, vm-00006/7) move, and what do its features hold?

Evaluation only: the true positions are used for scoring, never by the agent. STEP is a checkpoint (env steps).

  aim RUN... --step STEP     the 128 test episodes, with the moves of online.py --eval. Where the cursor is outside the
                             circle and at least 10 px from every wall: the cosine between the predicted mean move and
                             the direction to the circle (with the agent's command, its hunger, and with the command
                             0), and the same for the direction to the screen centre. A constant drift pins the cursor
                             to a wall, from where the circle always lies inward, hence the wall margin. Also the share
                             of steps with the cursor within 2 px of a wall. -> RUN/eval/ckpt_NNNNNNNN_aim.json
  probe RUN... --step STEP   ridge probes from the Transformer output z (one-token windows) and the visual features v
                             to the cursor, the circle and the unit direction between them; R^2 on held-out states.
  relief [RUN --step STEP]   the agent as it lives online (its own moves, babbling with probability explore_eps,
                             auto-reset) in n worlds. For the steps followed by a hit within 12 steps of the same
                             episode (the hindsight-labelled steps): the cosine between the move and the direction to
                             the circle, for own and for babbled moves. Without RUN: random babbling.
  inputs RUN... --step STEP  how much the predicted move changes with another image, with the command 0 and with other
                             drives (mean |d mu| over the test start states), and the norm of v.
  momentum RUN... --step STEP  the test episodes again: does the predicted mean move follow the previous sampled move?
                             The least-squares slope and the mean cosine between the two, with the cursor at least
                             10 px from every wall.

Usage:
    ./venv/bin/python analyze_self_actor.py aim RUN_DIR [RUN_DIR ...] --step 50000
"""
import argparse
import dataclasses
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import agent_lib  # noqa: E402
from evaluate import executed_action  # noqa: E402
from online import EVAL_SEED, sample_moves  # noqa: E402
from world import CircleWorld, EnvConfig  # noqa: E402

FREE = 10.0          # aim: steps with the cursor at least this far from every wall (px)
WALL = 2.0           # closer than this to a wall counts as at the wall (px)
LABEL_K = 12         # hindsight label: a hit within the next LABEL_K steps


def load(run, st):
    info = json.load(open(os.path.join(run, "info.json")))
    ec = dict(info["env_config"])
    for k in ("bg_color", "target_color", "cursor_color", "cursor_outline"):
        ec[k] = tuple(ec[k])
    return info, EnvConfig(**ec), agent_lib.Agent.load(os.path.join(run, "ckpt_%08d.bin" % st))


def name(run):
    return os.path.basename(run.rstrip("/"))


def forward(agent, win, cmd=None):
    x = [np.stack([w[k] for w in win], 1) for k in ("img", "drv", "pa", "cmd", "es")]
    if cmd is not None:
        x[3] = cmd
    return agent.forward(*x)


def aim(run, st, n=128):
    info, cfg, a = load(run, st)
    cfg = dataclasses.replace(cfg, auto_reset=False, bonuses=False)
    goal = info["config"]["online"]["goal"]
    env = CircleWorld(n, cfg, seed=EVAL_SEED)
    rng = np.random.default_rng(0)
    L = a.config["ctx"]
    obs = env.observe()
    hist, pa, es = [], np.zeros((n, 3), np.float32), np.ones(n, np.uint8)
    active = np.ones(n, bool)
    acc = {k: [] for k in ("cos", "along", "cos0", "along0", "cos_act", "cos_center", "dist")}
    wall = []
    for _ in range(cfg.max_steps):
        cur = dict(img=obs["image"], drv=obs["drives"], pa=pa, es=es,
                   cmd=(obs["drives"][:, 0] if goal else np.zeros(n)).astype(np.float32))
        win = hist + [cur]
        out = forward(a, win)
        out0 = forward(a, win, np.zeros((n, len(win)), np.float32)) if goal else out
        act = np.where(active[:, None], sample_moves(a, out, rng), 0.0).astype(np.float32)
        d = env.target - env.cursor
        dist = np.linalg.norm(d, axis=1)
        edge = np.minimum(env.cursor, cfg.size - env.cursor).min(1)
        wall.append(edge[active] < WALL)
        sel = active & (dist > env.radius) & (edge >= FREE)
        if sel.any():
            u = d[sel] / dist[sel, None]
            c = cfg.size / 2.0 - env.cursor[sel]
            uc = c / (np.linalg.norm(c, axis=1, keepdims=True) + 1e-9)
            for key, mu in (("", out["mu"][sel, -1]), ("0", out0["mu"][sel, -1])):
                acc["cos" + key].append((mu * u).sum(1) / (np.linalg.norm(mu, axis=1) + 1e-9))
                acc["along" + key].append((mu * u).sum(1))
            mu = out["mu"][sel, -1]
            acc["cos_center"].append((mu * uc).sum(1) / (np.linalg.norm(mu, axis=1) + 1e-9))
            m = act[sel, :2]
            acc["cos_act"].append((m * u).sum(1) / (np.linalg.norm(m, axis=1) + 1e-9))
            acc["dist"].append(dist[sel])
        obs, _, done, _ = env.step(act)
        active &= ~done
        hist = (hist + [cur])[-(L - 1):] if L > 1 else []
        pa = executed_action(act)
        es = np.zeros(n, np.uint8)
        if not active.any():
            break
    a.close()
    r = {k: float(np.concatenate(v).mean()) for k, v in acc.items() if v}
    near = np.concatenate(acc["dist"]) < 24
    r["cos_near"] = float(np.concatenate(acc["cos"])[near].mean()) if near.any() else float("nan")
    r["steps"] = int(sum(len(v) for v in acc["cos"]))
    r["wall_frac"] = float(np.concatenate(wall).mean())
    return r


def probe_states(cfg, n=4096, seed=99):
    env = CircleWorld(n, dataclasses.replace(cfg, auto_reset=False, bonuses=False), seed=seed)
    rng = np.random.default_rng(seed)
    for _ in range(int(rng.integers(0, 6))):
        env.step(np.concatenate([rng.normal(0, 0.6, (n, 2)), np.zeros((n, 1))], 1).astype(np.float32))
    obs = env.observe()
    d = env.target - env.cursor
    u = d / np.linalg.norm(d, axis=1, keepdims=True)
    return obs, dict(cursor=env.cursor / 64.0, circle=env.target / 64.0, direction=u)


def ridge_r2(X, Y, ntr=3072, lam=1e-2):
    X = np.concatenate([X, np.ones((len(X), 1))], 1)
    mu, sd = X[:ntr].mean(0), X[:ntr].std(0) + 1e-6
    sd[-1], mu[-1] = 1.0, 0.0
    Xs = (X - mu) / sd
    W = np.linalg.solve(Xs[:ntr].T @ Xs[:ntr] + lam * ntr * np.eye(X.shape[1]), Xs[:ntr].T @ Y[:ntr])
    P = Xs[ntr:] @ W
    return float(1 - ((P - Y[ntr:]) ** 2).sum() / ((Y[ntr:] - Y[ntr:].mean(0)) ** 2).sum())


def probe(run, st):
    info, cfg, a = load(run, st)
    obs, gt = probe_states(cfg)
    n = obs["image"].shape[0]
    cmd = obs["drives"][:, :1].astype(np.float32) if info["config"]["online"].get("goal") else np.zeros((n, 1), np.float32)
    out = a.forward(obs["image"][:, None], obs["drives"][:, None].astype(np.float32), np.zeros((n, 1, 3), np.float32),
                    cmd, np.ones((n, 1), np.uint8))
    feats = dict(z=out["z"][:, -1], v=a.encode(obs["image"])["vf"])
    a.close()
    return {"%s_%s" % (f, k): ridge_r2(X, Y) for f, X in feats.items() for k, Y in gt.items()}


def hindsight(hit, done, k=LABEL_K):
    """Steps until the next hit of the same episode (-1 if none within k): [T, n] from [T, n] hit/done."""
    T, n = hit.shape
    before = np.full((T, n), -1)
    nxt = np.full(n, 10 ** 9)
    for t in range(T - 1, -1, -1):
        nxt = np.where(done[t], 10 ** 9, nxt)
        nxt = np.where(hit[t], t, nxt)
        before[t] = np.where(nxt - t < k, nxt - t, -1)
    return before


def relief(run=None, st=None, n=None, T=None):
    if run is None:          # random babbling
        n, T = n or 512, T or 4000
        cfg = dataclasses.replace(EnvConfig(), drives=True, bonuses=False)
        env, rng, a, on = CircleWorld(n, cfg, seed=123), np.random.default_rng(0), None, None
    else:
        n, T = n or 256, T or 1000
        info, cfg, a = load(run, st)
        on = info["config"]["online"]
        cfg = dataclasses.replace(cfg, auto_reset=True, bonuses=False)
        env, rng = CircleWorld(n, cfg, seed=4242), np.random.default_rng(1)
        L = a.config["ctx"]
        hist, pa, es = [], np.zeros((n, 3), np.float32), np.ones(n, np.uint8)
    obs = env.observe()
    rec = {k: [] for k in ("cur", "tgt", "rad", "mv", "bab", "hit", "done")}
    for _ in range(T):
        if a is None:
            act = np.concatenate([rng.normal(0, 0.607, (n, 2)), (rng.random((n, 1)) < 0.1)], 1).astype(np.float32)
            bab = np.ones(n, bool)
        else:
            cur = dict(img=obs["image"], drv=obs["drives"], pa=pa, es=es,
                       cmd=(obs["drives"][:, 0] if on["goal"] else np.zeros(n)).astype(np.float32))
            act = sample_moves(a, forward(a, hist + [cur]), rng)
            bab = rng.random(n) < on["explore_eps"]
            babble = np.concatenate([rng.normal(0, on["babble_sigma"], (n, 2)),
                                     rng.random((n, 1)) < on["babble_click"]], 1)
            act = np.where(bab[:, None], babble, act).astype(np.float32)
        rec["cur"].append(env.cursor.copy())
        rec["tgt"].append(env.target.copy())
        rec["rad"].append(env.radius.copy())
        rec["mv"].append(act[:, :2].copy())
        rec["bab"].append(bab)
        obs, _, done, inf = env.step(act)
        rec["hit"].append(inf["success"].copy())
        rec["done"].append(done.copy())
        if a is not None:
            hist = (hist + [cur])[-(L - 1):]
            pa = executed_action(act)
            es = done.astype(np.uint8)          # the next token starts a new episode
    if a is not None:
        a.close()
    r = {k: np.array(v) for k, v in rec.items()}
    before = hindsight(r["hit"], r["done"])
    lab = before >= 0
    d = r["tgt"] - r["cur"]
    dist = np.linalg.norm(d, axis=2)
    out = dist > r["rad"]
    cos = (r["mv"] * d).sum(2) / (np.linalg.norm(r["mv"], axis=2) * dist + 1e-9)
    res = dict(hits_per_episode=float(r["hit"].sum() / max(r["done"].sum(), 1)), labelled=float(lab.mean()),
               labelled_babbled=float(r["bab"][lab].mean()),
               by_steps_before={j: float(cos[(before == j) & out].mean()) for j in range(1, LABEL_K)
                                if ((before == j) & out).any()})
    groups = [("labelled", lab), ("not labelled", ~lab)]
    if a is not None:
        groups = [("labelled, own moves", lab & ~r["bab"]), ("labelled, babbled", lab & r["bab"]),
                  ("not labelled, own moves", ~lab & ~r["bab"]), ("not labelled, babbled", ~lab & r["bab"])]
    res["cos"] = {g: (int((sel & out).sum()), float(cos[sel & out].mean())) for g, sel in groups}
    return res


def inputs(run, st, n=128):
    info, cfg, a = load(run, st)
    cfg = dataclasses.replace(cfg, auto_reset=False, bonuses=False)
    obs = CircleWorld(n, cfg, seed=EVAL_SEED).observe()
    img, drv = obs["image"][:, None], obs["drives"][:, None].astype(np.float32)
    pa, es = np.zeros((n, 1, 3), np.float32), np.ones((n, 1), np.uint8)
    goal = info["config"]["online"]["goal"]
    cmd = drv[:, :, 0].copy() if goal else np.zeros((n, 1), np.float32)
    roll = np.roll(np.arange(n), 1)
    o = a.forward(img, drv, pa, cmd, es)
    dmu = lambda x: float(np.abs(x["mu"][:, -1] - o["mu"][:, -1]).mean())  # noqa: E731
    r = dict(image=dmu(a.forward(img[roll], drv, pa, cmd, es)),
             command=dmu(a.forward(img, drv, pa, np.zeros_like(cmd), es)) if goal else 0.0,
             drives=dmu(a.forward(img, drv[roll], pa, cmd, es)))
    v = a.encode(CircleWorld(32, cfg, seed=EVAL_SEED).observe()["image"])["vf"]
    r.update(v_norm=float(np.linalg.norm(v, axis=1).mean()), v_active=int((v.max(0) > 0).sum()))
    a.close()
    return r


def momentum(run, st, n=128):
    info, cfg, a = load(run, st)
    cfg = dataclasses.replace(cfg, auto_reset=False, bonuses=False)
    goal = info["config"]["online"]["goal"]
    env = CircleWorld(n, cfg, seed=EVAL_SEED)
    rng = np.random.default_rng(0)
    L = a.config["ctx"]
    obs = env.observe()
    hist, pa, es = [], np.zeros((n, 3), np.float32), np.ones(n, np.uint8)
    active = np.ones(n, bool)
    xs, ys, prev = [], [], None
    for _ in range(cfg.max_steps):
        cur = dict(img=obs["image"], drv=obs["drives"], pa=pa, es=es,
                   cmd=(obs["drives"][:, 0] if goal else np.zeros(n)).astype(np.float32))
        out = forward(a, hist + [cur])
        act = sample_moves(a, out, rng)
        if prev is not None:
            sel = active & (np.minimum(env.cursor, cfg.size - env.cursor).min(1) >= FREE)
            xs.append(prev[sel])
            ys.append(out["mu"][sel, -1])
        prev = act[:, :2].copy()
        obs, _, done, _ = env.step(np.where(active[:, None], act, 0).astype(np.float32))
        active &= ~done
        hist = (hist + [cur])[-(L - 1):] if L > 1 else []
        pa = executed_action(act)
        es = np.zeros(n, np.uint8)
        if not active.any():
            break
    a.close()
    x, y = np.concatenate(xs), np.concatenate(ys)
    return dict(slope=float((x * y).sum() / (x * x).sum()),
                cos=float(((x * y).sum(1) / (np.linalg.norm(x, axis=1) * np.linalg.norm(y, axis=1) + 1e-9)).mean()),
                mu_norm=float(np.linalg.norm(y, axis=1).mean()), steps=len(x))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("what", choices=["aim", "probe", "relief", "inputs", "momentum"])
    ap.add_argument("runs", nargs="*")
    ap.add_argument("--step", type=int, default=50000)
    ap.add_argument("--n", type=int, default=None, help="relief: worlds (default 512 babbling, 256 agent)")
    ap.add_argument("--steps", type=int, default=None, help="relief: steps (default 4000 babbling, 1000 agent)")
    ap.add_argument("--threads", type=int, default=16)
    args = ap.parse_args()
    agent_lib.set_threads(args.threads)
    if args.what == "relief" and not args.runs:
        args.runs = [None]
    for run in args.runs:
        if args.what == "aim":
            out = os.path.join(run, "eval", "ckpt_%08d_aim.json" % args.step)
            if os.path.exists(out):
                r = json.load(open(out))
            else:
                r = aim(run, args.step)
                os.makedirs(os.path.dirname(out), exist_ok=True)
                with open(out, "w") as f:
                    json.dump(r, f)
            print("%-20s %7d  away from walls: cos(mu, to circle) %+.3f (command 0: %+.3f; within 24 px: %+.3f)"
                  "  cos(mu, to centre) %+.3f  along %+.3f (command 0: %+.3f)  cos(move) %+.3f  steps %d;"
                  "  at a wall %.2f" % (name(run), args.step, r["cos"], r["cos0"], r["cos_near"], r["cos_center"],
                                        r["along"], r["along0"], r["cos_act"], r["steps"], r["wall_frac"]), flush=True)
        elif args.what == "probe":
            r = probe(run, args.step)
            print("%-20s %7d  R^2 from z: cursor %.2f  circle %.2f  direction %.2f   from v: cursor %.2f  circle %.2f"
                  "  direction %.2f" % (name(run), args.step, r["z_cursor"], r["z_circle"], r["z_direction"],
                                        r["v_cursor"], r["v_circle"], r["v_direction"]), flush=True)
        elif args.what == "relief":
            r = relief(run, args.step if run else None, args.n, args.steps)
            print("%s: hits per episode %.2f, labelled steps %.3f, of which babbled %.2f" % (
                name(run) if run else "random babbling", r["hits_per_episode"], r["labelled"], r["labelled_babbled"]))
            for g, (k, c) in r["cos"].items():
                print("   %-26s steps %7d  cos(move, to circle) %+.3f" % (g, k, c))
            print("   cos by steps before the hit: " + " ".join("%d:%+.2f" % kv for kv in r["by_steps_before"].items()))
        elif args.what == "momentum":
            r = momentum(run, args.step)
            print("%-20s %7d  away from walls: mu vs the previous sampled move: slope %.3f  cos %+.3f  |mu| %.3f"
                  "  (%d steps)" % (name(run), args.step, r["slope"], r["cos"], r["mu_norm"], r["steps"]), flush=True)
        else:
            r = inputs(run, args.step)
            print("%-20s %7d  change of mu with: another image %.2e  command 0 %.2e  other drives %.2e;  |v| %.3g"
                  " (%d of 128 active)" % (name(run), args.step, r["image"], r["command"], r["drives"], r["v_norm"],
                                           r["v_active"]), flush=True)


if __name__ == "__main__":
    main()

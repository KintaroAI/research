"""What hunger does to the move, on frozen final checkpoints acting on the ring as they lived (a run of online.pieces, with its own rule). Asked of
vm-00069's lives. Rows of JSON on stdout, one per run or arm; `show` reads them back as tables, by arm (the run's name up to _s).

    reach  ticks with food one move away, by the body's real hunger: does the model predict the drop, does the drop clear online.instinct_tol, is the
           lowest predicted hunger a move that touches the food, and is it lost to the tolerance. Each piece's candidates are scored by the world's geometry.
    told   the same models told a false hunger (the world's own is untouched), and the real one with the tolerance at 0: food, poison, aim. Arms paired by seed.
    chain  is there a chain of moves, or chaos and a grab? seen: the model. blind: the same moves, tick by tick, in another world. grab: those blind moves, but
           straight at the food whenever one move can touch it. And the aim at the food by the gap to its edge, seen and blind (the blind aim is the baseline: it
           is above 0, since a path that nears a food is more often moving toward it).

Ground truth (the world's cursor, food and poison) is for scoring only and never reaches the model.

    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 ./venv/bin/python eval_hunger.py reach RUN [RUN ...] --ticks 1500 > reach.jsonl
    ./venv/bin/python eval_hunger.py show reach.jsonl told.jsonl chain.jsonl
"""
import argparse
import json
import os

import numpy as np

EDGES = (0.0, 0.005, 0.01, 0.02, 0.04, 0.1, 1.01)                 # real hunger
LEVELS = ("0-.005", ".005-.01", ".01-.02", ".02-.04", ".04-.1", ".1+")
GAPS = (-1e9, 0.0, 4.0, 8.0, 12.0, 16.0, 24.0, 1e9)               # pixels from the cursor to the food's edge
GAP_NAMES = ("in", "0-4", "4-8", "8-12", "12-16", "16-24", "24+")
TOLD = (("real", None, None), ("tol0", None, 0.0), ("told0", 0.0, None), ("told.05", 0.05, None), ("told.5", 0.5, None))


class Scored:
    """A CudaModel that keeps the instinct's last candidates and their predicted hungers."""

    def __init__(self, model):
        self.model, self.cand, self.pred = model, None, None

    def __getattr__(self, name):
        return getattr(self.model, name)

    @property
    def uni_candidates(self):
        run = self.model.uni_candidates

        def scored(cand):
            self.cand, self.pred = cand.copy(), run(cand).copy()
            return self.pred
        return scored


class Life:
    """A run's frozen pieces, one body per world, and one tick of its acting."""

    def __init__(self, path, bodies, seed):
        import agent_lib
        import cuda_lib
        from lifetime import ckpt_name
        from online import DEFAULT_ONLINE
        from world import make_config

        info = json.load(open(os.path.join(path, "info.json")))
        self.on = dict(DEFAULT_ONLINE, **info["config"]["online"])
        self.S = int(info.get("pieces", 1))
        agents = [agent_lib.Agent.load(os.path.join(path, ckpt_name("final", k, self.S))) for k in range(self.S)]
        self.bins = agents[0].config["bins"]
        self.models = [Scored(cuda_lib.CudaModel(a)) for a in agents]
        self.cfg = make_config(info["config"]["env"], auto_reset=True)
        self.bodies, self.rng = bodies, np.random.default_rng(seed)
        for m in self.models:
            m.uni_ring_reset(bodies)
        self.es = np.ones(bodies, np.uint8)

    def act(self, obs, tol=None, told=None):
        from lifetime import pieces_move
        from online import uni_act

        on, drv = self.on, obs["drives"]
        if told is not None:
            drv = drv.copy()
            drv[:, 0] = told
        act = np.zeros((self.bodies, 1, 3), np.float32)
        tol = on["instinct_tol"] if tol is None else tol
        props = [uni_act(m, obs["image"][:, None], drv[:, None], act, self.es[:, None], self.bins, self.rng, on["instinct_m"], tol, on["instinct_h"],
                         on["instinct_jacobi"], on["instinct_decode"], per_body=True, first="sample", ring=True) for m in self.models]
        a, _ = pieces_move(self.models, props, on["piece_rule"], self.rng, prev=None)
        for m in self.models:
            m.uni_ring_move(a)
        return a

    def close(self):
        for m in self.models:
            m.model.close()


def would_touch(env, cand):
    """cand [N, M, 3] -> food, poison [N, M] bool, as CircleWorld.step."""
    from world.circle_world import _inside, wrap_delta

    c, size = env.config, float(env.config.size)
    new = np.mod(env.cursor[:, None] + np.clip(cand[..., :2], -1, 1) * c.max_step_px, size)
    dx = wrap_delta(new[..., 0], env.target[:, None, 0], size)
    dy = wrap_delta(new[..., 1], env.target[:, None, 1], size)
    food = (dx * dx + dy * dy <= env.radius[:, None] ** 2) & env.target_alive[:, None]
    hx = wrap_delta(new[:, :, None, 0] - env.hz_pos[:, None, :, 0], 0.0, size)
    hy = wrap_delta(new[:, :, None, 1] - env.hz_pos[:, None, :, 1], 0.0, size)
    return food, _inside(hx, hy, env.hz_rad[:, None], c.hazard_shape == "square").any(2)


def in_reach(env):
    """[N] one move can end inside the food, and the move [N, 2] straight at it."""
    from eval_pieces import relative

    d = relative(env, env.target)
    p = env.config.max_step_px
    return np.linalg.norm(d - np.clip(d, -p, p), axis=1) <= env.radius * 0.98, np.clip(d / p, -1, 1)


def reach(path, ticks, bodies, seed):
    from world import make_world

    life = Life(path, bodies, seed)
    env = make_world(bodies, life.cfg, seed=seed)
    tol = life.on["instinct_tol"]
    obs = env.observe()
    T = {k: np.zeros(len(LEVELS)) for k in ("ticks", "reach", "ate", "ate_reach", "pair", "dip", "dip_ok", "best_food", "lost", "override", "spread")}
    P = dict(pair=0, rise=0.0, rise_ok=0, best_poison=0)
    for t in range(ticks):
        h0 = obs["drives"][:, 0].astype(np.float64)
        a = life.act(obs)
        g = np.digitize(h0, EDGES) - 1
        can = np.zeros(bodies, bool)
        for m in life.models:
            food, poison = would_touch(env, m.cand)
            clean = food & ~poison
            can |= clean.any(1)
            pred = m.pred.astype(np.float64)
            for b in range(bodies):
                j, best, other = g[b], pred[b].argmin(), ~food[b] & ~poison[b]
                T["spread"][j] += pred[b].max() - pred[b].min()
                T["override"][j] += pred[b, 0] - pred[b, best] > tol
                if clean[b].any() and other.any():
                    d = pred[b][other].mean() - pred[b][clean[b]].mean()
                    T["pair"][j] += 1
                    T["dip"][j] += d
                    T["dip_ok"][j] += d > tol
                    T["best_food"][j] += clean[b, best]
                    T["lost"][j] += (not clean[b, 0]) and clean[b, best] and pred[b, 0] - pred[b, best] <= tol
                if poison[b].any() and other.any():
                    d = pred[b][poison[b]].mean() - pred[b][other].mean()
                    P["pair"] += 1
                    P["rise"] += d
                    P["rise_ok"] += d > tol
                    P["best_poison"] += poison[b, best]
        obs, _, done, inf = env.step(a)
        life.es = done.astype(np.uint8)
        np.add.at(T["ticks"], g, 1)
        np.add.at(T["reach"], g, can)
        np.add.at(T["ate"], g, inf["success"])
        np.add.at(T["ate_reach"], g, can & inf["success"])
    life.close()
    yield dict(mode="reach", run=os.path.basename(path), pieces=life.S, T={k: v.tolist() for k, v in T.items()}, P=P)


def told(path, ticks, bodies, seed):
    from eval_pieces import relative
    from lifetime import cosine
    from world import make_world

    for arm, level, tol in TOLD:
        life = Life(path, bodies, seed)
        env = make_world(bodies, life.cfg, seed=seed)
        obs = env.observe()
        food = poison = 0
        size, near = [], []
        for t in range(ticks):
            a = life.act(obs, tol, level)
            d = relative(env, env.target)
            c = cosine(a[:, :2].astype(np.float64), d)
            close = np.linalg.norm(d, axis=1) <= 16
            size.append(np.abs(a[:, :2]).mean())
            if close.any():
                near.append(c[close].mean())
            obs, _, done, inf = env.step(a)
            life.es = done.astype(np.uint8)
            food += int(inf["success"].sum())
            poison += int(inf["gt_hazard_click"].sum())
        life.close()
        k = 1000.0 / (ticks * bodies)
        yield dict(mode="told", run=os.path.basename(path), arm=arm, food=food * k, poison=poison * k, size=float(np.mean(size)), cos_near=float(np.mean(near)))


def chain(path, ticks, bodies, seed):
    from eval_pieces import relative
    from lifetime import cosine
    from world import make_world

    life = Life(path, bodies, seed)
    env, blind, grab = (make_world(bodies, life.cfg, seed=seed + k) for k in range(3))
    obs = env.observe()
    food, poison = dict(seen=0, blind=0, grab=0), dict(seen=0, blind=0, grab=0)
    n = len(GAP_NAMES)
    aim, count = dict(seen=np.zeros(n), blind=np.zeros(n)), dict(seen=np.zeros(n), blind=np.zeros(n))
    smooth, last = [], None
    for t in range(ticks):
        a = life.act(obs)
        xy = a[:, :2].astype(np.float64)
        for name, e in (("seen", env), ("blind", blind)):
            d = relative(e, e.target)
            g = np.digitize(np.linalg.norm(d, axis=1) - e.radius, GAPS) - 1
            np.add.at(aim[name], g, cosine(xy, d))
            np.add.at(count[name], g, 1)
        if last is not None:
            smooth.append(cosine(xy, last).mean())
        last = xy
        can, straight = in_reach(grab)
        ag = a.copy()
        ag[can, :2] = straight[can]
        obs, _, done, inf = env.step(a)
        life.es = done.astype(np.uint8)
        infos = dict(seen=inf, blind=blind.step(a)[3], grab=grab.step(ag)[3])
        for name, i in infos.items():
            food[name] += int(i["success"].sum())
            poison[name] += int(i["gt_hazard_click"].sum())
    life.close()
    k = 1000.0 / (ticks * bodies)
    yield dict(mode="chain", run=os.path.basename(path), food={q: v * k for q, v in food.items()}, poison={q: v * k for q, v in poison.items()},
               smooth=float(np.mean(smooth)), share=(count["seen"] / count["seen"].sum()).tolist(),
               cos_seen=(aim["seen"] / np.maximum(count["seen"], 1)).tolist(), cos_blind=(aim["blind"] / np.maximum(count["blind"], 1)).tolist())


def interval(x):
    """The t 95% half-interval of the mean of up to 8 seeds."""
    x = np.asarray(x, float)
    t = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571, 7: 2.447, 8: 2.365}[len(x)]
    return t * x.std(ddof=1) / np.sqrt(len(x))


def show(paths):
    rows = [json.loads(line) for p in paths for line in open(p)]
    for arm in sorted({r["run"].split("_s")[0] for r in rows}):
        R = [r for r in rows if r["run"].split("_s")[0] == arm]
        A = [r for r in R if r["mode"] == "reach"]
        if A:
            T = {k: np.sum([r["T"][k] for r in A], 0) for k in A[0]["T"]}
            print(f"{arm}: {len(A)} seeds, {A[0]['pieces']} pieces. By real hunger, food one move away: share of ticks, in reach, eaten when in reach, predicted drop, "
                  "drop above tolerance, lowest is a food move, lost to tolerance; per piece: override, spread")
            per = lambda k, of: T[k] / np.maximum(of, 1)                                                  # noqa: E731
            for j, name in enumerate(LEVELS):
                print("  %-9s %.3f %.3f %.3f %.4f %.3f %.3f %.3f  %.3f %.4f" % (
                    name, T["ticks"][j] / T["ticks"].sum(), per("reach", T["ticks"])[j], per("ate_reach", T["reach"])[j], per("dip", T["pair"])[j],
                    per("dip_ok", T["pair"])[j], per("best_food", T["pair"])[j], per("lost", T["pair"])[j],
                    per("override", T["ticks"] * A[0]["pieces"])[j], per("spread", T["ticks"] * A[0]["pieces"])[j]))
            print("  share of the food by hunger", np.round(T["ate"] / T["ate"].sum(), 3).tolist())
            P = {k: sum(r["P"][k] for r in A) for k in A[0]["P"]}
            print("  poison in reach: predicted rise %.3f, above tolerance %.3f, lowest is a poison move %.3f" % (
                P["rise"] / P["pair"], P["rise_ok"] / P["pair"], P["best_poison"] / P["pair"]))
        A = [r for r in R if r["mode"] == "told"]
        if A:
            runs = sorted({r["run"] for r in A})
            get = lambda a, k: np.array([next(r[k] for r in A if r["run"] == x and r["arm"] == a) for x in runs])   # noqa: E731
            print(f"{arm}: {len(runs)} seeds. Against the real hunger, paired by seed")
            for a, _, _ in TOLD:
                line = "  %-8s food %.1f poison %.2f near aim %.3f" % (a, get(a, "food").mean(), get(a, "poison").mean(), get(a, "cos_near").mean())
                if a != "real":
                    line += "".join("  d%s %+.2f +- %.2f" % (k, (get(a, k) - get("real", k)).mean(), interval(get(a, k) - get("real", k))) for k in ("food", "poison"))
                print(line)
        A = [r for r in R if r["mode"] == "chain"]
        if A:
            get = lambda f, k: np.array([r[f][k] for r in A])                                             # noqa: E731
            print(f"{arm}: {len(A)} seeds, smoothness {np.mean([r['smooth'] for r in A]):.3f}")
            for k in ("seen", "blind", "grab"):
                print("  %-6s food %.1f +- %.1f poison %.2f +- %.2f" % (k, get("food", k).mean(), interval(get("food", k)), get("poison", k).mean(), interval(get("poison", k))))
            d = get("food", "seen") - get("food", "grab")
            print("  seen - grab %+.1f +- %.1f, seeds %s" % (d.mean(), interval(d), np.round(d, 1).tolist()))
            print("  gap to the food: share, aim seen, aim blind")
            for j, name in enumerate(GAP_NAMES[1:], 1):
                print("  %-6s %.3f %+.3f %+.3f" % (name, np.mean([r["share"][j] for r in A]), np.mean([r["cos_seen"][j] for r in A]), np.mean([r["cos_blind"][j] for r in A])))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("what", choices=("reach", "told", "chain", "show"))
    ap.add_argument("runs", nargs="+", help="run directories (show: the rows printed)")
    ap.add_argument("--ticks", type=int, default=1500)
    ap.add_argument("--bodies", type=int, default=16)
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()
    if args.what == "show":
        show(args.runs)
    else:
        for run in args.runs:
            for row in dict(reach=reach, told=told, chain=chain)[args.what](run, args.ticks, args.bodies, args.seed):
                print(json.dumps(row, default=float), flush=True)

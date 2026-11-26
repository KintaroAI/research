"""What hunger does to the move, on frozen final checkpoints acting on the ring as they lived (a run of online.pieces, with its own rule). Asked of
vm-00069's lives. Rows of JSON on stdout, one per run or arm; `show` reads them back as tables, by arm (the run's name up to _s).

    reach  ticks with food one move away, by the body's real hunger: does the model predict the drop, does the drop clear online.instinct_tol, is the
           lowest predicted hunger a move that touches the food, and is it lost to the tolerance. Each piece's candidates are scored by the world's geometry.
    told   the same models told a false hunger (the world's own is untouched), and the real one with the tolerance at 0: food, poison, aim. Arms paired by seed.
    chain  is there a chain of moves, or chaos and a grab? seen: the model. blind: the same moves, tick by tick, in another world. grab: those blind moves, but
           straight at the food whenever one move can touch it. And the aim at the food by the gap to its edge, seen and blind (the blind aim is the baseline: it
           is above 0, since a path that nears a food is more often moving toward it).
    poison what a touch really costs (vm-00071): the mean of the next 1, 4 and 16 hungers after a touch, against ticks with no touch or food that start at the
           same hunger, how soon the next meal comes, and the mean hunger of the run.
    value  does the label the model predicts for the move it takes match what follows (task 00096)? Each tick's predicted label of the move taken (the
           transformer that made it) against the realized one: with online.value_gamma, the discounted mean of the hungers after it, cut where gamma^i < 1e-3
           and renormalized; else the mean of the next future_k (1 if 0). Bias, mean absolute error, correlation; and the spread of the instinct's candidates.
    move   how the body moves (task 00106, for the inertia world): speed, the share of ticks at top speed, braking (the move against the velocity
           kept), and by the gap to the nearest food and poison: the speed, the move's and the motion's aim at the food, the motion away from poison.

--env KEY=VALUE (task 00106): test each run in its world with that setting changed (e.g. inertia=4); rows named ARM-inertia4_sN.

Ground truth (the world's cursor, food and poison) is for scoring only and never reaches the model.

    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 ./venv/bin/python eval_hunger.py reach RUN [RUN ...] --ticks 1500 > reach.jsonl
    ./venv/bin/python eval_hunger.py show reach.jsonl told.jsonl chain.jsonl [--base K1] [--plot aim.png]

--base: each arm's chain rows against that arm, paired by seed (vm-00071's readings). --plot: aim by the gap to the food, by arm.
compare DIR_BASE DIR_OTHER: two experiments' folders, same arm and seed, from their hunger_*.jsonl and metrics.
"""
import argparse
import dataclasses
import json
import os

import numpy as np

EDGES = (0.0, 0.005, 0.01, 0.02, 0.04, 0.1, 1.01)                 # real hunger
LEVELS = ("0-.005", ".005-.01", ".01-.02", ".02-.04", ".04-.1", ".1+")
GAPS = (-1e9, 0.0, 4.0, 8.0, 12.0, 16.0, 24.0, 1e9)               # pixels from the cursor to the food's edge
GAP_NAMES = ("in", "0-4", "4-8", "8-12", "12-16", "16-24", "24+")
SINCE = (1, 9, 17, 33, 65, 10 ** 9)                                 # task 00102: ticks since any food was last in the window
SINCE_NAMES = ("1-8", "9-16", "17-32", "33-64", "65+")              # task 00103: finer, for contexts of 16 and 32 (vm-00091 used 1-8, 9-32, 33+)
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


class MemoryScore:
    """Tasks 00102, 00103 (and 00105, in life): one world's aim at the nearest unseen food by ticks since any food was last in
    its window (the SINCE bins), on all out-of-view ticks and on those with no meal since food was last in view. Each tick:
    see(env, c) before the world steps (c: each body's cosine of its move to the nearest food), ate(success) after it."""

    def __init__(self, bodies):
        m = len(SINCE_NAMES)
        self.since, self.fed = np.full(bodies, SINCE[-1]), np.zeros(bodies, bool)   # fed: a food eaten since food was last in view
        self.out_aim, self.out_n, self.nm_aim, self.nm_n = np.zeros(m), np.zeros(m), np.zeros(m), np.zeros(m)

    def see(self, env, c):
        m = len(SINCE_NAMES)
        inv = env.foods_in_view().any(1)
        self.since = np.where(inv, 0, np.minimum(self.since + 1, SINCE[-1]))
        self.fed &= ~inv
        o = ~inv
        k = np.minimum(np.digitize(self.since[o], SINCE) - 1, m - 1)
        np.add.at(self.out_aim, k, c[o])
        np.add.at(self.out_n, k, 1)
        o = ~inv & ~self.fed
        k = np.minimum(np.digitize(self.since[o], SINCE) - 1, m - 1)
        np.add.at(self.nm_aim, k, c[o])
        np.add.at(self.nm_n, k, 1)

    def ate(self, success):
        self.fed |= np.asarray(success, bool)

    def take(self):
        """The sums and counts since the last take, then zeroed (the ticks since and the meals carry on)."""
        out = dict(out_aim=self.out_aim.tolist(), out_n=self.out_n.tolist(), nm_aim=self.nm_aim.tolist(), nm_n=self.nm_n.tolist())
        for a in (self.out_aim, self.out_n, self.nm_aim, self.nm_n):
            a[:] = 0
        return out


STATE_MODE = "live"   # task 00104: --state blank, every read of the state blank (0.5); shuffle, each body reads another's
ENV_SET = {}   # task 00106: --env key=value, world settings put over each run's own (rows named ARM-keyvalue_sN)
STEP = "final"   # task 00106: --step N tests the checkpoint saved at tick N (rows named ARM-stepN_sN)


class Life:
    """A run's frozen pieces, one body per world, and one tick of its acting. With the state (task 00104) each piece keeps
    its queue of writes in its ring (cuda_lib), and Life the reads' spread over ticks. A run that lived without the ring
    (online.ring = 0, task 00107) acts as it lived: on windows of its last ctx steps, with its own instinct_h."""

    def __init__(self, path, bodies, seed, clog=0.0, off=0.0, view=None):
        import agent_lib
        import cuda_lib
        from lifetime import ckpt_name
        from online import DEFAULT_ONLINE
        from world import make_config

        info = json.load(open(os.path.join(path, "info.json")))
        self.on = dict(DEFAULT_ONLINE, **info["config"]["online"])
        self.S = int(info.get("pieces", 1))
        agents = [agent_lib.Agent.load(os.path.join(path, ckpt_name(STEP, k, self.S))) for k in range(self.S)]
        self.bins = agents[0].config["bins"]
        self.models = [Scored(cuda_lib.CudaModel(a)) for a in agents]
        self.state = bool(agents[0].config.get("state"))   # task 00104
        for m in self.models:
            m.model.state_blank, m.model.state_shuffle = STATE_MODE == "blank", STATE_MODE == "shuffle"
        self.st1 = self.st2 = None
        self.st_n = self.t = 0
        self.ctx = agents[0].config["ctx"]
        from lifetime import env_at_end
        self.cfg = make_config(dict(env_at_end(info["config"]["env"], info["config"]["online"].get("env_sched")), **ENV_SET),
                               auto_reset=True)   # task 00106: the world the life ended in, and --env over it
        self.bodies, self.rng = bodies, np.random.default_rng(seed)
        self.ring = bool(self.on["ring"])   # task 00107
        if not self.ring and (clog or off or view is not None or self.state):
            raise ValueError("Life without the ring (online.ring = 0) takes no clog, outage, view or state")
        self.hist = []   # task 00107: without the ring, the last ctx - 1 steps
        for m in self.models:
            if self.ring:
                m.uni_ring_reset(bodies)
        self.es = np.ones(bodies, np.uint8)
        self.clogs = [None] * self.S
        if clog:   # task 00085: each piece acts on its own damaged view
            from online import ClogStream

            c = agents[0].config
            self.clogs = [ClogStream(np.random.default_rng([seed, k]), bodies, c["img"], c.get("img_w") or c["img"], c["patch"], clog) for k in range(self.S)]
        if view is not None:   # task 00091: one function of the frame for every piece (e.g. blanked copies)
            self.clogs = [view] * self.S
        # task 00089: pieces off at random, as lifetime.py's online.off, whatever the run lived with (its off_span)
        self.off, self.is_off, self.missed = off, np.zeros(self.S, bool), np.zeros(self.S, int)
        self.off_rng = np.random.default_rng([seed, 0x0FF])
        span = self.on.get("off_span", 32)
        self.p_back, self.p_down = 1.0 / span, (off / ((1.0 - off) * span) if off else 0.0)
        self.live = np.arange(self.S)

    @staticmethod
    def pieces(path):
        return int(json.load(open(os.path.join(path, "info.json"))).get("pieces", 1))

    def act(self, obs, tol=None, told=None):
        from lifetime import pieces_move
        from online import uni_act

        on, drv = self.on, obs["drives"]
        if told is not None:
            drv = drv.copy()
            drv[:, 0] = told
        act = np.zeros((self.bodies, 1, 3), np.float32)
        tol = on["instinct_tol"] if tol is None else tol
        if on.get("need"):   # task 00108: the label weighted by hunger when choosing, as the life chose
            from online import NEED_FLOOR
            tol = (tol / np.maximum(np.asarray(drv, np.float64)[:, 0], NEED_FLOOR)).astype(np.float32)
        if self.off:   # task 00089: each piece's chain; the ticks it missed are skipped in its ring when it is back
            r = self.off_rng.random(self.S)
            self.is_off = np.where(self.is_off, r >= self.p_back, r < self.p_down)
            if self.is_off.all():
                self.is_off[0] = False
        self.live = np.flatnonzero(~self.is_off)
        for k in self.live:
            if self.missed[k]:
                self.models[k].uni_ring_skip(self.missed[k])
                self.missed[k] = 0
        models = [self.models[k] for k in self.live]
        clogs = [self.clogs[k] for k in self.live]
        inst = (self.bins, self.rng, on["instinct_m"], tol, on["instinct_h"], on["instinct_jacobi"], on["instinct_decode"])
        if self.ring:
            props = [uni_act(m, (c(obs["image"]) if c else obs["image"])[:, None], drv[:, None], act, self.es[:, None], *inst,
                             per_body=True, first="sample", ring=True, temp=on.get("instinct_temp", 1.0)) for m, c in zip(models, clogs)]
        else:   # task 00107: the window of the last ctx steps, as forage_video and lifetime.py act without the ring
            cur = dict(img=obs["image"], drv=np.asarray(drv, np.float32), act=np.zeros((self.bodies, 3), np.float32), es=self.es.copy())
            win = self.hist + [cur]
            st = lambda k: np.stack([w[k] for w in win], 1)   # noqa: E731
            props = [uni_act(m, st("img"), st("drv"), st("act"), st("es"), *inst, per_body=True, first="sample",
                             temp=on.get("instinct_temp", 1.0), gamma=on.get("label_gamma", 0.95)) for m in models]
        a, won = pieces_move(models, props, on["piece_rule"], self.rng, prev=None)
        self.decided = True
        if on.get("act_every", 1) > 1:   # task 00131: as the life acted: decided every act_every ticks, held between
            from lifetime import hold_move
            a, won, self.held, self.decided = hold_move(a, won, getattr(self, "held", None), self.t % on["act_every"] == 0)
        hp = np.stack([p[3] for p in props])   # task 00096: the predicted label of the move taken (as lifetime.py stores it)
        self.vh = np.where(won[:, 0] >= 0, hp[np.maximum(won[:, 0], 0), np.arange(len(a))], hp.mean(0))
        if self.ring:
            for m in models:
                m.uni_ring_move(a)
        else:
            cur["act"] = np.asarray(a, np.float32)
            self.hist = (self.hist + [cur])[-(self.ctx - 1):] if self.ctx > 1 else []
        self.missed[self.is_off] += 1
        self.t += 1
        if self.state and self.t > self.ctx:   # task 00104: the reads, for their spread over ticks (after the first ctx, which are blank)
            r = np.stack([m.model.last_read for m in models], 1).astype(np.float64)   # [bodies, pieces, V]
            self.st1 = r if self.st1 is None else self.st1 + r
            self.st2 = r * r if self.st2 is None else self.st2 + r * r
            self.st_n += 1
        return a

    def state_spread(self):
        """Task 00104: the reads' standard deviation over the ticks so far, the mean over bodies, pieces and values; the first
        ctx ticks (blank reads, before the model's own writes come back) left out."""
        if not self.st_n:
            return None
        mu = self.st1 / self.st_n
        return float(np.sqrt(np.maximum(self.st2 / self.st_n - mu * mu, 0.0)).mean())

    def close(self):
        for m in self.models:
            m.model.close()


def would_touch(env, cand):
    """cand [N, M, 3] -> food, poison [N, M] bool, as CircleWorld.step."""
    from world.circle_world import _inside, wrap_delta

    c, size = env.config, float(env.config.size)
    new = np.mod(env.cursor[:, None] + env.push_of(cand), size)   # task 00106: with inertia, from the velocity kept
    dx = wrap_delta(new[..., 0], env.target[:, None, 0], size)
    dy = wrap_delta(new[..., 1], env.target[:, None, 1], size)
    food = (dx * dx + dy * dy <= env.radius[:, None] ** 2) & env.target_alive[:, None]
    for k in range(env.fx_pos.shape[1]):   # the other foods (task 00083)
        dx = wrap_delta(new[..., 0], env.fx_pos[:, None, k, 0], size)
        dy = wrap_delta(new[..., 1], env.fx_pos[:, None, k, 1], size)
        food |= dx * dx + dy * dy <= env.fx_rad[:, None, k] ** 2
    hx = wrap_delta(new[:, :, None, 0] - env.hz_pos[:, None, :, 0], 0.0, size)
    hy = wrap_delta(new[:, :, None, 1] - env.hz_pos[:, None, :, 1], 0.0, size)
    return food, _inside(hx, hy, env.hz_rad[:, None], c.hazard_shape == "square").any(2)


def in_reach(env):
    """[N] one move can end inside the food, and the move [N, 2] straight at it."""
    from eval_pieces import relative

    d = relative(env, env.target)
    c = env.config
    p = c.max_step_px
    if c.inertia > 0:   # task 00106: one push changes the velocity kept by at most p / k per axis, within +-p
        v, a = env.vel * (1.0 - c.inertia_drag), p / c.inertia
        q = np.clip(d, np.maximum(v - a, -p), np.minimum(v + a, p))
        return np.linalg.norm(d - q, axis=1) <= env.radius * 0.98, np.clip((q - v) / a, -1, 1)
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


def chain(path, ticks, bodies, seed, clog=0.0, noise=None, off=0.0, view=None):
    from eval_pieces import relative
    from lifetime import cosine
    from world import make_world

    life = Life(path, bodies, seed, clog, off, view)
    if noise is not None:   # task 00087: tested at a level of white noise, whatever it was raised on
        life.cfg = dataclasses.replace(life.cfg, noise=noise)
    env, blind, grab = (make_world(bodies, life.cfg, seed=seed + k) for k in range(3))
    obs = env.observe()
    food, poison = dict(seen=0, blind=0, grab=0), dict(seen=0, blind=0, grab=0)
    n = len(GAP_NAMES)
    aim, count = dict(seen=np.zeros(n), blind=np.zeros(n)), dict(seen=np.zeros(n), blind=np.zeros(n))
    smooth, last = [], None
    window = bool(getattr(life.cfg, "view", 0))   # task 00102: aim at food out of the window, by ticks since food was last in it
    mem = dict(seen=MemoryScore(bodies), blind=MemoryScore(bodies))
    for t in range(ticks):
        a = life.act(obs)
        xy = a[:, :2].astype(np.float64)
        for name, e in (("seen", env), ("blind", blind)):
            d = relative(e, e.target)
            c = cosine(xy, d)
            g = np.digitize(np.linalg.norm(d, axis=1) - e.radius, GAPS) - 1
            np.add.at(aim[name], g, c)
            np.add.at(count[name], g, 1)
            if window:
                mem[name].see(e, c)
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
            if name in mem:
                mem[name].ate(i["success"])
            poison[name] += int(i["gt_hazard_click"].sum())
    spread = life.state_spread()   # task 00104
    life.close()
    k = 1000.0 / (ticks * bodies)
    yield dict(mode="chain", run=os.path.basename(path), noise=life.cfg.noise, off=life.off, food={q: v * k for q, v in food.items()}, poison={q: v * k for q, v in poison.items()},
               **({} if spread is None else dict(state_spread=spread)),
               smooth=float(np.mean(smooth)), share=(count["seen"] / count["seen"].sum()).tolist(),
               cos_seen=(aim["seen"] / np.maximum(count["seen"], 1)).tolist(), cos_blind=(aim["blind"] / np.maximum(count["blind"], 1)).tolist(),
               **(dict(out_share=float(mem["seen"].out_n.sum() / (ticks * bodies)), out_n=mem["seen"].out_n.tolist(),
                       out_n_blind=mem["blind"].out_n.tolist(),
                       cos_out=(mem["seen"].out_aim / np.maximum(mem["seen"].out_n, 1)).tolist(),
                       cos_out_blind=(mem["blind"].out_aim / np.maximum(mem["blind"].out_n, 1)).tolist(),
                       out_n_nomeal=mem["seen"].nm_n.tolist(), cos_out_nomeal=(mem["seen"].nm_aim / np.maximum(mem["seen"].nm_n, 1)).tolist(),
                       cos_out_nomeal_blind=(mem["blind"].nm_aim / np.maximum(mem["blind"].nm_n, 1)).tolist(), since_names=list(SINCE_NAMES)) if window else {}))


def poison(path, ticks, bodies, seed):
    from world import make_world

    life = Life(path, bodies, seed)
    env = make_world(bodies, life.cfg, seed=seed)
    obs = env.observe()
    H, F, P = [], [], []
    for t in range(ticks):
        H.append(obs["drives"][:, 0].astype(np.float64).copy())
        a = life.act(obs)
        obs, _, done, inf = env.step(a)
        life.es = done.astype(np.uint8)
        F.append(np.asarray(inf["success"], bool).copy())
        P.append(np.asarray(inf["gt_hazard_click"], bool).copy())
    life.close()
    H, F, P = np.array(H), np.array(F), np.array(P)
    row = dict(mode="poison", run=os.path.basename(path), hunger=float(H.mean()), median=float(np.median(H)), food=F.sum() * 1000.0 / F.size, poison=P.sum() * 1000.0 / P.size)
    T = ticks - 1 - 16
    g = np.digitize(H, EDGES) - 1
    for k in (1, 4, 16):
        m = np.stack([H[t + 1:t + 1 + k].mean(0) for t in range(T)])
        touch, plain = P[:T], ~P[:T] & ~F[:T]
        d, n = 0.0, 0
        for j in range(len(EDGES) - 1):
            a, b = touch & (g[:T] == j), plain & (g[:T] == j)
            if a.sum() and b.sum():
                d += a.sum() * (m[a].mean() - m[b].mean())
                n += a.sum()
        row[f"rise{k}"] = d / max(n, 1)
    wait = [nxt[0] - t for b in range(bodies) for t in np.flatnonzero(P[:, b]) for nxt in [np.flatnonzero(F[t + 1:, b]) + t + 1] if len(nxt)]
    row["to_food"] = float(np.mean(wait)) if wait else None
    row["fed_16"] = sum(w <= 16 for w in wait) / max(int(P.sum()), 1)
    yield row


def value(path, ticks, bodies, seed):
    from world import make_world

    life = Life(path, bodies, seed)
    env = make_world(bodies, life.cfg, seed=seed)
    obs = env.observe()
    on = life.on
    gamma = float(on.get("value_gamma") or 0.0)
    if on.get("label") == "meal":   # task 00109: the stored meal label's gamma
        gamma = float(on.get("label_gamma") or 0.0)
    k = max(1, int(on.get("future_k") or 0))
    if gamma:
        M = int(np.ceil(np.log(1e-3) / np.log(gamma)))
        w = (1 - gamma) * gamma ** np.arange(M)
    else:
        M, w = k, np.full(k, 1.0 / k)
    w = w / w.sum()
    meal = on.get("value_target") == "meal" or on.get("label") == "meal"   # task 00108 (and 00109): realized from the hunger
    drift = on.get("value_target") == "meal"   # task 00109's stored label has no cost for the drift, only a poison's rise
    dose = bool(on.get("full_dose"))   # task 00110: the hunger the cap at 1 cut off is counted in the rise
    V, H, H0, F, P, C, spread = [], [], [], [], [], [], []
    for t in range(ticks):
        H0.append(obs["drives"][:, 0].astype(np.float64).copy())
        a = life.act(obs)
        V.append(life.vh.astype(np.float64).copy())
        spread += [float(np.median(m.pred.max(1) - m.pred.min(1))) for m in life.models if m.pred is not None]
        obs, _, done, inf = env.step(a)
        life.es = done.astype(np.uint8)
        H.append(inf["terminal_drives"][:, 0].astype(np.float64).copy())
        F.append(np.asarray(inf["success"], bool).copy())
        P.append(np.asarray(inf["gt_hazard_click"], bool).copy())
        C.append(np.asarray(inf["hunger_cut"], np.float64).copy() if dose else np.zeros(len(H[-1])))
    life.close()
    V, H = np.array(V), np.array(H)
    T = ticks - M
    R = np.stack([(w[:, None] * H[t:t + M]).sum(0) for t in range(T)])
    prof = None
    if meal:   # task 00108: V[t] = r[t] + gamma (1 - meal[t]) V[t + 1], backward from the end (cut at most gamma^M from the truth)
        H0 = np.array(H0)
        ate = H < H0
        from online import POISON_RISE
        rise = H - H0 + np.array(C)   # task 00110: with the full dose, what the cap cut off (0 otherwise)
        rr = np.where(ate, 1.0, np.where(drift | (rise > POISON_RISE), -rise, 0.0))
        Vm, Km = np.zeros_like(H), np.full(H.shape, 10 ** 6)
        nxt, kn = np.zeros(H.shape[1]), np.full(H.shape[1], 10 ** 6)
        for t in range(ticks - 1, -1, -1):
            nxt = rr[t] + gamma * (~ate[t]) * nxt
            kn = np.where(ate[t], 0, kn + 1)
            Vm[t], Km[t] = nxt, kn
        R = (1.0 - np.clip(Vm[:T], -1.0, 1.0)) / 2.0
        ks = list(range(31))   # ticks to the next real meal (0: this tick's move eats)
        pv, rv, nk = [], [], []
        for k_ in ks:
            m_ = Km[:T] == k_
            nk.append(int(m_.sum()))
            pv.append(float(np.mean(1.0 - 2.0 * V[:T][m_])) if m_.any() else None)
            rv.append(float(np.mean(np.clip(Vm[:T][m_], -1.0, 1.0))) if m_.any() else None)
        prof = dict(k=ks, pred_v=pv, real_v=rv, n=nk)
    v, r = V[:T].ravel(), R.ravel()
    yield dict(mode="value", run=os.path.basename(path), gamma=gamma, k=k, horizon=M, pred=float(v.mean()), real=float(r.mean()), bias=float((v - r).mean()),
               mae=float(np.abs(v - r).mean()), corr=float(np.corrcoef(v, r)[0, 1]), spread=float(np.median(spread)) if spread else None,
               food=np.sum(F) * 1000.0 / np.size(F), poison=np.sum(P) * 1000.0 / np.size(P), **({"meal": prof} if prof else {}))


def interval(x):
    """The t 95% half-interval of the mean of up to 12 seeds."""
    x = np.asarray(x, float)
    t = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571, 7: 2.447, 8: 2.365, 9: 2.306, 10: 2.262, 11: 2.228, 12: 2.201}[len(x)]
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
        A = [r for r in R if r["mode"] == "poison"]
        if A:
            print(f"{arm}: {len(A)} seeds. " + ", ".join("%s %.3f" % (k, np.mean([r[k] for r in A])) for k in ("hunger", "median", "rise1", "rise4", "rise16", "to_food", "fed_16")))
        A = [r for r in R if r["mode"] == "value"]
        if A:
            print(f"{arm}: {len(A)} seeds, gamma {A[0]['gamma']}, k {A[0]['k']}, horizon {A[0]['horizon']}. Predicted label of the move taken against the realized one")
            for r in A:
                print("  %-8s pred %.3f real %.3f bias %+.3f mae %.3f corr %.3f spread %.4f food %.1f poison %.2f" % (
                    r["run"], r["pred"], r["real"], r["bias"], r["mae"], r["corr"], float("nan") if r["spread"] is None else r["spread"], r["food"], r["poison"]))
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
            if "cos_out" in A[0]:   # task 00102
                o = np.array([r["out_share"] for r in A])
                print("  no food in view: share of ticks %.3f +- %.3f. Aim at the unseen nearest food by ticks since food was last in view: share of ticks, seen, blind, "
                      "seen - blind over seeds" % (o.mean(), interval(o)))
                names = A[0].get("since_names", ("1-8", "9-32", "33+"))   # rows of vm-00091 carry no names
                for j, name in enumerate(names):
                    n = np.array([r["out_n"][j] / sum(r["out_n"]) * r["out_share"] for r in A])
                    x, y = np.array([r["cos_out"][j] for r in A]), np.array([r["cos_out_blind"][j] for r in A])
                    print("  %-6s %.3f %+.3f %+.3f  %+.3f +- %.3f, positive %d of %d" % (name, n.mean(), x.mean(), y.mean(), (x - y).mean(), interval(x - y),
                                                                                    (x > y).sum(), len(A)))
                if "cos_out_nomeal" in A[0]:
                    print("  the same without the ticks after a meal (no food eaten since food was last in view): share of the out-of-view ticks, seen, blind, seen - blind")
                    for j, name in enumerate(names):
                        n = np.array([r["out_n_nomeal"][j] / max(r["out_n"][j], 1) for r in A])
                        x, y = np.array([r["cos_out_nomeal"][j] for r in A]), np.array([r["cos_out_nomeal_blind"][j] for r in A])
                        print("  %-6s %.3f %+.3f %+.3f  %+.3f +- %.3f, positive %d of %d" % (name, n.mean(), x.mean(), y.mean(), (x - y).mean(), interval(x - y),
                                                                                        (x > y).sum(), len(A)))


MOVE_GAPS = (-1e9, 0.0, 8.0, 16.0, 32.0, 1e9)   # task 00106: pixels from the cursor to the nearest food's or poison's edge
MOVE_GAP_NAMES = ("in", "0-8", "8-16", "16-32", "32+")


def moving_body(path, ticks, bodies, seed):
    """Task 00106: how the frozen agent's body moves in its world (with inertia: how it handles the velocity it keeps). Speed
    (pixels a tick), the share of ticks at top speed on an axis, braking (the move against the velocity kept, cosine below -0.5,
    among ticks at 1 px a tick or more), and by the gap to the nearest food and to the nearest poison: the speed, the move's
    aim and the motion's aim at the food, and the motion away from the poison."""
    from eval_pieces import relative
    from lifetime import cosine
    from world import make_world

    life = Life(path, bodies, seed)
    env = make_world(bodies, life.cfg, seed=seed)
    obs = env.observe()
    p, size, rows = life.cfg.max_step_px, float(life.cfg.size), np.arange(bodies)
    acc = {k: np.zeros(len(MOVE_GAP_NAMES)) for k in ("f_n", "f_speed", "f_aim", "f_motion", "p_n", "p_speed", "p_away")}
    speed, top, brake, moving, food, poisoned, hunger = [], [], 0, 0, 0, 0, []
    for t in range(ticks):
        hunger.append(float(obs["drives"][:, 0].mean()))   # task 00107: the mean hunger too, as the poison mode reads it
        a = life.act(obs)
        xy = a[:, :2].astype(np.float64)
        v0 = env.vel.copy()   # the velocity kept (with inertia), else the last displacement
        d = relative(env, env.target)
        gf = np.digitize(np.linalg.norm(d, axis=1) - env.radius, MOVE_GAPS) - 1
        h = (env.hz_pos - env.cursor[:, None] + size / 2) % size - size / 2   # [N, H, 2] the poisons from the cursor
        hd = np.linalg.norm(h, axis=2) - env.hz_rad
        j = hd.argmin(1)
        dh, gp = h[rows, j], np.digitize(hd[rows, j], MOVE_GAPS) - 1
        obs, _, done, inf = env.step(a)
        life.es = done.astype(np.uint8)
        v1 = env.vel   # the displacement this tick made
        s1 = np.linalg.norm(v1, axis=1)
        speed.append(s1.mean())
        top.append((np.abs(v1).max(1) >= p * 0.999).mean())
        m0 = np.linalg.norm(v0, axis=1) >= 1.0
        moving += int(m0.sum())
        brake += int((m0 & (cosine(xy, v0) < -0.5)).sum())
        for key, g, val in (("f_n", gf, 1.0), ("f_speed", gf, s1), ("f_aim", gf, cosine(xy, d)), ("f_motion", gf, cosine(v1, d)),
                            ("p_n", gp, 1.0), ("p_speed", gp, s1), ("p_away", gp, -cosine(v1, dh))):
            np.add.at(acc[key], g, val)
        food += int(inf["success"].sum())
        poisoned += int(inf["gt_hazard_click"].sum())
    life.close()
    k = 1000.0 / (ticks * bodies)
    per = lambda x, c: (acc[x] / np.maximum(acc[c], 1)).tolist()   # noqa: E731
    yield dict(mode="move", run=os.path.basename(path), inertia=float(life.cfg.inertia), food=food * k, poison=poisoned * k, hunger=float(np.mean(hunger)),
               speed=float(np.mean(speed)), top=float(np.mean(top)), brake=brake / max(moving, 1), moving=moving / (ticks * bodies),
               gaps=list(MOVE_GAP_NAMES), food_share=(acc["f_n"] / acc["f_n"].sum()).tolist(), food_speed=per("f_speed", "f_n"),
               food_aim=per("f_aim", "f_n"), food_motion=per("f_motion", "f_n"), poison_share=(acc["p_n"] / acc["p_n"].sum()).tolist(),
               poison_speed=per("p_speed", "p_n"), poison_away=per("p_away", "p_n"))


def chain_arms(rows):
    A = {}
    for r in rows:
        if r["mode"] == "chain":
            arm, seed = r["run"].split("_s")
            A.setdefault(arm, {})[int(seed)] = r
    return A


def versus(paths, base):
    """Each arm's chain rows against the base arm, seed by seed: aim at gaps of 12-24 pixels, weighted by ticks, and the lead over blind with a perfect grab."""
    A = chain_arms([json.loads(line) for p in paths for line in open(p)])
    far = lambda r: float(np.dot(r["cos_seen"][4:6], r["share"][4:6]) / np.sum(r["share"][4:6]))      # noqa: E731
    near = lambda r: float(np.dot(r["cos_blind"][4:6], r["share"][4:6]) / np.sum(r["share"][4:6]))    # noqa: E731
    lead = lambda r: r["food"]["seen"] - r["food"]["grab"]                                            # noqa: E731
    for arm in sorted(set(A) - {base}):
        seeds = sorted(set(A[arm]) & set(A[base]))
        print(f"{arm} - {base}, {len(seeds)} seeds")
        tests = [("aim 12-24", far), ("aim 12-24, blind", near), ("lead over grab", lead), ("food", lambda r: r["food"]["seen"]), ("poison", lambda r: r["poison"]["seen"])]
        r0 = A[arm][seeds[0]]
        if "cos_out_nomeal" in r0 and len(r0["cos_out_nomeal"]) == len(A[base][seeds[0]]["cos_out_nomeal"]):   # task 00103: memory, same bins only
            for j, nm in enumerate(r0.get("since_names", ("1-8", "9-32", "33+"))):
                tests.append(("unseen %s" % nm, lambda r, j=j: r["cos_out_nomeal"][j] - r["cos_out_nomeal_blind"][j]))
        for name, f in tests:
            d = np.array([f(A[arm][s]) - f(A[base][s]) for s in seeds])
            print("  %-17s %+.3f +- %.3f, seeds %s" % (name, d.mean(), interval(d), np.round(d, 3).tolist()))


def compare(base, other):
    """The arms of two experiments, same arm and seed (vm-00073): other minus base, from each folder's hunger_chain.jsonl, hunger_poison.jsonl and metrics, last 20k."""
    def rows(d, mode):
        out = {}
        for line in open(os.path.join(d, f"hunger_{mode}.jsonl")):
            r = json.loads(line)
            arm, seed = r["run"].split("_s")
            out[arm, int(seed)] = r
        return out

    def life(d, arm, seed, key):
        x = [json.loads(line) for line in open(os.path.join(d, f"{arm}_s{seed}", "metrics.jsonl"))]
        return float(np.mean([r[key] for r in x if r["step"] > x[-1]["step"] - 20000]))

    far = lambda r, k="cos_seen": float(np.dot(r[k][4:6], r["share"][4:6]) / np.sum(r["share"][4:6]))   # noqa: E731
    keys = sorted(rows(base, "chain"), key=lambda a: (int(a[0][1:]), a[1]))
    arms = sorted({a for a, _ in keys}, key=lambda a: int(a[1:]))
    chains, poisons = [rows(d, "chain") for d in (base, other)], [rows(d, "poison") for d in (base, other)]
    measures = (("food, last 20k", lambda i, q: life((base, other)[i], *q, "step_hits")), ("poison, last 20k", lambda i, q: life((base, other)[i], *q, "poison")),
                ("aim 12-24", lambda i, q: far(chains[i][q])), ("lead over grab", lambda i, q: chains[i][q]["food"]["seen"] - chains[i][q]["food"]["grab"]),
                ("mean hunger", lambda i, q: poisons[i][q]["hunger"]))
    for name, f in measures:
        print(name)
        for arm in arms:
            seeds = [s for a, s in keys if a == arm]
            x, y = (np.array([f(i, (arm, s)) for s in seeds]) for i in (0, 1))
            d = y - x
            print("  %-4s %.3f %.3f  %+.3f +- %.3f, seeds %s" % (arm, x.mean(), y.mean(), d.mean(), interval(d), np.round(d, 2).tolist()))
    print("aim by the gap: " + " ".join(GAP_NAMES[1:]))
    for arm in arms:
        for i, c in enumerate(chains):
            print("  %-4s %d  %s" % (arm, i, " ".join("%.2f" % np.mean([c[arm, s]["cos_seen"][j] for a, s in keys if a == arm]) for j in range(1, len(GAP_NAMES)))))


def plot_aim(paths, out):
    """Aim by the gap to the food, one line per arm, the blind aim of all arms as the baseline."""
    from readme_images import COLORS, GRID, INK, MUTED_INK, SURFACE, plt, save

    A = chain_arms([json.loads(line) for p in paths for line in open(p)])
    names = GAP_NAMES[1:]
    fig, ax = plt.subplots(figsize=(6.4, 3.4), dpi=100, facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    x = np.arange(len(names))
    arms = sorted(A, key=lambda a: a + "_")                       # the order of the run folders, so that an arm has readme_images.py's colour
    ends = []
    for i, arm in enumerate(arms):
        y = np.mean([r["cos_seen"][1:] for r in A[arm].values()], 0)
        ax.plot(x, y, color=COLORS[i], lw=2, marker="o", ms=5, mec=SURFACE, mew=1.5, label=arm, solid_capstyle="round")
        ends.append((y[2], arm))
    blind = np.mean([r["cos_blind"][1:] for a in arms for r in A[a].values()], 0)
    ax.plot(x, blind, color=MUTED_INK, lw=2, ls=(0, (1, 2)), label="blind, all arms")
    for y, arm in ends:
        ax.annotate(arm, (2, y), xytext=(0, 7), textcoords="offset points", ha="center", fontsize=8, color=MUTED_INK)
    ax.set_xticks(x, names)
    ax.set_xlabel("gap from the cursor to the food's edge, pixels (a move is at most 4 per axis)", fontsize=9, color=MUTED_INK)
    ax.set_title("Aim at the food: cosine of the move to it", fontsize=11, color=INK, loc="left")
    ax.grid(axis="y", color=GRID, lw=1)
    ax.set_axisbelow(True)
    for k in ("top", "right", "left"):
        ax.spines[k].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=MUTED_INK, labelsize=9, length=0)
    ax.legend(frameon=False, fontsize=8, labelcolor=MUTED_INK, ncol=len(arms) + 1, loc="upper right", handlelength=1.6)
    fig.tight_layout()
    save(fig, out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("what", choices=("reach", "told", "chain", "poison", "value", "move", "show", "compare"))
    ap.add_argument("runs", nargs="+", help="run directories (show: the rows printed)")
    ap.add_argument("--ticks", type=int, default=1500)
    ap.add_argument("--bodies", type=int, default=16)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--clog", type=float, default=0.0, help="chain: each piece acts on a view with this share of its patches damaged (task 00085)")
    ap.add_argument("--base", help="show: the arm the others are read against, paired by seed")
    ap.add_argument("--plot", help="show: where to write the plot of aim by gap")
    ap.add_argument("--state", choices=("live", "blank", "shuffle"), default="live",
                    help="task 00104: blank: every read of the state blank (0.5); shuffle: each body reads another body's note; "
                         "rows named ARM-blank_sN / ARM-shuffle_sN, to pair with ARM_sN")
    ap.add_argument("--step", type=int, help="task 00106: test the checkpoint saved at this tick (default: the final one); rows ARM-stepN_sN")
    ap.add_argument("--env", action="append", default=[], metavar="KEY=VALUE",
                    help="task 00106: test every run in its world with this setting changed (e.g. inertia=4); rows named ARM-inertia4_sN")
    args = ap.parse_args()
    STATE_MODE = args.state
    for kv in args.env:
        k_, _, v_ = kv.partition("=")
        try:
            ENV_SET[k_] = json.loads(v_)
        except ValueError:
            ENV_SET[k_] = v_
    if args.step is not None:
        STEP = args.step
    tags = ([STATE_MODE] if STATE_MODE != "live" else []) + (["step%d" % STEP] if STEP != "final" else []) + ["%s%s" % kv for kv in ENV_SET.items()]
    if args.what == "compare":
        compare(*args.runs)
    elif args.what == "show":
        show(args.runs)
        if args.base:
            versus(args.runs, args.base)
        if args.plot:
            plot_aim(args.runs, args.plot)
    else:
        for run in args.runs:
            how = dict(reach=reach, told=told, chain=lambda *a: chain(*a, args.clog), poison=poison, value=value, move=moving_body)[args.what]
            for row in how(run, args.ticks, args.bodies, args.seed):
                if tags and "run" in row:
                    arm, _, seed = row["run"].rpartition("_s")
                    tag = "-".join(tags)
                    row["run"] = "%s-%s_s%s" % (arm, tag, seed) if arm else row["run"] + "-" + tag
                print(json.dumps(row, default=float), flush=True)

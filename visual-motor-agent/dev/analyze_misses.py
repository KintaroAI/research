"""Why the trained agent misses in Breakout (task 00129, vm-00116).

Subcommands:
    ring DIR   — the ring control: DIR/{ring,full}/s1..12, lifetime.py with learning off on BD's (vm-00103) 60k weights
                 in new courts (seed 100 + s), acting on the ring as it lived and on full windows (online.ring = 0).
                 Prints the hit share by mode and the paired difference full - ring (mean, standard error, 95% interval
                 by t, seeds up and down) over the whole test, over its first 2000 ticks and over the rest. The first
                 2000 ticks repeat the earlier tests of the same weights in the same courts (vm-00103's frozen60k,
                 vm-00114's check), where the difference was first noticed, so the reading is the rest: the new ticks.
                 Equal counts in the first 2000 ticks mean the build did not change the play.
    record OUT.npz RUN [--step N] [--full] [--grid] [--ticks 1500] [--bodies 16] [--seed 1]
               — the miss autopsy's recording: RUN's frozen pieces (the checkpoint saved at tick N; the final one by
                 default) play `bodies` new courts for `ticks` ticks, acting as the run lived, or on full windows with
                 --full. Each tick, from the court's state (for scoring only; nothing of it reaches the model): where
                 the ball will land and in how many ticks, and how far the paddle would still have to travel to be
                 under it; from the agent: each transformer's candidates and their labels, each transformer's pick, the
                 move made and whose it was. --grid (task 00131, a test): each transformer's pick is the best dx of an
                 even grid of 16 by its own label (no dy, no click) in place of the best of its own candidates; the
                 choice among the transformers and the hold are as lived.
    read OUT.npz [OUT.npz ...]
               — the autopsy's tables over the recordings (one per seed): by ticks to landing, how often the move made
                 goes toward the ball and how well each transformer's label orders its candidates; who makes the move;
                 the hit share by how far the paddle had to go in the last 12 ticks; and each miss's kind.

With action repeat (task 00131, vm-00118) `record` also scores each candidate by where the hold leaves the paddle, and it
keeps of each transformer's 16 candidate dx their spread, the number of different values and the share that repeat the
move made on the tick before. `last_ticks`, `spans`, `moves` and `endgame` give one recording's measures for comparing
two agents by seed (`analyze_lives.py repeat` prints them).
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

D = os.path.expanduser("~/data/research/visual-motor-agent")
T95 = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571, 7: 2.447, 8: 2.365, 9: 2.306, 10: 2.262, 11: 2.228, 12: 2.201}
BINS = (1, 3, 5, 9, 17, 33, 10 ** 9)                       # ticks to landing
BIN_NAMES = ("1-2", "3-4", "5-8", "9-16", "17-32", "33+")
LAST = 12                                                  # a ball is read over its last 12 ticks: the paddle crosses its court in 11
FAR = (0.0, 1e-9, 8.0, 16.0, 24.0, 32.0, 1e9)              # px the paddle had to travel, 12 ticks before landing
FAR_NAMES = ("0", "0-8", "8-16", "16-24", "24-32", "32+")
SEEN = 2000                                                # the ticks the earlier frozen tests of BD's 60k weights already covered
CLEAR = 4.0                                                # px: a pair of candidates that end clearly apart (a full move is 4.5)


# ---------------------------------------------------------------------------------------------- the ring control
def counts(path, upto=None):
    """Hits and misses of a Breakout run up to tick `upto` (the whole run if None); the counters are cumulative."""
    rows = [json.loads(line) for line in open(os.path.join(path, "metrics.jsonl"))]
    if upto is not None:
        rows = [r for r in rows if r["step"] <= upto]
    return rows[-1]["bo_hits"], rows[-1]["bo_misses"], rows[-1]["step"]


def paired(a, b):
    """Mean of a - b over seeds, its standard error and 95% half-width, and the seeds above and below zero."""
    d = np.asarray(a) - np.asarray(b)
    se = d.std(ddof=1) / np.sqrt(len(d))
    return d.mean(), se, T95[len(d)] * se, int((d > 0).sum()), int((d < 0).sum())


def ring(out):
    seeds = [s for s in range(1, 13) if all(os.path.exists(f"{out}/{m}/s{s}/metrics.jsonl") for m in ("ring", "full"))]
    if len(seeds) < 2:
        raise SystemExit(f"fewer than 2 seeds with both modes under {out}")
    end = counts(f"{out}/ring/s{seeds[0]}")[2]

    def part(a, b):                                            # hits and misses of ticks a + 1 .. b, per mode and seed
        res = {}
        for m in ("ring", "full"):
            hi = [counts(f"{out}/{m}/s{s}", b)[:2] for s in seeds]
            lo = [counts(f"{out}/{m}/s{s}", a)[:2] if a else (0, 0) for s in seeds]
            res[m] = [(h1 - h0, x1 - x0) for (h1, x1), (h0, x0) in zip(hi, lo)]
        return res

    def line(name, res):
        sh = {m: np.array([h / max(h + x, 1) for h, x in res[m]]) for m in res}
        n = {m: sum(h + x for h, x in res[m]) for m in res}
        mean, se, half, up, down = paired(sh["full"], sh["ring"])
        print(f"{name}: ring {sh['ring'].mean():.3f} ({n['ring']} balls), full {sh['full'].mean():.3f} ({n['full']} balls); "
              f"full - ring {mean:+.3f} +- {half:.3f} (95%; standard error {se:.3f}), {up} up, {down} down of {len(seeds)}")
        return sh

    sh = line(f"all {end} ticks", part(0, end))
    for s, r, f in zip(seeds, sh["ring"], sh["full"]):
        print(f"  s{s:<2d} ring {r:.3f}  full {f:.3f}  {f - r:+.3f}")
    if end > SEEN:   # the first SEEN ticks are the courts and ticks of the earlier tests: the reading is the rest
        line(f"ticks 1-{SEEN}, seen before (vm-00103's frozen60k, vm-00114's check)", part(0, SEEN))
        line(f"ticks {SEEN + 1}-{end}, new (the reading)", part(SEEN, end))
    for m, old in (("ring", f"{D}/exp_00103/frozen60k/BD_s%d"), ("full", f"{D}/exp_00114/check/BD_full/s%d")):   # the build
        same = [(s, counts(f"{out}/{m}/s{s}", SEEN)[:2], counts(old % s)[:2]) for s in seeds if os.path.exists(os.path.join(old % s, "metrics.jsonl"))]
        if same:
            print(f"{m}: the first {SEEN} ticks against the earlier test of the same weights and courts: equal counts in "
                  f"{sum(a == b for _, a, b in same)} of {len(same)} seeds")


# ---------------------------------------------------------------------------------------------- the miss autopsy
def landing(w):
    """By the court's own rules (walls, top; stage 1, no bricks), without the paddle: the ticks until each live ball's
    bottom crosses the paddle's top row, and its left edge on that tick. -1 and NaN for a court with no ball in play,
    and for a ball already past the paddle's row (the world counts its miss a few ticks later, at the frame's edge)."""
    c = w.config
    p, v = w.ball.copy(), w.vel.copy()
    n = len(p)
    ttl, x = np.full(n, -1), np.full(n, np.nan)
    todo = w.live & ~((v[:, 1] > 0) & (p[:, 1] + c.ball > c.paddle_y))
    x_hi = w.hi - c.ball
    for k in range(1, 1000):
        if not todo.any():
            break
        q = p + v
        left, right = todo & (q[:, 0] < w.lo), todo & (q[:, 0] > x_hi)
        q[left, 0] = 2 * w.lo - q[left, 0]
        q[right, 0] = 2 * x_hi - q[right, 0]
        v[left | right, 0] *= -1
        top = todo & (q[:, 1] < c.wall)
        q[top, 1] = 2 * c.wall - q[top, 1]
        v[top, 1] *= -1
        cross = todo & (v[:, 1] > 0) & (p[:, 1] + c.ball <= c.paddle_y) & (q[:, 1] + c.ball > c.paddle_y)
        ttl[cross], x[cross] = k, q[cross, 0]
        todo &= ~cross
        p = np.where(todo[:, None], q, p)
    return ttl, x


def need_after(w, x_land, dx, hold=1):
    """After a move dx ([N] or [N, M], in the world's units): how far the paddle would still have to travel to be under
    the landing ball (0: it would be returned), and the signed gap between the ball's and the paddle's centres. hold
    (a number or [N]; task 00131): the move is made that many ticks running, as a held move is."""
    c = w.config
    dx = np.asarray(dx, np.float64)
    shape = (-1,) + (1,) * (dx.ndim - 1)
    ticks = np.broadcast_to(np.asarray(hold, np.float64), (len(w.paddle),)).reshape(shape)
    pad = np.clip(w.paddle.reshape(shape) + np.clip(dx, -1.0, 1.0) * c.paddle_speed * ticks, w.lo, w.hi - c.paddle_w)
    d = (np.asarray(x_land, np.float64).reshape(shape) + c.ball / 2.0) - (pad + c.paddle_w / 2.0)
    return np.maximum(0.0, np.abs(d) - (c.paddle_w + c.ball) / 2.0), d


def concordance(need, pred, gap=0.5):
    """Per body: of the pairs of candidates that leave the paddle different distances to travel (by more than `gap`
    pixels), the share in which the label (lower is better) prefers the nearer one; a tie in the label counts a half.
    NaN where no pair differs. Also the number of such pairs."""
    dn = need[:, :, None] - need[:, None, :]
    dp = pred[:, :, None] - pred[:, None, :]
    use = dn > gap                                             # the first leaves more to travel: each unordered pair once
    agree = np.where(dp > 0, 1.0, np.where(dp == 0, 0.5, 0.0))
    n = use.sum((1, 2))
    return np.where(n > 0, (agree * use).sum((1, 2)) / np.maximum(n, 1), np.nan), n


def corr(a, b):
    """Per body: the correlation across the candidates of a and b ([N, M]); NaN where either does not vary."""
    a, b = a - a.mean(1, keepdims=True), np.asarray(b, np.float64) - np.asarray(b, np.float64).mean(1, keepdims=True)
    den = np.sqrt((a * a).sum(1) * (b * b).sum(1))
    return np.where(den > 1e-12, (a * b).sum(1) / np.maximum(den, 1e-12), np.nan)


def variety(dx, last, same=0.05):
    """Per body, of its candidates' dx ([N, M]): the standard deviation, the mean, the number of different values, and
    the share that repeat `last` ([N], the dx of the move made on the tick before; NaN: none yet) to within `same`
    (the 32 move bins are .065 apart)."""
    dx = np.asarray(dx, np.float64)
    distinct = np.array([len(np.unique(np.round(r, 6))) for r in dx], np.float64)
    copy = np.where(np.isnan(last), np.nan, (np.abs(dx - np.nan_to_num(last)[:, None]) < same).mean(1))
    return dx.std(1), dx.mean(1), distinct, copy


def record(out, run, step, full, ticks, bodies, seed, grid_act=False):
    import eval_hunger
    import lifetime
    from world import make_world

    eval_hunger.STEP = step
    seen = {}
    orig = lifetime.pieces_move

    def spy(models, props, rule, rng, **kw):                   # Life.act's choice among the pieces: keep what it saw
        extra = []
        for m in models:                                       # before the ring takes the move: two more scorings of the label
            only = m.cand.copy()
            only[..., 1:] = 0.0                                # the same dx with no dy and no click (they do nothing in this world)
            grid = np.zeros_like(m.cand)
            grid[..., 0] = np.linspace(-1.0, 1.0, grid.shape[1])   # dx alone, evenly from full left to full right
            extra.append((np.array(m.model.uni_candidates(only), np.float64), np.array(m.model.uni_candidates(grid), np.float64), grid[..., 0]))
        if grid_act:                                           # the test (task 00131): each piece's pick is the best dx of its even grid
            n = np.arange(len(props[0][0]))
            for k, (p, (_, p_grid, g)) in enumerate(zip(props, extra)):
                mv, best = np.zeros_like(p[0]), p_grid.argmin(1)
                mv[:, 0] = g[n, best]
                props[k] = (mv,) + tuple(p[1:3]) + (p_grid[n, best].astype(np.asarray(p[3]).dtype),) + tuple(p[4:])
        a, won = orig(models, props, rule, rng, **kw)
        seen.update(props=props, won=won, extra=extra)
        return a, won

    lifetime.pieces_move = spy
    life = eval_hunger.Life(run, bodies, seed)
    if full:
        life.ring = False                                      # full windows of the last ctx steps (Life's own path)
    env = make_world(bodies, life.cfg, seed=seed)
    c = env.config
    if getattr(c, "kind", "") != "breakout" or c.bricks:
        raise SystemExit("the autopsy reads Breakout stage 1 (no bricks)")
    S = len(life.models)
    every = int(life.on.get("act_every", 1))                   # task 00131: a deciding tick's move is made this many ticks running
    hkeys = ("conc_hold", "clear_hold", "grid_hold", "grid_clear_hold", "r_need_hold", "best_hold", "made_hold") if every > 1 else ()
    keys = ("ttl", "d", "need", "dx", "won", "quarter", "vx", "conc", "pairs", "clear", "spread", "pick", "best", "made", "label",
            "only", "grid", "grid_clear", "grid_toward", "r_need", "r_dy", "r_click", "decided", "hit", "miss")
    ckeys = ("cand_sd", "cand_distinct", "cand_copy", "cand_mean")   # each piece's 16 candidate dx (task 00131): do they follow the last move?
    rec = {k: [] for k in keys + hkeys + ckeys}
    last_dx = np.full(bodies, np.nan)                          # the dx of the move made on the tick before
    obs = env.observe()
    try:
        for _ in range(ticks):
            ttl, x = landing(env)
            xs = np.where(ttl > 0, x, 0.0)
            need0, d0 = need_after(env, xs, np.zeros(bodies))
            centre = env.ball + c.ball / 2.0
            quarter = 2 * (centre[:, 1] >= c.size / 2.0).astype(int) + (centre[:, 0] >= c.size / 2.0).astype(int)
            vx = env.vel[:, 0].copy()
            a = life.act(obs)
            props, won = seen["props"], seen["won"]
            hold = np.clip(ttl, 1, every)                      # of a move decided now and held: its ticks up to the landing
            H = {k: np.full((bodies, S), np.nan) for k in hkeys}
            C = {k: np.full((bodies, S), np.nan) for k in ckeys}
            conc, pairs, clear, spread, pick, best, only, grid, grid_clear, grid_toward, r_need, r_dy, r_click = (
                np.full((bodies, S), np.nan) for _ in range(13))
            for k, m in enumerate(life.models):
                nc = need_after(env, xs, m.cand[..., 0])[0]
                pred = np.asarray(m.pred, np.float64)
                conc[:, k], pairs[:, k] = concordance(nc, pred)
                clear[:, k] = concordance(nc, pred, CLEAR)[0]
                p_only, p_grid, g = seen["extra"][k]
                only[:, k] = concordance(nc, p_only)[0]
                ng = need_after(env, xs, g)[0]
                grid[:, k], grid_clear[:, k] = concordance(ng, p_grid)[0], concordance(ng, p_grid, CLEAR)[0]
                grid_toward[:, k] = g[np.arange(bodies), p_grid.argmin(1)] * d0 > 0   # the grid's best move goes toward the ball
                r_need[:, k], r_dy[:, k], r_click[:, k] = corr(pred, nc), corr(pred, m.cand[..., 1]), corr(pred, m.cand[..., 2])
                spread[:, k] = pred.max(1) - pred.min(1)
                pick[:, k] = need_after(env, xs, np.asarray(props[k][0])[:, 0])[0]
                best[:, k] = nc.min(1)
                C["cand_sd"][:, k], C["cand_mean"][:, k], C["cand_distinct"][:, k], C["cand_copy"][:, k] = variety(m.cand[..., 0], last_dx)
                if hkeys:                                      # the same candidates and labels, scored by where the hold ends
                    nh, gh = need_after(env, xs, m.cand[..., 0], hold)[0], need_after(env, xs, g, hold)[0]
                    H["conc_hold"][:, k], H["clear_hold"][:, k] = concordance(nh, pred)[0], concordance(nh, pred, CLEAR)[0]
                    H["grid_hold"][:, k], H["grid_clear_hold"][:, k] = concordance(gh, p_grid)[0], concordance(gh, p_grid, CLEAR)[0]
                    H["r_need_hold"][:, k], H["best_hold"][:, k] = corr(pred, nh), nh.min(1)
            made = need_after(env, xs, a[:, 0])[0]
            if hkeys:
                H["made_hold"] = need_after(env, xs, a[:, 0], hold)[0]
            obs, _, done, inf = env.step(a)
            life.es = done.astype(np.uint8)
            for k, v in zip(keys, (ttl, d0, need0, a[:, 0].astype(np.float64), won[:, 0], quarter, vx, conc, pairs, clear, spread, pick, best,
                                   made, np.asarray(life.vh, np.float64), only, grid, grid_clear, grid_toward, r_need, r_dy, r_click,
                                   np.full(bodies, bool(getattr(life, "decided", True))), inf["gt_paddle_hit"], inf["gt_hazard_click"])):
                rec[k].append(np.array(v))
            for k in hkeys:
                rec[k].append(np.array(H[k]))
            for k in ckeys:
                rec[k].append(np.array(C[k]))
            last_dx = a[:, 0].astype(np.float64)
    finally:
        life.close()
        lifetime.pieces_move = orig
    np.savez_compressed(out, **{k: np.stack(v) for k, v in rec.items()}, speed=c.paddle_speed, half=(c.paddle_w + c.ball) / 2.0,
                        run=os.path.basename(run.rstrip("/")), step=str(step), full=int(full), seed=seed, act_every=every, grid_act=int(grid_act))
    hit, miss = np.stack(rec["hit"]).sum(), np.stack(rec["miss"]).sum()
    print(f"{out}: {ticks} ticks x {bodies} bodies, {int(hit)} hits, {int(miss)} misses, share {hit / max(hit + miss, 1):.3f}", flush=True)


def balls(z):
    """One row per ball that came down: returned or not, the px the paddle had to travel LAST ticks before the landing
    (or at the ball's first tick in play, if later), the travel it made in those ticks, the share of its moves that went
    toward the ball (of those made while it still had to travel), what was left at the end, and the ticks read."""
    ttl, need, d, dx, made, speed = z["ttl"], z["need"], z["d"], z["dx"], z["made"], float(z["speed"])
    rows = []
    for t, b in zip(*np.nonzero(ttl == 1)):                    # the tick whose move decides it: returned, or past the paddle
        js = [j for j in range(LAST) if t - j >= 0 and ttl[t - j, b] == j + 1]
        if not js or js != list(range(len(js))):
            continue                                           # the recording began inside this flight
        ts = t - np.array(js)
        act = need[ts, b] > 0
        toward = (dx[ts, b] * d[ts, b] > 0)[act]
        rows.append((bool(z["hit"][t, b]), need[ts[-1], b], np.abs(dx[ts, b]).sum() * speed,
                     toward.mean() if act.any() else np.nan, made[t, b], len(js)))
    return np.array(rows, np.float64).reshape(-1, 6)


def last_ticks(z, upto=8):
    """One recording's measures over the last `upto` ticks before a landing, taken as `read` takes them (on the deciding
    ticks when the run held its moves), for comparing two agents by seed (task 00131): the share of balls returned (the
    whole recording), of moves made toward the ball, the label's order of the candidates (all pairs, and pairs ending
    4+ px apart) by one tick of each candidate and by where the hold leaves the paddle (the same number for a run that
    decides every tick), the label's spread over the candidates (median), the share of the grid's best dx that goes
    toward the ball, and the label's correlation with the px left across the candidates."""
    ttl = z["ttl"]
    dec = z["decided"].astype(bool) if "decided" in z.files else np.ones(ttl.shape, bool)
    m = (ttl > 0) & (ttl <= upto) & dec
    act = m & (z["need"] > 0)
    held = "conc_hold" in z.files
    mean = lambda k, sel: float(np.nanmean(z[k][sel]))         # noqa: E731
    return dict(returned=float(z["hit"].sum()) / max(float((ttl == 1).sum()), 1.0),
                toward=float((z["dx"][act] * z["d"][act] > 0).mean()),
                order=mean("conc", m), clear=mean("clear", m),
                order_hold=mean("conc_hold" if held else "conc", m), clear_hold=mean("clear_hold" if held else "clear", m),
                spread=float(np.nanmedian(z["spread"][m])), grid_toward=mean("grid_toward", act),
                r_need=mean("r_need", m), r_need_hold=mean("r_need_hold" if held else "r_need", m))


def spans(z, k=4, hi=16, near=8.0):
    """What k ticks do to the paddle (task 00131, reading 5: a held move may carry the paddle past the ball). The spans
    of k ticks that start on a deciding tick (every k-th tick for a run that decides every tick) between k + 1 and `hi`
    ticks before a landing, so the same ball is still coming down at the span's end and its landing point has not moved.
    Of the spans that start with the paddle not under the landing point (all of them, those within `near` px of it and
    those further off): the shares that end under it, past it (outside the catching zone on the other side), nearer but
    short of it, and no nearer. Of the spans that start under it: the share that end outside the zone. Also the
    paddle's net travel over a span (px), the number of spans and the share that start away."""
    ttl, need, d = z["ttl"], z["need"], z["d"]
    every = int(z["act_every"]) if "act_every" in z.files else 1
    assert every in (1, k), "the spans must be the run's own holds"
    start = (z["decided"].astype(bool) if every > 1 else (np.arange(len(ttl)) % k == 0)[:, None] & np.ones(ttl.shape, bool))
    start = start & (ttl > k) & (ttl <= hi)
    start[len(ttl) - k:] = False
    t0, b0 = np.nonzero(start)
    assert (ttl[t0 + k, b0] == ttl[t0, b0] - k).all()          # the same flight at the span's end
    n0, n1, d0, d1 = need[t0, b0], need[t0 + k, b0], d[t0, b0], d[t0 + k, b0]
    same = d0 * d1 > 0
    out = dict(spans=len(t0), away=float((n0 > 0).mean()), travel=float(np.abs(d1 - d0).mean()),
               left=float((n1[n0 == 0] > 0).mean()) if (n0 == 0).any() else np.nan)
    for name, g in (("all", n0 > 0), ("near", (n0 > 0) & (n0 <= near)), ("far", n0 > near)):
        n = max(int(g.sum()), 1)
        out[name] = int(g.sum())
        for key, m in (("under", n1 == 0), ("past", (n1 > 0) & ~same), ("short", (n1 > 0) & same & (n1 < n0)), ("worse", (n1 > 0) & same & (n1 >= n0))):
            out[name + "_" + key] = float((g & m).sum()) / n
    return out


def moves(z, k=4, hi=16, near=8.0):
    """How one recording's paddle moves (task 00131), on the deciding ticks with a ball in play unless said otherwise:
    the share of all ticks on which the move is cut by a wall (the paddle travels less than the move asks), the size of
    the move (abs dx; 1 is full speed), the share of decisions that go the same way as the decision before (for a run
    that decides every tick: as the move k ticks before, the same gap in time, and `same1`: as the tick before), the
    share that go toward the ball between k + 1 and `hi` ticks before a landing with the paddle within `near` px of
    the catching zone and further off, and, where the candidates were recorded, of each transformer's 16 candidates
    the share that repeat the move made on the tick before, the number of different dx and their standard deviation
    (on the held ticks too, for a run that holds)."""
    ttl, d, need, dx, speed = z["ttl"], z["d"], z["need"], np.clip(z["dx"], -1.0, 1.0), float(z["speed"])
    every = int(z["act_every"]) if "act_every" in z.files else 1
    assert every in (1, k)
    dec = z["decided"].astype(bool) if "decided" in z.files else np.ones(ttl.shape, bool)
    live = ttl > 0
    flight = ttl[:-1] > 1                                      # the same ball is still coming down on the next tick
    cut = np.abs((d[:-1] - d[1:]) - dx[:-1] * speed) > 1e-6    # the landing point has not moved, so d changes by the paddle's travel
    m = dec[k:] & live[k:]
    out = dict(wall=float((cut & flight).sum()) / max(int(flight.sum()), 1), size=float(np.abs(dx[dec & live]).mean()),
               same=float((dx[k:][m] * dx[:-k][m] > 0).mean()),
               same1=float((dx[1:][live[1:]] * dx[:-1][live[1:]] > 0).mean()) if every == 1 else np.nan)
    at = dec & (ttl > k) & (ttl <= hi)
    for name, g in (("toward_near", (need > 0) & (need <= near)), ("toward_far", need > near)):
        out[name] = float((dx[at & g] * d[at & g] > 0).mean()) if (at & g).any() else np.nan
    if "cand_copy" in z.files:
        for tag, sel in (("", dec & live), ("_held", ~dec & live)):
            for key in ("cand_copy", "cand_distinct", "cand_sd"):
                out[key + tag] = float(np.nanmean(z[key][sel])) if sel.any() else np.nan
    return out


def endgame(z, at=4, near=8.0):
    """Per ball that came down, the paddle `at` ticks before the landing and the outcome (task 00131, reading 5: a
    decision every k ticks cannot correct late): the share of balls with the paddle under the landing point then, and
    the shares returned of the balls it was under, within `near` px of, and further from."""
    ttl, need = z["ttl"], z["need"]
    t0, b0 = np.nonzero(ttl[:len(ttl) - at + 1] == at)
    assert (ttl[t0 + at - 1, b0] == 1).all()
    hit, n0 = z["hit"][t0 + at - 1, b0].astype(bool), need[t0, b0]      # the tick whose move decides it is the one with 1 to go
    part = lambda m: float(hit[m].mean()) if m.any() else np.nan       # noqa: E731
    return dict(balls=len(t0), under=float((n0 == 0).mean()), ret_under=part(n0 == 0), ret_near=part((n0 > 0) & (n0 <= near)),
                ret_far=part(n0 > near), near=float(((n0 > 0) & (n0 <= near)).mean()))


def read(paths):
    Z = [np.load(p) for p in paths]
    S = Z[0]["conc"].shape[2]
    cat = lambda k: np.concatenate([z[k].reshape(-1, *z[k].shape[2:]) for z in Z])   # noqa: E731  [ticks x bodies, ...]
    ttl, need, d, dx, won, quarter = (cat(k) for k in ("ttl", "need", "d", "dx", "won", "quarter"))
    conc, pairs, clear, spread, pick, best, made = (cat(k) for k in ("conc", "pairs", "clear", "spread", "pick", "best", "made"))
    hits, misses = int(sum(z["hit"].sum() for z in Z)), int(sum(z["miss"].sum() for z in Z))
    print(f"{len(Z)} recordings ({Z[0]['run']}, step {Z[0]['step']}, {'full windows' if int(Z[0]['full']) else 'as lived'}"
          f"{', each pick the best dx of an even grid' if 'grid_act' in Z[0].files and int(Z[0]['grid_act']) else ''}): "
          f"{hits} hits, {misses} misses, hit share {hits / max(hits + misses, 1):.3f}")
    live = ttl > 0
    # task 00131: with action repeat the choice is made on the deciding ticks only; the held ticks' picks are not made
    dec = np.concatenate([(z["decided"] if "decided" in z.files else np.ones(z["ttl"].shape, bool)).reshape(-1) for z in Z]).astype(bool)
    if not dec.all():
        print(f"action repeat: {int((dec & live).sum())} of {int(live.sum())} ticks with a ball in play decide; the move toward the ball, "
              f"the label's order and the px lost are read on those")
    held = all("conc_hold" in z.files for z in Z)              # scored also by where a held move leaves the paddle
    HK = (("order", "conc_hold"), ("order_clear", "clear_hold"), ("order_grid", "grid_hold"), ("order_grid_clear", "grid_clear_hold"),
          ("r_need", "r_need_hold"))
    if held:
        H = {k: cat(k) for k in [k for _, k in HK] + ["best_hold", "made_hold"]}
    rows_h = {k: [] for k in [k for k, _ in HK] + ["regret"]}
    bin_of = np.digitize(ttl, BINS) - 1
    mine = np.take_along_axis(conc, np.maximum(won, 0)[:, None], 1)[:, 0]    # the label ordering of the piece whose move was made
    print("\nby ticks to landing     " + "".join(f"{n:>8s}" for n in BIN_NAMES))
    only, grid, grid_clear, grid_toward, r_need, r_dy, r_click = (cat(k) for k in ("only", "grid", "grid_clear", "grid_toward", "r_need", "r_dy", "r_click"))
    rows = {k: [] for k in ("ticks", "away", "closer", "toward", "order", "order_mover", "order_clear", "order_only", "order_grid", "order_grid_clear",
                            "grid_toward", "r_need", "r_dy", "r_click", "spread", "regret")}
    for i in range(len(BIN_NAMES)):
        m = live & (bin_of == i)
        act = m & (need > 0)
        rows["ticks"].append(m.sum())
        rows["away"].append(act.sum() / max(m.sum(), 1))                    # the paddle is not yet under the landing point
        rows["closer"].append((need[act] - made[act]).mean() if act.any() else np.nan)   # px nearer per tick (full speed 4.5)
        m, act = m & dec, act & dec                                          # from here on: the ticks that decide
        rows["toward"].append((dx[act] * d[act] > 0).mean() if act.any() else np.nan)
        rows["order"].append(np.nanmean(conc[m]) if m.any() else np.nan)
        rows["order_mover"].append(np.nanmean(mine[m]) if m.any() else np.nan)
        rows["order_clear"].append(np.nanmean(clear[m]) if m.any() and np.isfinite(clear[m]).any() else np.nan)
        for key, arr, sel in (("order_only", only, m), ("order_grid", grid, m), ("order_grid_clear", grid_clear, m), ("grid_toward", grid_toward, act),
                              ("r_need", r_need, m), ("r_dy", np.abs(r_dy), m), ("r_click", np.abs(r_click), m)):
            rows[key].append(np.nanmean(arr[sel]) if sel.any() and np.isfinite(arr[sel]).any() else np.nan)
        rows["spread"].append(np.nanmedian(spread[m]) if m.any() else np.nan)
        rows["regret"].append((made[act] - best[act].min(1)).mean() if act.any() else np.nan)
        if held:
            for key, k in HK:
                rows_h[key].append(np.nanmean(H[k][m]) if m.any() and np.isfinite(H[k][m]).any() else np.nan)
            rows_h["regret"].append((H["made_hold"][act] - H["best_hold"][act].min(1)).mean() if act.any() else np.nan)
    names = dict(ticks="ticks", away="paddle not yet under it", closer="  px nearer per tick", toward="move made goes toward",
                 order="label orders candidates", order_mover="  of the mover", order_clear="  pairs 4+ px apart",
                 order_only="  same dx, no dy or click", order_grid="label orders a grid of dx", order_grid_clear="  pairs 4+ px apart",
                 grid_toward="  its best dx goes toward", r_need="label's r with px left", r_dy="  abs r with dy", r_click="  abs r with click",
                 spread="label spread (median)",
                 regret="px lost to the best candidate")
    for k, name in names.items():
        fmt = "%8d" if k == "ticks" else ("%8.4f" if k == "spread" else "%8.2f")
        print(f"{name:24s}" + "".join(fmt % v for v in rows[k]))
    if held:
        print(f"\nscored by where the paddle is after the hold (the move made {int(Z[0]['act_every'])} ticks running, or up to the landing if sooner)")
        for k in rows_h:
            print(f"{names[k]:24s}" + "".join("%8.2f" % v for v in rows_h[k]))
    label = cat("label")
    print("\nthe label of the move made (lower is better), by px the paddle still has to travel")
    print("  ticks to landing      " + "".join(f"{n:>8s}" for n in FAR_NAMES))
    lab = []
    nb = np.digitize(need, FAR) - 1
    for name, m in (("1-4", live & dec & (ttl <= 4)), ("5-16", live & dec & (ttl > 4) & (ttl <= 16)), ("17+", live & dec & (ttl > 16))):
        lab.append([float(label[m & (nb == i)].mean()) if (m & (nb == i)).any() else None for i in range(len(FAR_NAMES))])
        print(f"  {name:22s}" + "".join(("%8.3f" % v) if v is not None else "       -" for v in lab[-1]))
    print("\nwho: the transformer whose move is made, by the quarter the ball is in (0 top left, 1 top right, 2 bottom left, 3 bottom right)")
    who = np.zeros((4, S))
    for q in range(4):
        m = live & (quarter == q) & (won >= 0)
        who[q] = np.bincount(won[m], minlength=S)[:S] / max(m.sum(), 1)
        print(f"  ball in {q}: " + "  ".join(f"p{k} {who[q, k]:.2f}" for k in range(S)))
    near = live & dec & (ttl <= 16)
    own = np.array([np.nanmean(conc[near & (quarter == k), k]) if (near & (quarter == k)).any() else np.nan for k in range(S)])
    other = np.array([np.nanmean(conc[near & (quarter != k), k]) if (near & (quarter != k)).any() else np.nan for k in range(S)])
    print("  label orders candidates in the last 16 ticks, ball in the transformer's own quarter: "
          + "  ".join(f"p{k} {own[k]:.2f}" for k in range(S)) + ";  elsewhere: " + "  ".join(f"p{k} {other[k]:.2f}" for k in range(S)))
    per = []                                                  # by recording: does the label's order go with the play?
    for z in Z:
        t, nd = z["ttl"], z["need"]
        m = (t > 0) & (t <= 16) & (z["decided"].astype(bool) if "decided" in z.files else True)
        act_ = m & (nd > 0)
        h, x = float(z["hit"].sum()), float((z["ttl"] == 1).sum())
        per.append((int(z["seed"]), h / max(x, 1), float((z["dx"][act_] * z["d"][act_] > 0).mean()), float(np.nanmean(z["conc"][m])),
                    float(np.nanmean(z["grid_toward"][act_]))) + ((float(np.nanmean(z["conc_hold"][m])),) if held else ()))
    per = np.array(per)
    if len(per) > 2:
        r = [float(np.corrcoef(per[:, 1], per[:, k])[0, 1]) for k in range(2, per.shape[1])]
        print("\nby recording, the last 16 ticks (seed: returned, move toward, label's order, the grid's best dx toward"
              + (", label's order after the hold" if held else "") + "):")
        print("  " + "  ".join(("%d:" + " %.2f" * (per.shape[1] - 1)) % tuple(x) for x in per))
        print(f"  across the recordings, returned goes with: move toward r {r[0]:+.2f}, label's order r {r[1]:+.2f}, the grid's best dx r {r[2]:+.2f}"
              + (f", label's order after the hold r {r[3]:+.2f}" if held else ""))
    B = np.concatenate([balls(z) for z in Z])
    hit, far, gross, toward, left = B[:, 0] > 0, B[:, 1], B[:, 2], B[:, 3], B[:, 4]
    print(f"\n{len(B)} balls read over their last {LAST} ticks: returned {hit.mean():.3f}")
    print("by px the paddle had to travel  " + "".join(f"{n:>8s}" for n in FAR_NAMES))
    fb = np.digitize(far, FAR) - 1
    print("  balls                         " + "".join("%8d" % (fb == i).sum() for i in range(len(FAR_NAMES))))
    print("  returned                      " + "".join(("%8.2f" % hit[fb == i].mean()) if (fb == i).any() else "       -" for i in range(len(FAR_NAMES))))
    ms = ~hit
    wrong = ms & (toward < 0.5)
    slow = ms & ~wrong & (gross < far)
    dither = ms & ~wrong & ~slow
    n = max(ms.sum(), 1)
    print(f"the {int(ms.sum())} misses: wrong way (under half of the moves toward the ball) {wrong.sum() / n:.2f}; too little travel "
          f"(toward, but the moves add up to less than the gap) {slow.sum() / n:.2f}; enough travel, not to the ball {dither.sum() / n:.2f}; "
          f"missed by under 2 px {(ms & (left < 2)).sum() / n:.2f}")
    print(f"  the paddle had to travel {far[hit].mean():.1f} px before a return, {far[ms].mean():.1f} px before a miss; "
          f"it travelled {gross[hit].mean():.1f} and {gross[ms].mean():.1f} px; moves toward the ball {np.nanmean(toward[hit]):.2f} and {np.nanmean(toward[ms]):.2f}")
    out = dict(mode="autopsy", run=str(Z[0]["run"]), step=str(Z[0]["step"]), full=int(Z[0]["full"]), recordings=len(Z), hits=hits, misses=misses,
               bins=BIN_NAMES, **{k: [float(v) for v in rows[k]] for k in rows}, who=who.tolist(), own=own.tolist(), other=other.tolist(),
               balls=len(B), far=FAR_NAMES, far_n=[int((fb == i).sum()) for i in range(len(FAR_NAMES))],
               far_hit=[float(hit[fb == i].mean()) if (fb == i).any() else None for i in range(len(FAR_NAMES))],
               wrong=float(wrong.sum() / n), slow=float(slow.sum() / n), dither=float(dither.sum() / n), label_by_need=lab,
               **({"hold": {k: [float(v) for v in rows_h[k]] for k in rows_h}} if held else {}))
    print(json.dumps(out))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("ring").add_argument("dir")
    r = sub.add_parser("record")
    r.add_argument("out")
    r.add_argument("run")
    r.add_argument("--step", type=int, default=None)
    r.add_argument("--full", action="store_true")
    r.add_argument("--grid", action="store_true")
    r.add_argument("--ticks", type=int, default=1500)
    r.add_argument("--bodies", type=int, default=16)
    r.add_argument("--seed", type=int, default=1)
    sub.add_parser("read").add_argument("paths", nargs="+")
    a = ap.parse_args()
    if a.cmd == "ring":
        ring(a.dir)
    elif a.cmd == "record":
        record(a.out, a.run, "final" if a.step is None else a.step, a.full, a.ticks, a.bodies, a.seed, a.grid)
    else:
        read(a.paths)


if __name__ == "__main__":
    main()

"""Why the trained agent misses in Breakout (task 00129, vm-00116).

Subcommands:
    ring DIR   — the ring control: DIR/{ring,full}/s1..12, lifetime.py with learning off on BD's (vm-00103) 60k weights
                 in new courts (seed 100 + s), acting on the ring as it lived and on full windows (online.ring = 0).
                 Prints the hit share by mode and the paired difference full - ring (mean, standard error, 95% interval
                 by t, seeds up and down) over the whole test, over its first 2000 ticks and over the rest. The first
                 2000 ticks repeat the earlier tests of the same weights in the same courts (vm-00103's frozen60k,
                 vm-00114's check), where the difference was first noticed, so the reading is the rest: the new ticks.
                 Equal counts in the first 2000 ticks mean the build did not change the play.
    record OUT.npz RUN [--step N] [--full] [--ticks 1500] [--bodies 16] [--seed 1]
               — the miss autopsy's recording: RUN's frozen pieces (the checkpoint saved at tick N; the final one by
                 default) play `bodies` new courts for `ticks` ticks, acting as the run lived, or on full windows with
                 --full. Each tick, from the court's state (for scoring only; nothing of it reaches the model): where
                 the ball will land and in how many ticks, and how far the paddle would still have to travel to be
                 under it; from the agent: each transformer's candidates and their labels, each transformer's pick, the
                 move made and whose it was.
    read OUT.npz [OUT.npz ...]
               — the autopsy's tables over the recordings (one per seed): by ticks to landing, how often the move made
                 goes toward the ball and how well each transformer's label orders its candidates; who makes the move;
                 the hit share by how far the paddle had to go in the last 12 ticks; and each miss's kind.
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


def need_after(w, x_land, dx):
    """After a move dx ([N] or [N, M], in the world's units): how far the paddle would still have to travel to be under
    the landing ball (0: it would be returned), and the signed gap between the ball's and the paddle's centres."""
    c = w.config
    dx = np.asarray(dx, np.float64)
    shape = (-1,) + (1,) * (dx.ndim - 1)
    pad = np.clip(w.paddle.reshape(shape) + np.clip(dx, -1.0, 1.0) * c.paddle_speed, w.lo, w.hi - c.paddle_w)
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


def record(out, run, step, full, ticks, bodies, seed):
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
    keys = ("ttl", "d", "need", "dx", "won", "quarter", "vx", "conc", "pairs", "clear", "spread", "pick", "best", "made", "label",
            "only", "grid", "grid_clear", "grid_toward", "r_need", "r_dy", "r_click", "hit", "miss")
    rec = {k: [] for k in keys}
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
            made = need_after(env, xs, a[:, 0])[0]
            obs, _, done, inf = env.step(a)
            life.es = done.astype(np.uint8)
            for k, v in zip(keys, (ttl, d0, need0, a[:, 0].astype(np.float64), won[:, 0], quarter, vx, conc, pairs, clear, spread, pick, best,
                                   made, np.asarray(life.vh, np.float64), only, grid, grid_clear, grid_toward, r_need, r_dy, r_click,
                                   inf["gt_paddle_hit"], inf["gt_hazard_click"])):
                rec[k].append(np.array(v))
    finally:
        life.close()
        lifetime.pieces_move = orig
    np.savez_compressed(out, **{k: np.stack(v) for k, v in rec.items()}, speed=c.paddle_speed, half=(c.paddle_w + c.ball) / 2.0,
                        run=os.path.basename(run.rstrip("/")), step=str(step), full=int(full), seed=seed)
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


def read(paths):
    Z = [np.load(p) for p in paths]
    S = Z[0]["conc"].shape[2]
    cat = lambda k: np.concatenate([z[k].reshape(-1, *z[k].shape[2:]) for z in Z])   # noqa: E731  [ticks x bodies, ...]
    ttl, need, d, dx, won, quarter = (cat(k) for k in ("ttl", "need", "d", "dx", "won", "quarter"))
    conc, pairs, clear, spread, pick, best, made = (cat(k) for k in ("conc", "pairs", "clear", "spread", "pick", "best", "made"))
    hits, misses = int(sum(z["hit"].sum() for z in Z)), int(sum(z["miss"].sum() for z in Z))
    print(f"{len(Z)} recordings ({Z[0]['run']}, step {Z[0]['step']}, {'full windows' if int(Z[0]['full']) else 'as lived'}): "
          f"{hits} hits, {misses} misses, hit share {hits / max(hits + misses, 1):.3f}")
    live = ttl > 0
    bin_of = np.digitize(ttl, BINS) - 1
    mine = np.take_along_axis(conc, np.maximum(won, 0)[:, None], 1)[:, 0]    # the label ordering of the piece whose move was made
    print("\nby ticks to landing     " + "".join(f"{n:>8s}" for n in BIN_NAMES))
    only, grid, grid_clear, grid_toward, r_need, r_dy, r_click = (cat(k) for k in ("only", "grid", "grid_clear", "grid_toward", "r_need", "r_dy", "r_click"))
    rows = {k: [] for k in ("ticks", "away", "toward", "closer", "order", "order_mover", "order_clear", "order_only", "order_grid", "order_grid_clear",
                            "grid_toward", "r_need", "r_dy", "r_click", "spread", "regret")}
    for i in range(len(BIN_NAMES)):
        m = live & (bin_of == i)
        act = m & (need > 0)
        rows["ticks"].append(m.sum())
        rows["away"].append(act.sum() / max(m.sum(), 1))                    # the paddle is not yet under the landing point
        rows["toward"].append((dx[act] * d[act] > 0).mean() if act.any() else np.nan)
        rows["closer"].append((need[act] - made[act]).mean() if act.any() else np.nan)   # px nearer per tick (full speed 4.5)
        rows["order"].append(np.nanmean(conc[m]) if m.any() else np.nan)
        rows["order_mover"].append(np.nanmean(mine[m]) if m.any() else np.nan)
        rows["order_clear"].append(np.nanmean(clear[m]) if m.any() and np.isfinite(clear[m]).any() else np.nan)
        for key, arr, sel in (("order_only", only, m), ("order_grid", grid, m), ("order_grid_clear", grid_clear, m), ("grid_toward", grid_toward, act),
                              ("r_need", r_need, m), ("r_dy", np.abs(r_dy), m), ("r_click", np.abs(r_click), m)):
            rows[key].append(np.nanmean(arr[sel]) if sel.any() and np.isfinite(arr[sel]).any() else np.nan)
        rows["spread"].append(np.nanmedian(spread[m]) if m.any() else np.nan)
        rows["regret"].append((made[act] - best[act].min(1)).mean() if act.any() else np.nan)
    names = dict(ticks="ticks", away="paddle not yet under it", toward="  move made goes toward", closer="  px nearer per tick",
                 order="label orders candidates", order_mover="  of the mover", order_clear="  pairs 4+ px apart",
                 order_only="  same dx, no dy or click", order_grid="label orders a grid of dx", order_grid_clear="  pairs 4+ px apart",
                 grid_toward="  its best dx goes toward", r_need="label's r with px left", r_dy="  abs r with dy", r_click="  abs r with click",
                 spread="label spread (median)",
                 regret="px lost to the best candidate")
    for k, name in names.items():
        fmt = "%8d" if k == "ticks" else ("%8.4f" if k == "spread" else "%8.2f")
        print(f"{name:24s}" + "".join(fmt % v for v in rows[k]))
    label = cat("label")
    print("\nthe label of the move made (lower is better), by px the paddle still has to travel")
    print("  ticks to landing      " + "".join(f"{n:>8s}" for n in FAR_NAMES))
    lab = []
    nb = np.digitize(need, FAR) - 1
    for name, m in (("1-4", live & (ttl <= 4)), ("5-16", live & (ttl > 4) & (ttl <= 16)), ("17+", live & (ttl > 16))):
        lab.append([float(label[m & (nb == i)].mean()) if (m & (nb == i)).any() else None for i in range(len(FAR_NAMES))])
        print(f"  {name:22s}" + "".join(("%8.3f" % v) if v is not None else "       -" for v in lab[-1]))
    print("\nwho: the transformer whose move is made, by the quarter the ball is in (0 top left, 1 top right, 2 bottom left, 3 bottom right)")
    who = np.zeros((4, S))
    for q in range(4):
        m = live & (quarter == q) & (won >= 0)
        who[q] = np.bincount(won[m], minlength=S)[:S] / max(m.sum(), 1)
        print(f"  ball in {q}: " + "  ".join(f"p{k} {who[q, k]:.2f}" for k in range(S)))
    near = live & (ttl <= 16)
    own = np.array([np.nanmean(conc[near & (quarter == k), k]) if (near & (quarter == k)).any() else np.nan for k in range(S)])
    other = np.array([np.nanmean(conc[near & (quarter != k), k]) if (near & (quarter != k)).any() else np.nan for k in range(S)])
    print("  label orders candidates in the last 16 ticks, ball in the transformer's own quarter: "
          + "  ".join(f"p{k} {own[k]:.2f}" for k in range(S)) + ";  elsewhere: " + "  ".join(f"p{k} {other[k]:.2f}" for k in range(S)))
    per = []                                                  # by recording: does the label's order go with the play?
    for z in Z:
        t, nd = z["ttl"], z["need"]
        m = (t > 0) & (t <= 16)
        act_ = m & (nd > 0)
        h, x = float(z["hit"].sum()), float((z["ttl"] == 1).sum())
        per.append((int(z["seed"]), h / max(x, 1), float((z["dx"][act_] * z["d"][act_] > 0).mean()), float(np.nanmean(z["conc"][m])),
                    float(np.nanmean(z["grid_toward"][act_]))))
    per = np.array(per)
    if len(per) > 2:
        r = [float(np.corrcoef(per[:, 1], per[:, k])[0, 1]) for k in (2, 3, 4)]
        print("\nby recording, the last 16 ticks (seed: returned, move toward, label's order, the grid's best dx toward):")
        print("  " + "  ".join("%d: %.2f %.2f %.2f %.2f" % tuple(x) for x in per))
        print(f"  across the recordings, returned goes with: move toward r {r[0]:+.2f}, label's order r {r[1]:+.2f}, the grid's best dx r {r[2]:+.2f}")
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
               wrong=float(wrong.sum() / n), slow=float(slow.sum() / n), dither=float(dither.sum() / n), label_by_need=lab)
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
    r.add_argument("--ticks", type=int, default=1500)
    r.add_argument("--bodies", type=int, default=16)
    r.add_argument("--seed", type=int, default=1)
    sub.add_parser("read").add_argument("paths", nargs="+")
    a = ap.parse_args()
    if a.cmd == "ring":
        ring(a.dir)
    elif a.cmd == "record":
        record(a.out, a.run, "final" if a.step is None else a.step, a.full, a.ticks, a.bodies, a.seed)
    else:
        read(a.paths)


if __name__ == "__main__":
    main()

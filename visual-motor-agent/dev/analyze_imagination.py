"""How good is the latent imagination? Imagined multi-step rollouts vs what really happens.

From the start states of the evaluation set, random action sequences (dx, dy ~ N(0, 0.607), click p 0.1) are
both imagined (vm_imagine: latent transition + hunger head, predicted hunger fed back) and executed in the world.
Scored per imagined step k (episodes that already ended are excluded):
  cursor_err / cursor_err_static
      Cursor position (px) read from the latent by a linear (ridge) probe fit on real frames' latents, vs the
      true position; "static" assumes the cursor never moves; "probe" applies the probe to the real latent at k
      (the probe's own error floor).
  target_err      the circle centre read the same way (the circle does not move); target_err_probe from the
                  real latent (the probe's floor).
  hunger_mae      |predicted - true hunger|.
  relief_*        At steps where a click inside really relieved hunger: the fraction whose imagined hunger falls
                  below half the true hunger before the step, and the mean relative drop. false_relief_frac: the
                  same criterion at all other steps. Steps with true hunger < 0.1 are not scored.
  cursor_err_nn   Cursor read by retrieval instead: the nearest latent among frames of the same circle with the
                  cursor on a 2-px grid. The linear probe can misread latents that leave the manifold of real
                  frames; retrieval cannot. cursor_err_nn_real: the same readout of the real latent (its floor).
Approach test: from the same start states, a scripted path (evaluation only: it uses the true positions) moves
straight to the circle at full speed and clicks once well inside. approach_* report, by the number of moves
before the click, whether the imagination shows the relief at the click, and false alarms on the path before it.
Edge test: the cursor is placed at a signed distance g from the circle's edge (g < 0 inside), hunger 0.4, and one
click is imagined. edge_relief = 1 - imagined hunger / 0.405 per g (truth: 1 inside, 0 outside). edge_cost: the
summed imagined hunger over 12 steps after that click (what the planner scores; doing nothing costs about 4.9).
Trap test (g = +2, +5 px outside): fraction of states where the plan "one move toward the centre, then click" costs
less than "click here" (trap_move_in).
Straight paths: 12 identical moves (8 directions, full and half speed, no click); straight_ratio = imagined
displacement along the true one / true displacement, per step (1 = no shrinkage).
Over the circle: one imagined random move with the cursor on the background vs inside the circle (vm-00001's
movement test states); move_err_* = retrieval error of the imagined cursor, move_corr_* = correlation of imagined
and true displacement (x, y).
Writes RUN_DIR/analysis/<ckpt>/imagination.json and imagination.png (curves + decoded imagined frames).

Usage:
    ./venv/bin/python analyze_imagination.py RUN_DIR [--ckpt ckpt_final.bin] [--n 256] [--horizon 12]
"""
import argparse
import dataclasses
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import agent_lib  # noqa: E402
from plan import EVAL_SEED  # noqa: E402
from record_episodes import env_config_from_info  # noqa: E402
from world import CircleWorld  # noqa: E402
from world.circle_world import render_state  # noqa: E402


def random_actions(rng, shape, click_p):
    return np.concatenate([rng.normal(0, 0.607, shape + (2,)), (rng.random(shape + (1,)) < click_p)], -1)


def fit_probe(agent, cfg, n_envs=128, steps=100, lam=1.0, seed=4242):
    """Ridge probe latent -> (cursor x, y, circle x, y), fit on real frames of random play (auto-reset on)."""
    env = CircleWorld(n_envs, dataclasses.replace(cfg, auto_reset=True), seed=seed)
    rng = np.random.default_rng(seed)
    X, Y = [], []
    obs = env.observe()
    for _ in range(steps):
        st = env.get_state()
        X.append(agent.encode(obs["image"])["vf"])
        Y.append(np.concatenate([st["cursor"], st["target"]], 1))
        obs, _, _, _ = env.step(random_actions(rng, (n_envs,), 0.1))
    X, Y = np.concatenate(X).astype(np.float64), np.concatenate(Y)
    mu, sd = X.mean(0), X.std(0) + 1e-6
    Xs = np.concatenate([(X - mu) / sd, np.ones((len(X), 1))], 1)
    Wt = np.linalg.solve(Xs.T @ Xs + lam * np.eye(Xs.shape[1]), Xs.T @ Y)
    return lambda Z: np.concatenate([(np.asarray(Z, np.float64) - mu) / sd, np.ones((len(Z), 1))], 1) @ Wt


def retrieval_bank(agent, cfg, st, n, grid=2):
    """Latents of each env's circle with the cursor at every grid point: [n, P, D] and the positions [P, 2]."""
    xs = np.arange(0, cfg.size, grid) + 0.5
    pos = np.stack(np.meshgrid(xs, xs), -1).reshape(-1, 2)
    P = len(pos)
    bank = np.stack([agent.encode(render_state(pos, np.repeat(st["target"][e:e + 1], P, 0),
                                               np.repeat(st["radius"][e], P), np.ones(P, bool), cfg))["vf"]
                     for e in range(n)])
    return bank, pos


def retrieve(bank, pos, z):
    """Cursor positions [m, 2] of the bank entries nearest to latents z [m, D] (bank [m, P, D])."""
    d = (bank * bank).sum(-1) - 2 * np.einsum("mpd,md->mp", bank, z)
    return pos[d.argmin(1)]


def relief_scores(pred, h_before, relief, valid, min_h=0.1):
    """Imagined relief at clicks that really relieved hunger, and false alarms at all other steps. A step shows
    relief when its imagined hunger is below half the true hunger before it. Only steps with true hunger
    >= min_h count: below that, relief and no relief look alike."""
    big = valid & (h_before >= min_h)
    ev, non = big & relief, big & ~relief
    det = pred < 0.5 * h_before
    nan = float("nan")
    return dict(events=int(ev.sum()), detected=float(det[ev].mean()) if ev.any() else nan,
                drop=float((1 - pred[ev] / h_before[ev]).mean()) if ev.any() else nan,
                false_alarm=float(det[non].mean()) if non.any() else nan)


def approach_test(agent, cfg, n=256, max_moves=14, depth=0.5, bank=None, pos=None):
    """Scripted approach: full-speed straight moves to the circle centre, click once within depth * radius.
    With a retrieval bank (first len(bank) envs), also the imagined cursor's error at the click."""
    env = CircleWorld(n, dataclasses.replace(cfg, auto_reset=False), seed=EVAL_SEED)
    obs = env.observe()
    s0, h0 = agent.encode(obs["image"])["vf"], obs["drives"][:, 0].copy()
    T = max_moves + 1
    acts = np.zeros((n, T, 3), np.float32)
    k_click = np.full(n, -1)
    live = np.ones(n, bool)
    hun = np.zeros((n, T))
    cur = np.zeros((n, T, 2))
    for k in range(T):
        st = env.get_state()
        cur[:, k] = st["cursor"]
        d = st["target"] - st["cursor"]
        inside = np.hypot(d[:, 0], d[:, 1]) <= depth * st["radius"]
        a = np.zeros((n, 3), np.float32)
        a[:, :2] = d / np.maximum(np.abs(d).max(1, keepdims=True), 4.0)      # at most 4 px per axis
        a[inside, :2] = 0
        a[inside, 2] = 1
        k_click[live & inside] = k
        acts[live, k] = a[live]
        _, _, done, inf = env.step(np.where(live[:, None], a, 0))
        hun[:, k] = inf["gt_drives"][:, 0]
        live &= ~done
    ok = k_click >= 0
    if not ok.any():
        return dict(approach_episodes=0)
    im = agent.imagine(s0[ok], h0[ok], acts[ok], latents=bank is not None)
    ph = im["hunger"]
    kc = k_click[ok]
    hb = np.concatenate([h0[ok, None], hun[ok, :-1]], 1)            # true hunger before each step
    t = np.arange(T)[None]
    click = t == kc[:, None]
    res = dict(approach_episodes=int(ok.sum()), approach_moves=[], approach_n=[], approach_relief_detected=[],
               approach_relief_drop=[], approach_cursor_err=[])
    err = np.full(len(kc), np.nan)                 # imagined cursor (retrieval) vs true, where the click happens
    if bank is not None:
        e = np.flatnonzero(ok)
        sel = (e < len(bank)) & (kc > 0)
        i = np.flatnonzero(sel)
        z = im["latents"][i, kc[i] - 1]            # latent after the last move = state at the click
        err[i] = np.linalg.norm(retrieve(bank[e[i]], pos, z) - cur[e[i], kc[i]], axis=1)
    for m in range(T):
        sel = kc == m
        r = relief_scores(ph[sel], hb[sel], click[sel], np.broadcast_to(t <= m, (int(sel.sum()), T)))
        if r["events"] >= 3:
            res["approach_moves"].append(m)
            res["approach_n"].append(r["events"])
            res["approach_relief_detected"].append(r["detected"])
            res["approach_relief_drop"].append(r["drop"])
            res["approach_cursor_err"].append(float(np.nanmean(err[sel])) if np.isfinite(err[sel]).any()
                                              else float("nan"))
    r = relief_scores(ph, hb, click, t <= kc[:, None])
    res["approach_relief_detected_all"] = r["detected"]
    res["approach_false_alarm"] = r["false_alarm"]
    return res


EDGE_G = (-8, -5, -3, -1, 1, 2, 3, 5, 8, 12)


def edge_test(agent, cfg, n=256, h0=0.4, seed=7):
    """Imagined relief of one click with the cursor at signed distance g from the circle's edge."""
    env = CircleWorld(n, dataclasses.replace(cfg, auto_reset=False), seed=EVAL_SEED)
    st = env.get_state()
    rng = np.random.default_rng(seed)
    res = dict(edge_g=[], edge_relief=[], edge_cost=[], edge_n=[], trap_g=[], trap_move_in=[])
    H = 12
    hh = np.full(n, h0, np.float32)
    for g in EDGE_G:
        ang = rng.uniform(0, 2 * np.pi, n)
        u = np.stack([np.cos(ang), np.sin(ang)], 1)
        cur = st["target"] + (st["radius"] + g)[:, None] * u
        ok = (cur >= 0.5).all(1) & (cur <= cfg.size - 0.5).all(1)       # cursor on screen
        s0 = agent.encode(render_state(cur, st["target"], st["radius"], np.ones(n, bool), cfg))["vf"]
        click = np.zeros((n, H, 3), np.float32)
        click[:, 0, 2] = 1
        h = agent.imagine(s0, hh, click)["hunger"]
        res["edge_g"].append(g)
        res["edge_relief"].append(float((1 - h[ok, 0] / (h0 + 0.005)).mean()))
        res["edge_cost"].append(float(h[ok].sum(1).mean()))
        res["edge_n"].append(int(ok.sum()))
        if g in (2, 5):
            move_in = np.zeros((n, H, 3), np.float32)
            move_in[:, 0, :2] = -u / np.abs(u).max(1, keepdims=True)    # full speed toward the centre
            move_in[:, 1, 2] = 1
            cost_in = agent.imagine(s0, hh, move_in)["hunger"].sum(1)
            res["trap_g"].append(g)
            res["trap_move_in"].append(float((cost_in[ok] < h[ok].sum(1)).mean()))
    return res


def straight_test(agent, cfg, n=128, H=12):
    """Imagined vs true displacement along straight paths at full and half speed (retrieval readout)."""
    env = CircleWorld(n, dataclasses.replace(cfg, auto_reset=False), seed=EVAL_SEED)
    obs, st0 = env.observe(), env.get_state()
    bank, pos = retrieval_bank(agent, cfg, st0, n)
    s0 = agent.encode(obs["image"])["vf"]
    res = {}
    for speed in (1.0, 0.5):
        true_l, proj = [], []
        for ang in np.arange(8) * np.pi / 4:
            d = np.array([np.cos(ang), np.sin(ang)])
            acts = np.zeros((n, H, 3), np.float32)
            acts[:, :, :2] = d / np.abs(d).max() * speed
            lat = agent.imagine(s0, obs["drives"][:, 0], acts, latents=True)["latents"]
            e2 = CircleWorld(n, dataclasses.replace(cfg, auto_reset=False), seed=EVAL_SEED)
            true = np.stack([e2.step(acts[:, k])[3]["gt_cursor"] for k in range(H)], 1) - st0["cursor"][:, None]
            imag = np.stack([retrieve(bank, pos, lat[:, k]) for k in range(H)], 1) - st0["cursor"][:, None]
            tl = np.linalg.norm(true, axis=-1)
            true_l.append(tl)
            proj.append((imag * true).sum(-1) / np.maximum(tl, 1e-6))
        tl, pr = np.concatenate(true_l), np.concatenate(proj)
        m = tl > 2
        res["straight_ratio_speed%g" % speed] = [float(pr[m[:, k], k].mean() / tl[m[:, k], k].mean()) for k in range(H)]
    return res


def over_circle_test(agent, cfg, n=128, seed=5):
    """One imagined random move, cursor on the background vs over the circle (retrieval readout)."""
    from analyze_dynamics import make_states
    rng = np.random.default_rng(seed)
    res = {}
    for name, inside in (("background", False), ("circle", True)):
        env = make_states(dataclasses.replace(cfg, auto_reset=False), n, 31337, inside)
        st, obs = env.get_state(), env.observe()
        bank, pos = retrieval_bank(agent, cfg, st, n)
        a = np.zeros((n, 1, 3), np.float32)
        a[:, 0, :2] = rng.uniform(-1, 1, (n, 2))
        z = agent.imagine(agent.encode(obs["image"])["vf"], obs["drives"][:, 0], a, latents=True)["latents"][:, 0]
        true = env.step(a[:, 0])[3]["gt_cursor"]
        im = retrieve(bank, pos, z)
        dt, di = true - st["cursor"], im - st["cursor"]
        res["move_err_" + name] = float(np.linalg.norm(im - true, axis=1).mean())
        res["move_err_static_" + name] = float(np.linalg.norm(dt, axis=1).mean())
        res["move_corr_" + name] = [float(np.corrcoef(dt[:, j], di[:, j])[0, 1]) for j in (0, 1)]
    return res


def analyze(agent, cfg, n=256, H=12, seed=1, n_nn=128):
    probe = fit_probe(agent, cfg)
    rng = np.random.default_rng(seed)
    env = CircleWorld(n, dataclasses.replace(cfg, auto_reset=False), seed=EVAL_SEED)
    obs = env.observe()
    st0 = env.get_state()
    h0 = obs["drives"][:, 0].copy()
    acts = random_actions(rng, (n, H), 0.1).astype(np.float32)
    s0 = agent.encode(obs["image"])["vf"]
    im = agent.imagine(s0, obs["drives"][:, 0], acts, latents=True, frames=True)
    cur, tgt, hun, alive, real_lat, relief = [], [], [], [], [], []
    live = np.ones(n, bool)
    frames = [obs["image"]]
    for k in range(H):
        obs, _, done, inf = env.step(acts[:, k])
        alive.append(live.copy())
        cur.append(inf["gt_cursor"]); tgt.append(inf["gt_target"]); hun.append(inf["gt_drives"][:, 0])
        relief.append(live & inf["success"])
        real_lat.append(agent.encode(inf["terminal_image"])["vf"])
        frames.append(inf["terminal_image"])
        live &= ~done
    res = dict(k=list(range(1, H + 1)), cursor_err=[], cursor_err_static=[], cursor_err_probe=[], target_err=[],
               target_err_probe=[], hunger_mae=[], alive_frac=[], cursor_err_nn=[], cursor_err_nn_real=[],
               cursor_err_nn_static=[])
    n_nn = min(n_nn, n)
    bank, pos = retrieval_bank(agent, cfg, st0, n_nn)
    for k in range(H):
        m = alive[k]
        pi = probe(im["latents"][m, k])
        pr = probe(real_lat[k][m])
        res["cursor_err"].append(float(np.linalg.norm(pi[:, :2] - cur[k][m], axis=1).mean()))
        res["cursor_err_probe"].append(float(np.linalg.norm(pr[:, :2] - cur[k][m], axis=1).mean()))
        res["cursor_err_static"].append(float(np.linalg.norm(st0["cursor"][m] - cur[k][m], axis=1).mean()))
        res["target_err"].append(float(np.linalg.norm(pi[:, 2:] - tgt[k][m], axis=1).mean()))
        res["target_err_probe"].append(float(np.linalg.norm(pr[:, 2:] - tgt[k][m], axis=1).mean()))
        res["hunger_mae"].append(float(np.abs(im["hunger"][m, k] - hun[k][m]).mean()))
        res["alive_frac"].append(float(m.mean()))
        mn = m[:n_nn]
        true = cur[k][:n_nn][mn]
        for key, z in (("cursor_err_nn", im["latents"][:n_nn, k]), ("cursor_err_nn_real", real_lat[k][:n_nn])):
            res[key].append(float(np.linalg.norm(retrieve(bank[mn], pos, z[mn]) - true, axis=1).mean()))
        res["cursor_err_nn_static"].append(float(np.linalg.norm(st0["cursor"][:n_nn][mn] - true, axis=1).mean()))
    hb = np.stack([h0] + hun[:-1])                               # [H, n] true hunger before each step
    r = relief_scores(im["hunger"].T, hb, np.array(relief), np.array(alive))
    res.update(relief_events=r["events"], relief_detected_frac=r["detected"], relief_pred_drop=r["drop"],
               false_relief_frac=r["false_alarm"])
    res.update(approach_test(agent, cfg, n, bank=bank, pos=pos))
    res.update(edge_test(agent, cfg, n))
    res.update(straight_test(agent, cfg))
    res.update(over_circle_test(agent, cfg))
    return res, im, np.stack(frames, 1), alive


def plot(res, im, frames, path, title, n_show=5):
    fig = plt.figure(figsize=(14, 7.5))
    ax = fig.add_subplot(2, 3, 1)
    ax.plot(res["k"], res["cursor_err"], "o-", label="imagined latent")
    ax.plot(res["k"], res["cursor_err_static"], "s--", label="no-motion baseline")
    ax.plot(res["k"], res["cursor_err_probe"], "^:", label="real latent (probe floor)")
    ax.plot(res["k"], res["cursor_err_nn"], "o-", c="C4", label="imagined latent, retrieval")
    ax.plot(res["k"], res["cursor_err_nn_real"], "^:", c="C5", label="real latent, retrieval")
    ax.set_xlabel("imagined step k")
    ax.set_ylabel("cursor position error (px)")
    ax.set_title("Cursor, read from the latent", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    ax = fig.add_subplot(2, 3, 2)
    if res.get("approach_moves"):
        ax.plot(res["approach_moves"], res["approach_relief_detected"], "o-", c="C3")
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("moves before the click")
    ax.set_ylabel("relief imagined at the click")
    ax.set_title("Approach and click: false alarms on the way %.0f%%"
                 % (100 * np.nan_to_num(res.get("approach_false_alarm", np.nan))), fontsize=10)
    ax.grid(alpha=0.3)
    ax = fig.add_subplot(2, 3, 3)
    ax.plot(res["k"], res["hunger_mae"], "o-", c="C2")
    ax.set_xlabel("imagined step k")
    ax.set_ylabel("|predicted - true hunger|")
    ax.set_title("Hunger: relief events %d, detected %.0f%%" % (res["relief_events"],
                                                               100 * np.nan_to_num(res["relief_detected_frac"])),
                 fontsize=10)
    ax.grid(alpha=0.3)
    H = frames.shape[1] - 1
    r = im["frames"].shape[2]
    f = frames.shape[2] // r
    for e in range(min(n_show, frames.shape[0])):
        real = [frames[e, k].reshape(r, f, r, f, 3).mean((1, 3)) / 255.0 for k in range(1, H + 1)]
        imag = [np.clip(im["frames"][e, k], 0, 1) for k in range(H)]
        strip = np.concatenate([np.concatenate(real, 1), np.concatenate(imag, 1)], 0)
        ax = fig.add_axes([0.05, 0.40 - 0.085 * e, 0.9, 0.08])
        ax.imshow(strip, interpolation="nearest")
        ax.set_axis_off()
        if e == 0:
            ax.set_title("top: real frames (coarse), bottom: decoded imagination, steps 1..%d" % H, fontsize=9)
    fig.suptitle(title, fontsize=11)
    fig.savefig(path, dpi=100)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir")
    ap.add_argument("--ckpt", default="ckpt_final.bin")
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--horizon", type=int, default=12)
    ap.add_argument("--n-nn", type=int, default=128, help="episodes scored by the retrieval readout")
    ap.add_argument("--threads", type=int, default=32)
    args = ap.parse_args()
    agent_lib.set_threads(args.threads)
    with open(os.path.join(args.run_dir, "info.json")) as f:
        info = json.load(f)
    cfg = dataclasses.replace(env_config_from_info(info), bonuses=False, drives=True)
    agent = agent_lib.Agent.load(os.path.join(args.run_dir, args.ckpt))
    res, im, frames, _ = analyze(agent, cfg, args.n, args.horizon, n_nn=args.n_nn)
    out = os.path.join(args.run_dir, "analysis", os.path.splitext(args.ckpt)[0])
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "imagination.json"), "w") as f:
        json.dump(res, f, indent=1)
    plot(res, im, frames, os.path.join(out, "imagination.png"),
         "%s  %s: imagined vs real, random actions" % (os.path.basename(args.run_dir.rstrip("/")), args.ckpt))
    print(json.dumps({k: (np.round(v, 2).tolist() if isinstance(v, list) else v) for k, v in res.items()}))


if __name__ == "__main__":
    main()

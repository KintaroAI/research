"""Did the model learn what its actions do? Counterfactual tests of the action-conditioned predictor.

For a set of held-out world states, the model predicts the next frame (and next drives) for each of
several candidate actions. The world, restored to the same state, produces the true outcome of each
action. Ground-truth geometry is used here only to *score* the predictions.

Metrics (JSON + figures in RUN_DIR/analysis/<ckpt>/):
  action_retrieval_acc   For each state, does the prediction for action i match the true outcome of
                         action i better than the outcomes of the other 8 moves? Chance = 1/9.
                         Requires the model to use the action input AND to know where its cursor is.
  outcome_explained_acc  For each true outcome j, is the prediction conditioned on action j closer
                         to it than the predictions conditioned on the other 8 moves? Chance = 1/9.
                         (Softer: correct-direction shifts count even if smaller than the truth.)
  outcome_margin_rel     Mean relative MSE advantage of the matching prediction over the others.
  disp_err / disp_err_static
                         Cursor displacement (px) read off predicted vs true frames, against a
                         predictor that assumes no motion. disp_corr = correlation of components.
  vanish_inside / vanish_outside
                         Fraction of the red circle that disappears in the predicted frame after a
                         click (relative to no click), with the cursor inside vs outside the
                         circle. Truth: 1.0 inside, 0.0 outside.
  hunger_drop_inside     Predicted next hunger without vs with a click while inside the circle.
  probe_r2_*             Ridge-regression R^2 of cursor / target position from z (held-out states).

Usage:
    ./venv/bin/python analyze_dynamics.py RUN_DIR [--ckpt ckpt_final.bin] [--states 256]
    ./venv/bin/python analyze_dynamics.py --untrained          # same tests on a random-init model
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
from world import CircleWorld, EnvConfig  # noqa: E402

MOVES = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1), (0, 0)]
MOVE_NAMES = ["right", "left", "down", "up", "down-right", "up-right", "down-left", "up-left", "stay"]


def downsample(img, f):
    """Same box filter as the C++ prediction target (rounding half up), returns float [0,1]."""
    H, W = img.shape[-3], img.shape[-2]
    x = img.astype(np.int64).reshape(img.shape[:-3] + (H // f, f, W // f, f, 3)).sum(axis=(-4, -2))
    return ((x + (f * f) // 2) // (f * f)).astype(np.float32) / 255.0


def cursor_centroid(frames):
    """Cursor position (in frame pixels) from the white crosshair: centroid of excess whiteness."""
    w = frames.min(axis=-1)                       # white is high in all channels; red/background are not
    w = np.clip(w - 0.3, 0, None) ** 2
    H, W = w.shape[-2:]
    tot = w.sum(axis=(-2, -1)) + 1e-9
    ys, xs = np.arange(H) + 0.5, np.arange(W) + 0.5
    cy = (w.sum(axis=-1) * ys).sum(axis=-1) / tot
    cx = (w.sum(axis=-2) * xs).sum(axis=-1) / tot
    return np.stack([cx, cy], axis=-1), tot


def redness(frames):
    """Soft count of target-red pixels (red high, green/blue low)."""
    r, g, b = frames[..., 0], frames[..., 1], frames[..., 2]
    return np.clip((r - np.maximum(g, b) - 0.2) / 0.5, 0, 1).sum(axis=(-2, -1))


def make_states(env_cfg, n, seed, inside):
    """n fresh episode-start states; if inside, the cursor is moved inside the circle."""
    env = CircleWorld(n, env_cfg, seed=seed)
    env.reset()
    if inside:
        rng = np.random.default_rng(seed + 1)
        st = env.get_state()
        ang = rng.uniform(0, 2 * np.pi, n)
        rad = st["radius"] * np.sqrt(rng.uniform(0, 1, n)) * 0.6
        cur = st["target"] + np.stack([np.cos(ang), np.sin(ang)], 1) * rad[:, None]
        env.set_state(dict(cursor=np.clip(cur, 0, env_cfg.size - 1e-4)))
    return env


def predict_actions(agent, obs, actions):
    """Predict next frame/drives for each candidate action from single-frame contexts.

    obs: {'image' [n,S,S,3], 'drives' [n,2]}; actions: [A,3]. Returns frames [n,A,p,p,3], drives [n,A,2], z [n,d].
    """
    n, A = obs["image"].shape[0], len(actions)
    img = np.repeat(obs["image"][:, None], A, axis=1).reshape(n * A, 1, *obs["image"].shape[1:])
    drv = np.repeat(obs["drives"][:, None], A, axis=1).reshape(n * A, 1, 2)
    zeros3 = np.zeros((n * A, 1, 3), np.float32)
    zeros1 = np.zeros((n * A, 1), np.float32)
    es = np.ones((n * A, 1), np.uint8)
    act = np.tile(np.asarray(actions, np.float32), (n, 1)).reshape(n * A, 1, 3)
    out = agent.forward(img, drv, zeros3, zeros1, es, pred_actions=act)
    p = out["pred_frames"].shape[-2]
    return (out["pred_frames"].reshape(n, A, p, p, 3), out["pred_drives"].reshape(n, A, 2),
            out["z"].reshape(n, A, -1)[:, 0])


def true_outcomes(env, actions):
    """True next frames/drives for each candidate action from the env's current state (restored each time)."""
    st = env.get_state()
    frames, drives = [], []
    n = env.num_envs
    for a in actions:
        env.set_state(st)
        _, _, _, inf = env.step(np.tile(np.asarray(a, np.float64), (n, 1)))
        frames.append(inf["terminal_image"])
        drives.append(inf["gt_drives"])
    env.set_state(st)
    return np.stack(frames, 1), np.stack(drives, 1)


def ridge_r2(X, Y, lam=1.0, split=0.75, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))
    k = int(split * len(X))
    tr, te = idx[:k], idx[k:]
    mu, sd = X[tr].mean(0), X[tr].std(0)
    keep = sd > 1e-4                                   # drop dead / constant features
    X, mu, sd = X[:, keep], mu[keep], np.maximum(sd[keep], 1e-2 * sd[keep].max())
    Xs = np.clip((X - mu) / sd, -10, 10)
    Xa = np.concatenate([Xs, np.ones((len(X), 1))], 1)
    A = Xa[tr].T @ Xa[tr] + lam * np.eye(Xa.shape[1])
    W = np.linalg.solve(A, Xa[tr].T @ Y[tr])
    P = Xa[te] @ W
    ss_res = ((P - Y[te]) ** 2).sum(0)
    ss_tot = ((Y[te] - Y[te].mean(0)) ** 2).sum(0)
    return 1 - ss_res / ss_tot


def probe(agent, env_cfg, n_envs=128, steps=64, seed=4242):
    """Collect z along random-policy trajectories and fit linear probes for cursor/target position."""
    env = CircleWorld(n_envs, dataclasses.replace(env_cfg, bonuses=False), seed=seed)
    obs = env.reset()
    rng = np.random.default_rng(seed)
    K = agent.config["ctx"]
    T = min(steps, K)
    imgs, drvs, pas, es, cur, tgt = [], [], [], [], [], []
    pa = np.zeros((n_envs, 3), np.float32)
    first = np.ones(n_envs, bool)
    cursor, target = env.cursor.copy(), env.target.copy()
    for t in range(T):
        imgs.append(obs["image"]); drvs.append(obs["drives"]); pas.append(pa.copy()); es.append(first.copy())
        cur.append(cursor.copy()); tgt.append(target.copy())
        a = np.concatenate([rng.normal(0, 0.6, (n_envs, 2)), (rng.random((n_envs, 1)) < 0.05)], 1)
        obs, _, done, inf = env.step(a)
        pa = executed_action(a.astype(np.float32))
        first = inf["first"].astype(bool)
        pa[first] = 0
        cursor, target = env.cursor.copy(), env.target.copy()
    st = lambda v: np.stack(v, 1)
    out = agent.forward(st(imgs), st(drvs), st(pas), np.zeros((n_envs, T), np.float32), st(es).astype(np.uint8))
    Z = out["z"].reshape(-1, out["z"].shape[-1]).astype(np.float64)
    Y = np.concatenate([st(cur).reshape(-1, 2), st(tgt).reshape(-1, 2)], 1)
    r2 = ridge_r2(Z, Y)
    res = dict(probe_r2_cursor_x=float(r2[0]), probe_r2_cursor_y=float(r2[1]),
               probe_r2_target_x=float(r2[2]), probe_r2_target_y=float(r2[3]))
    # the shared encoder (also used by the policy): conv-3 map and the 128-d visual feature
    enc = agent.encode(st(imgs).reshape(-1, *imgs[0].shape[1:]))
    for name, X in (("a3", enc["a3"]), ("vf", enc["vf"])):
        r2 = ridge_r2(X.astype(np.float64), Y, lam=10.0)
        res.update({"probe_%s_r2_cursor_x" % name: float(r2[0]), "probe_%s_r2_cursor_y" % name: float(r2[1]),
                    "probe_%s_r2_target_x" % name: float(r2[2]), "probe_%s_r2_target_y" % name: float(r2[3])})
    return res


def save_grid(rows, path, scale=4, pad=2):
    from PIL import Image
    h = max(im.shape[0] for r in rows for im in r)
    w = max(im.shape[1] for r in rows for im in r)
    cols = max(len(r) for r in rows)
    canvas = np.full(((h * scale + pad) * len(rows) + pad, (w * scale + pad) * cols + pad, 3), 255, np.uint8)
    for i, r in enumerate(rows):
        for j, im in enumerate(r):
            im = np.asarray(im)
            if im.dtype != np.uint8:
                im = (np.clip(im, 0, 1) * 255 + 0.5).astype(np.uint8)
            s = (h * scale) // im.shape[0]
            big = im.repeat(s, 0).repeat(s, 1)
            y0, x0 = pad + i * (h * scale + pad), pad + j * (w * scale + pad)
            canvas[y0:y0 + big.shape[0], x0:x0 + big.shape[1]] = big
    Image.fromarray(canvas).save(path)


def analyze(agent, env_cfg, n_states=256, seed=31337, out_dir=None, tag=""):
    env_cfg = dataclasses.replace(env_cfg, bonuses=False)
    f = env_cfg.size // agent.config["pred_img"]
    res = {}
    # ---- movement: 9 moves, no click, cursor outside the circle
    env = make_states(env_cfg, n_states, seed, inside=False)
    obs = env.observe()
    moves = [(dx, dy, 0.0) for dx, dy in MOVES]
    pred, _, _ = predict_actions(agent, obs, moves)
    true_img, _ = true_outcomes(env, moves)
    true = downsample(true_img, f)                                     # [n, A, p, p, 3]
    cur = downsample(obs["image"], f)                                  # [n, p, p, 3]
    mse = ((pred[:, :, None] - true[:, None, :]) ** 2).mean(axis=(-3, -2, -1))  # [n, i(pred), j(true)]
    diag = np.einsum("nii->ni", mse)
    res["action_retrieval_acc"] = float((diag <= mse.min(axis=2) + 1e-12).mean())
    # which action-conditioned prediction best explains each true outcome? (chance 1/9)
    res["outcome_explained_acc"] = float((diag <= mse.min(axis=1) + 1e-12).mean())
    # margin: how much closer the matching prediction is than the mean other prediction (relative)
    other = (mse.sum(axis=1) - diag) / (len(moves) - 1)
    res["outcome_margin_rel"] = float(np.mean((other - diag) / (other + 1e-12)))
    res["pred_mse_true_action"] = float(diag.mean())
    res["pred_mse_other_actions"] = float((mse.sum(2) - diag).mean() / (len(moves) - 1))
    c0, _ = cursor_centroid(cur)
    cp, wp = cursor_centroid(pred)
    ct, _ = cursor_centroid(true)
    d_pred = (cp - c0[:, None]) * f                                    # px in world units
    d_true = (ct - c0[:, None]) * f
    ok = wp > 1e-3
    err = np.linalg.norm(d_pred - d_true, axis=-1)
    res["disp_err"] = float(err[ok].mean())
    res["disp_err_static"] = float(np.linalg.norm(d_true, axis=-1).mean())
    res["disp_corr_x"] = float(np.corrcoef(d_pred[..., 0][ok], d_true[..., 0][ok])[0, 1])
    res["disp_corr_y"] = float(np.corrcoef(d_pred[..., 1][ok], d_true[..., 1][ok])[0, 1])
    res["cursor_visible_frac"] = float(ok.mean())
    # mean predicted displacement per commanded move (should point the commanded way, |d| ~ 4 px)
    res["mean_pred_disp_per_move"] = {MOVE_NAMES[i]: [float(v) for v in d_pred[:, i][ok[:, i]].mean(0)]
                                      for i in range(len(moves))}
    res["mean_true_disp_per_move"] = {MOVE_NAMES[i]: [float(v) for v in d_true[:, i].mean(0)] for i in range(len(moves))}
    fig_rows = []
    for s in range(min(6, n_states)):
        row = [cur[s]]
        for i in (0, 3, 8):                                            # right, up, stay
            row += [pred[s, i], true[s, i]]
        fig_rows.append(row)
    # ---- clicking: cursor inside vs outside the circle
    click_acts = [(0.0, 0.0, 0.0), (0.0, 0.0, 1.0)]
    for where in ("inside", "outside"):
        e2 = make_states(env_cfg, n_states, seed + (7 if where == "inside" else 0), inside=(where == "inside"))
        o2 = e2.observe()
        pr, pd, _ = predict_actions(agent, o2, click_acts)
        ti, td = true_outcomes(e2, click_acts)
        tr = downsample(ti, f)
        red_p = redness(pr)                                            # [n, 2]
        red_t = redness(tr)
        res["vanish_%s" % where] = float(np.mean(1 - red_p[:, 1] / (red_p[:, 0] + 1e-6)))
        res["vanish_%s_true" % where] = float(np.mean(1 - red_t[:, 1] / (red_t[:, 0] + 1e-6)))
        res["pred_hunger_noclick_%s" % where] = float(pd[:, 0, 0].mean())
        res["pred_hunger_click_%s" % where] = float(pd[:, 1, 0].mean())
        res["true_hunger_noclick_%s" % where] = float(td[:, 0, 0].mean())
        res["true_hunger_click_%s" % where] = float(td[:, 1, 0].mean())
        if where == "inside":
            for s in range(min(6, n_states)):
                fig_rows[s] += [downsample(o2["image"][s], f), pr[s, 0], pr[s, 1], tr[s, 1]]
    res["hunger_drop_inside"] = res["pred_hunger_noclick_inside"] - res["pred_hunger_click_inside"]
    res.update(probe(agent, env_cfg))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        save_grid(fig_rows, os.path.join(out_dir, "counterfactual%s.png" % tag))
        with open(os.path.join(out_dir, "dynamics%s.json" % tag), "w") as fp:
            json.dump(res, fp, indent=2)
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", nargs="?")
    ap.add_argument("--ckpt", default="ckpt_final.bin")
    ap.add_argument("--states", type=int, default=256)
    ap.add_argument("--untrained", action="store_true", help="analyze a random-init model (default config)")
    ap.add_argument("--drives", action="store_true", help="with --untrained: world exposes drives")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    if args.untrained:
        agent = agent_lib.Agent({}, seed=1)
        env_cfg = EnvConfig(drives=args.drives)
        out_dir = args.out
    else:
        with open(os.path.join(args.run_dir, "info.json")) as fp:
            info = json.load(fp)
        agent = agent_lib.Agent.load(os.path.join(args.run_dir, args.ckpt))
        ec = dict(info["env_config"])
        for k in ("bg_color", "target_color", "cursor_color", "cursor_outline"):
            ec[k] = tuple(ec[k])
        env_cfg = EnvConfig(**ec)
        out_dir = args.out or os.path.join(args.run_dir, "analysis", os.path.splitext(args.ckpt)[0])
    res = analyze(agent, env_cfg, args.states, out_dir=out_dir)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()

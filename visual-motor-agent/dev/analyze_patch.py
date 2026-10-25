"""Patch tokens and distractor dots (vm-00010): do patch tokens couple where things change, and does the model
tell its own cursor from the dots? The true cursor and dot positions are used for scoring only.

  coupling     When the cursor (or a dot) moves from patch A at step t to patch B at step t+1: the attention weight
               of B's token on A's token at step t, relative to its mean weight on the other patch tokens of step t
               (per layer, averaged over heads). Control: the same ratio for a random other patch C looking at A.
               B above C means coupling; both above 1 alone means A is merely salient. Patch-token models only.
  sensitivity  How the predicted next frame changes when the executed move is reversed: the share of that change
               near the cursor, near the dots and elsewhere, against the share of the area. Any model.
  tokens       Ridge probes of the cursor, the circle and the direction from each kind of token output (move,
               drives, command, the patch tokens' mean, all patch tokens side by side). Patch-token models only.
               With a detached policy head, no loss trains the move token's output (z) directly.

Worlds are driven by random moves without clicks, as configured in the run (dots included).

usage:
  ./venv/bin/python analyze_patch.py coupling RUN_DIR [RUN_DIR ...] --step 50000
  ./venv/bin/python analyze_patch.py sensitivity RUN_DIR [RUN_DIR ...] --step 50000
  ./venv/bin/python analyze_patch.py tokens RUN_DIR [RUN_DIR ...] --step 50000
"""
import argparse
import dataclasses
import json
import os

import numpy as np

import agent_lib
from analyze_self_actor import load, name
from world.circle_world import CircleWorld


def windows(cfg, n, T, seed):
    """n worlds, T + 1 steps of random moves: the model's inputs for steps 0..T-1 and the positions at 0..T."""
    env = CircleWorld(n, dataclasses.replace(cfg, auto_reset=False, bonuses=False, max_steps=10 ** 6), seed=seed)
    rng = np.random.default_rng(seed)
    keys = ("img", "drv", "pa", "act", "cursor", "dots")
    w = {k: [] for k in keys}
    obs, prev = env.observe(), np.zeros((n, 3), np.float32)
    for _ in range(T + 1):
        a = np.concatenate([rng.normal(0, 0.6, (n, 2)), np.zeros((n, 1))], 1).astype(np.float32)
        for k, v in zip(keys, (obs["image"], obs["drives"], prev, a, env.cursor.copy(), env.dot_pos.copy())):
            w[k].append(v)
        obs = env.step(a)[0]
        prev = np.concatenate([np.clip(a[:, :2], -1, 1), np.zeros((n, 1))], 1).astype(np.float32)
    return {k: np.stack(v, 1) for k, v in w.items()}


def inputs(w, T, goal):
    n = w["img"].shape[0]
    cmd = (w["drv"][:, :T, 0] if goal else np.zeros((n, T))).astype(np.float32)   # acting: ask for all hunger
    es = np.zeros((n, T), np.uint8)
    es[:, 0] = 1
    return w["img"][:, :T], w["drv"][:, :T].astype(np.float32), w["pa"][:, :T], cmd, es


def cell(xy, ps, g):
    c = np.clip(np.floor(xy).astype(int) // ps, 0, g - 1)
    return int(c[1] * g + c[0])


def coupling(run, st, n=128, seed=7):
    info, cfg, a = load(run, st)
    c = a.config
    if not c["patch"]:
        a.close()
        return None
    ps, g, T = c["patch"], c["img"] // c["patch"], c["ctx"]
    npch, nt, half = g * g, g * g + 3, cfg.dot_size / 2.0
    w = windows(cfg, n, T, seed)
    img, drv, pa, cmd, es = inputs(w, T, info["config"]["online"]["goal"])
    ratios = {"cursor": [], "dots": []}
    control = {"cursor": [], "dots": []}
    rng = np.random.default_rng(seed)
    for i in range(n):
        att = a.attention(img[i], drv[i], pa[i], cmd[i], es[i]).mean(1)      # [L, Tt, Tt], heads averaged
        for t in range(T):
            moves = [("cursor", cell(w["cursor"][i, t], ps, g), cell(w["cursor"][i, t + 1], ps, g))]
            moves += [("dots", cell(w["dots"][i, t, k] + half, ps, g), cell(w["dots"][i, t + 1, k] + half, ps, g))
                      for k in range(w["dots"].shape[2])]
            for kind, A, B in moves:
                if A == B:
                    continue
                row = att[:, t * nt + B, t * nt:t * nt + npch]                 # B's weights on step t's patches
                others = np.delete(row, [A, B], axis=1).mean(1)
                ratios[kind].append(row[:, A] / others)
                C = rng.choice([q for q in range(npch) if q not in (A, B)])    # an unrelated patch looking at A
                rc = att[:, t * nt + C, t * nt:t * nt + npch]
                control[kind].append(rc[:, A] / np.delete(rc, [A, C], axis=1).mean(1))
    a.close()
    return {k: dict(events=len(v), median=np.median(v, 0).tolist(), above_1=(np.array(v) > 1).mean(0).tolist(),
                    control_median=np.median(control[k], 0).tolist(),
                    b_above_c=(np.array(v) > np.array(control[k])).mean(0).tolist())
            for k, v in ratios.items() if v}


def sensitivity(run, st, n=256, seed=8, radius=6.0):
    info, cfg, a = load(run, st)
    c = a.config
    T, p = c["ctx"], c["pred_img"]
    w = windows(cfg, n, T, seed)
    x = inputs(w, T, info["config"]["online"]["goal"])
    acts = w["act"][:, :T].copy()
    base = a.forward(*x, pred_actions=acts)["pred_frames"][:, -1]
    acts[:, -1, :2] *= -1                                                       # the opposite move at the last step
    flip = a.forward(*x, pred_actions=acts)["pred_frames"][:, -1]
    a.close()
    diff = np.abs(flip - base).sum(-1)                                          # [n, p, p]
    f = cfg.size / p
    yy, xx = (np.mgrid[0:p, 0:p] + 0.5) * f                                     # pixel centres, image coordinates

    def near(pts):                                                              # pts [n, m, 2] -> mask [n, p, p]
        d2 = (xx[None, None] - pts[..., 0, None, None]) ** 2 + (yy[None, None] - pts[..., 1, None, None]) ** 2
        return (d2 <= radius ** 2).any(1)

    cur = w["cursor"][:, T - 1]
    step = np.clip(w["act"][:, T - 1, :2], -1, 1) * cfg.max_step_px
    cursor = near(np.stack([cur, np.clip(cur + step, 0, cfg.size), np.clip(cur - step, 0, cfg.size)], 1))
    dots = (near(np.concatenate([w["dots"][:, T - 1], w["dots"][:, T]], 1) + cfg.dot_size / 2.0)
            if w["dots"].shape[2] else np.zeros_like(cursor)) & ~cursor
    total = diff.sum()
    return dict(cursor_share=float(diff[cursor].sum() / total), cursor_area=float(cursor.mean()),
                dots_share=float(diff[dots].sum() / total), dots_area=float(dots.mean()),
                change=float(diff.mean()))


def tokens(run, st, n=16384, chunk=2048):
    """Ridge probes of the cursor, the circle and the direction from each kind of token output of a one-step window
    (probe states as analyze_self_actor.py probe, but 4x more of them for the wide readouts). The token outputs come
    from the PyTorch reference, which the tests match to the C++ model; the move token's (z) is checked here too."""
    import torch
    from reference import torch_reference as ref
    from analyze_self_actor import probe_states, ridge_r2
    torch.set_num_threads(agent_lib.get_threads())
    info, cfg, a = load(run, st)
    c = a.config
    if not c["patch"]:
        a.close()
        return None
    obs, gt = probe_states(cfg, n=n)
    goal = info["config"]["online"]["goal"]
    x = (obs["image"][:, None], obs["drives"][:, None].astype(np.float32), np.zeros((n, 1, 3), np.float32),
         (obs["drives"][:, :1] if goal else np.zeros((n, 1))).astype(np.float32), np.ones((n, 1), np.uint8))
    uni = bool(c.get("universal"))
    z_cpp = None if uni else a.forward(*x)["z"][:, -1]
    P = ref.params_from_flat(a.params(), a.param_layout(), c, dtype=torch.float32)
    a.close()
    hs = []
    with torch.no_grad():
        for i in range(0, n, chunk):
            if uni:   # the universal model: tokens body, patches, move; the move made is "stay"
                o = ref.forward_uni(P, c, x[0][i:i + chunk], x[1][i:i + chunk], np.zeros((len(x[0][i:i + chunk]), 1, 3)),
                                    x[4][i:i + chunk])
                hs.append(o["h"].reshape(len(x[0][i:i + chunk]), 1, -1, c["d_model"])[:, -1].numpy())
            else:
                hs.append(ref.forward_patch(P, c, *[v[i:i + chunk] for v in x])["h"][:, -1].numpy())
    h = np.concatenate(hs)                                                      # [n, ntok, d]
    npch = (c["img"] // c["patch"]) ** 2
    if uni:   # body, the patches, the move: the last patch's output predicts the move (the policy's input)
        feats = dict(body=h[:, 0], last_patch=h[:, npch], move=h[:, npch + 1], patch_mean=h[:, 1:npch + 1].mean(1),
                     patches=h[:, 1:npch + 1].reshape(n, -1))
        res = {"%s_%s" % (f, k): ridge_r2(X, Y, ntr=3 * n // 4) for f, X in feats.items() for k, Y in gt.items()}
        res["z_max_abs_diff"] = 0.0
        return res
    zi = npch + 3 if c.get("act_token") else npch + 1          # the token the policy reads (act token, else move)
    feats = dict(move=h[:, npch + 1], drives=h[:, npch], command=h[:, npch + 2], patch_mean=h[:, :npch].mean(1),
                 patches=h[:, :npch].reshape(n, -1))
    if c.get("act_token"):
        feats["act"] = h[:, zi]
    res = {"%s_%s" % (f, k): ridge_r2(X, Y, ntr=3 * n // 4) for f, X in feats.items() for k, Y in gt.items()}
    res["z_max_abs_diff"] = float(np.abs(h[:, zi] - z_cpp).max())
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("what", choices=["coupling", "sensitivity", "tokens"])
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--step", type=int, default=50000)
    ap.add_argument("--threads", type=int, default=8)
    args = ap.parse_args()
    agent_lib.set_threads(args.threads)
    for run in args.runs:
        out = os.path.join(run, "eval", "ckpt_%08d_%s.json" % (args.step, args.what))
        r = dict(coupling=coupling, sensitivity=sensitivity, tokens=tokens)[args.what](run, args.step)
        if r is None:
            print("%-20s %7d  (no patch tokens)" % (name(run), args.step))
            continue
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with open(out, "w") as fh:
            json.dump(r, fh)
        if args.what == "coupling":
            for kind, v in r.items():
                print("%-20s %7d  %-6s %5d crossings: B's attention on A / on other patches, by layer: median %s"
                      " (an unrelated patch C on A: %s); B above C in %s" % (
                          name(run), args.step, kind, v["events"], " ".join("%.2f" % m for m in v["median"]),
                          " ".join("%.2f" % m for m in v["control_median"]),
                          " ".join("%.2f" % m for m in v["b_above_c"])), flush=True)
        elif args.what == "tokens":
            print("%-20s %7d  R^2 (cursor circle direction) from the token outputs:%s   (z vs C++: %.1e)" % (
                name(run), args.step, "".join("  %s %.2f %.2f %.2f" % (f, r[f + "_cursor"], r[f + "_circle"],
                                                                        r[f + "_direction"])
                                               for f in ("act", "move", "drives", "command", "body", "last_patch",
                                                         "patch_mean", "patches")
                                               if f + "_cursor" in r),
                r["z_max_abs_diff"]), flush=True)
        else:
            print("%-20s %7d  reversing the move changes the prediction by %.4f per pixel; near the cursor %.2f of it"
                  " on %.2f of the area; near the dots %.2f on %.2f" % (name(run), args.step, r["change"],
                  r["cursor_share"], r["cursor_area"], r["dots_share"], r["dots_area"]), flush=True)


if __name__ == "__main__":
    main()

"""Does a planner agent's learned hunger model see the hazards (vm-00012)? The true cursor, circle and hazard positions
place the states (scoring only; the model sees only the rendered frames and its drives).

  hazard   For n states with the cursor inside a hazard and the same states with the cursor moved outside every hazard
           and the circle: imagine H steps of standing still (no move, no click) from each, and compare the imagined
           hunger rise with the world's (h_rate per step, plus hazard_cost per step inside a hazard). The hazards
           drift, so standing still inside one does not keep the cursor inside for all H steps; the world's rise is
           measured by stepping copies of the world.

  contact  Is hazard contact represented at all? For 4,096 placed states, half with the cursor inside a hazard: a
           linear probe of "inside a hazard" from v (the encoder's features) and from z (the Transformer output), and
           the drives head's predicted next hunger inside against outside (with no move), against the world's.
           Patch-token models (vm-00014) have no v: the probes read the token outputs instead (from the PyTorch
           reference, as analyze_patch.py tokens): z (the token the policy reads: the act token, else the move
           token), the move token, the drives token, the token of the patch under the cursor and the patch mean.

  click    Poison circles (env.hazard_click, vm-00017): does the model predict what a click does? Placed states with
           hunger 0.3 and the cursor in the red circle (food), in a hazard (poison) or on the background; for the move
           "stay and click" and "stay, no click": the drives head's predicted next hunger against the world's, the
           predicted redness of the circle's pixels (it vanishes when eaten), and for a planner agent its latent hunger
           head over one imagined step.

  click-real  The click test on real windows (universal model, task 00027): the frozen agent lives as in the forage
           test (64 continual worlds, 1,000 steps, its own moves and instinct); at every step its chosen move is tried
           with and without a click: the predicted effect (next hunger, click minus no click) from windows of its last
           1, 2, 4 and ctx steps, the true effect from copies of the world, by where the cursor lands (food, poison,
           nothing); also the error of the current patches' prediction per history length.

usage:
  ./venv/bin/python analyze_hazards.py hazard RUN_DIR [RUN_DIR ...] --step 40000 [--device cuda]
  ./venv/bin/python analyze_hazards.py contact RUN_DIR [RUN_DIR ...] --step 40000
  ./venv/bin/python analyze_hazards.py click RUN_DIR [RUN_DIR ...] --step 40000 [--device cuda]
  ./venv/bin/python analyze_hazards.py click-real RUN_DIR [RUN_DIR ...] --step 200000 --device cuda [--instinct 0]
"""
import argparse
import dataclasses
import json
import os

import numpy as np

import agent_lib
from analyze_self_actor import load, name, ridge_r2
from world.circle_world import CircleWorld, wrap_delta


def hazard(run, st, n=256, H=12, seed=11, device="cpu"):
    info, cfg, a = load(run, st)
    if not cfg.hazards or not a.config["dyn_hidden"]:
        a.close()
        return None
    m = a
    if device == "cuda":
        import cuda_lib
        m = cuda_lib.CudaModel(a)
    cfg = dataclasses.replace(cfg, auto_reset=False, bonuses=False, max_steps=10 ** 9)
    rng = np.random.default_rng(seed)
    env = CircleWorld(n, cfg, seed=seed)
    for _ in range(int(rng.integers(5, 40))):      # let the hazards drift away from their first positions
        env.step(np.zeros((n, 3)))
    S = float(cfg.size)
    hz = env.hz_pos[:, 0].copy()                     # cursor inside hazard 0: at its center
    inside_state = dict(cursor=hz, hunger=np.full(n, 0.3))
    out = rng.uniform(0, S, (n, 2))
    for _ in range(100):                             # outside every hazard and the circle
        dh = np.stack([np.hypot(wrap_delta(out[:, 0, None], env.hz_pos[..., 0], S) if cfg.wrap else out[:, 0, None] - env.hz_pos[..., 0],
                                wrap_delta(out[:, 1, None], env.hz_pos[..., 1], S) if cfg.wrap else out[:, 1, None] - env.hz_pos[..., 1])
                       - env.hz_rad], -1).min(-1)
        dx, dy = out[:, 0] - env.target[:, 0], out[:, 1] - env.target[:, 1]
        if cfg.wrap:
            dx, dy = wrap_delta(dx, 0.0, S), wrap_delta(dy, 0.0, S)
        dt = np.hypot(dx, dy) - env.radius
        bad = (dh.min(-1) < 2.0) | (dt < 2.0)
        if not bad.any():
            break
        out[bad] = rng.uniform(0, S, (bad.sum(), 2))
    outside_state = dict(cursor=out, hunger=np.full(n, 0.3))
    res = {}
    base = env.get_state()
    for kind, stt in (("inside", inside_state), ("outside", outside_state)):
        env.set_state(base)
        env.set_state(stt)
        obs = env.observe()
        s0 = m.encode(obs["image"])["vf"]
        im = m.imagine(s0, env.hunger.astype(np.float32), np.zeros((n, H, 3), np.float32))["hunger"]
        hs = []
        for _ in range(H):                            # the world's own rise, standing still
            env.step(np.zeros((n, 3)))
            hs.append(env.hunger.copy())
        res[kind] = dict(imagined_rise=float((im[:, -1] - 0.3).mean()), true_rise=float((hs[-1] - 0.3).mean()))
    if m is not a:
        m.close()
    a.close()
    res["imagined_extra"] = res["inside"]["imagined_rise"] - res["outside"]["imagined_rise"]
    res["true_extra"] = res["inside"]["true_rise"] - res["outside"]["true_rise"]
    return res


def contact(run, st, n=4096, seed=5):
    info, cfg, a = load(run, st)
    if not cfg.hazards:
        a.close()
        return None
    cfg = dataclasses.replace(cfg, auto_reset=False, bonuses=False, max_steps=10 ** 9)
    env = CircleWorld(n, cfg, seed=seed)
    rng = np.random.default_rng(seed)
    S = float(cfg.size)
    inside = rng.random(n) < 0.5
    cur = rng.uniform(0, S, (n, 2))
    cur[inside] = env.hz_pos[inside, 0] + rng.uniform(-2, 2, (int(inside.sum()), 2))
    env.set_state(dict(cursor=np.mod(cur, S) if cfg.wrap else np.clip(cur, 0, S - 1e-4), hunger=np.full(n, 0.4)))
    dx = env.cursor[:, None, 0] - env.hz_pos[..., 0]
    dy = env.cursor[:, None, 1] - env.hz_pos[..., 1]
    if cfg.wrap:
        dx, dy = wrap_delta(dx, 0.0, S), wrap_delta(dy, 0.0, S)
    lab = ((dx * dx + dy * dy) <= env.hz_rad ** 2).any(1).astype(np.float64)
    obs = env.observe()
    goal = info["config"]["online"].get("goal")
    cmd = obs["drives"][:, :1].astype(np.float32) if goal else np.zeros((n, 1), np.float32)
    x = (obs["image"][:, None], obs["drives"][:, None].astype(np.float32), np.zeros((n, 1, 3), np.float32), cmd,
         np.ones((n, 1), np.uint8))
    out = a.forward(*x, pred_actions=np.zeros((n, 1, 3), np.float32))
    ph = out["pred_drives"][:, 0, 0]
    res = dict(inside_share=float(lab.mean()), r2_z=ridge_r2(out["z"][:, 0], lab[:, None]),
               drives_head_inside=float(ph[lab > 0].mean()), drives_head_outside=float(ph[lab == 0].mean()),
               true_inside=float(0.4 + cfg.h_rate + cfg.hazard_cost), true_outside=float(0.4 + cfg.h_rate))
    c = a.config
    if not c["patch"]:
        res["r2_v"] = ridge_r2(a.encode(obs["image"])["vf"], lab[:, None])
        a.close()
        return res
    import torch
    from reference import torch_reference as ref
    torch.set_num_threads(agent_lib.get_threads())
    P = ref.params_from_flat(a.params(), a.param_layout(), c, dtype=torch.float32)
    a.close()
    with torch.no_grad():
        h = np.concatenate([ref.forward_patch(P, c, *[v[i:i + 1024] for v in x])["h"][:, -1].numpy()
                            for i in range(0, n, 1024)])                       # [n, ntok, d]
    g = c["img"] // c["patch"]
    npch = g * g
    zi = npch + 3 if c.get("act_token") else npch + 1          # the token the policy reads (act token, else move)
    view = np.full((n, 2), S / 2) if cfg.egocentric else env.cursor
    cell = (np.clip(np.floor(view[:, 1]).astype(int) // c["patch"], 0, g - 1) * g +
            np.clip(np.floor(view[:, 0]).astype(int) // c["patch"], 0, g - 1))
    res.update(r2_drives=ridge_r2(h[:, npch], lab[:, None]), r2_move=ridge_r2(h[:, npch + 1], lab[:, None]),
               r2_cursor_patch=ridge_r2(h[np.arange(n), cell], lab[:, None]),
               r2_patch_mean=ridge_r2(h[:, :npch].mean(1), lab[:, None]),
               z_max_abs_diff=float(np.abs(h[:, zi] - out["z"][:, 0]).max()))
    return res


def place_click_states(env, cfg, rng, n):
    """Cursor positions: a third in the circle (not in a hazard), a third in a hazard (not in the circle), a third on
    the background (2 px clear of both). Returns the kinds (0 food, 1 poison, 2 background)."""
    S = float(cfg.size)
    kind = np.arange(n) % 3

    def gap(p):   # distance outside the circle, and outside the nearest hazard (negative: inside)
        d = lambda a, b: wrap_delta(a, b, S) if cfg.wrap else a - b
        dt = np.hypot(d(p[:, 0], env.target[:, 0]), d(p[:, 1], env.target[:, 1])) - env.radius
        dh = (np.hypot(d(p[:, 0, None], env.hz_pos[..., 0]), d(p[:, 1, None], env.hz_pos[..., 1])) - env.hz_rad).min(1)
        return dt, dh
    cur = np.zeros((n, 2))
    todo = np.ones(n, bool)
    for _ in range(200):
        m = int(todo.sum())
        if not m:
            break
        c = rng.uniform(0, S, (m, 2))
        k = kind[todo]
        ang, rad = rng.uniform(0, 2 * np.pi, m), rng.uniform(0, 0.6, m)
        off = np.stack([np.cos(ang), np.sin(ang)], 1) * rad[:, None]
        c[k == 0] = (env.target[todo] + off * env.radius[todo, None])[k == 0]
        hz = env.hz_pos[todo, 0] + off * env.hz_rad[todo, 0, None]
        c[k == 1] = hz[k == 1]
        cur[todo] = np.mod(c, S) if cfg.wrap else np.clip(c, 0, S - 1e-4)
        dt, dh = gap(cur)
        ok = np.where(kind == 0, (dt < 0) & (dh > 1), np.where(kind == 1, (dh < 0) & (dt > 1), (dt > 2) & (dh > 2)))
        todo = ~ok
    keep = ~todo
    return kind, cur, keep


def click(run, st, n=1536, seed=13, device="cpu"):
    info, cfg, a = load(run, st)
    if cfg.hazard_click < 0 and cfg.hazard_click_add <= 0:
        a.close()
        return None
    m = a
    if device == "cuda":
        import cuda_lib
        m = cuda_lib.CudaModel(a)
    cfg = dataclasses.replace(cfg, auto_reset=False, bonuses=False, max_steps=10 ** 9)
    env = CircleWorld(n, cfg, seed=seed)
    rng = np.random.default_rng(seed)
    for _ in range(int(rng.integers(5, 40))):      # let the hazards drift from their first positions
        env.step(np.zeros((n, 3)))
    kind, cur, keep = place_click_states(env, cfg, rng, n)
    env.set_state(dict(cursor=cur, hunger=np.full(n, 0.3)))
    base = env.get_state()
    obs = env.observe()
    goal = info["config"]["online"].get("goal")
    cmd = obs["drives"][:, :1].astype(np.float32) if goal else np.zeros((n, 1), np.float32)
    f = cfg.size // a.config["pred_img"]
    from online import downsample_u8
    cur32 = downsample_u8(obs["image"], f).astype(np.float32)
    tc = np.array(cfg.target_color, np.float32)
    red = np.abs(cur32 - tc).sum(-1) < 30            # the circle's pixels in the prediction frame
    redness = lambda fr: np.array([(fr[i][red[i]][:, 0] - fr[i][red[i]][:, 2]).mean() if red[i].any() else np.nan
                                   for i in range(n)])
    res = {}
    uni = bool(a.config.get("universal"))
    for c in (0, 1):
        act = np.zeros((n, 1, 3), np.float32)
        act[..., 2] = c
        if uni:   # the move token's prediction of the next body; the next frame would need generating patch by patch
            ph = m.forward_uni(obs["image"][:, None], obs["drives"][:, None].astype(np.float32), act,
                               np.ones((n, 1), np.uint8), values=False, move_probs=False,
                               next_body=True)["next_body"][:, 0, 0]
            pf = np.full(n, np.nan)
        else:
            fwd = m.forward if m is not a else a.forward
            out = fwd(obs["image"][:, None], obs["drives"][:, None].astype(np.float32), np.zeros((n, 1, 3), np.float32),
                      cmd, np.ones((n, 1), np.uint8), pred_actions=act)
            ph = out["pred_drives"][:, 0, 0]
            pf = redness(out["pred_frames"][:, 0] * 255.0)
        env.set_state(base)
        nobs = env.step(act[:, 0])[0]
        th = env.hunger.copy()
        tf = redness(downsample_u8(nobs["image"], f).astype(np.float32))
        ih = None
        if a.config["dyn_hidden"]:
            s0 = m.encode(obs["image"])["vf"]
            ih = m.imagine(s0, np.full(n, 0.3, np.float32), act)["hunger"][:, 0]
        for k, nm in enumerate(("food", "poison", "background")):
            sel = keep & (kind == k)
            r = dict(n=int(sel.sum()), predicted=float(ph[sel].mean()), true=float(th[sel].mean()),
                     red_predicted=float(np.nanmean(pf[sel])) if k == 0 and not uni else None,
                     red_true=float(np.nanmean(tf[sel])) if k == 0 else None)
            if ih is not None:
                r["imagined"] = float(ih[sel].mean())
            res["%s_%s" % (nm, "click" if c else "stay")] = r
    if m is not a:
        m.close()
    a.close()
    return res


def patch_values(cfg, img):
    """The patch tokens' values [n, patches, V] in [0, 1] of frames [n, H, W, 3] (online.uni_patches)."""
    from online import uni_patches
    return uni_patches(cfg, img)


def branch_hunger(model, img, drv, act, es, cand, values=False):
    """The universal model's predicted next hunger [S, Mc] for each candidate cand [S, Mc, 3] in the windows' last move
    slot (as the instinct reads it), and the forward pass's values (values=True; the last move slot zeroed). With the
    GPU's uni_candidates the candidates reuse the forward pass; otherwise a full pass each."""
    a0 = act.copy()
    a0[:, -1] = 0.0
    out = model.forward_uni(img, drv, a0, es, values=values, move_probs=True)
    if hasattr(model, "uni_candidates"):
        h = model.uni_candidates(cand)
    else:
        h = np.empty(cand.shape[:2], np.float32)
        for j in range(cand.shape[1]):
            a0[:, -1] = cand[:, j]
            h[:, j] = model.forward_uni(img, drv, a0, es, values=False, move_probs=False,
                                        next_body=True)["next_body"][:, -1, 0]
    return h, out.get("values")


KINDS = ("food", "poison", "nothing")


def click_truth(env, mv):
    """One real step of moves mv [n, 3], plus the same moves with the other click on a copy of the world (restored
    before the real step; scoring only). Returns the real step's obs and info, where a click would land (0 food,
    1 poison, 2 nothing, 3 both) and the true effect of the click on the next hunger (click minus no click)."""
    clk = mv[:, 2] > 0.5
    state = env.get_state()
    other = mv.copy()
    other[:, 2] = (~clk).astype(np.float32)
    inf_o = env.step(other)[3]
    h_o = env.hunger.copy()
    none = np.zeros(env.num_envs, bool)
    food_o, pois_o = inf_o["success"].copy(), inf_o.get("gt_hazard_click", none).copy()
    env.set_state(state)
    obs, _, _, inf = env.step(mv)
    h_a = env.hunger
    food = np.where(clk, inf["success"], food_o)                    # what the click would hit
    pois = np.where(clk, inf.get("gt_hazard_click", none), pois_o)
    kind = np.where(food & pois, 3, np.where(food, 0, np.where(pois, 1, 2)))
    return obs, inf, kind, np.where(clk, h_a - h_o, h_o - h_a)


def click_real(run, st, n=64, steps=1000, device="cpu", instinct=None, seed=None, tol=0.005):
    """The click test on real windows (task 00027): the frozen universal model lives as in the forage test; at every
    step its chosen move is tried with and without a click, the predicted effect (next hunger with click 1 minus 0)
    read from windows of its last k steps, the true effect from copies of the world (scoring only)."""
    from online import DEFAULT_ONLINE, EVAL_SEED, eval_moves
    info, cfg, a = load(run, st)
    if not a.config.get("universal") or not cfg.respawn or (cfg.hazard_click < 0 and cfg.hazard_click_add <= 0):
        a.close()
        return None
    m = a
    if device == "cuda":
        import cuda_lib
        m = cuda_lib.CudaModel(a)
    ion = dict(DEFAULT_ONLINE, **info["config"]["online"])
    inst = (ion["instinct_m"] if instinct is None else instinct, ion["instinct_tol"], ion["instinct_h"],
            ion["instinct_jacobi"], ion["instinct_decode"])
    env = CircleWorld(n, dataclasses.replace(cfg, auto_reset=False, bonuses=False, max_steps=10 ** 9),
                      seed=EVAL_SEED if seed is None else seed)
    rng = np.random.default_rng(0)
    c = a.config
    L, pp = c["ctx"], agent_lib.uni_layout(c)["patch"] - 1   # the tokens that predict the patches (task 00047)
    hists = sorted({k for k in (1, 2, 4, L) if k <= L})
    hist, events, prev_pv = [], [], None   # the window's earlier steps, their clicks (1 food, 2 poison), patches
    obs, es = env.observe(), np.ones(n, np.uint8)
    rec = {key: [] for key in ("kind", "true", "seen_food", "seen_poison")}
    for k in hists:
        rec.update({"pred_%d" % k: [], "act_err_%d" % k: [], "patch_%d" % k: [], "changed_%d" % k: []})
    food_clicks = poison_clicks = 0
    for _ in range(steps):
        cur = dict(img=obs["image"], drv=obs["drives"].astype(np.float32), act=np.zeros((n, 3), np.float32), es=es)
        win = hist + [cur]
        mv = eval_moves(m, win, rng, inst)
        clk = mv[:, 2] > 0.5
        obs, inf, kind, true = click_truth(env, mv)
        h_a = env.hunger.copy()
        none = np.zeros(n, bool)
        food_a, pois_a = inf["success"], inf.get("gt_hazard_click", none)
        rec["kind"].append(kind)
        rec["true"].append(true)
        ev = np.stack(events[-(L - 1):], 1) if events and L > 1 else np.zeros((n, 1), np.int8)
        rec["seen_food"].append((ev == 1).any(1))
        rec["seen_poison"].append((ev == 2).any(1))
        # the predictions, from the window's last k steps (the context restarting there)
        cand = np.stack([mv, mv], 1)
        cand[:, 0, 2], cand[:, 1, 2] = 0.0, 1.0
        pv = patch_values(c, cur["img"])
        changed = None if prev_pv is None else np.abs(pv - prev_pv).max(-1) > 0   # [n, patches]
        for k in hists:
            w = win[-k:]
            sk = lambda key: np.stack([x[key] for x in w], 1)   # noqa: E731
            esk = sk("es").copy()
            esk[:, 0] = 1
            h, vals = branch_hunger(m, sk("img"), sk("drv"), sk("act"), esk, cand, values=True)
            rec["pred_%d" % k].append(h[:, 1] - h[:, 0])
            rec["act_err_%d" % k].append(np.abs(h[np.arange(n), clk.astype(int)] - h_a))
            err = ((vals[:, -1, pp] - pv) ** 2).mean(-1)                            # [n, patches]
            rec["patch_%d" % k].append(err.mean(1))
            rec["changed_%d" % k].append(np.full(n, np.nan) if changed is None else
                                         np.where(changed.any(1), (err * changed).sum(1) / np.maximum(changed.sum(1), 1),
                                                  np.nan))
        food_clicks += int(food_a.sum())
        poison_clicks += int(pois_a.sum())
        cur["act"] = mv
        hist = (hist + [cur])[-(L - 1):] if L > 1 else []
        events.append(np.where(food_a, 1, np.where(pois_a, 2, 0)).astype(np.int8))
        events = events[-(L - 1):] if L > 1 else []
        prev_pv, es = pv, np.zeros(n, np.uint8)
    if m is not a:
        m.close()
    a.close()
    r = {key: np.concatenate(v) for key, v in rec.items()}
    return summarize_click_real(r, hists, n, steps, inst[0], food_clicks, poison_clicks, tol)


def summarize_click_real(r, hists, n, steps, instinct_m, food_clicks, poison_clicks, tol=0.005):
    """Per kind and history length: cases, mean predicted and true click effect, and the share ranked right beyond
    tol among the felt cases (|true| > 0.05; food: lower, poison: higher) or, for nothing, predicted flat."""
    kind, true = r["kind"], r["true"]

    def stats(pred, sel, k):
        felt = sel & (np.abs(true) > 0.05)
        d = dict(n=int(sel.sum()), predicted=float(pred[sel].mean()) if sel.any() else None,
                 true=float(true[sel].mean()) if sel.any() else None, n_felt=int(felt.sum()))
        if k == 0:
            d["right"] = float((pred[felt] < -tol).mean()) if felt.any() else None
        elif k == 1:
            d["right"] = float((pred[felt] > tol).mean()) if felt.any() else None
        else:
            d["flat"] = float((np.abs(pred[sel]) < 0.05).mean()) if sel.any() else None
        return d
    res = dict(worlds=n, steps=steps, instinct_m=instinct_m, food_per_1k=food_clicks * 1000.0 / (n * steps),
               poison_per_1k=poison_clicks * 1000.0 / (n * steps), both=int((kind == 3).sum()), hist={})
    for k in hists:
        pred = r["pred_%d" % k]
        h = {nm: stats(pred, kind == j, j) for j, nm in enumerate(KINDS)}
        ch = r["changed_%d" % k]
        h.update(next_hunger_mae=float(r["act_err_%d" % k].mean()), patch_mse=float(r["patch_%d" % k].mean()),
                 patch_changed_mse=float(np.nanmean(ch)) if np.isfinite(ch).any() else None)
        res["hist"][str(k)] = h
    pred = r["pred_%d" % hists[-1]]
    for j, nm in ((0, "food"), (1, "poison")):
        seen = r["seen_" + nm]
        res["%s_seen" % nm] = stats(pred, (kind == j) & seen, j)
        res["%s_unseen" % nm] = stats(pred, (kind == j) & ~seen, j)
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("what", choices=["hazard", "contact", "click", "click-real"])
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--step", type=int, default=40000)
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--instinct", type=int, default=None, help="click-real: the instinct's candidates (default: the "
                    "run's; 0: off)")
    ap.add_argument("--worlds", type=int, default=64, help="click-real: worlds")
    ap.add_argument("--steps", type=int, default=1000, help="click-real: steps per world")
    args = ap.parse_args()
    agent_lib.set_threads(16)
    for run in args.runs:
        if args.what == "click-real":
            r = click_real(run, args.step, n=args.worlds, steps=args.steps, device=args.device, instinct=args.instinct)
            if r is None:
                print("%-14s %7d  (not a universal model in a continual world with poison)" % (name(run), args.step))
                continue
            os.makedirs(os.path.join(run, "eval"), exist_ok=True)
            tag = "" if args.instinct is None else "_instinct%d" % args.instinct
            with open(os.path.join(run, "eval", "ckpt_%08d_click_real%s.json" % (args.step, tag)), "w") as f:
                json.dump(r, f)
            fmt = lambda v: "  -  " if v is None else "%+.2f" % v   # noqa: E731
            shr = lambda v: " - " if v is None else "%.2f" % v      # noqa: E731
            parts = []
            for k, h in r["hist"].items():
                f, p, z = h["food"], h["poison"], h["nothing"]
                parts.append("k=%s food %s (%s) poison %s (%s) nothing %s (flat %s) patch %.4f/%s" % (
                    k, fmt(f["predicted"]), shr(f["right"]), fmt(p["predicted"]), shr(p["right"]),
                    fmt(z["predicted"]), shr(z["flat"]), h["patch_mse"],
                    "-" if h["patch_changed_mse"] is None else "%.4f" % h["patch_changed_mse"]))
            print("%-14s %7d  clicks/1k food %.1f poison %.1f; cases food %d poison %d (true %s, %s); %s;  poison "
                  "after a poison click in the window: right %s (n %d), otherwise %s (n %d)" % (
                      name(run), args.step, r["food_per_1k"], r["poison_per_1k"], r["hist"]["1"]["food"]["n"],
                      r["hist"]["1"]["poison"]["n"], fmt(r["hist"]["1"]["food"]["true"]),
                      fmt(r["hist"]["1"]["poison"]["true"]), ";  ".join(parts), shr(r["poison_seen"]["right"]),
                      r["poison_seen"]["n_felt"], shr(r["poison_unseen"]["right"]), r["poison_unseen"]["n_felt"]),
                  flush=True)
            continue
        if args.what == "click":
            r = click(run, args.step, device=args.device)
            if r is None:
                print("%-14s %7d  (no poison clicks)" % (name(run), args.step))
                continue
            os.makedirs(os.path.join(run, "eval"), exist_ok=True)
            with open(os.path.join(run, "eval", "ckpt_%08d_click.json" % args.step), "w") as f:
                json.dump(r, f)
            parts = []
            for k in ("food", "poison", "background"):
                for c in ("click", "stay"):
                    v = r["%s_%s" % (k, c)]
                    parts.append("%s/%s %.3f (true %.3f%s)" % (k, c, v["predicted"], v["true"],
                                 ", imagined %.3f" % v["imagined"] if "imagined" in v else ""))
            fc, fs = r["food_click"], r["food_stay"]
            red = ("" if fc["red_predicted"] is None else
                   ";  circle redness on a food click %.0f (true %.0f), staying %.0f (true %.0f)" % (
                       fc["red_predicted"], fc["red_true"], fs["red_predicted"], fs["red_true"]))
            print("%-14s %7d  next hunger: %s%s" % (name(run), args.step, "  ".join(parts), red), flush=True)
            continue
        if args.what == "contact":
            r = contact(run, args.step)
            if r is None:
                print("%-14s %7d  (no hazards)" % (name(run), args.step))
                continue
            os.makedirs(os.path.join(run, "eval"), exist_ok=True)
            with open(os.path.join(run, "eval", "ckpt_%08d_contact.json" % args.step), "w") as f:
                json.dump(r, f)
            probes = ("from v %.2f, from z %.2f" % (r["r2_v"], r["r2_z"]) if "r2_v" in r else
                      "from z (the policy's token) %.2f, move token %.2f, drives token %.2f, cursor's patch %.2f, "
                      "patch mean %.2f (z vs C++ %.0e)" % (r["r2_z"], r.get("r2_move", r["r2_z"]), r["r2_drives"],
                                                           r["r2_cursor_patch"], r["r2_patch_mean"],
                                                           r["z_max_abs_diff"]))
            print("%-14s %7d  inside a hazard: R^2 %s; drives head next hunger inside %.3f, outside %.3f (world %.3f, "
                  "%.3f)" % (name(run), args.step, probes, r["drives_head_inside"], r["drives_head_outside"],
                             r["true_inside"], r["true_outside"]), flush=True)
            continue
        r = hazard(run, args.step, device=args.device)
        if r is None:
            print("%-14s %7d  (no hazards or no latent dynamics)" % (name(run), args.step))
            continue
        os.makedirs(os.path.join(run, "eval"), exist_ok=True)
        with open(os.path.join(run, "eval", "ckpt_%08d_hazard.json" % args.step), "w") as f:
            json.dump(r, f)
        print("%-14s %7d  hunger rise over 12 still steps: inside a hazard imagined %+.3f (true %+.3f), outside "
              "imagined %+.3f (true %+.3f); extra imagined %+.3f, true %+.3f" % (
                  name(run), args.step, r["inside"]["imagined_rise"], r["inside"]["true_rise"],
                  r["outside"]["imagined_rise"], r["outside"]["true_rise"], r["imagined_extra"], r["true_extra"]),
              flush=True)


if __name__ == "__main__":
    main()

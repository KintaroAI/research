"""PyTorch reference of the C++ visual–motor agent (testing only — the agent itself is C++).

Rebuilds the model from the flat C++ parameter vector and recomputes the forward pass, PPO loss,
prediction losses and GAE exactly as src/vm_agent.cpp defines them, so tests can compare loss values
and every gradient tensor (float64 here vs float32 in C++).
"""
import math

import numpy as np
import torch
import torch.nn.functional as F

LS_MIN, LS_MAX = -5.0, 1.0


def param_shapes(cfg):
    d, L, K = cfg["d_model"], cfg["n_layer"], cfg["ctx"]
    c1, c2, c3, feat, dff = cfg["c1"], cfg["c2"], cfg["c3"], cfg["feat"], cfg["d_ff"]
    h3 = cfg["img"] // 8
    p0 = cfg["pred_img"] // 4
    pc1, pc2, ph = cfg["pc1"], cfg["pc2"], cfg["pred_hidden"]
    in_dim = feat + 6
    co = 1 if cfg.get("coord", 0) else 0
    pb = cfg.get("pb", 0) if co else 0
    shapes = {
        "enc_c1w": (c1, 3 + 2 * co, 4, 4), "enc_c1b": (c1,), "enc_c2w": (c2, c1, 4, 4), "enc_c2b": (c2,),
        "enc_c3w": (c3, c2, 4, 4), "enc_c3b": (c3,), "enc_fcw": (feat, c3 * h3 * h3), "enc_fcb": (feat,),
        "in_w": (d, in_dim), "in_b": (d,), "wpe": (K, d),
        "ln1w": (L, d), "ln1b": (L, d), "qkvw": (L, 3 * d, d), "qkvb": (L, 3 * d),
        "attprojw": (L, d, d), "attprojb": (L, d), "ln2w": (L, d), "ln2b": (L, d),
        "fcw": (L, dff, d), "fcb": (L, dff), "fcprojw": (L, d, dff), "fcprojb": (L, d),
        "lnfw": (d,), "lnfb": (d,), "pi_w": (3, cfg.get("pi_hidden", 0) or d), "pi_b": (3,),
        "pi_h_w": (cfg.get("pi_hidden", 0), d), "pi_h_b": (cfg.get("pi_hidden", 0),),
        "logstd": (2,), "v_w": (d,), "v_b": (1,),
        "pr1w": (ph, d + 3), "pr1b": (ph,), "pr2w": (pc2 * p0 * p0, ph), "pr2b": (pc2 * p0 * p0,),
        "pbw": (pb, ph), "pbb": (pb,),
        "dc1w": (pc2 + pb + 2 * co + (c3 + 3 if cfg.get("pred_skip", 0) else 0), pc1, 4, 4), "dc1b": (pc1,),
        "dc2w": (pc1 + 2 * co + (c2 + 3 if cfg.get("pred_skip", 0) else 0), 3, 4, 4), "dc2b": (3,),
        "drw": (2, ph), "drb": (2,),
        "iv1w": (cfg.get("inv_hidden", 0), 2 * feat), "iv1b": (cfg.get("inv_hidden", 0),),
        "iv2w": (2 if cfg.get("inv_hidden", 0) else 0, cfg.get("inv_hidden", 0)),
        "iv2b": (2 if cfg.get("inv_hidden", 0) else 0,),
        **dyn_shapes(cfg),
        **patch_shapes(cfg),
        **uni_shapes(cfg),
    }
    if cfg.get("patch", 0):   # no CNN encoder or decoder; pr2w outputs one patch's cell
        cell = cfg["pred_img"] // (cfg["img"] // cfg["patch"])
        for k in ("enc_c1w", "enc_c1b", "enc_c2w", "enc_c2b", "enc_c3w", "enc_c3b", "enc_fcw", "enc_fcb", "in_w",
                  "in_b", "pbw", "pbb", "dc1w", "dc1b", "dc2w", "dc2b"):
            shapes[k] = (0,)
        shapes["pr2w"], shapes["pr2b"] = (cell * cell * 3, ph), (cell * cell * 3,)
    if uni_on(cfg):   # the universal model: no per-type heads or embeddings
        for k in ("pi_w", "pi_b", "pi_h_w", "pi_h_b", "v_w", "v_b", "pr1w", "pr1b", "pr2w", "pr2b", "drw", "drb",
                  "pt_w", "pt_b", "pos2d", "tok_type", "drv_w", "drv_b", "mv_w", "mv_b", "cmd_w", "cmd_b"):
            shapes[k] = (0,)
    return shapes


def uni_on(cfg):
    return 1 if cfg.get("patch", 0) and cfg.get("universal", 0) else 0


def uni_dims(cfg):
    """(tokens per step, values per token, bins) of the universal model."""
    return len(uni_slot_table(cfg)["kind"]), cfg["patch"] ** 2 * (1 if cfg.get("gray", 0) else 3), cfg["bins"]


def uni_has_patch(cfg, r, c):
    """Patch (r, c) is a token (vm_uni_has_patch): not with sum_only (task 00054); with band_n (task 00057), only in
    band band_k of the rows (band_cols: of the columns)."""
    if cfg.get("sum_only", 0):
        return False
    n = cfg.get("band_n", 0)
    if n < 2:
        return True
    return (c if cfg.get("band_cols", 0) else r) // (cfg["img"] // cfg["patch"] // n) == cfg.get("band_k", 0)


def uni_slot_table(cfg):
    """The slots of a step (tasks 00047, 00048): arrays [nt] kind (0 the body, 1 a patch, 2 a summary, 3 the move),
    index (the patch, row-major, or the summary), level (patches 0) and r, c (the position on its level's grid; uni_sum's
    summaries: band, 0). The patches go row by row; each summary right after the last patch of its block (uni_sum: a band
    of rows; uni_pyr: a 2^l x 2^l block at level l = 1..uni_pyr, finer levels first)."""
    g, R, L = cfg["img"] // cfg["patch"], cfg.get("uni_sum", 0), cfg.get("uni_pyr", 0)
    rows = [(0, 0, 0, 0, 0)]
    ns = 0
    for r in range(g):
        for c in range(g):
            if uni_has_patch(cfg, r, c):   # not with sum_only (task 00054); band_n: one band (task 00057)
                rows.append((1, r * g + c, 0, r, c))
            if R > 0 and c == g - 1 and (r + 1) % R == 0:
                rows.append((2, ns, 1, r // R, 0))
                ns += 1
            for lv in range(1, L + 1):
                b = 2 ** lv
                if (r + 1) % b == 0 and (c + 1) % b == 0:
                    rows.append((2, ns, lv, r // b, c // b))
                    ns += 1
    rows.append((3, 0, 0, 0, 0))
    t = np.array(rows)
    return dict(kind=t[:, 0], index=t[:, 1], level=t[:, 2], r=t[:, 3], c=t[:, 4])


def uni_slots(cfg):
    """(kind [nt], index [nt]) of uni_slot_table."""
    t = uni_slot_table(cfg)
    return t["kind"], t["index"]


def attn_allow(cfg):
    """The attention regions (task 00044; vm_attn_allow in vm_agent.h): allow [nt, nt] bool, a token of slot q may attend
    to one of slot k; None when attention is not split. Slots per step: the body, the patches row by row, the move. The
    patch rows are cut into attn_split bands, each reaching attn_overlap rows past its edges; two patches meet if their
    rows share a band; the body and the move meet every token, and with attn_pred_global also the last patch (it
    predicts the move). attn_cols: the bands are of the grid's columns instead of its rows. attn_checker (task 00046):
    no bands, patch (row r, column c) in region (r + c) % attn_split. attn_win (task 00047; directional, allow[q, k]: q
    attends to k): two patches within attn_win rows and columns; a patch attends to every summary, a summary to the
    patches of its band and every summary; the body and the move to and from everything. attn_split with uni_pyr
    (task 00055): halves sharing the summaries; every token reads every summary, the body and the move, and a patch
    only if it shares a band (a summary is in every band its block spans)."""
    S, O =cfg.get("attn_split", 0), cfg.get("attn_overlap", 0)
    if cfg.get("universal") and cfg.get("attn_win", 0) > 0:   # local attention with summaries (tasks 00047, 00048)
        Wn, R, g = cfg["attn_win"], cfg.get("uni_sum", 0), cfg["img"] // cfg["patch"]
        t = uni_slot_table(cfg)
        kind, index, lev, rr, cc = t["kind"], t["index"], t["level"], t["r"], t["c"]
        nt = len(kind)
        allow = np.ones((nt, nt), bool)
        for q in range(nt):
            for k in range(nt):
                if kind[q] in (0, 3) or kind[k] in (0, 3):
                    continue                                    # body and move: to and from everything
                if kind[q] == 1 and kind[k] == 1:
                    allow[q, k] = abs(index[q] // g - index[k] // g) <= Wn and abs(index[q] % g - index[k] % g) <= Wn
                elif R > 0:                                     # row summaries: a summary reads its band's patches
                    allow[q, k] = not (kind[q] == 2 and kind[k] == 1) or index[k] // g // R == index[q]
                elif kind[k] == 2:                              # the pyramid: same or coarser level, within Wn there
                    d = lev[k] - lev[q]
                    allow[q, k] = d >= 0 and abs(rr[q] // 2 ** d - rr[k]) <= Wn and abs(cc[q] // 2 ** d - cc[k]) <= Wn
                else:                                           # a summary reads every patch of its block
                    allow[q, k] = rr[k] // 2 ** lev[q] == rr[q] and cc[k] // 2 ** lev[q] == cc[q]
        return allow
    if cfg.get("universal") and S > 1 and cfg.get("uni_pyr", 0) > 0:   # halves sharing the summaries (task 00055)
        t = uni_slot_table(cfg)
        band = cfg["img"] // cfg["patch"] // S
        pos = t["c"] if cfg.get("attn_cols", 0) else t["r"]
        size = np.where(t["kind"] == 2, 2 ** t["level"], 1)              # a summary spans its block's bands
        reg = [set(range(p * b // band, ((p + 1) * b - 1) // band + 1)) if k in (1, 2) else set(range(S))
               for k, p, b in zip(t["kind"], pos, size)]
        return np.array([[t["kind"][k] != 1 or bool(reg[q] & reg[k]) for k in range(len(reg))] for q in range(len(reg))])
    if not cfg.get("universal") or S <= 1:
        return None
    g = cfg["img"] // cfg["patch"]
    nt, band = g * g + 2, g // S
    bands = [(s * band - O, (s + 1) * band - 1 + O) for s in range(S)]
    allow = np.ones((nt, nt), bool)
    last = nt - 2 if cfg.get("attn_pred_global", 0) else nt - 1   # slots from here on are global
    for q in range(1, last):
        for k in range(1, last):
            if cfg.get("attn_checker", 0):
                allow[q, k] = sum(divmod(q - 1, g)) % S == sum(divmod(k - 1, g)) % S
                continue
            rq, rk = ((q - 1) % g, (k - 1) % g) if cfg.get("attn_cols", 0) else ((q - 1) // g, (k - 1) // g)
            allow[q, k] = any(lo <= rq <= hi and lo <= rk <= hi for lo, hi in bands)
    return allow


def uni_shapes(cfg):
    d, on = cfg["d_model"], uni_on(cfg)
    if not on:
        return {"uni_in_w": (0,), "uni_in_b": (0,), "uni_slot": (0,), "uni_out_w": (0,), "uni_out_b": (0,)}
    nt, V, K = uni_dims(cfg)
    VI = V * K if cfg.get("in_bins", 0) else V                      # bin-embedding input (task 00036)
    return {"uni_in_w": (d, VI), "uni_in_b": (d,), "uni_slot": (nt, d), "uni_out_w": (V * K, d), "uni_out_b": (V * K,)}


def patch_dims(cfg):
    """(patch side, grid side, patches, predicted cell side); zeros without patch tokens."""
    ps = cfg.get("patch", 0)
    if not ps:
        return 0, 0, 0, 0
    g = cfg["img"] // ps
    return ps, g, g * g, cfg["pred_img"] // g


def patch_shapes(cfg):
    d = cfg["d_model"]
    ps, g, npch, _ = patch_dims(cfg)
    on = 1 if ps else 0
    ac = act_on(cfg)
    ph, cfx = cfg["pred_hidden"], ACT_CG * ACT_CG * 3
    return {"pt_w": (d * on, ps * ps * 3), "pt_b": (d * on,), "pos2d": (npch, d), "tok_type": ((4 + ac) * on, d),
            "drv_w": (d * on, 2), "drv_b": (d * on,), "mv_w": (d * on, 3), "mv_b": (d * on,),
            "cmd_w": (d * on, 1), "cmd_b": (d * on,),
            "act_e": (d * ac,), "ax1w": (ph * ac, d + 3), "ax1b": (ph * ac,), "axdw": (2 * ac, ph), "axdb": (2 * ac,),
            "axfw": (cfx * ac, ph), "axfb": (cfx * ac,)}


ACT_CG = 8   # act token: side of the coarse next frame it predicts


def act_on(cfg):
    return 1 if cfg.get("patch", 0) and cfg.get("act_token", 0) else 0


def dyn_shapes(cfg):
    dh, feat = cfg.get("dyn_hidden", 0), cfg["feat"]
    rpx = (cfg["pred_img"] // 2) ** 2 * 3
    on = 1 if dh else 0
    return {"dt1w": (dh, feat + 3), "dt1b": (dh,), "dt2w": (feat * on, dh), "dt2b": (feat * on,),
            "dh1w": (dh, feat + 4), "dh1b": (dh,), "dh2w": (on, dh), "dh2b": (on,),
            "dx1w": (dh, feat), "dx1b": (dh,), "dx2w": (rpx * on, dh), "dx2b": (rpx * on,)}


def executed(a):
    """Raw actions [..., 3] -> executed (clipped dx, dy; click 0/1), float64 numpy."""
    a = np.asarray(a, np.float64)
    return np.stack([np.clip(a[..., 0], -1, 1), np.clip(a[..., 1], -1, 1), (a[..., 2] > 0.5).astype(np.float64)], -1)


def dyn_step(P, s, h, a):
    """One latent-dynamics step: latent s [..., feat], hunger h [...], executed action a [..., 3] ->
    (next latent, predicted next hunger)."""
    hh = torch.sigmoid(F.linear(F.relu(F.linear(torch.cat([s, h[..., None], a], -1), P["dh1w"], P["dh1b"])),
                                P["dh2w"], P["dh2b"]))[..., 0]
    s2 = s + F.linear(F.relu(F.linear(torch.cat([s, a], -1), P["dt1w"], P["dt1b"])), P["dt2w"], P["dt2b"])
    return s2, hh


def dyn_decode(P, s):
    """Coarse decoder: latent [..., feat] -> flattened HWC frame [..., (pred_img/2)^2 * 3]."""
    return F.linear(F.relu(F.linear(s, P["dx1w"], P["dx1b"])), P["dx2w"], P["dx2b"])


def drive_weights(pc, batch):
    """[S, T] weight of each step's drives loss: 1 + drive_changed_weight where hunger changes by more than
    drive_change_eps over the step (C++ drive_weight)."""
    w = pc.get("drive_changed_weight", 0.0)
    ch = np.abs(np.asarray(batch["next_drives"], np.float32)[..., 0] - np.asarray(batch["drives"], np.float32)[..., 0])
    return torch.tensor(np.where((w > 0) & (ch > np.float32(pc.get("drive_change_eps", 0.01))), 1.0 + w, 1.0))


def dyn_losses(P, cfg, pc, batch, vf):
    """Latent-dynamics losses (sums over chain steps / tokens divided by S*T), as in C++ seq_dyn.

    The heads are trained on top of the encoder: v is detached everywhere. Chains start at every token t; step k
    uses the action of token j = t + k - 1 and stops at the horizon, at the end of the sequence or after a terminal
    step. Latent targets are the (detached) features of token j + 1 when
    it continues the episode; hunger targets next_drives[j]; decoder targets next_small[j] downsampled by 2, plus
    every token's own frame for its real latent. With dyn_h_horizon > 0, only the first dyn_h_horizon steps of a
    chain train the hunger head.
    """
    vf = vf.detach()
    S, T, _ = vf.shape
    dt = vf.dtype
    H = min(pc["dyn_horizon"], T)
    Hh = pc.get("dyn_h_horizon", 0)          # chain steps that train the hunger head (0 = all)
    es = np.asarray(batch["ep_start"], bool)
    ex = torch.tensor(executed(batch["action"]), dtype=dt)
    drv = torch.tensor(np.asarray(batch["drives"], np.float64), dtype=dt)
    nd = torch.tensor(np.asarray(batch["next_drives"], np.float64), dtype=dt)
    tgt_next = torch.tensor(downsample_u8(np.asarray(batch["next_small"]), 2), dtype=dt).reshape(S, T, -1) / 255.0
    cur = downsample_u8(downsample_u8(np.asarray(batch["images"]), cfg["img"] // cfg["pred_img"]), 2)
    tgt_cur = torch.tensor(cur, dtype=dt).reshape(S, T, -1) / 255.0
    lat = hun = rec = torch.zeros((), dtype=dt)
    wd = drive_weights(pc, batch)
    for si in range(S):
        for t in range(T):
            s = vf[si, t]
            for k in range(1, H + 1):
                j = t + k - 1
                if j > T - 1 or (k > 1 and es[si, j]):
                    break
                s, hh = dyn_step(P, s, drv[si, j, 0], ex[si, j])
                if j + 1 <= T - 1 and not es[si, j + 1]:
                    lat = lat + ((s - vf[si, j + 1].detach()) ** 2).mean()
                if Hh <= 0 or k <= Hh:
                    hun = hun + wd[si, j] * (hh - nd[si, j, 0]) ** 2
                rec = rec + ((dyn_decode(P, s) - tgt_next[si, j]) ** 2).mean()
            rec = rec + ((dyn_decode(P, vf[si, t]) - tgt_cur[si, t]) ** 2).mean()
    n = S * T
    return lat / n, hun / n, rec / n


def params_from_flat(flat, layout, cfg, dtype=torch.float64):
    shapes = param_shapes(cfg)
    P = {}
    for name, off, size in layout:
        t = torch.tensor(np.asarray(flat[off:off + size]), dtype=dtype).reshape(shapes[name])
        t.requires_grad_(True)
        P[name] = t
    return P


def coord_maps(n, S, dtype):
    """[n, 2, S, S] constant x / y pixel-center coordinates in (-1, 1) (matches C++ fill_coords)."""
    c = (torch.arange(S, dtype=torch.float32) + 0.5) / S * 2 - 1
    xs = c[None, :].expand(S, S)
    ys = c[:, None].expand(S, S)
    return torch.stack([xs, ys])[None].expand(n, 2, S, S).to(dtype)


def segment_starts(ep_start):
    """ep_start [S,T] bool -> ss [S,T] (context start index for each position)."""
    S, T = ep_start.shape
    ss = np.zeros((S, T), np.int64)
    for s in range(S):
        cur = 0
        for t in range(T):
            if t == 0 or ep_start[s, t]:
                cur = t
            ss[s, t] = cur
    return ss


def downsample_u8(img, f):
    """Box-average u8 [...,H,W,3] by factor f with rounding half up (matches C++)."""
    H, W = img.shape[-3], img.shape[-2]
    x = img.astype(np.int64).reshape(img.shape[:-3] + (H // f, f, W // f, f, 3)).sum(axis=(-4, -2))
    return ((x + (f * f) // 2) // (f * f)).astype(np.uint8)


def policy(P, cfg, z):
    """Policy logits [..., 3] (mean dx, dy before tanh; click logit), through a ReLU hidden layer if pi_hidden."""
    if cfg.get("pi_hidden", 0):
        z = F.relu(F.linear(z, P["pi_h_w"], P["pi_h_b"]))
    return F.linear(z, P["pi_w"], P["pi_b"])


def transformer(P, cfg, h, mask):
    """The Transformer layers and final LayerNorm on token inputs h [S,N,d]; mask [S,1,N,N] (query, key)."""
    S, N = h.shape[:2]
    d, L, nh = cfg["d_model"], cfg["n_layer"], cfg["n_head"]
    hs = d // nh
    for l in range(L):
        y = F.layer_norm(h, (d,), P["ln1w"][l], P["ln1b"][l], eps=1e-5)
        qkv = F.linear(y, P["qkvw"][l], P["qkvb"][l])
        q, k, v = qkv.split(d, dim=-1)
        q = q.reshape(S, N, nh, hs).transpose(1, 2)
        k = k.reshape(S, N, nh, hs).transpose(1, 2)
        v = v.reshape(S, N, nh, hs).transpose(1, 2)
        att = (q @ k.transpose(-1, -2)) / math.sqrt(hs)
        att = att.masked_fill(~mask, float("-inf")).softmax(dim=-1)
        y = (att @ v).transpose(1, 2).reshape(S, N, d)
        h = h + F.linear(y, P["attprojw"][l], P["attprojb"][l])
        y = F.layer_norm(h, (d,), P["ln2w"][l], P["ln2b"][l], eps=1e-5)
        y = F.gelu(F.linear(y, P["fcw"][l], P["fcb"][l]), approximate="tanh")
        h = h + F.linear(y, P["fcprojw"][l], P["fcprojb"][l])
    return F.layer_norm(h, (d,), P["lnfw"], P["lnfb"], eps=1e-5)


def forward_patch(P, cfg, images, drives, prev_action, prev_reward, ep_start, pred_actions=None):
    """Patch tokens (cfg['patch'] > 0), as seq_fwd_patch in C++. Per step: the patches row by row, then the drives,
    previous-move and command tokens. A token attends to every token of its own step and of the earlier steps of its
    context block. Returns z (the move tokens' outputs [S,T,d]), h (all token outputs [S,T,ntok,d]), pi, value."""
    dt = P["pt_w"].dtype
    S, T = images.shape[:2]
    d = cfg["d_model"]
    ps, g, npch, _ = patch_dims(cfg)
    ac = act_on(cfg)
    nt = npch + 3 + ac
    x = torch.tensor(images, dtype=dt).reshape(S, T, g, ps, g, ps, 3) / 255.0
    x = x.permute(0, 1, 2, 4, 3, 5, 6).reshape(S, T, npch, ps * ps * 3)          # (y, x, c) within each patch
    tp = F.linear(x, P["pt_w"], P["pt_b"]) + P["pos2d"] + P["tok_type"][0]
    cmd = torch.tensor(prev_reward, dtype=dt)[..., None] * float(cfg["use_prev_reward"])
    td = F.linear(torch.tensor(drives, dtype=dt), P["drv_w"], P["drv_b"]) + P["tok_type"][1]
    tm = F.linear(torch.tensor(prev_action, dtype=dt), P["mv_w"], P["mv_b"]) + P["tok_type"][2]
    tc = F.linear(cmd, P["cmd_w"], P["cmd_b"]) + P["tok_type"][3]
    ss = segment_starts(np.asarray(ep_start, bool))
    if cfg.get("one_frame", 0):
        ss = np.broadcast_to(np.arange(T)[None, :], ss.shape).copy()
    pos = np.arange(T)[None, :] - ss
    toks = [tp, td[:, :, None], tm[:, :, None], tc[:, :, None]]
    if ac:   # the act token: a learned query
        toks.append((P["act_e"] + P["tok_type"][4]).expand(S, T, 1, d))
    h = torch.cat(toks, dim=2)
    h = (h + P["wpe"][torch.tensor(pos)][:, :, None]).reshape(S, T * nt, d)
    step = np.repeat(np.arange(T), nt)                                             # token -> its step
    first = np.repeat(ss, nt, axis=1)                                              # [S, T*nt]: first step it sees
    mask = (step[None, None, :] <= step[None, :, None]) & (step[None, None, :] >= first[:, :, None])
    hz = transformer(P, cfg, h, torch.tensor(mask)[:, None]).reshape(S, T, nt, d)
    z = hz[:, :, npch + 3 if ac else npch + 1]   # the policy reads the act token, else the move token
    out = dict(z=z, h=hz, pi=policy(P, cfg, z), value=z @ P["v_w"] + P["v_b"][0])
    if pred_actions is not None:
        out.update(predict_patch(P, cfg, hz, pred_actions, images))
    return out


def predict_patch(P, cfg, h, actions, images=None):
    """Patch tokens: each patch token predicts its cell of the next frame, the drives token the next drives, from
    [token output, executed action]. h: token outputs [S,T,ntok,d]."""
    dt = h.dtype
    S, T = h.shape[:2]
    ps, g, npch, cell = patch_dims(cfg)
    a = np.asarray(actions, np.float64)
    ae = np.stack([np.clip(a[..., 0], -1, 1), np.clip(a[..., 1], -1, 1), (a[..., 2] > 0.5).astype(np.float64)], -1)
    aet = torch.tensor(ae, dtype=dt)[:, :, None].expand(-1, -1, npch + 1, -1)
    h1 = F.relu(F.linear(torch.cat([h[:, :, :npch + 1], aet], dim=-1), P["pr1w"], P["pr1b"]))
    dl = F.linear(h1[:, :, :npch], P["pr2w"], P["pr2b"])                          # [S,T,np,cell*cell*3], (y, x, c)
    p = cfg["pred_img"]
    out = dl.reshape(S, T, g, g, cell, cell, 3).permute(0, 1, 6, 2, 4, 3, 5).reshape(S, T, 3, p, p)
    if cfg.get("pred_residual", 0):
        cur = downsample_u8(np.asarray(images), cfg["img"] // p)                    # [S,T,p,p,3]
        frames = out + torch.tensor(cur, dtype=dt).permute(0, 1, 4, 2, 3) / 255.0
    else:
        frames = torch.sigmoid(out)
    res = dict(pred_frames=frames, pred_drives=torch.sigmoid(F.linear(h1[:, :, npch], P["drw"], P["drb"])))
    if act_on(cfg):   # the act token's auxiliary heads: next drives, coarse next frame (residual, HWC)
        ha = F.relu(F.linear(torch.cat([h[:, :, npch + 3], aet[:, :, 0]], dim=-1), P["ax1w"], P["ax1b"]))
        cc = downsample_u8(downsample_u8(np.asarray(images), cfg["img"] // p), p // ACT_CG).reshape(S, T, -1)
        res.update(act_drives=torch.sigmoid(F.linear(ha, P["axdw"], P["axdb"])),
                   act_frame=F.linear(ha, P["axfw"], P["axfb"]) + torch.tensor(cc, dtype=dt) / 255.0)
    return res


def forward(P, cfg, images, drives, prev_action, prev_reward, ep_start, pred_actions=None):
    """images u8 [S,T,H,W,3]; returns dict of torch tensors (z, pi, value, pred_frames CHW, pred_drives)."""
    if cfg.get("patch", 0):
        return forward_patch(P, cfg, images, drives, prev_action, prev_reward, ep_start, pred_actions)
    dt = P["in_w"].dtype
    S, T = images.shape[:2]
    x = torch.tensor(images, dtype=dt).reshape(S * T, *images.shape[2:]).permute(0, 3, 1, 2) / 255.0
    if cfg.get("coord", 0):
        x = torch.cat([x, coord_maps(S * T, x.shape[-1], dt)], dim=1)
    a = F.relu(F.conv2d(x, P["enc_c1w"], P["enc_c1b"], stride=2, padding=1))
    a = F.relu(F.conv2d(a, P["enc_c2w"], P["enc_c2b"], stride=2, padding=1))
    a2 = a
    a = F.relu(F.conv2d(a, P["enc_c3w"], P["enc_c3b"], stride=2, padding=1))
    a3 = a
    vf = F.relu(F.linear(a.flatten(1), P["enc_fcw"], P["enc_fcb"])).reshape(S, T, -1)
    pr = torch.tensor(prev_reward, dtype=dt)[..., None] * float(cfg["use_prev_reward"])
    tin = torch.cat([vf, torch.tensor(drives, dtype=dt), torch.tensor(prev_action, dtype=dt), pr], dim=-1)
    ss = segment_starts(np.asarray(ep_start, bool))
    if cfg.get("one_frame", 0):                 # every token is its own context: attends only to itself
        ss = np.broadcast_to(np.arange(T)[None, :], ss.shape).copy()
    pos = np.arange(T)[None, :] - ss
    h = F.linear(tin, P["in_w"], P["in_b"]) + P["wpe"][torch.tensor(pos)]
    t_idx = np.arange(T)
    mask = (t_idx[None, None, :] <= t_idx[None, :, None]) & (t_idx[None, None, :] >= ss[:, :, None])  # [S,T,T]
    z = transformer(P, cfg, h, torch.tensor(mask)[:, None, :, :])
    out = dict(z=z, pi=policy(P, cfg, z), value=z @ P["v_w"] + P["v_b"][0], a2=a2, a3=a3, vf=vf)
    if pred_actions is not None:
        out.update(predict(P, cfg, z, pred_actions, images, a2, a3))
    return out


def predict(P, cfg, z, actions, images=None, a2=None, a3=None):
    """Prediction head on z [S,T,d] conditioned on actions [S,T,3] (raw; clipped like the world).

    images (u8 [S,T,H,W,3]) are required when cfg['pred_residual']: prediction = current frame + output.
    """
    dt = z.dtype
    S, T = z.shape[:2]
    a = np.asarray(actions, np.float64)
    ae = np.stack([np.clip(a[..., 0], -1, 1), np.clip(a[..., 1], -1, 1), (a[..., 2] > 0.5).astype(np.float64)], -1)
    pin = torch.cat([z, torch.tensor(ae, dtype=dt)], dim=-1)
    h1 = F.relu(F.linear(pin, P["pr1w"], P["pr1b"]))
    p0 = cfg["pred_img"] // 4
    h2 = F.relu(F.linear(h1, P["pr2w"], P["pr2b"])).reshape(S * T, cfg["pc2"], p0, p0)
    act = torch.tensor(ae.reshape(S * T, 3, 1, 1), dtype=dt)
    parts = [h2]
    if cfg.get("coord", 0):
        if cfg.get("pb", 0) > 0:
            ub = F.relu(F.linear(h1, P["pbw"], P["pbb"])).reshape(S * T, -1, 1, 1)
            parts.append(ub.expand(-1, -1, p0, p0))
        parts.append(coord_maps(S * T, p0, dt))
    if cfg.get("pred_skip", 0):
        parts.append(a3)
        parts.append(act.expand(-1, -1, p0, p0))
    h2 = torch.cat(parts, dim=1)
    d1 = F.relu(F.conv_transpose2d(h2, P["dc1w"], P["dc1b"], stride=2, padding=1))
    parts = [d1]
    if cfg.get("coord", 0):
        parts.append(coord_maps(S * T, d1.shape[-1], dt))
    if cfg.get("pred_skip", 0):
        q = d1.shape[-1]
        parts.append(a2)
        parts.append(act.expand(-1, -1, q, q))
    d1 = torch.cat(parts, dim=1)
    out = F.conv_transpose2d(d1, P["dc2w"], P["dc2b"], stride=2, padding=1)
    p = cfg["pred_img"]
    if cfg.get("pred_residual", 0):
        cur = downsample_u8(np.asarray(images), cfg["img"] // p).reshape(S * T, p, p, 3)
        frames = out + torch.tensor(cur, dtype=dt).permute(0, 3, 1, 2) / 255.0
    else:
        frames = torch.sigmoid(out)
    return dict(pred_frames=frames.reshape(S, T, 3, p, p), pred_drives=torch.sigmoid(F.linear(h1, P["drw"], P["drb"])))


def gae(reward, value, done, last_value, gamma, lam, reward_scale):
    """[T,N] arrays -> adv, ret (float64), matching compute_gae in C++."""
    T, N = reward.shape
    adv = np.zeros((T, N))
    g = np.zeros(N)
    for t in reversed(range(T)):
        nv = last_value if t == T - 1 else value[t + 1]
        nonterm = 1.0 - done[t]
        delta = reward[t] * reward_scale + gamma * nv * nonterm - value[t]
        g = delta + gamma * lam * nonterm * g
        adv[t] = g
    return adv, adv + value


def uni_values(cfg, images, drives, actions):
    """Token values [S, T, ntok, V] in [0, 1] and the used counts [ntok]: body (hunger, restlessness), the patches row
    by row ((y, x, c) within a patch, pixels / 255), the move ((dx+1)/2, (dy+1)/2, click)."""
    S, T = images.shape[:2]
    nt, V, _ = uni_dims(cfg)
    ps, g = cfg["patch"], cfg["img"] // cfg["patch"]
    v = np.zeros((S, T, nt, V))
    x = np.asarray(images, np.float64).reshape(S, T, g, ps, g, ps, 3) / 255.0
    if cfg.get("gray", 0):   # one channel per pixel: the first
        x = x[..., :1]
    kind, index = uni_slots(cfg)
    if (kind == 1).any():   # sum_only (task 00054): no patch tokens; band_n (task 00057): one band's, by index
        v[:, :, kind == 1] = x.transpose(0, 1, 2, 4, 3, 5, 6).reshape(S, T, g * g, V)[:, :, index[kind == 1]]
    v[:, :, 0, :2] = np.asarray(drives, np.float64)
    a = np.asarray(actions, np.float64)
    v[:, :, nt - 1, 0] = (np.clip(a[..., 0], -1, 1) + 1) / 2
    v[:, :, nt - 1, 1] = (np.clip(a[..., 1], -1, 1) + 1) / 2
    v[:, :, nt - 1, 2] = (a[..., 2] > 0.5).astype(np.float64)
    used = np.full(nt, V)
    used[0], used[nt - 1] = 2, 3
    used[kind == 2] = 0                                                            # summary tokens: no values
    if cfg.get("sum_thumb", 0):   # task 00049: a summary's block, box-averaged to one patch (a thumbnail)
        v[:, :, kind == 2] = uni_thumbs(cfg, images)
        used[kind == 2] = V
    return v, used


def uni_thumbs(cfg, images):
    """Each summary's thumbnail [S, T, nsum, V] (task 00049): the pixels of its block (uni_sum: R * patch rows by the
    full width; uni_pyr: a 2^l * patch square) averaged over boxes to one patch, values (y, x, c) / 255."""
    S, T = images.shape[:2]
    ps, g = cfg["patch"], cfg["img"] // cfg["patch"]
    x = np.asarray(images, np.float64) / 255.0
    if cfg.get("gray", 0):
        x = x[..., :1]
    ch = x.shape[-1]
    t = uni_slot_table(cfg)
    out = []
    for k in np.flatnonzero(t["kind"] == 2):
        if cfg.get("uni_sum", 0):
            R = cfg["uni_sum"]
            y0, x0, fy, fx = t["r"][k] * R * ps, 0, R, g
        else:
            b = 2 ** t["level"][k]
            y0, x0, fy, fx = t["r"][k] * b * ps, t["c"][k] * b * ps, b, b
        blk = x[:, :, y0:y0 + fy * ps, x0:x0 + fx * ps]
        out.append(blk.reshape(S, T, ps, fy, ps, fx, ch).mean((3, 5)).reshape(S, T, ps * ps * ch))
    return np.stack(out, 2)


def forward_uni(P, cfg, images, drives, actions, ep_start):
    """The universal model (as seq_fwd_uni in C++). Returns h (token outputs [S, T*ntok, d]) and logits
    [S, T*ntok, V, K]: every token's prediction of the next token's values."""
    dt = P["uni_in_w"].dtype
    S, T = images.shape[:2]
    nt, V, K = uni_dims(cfg)
    v, used = uni_values(cfg, images, drives, actions)
    m = (np.arange(V)[None, :] < used[:, None]).astype(np.float64)               # [nt, V]
    if cfg.get("in_bins", 0):   # bin-embedding input (task 00036): each used value two-hot over the K bins
        x = torch.tensor(two_hot(v, K) * m[..., None], dtype=dt).reshape(S, T * nt, V * K)
    else:
        x = torch.tensor((v - 0.5) * m, dtype=dt).reshape(S, T * nt, V)
    ss = segment_starts(np.asarray(ep_start, bool))
    pos = np.arange(T)[None, :] - ss
    h = F.linear(x, P["uni_in_w"], P["uni_in_b"]).reshape(S, T, nt, -1) + P["uni_slot"][None, None]
    h = (h + P["wpe"][torch.tensor(pos)][:, :, None]).reshape(S, T * nt, -1)
    step = np.repeat(np.arange(T), nt)
    first = np.repeat(ss, nt, axis=1) * nt                                         # first token each token sees
    idx = np.arange(T * nt)
    mask = (idx[None, None, :] <= idx[None, :, None]) & (idx[None, None, :] >= first[:, :, None])
    allow = attn_allow(cfg)
    if allow is not None:   # attention regions (task 00044): by the slots of query and key
        mask = mask & allow[idx[:, None] % nt, idx[None, :] % nt][None]
    hz = transformer(P, cfg, h, torch.tensor(mask)[:, None])
    logits = F.linear(hz, P["uni_out_w"], P["uni_out_b"]).reshape(S, T * nt, V, K)
    return dict(h=hz, logits=logits, values=v, used=used, step=step)


def two_hot(v, K):
    """Two-hot targets [..., K] for values in [0, 1] (centres k / (K - 1))."""
    pos = np.clip(v, 0, 1) * (K - 1)
    lo = np.minimum(np.floor(pos).astype(np.int64), K - 2)
    w = pos - lo
    t = np.zeros(v.shape + (K,))
    np.put_along_axis(t, lo[..., None], (1 - w)[..., None], -1)
    np.put_along_axis(t, (lo + 1)[..., None], w[..., None], -1)
    return t


def uni_loss(P, cfg, batch):
    """The universal model's loss: every token predicts the next token's values (the move token: the drives after its
    step); two-hot cross-entropy, the mean over the target's used values, the mean over tokens. Optional
    batch["input_images"] / ["input_drives"] (self-forcing, task 00039): the tokens' inputs, the targets still from
    batch["images"] / ["drives"]."""
    dt = P["uni_in_w"].dtype
    out = forward_uni(P, cfg, batch.get("input_images", batch["images"]), batch.get("input_drives", batch["drives"]),
                      batch["action"], batch["ep_start"])
    S, T = batch["images"].shape[:2]
    nt, V, K = uni_dims(cfg)
    v, used = uni_values(cfg, batch["images"], batch["drives"], batch["action"])
    tgt = np.zeros((S, T, nt, V))
    tgt[:, :, :nt - 1] = v[:, :, 1:]                                               # the next token in the step
    tgt[:, :, nt - 1, :2] = np.asarray(batch["next_drives"], np.float64)           # the move: the next drives
    tu = np.concatenate([used[1:], [2]])                                           # values used by each target
    m = (np.arange(V)[None, :] < tu[:, None]).astype(np.float64)                   # [nt, V]
    logp = out["logits"].log_softmax(-1).reshape(S, T, nt, V, K)
    ce = -(torch.tensor(two_hot(tgt, K), dtype=dt) * logp).sum(-1)                # [S, T, nt, V]
    per_tok = (ce * torch.tensor(m, dtype=dt)).sum(-1) / torch.tensor(np.maximum(tu, 1), dtype=dt)
    has = tu > 0                                                                   # a summary next: no target
    return per_tok[:, :, has].mean(), dict(per_token=per_tok, logp=logp, tgt=tgt, used=tu)


def ppo_loss(P, cfg, pc, batch):
    """Total loss (mean per transition) for a batch of sequences, as in C++ seq_loss.

    batch: dict with [S,T,...] arrays: images, drives, prev_action, prev_reward, ep_start, action, old_logp,
    adv, ret, next_small (u8 HWC), next_drives.
    """
    if uni_on(cfg):
        loss, terms = uni_loss(P, cfg, batch)
        return loss, terms
    dt = P["in_w"].dtype
    have_pred = pc["pred_coef"] > 0 or pc["drive_coef"] > 0
    out = forward(P, cfg, batch["images"], batch["drives"], batch["prev_action"], batch["prev_reward"],
                  batch["ep_start"])
    z = out["z"]
    terms = {}
    total = torch.zeros((), dtype=dt)
    move_coef = pc.get("move_coef", 0.0)
    if pc["use_ppo"] or move_coef > 0:
        a = torch.tensor(batch["action"], dtype=dt)
        pi = policy(P, cfg, z.detach()) if pc.get("pi_detach", 0) else out["pi"]
        mu = torch.tanh(pi[..., :2])
        ls = torch.clamp(P["logstd"], LS_MIN, LS_MAX)
        sd = torch.exp(ls)
        zz = (a[..., :2] - mu) / sd
        logp = (-0.5 * zz * zz - ls - 0.5 * math.log(2 * math.pi)).sum(-1)
        l = pi[..., 2]
        c = a[..., 2] > 0.5
        logp = logp + torch.where(c, -F.softplus(-l), -F.softplus(l))
    if move_coef > 0:                           # next-move prediction: the move actually made
        total = total + move_coef * (-logp).mean()
        terms.update(move_nll=(-logp).mean().item())
    if pc["use_ppo"]:
        p = torch.sigmoid(l)
        ent = (0.5 + 0.5 * math.log(2 * math.pi) + ls).sum() + p * F.softplus(-l) + (1 - p) * F.softplus(l)
        A = torch.tensor(batch["adv"], dtype=dt)
        if pc["norm_adv"]:
            A = (A - A.mean()) / (A.std(unbiased=True) + 1e-8)
        ratio = torch.exp(torch.clamp(logp - torch.tensor(batch["old_logp"], dtype=dt), -20, 20))
        pg = -torch.minimum(ratio * A, torch.clamp(ratio, 1 - pc["clip_eps"], 1 + pc["clip_eps"]) * A)
        R = torch.tensor(batch["ret"], dtype=dt)
        vloss = 0.5 * (out["value"] - R) ** 2
        loss_t = pg + pc["vf_coef"] * vloss - pc["ent_coef"] * ent
        total = total + loss_t.mean()
        terms.update(pg=pg.mean().item(), v=vloss.mean().item(), ent=ent.mean().item())
    if have_pred:
        s = pc["pred_backbone_scale"]
        if cfg.get("patch", 0):
            hp = out["h"] if s == 1.0 else s * out["h"] + (1 - s) * out["h"].detach()
            pr = predict_patch(P, cfg, hp, batch["action"], batch["images"])
        else:
            zp = z if s == 1.0 else s * z + (1 - s) * z.detach()
            a2 = out["a2"] if s == 1.0 else s * out["a2"] + (1 - s) * out["a2"].detach()
            a3 = out["a3"] if s == 1.0 else s * out["a3"] + (1 - s) * out["a3"].detach()
            pr = predict(P, cfg, zp, batch["action"], batch["images"], a2, a3)
        tgt = torch.tensor(batch["next_small"], dtype=dt).permute(0, 1, 4, 2, 3) / 255.0
        f = cfg["img"] // cfg["pred_img"]
        cur = downsample_u8(np.asarray(batch["images"]), f)
        changed = (cur != np.asarray(batch["next_small"])).any(-1)                       # [S,T,p,p]
        w = torch.tensor(1.0 + pc.get("pred_changed_weight", 0.0) * changed, dtype=dt)[:, :, None]
        mse = (w * (pr["pred_frames"] - tgt) ** 2).mean(dim=(2, 3, 4))
        dmse = ((pr["pred_drives"] - torch.tensor(batch["next_drives"], dtype=dt)) ** 2).mean(-1)
        total = total + pc["pred_coef"] * mse.mean() + pc["drive_coef"] * (drive_weights(pc, batch) * dmse).mean()
        terms.update(pred=mse.mean().item(), drive=dmse.mean().item())
        if act_on(cfg):
            p = cfg["pred_img"]
            tc = downsample_u8(np.asarray(batch["next_small"]), p // ACT_CG).reshape(*tgt.shape[:2], -1)
            fm = ((pr["act_frame"] - torch.tensor(tc, dtype=dt) / 255.0) ** 2).mean(-1)
            am = ((pr["act_drives"] - torch.tensor(batch["next_drives"], dtype=dt)) ** 2).mean(-1)
            total = total + pc.get("act_coef", 0.0) * ((drive_weights(pc, batch) * am).mean() + fm.mean())
            terms.update(act_drive=am.mean().item(), act_frame=fm.mean().item())
    if pc.get("inv_coef", 0) > 0 and cfg.get("inv_hidden", 0) > 0:
        vf = out["vf"]
        S, T = vf.shape[:2]
        inp = torch.cat([vf[:, :-1], vf[:, 1:]], dim=-1)
        o = F.linear(F.relu(F.linear(inp, P["iv1w"], P["iv1b"])), P["iv2w"], P["iv2b"])
        tgt = torch.tensor(np.clip(np.asarray(batch["action"], np.float64)[:, :-1, :2], -1, 1), dtype=dt)
        valid = torch.tensor(~np.asarray(batch["ep_start"], bool)[:, 1:], dtype=dt)
        inv = (valid * 0.5 * ((o - tgt) ** 2).sum(-1)).sum() / (S * T)
        total = total + pc["inv_coef"] * inv
        terms.update(inv=inv.item())
    dyn_on = pc.get("dyn_coef", 0) > 0 or pc.get("dyn_h_coef", 0) > 0 or pc.get("dyn_x_coef", 0) > 0
    if cfg.get("dyn_hidden", 0) > 0 and pc.get("dyn_horizon", 0) > 0 and dyn_on:
        lat, hun, rec = dyn_losses(P, cfg, pc, batch, out["vf"])
        total = total + pc["dyn_coef"] * lat + pc["dyn_h_coef"] * hun + pc["dyn_x_coef"] * rec
        terms.update(dyn=lat.item(), dyn_hunger=hun.item(), dyn_recon=rec.item())
    return total, terms

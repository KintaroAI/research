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
    return {
        "enc_c1w": (c1, 3 + 2 * co, 4, 4), "enc_c1b": (c1,), "enc_c2w": (c2, c1, 4, 4), "enc_c2b": (c2,),
        "enc_c3w": (c3, c2, 4, 4), "enc_c3b": (c3,), "enc_fcw": (feat, c3 * h3 * h3), "enc_fcb": (feat,),
        "in_w": (d, in_dim), "in_b": (d,), "wpe": (K, d),
        "ln1w": (L, d), "ln1b": (L, d), "qkvw": (L, 3 * d, d), "qkvb": (L, 3 * d),
        "attprojw": (L, d, d), "attprojb": (L, d), "ln2w": (L, d), "ln2b": (L, d),
        "fcw": (L, dff, d), "fcb": (L, dff), "fcprojw": (L, d, dff), "fcprojb": (L, d),
        "lnfw": (d,), "lnfb": (d,), "pi_w": (3, d), "pi_b": (3,), "logstd": (2,), "v_w": (d,), "v_b": (1,),
        "pr1w": (ph, d + 3), "pr1b": (ph,), "pr2w": (pc2 * p0 * p0, ph), "pr2b": (pc2 * p0 * p0,),
        "pbw": (pb, ph), "pbb": (pb,),
        "dc1w": (pc2 + pb + 2 * co + (c3 + 3 if cfg.get("pred_skip", 0) else 0), pc1, 4, 4), "dc1b": (pc1,),
        "dc2w": (pc1 + 2 * co + (c2 + 3 if cfg.get("pred_skip", 0) else 0), 3, 4, 4), "dc2b": (3,),
        "drw": (2, ph), "drb": (2,),
        "iv1w": (cfg.get("inv_hidden", 0), 2 * feat), "iv1b": (cfg.get("inv_hidden", 0),),
        "iv2w": (2 if cfg.get("inv_hidden", 0) else 0, cfg.get("inv_hidden", 0)),
        "iv2b": (2 if cfg.get("inv_hidden", 0) else 0,),
    }


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


def forward(P, cfg, images, drives, prev_action, prev_reward, ep_start, pred_actions=None):
    """images u8 [S,T,H,W,3]; returns dict of torch tensors (z, pi, value, pred_frames CHW, pred_drives)."""
    dt = P["in_w"].dtype
    S, T = images.shape[:2]
    d, L, nh = cfg["d_model"], cfg["n_layer"], cfg["n_head"]
    hs = d // nh
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
    pos = np.arange(T)[None, :] - ss
    h = F.linear(tin, P["in_w"], P["in_b"]) + P["wpe"][torch.tensor(pos)]
    t_idx = np.arange(T)
    mask = (t_idx[None, None, :] <= t_idx[None, :, None]) & (t_idx[None, None, :] >= ss[:, :, None])  # [S,T,T]
    mask = torch.tensor(mask)[:, None, :, :]
    for l in range(L):
        y = F.layer_norm(h, (d,), P["ln1w"][l], P["ln1b"][l], eps=1e-5)
        qkv = F.linear(y, P["qkvw"][l], P["qkvb"][l])
        q, k, v = qkv.split(d, dim=-1)
        q = q.reshape(S, T, nh, hs).transpose(1, 2)
        k = k.reshape(S, T, nh, hs).transpose(1, 2)
        v = v.reshape(S, T, nh, hs).transpose(1, 2)
        att = (q @ k.transpose(-1, -2)) / math.sqrt(hs)
        att = att.masked_fill(~mask, float("-inf")).softmax(dim=-1)
        y = (att @ v).transpose(1, 2).reshape(S, T, d)
        h = h + F.linear(y, P["attprojw"][l], P["attprojb"][l])
        y = F.layer_norm(h, (d,), P["ln2w"][l], P["ln2b"][l], eps=1e-5)
        y = F.gelu(F.linear(y, P["fcw"][l], P["fcb"][l]), approximate="tanh")
        h = h + F.linear(y, P["fcprojw"][l], P["fcprojb"][l])
    z = F.layer_norm(h, (d,), P["lnfw"], P["lnfb"], eps=1e-5)
    out = dict(z=z, pi=F.linear(z, P["pi_w"], P["pi_b"]), value=z @ P["v_w"] + P["v_b"][0], a2=a2, a3=a3, vf=vf)
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


def ppo_loss(P, cfg, pc, batch):
    """Total loss (mean per transition) for a batch of sequences, as in C++ seq_loss.

    batch: dict with [S,T,...] arrays: images, drives, prev_action, prev_reward, ep_start, action, old_logp,
    adv, ret, next_small (u8 HWC), next_drives.
    """
    dt = P["in_w"].dtype
    have_pred = pc["pred_coef"] > 0 or pc["drive_coef"] > 0
    out = forward(P, cfg, batch["images"], batch["drives"], batch["prev_action"], batch["prev_reward"],
                  batch["ep_start"])
    z = out["z"]
    terms = {}
    total = torch.zeros((), dtype=dt)
    if pc["use_ppo"]:
        a = torch.tensor(batch["action"], dtype=dt)
        pi = out["pi"]
        mu = torch.tanh(pi[..., :2])
        ls = torch.clamp(P["logstd"], LS_MIN, LS_MAX)
        sd = torch.exp(ls)
        zz = (a[..., :2] - mu) / sd
        logp = (-0.5 * zz * zz - ls - 0.5 * math.log(2 * math.pi)).sum(-1)
        l = pi[..., 2]
        c = a[..., 2] > 0.5
        logp = logp + torch.where(c, -F.softplus(-l), -F.softplus(l))
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
        total = total + pc["pred_coef"] * mse.mean() + pc["drive_coef"] * dmse.mean()
        terms.update(pred=mse.mean().item(), drive=dmse.mean().item())
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
    return total, terms

"""Prediction on held-out data for any world (task 00032): how well a frozen universal model predicts.

The frozen agent lives in the world (its config from the run, with --env-set overrides, e.g. the text world's val
split) for --steps steps in --bodies copies, making its own moves (the policy alone unless --instinct). The trajectory is
cut into non-overlapping windows of ctx steps, each starting a context (as the classical reference's blocks), and every
token's prediction of the next token's values is scored by the PyTorch reference: the two-hot cross-entropy per value
(nats), as the training loss. Reported by what the token predicts (patch, move, body), for the patch targets also by
value position within a token and by step within the window. In the text world a patch value is a letter, so the patch
cross-entropy is nats per letter. With a world that draws symbols as glyphs, one per token (the text world with cell =
patch), also by glyph (task 00033): each glyph's factorized probability (the product over the token's values of the
probability of the glyph's value), normalized over the glyphs; the true glyph's -log (nll_glyph). For one-hot glyphs
this is the one-vs-all normalization; for glyphs of equal values it multiplies copies of one prediction.

    ./venv/bin/python eval_predict.py RUN_DIR --ckpts 50000 100000 --env-set split=val --env-set tile=true
    -> RUN_DIR/eval/ckpt_NNNNNNNN_predict[_TAG].json (or ckpt_final_...)
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import agent_lib  # noqa: E402
from online import _GpuEval, eval_moves, executed_action  # noqa: E402
from train import set_path  # noqa: E402
from world import make_config, make_world  # noqa: E402

KINDS = ("patch", "move", "body")


def rollout(agent, world, steps, rng, instinct=(0, 0.005)):
    """The frozen agent in the world: images [n, steps, S, S, 3], drives, moves made, drives after each move."""
    n, L = world.num_envs, agent.config["ctx"]
    hist, obs = [], world.observe()
    prev_a, es = np.zeros((n, 3), np.float32), np.ones(n, np.uint8)
    out = dict(img=[], drv=[], act=[], ndrv=[])
    for _ in range(steps):
        cur = dict(img=obs["image"], drv=obs["drives"], pa=prev_a, cmd=np.zeros(n, np.float32), es=es,
                   act=np.zeros((n, 3), np.float32))
        a = eval_moves(agent, hist + [cur], rng, instinct)
        cur["act"] = a
        out["img"].append(obs["image"])
        out["drv"].append(obs["drives"])
        out["act"].append(a)
        obs, _, _, inf = world.step(a)
        out["ndrv"].append(inf["terminal_drives"])
        hist = (hist + [cur])[-(L - 1):] if L > 1 else []
        prev_a, es = executed_action(a), np.zeros(n, np.uint8)
    return {k: np.stack(v, 1) for k, v in out.items()}


def windows(ro, L):
    """Non-overlapping windows of L steps, each starting a context: dict of [W, L, ...]."""
    n, T = ro["img"].shape[:2]
    k = T // L
    w = {f: v[:, :k * L].reshape((n * k, L) + v.shape[2:]) for f, v in ro.items()}
    w["es"] = np.zeros((n * k, L), np.uint8)
    w["es"][:, 0] = 1
    return w


def glyph_ll(logp, gl, K):
    """logp [..., V, K] of patch-target tokens, glyphs gl [n, V] in [0, 1] -> [..., n]: the sum over values of
    log p(the glyph's value) (two-hot: the two neighbouring bins' probabilities, weighted)."""
    import torch
    pos = np.clip(gl, 0, 1) * (K - 1)
    lo = np.minimum(np.floor(pos).astype(np.int64), K - 2)
    wt = torch.tensor(pos - lo, dtype=logp.dtype)
    p = logp.exp().unsqueeze(-3)                                          # [..., 1, V, K]
    shape = p.shape[:-3] + (len(gl),) + p.shape[-2:-1]
    g = lambda i: p.expand(shape + (K,)).gather(-1, torch.tensor(i).expand(shape)[..., None])[..., 0]  # noqa: E731
    return torch.log((1 - wt) * g(lo) + wt * g(lo + 1) + 1e-30).sum(-1)


def score(P, cfg, w, batch=64, glyphs=None):
    """Per-value two-hot cross-entropy of every token's prediction of the next token, summed by target kind, by value
    position (patch targets) and by step in the window (patch targets); with glyphs [n, V] (values in [0, 1]) the
    patch targets' glyph NLL as well."""
    import torch
    from reference import torch_reference as ref
    import agent_lib
    nt, V, K = ref.uni_dims(cfg)
    lay = agent_lib.uni_layout(cfg)
    pp, npch = lay["patch"] - 1, len(lay["patch"])                   # the tokens that predict the patches, in order
    T = w["img"].shape[1]
    if glyphs is not None:
        batch = max(1, min(batch, int(2e8 // (T * npch * len(glyphs) * V))))
    gsum, gn = 0.0, 0
    kind = np.full(nt, -1)                                           # what token i predicts (-1: a summary, task 00047)
    kind[pp], kind[nt - 2], kind[nt - 1] = 0, 1, 2
    used = np.where(kind == 0, V, np.where(kind == 1, 3, 2))         # values used by token i's target
    m = np.arange(V)[None, :] < used[:, None]                        # [nt, V]
    acc = dict(sum=np.zeros(3), n=np.zeros(3), by_value=np.zeros(V), by_step=np.zeros(T), n_value=0, n_step=0,
               by_patch=np.zeros(npch), n_patch=0)
    for j in range(0, len(w["img"]), batch):
        b = {f: v[j:j + batch] for f, v in w.items()}
        with torch.no_grad():
            out = ref.forward_uni(P, cfg, b["img"], b["drv"], b["act"], b["es"])
            S = len(b["img"])
            v = out["values"]
            tgt = np.zeros((S, T, nt, V))
            tgt[:, :, :nt - 1] = v[:, :, 1:]
            tgt[:, :, nt - 1, :2] = b["ndrv"]
            logp = out["logits"].float().log_softmax(-1).reshape(S, T, nt, V, K)
            ce = -(torch.tensor(ref.two_hot(tgt, K), dtype=logp.dtype) * logp).sum(-1).numpy()   # [S, T, nt, V]
            if glyphs is not None:   # the patch targets (tokens pp predict the patches, slots pp + 1)
                gll = glyph_ll(logp[:, :, pp], glyphs, K).log_softmax(-1)                   # [S, T, np, n]
                true = ((v[:, :, pp + 1, None, :] - glyphs) ** 2).sum(-1).argmin(-1)        # [S, T, np]
                gsum -= gll.gather(-1, torch.tensor(true)[..., None]).sum().item()
                gn += true.size
        for k in range(3):
            sel = ce[:, :, kind == k] * m[kind == k]
            acc["sum"][k] += sel.sum()
            acc["n"][k] += S * T * m[kind == k].sum()
        pc = ce[:, :, pp]                                              # [S, T, npatch, V], patches in order
        acc["by_value"] += pc.sum((0, 1, 2))
        acc["n_value"] += S * T * pc.shape[2]
        acc["by_step"] += pc.sum((0, 2, 3))
        acc["n_step"] += S * pc.shape[2] * V
        acc["by_patch"] += pc.sum((0, 1, 3))                           # by the predicted patch (task 00044)
        acc["n_patch"] += S * T * V
    res = {"ce_" + KINDS[k]: float(acc["sum"][k] / acc["n"][k]) for k in range(3)}
    res.update(ce_patch_by_value=(acc["by_value"] / acc["n_value"]).tolist(),
               ce_patch_by_step=(acc["by_step"] / acc["n_step"]).tolist(), windows=len(w["img"]))
    g = int(round(np.sqrt(npch)))                                      # the patch grid: by row, top to bottom
    res["ce_patch_by_row"] = (acc["by_patch"] / acc["n_patch"]).reshape(g, g).mean(1).tolist()
    res["ce_patch_by_col"] = (acc["by_patch"] / acc["n_patch"]).reshape(g, g).mean(0).tolist()
    if glyphs is not None:
        res["nll_glyph"] = gsum / gn
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run")
    ap.add_argument("--ckpts", nargs="+", required=True, help="checkpoint steps, or final")
    ap.add_argument("--env-set", action="append", default=[], help="world config override, e.g. split=val")
    ap.add_argument("--bodies", type=int, default=16)
    ap.add_argument("--steps", type=int, default=0,
                    help="steps per body (0: a text world's whole pages / bodies, else 128), cut to whole windows")
    ap.add_argument("--instinct", type=int, default=0, help="candidates of the hunger instinct (0: the policy alone)")
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--device", default="cpu", help="the rollout's forward: cpu or cuda (scoring: PyTorch, CPU)")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    import torch
    from reference import torch_reference as ref
    with open(os.path.join(args.run, "info.json")) as f:
        info = json.load(f)
    env = dict(info["config"]["env"])
    for ov in args.env_set:
        k, v = ov.split("=", 1)
        set_path(env, k, v)
    os.makedirs(os.path.join(args.run, "eval"), exist_ok=True)
    for st in args.ckpts:
        name = "ckpt_final" if st == "final" else "ckpt_%08d" % int(st)
        agent = agent_lib.Agent.load(os.path.join(args.run, name + ".bin"))
        cfg, L = agent.config, agent.config["ctx"]
        world = make_world(args.bodies, make_config(env), seed=args.seed)
        steps = args.steps or (len(world.codes) // world.config.page // args.bodies if hasattr(world, "codes")
                               else 128)
        steps -= steps % L
        model = agent
        if args.device == "cuda":
            import cuda_lib
            model = _GpuEval(agent, cuda_lib.CudaModel(agent))
        ro = rollout(model, world, steps, np.random.default_rng(args.seed), (args.instinct, 0.005))
        P = ref.params_from_flat(agent.params(), agent.param_layout(), cfg, dtype=torch.float32)
        gw = world.config if hasattr(world, "glyphs") else None
        one = gw is not None and gw.cell == cfg["patch"] and gw.size == cfg["img"] and cfg.get("gray")   # a glyph a token
        res = score(P, cfg, windows(ro, L), glyphs=world.glyphs.reshape(len(world.glyphs), -1) / 255.0 if one else None)
        res.update(step=st, env=env, bodies=args.bodies, steps=steps, instinct=args.instinct,
                   values_scored=int(res["windows"] * L * ((cfg["img"] // cfg["patch"]) ** 2)
                                     * cfg["patch"] ** 2 * (1 if cfg.get("gray") else 3)))
        tag = "_" + args.tag if args.tag else ""
        with open(os.path.join(args.run, "eval", "%s_predict%s.json" % (name, tag)), "w") as f:
            json.dump(res, f)
        print("%s %8s  ce patch %.4f  move %.4f  body %.4f%s  (%d windows)" % (
            os.path.basename(os.path.normpath(args.run)), st, res["ce_patch"], res["ce_move"], res["ce_body"],
            "  glyph nll %.4f" % res["nll_glyph"] if "nll_glyph" in res else "", res["windows"]), flush=True)
        print("    ce patch by row " + " ".join("%.4f" % x for x in res["ce_patch_by_row"]), flush=True)
        print("    ce patch by col " + " ".join("%.4f" % x for x in res["ce_patch_by_col"]), flush=True)


if __name__ == "__main__":
    main()

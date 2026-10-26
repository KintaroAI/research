"""Prediction on held-out data for any world (task 00032): how well a frozen universal model predicts.

The frozen agent lives in the world (its config from the run, with --env-set overrides, e.g. the text world's val
split) for --steps steps in --bodies copies, making its own moves (the policy alone unless --instinct). The trajectory is
cut into non-overlapping windows of ctx steps, each starting a context (as the classical reference's blocks), and every
token's prediction of the next token's values is scored by the PyTorch reference: the two-hot cross-entropy per value
(nats), as the training loss. Reported by what the token predicts (patch, move, body), for the patch targets also by
value position within a token and by step within the window. In the text world a patch value is a letter, so the patch
cross-entropy is nats per letter.

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


def score(P, cfg, w, batch=64):
    """Per-value two-hot cross-entropy of every token's prediction of the next token, summed by target kind, by value
    position (patch targets) and by step in the window (patch targets)."""
    import torch
    from reference import torch_reference as ref
    nt, V, K = ref.uni_dims(cfg)
    T = w["img"].shape[1]
    kind = np.array([0] * (nt - 2) + [1, 2])                          # what token i predicts
    used = np.concatenate([np.full(nt - 2, V), [3, 2]])             # values used by token i's target
    m = np.arange(V)[None, :] < used[:, None]                        # [nt, V]
    acc = dict(sum=np.zeros(3), n=np.zeros(3), by_value=np.zeros(V), by_step=np.zeros(T), n_value=0, n_step=0)
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
        for k in range(3):
            sel = ce[:, :, kind == k] * m[kind == k]
            acc["sum"][k] += sel.sum()
            acc["n"][k] += S * T * m[kind == k].sum()
        pc = ce[:, :, kind == 0]                                       # [S, T, npatch, V]
        acc["by_value"] += pc.sum((0, 1, 2))
        acc["n_value"] += S * T * pc.shape[2]
        acc["by_step"] += pc.sum((0, 2, 3))
        acc["n_step"] += S * pc.shape[2] * V
    res = {"ce_" + KINDS[k]: float(acc["sum"][k] / acc["n"][k]) for k in range(3)}
    res.update(ce_patch_by_value=(acc["by_value"] / acc["n_value"]).tolist(),
               ce_patch_by_step=(acc["by_step"] / acc["n_step"]).tolist(), windows=len(w["img"]))
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
        res = score(P, cfg, windows(ro, L))
        res.update(step=st, env=env, bodies=args.bodies, steps=steps, instinct=args.instinct,
                   values_scored=int(res["windows"] * L * ((cfg["img"] // cfg["patch"]) ** 2)
                                     * cfg["patch"] ** 2 * (1 if cfg.get("gray") else 3)))
        tag = "_" + args.tag if args.tag else ""
        with open(os.path.join(args.run, "eval", "%s_predict%s.json" % (name, tag)), "w") as f:
            json.dump(res, f)
        print("%s %8s  ce patch %.4f  move %.4f  body %.4f  (%d windows)" % (
            os.path.basename(os.path.normpath(args.run)), st, res["ce_patch"], res["ce_move"], res["ce_body"],
            res["windows"]), flush=True)


if __name__ == "__main__":
    main()

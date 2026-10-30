"""Next-letter cross-entropy of text-world runs at any scale and token layout (vm-00046, vm-00047): on the same val
windows for every run, the two-hot cross-entropy per value (nats per letter) at the tokens that predict each letter's
first appearance in reading order: its top-left patch, or with model.sum_only its B1 summary. A run's training loss
is not comparable across layouts (in a scaled world most patch targets are copies of a letter already seen); this is.
Also by letter column and by line. The frozen agent lives in the val world (4 copies, seed 0) for --steps ticks on the
GPU; the scoring is the PyTorch reference's forward (float32; on the CPU, or with --device cuda the same computation
on the GPU, run with ~/venvs/vm-gpu/bin/python).

A run of the revealing text world (env.reveal, task 00056) is scored at the one new letter of each step (with tile,
every reader's step s adds letter s % page; checked against the frames). A band model (model.band_n, task 00057) scores
the letters whose top-left patch is one of its tokens; --bands takes the runs as the bands of one model and scores
every letter by its own band's run.

    ./venv/bin/python letter_ce.py RUN [RUN ...] --ckpts 20000 30000 40000 100000
    ~/venvs/vm-gpu/bin/python letter_ce.py --bands LEFT RIGHT --ckpts 30000 --device cuda
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

DEV = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, DEV)

import agent_lib  # noqa: E402
import cuda_lib  # noqa: E402
from eval_predict import rollout, windows  # noqa: E402
from reference import torch_reference as ref  # noqa: E402
from world import make_config, make_world  # noqa: E402


def logits_on(P, cfg, img, drv, act, device):
    """torch_reference.forward_uni's logits [S, T * nt, V, K] and values, on the device (each window one context)."""
    S, T = img.shape[:2]
    nt, V, K = ref.uni_dims(cfg)
    v, used = ref.uni_values(cfg, img, drv, act)
    m = torch.tensor(np.arange(V)[None, :] < used[:, None], dtype=torch.float32, device=device)   # [nt, V]
    x = torch.tensor(ref.two_hot(v, K), dtype=torch.float32, device=device) * m[..., None]
    h = F.linear(x.reshape(S, T * nt, V * K), P["uni_in_w"], P["uni_in_b"]).reshape(S, T, nt, -1) + P["uni_slot"]
    h = (h + P["wpe"][:T][None, :, None]).reshape(S, T * nt, -1)
    mask = torch.ones(T * nt, T * nt, dtype=torch.bool, device=device).tril()
    allow = ref.attn_allow(cfg)
    if allow is not None:
        sl = torch.arange(T * nt, device=device) % nt
        mask = mask & torch.tensor(allow, device=device)[sl[:, None], sl[None, :]]
    hz = ref.transformer(P, cfg, h, mask[None, None])
    return F.linear(hz, P["uni_out_w"], P["uni_out_b"]).reshape(S, T * nt, V, K), v


def score(run, ckpt, steps=1600, seed=0, batch=4, device="cpu"):
    """(summed CE per value [n * n], letters scored [n * n]) per letter position of run's checkpoint at tick ckpt."""
    info = json.load(open(os.path.join(run, "info.json")))
    env = dict(info["config"]["env"], split="val", tile=True)
    reveal = bool(env.get("reveal", False))
    agent = agent_lib.Agent.load(os.path.join(run, "ckpt_%08d.bin" % ckpt))
    cfg = agent.config
    model = cuda_lib.CudaModel(agent)
    world = make_world(4, make_config(env), seed=seed)
    ro = rollout(model, world, steps, np.random.default_rng(seed))
    model.close()
    L = cfg["ctx"]
    w = windows(ro, L)
    if reveal:   # the new letter of each window step: rollout step s = chunk * L + tau adds letter s % page
        k = steps // L
        new = ((np.arange(len(w["img"])) % k)[:, None] * L + np.arange(L)[None]) % world.config.page   # [W, L]
        shown = (world.letters(w["img"].reshape((-1,) + w["img"].shape[2:])) < world.empty).sum(-1)
        assert (shown.reshape(new.shape) == new + 1).all(), "the frames do not fill one letter per step"
    t = ref.uni_slot_table(cfg)
    n, lp = env["size"] // env["cell"], env["cell"] // cfg["patch"]   # letters per side, patches per letter side
    if (t["kind"] == 1).any():
        tgt = np.flatnonzero((t["kind"] == 1) & (t["r"] % lp == 0) & (t["c"] % lp == 0))
        lid = (t["r"][tgt] // lp) * n + t["c"][tgt] // lp                    # the letter of each target
    else:
        tgt = np.flatnonzero((t["kind"] == 2) & (t["level"] == 1))
        lid = t["r"][tgt] * n + t["c"][tgt]
    assert len(tgt) == n * n or cfg.get("band_n", 0) >= 2, len(tgt)   # a band model: its band's letters
    nt, V, K = ref.uni_dims(cfg)
    P = ref.params_from_flat(agent.params(), agent.param_layout(), cfg, dtype=torch.float32)
    P = {key: val.detach().to(device) for key, val in P.items()}
    tot, cnt = np.zeros(n * n), np.zeros(n * n)
    for j in range(0, len(w["img"]), batch):
        b = {f: v[j:j + batch] for f, v in w.items()}
        with torch.no_grad():
            if device == "cpu":
                out = ref.forward_uni(P, cfg, b["img"], b["drv"], b["act"], b["es"])
                lg, vals = out["logits"], out["values"]
            else:
                lg, vals = logits_on(P, cfg, b["img"], b["drv"], b["act"], device)
        S, T = b["img"].shape[:2]
        logp = lg.float().log_softmax(-1).reshape(S, T, nt, V, K)[:, :, tgt - 1]               # the predictors
        two = torch.tensor(ref.two_hot(vals[:, :, tgt], K), dtype=logp.dtype, device=logp.device)
        ce = (-(two * logp).sum(-1).mean(-1)).cpu().numpy()                                    # [S, T, n*n] per value
        if reveal:   # only the new letter of each step
            sel = lid[None, None, :] == new[j:j + batch, :, None]
            ce, c_ = ce * sel, sel.sum((0, 1))
        else:
            c_ = np.full(len(tgt), S * T)
        np.add.at(tot, lid, ce.sum((0, 1)))
        np.add.at(cnt, lid, c_)
    agent.close()
    return tot, cnt


def report(name, ckpt, tot, cnt):
    n = int(round(np.sqrt(len(tot))))
    pos = (tot / np.maximum(cnt, 1)).reshape(n, n)
    print("%-14s %6dk next-letter CE per value %.4f  (%d val letters)" % (name, ckpt // 1000, tot.sum() / cnt.sum(),
                                                                           cnt.sum()))
    print("  by letter column:", " ".join("%.3f" % v for v in pos.mean(0)))
    print("  by line:         ", " ".join("%.3f" % v for v in pos.mean(1)), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--ckpts", type=int, nargs="+", default=[100000])
    ap.add_argument("--steps", type=int, default=1600)
    ap.add_argument("--threads", type=int, default=32)
    ap.add_argument("--device", default="cpu", help="cpu, or cuda (with ~/venvs/vm-gpu/bin/python)")
    ap.add_argument("--bands", action="store_true", help="the runs are the bands of one model (task 00057)")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    if a.bands:
        for k in a.ckpts:
            parts = [score(r, k, a.steps, device=a.device) for r in a.runs]
            cnt = np.stack([c for _, c in parts])
            assert ((cnt > 0).sum(0) == 1).all(), "every letter in exactly one band"
            report("+".join(os.path.basename(r) for r in a.runs), k, sum(t for t, _ in parts), cnt.sum(0))
        return
    for r in a.runs:
        for k in a.ckpts:
            report(os.path.basename(r), k, *score(r, k, a.steps, device=a.device))


if __name__ == "__main__":
    main()

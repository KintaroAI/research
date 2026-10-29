"""Next-letter cross-entropy of text-world runs at any scale and token layout (vm-00046, vm-00047): on the same val
windows for every run, the two-hot cross-entropy per value (nats per letter) at the tokens that predict each letter's
first appearance in reading order: its top-left patch, or with model.sum_only its B1 summary. A run's training loss
is not comparable across layouts (in a scaled world most patch targets are copies of a letter already seen); this is.
Also by letter column and by line. The frozen agent lives in the val world (4 copies, seed 0) for --steps ticks on the
GPU; the scoring is the PyTorch reference's forward (CPU, float32).

    ./venv/bin/python letter_ce.py RUN [RUN ...] --ckpts 20000 30000 40000 100000
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

DEV = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, DEV)

import agent_lib  # noqa: E402
import cuda_lib  # noqa: E402
from eval_predict import rollout, windows  # noqa: E402
from reference import torch_reference as ref  # noqa: E402
from world import make_config, make_world  # noqa: E402


def score(run, ckpt, steps=1600, seed=0, batch=4):
    """(CE per value, windows, CE per letter position [n, n]) of run's checkpoint at tick ckpt."""
    info = json.load(open(os.path.join(run, "info.json")))
    env = dict(info["config"]["env"], split="val", tile=True)
    agent = agent_lib.Agent.load(os.path.join(run, "ckpt_%08d.bin" % ckpt))
    cfg = agent.config
    model = cuda_lib.CudaModel(agent)
    ro = rollout(model, make_world(4, make_config(env), seed=seed), steps, np.random.default_rng(seed))
    model.close()
    w = windows(ro, cfg["ctx"])
    t = ref.uni_slot_table(cfg)
    n, lp = env["size"] // env["cell"], env["cell"] // cfg["patch"]   # letters per side, patches per letter side
    if (t["kind"] == 1).any():
        tgt = np.flatnonzero((t["kind"] == 1) & (t["r"] % lp == 0) & (t["c"] % lp == 0))
    else:
        tgt = np.flatnonzero((t["kind"] == 2) & (t["level"] == 1))
    assert len(tgt) == n * n, len(tgt)
    nt, V, K = ref.uni_dims(cfg)
    P = ref.params_from_flat(agent.params(), agent.param_layout(), cfg, dtype=torch.float32)
    pos, cnt = np.zeros(n * n), 0
    for j in range(0, len(w["img"]), batch):
        b = {f: v[j:j + batch] for f, v in w.items()}
        with torch.no_grad():
            out = ref.forward_uni(P, cfg, b["img"], b["drv"], b["act"], b["es"])
        S, T = b["img"].shape[:2]
        logp = out["logits"].float().log_softmax(-1).reshape(S, T, nt, V, K)[:, :, tgt - 1]   # the predictors
        two = torch.tensor(ref.two_hot(out["values"][:, :, tgt], K), dtype=logp.dtype)
        pos += (-(two * logp).sum(-1).mean(-1)).sum((0, 1)).numpy()                          # per value
        cnt += S * T
    agent.close()
    pos = (pos / cnt).reshape(n, n)
    return pos.mean(), len(w["img"]), pos


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--ckpts", type=int, nargs="+", default=[100000])
    ap.add_argument("--steps", type=int, default=1600)
    ap.add_argument("--threads", type=int, default=32)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    for r in a.runs:
        for k in a.ckpts:
            ce, nw, pos = score(r, k, a.steps)
            print("%-14s %6dk next-letter CE per value %.4f  (%d val windows)" % (os.path.basename(r), k // 1000, ce, nw))
            print("  by letter column:", " ".join("%.3f" % v for v in pos.mean(0)))
            print("  by line:         ", " ".join("%.3f" % v for v in pos.mean(1)), flush=True)


if __name__ == "__main__":
    main()

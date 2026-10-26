"""The classical reference for the text world (task 00032): a character-level GPT with the universal model's Transformer
(torch_reference.transformer: the same layers, widths, initialisation, optimizer and gradient clipping), but a token is
a letter's index: an embedding table in, a learned position per context slot, and one softmax over the alphabet out.

Trained offline on the text world's train split (random windows of ctx letters, batch windows per update); scored on
exactly the windows eval_predict.py scores with --env-set split=val --env-set tile=true (bodies readers, evenly spaced
pages, non-overlapping windows of the universal model's ctx steps), each window a fresh context. Nats per letter.

    ~/venvs/vm-gpu/bin/python reference/char_gpt.py --config configs/T_text_B.json --updates 100000 -o OUT_DIR
"""

import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

DEV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, DEV)

from reference.torch_reference import transformer  # noqa: E402
from world import make_config, make_world  # noqa: E402


def init_params(n_vocab, ctx, d, L, d_ff, gen, device):
    """GPT-2 initialisation, as the C++ model's Transformer (0.02; residual projections 0.02 / sqrt(2 L))."""
    def nrm(*shape, std=0.02):
        return (torch.randn(*shape, generator=gen) * std).to(device).requires_grad_()

    def full(v, *shape):
        return torch.full(shape, float(v), device=device).requires_grad_()
    r = 0.02 / math.sqrt(2 * L)
    return dict(wte=nrm(n_vocab, d), wpe=nrm(ctx, d),
                ln1w=full(1, L, d), ln1b=full(0, L, d), qkvw=nrm(L, 3 * d, d), qkvb=full(0, L, 3 * d),
                attprojw=nrm(L, d, d, std=r), attprojb=full(0, L, d), ln2w=full(1, L, d), ln2b=full(0, L, d),
                fcw=nrm(L, d_ff, d), fcb=full(0, L, d_ff), fcprojw=nrm(L, d, d_ff, std=r), fcprojb=full(0, L, d),
                lnfw=full(1, d), lnfb=full(0, d), out_w=nrm(n_vocab, d), out_b=full(0, n_vocab))


def logits(P, cfg, x):
    """x [S, N] letter codes -> logits [S, N, n_vocab] of the next letter."""
    S, N = x.shape
    h = P["wte"][x] + P["wpe"][:N][None]
    mask = torch.ones(N, N, dtype=torch.bool, device=x.device).tril()[None, None]
    return F.linear(transformer(P, cfg, h, mask), P["out_w"], P["out_b"])


def val_windows(world_cfg, bodies, L):
    """The letters of the windows eval_predict.py scores: [W, L * page] codes."""
    w = make_world(bodies, make_config(dict(world_cfg, split="val", tile=True)), seed=0)
    page = w.config.page
    steps = len(w.codes) // page // bodies
    steps -= steps % L
    starts = (w.pos[:, None] + np.arange(0, steps, L)[None] * page).reshape(-1)
    idx = (starts[:, None] + np.arange(L * page)[None]) % len(w.codes)
    return w.codes[idx]


def evaluate(P, cfg, win, device, batch=64):
    """Nats per letter over the windows, letters 1.. of each (eval_predict.py also scores letter 0, which the universal
    model predicts from no letters at all, at its first body token: 1 of 512), and by position in the window."""
    tot, n, by = 0.0, 0, None
    with torch.no_grad():
        for j in range(0, len(win), batch):
            x = torch.tensor(win[j:j + batch], device=device)
            lp = logits(P, cfg, x[:, :-1]).log_softmax(-1)
            ce = -lp.gather(-1, x[:, 1:, None])[..., 0]                     # [S, N]: letter t+1 from letters 0..t
            tot += ce.sum().item()
            n += ce.numel()
            by = ce.sum(0) if by is None else by + ce.sum(0)
    return tot / n, (by / len(win)).cpu().numpy()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, help="a text-world config (the world, the model's sizes, ppo.lr)")
    ap.add_argument("--updates", type=int, default=100000)
    ap.add_argument("--batch", type=int, default=0, help="windows per update (0: the config's online.batch)")
    ap.add_argument("--eval-every", type=int, default=10000)
    ap.add_argument("--bodies", type=int, default=16, help="eval_predict.py's readers (the val windows)")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("-o", "--out", required=True)
    args = ap.parse_args()
    cfg = json.load(open(args.config))
    m = dict(d_model=128, n_layer=2, n_head=4, d_ff=512)
    m.update({k: v for k, v in cfg["model"].items() if k in m})
    world = make_world(1, make_config(dict(cfg["env"], split="train")), seed=0)
    page, L = world.config.page, cfg["model"]["ctx"]
    ctx = L * page
    n_vocab = len(world.alphabet)
    B = args.batch or cfg["online"].get("batch", 8)
    ppo = dict(lr=2.5e-4, beta1=0.9, beta2=0.999, adam_eps=1e-5, weight_decay=0.0, max_grad_norm=0.5)
    ppo.update({k: v for k, v in cfg.get("ppo", {}).items() if k in ppo})
    torch.manual_seed(args.seed)
    gen = torch.Generator().manual_seed(args.seed)
    P = init_params(n_vocab, ctx, m["d_model"], m["n_layer"], m["d_ff"], gen, args.device)
    opt = torch.optim.AdamW(list(P.values()), lr=ppo["lr"], betas=(ppo["beta1"], ppo["beta2"]), eps=ppo["adam_eps"],
                            weight_decay=ppo["weight_decay"])
    win = val_windows(cfg["env"], args.bodies, L)
    train = torch.tensor(world.codes, device=args.device)
    rng = np.random.default_rng(args.seed)
    os.makedirs(args.out, exist_ok=True)
    nparams = sum(p.numel() for p in P.values())
    info = dict(command=" ".join(sys.argv), config=args.config, model=m, ctx_letters=ctx, n_vocab=n_vocab,
                batch=B, ppo=ppo, params=nparams, train_letters=len(world.codes), val_windows=len(win))
    json.dump(info, open(os.path.join(args.out, "info.json"), "w"), indent=2)
    print("params %d  vocab %d  context %d letters  batch %d  val windows %d" % (nparams, n_vocab, ctx, B, len(win)),
          flush=True)
    log = open(os.path.join(args.out, "metrics.jsonl"), "w")
    t0 = time.time()
    for u in range(args.updates + 1):
        if u % args.eval_every == 0:
            v, by = evaluate(P, m, win, args.device)
            row = dict(update=u, letters=u * B * ctx, val=v, val_by_page=[float(by[max(i - 1, 0):i + page - 1].mean())
                                                                           for i in range(0, len(by), page)],
                       time=time.time() - t0)
            log.write(json.dumps(row) + "\n")
            log.flush()
            print("update %7d  val %.4f nats/letter  %.0f s" % (u, v, row["time"]), flush=True)
        if u == args.updates:
            break
        st = torch.tensor(rng.integers(0, len(train) - ctx - 1, B), device=args.device)
        x = train[st[:, None] + torch.arange(ctx + 1, device=args.device)[None]]
        loss = F.cross_entropy(logits(P, m, x[:, :-1]).reshape(-1, n_vocab), x[:, 1:].reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(P.values()), ppo["max_grad_norm"])
        opt.step()
    torch.save({k: v.detach().cpu() for k, v in P.items()}, os.path.join(args.out, "params.pt"))


if __name__ == "__main__":
    main()

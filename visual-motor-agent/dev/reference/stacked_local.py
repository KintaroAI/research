"""Two stacked boxes with backprop cut between them (task 00045): can the stack learn, and what crosses the cut?

A box maps V numbers per position to V numbers per position (x = y = V, the text world's 65 letters): W_in (d x V)
plus a learned position, the universal model's Transformer (torch_reference.transformer), W_out (V x d). The bottom
reads the letters one-hot; its output h (V per position) is the top's input; the top's output is the logits of the
next letter. Five ways of training (--arm):

  e2e     cross-entropy of the top's output, backprop through both boxes (the ceiling)
  frozen  the bottom stays random; the top trains on h (the floor)
  local   the bottom trains on its own next-letter cross-entropy (h read as logits); the top on h, detached
  dtp     difference target propagation: an inverse box g learns to undo the top (|g(top(h + noise)) - (h + noise)|^2);
          the output target t = y_hat - beta (softmax(y_hat) - onehot(y)); the bottom's target
          h* = h + g(t) - g(y_hat), its loss |h - h*|^2; the top trains on h, detached
  dni     synthetic gradients (conditioned): a box M reads [h, onehot(next letter)] over the whole window and predicts
          the top loss's gradient at h; the bottom follows the prediction; M trains on the true gradient at the cut

--tie (the owner's idea): the bottom's output map and the top's input map are one matrix W (the bottom's W_out = the
top's W_in transposed, as GPT-2's wte), a module of its own learning at the full rate from every loss that reaches it.
The top's own loss then reshapes W, so the top shapes what the bottom outputs, without any gradient through the bottom.

--cut-norm: h is layer-normalized (no parameters) at the cut, so it cannot grow (in dtp the unbounded h grew without
limit: each target pushed it outward, the inverse trained near the old h lagged, the next target was worse).

The bottom of dtp and dni learns at --bottom-lr-scale (default 0.1) times the learning rate: AdamW makes every step
about lr in size, so while the inverse or the gradient predictor is still untrained its noise moves the bottom at full
speed and the top chases a drifting input (a toy sequence: at scale 1 neither learns; at 0.1 dtp does).

Data as reference/char_gpt.py (random 512-letter windows of the train split, 8 per update; val nats per letter on
eval_predict's windows). Usage (from visual-motor-agent/dev):

    ~/venvs/vm-gpu/bin/python reference/stacked_local.py --arm dtp --seed 1 -o OUT_DIR
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

from reference.char_gpt import val_windows  # noqa: E402
from reference.torch_reference import transformer  # noqa: E402
from world import make_config, make_world  # noqa: E402

ARMS = ("e2e", "frozen", "local", "dtp", "dni")
MODEL = dict(d_model=128, n_layer=2, n_head=4, d_ff=512)


def box_params(n_in, n_out, ctx, gen, device, m=MODEL):
    """A box's parameters, GPT-2 initialisation as char_gpt.py (0.02; residual projections 0.02 / sqrt(2 L))."""
    d, L, dff = m["d_model"], m["n_layer"], m["d_ff"]

    def nrm(*shape, std=0.02):
        return (torch.randn(*shape, generator=gen) * std).to(device).requires_grad_()

    def full(v, *shape):
        return torch.full(shape, float(v), device=device).requires_grad_()
    r = 0.02 / math.sqrt(2 * L)
    return dict(in_w=nrm(d, n_in), in_b=full(0, d), wpe=nrm(ctx, d),
                ln1w=full(1, L, d), ln1b=full(0, L, d), qkvw=nrm(L, 3 * d, d), qkvb=full(0, L, 3 * d),
                attprojw=nrm(L, d, d, std=r), attprojb=full(0, L, d), ln2w=full(1, L, d), ln2b=full(0, L, d),
                fcw=nrm(L, dff, d), fcb=full(0, L, dff), fcprojw=nrm(L, d, dff, std=r), fcprojb=full(0, L, d),
                lnfw=full(1, d), lnfb=full(0, d), out_w=nrm(n_out, d), out_b=full(0, n_out))


def box(P, x, causal=True, m=MODEL):
    """x [S, N, n_in] -> [S, N, n_out]; causal: position i sees positions <= i, else the whole window."""
    S, N, _ = x.shape
    h = F.linear(x, P["in_w"], P["in_b"]) + P["wpe"][:N][None]
    mask = torch.ones(N, N, dtype=torch.bool, device=x.device)
    if causal:
        mask = mask.tril()
    return F.linear(transformer(P, m, h, mask[None, None]), P["out_w"], P["out_b"])


def ce(logits, y):
    return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1))


def output_target(y_hat, y, beta):
    """The top's output nudged toward the right letter: y_hat - beta * dCE/dy_hat (per position, local to the output)."""
    return y_hat - beta * (y_hat.softmax(-1) - F.one_hot(y, y_hat.shape[-1]).to(y_hat.dtype))


def dtp_target(h, y_hat, t, g):
    """Difference target propagation: h* = h + g(t) - g(y_hat) (g: the top's learned inverse; its error cancels)."""
    return h + g(t) - g(y_hat)


def grad_at_cut(top, h, y):
    """The top's loss and its gradient at its input h (the top's own backward; h detached from the bottom)."""
    hr = h.detach().requires_grad_()
    loss = ce(top(hr), y)
    g, = torch.autograd.grad(loss, hr)
    return loss, g


class Stack:
    """The two boxes and, for dtp / dni, the inverse g or the gradient predictor M; one AdamW per module."""

    def __init__(self, arm, V, ctx, seed, device, lr=2.5e-4, beta=1.0, sigma=0.1, clip=0.5, bottom_lr_scale=None,
                 tie=False, cut_norm=False):
        if arm not in ARMS:
            raise ValueError("arm %r" % arm)
        gen = torch.Generator().manual_seed(seed)
        self.arm, self.V, self.beta, self.sigma, self.clip = arm, V, beta, sigma, clip
        self.mods = dict(bottom=box_params(V, V, ctx, gen, device), top=box_params(V, V, ctx, gen, device))
        self.tie, self.cut_norm = tie, cut_norm
        if tie:   # one matrix W [V, d]: the bottom's out_w and the top's in_w transposed, owned by neither box
            self.mods["cut"] = dict(w=self.mods["bottom"].pop("out_w"))
            del self.mods["top"]["in_w"]
        if arm == "dtp":
            self.mods["inv"] = box_params(V, V, ctx, gen, device)
        if arm == "dni":
            self.mods["sg"] = box_params(2 * V, V, ctx, gen, device)
        if bottom_lr_scale is None:
            bottom_lr_scale = 0.1 if arm in ("dtp", "dni") else 1.0
        self.bottom_lr_scale = bottom_lr_scale
        self.opt = {k: torch.optim.AdamW(list(P.values()), lr=lr * (bottom_lr_scale if k == "bottom" else 1.0),
                                         betas=(0.9, 0.999), eps=1e-5, weight_decay=0.0)
                    for k, P in self.mods.items() if not (arm == "frozen" and k == "bottom")}

    def bottom(self, x):
        P = self.mods["bottom"]
        h = box(dict(P, out_w=self.mods["cut"]["w"]) if self.tie else P, x)
        return F.layer_norm(h, (h.shape[-1],)) if self.cut_norm else h

    def top(self, h):
        P = self.mods["top"]
        return box(dict(P, in_w=self.mods["cut"]["w"].t()) if self.tie else P, h)

    def inv(self, y):
        return box(self.mods["inv"], y)

    def sg(self, h, y):
        return box(self.mods["sg"], torch.cat([h, F.one_hot(y, self.V).to(h.dtype)], -1), causal=False)

    def logits(self, x):
        return self.top(self.bottom(x))

    def losses(self, x, y, diag=False):
        """One update's losses, each module's own (no gradient crosses the cut except in e2e); returns
        {module: loss}, stats. x: one-hot letters [S, N, V], y: next letters [S, N]."""
        arm, st = self.arm, {}
        h = self.bottom(x)
        if arm == "e2e":
            loss = ce(self.top(h), y)
            st["top"] = loss.item()
            return dict(top=loss), st
        hd = h.detach()
        need_g = arm == "dni" or diag
        if need_g:
            hr = hd.clone().requires_grad_()
            out = self.top(hr)
            lt = ce(out, y)
            g_true, = torch.autograd.grad(lt, hr, retain_graph=True)   # dL_top/dh (top's own backward)
        else:
            out = self.top(hd)
            lt = ce(out, y)
        L = dict(top=lt)
        st["top"] = lt.item()
        if arm == "local":
            L["bottom"] = ce(h, y)
            st["bottom_ce"] = L["bottom"].item()
        elif arm == "dtp":
            with torch.no_grad():
                y_hat = out.detach()
                hstar = dtp_target(hd, y_hat, output_target(y_hat, y, self.beta), self.inv)
            L["bottom"] = ((h - hstar) ** 2).sum(-1).mean()
            ht = hd + self.sigma * torch.randn_like(hd)
            with torch.no_grad():
                yt = self.top(ht)
            L["inv"] = ((self.inv(yt) - ht) ** 2).sum(-1).mean()
            st["inv_rec"] = L["inv"].item()
            if diag:
                st["cos_tp"] = F.cosine_similarity((hstar - hd).reshape(1, -1), -g_true.reshape(1, -1)).item()
        elif arm == "dni":
            scale = float(y.numel())                      # dL/dh of a mean over S N positions is O(1 / (S N))
            pred = self.sg(hd, y)
            L["sg"] = ((pred - scale * g_true) ** 2).mean()
            L["bottom"] = (h * pred.detach()).sum() / scale   # its gradient at h: the predicted gradient
            st["sg_mse"] = L["sg"].item()
            if diag:
                st["cos_sg"] = F.cosine_similarity(pred.detach().reshape(1, -1), g_true.reshape(1, -1)).item()
        return L, st

    def step(self, x, y, diag=False):
        for o in self.opt.values():
            o.zero_grad(set_to_none=True)
        L, st = self.losses(x, y, diag)
        sum(L.values()).backward()                        # the losses touch disjoint modules (cut arms)
        for k, o in self.opt.items():
            torch.nn.utils.clip_grad_norm_(list(self.mods[k].values()), self.clip)
            o.step()
        return st


def evaluate(stack, win, V, device, batch=64):
    """Val nats per letter of the top (and of the bottom read as logits, meaningful for local), letters 1.. of each
    window (as char_gpt.evaluate)."""
    tot, tot_b, n = 0.0, 0.0, 0
    with torch.no_grad():
        for j in range(0, len(win), batch):
            w = torch.tensor(win[j:j + batch], device=device)
            x = F.one_hot(w[:, :-1], V).float()
            h = stack.bottom(x)
            tot += F.cross_entropy(stack.top(h).reshape(-1, V), w[:, 1:].reshape(-1), reduction="sum").item()
            tot_b += F.cross_entropy(h.reshape(-1, V), w[:, 1:].reshape(-1), reduction="sum").item()
            n += w[:, 1:].numel()
    return tot / n, tot_b / n


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", required=True, choices=ARMS)
    ap.add_argument("--config", default=os.path.join(DEV, "configs", "T_text_B.json"))
    ap.add_argument("--updates", type=int, default=30000)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--eval-every", type=int, default=2500)
    ap.add_argument("--lr", type=float, default=2.5e-4)
    ap.add_argument("--beta", type=float, default=1.0, help="dtp: the output target's step")
    ap.add_argument("--sigma", type=float, default=0.1, help="dtp: the noise the inverse learns on")
    ap.add_argument("--tie", action="store_true", help="one matrix for the bottom's W_out and the top's W_in")
    ap.add_argument("--cut-norm", action="store_true", help="layer-normalize h at the cut (no parameters)")
    ap.add_argument("--bottom-lr-scale", type=float, default=None, help="the bottom's lr / lr (default: 0.1 for dtp "
                                                                        "and dni, else 1)")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("-o", "--out", required=True)
    args = ap.parse_args()
    cfg = json.load(open(args.config))
    world = make_world(1, make_config(dict(cfg["env"], split="train")), seed=0)
    page, Lw = world.config.page, cfg["model"]["ctx"]
    ctx, V = Lw * page, len(world.alphabet)
    torch.manual_seed(args.seed)
    stack = Stack(args.arm, V, ctx, args.seed, args.device, lr=args.lr, beta=args.beta, sigma=args.sigma,
                  bottom_lr_scale=args.bottom_lr_scale, tie=args.tie, cut_norm=args.cut_norm)
    win = val_windows(cfg["env"], 16, Lw)
    train = torch.tensor(world.codes, device=args.device)
    rng = np.random.default_rng(args.seed)
    os.makedirs(args.out, exist_ok=True)
    info = dict(command=" ".join(sys.argv), arm=args.arm, model=MODEL, ctx_letters=ctx, n_vocab=V, batch=args.batch,
                lr=args.lr, beta=args.beta, sigma=args.sigma, bottom_lr_scale=stack.bottom_lr_scale, tie=args.tie, cut_norm=args.cut_norm, seed=args.seed,
                params={k: sum(p.numel() for p in P.values()) for k, P in stack.mods.items()}, val_windows=len(win))
    json.dump(info, open(os.path.join(args.out, "info.json"), "w"), indent=2)
    print("arm %s  params %s  context %d letters  val windows %d" % (args.arm, info["params"], ctx, len(win)),
          flush=True)
    log = open(os.path.join(args.out, "metrics.jsonl"), "w")
    t0, acc = time.time(), {}
    for u in range(args.updates + 1):
        if u % args.eval_every == 0:
            v, vb = evaluate(stack, win, V, args.device)
            row = dict(update=u, val=v, val_bottom=vb, time=time.time() - t0,
                       **{k: float(np.mean(a)) for k, a in acc.items()})
            log.write(json.dumps(row) + "\n")
            log.flush()
            print("update %6d  val %.4f  bottom %.4f  %s  %.0f s" % (
                u, v, vb, " ".join("%s %.4f" % (k, row[k]) for k in sorted(acc)), row["time"]), flush=True)
            acc = {}
        if u == args.updates:
            break
        st_ = torch.tensor(rng.integers(0, len(train) - ctx - 1, args.batch), device=args.device)
        w = train[(st_[:, None] + torch.arange(ctx + 1, device=args.device)[None]) % len(train)]
        st = stack.step(F.one_hot(w[:, :-1], V).float(), w[:, 1:], diag=u % 50 == 0)
        for k, val in st.items():
            acc.setdefault(k, []).append(val)
    torch.save({m: {k: v.detach().cpu() for k, v in P.items()} for m, P in stack.mods.items()},
               os.path.join(args.out, "params.pt"))


if __name__ == "__main__":
    main()

"""The ablation ladder (task 00035): from the universal model to the classical GPT on the text world, one change at a
time, to find what costs the universal model 0.17 nats per letter (vm-00028).

One PyTorch model with five switches; all at the first value it is the universal model of configs/T_text_B.json
(initialised from the C++ model's own weights at birth), all at the second it is reference/char_gpt.py's GPT:

  tokens  universal: per step a body token, the page's 64 letter tokens, a move token (body 0, move (0.5, 0.5, 0)), as
          a text lifetime trained offline (train_text_offline.py) | letters: the letter tokens only
  input   linear: W_in . (values - 0.5), a letter's values its grey level, 4 times (the 2x2 cell) | table: an embedding
          per letter (body and move tokens keep the linear map) | bins: every token, the sum over its used values of a
          learned embedding per (value slot, bin), weighted two-hot (the output's bins, on the input; for letters on
          bin centres a table, for continuous values a learned piecewise-linear function; W_in is the case of
          embeddings linear in the bin)
  pos     factorized: a slot embedding (position in the step) + a step embedding | absolute: one per position
  output  bins: each value's 256-bin distribution, two-hot cross-entropy, a token's loss the mean over its values |
          classes: a 65-way softmax for the letter targets (body and move targets keep the bins)
  init    cpp: the C++ initialisation (W_in std sqrt(3 / V): content at unit scale; slots 0.02 plus a 0.1 sine-cosine
          code; steps 0.02) | gpt: content and positions at 0.02, as GPT-2

Training: offline, 8 random windows of 8 pages (512 letters) of the train split per update, AdamW 3e-4, gradient clip
0.5, the same Transformer (torch_reference.transformer). Scoring: char_gpt's 208 val windows, every letter but each
window's first, nats per letter (bins: the mean of the letter's 4 values' cross-entropies, as eval_predict.py).

    ~/venvs/vm-gpu/bin/python reference/text_ladder.py --name R0 --updates 30000 -o OUT_DIR [--tokens letters ...]
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

import agent_lib  # noqa: E402
from reference import torch_reference as ref  # noqa: E402
from reference.char_gpt import val_windows  # noqa: E402
from world import make_config, make_world  # noqa: E402

TRANSFORMER = ("ln1w", "ln1b", "qkvw", "qkvb", "attprojw", "attprojb", "ln2w", "ln2b", "fcw", "fcb", "fcprojw",
               "fcprojb", "lnfw", "lnfb")


class Ladder:
    def __init__(self, cfg, world, sw, seed, device):
        self.sw, self.dev = sw, device
        m = cfg["model"]
        agent = agent_lib.Agent(m, seed=seed)                      # the C++ model at birth
        self.mc = agent.config
        nt, V, K = ref.uni_dims(self.mc)
        self.page, self.L, self.V, self.K = world.config.page, self.mc["ctx"], V, K
        self.levels = torch.tensor(world.levels / 255.0, dtype=torch.float32, device=device)
        self.nv = len(world.alphabet)
        P0 = ref.params_from_flat(agent.params(), agent.param_layout(), self.mc, dtype=torch.float32)
        agent.close()
        d = self.mc["d_model"]
        uni = sw["tokens"] == "universal"
        self.spt = self.page + 2 if uni else self.page               # tokens per step
        g = torch.Generator().manual_seed(seed + 1000)
        nrm = lambda *s, std: torch.randn(*s, generator=g) * std    # noqa: E731
        P = {k: P0[k].detach().clone() for k in TRANSFORMER}
        P["uni_in_w"], P["uni_in_b"] = P0["uni_in_w"].detach().clone(), P0["uni_in_b"].detach().clone()
        P["uni_out_w"], P["uni_out_b"] = P0["uni_out_w"].detach().clone(), P0["uni_out_b"].detach().clone()
        gpt = sw["init"] == "gpt"
        if gpt:
            P["uni_in_w"] = nrm(d, V, std=0.02)
        if sw["input"] == "table":
            P["table"] = nrm(self.nv, d, std=0.02 if gpt else 0.5)  # 0.5: the linear map's output scale here
        if sw["input"] == "bins":                                   # [V, K, d]; a token's sum over its values
            P["in_bins"] = nrm(V, K, d, std=0.02 if gpt else 0.5 / math.sqrt(V))
        slot = P0["uni_slot"].detach().clone()                      # [nt, d]: body, 64 patches (sine-cosine), move
        if not uni:
            slot = slot[1:-1]
        if gpt:
            slot = nrm(*slot.shape, std=0.02)
        if sw["pos"] == "factorized":
            P["slot"], P["step"] = slot, (P0["wpe"].detach().clone() if not gpt else nrm(self.L, d, std=0.02))
        else:
            P["pos"] = nrm(self.L * self.spt, d, std=0.02)
        if sw["output"] == "classes":
            P["cls_w"], P["cls_b"] = nrm(self.nv, d, std=0.02), torch.zeros(self.nv)
        self.P = {k: v.to(device).requires_grad_() for k, v in P.items()}
        # the token layout of one window: kind (0 body, 1 letter, 2 move) and the letter's index in the window
        kind, li = [], []
        for t in range(self.L):
            if uni:
                kind.append(0); li.append(-1)                       # noqa: E702
            kind += [1] * self.page
            li += list(range(t * self.page, (t + 1) * self.page))
            if uni:
                kind.append(2); li.append(-1)                       # noqa: E702
        self.kind = torch.tensor(kind, device=device)
        self.li = torch.tensor(li, device=device)
        self.N = len(kind)
        self.mask = torch.ones(self.N, self.N, dtype=torch.bool, device=device).tril()[None, None]
        used = torch.tensor([2, V, 3], device=device)[self.kind]   # values each token uses
        self.vmask = (torch.arange(V, device=device)[None] < used[:, None]).float()   # [N, V]

    def values(self, codes):
        """Token values [S, N, V] of windows of letter codes [S, L * page]."""
        S = codes.shape[0]
        v = torch.zeros(S, self.N, self.V, device=self.dev)
        lt = self.kind == 1
        v[:, lt] = self.levels[codes[:, self.li[lt]]][..., None].expand(-1, -1, self.V)
        mv = self.kind == 2
        v[:, mv, 0] = 0.5
        v[:, mv, 1] = 0.5
        return v

    def forward(self, codes):
        P, sw = self.P, self.sw
        v = self.values(codes)
        if sw["input"] == "bins":
            pos_ = v.clamp(0, 1) * (self.K - 1)
            lo = pos_.floor().long().clamp(max=self.K - 2)
            w = (pos_ - lo) * self.vmask
            E = P["in_bins"]                                           # [V, K, d]
            vi = torch.arange(self.V, device=self.dev)
            x = (((1 - w) * self.vmask)[..., None] * E[vi, lo] + w[..., None] * E[vi, lo + 1]).sum(-2)
            x = x + P["uni_in_b"]
        else:
            x = F.linear((v - 0.5) * self.vmask, P["uni_in_w"], P["uni_in_b"])
        if sw["input"] == "table":
            lt = self.kind == 1
            x = x.clone()
            x[:, lt] = P["table"][codes[:, self.li[lt]]]
        if sw["pos"] == "factorized":
            pos = P["slot"].repeat(self.L, 1) + P["step"].repeat_interleave(self.spt, 0)
        else:
            pos = P["pos"]
        h = ref.transformer(P, self.mc, x + pos[None], self.mask)
        return h, v

    def losses(self, codes):
        """Per target token: the training loss [S, N-1] (token n predicts token n+1) and, for letter targets, the
        letter's nats [S, N-1] (NaN elsewhere)."""
        P, sw, K = self.P, self.sw, self.K
        h, v = self.forward(codes)
        h, tv = h[:, :-1], v[:, 1:]
        tk = self.kind[1:]
        lg = F.linear(h, P["uni_out_w"], P["uni_out_b"]).reshape(h.shape[0], h.shape[1], self.V, K)
        pos = tv.clamp(0, 1) * (K - 1)
        lo = pos.floor().long().clamp(max=K - 2)
        w = pos - lo
        lp = lg.log_softmax(-1)
        ce = -((1 - w) * lp.gather(-1, lo[..., None])[..., 0] + w * lp.gather(-1, (lo + 1)[..., None])[..., 0])
        vm = self.vmask[1:]
        tok = (ce * vm).sum(-1) / vm.sum(-1)                         # [S, N-1]
        if sw["output"] == "classes":
            lt = tk == 1
            cl = F.linear(h[:, lt], P["cls_w"], P["cls_b"])
            tok = tok.clone()
            tok[:, lt] = F.cross_entropy(cl.transpose(1, 2), codes[:, self.li[1:][lt]], reduction="none")
        letter = torch.where(tk == 1, tok, torch.full_like(tok, float("nan")))
        return tok, letter


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=os.path.join(DEV, "configs", "T_text_B.json"))
    ap.add_argument("--name", required=True)
    for k, ch in (("tokens", ("universal", "letters")), ("input", ("linear", "table", "bins")),
                  ("pos", ("factorized", "absolute")), ("output", ("bins", "classes")), ("init", ("cpp", "gpt"))):
        ap.add_argument("--" + k, default=ch[0], choices=ch)
    ap.add_argument("--updates", type=int, default=30000)
    ap.add_argument("--eval-every", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("-o", "--out", required=True)
    args = ap.parse_args()
    sw = {k: getattr(args, k) for k in ("tokens", "input", "pos", "output", "init")}
    cfg = json.load(open(args.config))
    world = make_world(1, make_config(dict(cfg["env"], split="train")), seed=0)
    torch.manual_seed(args.seed)
    M = Ladder(cfg, world, sw, args.seed, args.device)
    L, page, B = M.L, M.page, cfg["online"].get("batch", 8)
    ppo = dict(cfg.get("ppo", {}))
    opt = torch.optim.AdamW(list(M.P.values()), lr=ppo.get("lr", 3e-4), betas=(0.9, 0.999), eps=1e-5, weight_decay=0.0)
    train = torch.tensor(world.codes, device=args.device)
    win = torch.tensor(val_windows(cfg["env"], 16, L), device=args.device)
    rng = np.random.default_rng([args.seed, 35])
    os.makedirs(args.out, exist_ok=True)
    nparams = sum(p.numel() for p in M.P.values())
    json.dump(dict(command=" ".join(sys.argv), switches=sw, params=nparams, tokens_per_step=M.spt),
              open(os.path.join(args.out, "info.json"), "w"), indent=2)
    print("%s %s params %d tokens/step %d" % (args.name, sw, nparams, M.spt), flush=True)
    log = open(os.path.join(args.out, "metrics.jsonl"), "w")
    t0 = time.time()
    for u in range(args.updates + 1):
        if u % args.eval_every == 0:
            tot, n = 0.0, 0
            with torch.no_grad():
                for j in range(0, len(win), 16):
                    _, let = M.losses(win[j:j + 16])
                    if sw["tokens"] == "universal":
                        let = let[:, 1:]          # target 0 is the window's first letter (from the body token): skipped
                    ok = ~torch.isnan(let)
                    tot += let[ok].sum().item()
                    n += int(ok.sum())
            row = dict(update=u, val=tot / n, letters_scored=n, time=time.time() - t0)
            log.write(json.dumps(row) + "\n")
            log.flush()
            print("%s update %6d  val %.4f nats/letter  %.0f s" % (args.name, u, row["val"], row["time"]), flush=True)
        if u == args.updates:
            break
        st = torch.tensor(rng.integers(0, len(train), B), device=args.device)
        codes = train[(st[:, None] + torch.arange(L * page, device=args.device)[None]) % len(train)]
        tok, _ = M.losses(codes)
        loss = tok.mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(M.P.values()), 0.5)
        opt.step()


if __name__ == "__main__":
    main()

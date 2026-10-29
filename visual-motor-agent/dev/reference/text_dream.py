"""Predicting the future in the text world (task 00040): the universal model dreams (generates) the next pages, a
classical character-level GPT generates the same, from the same contexts.

The universal model generates as it dreams: its window is its last ctx steps (pages); a new page's letters come one
token at a time, each from a forward pass over the window so far (exactly sequential), as a letter (the mean over
the token's values of the log-probability of the letter's level, normalized over the alphabet: letter_logp). A new step's body is the move token's expected prediction (the drives, ~0); each step's move is sampled
from the model's move prediction (as in its lifetime, no instinct), also for the real context pages. GPT generates
letter by letter from its last 512 letters. Starts: the val windows of eval_predict / char_gpt (8 real pages each).

Measures: (1) greedy continuations against the real text: the share of letters right at each position after the
context, and the first error's position; (2) continuations sampled at temperature 1 (--temp), scored by both models
teacher-forced (nats per letter), against the real continuations; (3) text samples.

    ~/venvs/vm-gpu/bin/python reference/text_dream.py --uni RUN/ckpt_00030000.bin --gpt GPT_RUN/params.pt \\
        --config configs/T_text_B.json -o OUT.json
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

DEV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, DEV)

import agent_lib  # noqa: E402
from reference import torch_reference as ref  # noqa: E402
from reference.char_gpt import logits as gpt_logits, val_windows  # noqa: E402
from world import make_config, make_world  # noqa: E402


class Uni:
    """The universal model's forward pass in torch on the device (as torch_reference.forward_uni, which checks it), over
    token values v [S, T, nt, V] in [0, 1]; the context restarts only at the window's first step."""

    def __init__(self, agent, device):
        c = self.cfg = agent.config
        if not c.get("in_bins") or not c.get("gray"):
            raise ValueError("text_dream expects the text configs (gray, in_bins)")
        P = ref.params_from_flat(agent.params(), agent.param_layout(), c, dtype=torch.float32)
        self.P = {k: v.detach().to(device) for k, v in P.items()}
        self.nt, self.V, self.K = ref.uni_dims(c)
        self.ps = torch.tensor(agent_lib.uni_layout(c)["patch"], device=device)   # the letters' slots (task 00047)
        self.page = len(self.ps)
        self.m = torch.zeros(self.nt, self.V, device=device)
        self.m[0, :2] = 1
        self.m[self.ps] = 1
        lay = agent_lib.uni_layout(c)
        self.ss = torch.tensor(lay["summary"], device=device)                    # the summaries' slots
        if c.get("sum_thumb"):                                                   # task 00049: they carry thumbnails
            self.m[self.ss] = 1
        self.m[self.nt - 1, :3] = 1
        al = ref.attn_allow(c)                            # attention regions (task 00044), by slot; None = none
        self.allow = None if al is None else torch.tensor(al, device=device)
        self.device = device

    def fill_thumbs(self, v, t=None):
        """Task 00049: the summaries' values of v [S, T, nt, V] (all steps, or step t) from its patches' values, as
        torch_reference.uni_thumbs: the page rebuilt from the patches (gray), each summary's block box-averaged to one
        patch. A no-op without thumbnails."""
        if not self.cfg.get("sum_thumb"):
            return
        c, ps_, g = self.cfg, self.cfg["patch"], self.cfg["img"] // self.cfg["patch"]
        sel = slice(None) if t is None else slice(t, t + 1 if t != -1 else None)
        pv = v[:, sel][:, :, self.ps]                                            # [S, T', g*g, ps*ps]
        S, T = pv.shape[:2]
        img = pv.reshape(S, T, g, g, ps_, ps_).permute(0, 1, 2, 4, 3, 5).reshape(S, T, g * ps_, g * ps_)
        tab = ref.uni_slot_table(c)
        out = []
        for k in self.ss.tolist():
            if c.get("uni_sum", 0):
                R = c["uni_sum"]
                y0, x0, fy, fx = int(tab["r"][k]) * R * ps_, 0, R, g
            else:
                b = 2 ** int(tab["level"][k])
                y0, x0, fy, fx = int(tab["r"][k]) * b * ps_, int(tab["c"][k]) * b * ps_, b, b
            blk = img[:, :, y0:y0 + fy * ps_, x0:x0 + fx * ps_]
            out.append(blk.reshape(S, T, ps_, fy, ps_, fx).mean((3, 5)).reshape(S, T, ps_ * ps_))
        vv = v[:, sel]
        vv[:, :, self.ss] = torch.stack(out, 2)
        v[:, sel] = vv

    def logits(self, v):
        S, T, nt, V = v.shape
        K, P = self.K, self.P
        pos = v.clamp(0, 1) * (K - 1)
        lo = pos.floor().clamp(max=K - 2)
        w = pos - lo
        hot = torch.zeros(S, T, nt, V, K, device=self.device)
        hot.scatter_(-1, lo.long()[..., None], (1 - w)[..., None])
        hot.scatter_add_(-1, lo.long()[..., None] + 1, w[..., None])
        x = (hot * self.m[..., None]).reshape(S, T * nt, V * K)
        h = F.linear(x, P["uni_in_w"], P["uni_in_b"]).reshape(S, T, nt, -1) + P["uni_slot"][None, None]
        h = (h + P["wpe"][:T][None, :, None]).reshape(S, T * nt, -1)
        mask = torch.ones(T * nt, T * nt, dtype=torch.bool, device=self.device).tril()
        if self.allow is not None:
            slot = torch.arange(T * nt, device=self.device) % nt
            mask = mask & self.allow[slot[:, None], slot[None, :]]
        mask = mask[None, None]
        return F.linear(ref.transformer(P, self.cfg, h, mask), P["uni_out_w"], P["uni_out_b"]).reshape(S, T * nt, V, K)


def letter_logp(lg, lev):
    """Letter log-probabilities [..., n] from a patch target's logits [..., V, K]: each of the token's V values predicts
    the same letter (every value of letter k's glyph is at bin lev[k]: the levels are bin centres with 256 bins); the
    mean of their log-probabilities, normalized over the alphabet. (Their sum, eval_predict's glyph probability, counts
    one prediction V times: a temperature of 1 / V.)"""
    lp = lg.log_softmax(-1)[..., lev]                                      # [..., V, n]
    return lp.mean(-2).log_softmax(-1)


def sample_bins(lg, gen):
    """Sample each value's bin from logits [S, V, K] -> values [S, V] (bin centres)."""
    p = lg.softmax(-1)
    S, V, K = p.shape
    b = torch.multinomial(p.reshape(S * V, K), 1, generator=gen).reshape(S, V)
    return b.float() / (K - 1)


def uni_generate(uni, codes, lev, G, mode, gen, temp=1.0):
    """codes [S, L*page] real context letters (L pages) -> generated letters [S, G*page]. mode: greedy | sample."""
    c, nt, V, ps = uni.cfg, uni.nt, uni.V, uni.ps
    S = codes.shape[0]
    L, page = c["ctx"], uni.page
    levf = torch.tensor(lev, device=uni.device, dtype=torch.float32) / 255.0
    v = torch.zeros(S, L, nt, V, device=uni.device)
    v[:, :, ps] = levf[torch.tensor(codes, device=uni.device).reshape(S, L, page)][..., None]
    uni.fill_thumbs(v)
    for t in range(L):   # the context's moves, sampled step by step, as in its lifetime
        mv = sample_bins(uni.logits(v[:, :t + 1])[:, t * nt + nt - 2, :3], gen)
        v[:, t, nt - 1, :2], v[:, t, nt - 1, 2] = mv[:, :2], (mv[:, 2] > 0.5).float()
    out = []
    lev_t = torch.tensor(lev, device=uni.device)
    for g in range(G):
        lg = uni.logits(v)
        probs = lg[:, -1].softmax(-1)                                      # the last move token: the next body
        body = (probs * torch.linspace(0, 1, uni.K, device=uni.device)).sum(-1)[:, :2]
        v = torch.cat([v[:, 1:], torch.zeros(S, 1, nt, V, device=uni.device)], 1)
        v[:, -1, 0, :2] = body.clamp(0, 1)
        T = v.shape[1]
        for p in range(page):
            lg = uni.logits(v)[:, (T - 1) * nt + ps[p] - 1]                # predicts patch p of the new page
            lp = letter_logp(lg, lev_t)
            k = lp.argmax(-1) if mode == "greedy" else torch.multinomial((lp / temp).softmax(-1), 1, generator=gen)[:, 0]
            v[:, -1, ps[p]] = levf[k][:, None]
            uni.fill_thumbs(v, -1)
            out.append(k)
        mv = sample_bins(uni.logits(v)[:, (T - 1) * nt + nt - 2, :3], gen)
        v[:, -1, nt - 1, :2], v[:, -1, nt - 1, 2] = mv[:, :2], (mv[:, 2] > 0.5).float()
    return torch.stack(out, 1).cpu().numpy()


def uni_score(uni, codes, lev, n_score, gen):
    """Teacher-forced nats per letter of the last n_score letters of codes [S, L*page] (L pages), moves sampled."""
    c, nt, V, ps = uni.cfg, uni.nt, uni.V, uni.ps
    S = codes.shape[0]
    L, page = c["ctx"], uni.page
    levf = torch.tensor(lev, device=uni.device, dtype=torch.float32) / 255.0
    ct = torch.tensor(codes, device=uni.device)
    v = torch.zeros(S, L, nt, V, device=uni.device)
    v[:, :, ps] = levf[ct.reshape(S, L, page)][..., None]
    uni.fill_thumbs(v)
    for t in range(L):
        mv = sample_bins(uni.logits(v[:, :t + 1])[:, t * nt + nt - 2, :3], gen)
        v[:, t, nt - 1, :2], v[:, t, nt - 1, 2] = mv[:, :2], (mv[:, 2] > 0.5).float()
    lg = uni.logits(v).reshape(S, L, nt, V, -1)[:, :, ps - 1]              # the tokens that predict the patches
    lp = letter_logp(lg, torch.tensor(lev, device=uni.device))             # [S, L, page, n]
    nll = -lp.gather(-1, ct.reshape(S, L, page)[..., None])[..., 0].reshape(S, L * page)
    return nll[:, -n_score:].mean().item()


def gpt_generate(P, m, codes, G, mode, gen, temp=1.0):
    x = torch.tensor(codes, device=P["wte"].device)
    ctx = x.shape[1]
    out = []
    for _ in range(G):
        lp = gpt_logits(P, m, x[:, -ctx:])[:, -1].log_softmax(-1)
        k = lp.argmax(-1) if mode == "greedy" else torch.multinomial((lp / temp).softmax(-1), 1, generator=gen)[:, 0]
        out.append(k)
        x = torch.cat([x, k[:, None]], 1)
    return torch.stack(out, 1).cpu().numpy()


def gpt_score(P, m, codes, n_score):
    x = torch.tensor(codes, device=P["wte"].device)
    lp = gpt_logits(P, m, x[:, :-1]).log_softmax(-1)
    nll = -lp.gather(-1, x[:, 1:, None])[..., 0]
    return nll[:, -n_score:].mean().item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--uni", required=True, help="the universal model's checkpoint (a text lifetime)")
    ap.add_argument("--gpt", required=True, help="char_gpt.py's params.pt")
    ap.add_argument("--config", required=True)
    ap.add_argument("--windows", type=int, default=32)
    ap.add_argument("--pages", type=int, default=4, help="pages generated (64 letters each)")
    ap.add_argument("--samples", type=int, default=3, help="texts printed")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--temp", type=float, default=1.0, help="the samples' temperature (both models)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("-o", "--out", required=True)
    args = ap.parse_args()
    with torch.no_grad():
        run(args)


def run(args):
    cfg = json.load(open(args.config))
    agent = agent_lib.Agent.load(args.uni)
    uni = Uni(agent, args.device)
    c = agent.config
    L, page = c["ctx"], uni.page
    world = make_world(1, make_config(dict(cfg["env"], split="val")), seed=0)
    lev = np.asarray(world.levels, np.int64)
    alpha = world.alphabet
    # the torch forward against the reference's (float64, numpy two-hot)
    wv = val_windows(cfg["env"], 16, L)
    rng = np.random.default_rng(args.seed)
    pick = rng.choice(len(wv), min(args.windows, len(wv)), replace=False)
    codes = wv[pick]
    S = 2
    img = np.repeat(np.repeat(lev[codes[:S]].reshape(S, L, 8, 8), 2, -1), 2, -2)[..., None].repeat(3, -1).astype(np.uint8)
    act = rng.uniform(-1, 1, (S, L, 3)).astype(np.float32)
    act[..., 2] = rng.random((S, L)) < 0.5
    want = ref.forward_uni(ref.params_from_flat(agent.params(), agent.param_layout(), c), c, img,
                           np.zeros((S, L, 2), np.float32), act, np.eye(1, L, dtype=np.uint8).repeat(S, 0))["logits"]
    v, _ = ref.uni_values(c, img, np.zeros((S, L, 2)), act)
    got = uni.logits(torch.tensor(v, dtype=torch.float32, device=args.device)).cpu().double()
    err = (got - want.detach()).abs().max().item()
    assert err < 1e-3, "the torch forward differs from the reference: %g" % err
    gp = torch.load(args.gpt)
    P = {k: t.to(args.device) for k, t in gp.items()}
    m = dict(d_model=c["d_model"], n_layer=c["n_layer"], n_head=c["n_head"], d_ff=c["d_ff"])
    gen = torch.Generator(device=args.device).manual_seed(args.seed)
    G, n = args.pages, args.pages * page
    # the real continuations: the G pages after each window
    real = np.stack([world.codes[(i0 + np.arange(L * page, (L + G) * page)) % len(world.codes)]
                     for i0 in window_offsets(cfg, wv, pick, L, page)])
    res = dict(uni=args.uni, gpt=args.gpt, windows=len(codes), pages=G, temp=args.temp, torch_vs_reference=err)
    gens = {}
    for name, f in (("uni", lambda mode: uni_generate(uni, codes, lev, G, mode, gen, args.temp)),
                    ("gpt", lambda mode: gpt_generate(P, m, codes, n, mode, gen, args.temp))):
        for mode in ("greedy", "sample"):
            gens[name, mode] = f(mode)
        right = gens[name, "greedy"] == real
        first = np.where(right.all(1), n, (~right).argmax(1))
        res[name + "_greedy_right"] = {"1": float(right[:, 0].mean()), "2-8": float(right[:, 1:8].mean()),
                                       "9-64": float(right[:, 8:64].mean()), "65-%d" % n: float(right[:, 64:].mean())}
        res[name + "_first_error_median"] = float(np.median(first))
    # scoring: each continuation after the last L - G real pages, by both models
    for name in ("real", "uni", "gpt"):
        cont = real if name == "real" else gens[name, "sample"]
        seq = np.concatenate([codes[:, G * page:], cont], 1)              # L pages: L - G real, G continued
        res["nll_under_gpt_" + name] = gpt_score(P, m, seq, n)
        res["nll_under_uni_" + name] = uni_score(uni, seq, lev, n, gen)
    txt = lambda a: "".join(alpha[k] for k in a)   # noqa: E731
    res["texts"] = [dict(context=txt(codes[i, -192:]), real=txt(real[i]), uni_greedy=txt(gens["uni", "greedy"][i]),
                         gpt_greedy=txt(gens["gpt", "greedy"][i]), uni_sample=txt(gens["uni", "sample"][i]),
                         gpt_sample=txt(gens["gpt", "sample"][i])) for i in range(args.samples)]
    json.dump(res, open(args.out, "w"), indent=1)
    for k, v in res.items():
        if k != "texts":
            print(k, v)
    for t in res["texts"]:
        print("\n" + "=" * 100)
        for k, v in t.items():
            print("--- %s:\n%s" % (k, v))


def window_offsets(cfg, wv, pick, L, page):
    """The val split's letter offset of each picked window (val_windows' layout: 16 tiled readers)."""
    w = make_world(16, make_config(dict(cfg["env"], split="val", tile=True)), seed=0)
    steps = len(w.codes) // page // 16
    steps -= steps % L
    starts = (w.pos[:, None] + np.arange(0, steps, L)[None] * page).reshape(-1)
    return starts[pick]


if __name__ == "__main__":
    main()

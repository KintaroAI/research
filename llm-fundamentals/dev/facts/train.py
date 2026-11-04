"""Task 00007: a GPT-2-shaped model trained from scratch on TinyStories with synthetic facts in the stream; probes of every
kind of fact as it learns. One run covers the three parts of the task:

    base     (part B) 500 facts, each shown 32 times in the first 4,000 steps (a fifth of a shorter run), then never: how fast the stream erodes them
    stream   (part A) 3 new facts a step, each shown n times (1, 2, 4, 8, 16, 32) with g steps between (20, 200)
    test     (part C) 300 facts learned adaptively: shown every 10 steps until recalled (X showings), then not shown until lost
             (not recalled at two probes in a row, 50 steps apart), then after a delay (1, 2 or 4 times the time it took to
             lose it, counted from the loss) shown again until recalled (Y showings). Each relearning starts a fresh fact's
             first learning at the same step, the control.
    reverse  32 facts, each with its own city, shown 16 times 100 steps apart: "Who was born in X? It was" against 32 names

A fact is recalled when its city ranks first among the cities of its kind on the exact probe. Probes are forward passes only.
Outputs (in --out): meta.json, train.jsonl (losses), probe_STEP.npz (base, sampled stream and reverse facts), test.jsonl
(the test facts' events).

    ./venv/bin/python facts/train.py --size S --steps 20000 --out runs/facts/S_s1 --seed 1
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.utils.checkpoint
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from facts_lib import EOT, ENC, Facts, make, ts_tokens  # noqa: E402

SIZES = dict(S=(2, 128, 4), M=(4, 256, 4), L=(6, 384, 6), X=(8, 512, 8), G=(12, 768, 12))   # n_layer, n_embd, n_head
DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "tinystories")
NS, GS = (1, 2, 4, 8, 16, 32), (20, 200)


class Block(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.h, self.ln1, self.ln2 = h, nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv, self.proj = nn.Linear(d, 3 * d), nn.Linear(d, d)
        self.fc, self.out = nn.Linear(d, 4 * d), nn.Linear(4 * d, d)

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(self.ln1(x)).split(C, 2)
        q, k, v = (t.view(B, T, self.h, C // self.h).transpose(1, 2) for t in (q, k, v))
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True).transpose(1, 2).reshape(B, T, C)
        x = x + self.proj(y)
        return x + self.out(F.gelu(self.fc(self.ln2(x)), approximate="tanh"))


class GPT(nn.Module):
    def __init__(self, n_layer, d, h, vocab=50257, block=256):
        super().__init__()
        self.wte, self.wpe = nn.Embedding(vocab, d), nn.Embedding(block, d)
        self.blocks = nn.ModuleList([Block(d, h) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(d)
        for p in self.parameters():
            if p.dim() >= 2:
                nn.init.normal_(p, 0.0, 0.02)

    def hidden(self, idx):
        x = self.wte(idx) + self.wpe(torch.arange(idx.shape[1], device=idx.device))
        for b in self.blocks:
            x = b(x)
        return self.ln_f(x)

    def forward(self, idx):
        return self.hidden(idx) @ self.wte.weight.T                              # tied output

    def loss(self, idx, y, chunks=4):
        """The mean next-token cross-entropy, the logits made a chunk of rows at a time and remade in the backward pass, so
        that only a quarter of the full-vocabulary logits are held at once."""
        h = self.hidden(idx).reshape(-1, self.wte.weight.shape[1])
        y = y.reshape(-1)
        part = lambda hh, yy: F.cross_entropy((hh @ self.wte.weight.T).float(), yy, reduction="sum")   # noqa: E731
        n = (len(y) + chunks - 1) // chunks
        total = sum(torch.utils.checkpoint.checkpoint(part, h[i:i + n], y[i:i + n], use_reentrant=False) for i in range(0, len(y), n))
        return total / len(y)


class Test:
    """One part-C fact (and its fresh control when relearning starts). phase: wait, learn, forget, delay, relearn, done."""

    def __init__(self, i, start, k):
        self.i, self.start, self.k = i, start, k
        self.phase, self.shows, self.next, self.miss = "wait", 0, start, 0
        self.ev = dict(fact=i, start=start, delay_factor=k)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--size", choices=SIZES, default="S")
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=6e-4)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--probe-every", type=int, default=500)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    dev = "cuda"
    T = 256

    # facts: base, stream (3 a step), test and their controls, reverse
    steps = a.steps
    n_base, n_test, n_rev = 500, 300, 32
    n_stream = 3 * steps
    n_neutral = 8                                                                # never shown: the probes' baseline
    n_all = n_base + n_stream + 2 * n_test + n_rev + n_neutral
    cities, names = make(1234, n_all, os.path.join(DATA, "TinyStories_val.bin"))     # the same facts in every run
    frng = np.random.default_rng(1234)
    city_of = frng.integers(0, 256, n_all)
    kind = np.array(["base"] * n_base + ["stream"] * n_stream + ["test"] * n_test + ["fresh"] * n_test + ["reverse"] * n_rev + ["neutral"] * n_neutral)
    rev = np.flatnonzero(kind == "reverse")
    neutral = np.flatnonzero(kind == "neutral")
    city_of[rev] = 256 + np.arange(n_rev)
    facts = Facts(names, cities, city_of)
    n_of, g_of, arrive = np.zeros(n_all, int), np.zeros(n_all, int), np.full(n_all, -1)
    show = [[] for _ in range(steps)]
    for i in range(n_base):                                                      # base: 32 showings in the first 4,000 steps
        n_of[i] = 32
        for t in frng.integers(0, min(4000, steps // 5), 32):
            show[t].append(i)
        arrive[i] = 0
    st = np.flatnonzero(kind == "stream")
    for j, i in enumerate(st):                                                   # stream: 3 new facts a step
        t0 = j // 3
        n_of[i], g_of[i], arrive[i] = NS[j % 6], GS[(j // 6) % 2], t0
        for k in range(n_of[i]):
            if t0 + k * g_of[i] < steps:
                show[t0 + k * g_of[i]].append(i)
    for j, i in enumerate(rev):                                                  # reverse: 16 showings, 100 steps apart
        n_of[i], g_of[i], arrive[i] = 16, 100, 1000 + j
        for k in range(16):
            if 1000 + j + 100 * k < steps:
                show[1000 + j + 100 * k].append(i)
    probe_ids = np.concatenate([np.arange(n_base), st[st % 5 == 0], rev])          # probed every probe_every steps
    tests = [Test(i, int(s), int(k)) for i, s, k in zip(np.flatnonzero(kind == "test"), frng.integers(int(0.1 * steps), int(0.6 * steps), n_test),
                                                        frng.choice([1, 2, 4], n_test))]
    fresh = list(np.flatnonzero(kind == "fresh"))
    json.dump(dict(args=vars(a), n_all=n_all, kind=kind.tolist(), n=n_of.tolist(), g=g_of.tolist(), arrive=arrive.tolist(),
                   city=city_of.tolist(), cities=[ENC.decode([int(c)]) for c in cities], names_head=names[:10]), open(os.path.join(a.out, "meta.json"), "w"))

    L, d, h = SIZES[a.size]
    model = GPT(L, d, h).to(dev)
    non_emb = sum(p.numel() for n_, p in model.named_parameters() if not n_.startswith(("wte", "wpe")))
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, betas=(0.9, 0.95), weight_decay=0.1)
    train, val = ts_tokens(os.path.join(DATA, "TinyStories_train.bin")), ts_tokens(os.path.join(DATA, "TinyStories_val.bin"))
    cand_plain, cand_rev = torch.tensor(cities[:256], device=dev), torch.tensor(cities[256:], device=dev)
    log = open(os.path.join(a.out, "train.jsonl"), "w")
    tlog = open(os.path.join(a.out, "test.jsonl"), "w")
    print("size %s: %d layers, d %d, %d non-embedding parameters, %d facts" % (a.size, L, d, non_emb, n_all), flush=True)

    @torch.no_grad()
    def last_logits(prompts):
        out = torch.zeros(len(prompts), 50257, device=dev)
        by_len = {}
        for j, pr in enumerate(prompts):
            by_len.setdefault(len(pr), []).append(j)
        for n, js in by_len.items():
            x = torch.tensor([prompts[j] for j in js], device=dev)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out[js] = (model.hidden(x)[:, -1] @ model.wte.weight.T).float()
        return out

    @torch.no_grad()
    def probe(ids, form):
        """(p, rank) of each fact's city on the exact or paraphrase probe. p is the city's probability; the rank is among its
        kind's cities, by the city's logit minus its mean logit for the never-shown names in the same form, so that a city
        made likely by its own recent showing, whoever is asked about, does not count as recall."""
        model.eval()
        p, r = np.zeros(len(ids)), np.zeros(len(ids), int)
        prompts = [(facts.exact if form == "exact" else facts.para)[i] for i in ids]
        base = last_logits([(facts.exact if form == "exact" else facts.para)[i] for i in neutral]).mean(0)
        by_len = {}
        for j, pr in enumerate(prompts):
            by_len.setdefault(len(pr), []).append(j)
        for n, js in by_len.items():
            for c in range(0, len(js), 2048):
                jj = js[c:c + 2048]
                x = torch.tensor([prompts[j] for j in jj], device=dev)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    lg = (model.hidden(x)[:, -1] @ model.wte.weight.T).float()
                prob = lg.softmax(-1)
                lg = lg - base
                ci = torch.tensor([int(cities[city_of[ids[j]]]) for j in jj], device=dev)
                own = lg.gather(1, ci[:, None])
                is_rev = torch.tensor([city_of[ids[j]] >= 256 for j in jj], device=dev)
                rank = torch.where(is_rev, (lg[:, cand_rev] > own).sum(1), (lg[:, cand_plain] > own).sum(1))
                r[jj] = rank.cpu().numpy()
                p[jj] = prob.gather(1, ci[:, None])[:, 0].cpu().numpy()
        model.train()
        return p, r

    @torch.no_grad()
    def probe_reverse():
        """Each reverse fact's rank of its own name among the 32 reverse names, by the names' summed log-probability minus
        each name's mean over the 32 questions (a name made likely by its own showing, whatever city is asked, does not count)."""
        model.eval()
        score = np.zeros((n_rev, n_rev))
        for qi, i in enumerate(rev):
            pr = facts.reverse_prompt(i)
            seqs = [pr + facts.name_tokens[j] for j in rev]
            n = max(len(s) for s in seqs)
            x = torch.tensor([s + [EOT] * (n - len(s)) for s in seqs], device=dev)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                lp = model(x).float().log_softmax(-1)
            for cj, s in enumerate(seqs):
                tok = torch.tensor(s[len(pr):], device=dev)
                pos = torch.arange(len(pr) - 1, len(s) - 1, device=dev)
                score[qi, cj] = lp[cj, pos, tok].sum().item()
        model.train()
        score = score - score.mean(0, keepdims=True)
        return (score > np.diag(score)[:, None]).sum(1)

    def batch(t, extra):
        due = show[t] + extra
        per = [[] for _ in range(a.batch)]
        for i in due:
            per[rng.integers(a.batch)].append(facts.sentence[i])
        rows = []
        for fs in per:
            n_text = T + 1 - sum(len(f) for f in fs)
            s = int(rng.integers(0, len(train) - n_text))
            text = np.asarray(train[s:s + n_text]).tolist()
            cuts = sorted(rng.integers(0, n_text, len(fs)).tolist())
            seq, last = [], 0
            for c, f in zip(cuts, fs):
                seq += text[last:c] + f
                last = c
            seq += text[last:]
            rows.append(seq[:T + 1])
        x = torch.tensor(rows, device=dev)
        return x[:, :-1], x[:, 1:]

    pe, re_ = probe(np.arange(n_base), "exact")                                   # before any training: chance
    np.savez_compressed(os.path.join(a.out, "probe_000000.npz"), ids=np.arange(n_base), p_exact=pe, rank_exact=re_, rank_reverse=probe_reverse())
    t0 = time.time()
    for t in range(steps):
        lr = a.lr * min(1.0, (t + 1) / 500)
        for gr in opt.param_groups:
            gr["lr"] = lr
        extra = []
        for q in tests:                                                          # part C: who is shown this step
            if q.phase == "wait" and t >= q.start:
                q.phase, q.next = "learn", t
            if q.phase in ("learn", "relearn") and t >= q.next:
                extra.append(q.i)
                q.shows += 1
                q.next = t + 10
            if q.phase == "delay" and t >= q.wake:
                q.phase, q.shows, q.next = "relearn", 0, t
                f = Test(fresh.pop(), t, 0)                                      # the control: a fresh fact, first learned now
                f.phase, f.next, f.ev["control_for"] = "learn", t, int(q.i)
                f.fresh = True
                tests.append(f)
        x, y = batch(t, extra)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = model.loss(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if t % 10 == 9:                                                          # part C: probe the active test facts
            act = [q for q in tests if q.phase in ("learn", "relearn", "forget")]
            if act:
                p, r = probe([q.i for q in act], "exact")
                for q, pi, ri in zip(act, p, r):
                    hit = ri == 0
                    if q.phase in ("learn", "relearn"):
                        key = "X" if q.phase == "learn" else "Y"
                        if hit or q.shows >= 64:
                            q.ev[key] = q.shows if hit else None
                            q.ev["t_" + key] = t
                            if q.phase == "learn" and hit and not getattr(q, "fresh", False):
                                q.phase, q.miss, q.t_learned = "forget", 0, t
                            else:
                                q.phase = "done"
                                tlog.write(json.dumps(q.ev, default=lambda o: o.item()) + "\n")
                    elif t % 50 == 49:                                           # forgetting: two misses, 50 steps apart
                        q.miss = q.miss + 1 if not hit else 0
                        if q.miss >= 2:
                            q.ev.update(t_lost=t, p_lost=float(pi), lost_after=t - q.t_learned)
                            q.phase, q.wake = "delay", t + (q.k - 1) * (t - q.t_learned)
        if t % 100 == 0:
            log.write(json.dumps(dict(step=t, loss=float(loss.detach()), lr=lr, time=time.time() - t0)) + "\n")
            log.flush()
        if t % 1000 == 999:
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                model.eval()
                vl = []
                for _ in range(10):
                    s = rng.integers(0, len(val) - T - 1, a.batch)
                    v = torch.tensor(np.stack([np.asarray(val[j:j + T + 1]) for j in s]).astype(np.int64), device=dev)
                    vl.append(float(model.loss(v[:, :-1], v[:, 1:])))
                model.train()
            log.write(json.dumps(dict(step=t, val=float(np.mean(vl)))) + "\n")
            print("step %6d  loss %.3f  val %.3f  %.1f steps/s" % (t + 1, float(loss.detach()), np.mean(vl), (t + 1) / (time.time() - t0)), flush=True)
        if t % a.probe_every == a.probe_every - 1 or t == steps - 1:
            ids = probe_ids[arrive[probe_ids] <= t]
            pe, re_ = probe(ids, "exact")
            pp, rp = probe(ids, "para")
            np.savez_compressed(os.path.join(a.out, "probe_%06d.npz" % (t + 1)), ids=ids, p_exact=pe, rank_exact=re_, p_para=pp, rank_para=rp,
                                rank_reverse=probe_reverse())
    for q in tests:                                                              # the unfinished, as they stand
        if q.phase != "done":
            q.ev["phase_at_end"] = q.phase
            tlog.write(json.dumps(q.ev, default=lambda o: o.item()) + "\n")
    tlog.close()
    log.close()
    torch.save(model.state_dict(), os.path.join(a.out, "model.pt"))


if __name__ == "__main__":
    main()

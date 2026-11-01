"""Saying it out loud, in text (task 00078, vm-00068): one part of the model teaches another.

Half 1 predicts the next letter it hears. Half 2 reads the same input and is trained only against half 1's distribution.
On every tick half 2 says a letter; the world masks it: the real letter is heard where it comes, half 2's where the
word is missing, so there are no gaps while the model speaks. It speaks along, and is heard only when the world stops.

    ~/venvs/vm-gpu/bin/python reference/mimic.py --arm B --q 0.3 --seed 1 -o OUT_DIR

Arms: Z no gaps; N half 2 silent, blanks heard; A half 1 frozen when the gaps begin; B half 1 learns from all it hears;
S as B, a letter weighted by min(1, CE / --slow); E as B, half 1 learns only the world's letters (it is told which are
its own, a control).
"""

import argparse
import json
import math
import os
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

CORPUS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "llm-fundamentals", "dev", "data",
                      "tinyshakespeare", "input.txt")
ARMS = ("Z", "N", "A", "B", "S", "E")
GAP_LENGTHS = (1, 2, 4, 8, 16)


class Block(nn.Module):
    def __init__(self, d, heads):
        super().__init__()
        self.heads = heads
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv, self.proj = nn.Linear(d, 3 * d), nn.Linear(d, d)
        self.ff = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, x):
        B, T, D = x.shape
        q, k, v = self.qkv(self.ln1(x)).view(B, T, 3, self.heads, D // self.heads).permute(2, 0, 3, 1, 4)
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.proj(a.transpose(1, 2).reshape(B, T, D))
        return x + self.ff(self.ln2(x))


class Half(nn.Module):
    """A causal character transformer. Its vocabulary has one more entry than the alphabet: the blank."""

    def __init__(self, vocab, ctx, d=128, layers=4, heads=4):
        super().__init__()
        self.tok, self.pos = nn.Embedding(vocab, d), nn.Embedding(ctx, d)
        self.blocks = nn.ModuleList(Block(d, heads) for _ in range(layers))
        self.ln, self.out = nn.LayerNorm(d), nn.Linear(d, vocab)

    def forward(self, x):
        h = self.tok(x) + self.pos(torch.arange(x.shape[1], device=x.device))
        for b in self.blocks:
            h = b(h)
        return self.out(self.ln(h))


def load_corpus(path=CORPUS, frac=0.9):
    """Train and val letters and the alphabet, as the text world splits them."""
    text = open(path, encoding="utf-8").read()
    chars = sorted(set(text))
    ix = {c: i for i, c in enumerate(chars)}
    data = torch.tensor([ix[c] for c in text], dtype=torch.long)
    n = int(len(data) * frac)
    return data[:n], data[n:], chars


def omit_rate(tick, warm, ramp, q):
    """The chance that a word is omitted: none in the warm-up, then rising to q over ramp ticks."""
    return q * min(1.0, max(0.0, (tick - warm) / ramp))


def world_step(letters, spoken, in_gap, starts_word, is_space, rate, gen, silent, blank):
    """The world's one tick for R readers. letters: the true next letters. A reader that is at a word start may enter a
    gap, which lasts until the next space. Returns what each hears, whether it was its own, and the new gap state."""
    enter = starts_word & ~is_space & (torch.rand(letters.shape, device=letters.device, generator=gen) < rate)
    gap = (in_gap | enter) & ~is_space
    heard = torch.where(gap, torch.full_like(letters, blank) if silent else spoken, letters)
    return heard, gap


def half1_weights(ce, own, arm, slow):
    """What each heard letter counts for in half 1's loss."""
    if arm == "S":
        return (ce.detach() / slow).clamp(max=1.0)
    if arm == "E":
        return (~own).float()
    return torch.ones_like(ce)


def update(h1, h2, o1, o2, x, y, own, arm, slow, freeze, clip=1.0):
    """One update of both halves on one batch of heard windows. Half 2's only target is half 1's distribution."""
    l1 = h1(x)
    ce = F.cross_entropy(l1.flatten(0, 1), y.flatten(), reduction="none").view_as(y)
    w = half1_weights(ce, own, arm, slow)
    if not freeze:
        loss1 = (ce * w).mean()
        o1.zero_grad(set_to_none=True)
        loss1.backward()
        nn.utils.clip_grad_norm_(h1.parameters(), clip)
        o1.step()
    p = F.softmax(l1.detach(), -1)
    loss2 = -(p * F.log_softmax(h2(x), -1)).sum(-1).mean()
    o2.zero_grad(set_to_none=True)
    loss2.backward()
    nn.utils.clip_grad_norm_(h2.parameters(), clip)
    o2.step()
    return dict(ce=ce.mean().item(), ce_own=ce[own].mean().item() if own.any() else None, weight=w.mean().item(),
                kl=(loss2 + (p * torch.log(p.clamp_min(1e-12))).sum(-1).mean()).item())


@torch.no_grad()
def say(h, ctx, blank, greedy, gen):
    """The letter a half says next, never the blank."""
    logits = h(ctx)[:, -1].clone()
    logits[:, blank] = -math.inf
    if greedy:
        return logits.argmax(-1)
    return torch.multinomial(F.softmax(logits, -1), 1, generator=gen)[:, 0]


@torch.no_grad()
def clean_ce(h, val, ctx, device, windows=256):
    n = (len(val) - 1) // ctx
    idx = torch.linspace(0, n - 1, min(windows, n)).long() * ctx
    x = torch.stack([val[i:i + ctx] for i in idx]).to(device)
    y = torch.stack([val[i + 1:i + ctx + 1] for i in idx]).to(device)
    return F.cross_entropy(h(x).flatten(0, 1), y.flatten()).item()


@torch.no_grad()
def rollout(speaker, ctx, truth, blank, greedy, gen):
    """The speaker fills len(truth[0]) letters, hearing itself. Returns the letters it said."""
    x, out = ctx.clone(), []
    for _ in range(truth.shape[1]):
        c = say(speaker, x[:, -ctx.shape[1]:], blank, greedy, gen)
        out.append(c)
        x = torch.cat([x, c[:, None]], 1)
    return torch.stack(out, 1)


@torch.no_grad()
def after_gap(h1, ctx, fill, after, T):
    """Half 1's CE on the real letters after a gap, with the given letters in the gap."""
    s = torch.cat([ctx, fill, after], 1)
    n, ces = after.shape[1], []
    for k in range(n):
        end = s.shape[1] - n + k
        ces.append(F.cross_entropy(h1(s[:, end - T:end])[:, -1], s[:, end], reduction="none"))
    return torch.stack(ces, 1).mean().item()


@torch.no_grad()
def evaluate_gaps(h1, h2, val, chars, T, blank, device, seed, n=2048, tail=16):
    """Filling, whole words and repair, on starts of the held-out text."""
    gen = torch.Generator(device=device).manual_seed(seed)
    m = max(GAP_LENGTHS)
    idx = torch.linspace(T, len(val) - m - tail - 1, n).long()
    ctx = torch.stack([val[i - T:i] for i in idx]).to(device)
    truth = torch.stack([val[i:i + m] for i in idx]).to(device)
    after = torch.stack([val[i + 8:i + 8 + tail] for i in idx]).to(device)
    common = torch.bincount(val).argmax().item()
    r = dict(bottom_common=(truth == common).float().mean().item())
    g2, s2 = rollout(h2, ctx, truth, blank, True, gen), rollout(h2, ctx, truth, blank, False, gen)
    g1, s1 = rollout(h1, ctx, truth, blank, True, gen), rollout(h1, ctx, truth, blank, False, gen)
    full = torch.cat([ctx, truth], 1)
    tf = torch.stack([h1(full[:, k:k + T])[:, -1].argmax(-1) for k in range(m)], 1)
    for L in GAP_LENGTHS:
        for name, f in (("h2_greedy", g2), ("h2_sampled", s2), ("h1_greedy", g1), ("h1_sampled", s1), ("h1_true_ctx", tf)):
            r[f"acc{L}_{name}"] = (f[:, :L] == truth[:, :L]).float().mean().item()
    fill8 = dict(true=truth[:, :8], h2=s2[:, :8], h1=s1[:, :8], blank=torch.full_like(truth[:, :8], blank))
    for name, f in fill8.items():
        r[f"after_{name}"] = after_gap(h1, ctx, f, after, T)
    # whole words: from each word start of up to 8 letters, greedy
    space = [i for i, c in enumerate(chars) if c in " \n"]
    sp = torch.isin(val, torch.tensor(space))
    starts = torch.nonzero(sp[T - 1:-m - 2] & ~sp[T:-m - 1])[:, 0] + T
    starts = starts[torch.linspace(0, len(starts) - 1, n).long()]
    c = torch.stack([val[i - T:i] for i in starts]).to(device)
    t = torch.stack([val[i:i + m] for i in starts]).to(device)
    isp = torch.isin(t, torch.tensor(space, device=device))
    length = isp.float().argmax(1)
    ok = (length >= 2) & (length <= 8) & isp.any(1)
    for name, h in (("h2", h2), ("h1", h1)):
        f = rollout(h, c, t, blank, True, gen)
        right = torch.stack([(f[i, :length[i]] == t[i, :length[i]]).all() for i in torch.nonzero(ok)[:, 0]])
        r[f"word_{name}"] = right.float().mean().item()
    r["words"] = int(ok.sum())
    return r


@torch.no_grad()
def sample_text(h2, val, chars, T, blank, device, seed, n=6):
    """Held-out lines with a word or two taken away and said by half 2."""
    gen = torch.Generator(device=device).manual_seed(seed)
    lines = []
    sp = [c in " \n" for c in chars]
    for k in range(n):
        i = T + (len(val) - T - 200) * (k + 1) // (n + 1)
        s = val[i - T:i].tolist()
        shown, gap_left = [], 0
        for j in range(100):
            c = val[i + j].item()
            if gap_left == 0 and sp[s[-1]] and not sp[c] and j % 3 == 0:
                gap_left = 1
            if gap_left and sp[c]:
                gap_left = 0
            if gap_left:
                x = torch.tensor([s[-T:]], device=device)
                c = say(h2, x, blank, False, gen).item()
                shown.append("[" + chars[c] + "]")
            else:
                shown.append(chars[c])
            s.append(c)
        lines.append("".join(shown).replace("][", ""))
    return lines


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", required=True, choices=ARMS)
    ap.add_argument("--q", type=float, default=0.3)
    ap.add_argument("--ticks", type=int, default=12000)
    ap.add_argument("--warm", type=int, default=4000)
    ap.add_argument("--ramp", type=int, default=2000)
    ap.add_argument("--readers", type=int, default=256)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--ctx", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--slow", type=float, default=2.0)
    ap.add_argument("--eval-every", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("-o", "--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    json.dump(vars(a), open(os.path.join(a.out, "args.json"), "w"), indent=1)
    dev = torch.device(a.device)
    torch.manual_seed(a.seed)
    gen = torch.Generator(device=dev).manual_seed(a.seed + 1000)
    train, val, chars = load_corpus()
    V, blank, T, R = len(chars), len(chars), a.ctx, a.readers
    h1, h2 = Half(V + 1, T).to(dev), Half(V + 1, T).to(dev)
    o1, o2 = torch.optim.Adam(h1.parameters(), lr=a.lr), torch.optim.Adam(h2.parameters(), lr=a.lr)
    corpus = train.to(dev)
    space = torch.tensor([i for i, c in enumerate(chars) if c in " \n"], device=dev)
    N = a.ticks + T + 1
    heard = torch.zeros(R, N, dtype=torch.long, device=dev)
    own = torch.zeros(R, N, dtype=torch.bool, device=dev)
    pos = torch.randint(0, len(train) - 1, (R,), device=dev, generator=gen)
    for k in range(T):
        heard[:, k] = corpus[(pos + k) % len(train)]
    pos = pos + T
    in_gap = torch.zeros(R, dtype=torch.bool, device=dev)
    log = open(os.path.join(a.out, "metrics.jsonl"), "w")
    t0, stats, spoken_total = time.time(), [], 0
    for tick in range(a.ticks):
        now = T + tick
        rate = 0.0 if a.arm == "Z" else omit_rate(tick, a.warm, a.ramp, a.q)
        letters = corpus[pos % len(train)]
        spoken = say(h2, heard[:, now - T:now], blank, False, gen) if rate > 0 else letters
        prev_space = torch.isin(heard[:, now - 1], space)
        is_space = torch.isin(letters, space)
        got, in_gap = world_step(letters, spoken, in_gap, prev_space, is_space, rate, gen, a.arm == "N", blank)
        heard[:, now], own[:, now] = got, in_gap
        spoken_total += int(in_gap.sum())
        pos = pos + 1
        rr = torch.randint(0, R, (a.batch,), device=dev, generator=gen)
        tt = torch.randint(0, now - T + 1, (a.batch,), device=dev, generator=gen)
        ix = tt[:, None] + torch.arange(T + 1, device=dev)
        w = heard[rr[:, None], ix]
        stats.append(update(h1, h2, o1, o2, w[:, :-1], w[:, 1:], own[rr[:, None], ix[:, 1:]], a.arm, a.slow,
                            a.arm == "A" and tick >= a.warm))
        if (tick + 1) % a.eval_every == 0:
            h1.eval(), h2.eval()
            rec = dict(step=tick + 1, val_h1=clean_ce(h1, val, T, dev), val_h2=clean_ce(h2, val, T, dev), rate=rate,
                       own_share=spoken_total / (R * (tick + 1)), tps=(tick + 1) / (time.time() - t0),
                       ce=sum(s["ce"] for s in stats) / len(stats), kl=sum(s["kl"] for s in stats) / len(stats),
                       weight=sum(s["weight"] for s in stats) / len(stats))
            co = [s["ce_own"] for s in stats if s["ce_own"] is not None]
            rec["ce_own"] = sum(co) / len(co) if co else None
            stats = []
            log.write(json.dumps(rec) + "\n"), log.flush()
            print(json.dumps({k: round(v, 3) if isinstance(v, float) else v for k, v in rec.items()}), flush=True)
    r = evaluate_gaps(h1, h2, val, chars, T, blank, dev, a.seed)
    r.update(val_h1=clean_ce(h1, val, T, dev), val_h2=clean_ce(h2, val, T, dev), samples=sample_text(h2, val, chars, T, blank, dev, a.seed))
    json.dump(r, open(os.path.join(a.out, "final.json"), "w"), indent=1)
    torch.save(dict(h1=h1.state_dict(), h2=h2.state_dict()), os.path.join(a.out, "halves.pt"))
    print("finished", flush=True)


if __name__ == "__main__":
    main()

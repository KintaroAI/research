"""Task 00007: tables from facts/train.py's runs. Each RUN is a run directory; runs are grouped by their size.

    ./venv/bin/python facts/analyze.py RUN [RUN ...]

    repetition  stream facts: recall (exact, paraphrase) by showings n, at the end and at their best, by size
    forgetting  stream facts with n = 32: recall by probe steps since their last showing
    base        the 500 base facts: recall over the run (part B)
    reverse     recall of the 32 reverse facts at the end
    capacity    facts recalled at once at the end, and the bits that makes (8 per fact) per non-embedding parameter
    savings     part C: X, Y and the fresh controls' X, fact by fact; savings (X - Y) / X; by the delay factor
"""
import glob
import json
import os
import sys

import numpy as np

SIZES = dict(S=(2, 128), M=(4, 256), L=(6, 384), X=(8, 512), G=(12, 768))


def non_emb(size):
    L, d = SIZES[size]
    return L * (12 * d * d + 13 * d) + 2 * d


def load(run):
    m = json.load(open(os.path.join(run, "meta.json")))
    probes = {int(os.path.basename(f)[6:12]): dict(np.load(f)) for f in sorted(glob.glob(os.path.join(run, "probe_*.npz")))}
    ev = [json.loads(line) for line in open(os.path.join(run, "test.jsonl"))]
    return m, probes, ev


def last_show(m, i, t):
    a, n, g = m["arrive"][i], m["n"][i], m["g"][i]
    k = min(n - 1, (t - a) // g) if g else n - 1
    return a + k * g


def main(runs):
    by = {}
    for r in runs:
        m, P, ev = load(r)
        by.setdefault(m["args"]["size"], []).append((m, P, ev))
    for size, R in sorted(by.items(), key=lambda kv: list(SIZES).index(kv[0])):
        print("== size %s, %d non-embedding parameters, %d runs" % (size, non_emb(size), len(R)))
        rep, forget, base, revr, cap, sav = {}, {}, [], [], [], []
        for m, P, ev in R:
            kind, n = np.array(m["kind"]), np.array(m["n"])
            steps = sorted(P)
            end = P[steps[-1]]
            ids, re_, rp = end["ids"], end["rank_exact"], end["rank_para"]
            for nn in (1, 2, 4, 8, 16, 32):
                sel = (kind[ids] == "stream") & (n[ids] == nn)
                best = np.zeros(sel.sum(), bool)
                for t in steps[1:]:
                    z = P[t]
                    pos = {int(i): j for j, i in enumerate(z["ids"])}
                    best |= np.array([z["rank_exact"][pos[int(i)]] == 0 if int(i) in pos else False for i in ids[sel]])
                rep.setdefault(nn, []).append(((re_[sel] == 0).mean(), (rp[sel] == 0).mean(), best.mean()))
            for t in steps[1:]:
                z = P[t]
                sel = (kind[z["ids"]] == "stream") & (n[z["ids"]] == 32)
                for i, rk in zip(z["ids"][sel], z["rank_exact"][sel]):
                    ls = last_show(m, int(i), t)
                    if ls <= t:
                        forget.setdefault(int((t - ls) // 1000), []).append(rk == 0)
            base.append([(t, (P[t]["rank_exact"][kind[P[t]["ids"]] == "base"] == 0).mean()) for t in steps])
            revr.append((end["rank_reverse"] == 0).mean() if end["rank_reverse"].size else np.nan)
            cap.append(int((re_ == 0).sum()))
            X = {e["fact"]: e for e in ev}
            for e in ev:
                if "control_for" in e and e.get("X") is not None:
                    q = X.get(e["control_for"])
                    if q and q.get("X") and q.get("Y") is not None:
                        sav.append((q["X"], q["Y"], e["X"], q["delay_factor"], q.get("p_lost", np.nan)))
        print("  repetition: showings n -> recall at the end (exact, paraphrase), ever recalled")
        for nn, v in rep.items():
            v = np.array(v)
            print("    n %2d  %.3f  %.3f  %.3f" % (nn, *v.mean(0)))
        print("  forgetting (n = 32): recall by thousands of steps since the last showing")
        print("    " + "  ".join("%dk %.3f (%d)" % (k, np.mean(v), len(v)) for k, v in sorted(forget.items())))
        print("  base facts' recall over the run: " + "  ".join("%d %.3f" % (t, np.mean([b[j][1] for b in base])) for j, (t, _) in enumerate(base[0])))
        print("  reverse recall at the end %.3f (chance .031); facts recalled at the end %.0f, %.2f bits per non-embedding parameter" % (
            np.nanmean(revr), np.mean(cap), np.mean(cap) * 8 / non_emb(size)))
        if sav:
            s = np.array(sav)
            print("  savings: %d facts relearned with a control.  X median %.1f  Y median %.1f  fresh X median %.1f" % (
                len(s), np.median(s[:, 0]), np.median(s[:, 1]), np.median(s[:, 2])))
            print("    Y < X %.2f, Y = X %.2f, Y > X %.2f;  Y < fresh X %.2f;  mean savings (X - Y) / X %.2f, against the fresh fact (Xf - Y) / Xf %.2f" % (
                (s[:, 1] < s[:, 0]).mean(), (s[:, 1] == s[:, 0]).mean(), (s[:, 1] > s[:, 0]).mean(), (s[:, 1] < s[:, 2]).mean(),
                ((s[:, 0] - s[:, 1]) / s[:, 0]).mean(), ((s[:, 2] - s[:, 1]) / s[:, 2]).mean()))
            for k in (1, 2, 4):
                q = s[s[:, 3] == k]
                if len(q):
                    print("    delay factor %d: %d facts, Y median %.1f, fresh X median %.1f, (Xf - Y) / Xf %.2f" % (
                        k, len(q), np.median(q[:, 1]), np.median(q[:, 2]), ((q[:, 2] - q[:, 1]) / q[:, 2]).mean()))


if __name__ == "__main__":
    main(sys.argv[1:])

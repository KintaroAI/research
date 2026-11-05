"""Task 00007: tables from facts/train.py's runs. Each RUN is a run directory; runs are grouped by their size.

    ./venv/bin/python facts/analyze.py RUN [RUN ...]

    repetition  stream facts: recall (exact, paraphrase) by showings n, at the end and at their best, by size
    forgetting  stream facts with n = 32: recall by probe steps since their last showing
    base        the 500 base facts: recall over the run (part B)
    reverse     recall of the 32 reverse facts at the end
    capacity    facts recalled at once at the end, and the bits that makes (8 per fact) per non-embedding parameter
    savings     part C: X, Y and the fresh controls' X, fact by fact; savings (X - Y) / X; by the delay factor
    spacing     task 00008 (--plus 1): facts shown 8 times all in one step, 10, 100 or 1,000 steps apart: recall by thousands of steps since the
                last showing and ever, 95% intervals by bootstrap over facts (seeds pooled), and each seed's mean over 1-4k
    wordings    task 00008: facts shown 8 times 100 steps apart in four wordings against one wording: exact and held-out paraphrase recall

Runs are grouped by size and arm: C followed by the learning rate's first digit (constant) or D (cosine decay); 00020's runs are C6.
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


def arm(m):
    a = m["args"]
    return "D" if a.get("lr_sched", "const") == "cos" else "C%s" % ("%.0e" % a["lr"])[0]


def boot(groups, n=1000, seed=0):
    """The mean of the pooled hits and a 95% interval, resampling facts (each a list of hits)."""
    groups = [g for g in groups if len(g)]
    if not groups:
        return np.nan, np.nan, np.nan
    s, c = np.array([np.sum(g) for g in groups], float), np.array([len(g) for g in groups], float)
    rng = np.random.default_rng(seed)
    bs = [s[j].sum() / c[j].sum() for j in (rng.integers(0, len(s), len(s)) for _ in range(n))]
    return s.sum() / c.sum(), *np.percentile(bs, [2.5, 97.5])


BINS = ((0, 1), (1, 2), (2, 4), (4, 8), (8, 99))


def plus_tables(R):
    """Task 00008's spacing and wording facts: {condition: {bin: {fact: [hit...]}}} for the exact and paraphrase probes, ever recalled,
    and each seed's 1-4k mean."""
    rec = {}
    for si, (m, P, ev) in enumerate(R):
        kind, g, arr = np.array(m["kind"]), np.array(m["g"]), np.array(m["arrive"])
        steps = m["args"]["steps"]
        for t, z in P.items():
            if t == 0:
                continue
            ids = z["ids"]
            sel = np.isin(kind[ids], ("space", "word"))
            for i, re_, rp in zip(ids[sel], z["rank_exact"][sel], z["rank_para"][sel]):
                last = arr[i] + 7 * g[i]
                if last >= steps:
                    continue                                                     # not all 8 showings in the run
                cond = ("words" if kind[i] == "word" else "gap %d" % g[i])
                key = (si, int(i))
                d = rec.setdefault(cond, dict(exact={}, para={}, ever_e={}, ever_p={}, seed={}))
                d["ever_e"][key] = d["ever_e"].get(key, False) or (t > arr[i] and re_ == 0)
                d["ever_p"][key] = d["ever_p"].get(key, False) or (t > arr[i] and rp == 0)
                if t <= last:
                    continue
                k = (t - last) / 1000
                for b in BINS:
                    if b[0] <= k < b[1]:
                        d["exact"].setdefault(b, {}).setdefault(key, []).append(re_ == 0)
                        d["para"].setdefault(b, {}).setdefault(key, []).append(rp == 0)
                if 1 <= k < 4:
                    d["seed"].setdefault(si, []).append(re_ == 0)
    return rec


def show_plus(R):
    rec = plus_tables(R)
    if not rec:
        return
    order = [c for c in ("gap 0", "gap 10", "gap 100", "gap 1000", "words") if c in rec]
    fmt = lambda v: "%.3f [%.3f, %.3f]" % v                                       # noqa: E731
    print("  spacing and wordings (task 00008): recall by thousands of steps since the last of 8 showings, 95% by facts")
    print("    %-9s %s   ever" % ("", "  ".join("%-22s" % ("%g-%gk" % b if b[1] < 99 else "%g+k" % b[0]) for b in BINS)))
    for probe, ever in (("exact", "ever_e"), ("para", "ever_p")):
        print("    %s probe" % probe)
        for c in order:
            d = rec[c]
            row = [fmt(boot(list(d[probe].get(b, {}).values()))) for b in BINS]
            ev = list(d[ever].values())
            print("    %-9s %s   %.3f (%d facts)" % (c, "  ".join("%-22s" % r for r in row), np.mean(ev), len(ev)))
    print("    1-4k, exact, by seed: " + "  ".join("%s %s" % (c, " ".join("%.3f" % np.mean(v) for _, v in sorted(rec[c]["seed"].items()))) for c in order))
    if "gap 0" in rec and "gap 100" in rec:
        a = {k: np.mean(v) for b in BINS[1:3] for k, v in rec["gap 100"]["exact"].get(b, {}).items()}
        z = {k: np.mean(v) for b in BINS[1:3] for k, v in rec["gap 0"]["exact"].get(b, {}).items()}
        rng = np.random.default_rng(1)
        A, Z = np.array(list(a.values())), np.array(list(z.values()))
        bs = [rng.choice(A, len(A)).mean() - rng.choice(Z, len(Z)).mean() for _ in range(1000)]
        print("    gap 100 minus gap 0, exact, 1-4k (per-fact means): %+.3f [%.3f, %.3f]" % (A.mean() - Z.mean(), *np.percentile(bs, [2.5, 97.5])))
    if "words" in rec and "gap 100" in rec:
        W, O = (np.array(list(rec[c]["ever_p"].values()), float) for c in ("words", "gap 100"))
        rng = np.random.default_rng(2)
        bs = [rng.choice(W, len(W)).mean() - rng.choice(O, len(O)).mean() for _ in range(1000)]
        print("    words minus gap 100, ever recalled on the paraphrase: %+.3f [%.3f, %.3f] (words %.3f, chance .004)" % (
            W.mean() - O.mean(), *np.percentile(bs, [2.5, 97.5]), W.mean()))


def main(runs):
    by = {}
    for r in runs:
        m, P, ev = load(r)
        by.setdefault((m["args"]["size"], arm(m)), []).append((m, P, ev))
    for (size, am), R in sorted(by.items(), key=lambda kv: (list(SIZES).index(kv[0][0]), kv[0][1])):
        print("== size %s, arm %s, %d non-embedding parameters, %d runs" % (size, am, non_emb(size), len(R)))
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
        if 0 in forget and 1 in forget:
            print("    kept at 1-2k as a share of 0-1k: %.2f" % (np.mean(forget[1]) / np.mean(forget[0])))
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
        show_plus(R)


if __name__ == "__main__":
    main(sys.argv[1:])

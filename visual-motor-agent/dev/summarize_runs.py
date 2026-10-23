"""Sample-efficiency table from eval.jsonl of several runs (Markdown to stdout).

Usage:
    ./venv/bin/python summarize_runs.py RUN1 RUN2 ... [--labels A A C C] [--budgets 262144 507904]
Columns: env steps until eval success first reaches the thresholds (stochastic >= 0.95, deterministic
>= 0.9), stochastic/deterministic success at fixed budgets (nearest eval at or before the budget), and
the final median steps to success.
"""
import argparse
import json
import os


def first_reach(rows, key, thr):
    for r in rows:
        if r.get(key, 0) >= thr:
            return r["steps"]
    return None


def at_budget(rows, key, budget):
    best = None
    for r in rows:
        if r["steps"] <= budget:
            best = r.get(key)
    return best


def fmt_steps(s):
    return "—" if s is None else "%dk" % round(s / 1000)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--labels", nargs="*")
    ap.add_argument("--budgets", nargs="*", type=int, default=[262144, 507904])
    args = ap.parse_args()
    labels = args.labels or [os.path.basename(r.rstrip("/")) for r in args.runs]
    head = ["run", "steps to succ ≥ .95", "steps to det ≥ .9"]
    for b in args.budgets:
        head += ["succ / det @%dk" % round(b / 1000)]
    head += ["final median steps", "last eval step"]
    print("| " + " | ".join(head) + " |")
    print("|" + "---|" * len(head))
    for run, lab in zip(args.runs, labels):
        p = os.path.join(run, "eval.jsonl")
        rows = [json.loads(l) for l in open(p)] if os.path.exists(p) else []
        if not rows:
            print("| %s | (no eval yet) |" % lab)
            continue
        cells = [lab, fmt_steps(first_reach(rows, "success", 0.95)), fmt_steps(first_reach(rows, "det_success", 0.9))]
        for b in args.budgets:
            s, d = at_budget(rows, "success", b), at_budget(rows, "det_success", b)
            cells.append("—" if s is None else "%.2f / %.2f" % (s, d))
        cells.append("%.0f" % rows[-1].get("steps_to_success_median", float("nan")))
        cells.append(fmt_steps(rows[-1]["steps"]))
        print("| " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()

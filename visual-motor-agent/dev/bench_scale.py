"""Does the cost of a tick grow in line with the world when the transformers grow with it (task 00084)? The world's side doubles, so its area, its things and its
transformers go up 4 times, and each transformer keeps the same patches. Every summary against summaries nearby (model.band_win). Alone on one GPU, without MPS:
a short life, timed after the warm-up, by its parts. Rows of JSON on stdout.

    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 ./venv/bin/python bench_scale.py --ticks 1500 -o OUT > scaling.jsonl
    ./venv/bin/python bench_scale.py --plot scaling.jsonl scaling.png
"""
import argparse
import json
import os
import subprocess
import sys
import threading

import agent_lib

CELLS = ((64, 4, 3, 1, 2, 8), (128, 16, 4, 4, 8, 32), (256, 64, 5, 16, 32, 128))   # side, transformers, pyramid levels, foods, poisons, dots


def tokens(side, pieces, pyr, win):
    """Tokens of each piece per step, fewest and most."""
    n = []
    for k in range(pieces):
        c = dict(img=side, patch=8, band_n=pieces, band_k=k, band_cols=2, band_win=win, uni_pyr=pyr)
        n.append(16 + 2 + sum(agent_lib.has_sum(c, lv, r, q) for lv in range(1, pyr + 1) for r in range(side // 8 >> lv) for q in range(side // 8 >> lv)))
    return min(n), max(n)


def plot(path, out):
    """The model's time per tick by world, every summary against log summaries, with what linear would be."""
    from readme_images import COLORS, GRID, INK, MUTED_INK, SURFACE, plt, save

    rows = [json.loads(line) for line in open(path)]
    fig, ax = plt.subplots(figsize=(6.4, 3.6), dpi=100, facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    for i, (win, name) in enumerate(((0, "every summary"), (1, "log summaries"))):
        r = sorted((d for d in rows if d["win"] == win), key=lambda d: d["side"])
        x, y = [d["side"] for d in r], [d["ms_model"] for d in r]
        ax.plot(x, y, color=COLORS[i], lw=2, marker="o", ms=6, mec=SURFACE, mew=1.5, label=name, solid_capstyle="round")
        for n, (a, b) in enumerate(zip(x, y)):
            at = (0, 8) if i == 0 else (0, -15) if n == 0 else (-20, 7)
            ax.annotate("%.0f ms" % b, (a, b), xytext=at, textcoords="offset points", ha="center", fontsize=8, color=MUTED_INK)
        if win == 1:
            ax.plot(x, [y[0] * 4 ** k for k in range(len(x))], color=MUTED_INK, lw=2, ls=(0, (1, 2)), label="linear, from log's first point")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks([64, 128, 256], ["64 world\n4 transformers", "128 world\n16 transformers", "256 world\n64 transformers"])
    ax.minorticks_off()
    ax.set_title("Model time per tick, ms, one GPU", fontsize=11, color=INK, loc="left")
    ax.grid(axis="y", color=GRID, lw=1)
    ax.set_axisbelow(True)
    for k in ("top", "right", "left"):
        ax.spines[k].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=MUTED_INK, labelsize=9, length=0)
    ax.margins(x=0.12, y=0.14)
    ax.legend(frameon=False, fontsize=8, labelcolor=MUTED_INK, loc="upper left")
    fig.tight_layout()
    save(fig, out)


def peak_memory(proc, out):
    gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
    while proc.poll() is None:
        q = subprocess.run(["nvidia-smi", "-i", gpu, "--query-gpu=memory.used", "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout
        if q.strip().isdigit():
            out[0] = max(out[0], int(q))
        try:
            proc.wait(timeout=2)
        except subprocess.TimeoutExpired:
            pass


def one(side, pieces, pyr, foods, poisons, dots, win, args):
    name = f"S{side}_T{pieces}_W{win}"
    out = os.path.join(args.out, name)
    cmd = [sys.executable, "lifetime.py", "--config", "configs/U_default.json", "--seed", "1", "-o", out]
    for kv in (f"name={name}", f"model.img={side}", f"model.uni_pyr={pyr}", f"model.band_n={pieces}", f"model.band_win={win}", f"env.size={side}", f"env.foods={foods}",
               f"env.hazards={poisons}", f"env.distractors={dots}", f"online.steps={args.ticks}", "online.log_every=100", "online.save_every=100000000"):
        cmd += ["--set", kv]
    env = dict(os.environ)
    env.pop("CUDA_MPS_PIPE_DIRECTORY", None)
    with open(out + ".log", "w") as log:
        p = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env)
        mem = [0]
        t = threading.Thread(target=peak_memory, args=(p, mem))
        t.start()
        try:
            p.wait(timeout=args.limit)
        except subprocess.TimeoutExpired:
            p.terminate()
        p.wait()
        t.join()
    lo, hi = tokens(side, pieces, pyr, win)
    row = dict(side=side, pieces=pieces, win=win, tokens_min=lo, tokens_max=hi, peak_mb=mem[0], code=p.returncode)
    path = os.path.join(out, "metrics.jsonl")
    rows = [json.loads(line) for line in open(path)] if os.path.exists(path) else []
    warm = [r for r in rows if r["step"] > args.warm]
    row["warmed"] = bool(warm)                                         # False: the limit came first, and these are the last rows before it
    rows = warm or rows[-3:]
    if rows:                                                           # the t_ columns are seconds per 1,000 ticks, so milliseconds per tick
        mean = lambda *k: sum(sum(r[q] for q in k) for r in rows) / len(rows)                          # noqa: E731
        row.update(ticks=100 * len(rows), ms_act=mean("t_act"), ms_update=mean("t_update"), ms_world=mean("t_world"), ms_memory=mean("t_memory", "t_batch"), sps=mean("sps"))
        row["ms_model"] = row["ms_act"] + row["ms_update"]
    return row


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-o", "--out", help="the folder of the runs")
    ap.add_argument("--plot", nargs=2, metavar=("ROWS", "PNG"), help="draw the rows and stop")
    ap.add_argument("--ticks", type=int, default=1500)
    ap.add_argument("--warm", type=int, default=1100, help="ticks not timed: the memory is still filling")
    ap.add_argument("--limit", type=int, default=1200, help="seconds a cell may run")
    ap.add_argument("--sides", type=int, nargs="+", default=[64, 128, 256])
    args = ap.parse_args()
    if args.plot:
        plot(*args.plot)
        sys.exit()
    os.makedirs(args.out, exist_ok=True)
    for win in (0, 1):
        for c in CELLS:
            if c[0] in args.sides:
                print(json.dumps(one(*c, win, args), default=float), flush=True)

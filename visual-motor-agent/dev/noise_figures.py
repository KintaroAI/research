"""The README figures of vm-00076 (task 00087) and vm-00077 (task 00088): the world at six levels, food and poison by level, and the cross-tests as a lead over a blind walk.

Run from dev/ after `make -C ../00076-white-noise frozen`:  ./venv/bin/python noise_figures.py
vm-00077 (task 00088), after `make -C ../00077-input-noise frozen`:  ./venv/bin/python noise_figures.py input
"""
import json, os, sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
sys.path.insert(0, ".")
from readme_images import fit, label_strip
from world import make_config, make_world
D = os.path.expanduser("~/data/research/visual-motor-agent/")
IMG = "../00076-white-noise/images/"
LV = (0, .05, .1, .25, .5, .75, .9, .95, 1)
ARMS = ["K4"] + ["N%d" % round(q * 100) for q in LV[1:]]
path = lambda a, s: D + ("exp_00073/" if a == "K4" else "exp_00077/" if a[0] == "I" else "exp_00076/") + "%s_s%d" % (a, s)


def input_figure(out="../00077-input-noise/images/input.png"):
    """vm-00077 (task 00088): frozen food and poison by test level, noise in the world (N) against noise on the input only (I)."""
    C = {}
    for f in ("exp_00076/chain_a.jsonl", "exp_00076/chain_b.jsonl", "exp_00077/chain3.jsonl"):
        for line in open(D + f):
            r = json.loads(line)
            a, s = r["run"].split("_s")
            C[a, int(s), r["noise"]] = r
    TL = (0.0, 0.1, 0.5, 1.0)
    style = {"K4": ("#555555", "-", "raised clean (K4)"), "N10": ("#2f6fdb", "--", ".1 in the world"), "I10": ("#2f6fdb", "-", ".1 on the input only"),
             "N50": ("#d9822b", "--", ".5 in the world"), "I50": ("#d9822b", "-", ".5 on the input only")}
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
    fig, axs = plt.subplots(1, 2, figsize=(10.5, 4.2))
    for ax, ck, title in ((axs[0], "food", "Food per 1,000 ticks, frozen"), (axs[1], "poison", "Poison per 1,000 ticks, frozen")):
        for a, (col, ls, lab) in style.items():
            ax.plot(TL, [np.mean([C[a, s, t][ck]["seen"] for s in (1, 2, 3)]) for t in TL], ls, color=col, lw=2, marker="o", ms=5, label=lab)
        blind = np.mean([C["K4", s, t][ck]["blind"] for s in (1, 2, 3) for t in TL])
        ax.axhline(blind, color="#999", lw=1.2, ls=":")
        ax.text(1.0, blind, "blind walk", ha="right", va="bottom", color="#777", fontsize=10)
        ax.set_title(title, loc="left", fontsize=12, color="#222")
        ax.set_xlabel("tested at q")
        ax.set_xticks(TL, ["0", ".1", ".5", "1"])
        ax.grid(axis="y", color="#e4e4e4", lw=.8)
        ax.set_axisbelow(True)
        ax.set_ylim(bottom=0)
    axs[0].legend(frameon=False, fontsize=9.5, loc="upper right")
    fig.tight_layout()
    tmp = out.replace(".png", "_tmp.png")
    fig.savefig(tmp, dpi=110)
    plt.close(fig)
    fit(Image.open(tmp).convert("RGB"), out)
    os.remove(tmp)


if sys.argv[1:] == ["input"]:
    input_figure()
    sys.exit()

# 1. the same world at each level
env = json.load(open(path("K4", 1) + "/info.json"))
env = env.get("config", env).get("env", env.get("env"))
shots, labs = [], []
for q in (0, .1, .25, .5, .75, 1):
    w = make_world(1, make_config(dict(env, noise=q), auto_reset=True), seed=11)
    rng = np.random.default_rng(11)
    for t in range(30):
        if t % 6 == 0:
            m = np.array([[rng.uniform(-1, 1), rng.uniform(-1, 1), 0.0]], np.float32)
        obs = w.step(m)[0]
    shots.append(Image.fromarray(obs["image"][0]).resize((148, 148), Image.NEAREST)); labs.append("q = %g" % q)
fit(label_strip(shots, labs), IMG + "noise.png")

# data
C = {}
for f in ("chain_a.jsonl", "chain_b.jsonl"):
    for l in open(D + "exp_00076/" + f):
        r = json.loads(l); a, s = r["run"].split("_s"); C[a, int(s), r["noise"]] = r
def lived(a, s, k):
    R = [json.loads(l) for l in open(path(a, s) + "/metrics.jsonl")]
    return np.mean([r[k] for r in R if r["step"] > 40000])
TL = (0.0, 0.1, 0.5, 0.9, 0.95, 1.0)
own = [(a, q) for a, q in zip(ARMS, LV) if q in TL]
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#888", "axes.labelcolor": "#333", "xtick.color": "#555", "ytick.color": "#555"})
COL = {"lived": "#2f6fdb", "own": "#d9822b", "clean": "#2a9d6e"}

# 2. food and poison against the level
fig, axs = plt.subplots(1, 2, figsize=(10.5, 4.2))
for ax, key, ck, title in ((axs[0], "step_hits", "food", "Food per 1,000 ticks"), (axs[1], "poison", "poison", "Poison per 1,000 ticks")):
    y = np.array([[lived(a, s, key) for s in (1, 2, 3)] for a in ARMS])
    ax.plot(LV, y.mean(1), "-o", color=COL["lived"], lw=2, ms=6, label="lived, last 20k ticks, at its own level")
    ax.scatter(np.repeat(LV, 3), y.ravel(), s=10, color=COL["lived"], alpha=.35, lw=0)
    xs = [q for a, q in own]
    ax.plot(xs, [np.mean([C[a, s, q][ck]["seen"] for s in (1, 2, 3)]) for a, q in own], "-s", color=COL["own"], lw=2, ms=6, label="frozen, tested at its own level")
    yc = [np.mean([C[a, s, 0.0][ck]["seen"] for s in (1, 2, 3)]) for a in ARMS]
    ax.plot(LV, yc, "-^", color=COL["clean"], lw=2, ms=6, label="frozen, tested on clean frames")
    blind = np.mean([r[ck]["blind"] for r in C.values()])
    ax.axhline(blind, color="#999", lw=1.2, ls="--"); ax.text(1.0, blind, "blind walk", ha="right", va="bottom", color="#777", fontsize=10)
    ax.set_title(title, loc="left", fontsize=12, color="#222"); ax.set_xlabel("noise level q the model lived in"); ax.grid(axis="y", color="#e4e4e4", lw=.8); ax.set_axisbelow(True)
    ax.set_ylim(bottom=0)
axs[0].legend(frameon=False, fontsize=9.5, loc="upper right")
fig.tight_layout(); fig.savefig(IMG + "_levels.png", dpi=110); plt.close(fig)
fit(Image.open(IMG + "_levels.png").convert("RGB"), IMG + "levels.png"); os.remove(IMG + "_levels.png")

# 3. raised on by tested at, lead over a blind walk
fig, axs = plt.subplots(1, 2, figsize=(10.5, 5.0))
for ax, ck, sign, title, cmap in ((axs[0], "food", 1, "Food above a blind walk", "Greens"), (axs[1], "poison", -1, "Poison below a blind walk", "Oranges")):
    M = np.array([[np.mean([sign * (C[a, s, t][ck]["seen"] - C[a, s, t][ck]["blind"]) for s in (1, 2, 3)]) for t in TL] for a in ARMS])
    top = M.max()
    ax.imshow(np.clip(M, 0, None), cmap=cmap, vmin=0, vmax=top, aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, "%.0f" % M[i, j] if abs(M[i, j]) >= 1 else "0", ha="center", va="center", fontsize=9.5, color="white" if M[i, j] > .55 * top else "#333")
        k = TL.index(LV[i]) if LV[i] in TL else None
        if k is not None:
            ax.add_patch(plt.Rectangle((k - .5, i - .5), 1, 1, fill=False, ec="#222", lw=1.6))
    ax.set_xticks(range(len(TL)), ["%g" % t for t in TL]); ax.set_yticks(range(len(ARMS)), ["%g" % q for q in LV])
    ax.set_xlabel("tested at q"); ax.set_ylabel("raised on q"); ax.set_title(title, loc="left", fontsize=12, color="#222")
    for s in ax.spines.values(): s.set_visible(False)
fig.tight_layout(); fig.savefig(IMG + "_cross.png", dpi=110); plt.close(fig)
fit(Image.open(IMG + "_cross.png").convert("RGB"), IMG + "cross.png"); os.remove(IMG + "_cross.png")

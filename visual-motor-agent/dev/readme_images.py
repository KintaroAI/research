"""Task 00069: small PNGs for the experiment READMEs, from what each experiment left in the data folder.

images/curves.png (metrics by arm), images/world.png (the run's world under random moves), images/sheet.png (an existing picture, or stills of a video), and a
## Figures section in the README between markers. Repeatable.

    MUJOCO_EGL_DEVICE_ID=1 ./venv/bin/python readme_images.py [--only 00057 00060] [--no-readme]
"""
import argparse
import glob
import io
import json
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

DEV = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, DEV)
ROOT = os.path.dirname(DEV)
DATA = os.path.expanduser("~/data/research/visual-motor-agent")
MAX_W, MAX_KB = 960, 200
START, END = "<!-- figures -->", "<!-- /figures -->"

COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]   # the fixed order
SURFACE, INK, MUTED_INK, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
FONT = matplotlib.font_manager.findfont("DejaVu Sans")

VIDEOS = {"00054": "videos/compare.mp4", "00056": "thought_T_s2.mp4", "00057": "videos/grey_fix_s3.mp4", "00059": "videos/car_s3.mp4",
          "00060": "videos/car_D_s1.mp4", "00061": "videos/car_SD_s1.mp4", "00062": "videos/car_FD_s1.mp4",
          "00091": "../videos/vm91_s5/compare.mp4", "00092": "../videos/vm92_s5/compare.mp4",
          "00093": "../videos/vm93_s5/compare.mp4",
          "00094": "../videos/vm94_s5/compare.mp4", "00095": "../videos/vm95_s5/compare.mp4",
          "00096": "../videos/vm96_s5/compare.mp4", "00097": "../videos/vm97_inertia_s5/compare.mp4",
          "00098": "../videos/vm98_plain_s5/compare.mp4", "00099": "../videos/vm99_s5/compare.mp4",
          "00100": "../videos/vm100_s5/compare.mp4", "00101": "../videos/vm101_s5/compare.mp4",
          "00102": "../videos/vm102/panels_60k.mp4", "00103": "../videos/vm103/play_60k.mp4",
          "00104": "../videos/vm104/play_60k.mp4", "00105": "../videos/vm105/BL_gap_s1_60k.mp4",
          "00106": "../videos/vm106/GB_gap_s1_60k.mp4", "00107": "../videos/vm107/play_60k.mp4",
          "00109": "../videos/vm109/play_1M.mp4"}
CONTROLS = {"00064": [("00063", ("NA", "CO"))], "00065": [("00064", ("G16", "C16")), ("00063", ("NA", "CO"))], "00066": [("00064", ("C16",))], "00067": [("00064", ("G16",)), ("00063", ("NA",))],
            "00080": [("00073", ("K4",))], "00081": [("00073", ("K4",))],
            "00082": [("00073", ("K4",))], "00083": [("00073", ("K4",)), ("00081", ("H4",))],
            "00084": [("00082", ("C8",))], "00086": [("00085", ("K4", "V90"))],
            "00087": [("00085", ("K4", "V90"))], "00088": [("00086", ("K4D8",))],
            "00089": [("00086", ("K4D8",)), ("00088", ("K4D8", "V90RD8"))], "00090": [("00072", ("K4",))],
            "00095": [("00094", ("C8r",))], "00096": [("00095", ("IB",))],
            "00097": [("00094", ("C8r",)), ("00095", ("IB",))],
            "00098": [("00094", ("C8r",)), ("00095", ("IB",)), ("00097", ("M0", "MB"))],
            "00099": [("00095", ("IB",)), ("00097", ("MB",)), ("00098", ("SB",))],
            "00100": [("00095", ("IB",)), ("00097", ("MB",)), ("00099", ("SD",))],
            "00101": [("00095", ("IB",)), ("00097", ("MB",)), ("00100", ("RD",))],
            "00102": [("00101", ("TT",))], "00107": [("00104", ("BK", "BS"))],
            "00110": [("00103", ("BD",)), ("00108", ("DR_breakout",))]}   # arms of an earlier experiment that this one compares with and did not rerun
SHEET_TEXT = {"forage_sheet": "the frozen agent foraging", "dream_pair_sheet": "the model's dream beside the real frames",
              "imagination": "imagined against real frames", "video": "stills from this experiment's video, in time order"}


def arm_of(name):
    return re.sub(r"_s\d+$", "", re.sub(r"^\d+_", "", name))


def load_runs(e):
    """Runs of an experiment with metrics, and the control arms it reuses (CONTROLS): {arm: [(rows, info), ...]}."""
    arms = {}
    ms = sorted(glob.glob(os.path.join(DATA, "exp_" + e, "*", "metrics.jsonl")))
    for other, names in CONTROLS.get(e, []):
        ms += [m for n in names for m in sorted(glob.glob(os.path.join(DATA, "exp_" + other, n + "_s*", "metrics.jsonl")))]
    for m in ms:
        rows = []
        for line in open(m):
            try:
                rows.append(json.loads(line))
            except ValueError:
                pass
        if not rows:
            continue
        d = os.path.dirname(m)
        p = os.path.join(d, "info.json")
        info = json.load(open(p)) if os.path.exists(p) else {}
        if info.get("ppo", {}).get("lr", 1) == 0:   # a frozen test (learning off), e.g. a video's life: not a life to plot
            continue
        arms.setdefault(arm_of(os.path.basename(d)), []).append((rows, info, d))
    return arms


def env_of(info):
    c = info.get("config")
    return c.get("env", {}) if isinstance(c, dict) else {}


def is_text(info):
    return env_of(info).get("kind") == "text"


def has(rows, k):
    return any(r.get(k) not in (None, 0) for r in rows)


def x_of(r):
    for k in ("step", "steps", "update"):
        if r.get(k) is not None:
            return float(r[k])
    return None


def smooth(y):
    w = max(1, len(y) // 25)
    if w == 1:
        return y
    k = np.ones(w)
    return np.convolve(np.nan_to_num(y), k, "same") / np.convolve(np.ones(len(y)), k, "same")   # fewer points at the ends, no false plateau


def delta(rows, key):
    """A cumulative counter's change per row (task 00117: the Breakout world's bo_* stats)."""
    v = np.array([r.get(key) or 0.0 for r in rows], float)
    return np.diff(v, prepend=0.0)


def x_of_all(rows):
    return np.array([x_of(r) for r in rows], float)


def per_1k(rows, key):
    x = np.array([x_of(r) for r in rows], float)
    v = np.array([r.get(key) or 0.0 for r in rows], float)
    dt = np.diff(x, prepend=0.0)
    return x, v * 1000.0 / np.maximum(dt, 1.0)


def plain(rows, key):
    rows = [r for r in rows if r.get(key) is not None]
    return np.array([x_of(r) for r in rows], float), np.array([r[key] for r in rows], float)


def mean_curve(runs, fn):
    """Seeds' mean over the ticks they share, smoothed."""
    cs = [fn(r[0]) for r in runs]
    cs = [c for c in cs if len(c[0])]
    if not cs:
        return None
    n = min(len(c[0]) for c in cs)
    return cs[0][0][:n], smooth(np.mean([c[1][:n] for c in cs], 0))


def panels(arms):
    """[(title, {arm: (x, y)}, higher is better)]; which arms go where follows each run's world and what its metrics hold."""
    out = []
    forage, hits, text, ppo, val, court = {}, {}, {}, {}, {}, {}
    for a, runs in arms.items():
        rows, info = runs[0][0], runs[0][1]
        if has(rows, "val") or "val" in rows[-1]:
            val[a] = runs
        elif "train_success" in rows[-1]:
            ppo[a] = runs
        elif is_text(info) or (has(rows, "uni_ce_patch") and not has(rows, "step_hits") and not has(rows, "poison")):
            text[a] = runs
        elif env_of(info).get("kind") == "breakout" or "bo_hits" in rows[-1]:   # a hit is a meal, a miss a poison (bo_*: runs without info.json, vm-00108's)
            court[a] = runs
        elif has(rows, "poison"):
            forage[a] = runs
        elif has(rows, "step_hits") or has(rows, "hits"):
            hits[a] = runs
    hit = lambda rows: "step_hits" if has(rows, "step_hits") else "hits"   # noqa: E731
    if forage:
        out.append(("Food touched per 1k ticks", {a: mean_curve(r, lambda q: per_1k(q, hit(q))) for a, r in forage.items()}, True))
        out.append(("Poison touched per 1k ticks", {a: mean_curve(r, lambda q: per_1k(q, "poison")) for a, r in forage.items()}, False))
    if court:   # the world's cumulative counters (bo_*), per row; stage 1's paddle hits are its meals (task 00117: bricks feed)
        share = lambda q: (x_of_all(q), np.array([h / max(h + m, 1) for h, m in zip(delta(q, "bo_hits"), delta(q, "bo_misses"))], float))  # noqa: E731
        out.append(("Ball hit share (paddle hits over hits and misses)", {a: mean_curve(r, share) for a, r in court.items()}, True))
        bricks = {a: r for a, r in court.items() if "bo_food" in r[0][0][-1]}
        if bricks:
            out.append(("Food per 1k ticks (bricks, and paddle hits where they feed)", {a: mean_curve(r, lambda q: (x_of_all(q), delta(q, "bo_food") * 1000.0 / np.maximum(np.diff(x_of_all(q), prepend=0.0), 1.0)))
                                                    for a, r in bricks.items()}, True))
        else:
            out.append(("Balls hit per 1k ticks", {a: mean_curve(r, lambda q: per_1k(q, "step_hits")) for a, r in court.items()}, True))
    if hits:
        out.append(("Targets hit per 1k ticks", {a: mean_curve(r, lambda q: per_1k(q, hit(q))) for a, r in hits.items()}, True))
    if text:
        out.append(("Text: patch cross-entropy (training)", {a: mean_curve(r, lambda q: plain(q, "uni_ce_patch")) for a, r in text.items()}, False))
    if ppo:
        out.append(("Success per episode", {a: mean_curve(r, lambda q: plain(q, "train_success")) for a, r in ppo.items()}, True))
    if val:
        out.append(("Validation error", {a: mean_curve(r, lambda q: plain(q, "val")) for a, r in val.items()}, False))
    return [(t, {a: c for a, c in s.items() if c is not None}, up) for t, s, up in out]


def fold(series, up):
    """At most 8 arms, the best by the last value; colours follow the arm, in the experiment's order."""
    if len(series) <= 8:
        return series, len(series)
    last = sorted(series, key=lambda a: series[a][1][-1], reverse=up)[:8]
    return {a: series[a] for a in series if a in last}, len(series)


def curves(arms, path):
    ps = [p for p in panels(arms) if p[1]]
    if not ps:
        return None
    names = {}
    for _, s, up in ps:
        for a in fold(s, up)[0]:
            names.setdefault(a, COLORS[len(names) % 8])
    fig, axs = plt.subplots(1, len(ps), figsize=(9.2, 3.3), dpi=100, squeeze=False, facecolor=SURFACE)
    total, shown = 0, 0
    for ax, (title, s, up) in zip(axs[0], ps):
        kept, n = fold(s, up)
        total, shown = max(total, n), max(shown, len(kept))
        ax.set_facecolor(SURFACE)
        for a, (x, y) in kept.items():
            ax.plot(x, y, color=names.get(a, INK), lw=2, label=a, solid_capstyle="round")
        ax.set_title(title, fontsize=11, color=INK, loc="left")
        ax.grid(axis="y", color=GRID, lw=1)
        ax.set_axisbelow(True)
        for k in ("top", "right", "left"):
            ax.spines[k].set_visible(False)
        ax.spines["bottom"].set_color(GRID)
        ax.tick_params(colors=MUTED_INK, labelsize=9, length=0)
        ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(
            lambda v, _: "%gM" % (v / 1e6) if v >= 1e6 else ("%gk" % (v / 1e3) if v >= 1e3 else "%g" % v)))
        if 2 <= len(kept) <= 4:   # direct labels at the lines' ends, spread so that they do not collide
            lo, hi = ax.get_ylim()
            gap = (hi - lo) * 0.075
            ends = sorted(kept, key=lambda a: kept[a][1][-1])
            place = []
            for a in ends:
                y = kept[a][1][-1]
                place.append(max(y, place[-1] + gap) if place else y)
            for a, y in zip(ends, place):
                ax.annotate(a, (kept[a][0][-1], y), xytext=(5, 0), textcoords="offset points", va="center", fontsize=8, color=MUTED_INK,
                            annotation_clip=False)
            ax.margins(x=0.09)
        if len(kept) >= 2:
            ax.legend(frameon=False, fontsize=8, labelcolor=MUTED_INK, ncol=min(4, len(kept)), loc="upper center",
                      bbox_to_anchor=(0.5, -0.1), handlelength=1.4, columnspacing=1.0)
    fig.tight_layout()
    save(fig, path)
    return ps, shown, total


def save(fig, path):
    b = io.BytesIO()
    fig.savefig(b, format="png", facecolor=SURFACE)
    plt.close(fig)
    fit(Image.open(b).convert("RGB"), path)


def fit(im, path):
    """At most MAX_W wide and MAX_KB, with a palette."""
    if im.width > MAX_W:
        im = im.resize((MAX_W, im.height * MAX_W // im.width), Image.LANCZOS)
    for colors in (128, 64, 32, 16):
        for w in (im.width, im.width * 3 // 4, im.width // 2):
            q = (im if w == im.width else im.resize((w, im.height * w // im.width), Image.LANCZOS)).quantize(colors, dither=Image.NONE)
            b = io.BytesIO()
            q.save(b, format="PNG", optimize=True)
            if b.tell() <= MAX_KB * 1024:
                os.makedirs(os.path.dirname(path), exist_ok=True)
                open(path, "wb").write(b.getvalue())
                return True
    raise ValueError("cannot fit " + path)


def label_strip(tiles, labels, gap=8, pad=6):
    f = ImageFont.truetype(FONT, 13)
    w, h = tiles[0].size
    n = len(tiles)
    im = Image.new("RGB", (n * w + (n - 1) * gap + 2 * pad, h + 24 + 2 * pad), SURFACE)
    d = ImageDraw.Draw(im)
    for i, (t, lab) in enumerate(zip(tiles, labels)):
        x = pad + i * (w + gap)
        d.text((x, pad), lab, font=f, fill=MUTED_INK)
        im.paste(t, (x, pad + 22))
    return im


def world_strip(arms, path):
    """The first run of the experiment whose world the current code can still build, under random moves held for a while."""
    from world import make_config, make_world
    tried = None
    for a, runs in arms.items():
        env_d = env_of(runs[0][1])
        if not env_d:
            continue
        seed = 11
        try:
            w = make_world(1, make_config(env_d, auto_reset=True), seed=seed)
            obs = w.observe()
            rng = np.random.default_rng(seed)
            H, W = obs["image"].shape[1:3]
            n = 6 if W == H else 3
            scale = max(1, round((930 - (n - 1) * 8) / n / W))
            shots, labs = [], []
            m = np.zeros((1, 3), np.float32)
            for t in range(n * 12 + 1):
                if t % 12 == 0 and len(shots) < n:
                    shots.append(Image.fromarray(obs["image"][0]).resize((W * scale, H * scale), Image.NEAREST))
                    labs.append("tick %d" % t)
                if t % 6 == 0:
                    m = np.array([[rng.uniform(-1, 1), rng.uniform(-1, 1), 0.0]], np.float32)
                obs = w.step(m)[0]
                if hasattr(w, "think"):
                    obs = dict(obs, image=w.think(None))
            if hasattr(w, "close"):
                w.close()
            fit(label_strip(shots, labs), path)
            return a, env_d.get("kind", "circle"), True
        except Exception as ex:   # an old world's config, or code that moved on
            tried = "%s: %s" % (a, str(ex)[:90])
    return tried, None, False


def existing_sheet(e):
    d = os.path.join(DATA, "exp_" + e)
    for pat, kind in (("*/forage_sheet.png", "forage_sheet"), ("*/dream_pair_sheet.png", "dream_pair_sheet")):
        g = sorted(glob.glob(os.path.join(d, pat)))
        if g:
            return g[0], kind
    g = sorted(glob.glob(os.path.join(d, "*", "analysis", "*", "imagination.png")))
    return (g[-1], "imagination") if g else (None, None)


def video_sheet(path, out, n=4):
    import imageio_ffmpeg
    gen = imageio_ffmpeg.read_frames(path)
    meta = next(gen)
    W, H = meta["size"]
    frames = [np.frombuffer(b, np.uint8).reshape(H, W, 3).copy() for b in gen]
    ids = np.linspace(len(frames) // 8, len(frames) - 1, n).astype(int)
    tw = (MAX_W - 6) // 2
    tiles = [Image.fromarray(frames[i]).resize((tw, H * tw // W), Image.LANCZOS) for i in ids]
    im = Image.new("RGB", (MAX_W, 2 * tiles[0].height + 6), SURFACE)
    for k, t in enumerate(tiles):
        im.paste(t, ((k % 2) * (tw + 6), (k // 2) * (tiles[0].height + 6)))
    fit(im, out)


def world_caption(kind):
    if kind == "text":
        return "The text world as this experiment's model sees it: each frame is a page, one letter per block of grey."
    if kind == "car":
        return "What the car's camera sees, under random moves."
    return "The world of this experiment as its model sees it, every 12 ticks. Moves are random: this shows the world, not the trained agent."


def readme_section(figs):
    lines = [START, "## Figures", ""]
    for name, alt, cap in figs:
        lines += ["![%s](images/%s)" % (alt, name), "", "*%s*" % cap, ""]
    return "\n".join(lines) + END + "\n"


def put_section(path, text):
    s = open(path).read()
    if START in s:
        i, j = s.index(START), s.index(END) + len(END)
        s = s[:i] + text.rstrip("\n") + s[j:]
    else:
        for head in ("## Analysis", "## Conclusions", "## Next"):
            m = re.search(r"^%s" % head, s, re.M)
            if m:
                s = s[:m.start()] + text + "\n" + s[m.start():]
                break
        else:
            s = s.rstrip("\n") + "\n\n" + text
    open(path, "w").write(s)


def baselines_figs(img):
    """vm-00108 (task 00120): the learned baselines beside ours, one figure per world. Their runs name arm and world
    (DR_breakout_s1) and PL counts samples of 16 bodies, so the generic curves don't fit them."""
    def runs(e, name):
        out = []
        for sd in range(1, 7):   # seeds 1-6 (ours ran 12: the baselines' seeds are 1-6)
            m = os.path.join(DATA, "exp_" + e, "%s_s%d" % (name, sd), "metrics.jsonl")
            out.append(([json.loads(x) for x in open(m) if x.strip()],))
        return out
    def share(q):   # over the last 5 rows (5k ticks; PL's rows are 16k samples each: one row)
        k = 1 if q[0].get("bodies", 1) > 1 else 5
        h, m = (np.array([r["bo_" + c] for r in q], float) for c in ("hits", "misses"))
        h, m = h - np.concatenate([np.zeros(k), h[:-k]]), m - np.concatenate([np.zeros(k), m[:-k]])
        return x_of_all(q)[k - 1:], (h / np.maximum(h + m, 1))[k - 1:]
    food = lambda q: (x_of_all(q), np.array([r["step_hits"] for r in q], float))  # noqa: E731
    pois = lambda q: (x_of_all(q), np.array([r["poison"] for r in q], float))  # noqa: E731
    first = lambda fn, n: (lambda q: fn([r for r in q if r["step"] <= n]))  # noqa: E731
    col = {"ours": COLORS[0], "DR": COLORS[1], "PP": COLORS[2], "PL": COLORS[3]}
    worlds = (("breakout", "BD", "00103", [("Ball hit share", share, True)], 0.235, "random paddle"),
              ("food", "TT", "00101", [("Food per 1k ticks", food, True), ("Poison per 1k ticks", pois, False)], None, None))
    figs = []
    for w, ours, oe, ms, ref, ref_name in worlds:
        series = {"ours": runs(oe, ours), "DR": runs("00108", "DR_" + w), "PP": runs("00108", "PP_" + w), "PL": runs("00108", "PL_" + w)}
        fig, axs = plt.subplots(1, len(ms) + 1, figsize=(9.2, 3.3), dpi=100, squeeze=False, facecolor=SURFACE)
        for i, (title, fn, up) in enumerate(ms + [ms[0]]):
            ax = axs[0][i]
            ax.set_facecolor(SURFACE)
            whole = i == len(ms)
            if whole:   # PL's whole run, by samples, with ours and DR at 50-60k as levels
                c = mean_curve(series["PL"], fn)
                ax.plot(c[0], c[1], color=col["PL"], lw=2, label="PL", solid_capstyle="round")
                for a in ("ours", "DR"):
                    if fn is share:   # the 50-60k level, from the counters
                        lvl = float(np.mean([(q[59]["bo_hits"] - q[49]["bo_hits"]) / max(1, q[59]["bo_hits"] + q[59]["bo_misses"]
                                             - q[49]["bo_hits"] - q[49]["bo_misses"]) for (q,) in series[a]]))
                    else:
                        lvl = float(np.mean([np.mean(fn(q)[1][50:60]) for (q,) in series[a]]))
                    ax.axhline(lvl, color=col[a], lw=1.5, ls=(0, (4, 3)))
                    ax.annotate("%s at 50-60k" % (ours if a == "ours" else a), (c[0][-1], lvl), xytext=(-2, 4), textcoords="offset points",
                                ha="right", fontsize=8, color=MUTED_INK)
                ax.set_title("PL's whole run: " + title[0].lower() + title[1:], fontsize=11, color=INK, loc="left")
                ax.set_xlabel("samples (16 bodies)", fontsize=9, color=MUTED_INK)
            else:
                for a in ("ours", "DR", "PP", "PL"):
                    c = mean_curve(series[a], first(fn, 64000) if a == "PL" else fn)
                    if c is not None:
                        ax.plot(c[0], c[1], color=col[a], lw=2, label=ours if a == "ours" else (a + " (samples)" if a == "PL" else a),
                                solid_capstyle="round", marker="o" if a == "PL" else None, ms=4)
                ax.set_title(title, fontsize=11, color=INK, loc="left")
                ax.set_xlabel("ticks lived", fontsize=9, color=MUTED_INK)
                ax.legend(frameon=False, fontsize=8, labelcolor=MUTED_INK, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.2),
                          handlelength=1.4, columnspacing=1.0)
            if ref is not None:
                ax.axhline(ref, color=MUTED_INK, lw=1, ls=":")
                if not whole:
                    ax.annotate(ref_name, (60000, ref), xytext=(0, -11), textcoords="offset points", ha="right", fontsize=8, color=MUTED_INK)
            ax.grid(axis="y", color=GRID, lw=1)
            ax.set_axisbelow(True)
            for k in ("top", "right", "left"):
                ax.spines[k].set_visible(False)
            ax.spines["bottom"].set_color(GRID)
            ax.tick_params(colors=MUTED_INK, labelsize=9, length=0)
            ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(
                lambda v, _: "%gM" % (v / 1e6) if v >= 1e6 else ("%gk" % (v / 1e3) if v >= 1e3 else "%g" % v)))
        fig.tight_layout()
        name = "curves_%s.png" % w
        save(fig, os.path.join(img, name))
        if w == "breakout":
            cap = ("Breakout stage 1, seeds 1-6, the mean by arm, smoothed: BD the default agent (vm-00103), DR DreamerV3, PP PPO with"
                   " one body, PL PPO with 16 bodies (its first 64k samples on the ticks' axis; right, its whole run). The dotted line"
                   " is a random paddle (.235).")
        else:
            cap = ("The food world, seeds 1-6, the mean by arm, smoothed: TT the default agent (vm-00101), DR DreamerV3, PP PPO with"
                   " one body, PL PPO with 16 bodies (its first 64k samples on the ticks' axis; right, its whole run, food per 1k"
                   " ticks of each body).")
        figs.append((name, "%s: ours, DreamerV3 and PPO" % w.capitalize(), cap))
    return figs


def one(d, write_readme):
    e = os.path.basename(d)[:5]
    arms = load_runs(e)
    img = os.path.join(d, "images")
    figs, notes = [], []
    if e == "00108":
        figs, arms = baselines_figs(img), {}
    if arms:
        r, kind, ok = world_strip(arms, os.path.join(img, "world.png"))
        if ok:
            figs.append(("world.png", "Frames of the world", world_caption(kind)))
        elif r:
            notes.append("no world: " + r)
        else:
            notes.append("no world config")
        c = curves(arms, os.path.join(img, "curves.png"))
        if c:
            ps, shown, total = c
            cap = "From each run's `metrics.jsonl`, the mean of its seeds by arm, smoothed."
            if e == "00095":
                cap += " C8r is vm-00094's control in the world without inertia; IS lives without inertia until tick 30k, then with it."
            if e == "00096":
                cap += " All in the world with inertia from birth; IB is vm-00095's (the ring), IR and IH act without it."
            if e == "00097":
                cap += (" C8r, M0 and M0N live without inertia, IB, MB and MBN with it; C8r and IB have the hunger value, the M arms"
                        " the meal label, the N arms weighted by hunger when choosing.")
            if e == "00098":
                cap += (" C8r, M0 and S0 live without inertia, IB, MB and SB with it; C8r and IB have the hunger value, M0 and MB the"
                        " meal label re-labeled at every replay (vm-00097), S0 and SB the meal label stored once, in its own body channel.")
            if e == "00099":
                cap += (" All in the world with inertia from birth: IB the hunger value (vm-00095), MB the meal label re-labeled at every"
                        " replay (vm-00097), SB the meal label stored once (vm-00098), SD stored once with the full dose of poison.")
            if e == "00100":
                cap += (" All in the world with inertia from birth: IB the hunger value (vm-00095), MB the meal label re-labeled at every"
                        " replay in the hunger channel (vm-00097), SD stored once in its own channel with the full dose (vm-00099), RD"
                        " re-labeled at every replay in its own channel with the full dose.")
            if e == "00101":
                cap += (" All in the world with inertia from birth: IB the hunger value (vm-00095), MB the meal label re-labeled at every"
                        " replay in the hunger channel (vm-00097), RD re-labeled in its own channel with the full dose (vm-00100), TT RD"
                        " with the instinct's candidates after the first drawn at temperature 2.")
            if e == "00102":
                cap += (" All in the world with inertia, on TT's settings (vm-00101: RD's label with the candidates at temperature 2): PB and PN"
                        " see a 128 x 128 composite of four 64 x 64 panels, panel 1 the real view and panels 2-4 noise (PB) or the"
                        " model's own forecasts (PN); TT is vm-00101's 64 x 64 agent without panels (its 12 seeds).")
            if e == "00104":
                cap += (" Stage 2 (a ball at 2.25 px a tick, no food for a paddle hit, 24 bricks of 3 looks feeding .1, .25 and .5): BK"
                        " the default agent unchanged, BS with a meal worth its size; 12 seeds each. A random paddle's hit share is"
                        " .229 and its food 3.1 per 1k; the aimer's food 7.0.")
            if e == "00107":
                cap += (" Stage 2 with food for a paddle hit (.25 of hunger; bricks .1, .25 and .5 by look): PK the default agent"
                        " unchanged, PS with a meal worth its size; 12 seeds each. BK and BS are vm-00104's same arms without paddle"
                        " food. A random paddle's hit share is .229.")
            if e == "00109":
                cap += (" One life of 1M ticks in vm-00107's court (stage 2, paddle food .25): PS seed 1 (a meal worth its size, .5),"
                        " smoothed over about 40k ticks. vm-00107's PS returned .47 at 50-60k; a random paddle .229.")
            if e == "00110":
                cap += (" Breakout stage 1: RB the default agent with online.batch 56 (Dreamer's replayed steps per tick), BD the"
                        " default agent (vm-00103, batch 8), 12 seeds each; DR_breakout DreamerV3 (vm-00108, seeds 1-6). A random"
                        " paddle's hit share is .235.")
            if e == "00105":
                cap += (" BL is the default agent with a fixed 12 x 12 px blind spot in everything the model sees (targets whole), 12 seeds;"
                        " its control, vm-00101's TT, eats 71.6 food and 8.3 poison per 1k at 40-60k.")
            if e == "00106":
                cap += (" The default agent with the vm-00105 blind spot in everything it sees and in its targets: G0 learns against"
                        " the black gap, GB against the gap filled by the motion teacher; 12 seeds each. TT (no gap) eats 71.6 food"
                        " and 8.3 poison per 1k at 40-60k.")
            if e == "00103":
                cap += (" BD is the default agent (vm-00101's TT) in the Breakout court, 12 seeds; the random paddle's hit share is .235,"
                        " the tracker's 1.0.")
            if total > shown:
                cap += " The %d best arms of %d by their last value." % (shown, total)
            figs.append(("curves.png", "; ".join(p[0] for p in ps), cap))
        else:
            notes.append("no curves")
    v = VIDEOS.get(e)
    src = os.path.join(DATA, "exp_" + e, v) if v else None
    if src and os.path.exists(src):
        video_sheet(src, os.path.join(img, "sheet.png"))
        cap = SHEET_TEXT["video"].capitalize() + "."
        if e == "00059":
            cap += " The tilted horizon and the dark floor are the bug described in the correction above: the car had tipped over."
        if e == "00091":
            cap += (" Seed 5 at 60k, frozen, KW and VW from the same start. Each arm's 64 x 64 view (the blue square: the quarter transformer whose move it was),"
                    " and beside it the whole world, centered on the cursor the same way, with the window outlined: plain circles are food, checkered ones poison.")
        if e == "00098":
            cap += (" Seed 5 at 60k, frozen, from the same start, in the world without inertia: C8r (the hunger value), M0 (the meal"
                    " label, re-labeled at every replay) and S0 (the meal label stored once, in its own body channel). Each arm's 64 x 64"
                    " view (the blue square: the quarter transformer whose move it was), and beside it the whole world with the window"
                    " outlined. Plain circles are food, checkered ones poison. The same comparison with inertia (IB, MB, SB) is in"
                    " videos/vm98_inertia_s5.")
        if e == "00099":
            cap += (" Seed 5 at 60k, frozen, from the same start, in the world with inertia: IB (the hunger value), MB (the meal label,"
                    " re-labeled at every replay), SB (stored once, capped poison cost) and SD (stored once, full dose). Each arm's 64 x 64"
                    " view (the blue square: the quarter transformer whose move it was), and beside it the whole world with the window"
                    " outlined. Plain circles are food, checkered ones poison. Seed 3 (where SB starved) is in videos/vm99_s3.")
        if e == "00100":
            cap += (" Seed 5 at 60k, frozen, from the same start, in the world with inertia: IB (the hunger value), MB (the meal label,"
                    " re-labeled, in the hunger channel), SD (stored once, own channel, full dose) and RD (re-labeled, own channel, full"
                    " dose). Each arm's 64 x 64 view (the blue square: the quarter transformer whose move it was), and beside it the whole"
                    " world with the window outlined. Plain circles are food, checkered ones poison. Seed 3 is in videos/vm100_s3.")
        if e == "00101":
            cap += (" Seed 5 at 60k, frozen, from the same start, in the world with inertia: IB (the hunger value), MB (the meal label,"
                    " re-labeled, in the hunger channel), RD (re-labeled, own channel, full dose) and TT (RD with the candidates at"
                    " temperature 2). Each arm's 64 x 64 view (the blue square: the quarter transformer whose move it was), and beside it"
                    " the whole world with the window outlined. Plain circles are food, checkered ones poison. Seed 3 is in"
                    " videos/vm101_s3.")
        if e == "00102":
            cap += (" PN seed 1 at 60k, frozen, 400 ticks in a new world. Left: the 128 x 128 frame as the model sees it (panel 1 the real"
                    " view; panels 2-4 the forecasts of the frame now, 1 tick ahead and 2 ahead, made on the tick before, with noise .03)."
                    " Right: the real frames those forecasts aim at. The forecast further out hedges toward the present (R6).")
        if e == "00104":
            cap += (" BK (left) and BS (right) seed 1 at 60k, frozen, 900 ticks from the same court (seed 201): the frame as the"
                    " model sees it, the bricks' three looks (solid, checker, hollow); which look feeds how much rotates with the seed.")
        if e == "00107":
            cap += (" PK (left) and PS (right) seed 1 at 60k, frozen, 900 ticks from the same court (seed 201): the frame as the"
                    " model sees it, the bricks' three looks (solid, checker, hollow); which look feeds how much rotates with the seed.")
        if e == "00109":
            cap += (" PS seed 1 at 60k (vm-00107, left) and at 1M (right), frozen, 900 ticks from the same court (seed 201): the"
                    " frame as the model sees it, the bricks' three looks (solid, checker, hollow).")
        if e == "00105":
            cap += (" BL seed 1 at 60k, frozen, 600 ticks in a new world (seed 101). Left: what the model sees, the gap black (red box);"
                    " middle: its predicted frame (each quarter transformer its own quarter, decoded pair); right: the true frame."
                    " TT with the gap put in, never trained with it, is in videos/vm105/TT_gap_s1_60k.mp4.")
        if e == "00106":
            cap += (" GB seed 1 at 60k, frozen, 600 ticks in a new world (seed 101). Left: what the model sees, the gap black (red box);"
                    " middle: its predicted frame; right: the true frame. GB never learned from the true gap, only from the teacher's"
                    " fill. G0 (trained against the black gap) predicts black there: videos/vm106/G0_gap_s1_60k.mp4.")
        if e == "00103":
            cap += (" BD seeds 1-3 at 60k, frozen, 600 ticks each in new courts: the 64 x 64 frame as the model sees it (walls, the"
                    " paddle at the bottom, the ball). Each life hit 6 balls and missed 2 here.")
        if e == "00097":
            cap += (" Seed 5 at 60k, frozen, from the same start, in the world with inertia: IB (the hunger value), MB (the meal"
                    " label) and MBN (the meal label weighted by hunger when choosing). Each arm's 64 x 64 view (the blue square: the"
                    " quarter transformer whose move it was), and beside it the whole world with the window outlined. Plain circles"
                    " are food, checkered ones poison. The same arms without inertia are in videos/vm97_plain_s5.")
        if e == "00096":
            cap += (" Seed 5 at 60k, frozen, from the same start, all in the world with inertia: IB (vm-00095, the ring, the one-move"
                    " instinct), IR (no ring, the one-move instinct) and IH (no ring, each candidate held for 4 imagined ticks). Each"
                    " arm's 64 x 64 view (the blue square: the quarter transformer whose move it was), and beside it the whole world"
                    " with the window outlined. Plain circles are food, checkered ones poison.")
        if e == "00095":
            cap += (" Seed 5 at 60k, frozen, from the same start: C8r in its own world (no inertia), IB and IS in the world with"
                    " inertia (each arm's world as its life ended). Each arm's 64 x 64 view (the blue square: the quarter transformer"
                    " whose move it was), and beside it the whole world with the window outlined. Plain circles are food, checkered"
                    " ones poison.")
        if e == "00094":
            cap += (" Seed 5 at 60k, frozen, C8r and F from the same start, laid out as in vm-00091: each arm's 64 x 64 view (the blue"
                    " square: the quarter transformer whose move it was), and beside it the whole world with the window outlined. Plain"
                    " circles are food, checkered ones poison. F is frozen at its value at 60k: it does not learn here.")
        if e == "00093":
            cap += (" Seed 5 at 60k, frozen, C8, SA and SB from the same start, laid out as in vm-00091: each arm's 64 x 64 view (the blue"
                    " square: the quarter transformer whose move it was), and beside it the whole world with the window outlined. Plain"
                    " circles are food, checkered ones poison.")
        if e == "00092":
            cap += (" Seed 5 at 60k, frozen, C8 and C32 from the same start, laid out as in vm-00091: each arm's 64 x 64 view (the blue square: the quarter"
                    " transformer whose move it was), and beside it the whole world with the window outlined. Plain circles are food, checkered ones poison.")
        figs.append(("sheet.png", "Stills from a video", cap))
    else:
        p, kind = existing_sheet(e)
        if p:
            fit(Image.open(p).convert("RGB"), os.path.join(img, "sheet.png"))
            figs.append(("sheet.png", SHEET_TEXT[kind].capitalize(), SHEET_TEXT[kind].capitalize() + " (`%s`)." % os.path.relpath(p, os.path.join(DATA, "exp_" + e))))
    readme = os.path.join(d, "README.md")
    if write_readme and figs and os.path.exists(readme):
        put_section(readme, readme_section(figs))
    print(e, " ".join(f[0][:-4] for f in figs) or "NOTHING", " | ".join(notes)[:130], flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--no-readme", action="store_true")
    a = ap.parse_args()
    for d in sorted(glob.glob(os.path.join(ROOT, "0*-*"))):
        if a.only and os.path.basename(d)[:5] not in a.only:
            continue
        one(d, not a.no_readme)


if __name__ == "__main__":
    main()

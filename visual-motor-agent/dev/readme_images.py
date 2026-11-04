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
          "00060": "videos/car_D_s1.mp4", "00061": "videos/car_SD_s1.mp4", "00062": "videos/car_FD_s1.mp4"}
CONTROLS = {"00064": [("00063", ("NA", "CO"))], "00065": [("00064", ("G16", "C16")), ("00063", ("NA", "CO"))], "00066": [("00064", ("C16",))], "00067": [("00064", ("G16",)), ("00063", ("NA",))],
            "00080": [("00073", ("K4",))], "00081": [("00073", ("K4",))],
            "00082": [("00073", ("K4",))], "00083": [("00073", ("K4",)), ("00081", ("H4",))]}   # arms of an earlier experiment that this one compares with and did not rerun
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
    forage, hits, text, ppo, val = {}, {}, {}, {}, {}
    for a, runs in arms.items():
        rows, info = runs[0][0], runs[0][1]
        if has(rows, "val") or "val" in rows[-1]:
            val[a] = runs
        elif "train_success" in rows[-1]:
            ppo[a] = runs
        elif is_text(info) or (has(rows, "uni_ce_patch") and not has(rows, "step_hits") and not has(rows, "poison")):
            text[a] = runs
        elif has(rows, "poison"):
            forage[a] = runs
        elif has(rows, "step_hits") or has(rows, "hits"):
            hits[a] = runs
    hit = lambda rows: "step_hits" if has(rows, "step_hits") else "hits"   # noqa: E731
    if forage:
        out.append(("Food touched per 1k ticks", {a: mean_curve(r, lambda q: per_1k(q, hit(q))) for a, r in forage.items()}, True))
        out.append(("Poison touched per 1k ticks", {a: mean_curve(r, lambda q: per_1k(q, "poison")) for a, r in forage.items()}, False))
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


def one(d, write_readme):
    e = os.path.basename(d)[:5]
    arms = load_runs(e)
    img = os.path.join(d, "images")
    figs, notes = [], []
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

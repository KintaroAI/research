"""Task 00134: a video of a watching model's dreams beside the world (the owner: "can you generate videos with examples?").

A few stretches of the exam, one a row. First the last real ticks, which every model watches; then the dream, tick by
tick: the world's own continuation in the first column and each model's dream beside it. On the dreams the true places
of the things are outlined (a disc's rim, the ball's box), so a lag or a drift is seen as the thing leaving its outline.

    ./venv/bin/python wm_video.py RUN:LABEL [RUN:LABEL ...] -o OUT.mp4 [--set work|home] [--starts 4] [--real 8]
                                  [--warm 64] [--dream 32] [--outline N] [--scale 4] [--fps 6] [--step 60000]
                                  [--chunk 64] [--title TEXT]

--set work: the exam's working set (seed 1001; in the disc world its discs and dots are new to every life).
--set home: the runs' own world after their lives' end (the runs must share their seed: it is the same world).
The stretches are fixed by the number asked for, not chosen by looking: copies spread over the set, starts in turn.
--real: the watched ticks shown before the dream. --warm: the real ticks the model is given before it dreams (64, as at
the exam's regular starts; the model's window is its last 8). A long dream (the owner, task 00135: "it starts for 20
ticks and then dream 1000 ticks") is --real 20 --warm 20 --dream 1000: the exam is then made as much longer as the
dream, its first ticks the same. --outline N: the true places outlined on the first N dreamed ticks only (a long dream
and the world part ways, and the outlines then say nothing).
"""
import argparse
import itertools
import json
import os

import numpy as np
from PIL import Image, ImageDraw, ImageFont

import wm_measures as W

BG, INK, MUTED = (16, 16, 18), (235, 235, 235), (150, 150, 155)
REAL, DREAM = (120, 200, 255), (255, 170, 70)                 # the caption's colour while watching, while dreaming
RIM = dict(food=(110, 230, 140), poison=(255, 110, 95), ball=(255, 215, 80))
PAD, GAP, HEAD, CAPTION = 8, 6, 22, 24


def font(size=13):
    from readme_images import FONT
    return ImageFont.truetype(FONT, size)


def stretches(exam, n, home=False):
    """n stretches (copy, start tick): copies spread over the set, the regular starts taken in turn."""
    k = len(exam.starts)
    if home or exam.E == 1:
        return np.zeros(n, int), np.array([exam.starts[(4 * i) % k] for i in range(n)])
    return np.round(np.linspace(0, exam.E - 1, n)).astype(int), np.array([exam.starts[(3 * i) % k] for i in range(n)])


def outlines(d, exam, e, t, x0, y0, scale):
    """The true places of the things at tick t of copy e, drawn on a panel whose corner is (x0, y0)."""
    S = exam.S
    if exam.kind == "watch":
        pos, rad = exam.state["pos"][e, t], exam.rad[e]
        for j, k in enumerate(exam.kinds):
            name = exam.thing_kinds[int(k)]
            for ox in (-S, 0, S):                                 # a torus: a disc over the edge shows on both sides
                for oy in (-S, 0, S):
                    cx, cy, r = (pos[j, 0] + ox) * scale, (pos[j, 1] + oy) * scale, rad[j] * scale
                    if cx + r < 0 or cy + r < 0 or cx - r > S * scale or cy - r > S * scale:
                        continue
                    d.ellipse((x0 + cx - r, y0 + cy - r, x0 + cx + r, y0 + cy + r), outline=RIM[name])
    elif exam.state["live"][e, t]:
        bx, by = np.rint(exam.state["ball"][e, t]).astype(int)
        b = exam.cfg.ball
        d.rectangle((x0 + (bx - 1) * scale, y0 + (by - 1) * scale, x0 + (bx + b + 1) * scale - 1, y0 + (by + b + 1) * scale - 1),
                    outline=RIM["ball"])


def frames(exam, e, s, dreamed, real=8, scale=4, title="", hold=4, rims=True):
    """The video's frames [h, w, 3]: `real` watched ticks, then the dream. dreamed: [(label, frames [M, H, S, S])].
    rims: the true places outlined on the dreams: True on every dreamed tick, a number on the first that many, False never."""
    M, H = len(e), dreamed[0][1].shape[1]
    side, ncol = exam.S * scale, 1 + len(dreamed)
    w = 2 * PAD + ncol * side + (ncol - 1) * GAP
    h = 2 * PAD + (HEAD if title else 0) + HEAD + CAPTION + M * side + (M - 1) * GAP
    w8, h8 = -(-w // 8) * 8, -(-h // 8) * 8                       # the encoder wants multiples of 8
    f, small = font(13), font(12)
    top = PAD + (HEAD if title else 0)
    labels = ["the world"] + [lab for lab, _ in dreamed]

    def tile(a):
        return Image.fromarray(np.clip(np.asarray(a), 0, 255).astype(np.uint8), "L").resize((side, side), Image.NEAREST).convert("RGB")

    def one(t, phase):
        """t: the tick relative to the last real one (0: the last real tick; h > 0: dreamed tick h)."""
        im = Image.new("RGB", (w8, h8), BG)
        d = ImageDraw.Draw(im)
        if title:
            d.text((PAD, PAD), title, font=f, fill=INK)
        for c, lab in enumerate(labels):
            d.text((PAD + c * (side + GAP), top), lab if (c == 0 or t > 0) else lab + " (watching)", font=small, fill=INK if c == 0 else MUTED if t <= 0 else INK)
        cap = ("watching: %d tick%s before the dream" % (-t, "" if t == -1 else "s") if t < 0 else "watching: the last real tick") if t <= 0 else "dreaming: tick +%d" % t
        d.text((PAD, top + HEAD), cap + phase, font=f, fill=REAL if t <= 0 else DREAM)
        drawn = t > 0 and (rims is True or (rims is not False and t <= rims))
        if drawn:
            d.text((w - PAD - 250, top + HEAD), "outline: where the thing really is", font=small, fill=MUTED)
        elif t > 0 and rims:
            d.text((w - PAD - 250, top + HEAD), "(outlined up to tick +%d)" % rims, font=small, fill=MUTED)
        for m in range(M):
            y = top + HEAD + CAPTION + m * (side + GAP)
            truth = exam.frames[e[m], s[m] + t]
            for c in range(ncol):
                x = PAD + c * (side + GAP)
                im.paste(tile(truth if (c == 0 or t <= 0) else dreamed[c - 1][1][m, t - 1]), (x, y))
                if c > 0 and drawn:
                    layer = Image.new("RGB", (side, side), (0, 0, 0))
                    outlines(ImageDraw.Draw(layer), exam, e[m], s[m] + t, 0, 0, scale)
                    mask = Image.fromarray((np.asarray(layer).sum(2) > 0).astype(np.uint8) * 255, "L")
                    im.paste(layer, (x, y), mask)
        return np.asarray(im)

    for t in range(-real + 1, 1):
        yield one(t, "")
    for _ in range(hold):
        yield one(0, "   (now it dreams)")
    for t in range(1, H + 1):
        yield one(t, "")
    last = one(H, "")
    for _ in range(hold):
        yield last


def write_mp4(path, images, fps):
    import imageio_ffmpeg
    images = iter(images)
    first = next(images)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    gen = imageio_ffmpeg.write_frames(path, (first.shape[1], first.shape[0]), fps=fps, codec="libx264", quality=None, bitrate="3M", macro_block_size=8)
    gen.send(None)
    n = 0
    for im in itertools.chain([first], images):       # one frame at a time: a long dream is a thousand frames
        gen.send(np.ascontiguousarray(im).tobytes())
        n += 1
    gen.close()
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", help="RUN:LABEL")
    ap.add_argument("-o", "--out", required=True)
    ap.add_argument("--set", dest="which", default="work", choices=["work", "home"])
    ap.add_argument("--starts", type=int, default=4)
    ap.add_argument("--real", type=int, default=8, help="watched ticks shown before the dream")
    ap.add_argument("--warm", type=int, default=W.WARM, help="real ticks the model is given before it dreams")
    ap.add_argument("--dream", type=int, default=32)
    ap.add_argument("--outline", type=int, default=None, help="outline the true places on the first N dreamed ticks only (0: never)")
    ap.add_argument("--scale", type=int, default=4)
    ap.add_argument("--fps", type=int, default=6)
    ap.add_argument("--step", type=int, default=60000)
    ap.add_argument("--chunk", type=int, default=64)
    ap.add_argument("--title", default="")
    a = ap.parse_args()
    W.CHUNK = a.chunk
    runs = [(r.rsplit(":", 1) + [None])[:2] for r in a.runs]
    infos = [json.load(open(os.path.join(r, "info.json"))) for r, _ in runs]
    env = infos[0]["config"]["env"]
    if any(i["config"]["env"] != env for i in infos):
        raise SystemExit("the runs watch different worlds")
    ticks = W.TICKS + max(0, a.dream - W.HORIZON)             # a dream longer than the exam's: the same exam, continued
    if a.which == "home":
        if len({i["seed"] for i in infos}) != 1:
            raise SystemExit("--set home shows one life's own world: the runs must share their seed")
        exam = W.Exam(env, seed=infos[0]["seed"], copies=1, skip=max(W.home_skip(i) for i in infos), ticks=ticks)
    else:
        exam = W.Exam(env, seed=W.WORK_SEED, ticks=ticks)
    e, s = stretches(exam, a.starts, a.which == "home")
    if s.min() - max(a.real, a.warm) + 1 < 0 or s.max() + a.dream >= exam.T:
        raise SystemExit("a stretch does not fit the exam: %d ticks before a start, %d after it" % (max(a.real, a.warm), a.dream))
    dreamed = []
    for (run, label), info in zip(runs, infos):
        m = W.Pieces(run, a.step, "pair")
        name = label or info["config"].get("name", os.path.basename(run))
        dreamed.append(("dream: " + name, W.dreams(m, exam, e, s, a.dream, warm=a.warm)[0]))
        print("%s: %.1f passes a dreamed tick, %d frames of %d not settled" % (name, float(np.mean(m.passes)), m.unsettled, len(e) * a.dream))
        m.close()
    rims = True if a.outline is None else a.outline if a.outline > 0 else False
    n = write_mp4(a.out, frames(exam, e, s, dreamed, a.real, a.scale, a.title, rims=rims), a.fps)
    print("wrote %s: %d frames at %d a second; stretches (copy, tick): %s" % (a.out, n, a.fps, ", ".join("(%d, %d)" % (x, y) for x, y in zip(e, s))))


if __name__ == "__main__":
    main()

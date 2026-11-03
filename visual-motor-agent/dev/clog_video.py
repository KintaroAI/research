"""A video of frozen agents acting with damaged channels (task 00085), a row per run: the world, what the transformers get, and what they predict.

Each transformer gets its own damage in its own quarter, black or speckled patches, 30% by default, as eval_hunger.py chain --clog does: half held for 16 ticks, half
for one. The runs get the same damage, tick by tick, and their worlds start alike. "What they predict" is each transformer's prediction of its own patches, the same one
eval_clog.py scores: each patch from the steps before it, its neighbours in reading order and the summaries. It is not a dream. Orange squares mark the patches damaged
this tick. Error is the mean absolute difference from the clean frame on damaged patches that hold something, in this world, so far.

Usage (from visual-motor-agent/dev):
    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 ./venv/bin/python clog_video.py RUN:LABEL RUN:LABEL --ticks 600 -o OUT.mp4
"""
import argparse
import os
import sys

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import agent_lib  # noqa: E402
from eval_hunger import Life  # noqa: E402
from forage_video import BG, FOOD, MUTED, POISON, TEXT, rings, trail, write_mp4  # noqa: E402
from make_video import font  # noqa: E402
from online import DECODE  # noqa: E402
from world import make_world  # noqa: E402

SIZE = (1280, 848)
SC, HEAD, GAP = 5, 40, 24             # 64 x 64 at 320 x 320
MARK = (255, 150, 60)


class Keep:
    """A ClogStream that remembers what it gave."""

    def __init__(self, stream):
        self.stream, self.last = stream, None

    def __call__(self, frame):
        self.last = self.stream(frame)
        return self.last


def run(path, ticks, bodies, seed, p):
    life = Life(path, bodies, seed, p)
    life.clogs = [Keep(c) for c in life.clogs]
    env = make_world(bodies, life.cfg, seed=seed)
    cfgs = [m.model.config for m in life.models]
    c = cfgs[0]
    L, ps, H = c["ctx"], c["patch"], c["img"]
    own = [agent_lib.band_piece(q) for q in cfgs]
    slots = [agent_lib.uni_layout(q)["patch"] - 1 for q in cfgs]
    bg = life.cfg.bg_color[0] / 255.0
    obs = env.observe()
    rec, hist, foods, poisons, err, n = [], [], np.zeros(bodies), np.zeros(bodies), 0.0, 0
    for t in range(ticks):
        es = life.es.copy()
        cursor = env.cursor[0].copy()
        a = life.act(obs)
        real = obs["image"][0].copy()
        hist = (hist + [dict(img=np.stack([s.last[0] for s in life.clogs]), drv=obs["drives"][0], act=a[0].astype(np.float32), es=es[0])])[-L:]
        seen, pred, hit = np.zeros((H, H), np.uint8), np.zeros((H, H)), np.zeros((H, H), bool)
        for k, ((r0, r1, c0, c1), sl) in enumerate(zip(own, slots)):
            y0, y1, x0, x1 = r0 * ps, (r1 + 1) * ps, c0 * ps, (c1 + 1) * ps
            got = hist[-1]["img"][k]
            seen[y0:y1, x0:x1] = got[y0:y1, x0:x1, 0]
            for r in range(r0, r1 + 1):
                for q in range(c0, c1 + 1):
                    a0, b0 = r * ps, q * ps
                    hit[a0:a0 + ps, b0:b0 + ps] = (got[a0:a0 + ps, b0:b0 + ps] != real[a0:a0 + ps, b0:b0 + ps]).any()
            if len(hist) == L:
                st = lambda f: np.stack([w[f] for w in hist])[None]                                  # noqa: E731
                v = life.models[k].model.forward_uni_step(np.stack([w["img"][k] for w in hist])[None], st("drv"), st("act"), st("es"), L - 1,
                                                          mode=DECODE["pair"])["values_t"][0, sl]
                v = v.reshape(r1 - r0 + 1, c1 - c0 + 1, ps, ps).transpose(0, 2, 1, 3).reshape((r1 - r0 + 1) * ps, (c1 - c0 + 1) * ps)
                pred[y0:y1, x0:x1] = v
        if len(hist) == L:
            clean = real[..., 0] / 255.0
            on = hit & (np.abs(clean - bg) > 3.0 / 255)
            err += float(np.abs(pred - clean)[on].sum())
            n += int(on.sum())
        obs, _, done, inf = env.step(a)
        life.es = done.astype(np.uint8)
        foods += inf["success"]
        poisons += inf["gt_hazard_click"]
        kind = "food" if inf["success"][0] else "poison" if inf["gt_hazard_click"][0] else None
        rec.append(dict(real=real[..., 0], seen=seen, pred=np.clip(pred * 255, 0, 255).astype(np.uint8), hit=hit, cursor=cursor, kind=kind,
                        food=int(foods[0]), poison=int(poisons[0]), hunger=float(inf["gt_drives"][0, 0]), err=err / max(n, 1)))
    life.close()
    return rec, foods, poisons


def picture(gray, side):
    return Image.fromarray(np.repeat(gray[..., None], 3, 2)).resize((side, side), Image.NEAREST)


def marks(d, hit, x0, y0, patch):
    for r in range(hit.shape[0] // patch):
        for c in range(hit.shape[1] // patch):
            if hit[r * patch, c * patch]:
                x, y = x0 + c * patch * SC, y0 + r * patch * SC
                d.rectangle([x, y, x + patch * SC - 1, y + patch * SC - 1], outline=MARK)


def frame(recs, labels, t, patch):
    im = Image.new("RGB", SIZE, BG)
    d = ImageDraw.Draw(im)
    size = recs[0][0]["real"].shape[0]
    side = size * SC
    xs = [GAP + i * (side + GAP) for i in range(3)]
    d.text((GAP, 6), "step %d / %d" % (t + 1, len(recs[0])), font=font(16, True), fill=TEXT)
    for i, (rec, label) in enumerate(zip(recs, labels)):
        r = rec[t]
        y0 = 2 * HEAD - 8 + i * (side + HEAD + 12)
        for x, name in zip(xs, ("%s: the world" % label, "what the transformers get", "what they predict")):
            d.text((x, y0 - 22), name, font=font(16, True), fill=TEXT)
        for x, key in zip(xs, ("real", "seen", "pred")):
            im.paste(picture(r[key], side), (x, y0))
        trail(d, rec, t, xs[0], y0, SC, r["cursor"])
        rings(d, rec, t, xs[0], y0, SC, r["cursor"])
        marks(d, r["hit"], xs[1], y0, patch)
        marks(d, r["hit"], xs[2], y0, patch)
        px = xs[2] + side + GAP
        d.text((px, y0 + 10), "food %d" % r["food"], font=font(22, True), fill=FOOD)
        d.text((px, y0 + 44), "poison %d" % r["poison"], font=font(22, True), fill=POISON)
        d.text((px, y0 + 86), "hunger %.2f" % r["hunger"], font=font(16), fill=MUTED)
        d.text((px, y0 + 116), "error on damaged", font=font(16), fill=MUTED)
        d.text((px, y0 + 138), "things %.3f" % r["err"], font=font(18, True), fill=TEXT)
    d.text((GAP, SIZE[1] - 26), "orange: patches damaged this tick, each transformer its own, the same for both rows", font=font(14), fill=MUTED)
    return im


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("runs", nargs="+", help="RUN:LABEL, two at most")
    ap.add_argument("--ticks", type=int, default=600)
    ap.add_argument("--bodies", type=int, default=8)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--p", type=float, default=0.3)
    ap.add_argument("--slow", type=int, default=2, help="frames per tick: at one, the damage flickers too fast to see")
    ap.add_argument("-o", "--out", required=True)
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    recs, labels, patch = [], [], 8
    for r in a.runs:
        path, label = r.split(":")
        rec, foods, poisons = run(path, a.ticks, a.bodies, a.seed, a.p)
        k = 1000.0 / a.ticks
        print("%-24s world 0: food %d poison %d, error %.3f; all %d worlds: food %.1f poison %.1f per 1,000 ticks" % (
            label, rec[-1]["food"], rec[-1]["poison"], rec[-1]["err"], a.bodies, foods.mean() * k, poisons.mean() * k), flush=True)
        recs.append(rec)
        labels.append(label)
    write_mp4(a.out, (frame(recs, labels, t, patch) for t in range(a.ticks) for _ in range(a.slow)), SIZE, "5M")


if __name__ == "__main__":
    main()

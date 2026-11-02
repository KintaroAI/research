"""A video of frozen two-senses agents (task 00081) side by side: for each run its view and text sense as the model gets them, the
teacher's words, what is true, and its food and poison, with the eyes open and closed. The worlds start alike. While the eyes are closed
the model sees black: the dim picture is the real world, drawn for the viewer. Those ticks repeat (--slow), since a sense stays off
for at most 16 ticks. Moves as eval_steer.py: gain, and windows without the ring.

Usage (from visual-motor-agent/dev):
    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 ./venv/bin/python sense_video.py RUN:LABEL RUN:LABEL --ticks 1500 -o OUT
"""
import argparse
import json
import os
import sys

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import agent_lib  # noqa: E402
import cuda_lib  # noqa: E402
from forage_video import BG, FOOD, MUTED, POISON, TEXT, rings, trail, write_mp4  # noqa: E402
from lifetime import ckpt_name, pieces_move  # noqa: E402
from make_video import font  # noqa: E402
from online import DEFAULT_ONLINE, uni_act  # noqa: E402
from world import make_config, make_world  # noqa: E402
from world.sense_world import WORDS  # noqa: E402

SIZE = (1280, 816)
SC = 5                                # 64 x 64 at 320 x 320
OFF = (255, 150, 60)


def words(ix):
    return [ws[int(i)] for ws, i in zip(WORDS, ix)]


def run(path, ticks, bodies, seed):
    """Body 0's life, tick by tick, and running counts over all the bodies."""
    info = json.load(open(os.path.join(path, "info.json")))
    on = dict(DEFAULT_ONLINE, **info["config"]["online"])
    S = int(info.get("pieces", 2))
    agents = [agent_lib.Agent.load(os.path.join(path, ckpt_name("final", k, S))) for k in range(S)]
    models = [cuda_lib.CudaModel(a) for a in agents]
    c = agents[0].config
    L, bins = c["ctx"], c["bins"]
    env = make_world(bodies, make_config(info["config"]["env"], auto_reset=True), seed=seed)
    rng = np.random.default_rng(seed)
    obs = env.observe()
    hist, es = [], np.ones(bodies, np.uint8)
    count = dict(food=0, poison=0, food_blind=0, poison_blind=0, all_food=0, all_poison=0, all_food_blind=0, all_poison_blind=0)
    rec = []
    for t in range(ticks):
        cur = dict(img=obs["image"], drv=obs["drives"], act=np.zeros((bodies, 3), np.float32), es=es)
        win = hist + [cur]
        st = lambda k: np.stack([w[k] for w in win], 1)   # noqa: E731,B023
        props = [uni_act(m, st("img"), st("drv"), st("act"), st("es"), bins, rng, on["instinct_m"], on["instinct_tol"], on["instinct_h"],
                         on["instinct_jacobi"], on["instinct_decode"], per_body=True, first="sample", ring=False) for m in models]
        a, _ = pieces_move(models, props, "gain", rng, prev=None)
        blind = env.eyes > 0
        cur = dict(cur, act=a)
        hist = (hist + [cur])[-(L - 1):]
        obs, _, done, inf = env.step(a)
        es = done.astype(np.uint8)
        hit, bad = inf["success"], inf["gt_hazard_click"]
        count["food"] += int(hit[0])
        count["poison"] += int(bad[0])
        count["food_blind"] += int(hit[0] and blind[0])
        count["poison_blind"] += int(bad[0] and blind[0])
        count["all_food"] += int(hit.sum())
        count["all_poison"] += int(bad.sum())
        count["all_food_blind"] += int((hit & blind).sum())
        count["all_poison_blind"] += int((bad & blind).sum())
        rec.append(dict(img=obs["image"][0].copy(), real=env.world.observe()["image"][0].copy(), cursor=inf["gt_cursor"][0].copy(),
                        kind="food" if hit[0] else "poison" if bad[0] else None, hunger=float(inf["gt_drives"][0, 0]),
                        eyes=int(env.eyes[0]), quiet=int(env.quiet[0]), shown=env.shown[0].copy(), said=env.said[0].copy(), **count))
    for m in models:
        m.close()
    return rec, c["img"]


def big(a):
    a = np.asarray(a).astype(np.uint8)
    return Image.fromarray(np.repeat(a[..., :1], 3, -1) if a.shape[-1] != 3 else a).resize((64 * SC, 64 * SC), Image.NEAREST)


def panel(im, d, rec, t, label, H, y0):
    r = rec[t]
    x0, x1, x2 = 40, 40 + 64 * SC + 24, 40 + 2 * 64 * SC + 56
    ty = y0 + 26
    d.text((x0, y0), label, font=font(18, True), fill=TEXT)
    d.text((x0 + 64 * SC - 36, y0 + 3), "view", font=font(15), fill=MUTED)
    d.text((x1, y0 + 3), "text sense", font=font(15), fill=MUTED)
    if r["eyes"]:                     # the model sees black; the viewer sees the world dimly
        im.paste(big((r["real"][:, :H] * 0.35)), (x0, ty))
    else:
        im.paste(big(r["img"][:, :H]), (x0, ty))
    trail(d, rec, t, x0, ty, SC, r["cursor"])
    rings(d, rec, t, x0, ty, SC, r["cursor"])
    if r["eyes"]:
        d.rectangle([x0, ty, x0 + 64 * SC - 1, ty + 64 * SC - 1], outline=OFF, width=3)
        d.text((x0 + 10, ty + 8), "EYES CLOSED (%d)" % r["eyes"], font=font(17, True), fill=OFF)
    im.paste(big(r["img"][:, H:2 * H]), (x1, ty))
    said, true = words(r["shown"]), words(r["said"])
    if r["quiet"]:
        d.rectangle([x1, ty, x1 + 64 * SC - 1, ty + 64 * SC - 1], outline=OFF, width=3)
        d.text((x1 + 10, ty + 8), "TEACHER SILENT (%d)" % r["quiet"], font=font(17, True), fill=OFF)
    else:
        b = H // 8 * SC
        for row, w in enumerate(said):
            for col, ch in enumerate(w):
                level = int(r["img"][row * (H // 8), H + col * (H // 8), 0])
                d.text((x1 + col * b + 12, ty + row * b + 6), ch, font=font(20, True), fill=(0, 0, 0) if level > 170 else (255, 255, 255))
    y = ty
    d.text((x2, y), "the teacher says", font=font(15), fill=MUTED)
    d.text((x2, y + 20), "(silent)" if r["quiet"] else " ".join(said), font=font(20, True), fill=MUTED if r["quiet"] else TEXT)
    d.text((x2, y + 56), "what is true", font=font(15), fill=MUTED)
    d.text((x2, y + 76), " ".join(true), font=font(20, True), fill=TEXT if said == true else POISON)
    y += 124
    d.text((x2, y), "this world", font=font(15), fill=MUTED)
    d.text((x2, y + 20), "food %d" % r["food"], font=font(20, True), fill=FOOD)
    d.text((x2 + 120, y + 20), "poison %d" % r["poison"], font=font(20, True), fill=POISON)
    d.text((x2 + 270, y + 24), "hunger %.2f" % r["hunger"], font=font(15), fill=MUTED)
    y += 66
    d.text((x2, y), "all 16 worlds, eyes closed", font=font(15), fill=MUTED)
    d.text((x2, y + 20), "food %d" % r["all_food_blind"], font=font(20, True), fill=FOOD)
    d.text((x2 + 120, y + 20), "poison %d" % r["all_poison_blind"], font=font(20, True), fill=POISON)


def frame(recs, labels, H, t):
    im = Image.new("RGB", SIZE, BG)
    d = ImageDraw.Draw(im)
    d.text((40, 6), "tick %d / %d" % (t + 1, len(recs[0])), font=font(16, True), fill=TEXT)
    d.text((220, 8), "while the eyes are closed the model sees black: the dim picture is for you, at quarter speed", font=font(14), fill=MUTED)
    for i, (rec, label) in enumerate(zip(recs, labels)):
        panel(im, d, rec, t, label, H, 30 + i * 392)
    return im


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("runs", nargs="+", help="RUN:LABEL, one or two")
    ap.add_argument("--ticks", type=int, default=1500)
    ap.add_argument("--bodies", type=int, default=16)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--slow", type=int, default=4, help="how many times a tick with the eyes closed is shown")
    ap.add_argument("-o", "--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    recs, labels = [], []
    for spec in a.runs:
        p, label = spec.split(":", 1)
        rec, H = run(p, a.ticks, a.bodies, a.seed)
        recs.append(rec)
        labels.append(label)
        e = rec[-1]
        print("%-28s world 0: food %d poison %d; all %d worlds, per 1k ticks: food %.1f poison %.1f; eyes closed, count: food %d poison %d" % (
            label, e["food"], e["poison"], a.bodies, 1000.0 * e["all_food"] / a.ticks / a.bodies, 1000.0 * e["all_poison"] / a.ticks / a.bodies,
            e["all_food_blind"], e["all_poison_blind"]), flush=True)

    def frames():
        for t in range(a.ticks):
            n = a.slow if any(rec[t]["eyes"] for rec in recs) else 1
            im = frame(recs, labels, H, t)
            for _ in range(n):
                yield im

    write_mp4(os.path.join(a.out, "compare.mp4"), frames(), size=SIZE)
    t = next(i for i in range(a.ticks // 2, a.ticks) if recs[0][i]["eyes"] > 4)
    frame(recs, labels, H, t).save(os.path.join(a.out, "still.png"))


if __name__ == "__main__":
    main()

"""A video of a frozen universal agent foraging (as online.py's forage test: a continual world, the instinct on).

Usage (from visual-motor-agent/dev):
    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 ./venv/bin/python forage_video.py RUN --step 500000 -o OUT_DIR
"""
import argparse
import dataclasses
import json
import os
import sys

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import agent_lib  # noqa: E402
import cuda_lib  # noqa: E402
from evaluate import executed_action  # noqa: E402
from make_video import font  # noqa: E402
from online import DEFAULT_ONLINE, EVAL_SEED, _GpuEval, eval_moves  # noqa: E402
from world import CircleWorld, EnvConfig  # noqa: E402

W, H, FPS = 1280, 720, 15
BG, PANEL = (12, 12, 18), (26, 26, 36)
TEXT, MUTED = (232, 232, 238), (150, 152, 168)
FOOD, POISON, EMPTY, TRAIL = (90, 225, 125), (255, 95, 75), (200, 200, 210), (255, 205, 60)
SCALE, VX, VY = 10, 40, 40          # the agent's 64x64 view, upscaled to 640x640


def run(args):
    with open(os.path.join(args.run, "info.json")) as f:
        info = json.load(f)
    ec = dict(info["env_config"])
    for k in ("bg_color", "target_color", "cursor_color", "cursor_outline", "hazard_color"):
        if k in ec:
            ec[k] = tuple(ec[k])
    agent = agent_lib.Agent.load(os.path.join(args.run, "ckpt_%08d.bin" % args.step))
    ev = _GpuEval(agent, cuda_lib.CudaModel(agent))
    ion = dict(DEFAULT_ONLINE, **info["config"]["online"])
    inst = (ion["instinct_m"], ion["instinct_tol"], ion["instinct_h"], ion["instinct_jacobi"], ion["instinct_decode"])
    goal = info["config"]["online"]["goal"]
    n = args.worlds
    cfg = EnvConfig(**ec)
    env = CircleWorld(n, dataclasses.replace(cfg, auto_reset=False, bonuses=False, max_steps=10 ** 9), seed=EVAL_SEED)
    rng = np.random.default_rng(0)
    L = agent.config["ctx"]
    hist, obs = [], env.observe()
    prev_a = np.zeros((n, 3), np.float32)
    es = np.ones(n, np.uint8)
    rec = []    # per step, for world 0: the frame and cursor after the move, the click's kind, hunger
    food, poison = np.zeros(n, int), np.zeros(n, int)
    for t in range(args.steps):
        cur = dict(img=obs["image"], drv=obs["drives"], pa=prev_a,
                   cmd=(obs["drives"][:, 0] if goal else np.zeros(n)).astype(np.float32), es=es,
                   act=np.zeros((n, 3), np.float32))
        a = eval_moves(ev, hist + [cur], rng, inst)
        cur["act"] = a
        obs, _, _, inf = env.step(a)
        hz = inf.get("gt_hazard_click", np.zeros(n, bool))
        food += inf["success"]
        poison += hz
        kind = "food" if inf["success"][0] else "poison" if hz[0] else "empty" if inf["click"][0] else None
        rec.append(dict(img=obs["image"][0].copy(), cursor=inf["gt_cursor"][0].copy(), kind=kind, hunger=float(inf["gt_drives"][0, 0]),
                        food=int(food[0]), poison=int(poison[0])))
        hist = (hist + [cur])[-(L - 1):] if L > 1 else []
        prev_a = executed_action(a)
        es = np.zeros(n, np.uint8)
    return rec, food, poison


def view(img):
    a = np.asarray(img)
    if a.ndim == 2 or a.shape[-1] == 1:
        a = np.repeat(a.reshape(a.shape[0], a.shape[1], 1), 3, 2)
    return Image.fromarray(a.astype(np.uint8)).resize((64 * SCALE, 64 * SCALE), Image.NEAREST)


def frame(rec, t, title):
    im = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(im)
    im.paste(view(rec[t]["img"]), (VX, VY))
    # the cursor's last 30 positions (wrapping world: break the line where it jumps across an edge)
    pts = [r["cursor"] for r in rec[max(0, t - 30):t + 1]]
    for p, q in zip(pts, pts[1:]):
        if np.abs(p - q).max() < 20:
            d.line([(VX + p[0] * SCALE, VY + p[1] * SCALE), (VX + q[0] * SCALE, VY + q[1] * SCALE)], fill=TRAIL, width=2)
    for k in range(max(0, t - 8), t + 1):        # clicks fade over 8 steps
        r = rec[k]
        if r["kind"]:
            c = dict(food=FOOD, poison=POISON, empty=EMPTY)[r["kind"]]
            rad = 14 + 3 * (t - k)
            x, y = VX + r["cursor"][0] * SCALE, VY + r["cursor"][1] * SCALE
            d.ellipse([x - rad, y - rad, x + rad, y + rad], outline=c, width=3 if t - k < 4 else 1)
    # the panel
    px = VX + 64 * SCALE + 40
    d.rectangle([px, VY, W - 40, VY + 640], fill=PANEL)
    x0 = px + 24
    d.text((x0, VY + 20), title, font=font(24, True), fill=TEXT)
    d.text((x0, VY + 56), "frozen, the 1-step instinct; the agent's own 64x64 view", font=font(16), fill=MUTED)
    d.text((x0, VY + 100), "step %4d / %d" % (t + 1, len(rec)), font=font(22), fill=TEXT)
    r = rec[t]
    d.text((x0, VY + 140), "food clicks  %d" % r["food"], font=font(22, True), fill=FOOD)
    d.text((x0 + 260, VY + 140), "poison clicks  %d" % r["poison"], font=font(22, True), fill=POISON)
    # hunger bar
    d.text((x0, VY + 190), "hunger %.2f" % r["hunger"], font=font(18), fill=TEXT)
    bw = W - 40 - 24 - x0
    d.rectangle([x0, VY + 218, x0 + bw, VY + 234], outline=MUTED)
    d.rectangle([x0, VY + 218, x0 + int(bw * min(1.0, r["hunger"])), VY + 234],
                fill=POISON if r["hunger"] >= 0.999 else TRAIL)
    # hunger over the whole video so far, with click marks
    cx, cy, cw, ch = x0, VY + 280, bw, 220
    d.rectangle([cx, cy, cx + cw, cy + ch], outline=(60, 62, 78))
    d.text((cx, cy - 22), "hunger over the 1,000 steps", font=font(16), fill=MUTED)
    T = len(rec)
    hs = [(cx + cw * k / (T - 1), cy + ch * (1 - min(1.0, rec[k]["hunger"]))) for k in range(t + 1)]
    if len(hs) > 1:
        d.line(hs, fill=TRAIL, width=2)
    for k in range(t + 1):
        if rec[k]["kind"] in ("food", "poison"):
            xk = cx + cw * k / (T - 1)
            d.line([(xk, cy + ch + 4), (xk, cy + ch + 16)], fill=FOOD if rec[k]["kind"] == "food" else POISON, width=2)
    # legend
    ly = VY + 540
    for i, (c, s) in enumerate(((FOOD, "click on food: hunger to 0"), (POISON, "click on poison (checkered): +0.5"),
                                (EMPTY, "click on nothing"), (TRAIL, "cursor path"))):
        y = ly + 24 * i
        d.ellipse([x0, y + 3, x0 + 14, y + 17], outline=c, width=3)
        d.text((x0 + 24, y), s, font=font(16), fill=MUTED)
    return im


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--step", type=int, default=500000)
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--worlds", type=int, default=8, help="worlds run at once (the video shows the first)")
    ap.add_argument("--title", default="")
    ap.add_argument("-o", "--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    rec, food, poison = run(args)
    title = args.title or "%s at %dk ticks" % (os.path.basename(args.run.rstrip("/")), args.step // 1000)
    print("world 0: clicks on nothing %d;" % sum(r["kind"] == "empty" for r in rec), end=" ")
    print("world 0: food %d poison %d; all %d worlds: food %.1f poison %.1f per %d steps"
          % (food[0], poison[0], len(food), food.mean(), poison.mean(), args.steps))
    import imageio_ffmpeg
    gen = imageio_ffmpeg.write_frames(os.path.join(args.out, "forage.mp4"), (W, H), fps=FPS, codec="libx264",
                                      quality=None, bitrate="4M", macro_block_size=8)
    gen.send(None)
    for t in range(len(rec)):
        gen.send(np.asarray(frame(rec, t, title)).tobytes())
    gen.close()
    # a sheet: every 100th step, the view with the last 100 steps' path and clicks
    sheet = Image.new("RGB", (5 * 330 + 10, 2 * 360 + 10), BG)
    for i, t in enumerate(range(99, len(rec), 100)):
        v = view(rec[t]["img"]).resize((320, 320), Image.NEAREST)
        dv = ImageDraw.Draw(v)
        pts = [r["cursor"] for r in rec[t - 99:t + 1]]
        for p, q in zip(pts, pts[1:]):
            if np.abs(p - q).max() < 20:
                dv.line([(p[0] * 5, p[1] * 5), (q[0] * 5, q[1] * 5)], fill=TRAIL, width=1)
        for r in rec[t - 99:t + 1]:
            if r["kind"]:
                c = dict(food=FOOD, poison=POISON, empty=EMPTY)[r["kind"]]
                x, y = r["cursor"] * 5
                dv.ellipse([x - 5, y - 5, x + 5, y + 5], outline=c, width=2)
        x, y = 10 + (i % 5) * 330, 10 + (i // 5) * 360
        sheet.paste(v, (x, y))
        ds = ImageDraw.Draw(sheet)
        f100 = sum(r["kind"] == "food" for r in rec[t - 99:t + 1])
        p100 = sum(r["kind"] == "poison" for r in rec[t - 99:t + 1])
        ds.text((x, y + 324), "steps %d-%d: food %d, poison %d" % (t - 98, t + 1, f100, p100), font=font(15), fill=TEXT)
    sheet.save(os.path.join(args.out, "forage_sheet.png"))


if __name__ == "__main__":
    main()

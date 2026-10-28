"""A video of a frozen universal agent foraging (as online.py's forage test: a continual world, the instinct on).

Usage (from visual-motor-agent/dev):
    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 ./venv/bin/python forage_video.py RUN --step 500000 -o OUT_DIR

--horizon / --decode override the run's own instinct (instinct_h, instinct_decode). --vs RUN2 also runs a second
agent in the same worlds (same seeds) and writes compare.mp4: the two views side by side.
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


def run(args, path):
    with open(os.path.join(path, "info.json")) as f:
        info = json.load(f)
    ec = dict(info["env_config"])
    for k in ("bg_color", "target_color", "cursor_color", "cursor_outline", "hazard_color"):
        if k in ec:
            ec[k] = tuple(ec[k])
    agent = agent_lib.Agent.load(os.path.join(path, "ckpt_%08d.bin" % args.step))
    ev = _GpuEval(agent, cuda_lib.CudaModel(agent))
    ion = dict(DEFAULT_ONLINE, **info["config"]["online"])
    if args.horizon:
        ion["instinct_h"] = args.horizon
    if args.decode:
        ion["instinct_decode"] = args.decode
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
        kind = ("food" if inf["success"][0] else "poison" if hz[0]
                else "empty" if inf["click"][0] and not cfg.touch else None)
        rec.append(dict(img=obs["image"][0].copy(), cursor=inf["gt_cursor"][0].copy(), kind=kind, hunger=float(inf["gt_drives"][0, 0]),
                        food=int(food[0]), poison=int(poison[0])))
        hist = (hist + [cur])[-(L - 1):] if L > 1 else []
        prev_a = executed_action(a)
        es = np.zeros(n, np.uint8)
    m = ion["instinct_m"]
    how = ("the instinct off" if not m else "the 1-step instinct" if ion["instinct_h"] == 1
           else "a %d-step lookahead (%s)" % (ion["instinct_h"], ion["instinct_decode"]))
    mc = agent.config
    meta = dict(touch=cfg.touch, ego=cfg.egocentric, verb="touches" if cfg.touch else "clicks",
                sub="frozen, %s%s; the agent's own 64x64 view" % (how, ", egocentric" if cfg.egocentric else ""),
                poison_add=cfg.hazard_click_add, split=split_lines(mc))
    return rec, food, poison, meta


def view(img):
    a = np.asarray(img)
    if a.ndim == 2 or a.shape[-1] == 1:
        a = np.repeat(a.reshape(a.shape[0], a.shape[1], 1), 3, 2)
    return Image.fromarray(a.astype(np.uint8)).resize((64 * SCALE, 64 * SCALE), Image.NEAREST)


def frame(rec, t, title, meta):
    im = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(im)
    im.paste(view(rec[t]["img"]), (VX, VY))
    ref = rec[t]["cursor"] if meta["ego"] else None
    draw_split(d, meta, VX, VY, SCALE)
    trail(d, rec, t, VX, VY, SCALE, ref)
    rings(d, rec, t, VX, VY, SCALE, ref)
    # the panel
    px = VX + 64 * SCALE + 40
    d.rectangle([px, VY, W - 40, VY + 640], fill=PANEL)
    x0 = px + 24
    d.text((x0, VY + 20), title, font=font(24, True), fill=TEXT)
    d.text((x0, VY + 56), meta["sub"], font=font(16), fill=MUTED)
    d.text((x0, VY + 100), "step %4d / %d" % (t + 1, len(rec)), font=font(22), fill=TEXT)
    r = rec[t]
    d.text((x0, VY + 140), "food %s  %d" % (meta["verb"], r["food"]), font=font(22, True), fill=FOOD)
    d.text((x0 + 260, VY + 140), "poison %s  %d" % (meta["verb"], r["poison"]), font=font(22, True), fill=POISON)
    # hunger bar
    d.text((x0, VY + 190), "hunger %.2f" % r["hunger"], font=font(18), fill=TEXT)
    bw = W - 40 - 24 - x0
    d.rectangle([x0, VY + 218, x0 + bw, VY + 234], outline=MUTED)
    d.rectangle([x0, VY + 218, x0 + int(bw * min(1.0, r["hunger"])), VY + 234],
                fill=POISON if r["hunger"] >= 0.999 else TRAIL)
    # hunger over the whole video so far, with click marks
    cx, cy, cw, ch = x0, VY + 280, bw, 220
    d.rectangle([cx, cy, cx + cw, cy + ch], outline=(60, 62, 78))
    T = len(rec)
    d.text((cx, cy - 22), "hunger over the %d steps" % T, font=font(16), fill=MUTED)
    hs = [(cx + cw * k / (T - 1), cy + ch * (1 - min(1.0, rec[k]["hunger"]))) for k in range(t + 1)]
    if len(hs) > 1:
        d.line(hs, fill=TRAIL, width=2)
    for k in range(t + 1):
        if rec[k]["kind"] in ("food", "poison"):
            xk = cx + cw * k / (T - 1)
            d.line([(xk, cy + ch + 4), (xk, cy + ch + 16)], fill=FOOD if rec[k]["kind"] == "food" else POISON, width=2)
    # legend
    ly = VY + 540
    for i, (c, s) in enumerate(legend(meta)):
        y = ly + 24 * i
        d.ellipse([x0, y + 3, x0 + 14, y + 17], outline=c, width=3)
        d.text((x0 + 24, y), s, font=font(16), fill=MUTED)
    return im


def legend(meta):
    v = "touch" if meta["touch"] else "click on"
    return ((FOOD, "%s food: hunger to 0" % v), (POISON, "%s poison (checkered): +%g" % (v, meta["poison_add"]))) \
        + (() if meta["touch"] else ((EMPTY, "click on nothing"),)) + ((TRAIL, "cursor path"),)


def split_lines(mc):
    """Attention regions (task 00044): the overlap bands' edges in view pixels, and whether they are columns."""
    S, O = mc.get("attn_split", 0), mc.get("attn_overlap", 0)
    if not mc.get("universal") or S <= 1:
        return None
    g, P = mc["img"] // mc["patch"], mc["patch"]
    if mc.get("attn_checker", 0):                                    # task 00046: the regions' patches
        return "checker", P, S
    band = g // S
    edges = sorted({e for s in range(1, S) for e in ((s * band - O) * P, (s * band + O) * P)})
    return edges, bool(mc.get("attn_cols", 0))


def draw_split(d, meta, x0, y0, sc):
    """The overlap band(s) of a split model's attention, dashed, on its view."""
    if not meta.get("split"):
        return
    if meta["split"][0] == "checker":                                # region 0's patches: a dot at each centre
        _, P, S = meta["split"]
        for r in range(64 // P):
            for c in range(64 // P):
                if (r + c) % S == 0:
                    x, y = x0 + (c + 0.5) * P * sc, y0 + (r + 0.5) * P * sc
                    d.ellipse([x - 2, y - 2, x + 2, y + 2], fill=(120, 170, 255))
        return
    edges, cols = meta["split"]
    for e in edges:
        for k in range(0, 64 * sc, 12):
            a, b = k, min(k + 6, 64 * sc)
            seg = [(x0 + e * sc, y0 + a), (x0 + e * sc, y0 + b)] if cols else [(x0 + a, y0 + e * sc), (x0 + b, y0 + e * sc)]
            d.line(seg, fill=(120, 170, 255), width=2)


def in_view(p, ref):
    """A world position in the view: as is, or (egocentric, ref = the current cursor) relative to the cursor drawn at
    the center, wrapped."""
    return p if ref is None else np.mod(p - ref + 32.0, 64.0)


def trail(d, rec, t, x0, y0, sc, ref=None, n=30, width=2):
    """The cursor's last n positions (wrapping world: break the line where it jumps across an edge)."""
    pts = [in_view(r["cursor"], ref) for r in rec[max(0, t - n):t + 1]]
    for p, q in zip(pts, pts[1:]):
        if np.abs(p - q).max() < 20:
            d.line([(x0 + p[0] * sc, y0 + p[1] * sc), (x0 + q[0] * sc, y0 + q[1] * sc)], fill=TRAIL, width=width)


def rings(d, rec, t, x0, y0, sc, ref=None):
    """Each touch or click as a ring at the cursor, fading over 8 steps."""
    for k in range(max(0, t - 8), t + 1):
        r = rec[k]
        if r["kind"]:
            c = dict(food=FOOD, poison=POISON, empty=EMPTY)[r["kind"]]
            rad = 14 + 3 * (t - k)
            x, y = x0 + in_view(r["cursor"], ref)[0] * sc, y0 + in_view(r["cursor"], ref)[1] * sc
            d.ellipse([x - rad, y - rad, x + rad, y + rad], outline=c, width=3 if t - k < 4 else 1)


def frame_vs(recs, t, titles, metas):
    """Two agents in the same world: each view (512x512) with its path, touches and counts; hunger of both below."""
    S2, top = 8, 64
    im = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(im)
    xs = (40, W - 40 - 64 * S2)
    T = len(recs[0])
    for rec, x0, title, meta in zip(recs, xs, titles, metas):
        im.paste(view(rec[t]["img"]).resize((64 * S2, 64 * S2), Image.NEAREST), (x0, top))
        ref = rec[t]["cursor"] if meta["ego"] else None
        draw_split(d, meta, x0, top, S2)
        trail(d, rec, t, x0, top, S2, ref)
        rings(d, rec, t, x0, top, S2, ref)
        d.text((x0, 8), title, font=font(20, True), fill=TEXT)
        d.text((x0, 36), meta["sub"].split(";")[0], font=font(15), fill=MUTED)
        r = rec[t]
        y = top + 64 * S2 + 10
        d.text((x0, y), "food %d" % r["food"], font=font(22, True), fill=FOOD)
        d.text((x0 + 140, y), "poison %d" % r["poison"], font=font(22, True), fill=POISON)
        d.text((x0 + 300, y + 2), "hunger %.2f" % r["hunger"], font=font(18), fill=TEXT)
        # hunger over the video so far, with touch marks
        cy, ch, cw = y + 34, 64, 64 * S2
        d.rectangle([x0, cy, x0 + cw, cy + ch], outline=(60, 62, 78))
        hs = [(x0 + cw * k / (T - 1), cy + ch * (1 - min(1.0, rec[k]["hunger"]))) for k in range(t + 1)]
        if len(hs) > 1:
            d.line(hs, fill=TRAIL, width=2)
        for k in range(t + 1):
            if rec[k]["kind"] in ("food", "poison"):
                xk = x0 + cw * k / (T - 1)
                d.line([(xk, cy + ch + 3), (xk, cy + ch + 13)], fill=FOOD if rec[k]["kind"] == "food" else POISON,
                       width=2)
    mx = xs[0] + 64 * S2 + 12
    d.text((mx, top + 10), "step", font=font(16), fill=MUTED)
    d.text((mx, top + 30), "%d" % (t + 1), font=font(22, True), fill=TEXT)
    d.text((mx, top + 58), "/ %d" % T, font=font(16), fill=MUTED)
    for i, (c, s) in enumerate(legend(metas[0])):
        yy = top + 120 + 80 * i
        d.ellipse([mx + 40, yy, mx + 60, yy + 20], outline=c, width=3)
        for j, w in enumerate(s.replace(": ", "\n").replace(" (", "\n(").split("\n")):
            d.text((mx, yy + 26 + 17 * j), w, font=font(13), fill=MUTED)
    if any(m.get("split") for m in metas):   # attention regions (task 00044)
        yy = top + 120 + 80 * len(legend(metas[0]))
        d.line([(mx + 30, yy + 10), (mx + 36, yy + 10)], fill=(120, 170, 255), width=2)
        d.line([(mx + 42, yy + 10), (mx + 48, yy + 10)], fill=(120, 170, 255), width=2)
        for j, w in enumerate(("overlap of the", "attention halves")):
            d.text((mx, yy + 26 + 17 * j), w, font=font(13), fill=MUTED)
    return im


def write_mp4(path, frames):
    import imageio_ffmpeg
    gen = imageio_ffmpeg.write_frames(path, (W, H), fps=FPS, codec="libx264", quality=None, bitrate="4M",
                                      macro_block_size=8)
    gen.send(None)
    for im in frames:
        gen.send(np.asarray(im).tobytes())
    gen.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--step", type=int, default=500000)
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--worlds", type=int, default=8, help="worlds run at once (the video shows the first)")
    ap.add_argument("--title", default="")
    ap.add_argument("--horizon", type=int, default=0, help="override the run's instinct_h")
    ap.add_argument("--decode", default="", choices=["", "mean", "mode", "pair"], help="override instinct_decode")
    ap.add_argument("--vs", default="", help="a second run: compare.mp4 shows both in the same worlds")
    ap.add_argument("--vs-title", default="")
    ap.add_argument("-o", "--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    rec, food, poison, meta = run(args, args.run)
    title = args.title or "%s at %dk ticks" % (os.path.basename(args.run.rstrip("/")), args.step // 1000)
    print("world 0: clicks on nothing %d;" % sum(r["kind"] == "empty" for r in rec), end=" ")
    print("world 0: food %d poison %d; all %d worlds: food %.1f poison %.1f per %d steps"
          % (food[0], poison[0], len(food), food.mean(), poison.mean(), args.steps))
    write_mp4(os.path.join(args.out, "forage.mp4"), (frame(rec, t, title, meta) for t in range(len(rec))))
    if args.vs:
        rec2, food2, poison2, meta2 = run(args, args.vs)
        title2 = args.vs_title or "%s at %dk ticks" % (os.path.basename(args.vs.rstrip("/")), args.step // 1000)
        print("vs: world 0: food %d poison %d; all %d worlds: food %.1f poison %.1f per %d steps"
              % (food2[0], poison2[0], len(food2), food2.mean(), poison2.mean(), args.steps))
        write_mp4(os.path.join(args.out, "compare.mp4"),
                  (frame_vs((rec, rec2), t, (title, title2), (meta, meta2)) for t in range(len(rec))))
    # a sheet: every 100th step, the view with the last 100 steps' path and clicks
    sheet = Image.new("RGB", (5 * 330 + 10, 2 * 360 + 10), BG)
    for i, t in enumerate(range(99, len(rec), 100)):
        v = view(rec[t]["img"]).resize((320, 320), Image.NEAREST)
        dv = ImageDraw.Draw(v)
        ref = rec[t]["cursor"] if meta["ego"] else None
        pts = [in_view(r["cursor"], ref) for r in rec[t - 99:t + 1]]
        for p, q in zip(pts, pts[1:]):
            if np.abs(p - q).max() < 20:
                dv.line([(p[0] * 5, p[1] * 5), (q[0] * 5, q[1] * 5)], fill=TRAIL, width=1)
        for r in rec[t - 99:t + 1]:
            if r["kind"]:
                c = dict(food=FOOD, poison=POISON, empty=EMPTY)[r["kind"]]
                x, y = in_view(r["cursor"], ref) * 5
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

"""Forage for R real steps (the frozen agent with its instinct), then dream D steps: the model alone, from its last ctx
steps, predicts the next step and feeds it back. Each dreamed step: the body is the previous move token's predicted
next body; the frame is decoded patch by patch (Jacobi iterations until the frame stops changing: exactly sequential
decoding), each pixel its expected value (mean), most probable bin (mode) or most probable two-hot pair (pair, task
00038); the move is sampled from the move token's
predicted bins (the policy, no instinct). The same moves are also made in the real world, for comparison.

Usage (from visual-motor-agent/dev):
    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 ./venv/bin/python dream_video.py RUN --step 500000 -o OUT
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
from online import DECODE, DEFAULT_ONLINE, EVAL_SEED, _GpuEval, eval_moves, uni_frame  # noqa: E402
from world import CircleWorld, EnvConfig  # noqa: E402

W, H, FPS = 1280, 720, 10
BG, PANEL = (12, 12, 18), (26, 26, 36)
TEXT, MUTED = (232, 232, 238), (150, 152, 168)
FOOD, POISON, EMPTY, TRAIL, DREAM = (90, 225, 125), (255, 95, 75), (200, 200, 210), (255, 205, 60), (120, 170, 255)
S, VY = 8, 70                           # views 512x512
LX, RX = 60, W - 60 - 64 * S


def sample_move(probs, rng):
    K = probs.shape[-1]
    idx = np.minimum((np.cumsum(probs, -1) < rng.random((3, 1))).sum(-1), K - 1)
    c = idx / (K - 1)
    return np.array([2 * c[0] - 1, 2 * c[1] - 1, float(c[2] > 0.5)], np.float32)


def run(args):
    info = json.load(open(os.path.join(args.run, "info.json")))
    ec = {k: tuple(v) if isinstance(v, list) else v for k, v in info["env_config"].items()}
    agent = agent_lib.Agent.load(os.path.join(args.run, "ckpt_%08d.bin" % args.step))
    gpu = cuda_lib.CudaModel(agent)
    ev = _GpuEval(agent, gpu)
    c = agent.config
    P = (c["img"] // c["patch"]) ** 2
    ion = dict(DEFAULT_ONLINE, **info["config"]["online"])
    inst = (ion["instinct_m"], ion["instinct_tol"], ion["instinct_h"], ion["instinct_jacobi"], ion["instinct_decode"])
    env = CircleWorld(1, dataclasses.replace(EnvConfig(**ec), auto_reset=False, bonuses=False, max_steps=10 ** 9),
                      seed=EVAL_SEED + args.world)
    rng = np.random.default_rng(args.world)
    L = c["ctx"]
    hist, obs = [], env.observe()
    prev_a, es = np.zeros((1, 3), np.float32), np.ones(1, np.uint8)
    rec = []
    pending = dict(kind=None, cursor=env.cursor[0].copy(), dkind=None)   # the last move's outcome, shown with its frame

    def outcome(inf):
        return ("food" if inf["success"][0] else "poison" if inf.get("gt_hazard_click", [False])[0]
                else "empty" if inf["click"][0] else None)
    # 1. forage: the agent acts (instinct on); the right view shows the model's prediction of each frame
    for t in range(args.real):
        cur = dict(img=obs["image"], drv=obs["drives"], pa=prev_a, cmd=np.zeros(1, np.float32), es=es,
                   act=np.zeros((1, 3), np.float32))
        win = hist + [cur]
        st = lambda k: np.stack([w[k] for w in win], 1)   # noqa: E731
        pred = gpu.forward_uni_step(st("img"), st("drv").astype(np.float32), st("act").astype(np.float32), st("es"),
                                    len(win) - 1, mode=DECODE[args.decode])
        a = eval_moves(ev, win, rng, inst)
        cur["act"] = a
        rec.append(dict(phase="real", real=obs["image"][0], pred=uni_frame(c, pred["values_t"][:, :P])[0],
                        real_h=float(obs["drives"][0, 0]), pred_h=float(pred["next_body"][0, -2, 0]) if len(win) > 1
                        else float("nan"), **pending))
        obs, _, _, inf = env.step(a)
        pending = dict(kind=outcome(inf), cursor=inf["gt_cursor"][0].copy(), dkind="click" if a[0, 2] > 0.5 else None)
        hist = (hist + [cur])[-(L - 1):]
        prev_a, es = executed_action(a), np.zeros(1, np.uint8)
    # 2. dream: the window is the last ctx steps (real ones first); each new step is the model's own prediction
    win = hist + [dict(img=obs["image"], drv=obs["drives"], act=np.zeros((1, 3), np.float32), es=es)]
    img = np.stack([w["img"] for w in win], 1)
    drv = np.stack([w["drv"] for w in win], 1).astype(np.float32)
    act = np.stack([w["act"] for w in win], 1).astype(np.float32)
    e = np.stack([w["es"] for w in win], 1)
    # the dream's first step is the next one: its body from the last real step's move token
    for t in range(args.dream):
        if t == 0:   # the last real step is known: the dream starts after it
            probs = gpu.forward_uni(img, drv, act, e, values=False, move_probs=True)["move_probs"][0, -1]
            act[0, -1] = sample_move(probs, rng)
        nb = gpu.forward_uni(img, drv, act, e, values=False, move_probs=False, next_body=True)["next_body"][0, -1]
        real_a = act[:, -1].copy()
        obs, _, _, inf = env.step(real_a)      # the same move in the real world
        # shift: a new step with the predicted body, the frame decoded patch by patch
        img = np.concatenate([img[:, 1:], img[:, -1:]], 1)
        drv = np.concatenate([drv[:, 1:], np.clip(nb, 0, 1)[None, None]], 1)
        act = np.concatenate([act[:, 1:], np.zeros((1, 1, 3), np.float32)], 1)
        e = np.concatenate([e[:, 1:], np.zeros((1, 1), np.uint8)], 1)
        for it in range(P + 1):
            out = gpu.forward_uni_step(img, drv, act, e, L - 1, mode=DECODE[args.decode])
            fr = uni_frame(c, out["values_t"][:, :P])
            done = np.array_equal(fr[0], img[0, -1])
            img[:, -1] = fr
            if done:
                break
        probs = gpu.forward_uni(img, drv, act, e, values=False, move_probs=True)["move_probs"][0, -1]
        act[0, -1] = sample_move(probs, rng)
        rec.append(dict(phase="dream", real=obs["image"][0], pred=img[0, -1].copy(), real_h=float(obs["drives"][0, 0]),
                        pred_h=float(drv[0, -1, 0]), kind=outcome(inf), cursor=inf["gt_cursor"][0].copy(),
                        dkind="click" if real_a[0, 2] > 0.5 else None, iters=it + 1))
    return rec


def view(a):
    a = np.asarray(a)
    return Image.fromarray(a.astype(np.uint8)).resize((64 * S, 64 * S), Image.NEAREST)


def frame(rec, t, title, R):
    im = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(im)
    r = rec[t]
    dream = r["phase"] == "dream"
    d.text((LX, 18), title, font=font(22, True), fill=TEXT)
    d.text((W - 60, 22), "step %d / %d" % (t + 1, len(rec)), font=font(20), fill=TEXT, anchor="ra")
    d.text((LX, VY - 26), "the real world" + (" (the dreamed moves made in it)" if dream else " (the agent acts)"),
           font=font(16), fill=MUTED)
    d.text((RX, VY - 26), ("DREAM: the model's own next frame, fed back" if dream
                           else "the model's prediction of this frame"), font=font(16, dream),
           fill=DREAM if dream else MUTED)
    im.paste(view(r["real"]), (LX, VY))
    im.paste(view(r["pred"]), (RX, VY))
    if dream:
        d.rectangle([RX - 3, VY - 3, RX + 64 * S + 2, VY + 64 * S + 2], outline=DREAM, width=3)
    # clicks (fade over 6 steps): real outcome on the left, the model's own click on the right
    for k in range(max(0, t - 6), t + 1):
        q = rec[k]
        w = 3 if t - k < 3 else 1
        if q["kind"]:
            x, y = LX + q["cursor"][0] * S, VY + q["cursor"][1] * S
            col = dict(food=FOOD, poison=POISON, empty=EMPTY)[q["kind"]]
            d.ellipse([x - 12, y - 12, x + 12, y + 12], outline=col, width=w)
    # hunger: real (yellow) and the model's (blue: predicted in phase 1, dreamed in phase 2)
    cx, cy, cw, ch = LX, VY + 64 * S + 40, W - 2 * LX, 80
    d.rectangle([cx, cy, cx + cw, cy + ch], outline=(60, 62, 78))
    T = len(rec)
    xs = lambda k: cx + cw * k / (T - 1)   # noqa: E731
    d.line([(xs(R), cy), (xs(R), cy + ch)], fill=DREAM, width=1)
    for key, col in (("real_h", TRAIL), ("pred_h", DREAM)):
        pts = [(xs(k), cy + ch * (1 - min(1.0, rec[k][key]))) for k in range(t + 1) if np.isfinite(rec[k][key])]
        if len(pts) > 1:
            d.line(pts, fill=col, width=2)
    d.text((cx, cy - 22), "hunger: real %.2f" % r["real_h"], font=font(16), fill=TRAIL)
    d.text((cx + 200, cy - 22), "model's %.2f" % r["pred_h"], font=font(16), fill=DREAM)
    d.text((cx + cw, cy - 22), "dream starts", font=font(14), fill=DREAM, anchor="ra")
    for k in range(t + 1):   # the model's clicks
        if rec[k]["dkind"]:
            d.line([(xs(k), cy + ch + 4), (xs(k), cy + ch + 12)], fill=DREAM if rec[k]["phase"] == "dream" else MUTED)
    return im


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--step", type=int, default=500000)
    ap.add_argument("--real", type=int, default=200)
    ap.add_argument("--dream", type=int, default=200)
    ap.add_argument("--decode", default="pair", choices=["mean", "mode", "pair"])
    ap.add_argument("--world", type=int, default=0)
    ap.add_argument("--title", default="")
    ap.add_argument("-o", "--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    rec = run(args)
    np.savez_compressed(os.path.join(args.out, "dream_%s.npz" % args.decode),
                        pred=np.stack([r["pred"] for r in rec]), real=np.stack([r["real"] for r in rec]))
    its = [r["iters"] for r in rec if r["phase"] == "dream"]
    print("decode iterations per dreamed step: mean %.1f max %d" % (np.mean(its), max(its)))
    title = (args.title or os.path.basename(args.run.rstrip("/"))) + "  (pixels: %s)" % args.decode
    import imageio_ffmpeg
    path = os.path.join(args.out, "dream_%s.mp4" % args.decode)
    gen = imageio_ffmpeg.write_frames(path, (W, H), fps=FPS, codec="libx264", quality=None, bitrate="4M",
                                      macro_block_size=8)
    gen.send(None)
    for t in range(len(rec)):
        gen.send(np.asarray(frame(rec, t, title, args.real)).tobytes())
    gen.close()
    # a sheet: the dream every 20 steps, real above, dreamed below
    ks = list(range(args.real - 1, len(rec), 20))[:11]
    sh = Image.new("RGB", (len(ks) * 138 + 10, 2 * 150 + 40), BG)
    ds = ImageDraw.Draw(sh)
    for i, k in enumerate(ks):
        sh.paste(Image.fromarray(rec[k]["real"]).resize((128, 128), Image.NEAREST), (10 + i * 138, 30))
        sh.paste(Image.fromarray(rec[k]["pred"]).resize((128, 128), Image.NEAREST), (10 + i * 138, 30 + 150))
        ds.text((10 + i * 138, 8), "step %d" % (k + 1), font=font(14), fill=TEXT)
    ds.text((10, 30 + 130), "real (dreamed moves)", font=font(12), fill=MUTED)
    ds.text((10, 30 + 280), "dreamed", font=font(12), fill=DREAM)
    sh.save(os.path.join(args.out, "dream_%s_sheet.png" % args.decode))


if __name__ == "__main__":
    main()

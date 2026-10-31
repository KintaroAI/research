"""Frozen thought runs (task 00064): the run's pieces live on in the world beside their thought field, without learning.
  Food and poison, thought's change per tick. Does thought hold the world: ridge regressions, trained on half the ticks and scored
  on the rest, from thought alone (block-averaged to 16 x 16) to where the food was lag ticks ago, against the same from the
  world half at lag 0 as the ceiling. --shuffle gives each body another body's thought every tick, so that any use of thought
  breaks. --video OUT writes a video of body 0: world, thought, and what changed in thought.

    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 ./venv/bin/python eval_thought.py RUN [RUN ...] --ticks 1500
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
from forage_video import BG, FOOD, MUTED, POISON, TEXT, write_mp4  # noqa: E402
from lifetime import ckpt_name, pieces_move  # noqa: E402
from make_video import font  # noqa: E402
from online import DEFAULT_ONLINE, thought_dream, uni_act  # noqa: E402
from world import make_config, make_world  # noqa: E402

LAGS = (0, 4, 8, 16, 32, 64)


def r2(x, y, train):
    """R^2 on the test ticks of a ridge regression from x [n, f] to y [n, k]."""
    x = np.concatenate([x, np.ones((len(x), 1))], 1)
    a, b, t, u = x[train], y[train], x[~train], y[~train]
    w = np.linalg.solve(a.T @ a + 1.0 * np.eye(a.shape[1]), a.T @ b)
    return float(1.0 - ((u - t @ w) ** 2).sum() / max(((u - b.mean(0)) ** 2).sum(), 1e-12))


def blocks(frames, f=4):
    """[n, H, W] u8 to [n, (H // f) * (W // f)] in 0..1."""
    n, H, W = frames.shape
    return frames.reshape(n, H // f, f, W // f, f).mean((2, 4)).reshape(n, -1) / 255.0


def video(path, rec, title):
    S, W, H = 5, 1280, 480
    frames_n = len(rec)

    def frame(t):
        im = Image.new("RGB", (W, H), BG)
        d = ImageDraw.Draw(im)
        r = rec[t]
        x = [40, 440, 840]
        for i, (a, lab) in enumerate(((r["world"], "world"), (r["thought"], "thought"), (r["delta"], "what changed in thought"))):
            im.paste(Image.fromarray(a).convert("RGB").resize((64 * S, 64 * S), Image.NEAREST), (x[i], 70))
            d.text((x[i], 44), lab, font=font(18, True), fill=TEXT)
        d.text((40, 10), "%s   step %d / %d" % (title, t + 1, frames_n), font=font(18, True), fill=TEXT)
        d.text((40, 400), "food %d" % r["food"], font=font(20, True), fill=FOOD)
        d.text((180, 400), "poison %d" % r["poison"], font=font(20, True), fill=POISON)
        d.text((340, 402), "hunger %.2f" % r["hunger"], font=font(16), fill=MUTED)
        d.text((480, 402), "waves running %d" % r["waves"], font=font(16), fill=MUTED)
        return im
    write_mp4(path, (frame(t) for t in range(frames_n)), (W, H))


def run(path, ticks, bodies, seed, shuffle, vid):
    info = json.load(open(os.path.join(path, "info.json")))
    on = dict(DEFAULT_ONLINE, **info["config"]["online"])
    S = int(info.get("pieces", 1))
    agents = [agent_lib.Agent.load(os.path.join(path, ckpt_name("final", k, S))) for k in range(S)]
    models = [cuda_lib.CudaModel(a) for a in agents]
    L, bins = agents[0].config["ctx"], agents[0].config["bins"]
    env = make_world(bodies, make_config(info["config"]["env"], auto_reset=True), seed=seed)
    H = env.size
    rng = np.random.default_rng(seed)
    obs, hist, es = env.observe(), [], np.ones(bodies, np.uint8)
    food = poison = 0
    th, wo, rel = [], [], []
    rec = []
    for t in range(ticks):
        img = obs["image"]
        if shuffle:   # each body another's thought
            img = np.concatenate([img[:, :, :H], img[rng.permutation(bodies)][:, :, H:]], 2)
        cur = dict(img=img, drv=obs["drives"], act=np.zeros((bodies, 3), np.float32), es=es)
        win = hist + [cur]
        st = lambda k: np.stack([w[k] for w in win], 1)   # noqa: E731
        props = [uni_act(m, st("img"), st("drv"), st("act"), st("es"), bins, rng, on["instinct_m"], on["instinct_tol"],
                         on["instinct_h"], on["instinct_jacobi"], on["instinct_decode"], per_body=True) for m in models]
        a, _ = pieces_move(models, props, on["piece_rule"], rng)
        d = (env.world.target - env.world.cursor + H / 2) % H - H / 2
        th.append(blocks(img[:, :, H:, 0])); wo.append(blocks(img[:, :, :H, 0]))
        rel.append(np.stack([np.sin(2 * np.pi * d[:, 0] / H), np.cos(2 * np.pi * d[:, 0] / H),
                             np.sin(2 * np.pi * d[:, 1] / H), np.cos(2 * np.pi * d[:, 1] / H)], 1))
        cur = dict(cur, act=a)
        hist = (hist + [cur])[-(L - 1):]
        before = env.T.copy()
        obs, _, done, inf = env.step(a)
        dream = None
        if env.cfg.mode == "dream" and S == 2:
            w = hist + [dict(img=obs["image"], drv=obs["drives"], act=np.zeros((bodies, 3), np.float32),
                             es=done.astype(np.uint8))]
            s2 = lambda k: np.stack([x[k] for x in w], 1)   # noqa: E731
            dream = thought_dream(models[-1], s2("img"), s2("drv"), s2("act"), s2("es"), on["thought_jacobi"])
        obs = dict(obs, image=env.think(dream))
        es = done.astype(np.uint8)
        food += int(inf["success"].sum())
        poison += int(inf["gt_hazard_click"].sum())
        if vid:
            dl = np.clip(128 + 1000.0 * (env.T[0] - before[0]), 0, 255).astype(np.uint8)
            rec.append(dict(world=obs["image"][0, :, :H, 0].copy(), thought=obs["image"][0, :, H:, 0].copy(), delta=dl,
                            food=(rec[-1]["food"] if rec else 0) + int(inf["success"][0]),
                            poison=(rec[-1]["poison"] if rec else 0) + int(inf["gt_hazard_click"][0]),
                            hunger=float(inf["gt_drives"][0, 0]), waves=len(env.waves[0])))
    st_ = env.stats()
    n = ticks * bodies / 1000.0
    name = os.path.basename(path)
    print("%-10s %s food %5.1f poison %4.1f  change %.4f quiet %.2f" % (name, "shuffled" if shuffle else "        ", food / n,
                                                                       poison / n, st_["thought_change"], st_["thought_quiet"]),
          flush=True)
    th, wo, rel = np.stack(th), np.stack(wo), np.stack(rel)                      # [t, bodies, ...]
    burn = 100
    out = []
    for lag in LAGS:
        x = th[burn + lag:].reshape(-1, th.shape[-1])
        y = rel[burn:ticks - lag].reshape(-1, 4)
        train = np.repeat(np.arange(ticks - burn - lag) < (ticks - burn - lag) // 2, bodies)
        out.append(r2(x, y, train))
    x = wo[burn:].reshape(-1, wo.shape[-1])
    y = rel[burn:].reshape(-1, 4)
    print("  R2 of the food's place from thought, lag %s: %s;  from the world, lag 0: %.3f" % (
        list(LAGS), " ".join("%.3f" % v for v in out), r2(x, y, np.repeat(np.arange(ticks - burn) < (ticks - burn) // 2, bodies))),
        flush=True)
    for m in models:
        m.close()
    return rec


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--ticks", type=int, default=1500)
    ap.add_argument("--bodies", type=int, default=16)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--shuffle", action="store_true")
    ap.add_argument("--video", default="", help="a video of body 0, this many ticks only")
    a = ap.parse_args()
    for r in a.runs:
        rec = run(r, a.ticks, a.bodies, a.seed, a.shuffle, bool(a.video))
        if a.video:
            video(a.video, rec, os.path.basename(r))


if __name__ == "__main__":
    main()

"""Task 00067: a video of a frozen car: what its camera sees, and the view from above.

    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 MUJOCO_EGL_DEVICE_ID=0 ./venv/bin/python car_video.py RUN -o OUT.mp4 --ticks 1000
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
from online import DEFAULT_ONLINE, uni_act  # noqa: E402
from world import make_config, make_world  # noqa: E402

W, H, EYE, TOP = 880, 500, 384, 416


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("run")
    ap.add_argument("-o", "--out", required=True)
    ap.add_argument("--ticks", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--ckpt", default="final", help="final, or a tick")
    a = ap.parse_args()
    info = json.load(open(os.path.join(a.run, "info.json")))
    on = dict(DEFAULT_ONLINE, **info["config"]["online"])
    S = int(info.get("pieces", 1))
    tick = a.ckpt if a.ckpt == "final" else int(a.ckpt)
    agents = [agent_lib.Agent.load(os.path.join(a.run, ckpt_name(tick, k, S))) for k in range(S)]
    models = [cuda_lib.CudaModel(x) for x in agents]
    L, bins = agents[0].config["ctx"], agents[0].config["bins"]
    env = make_world(1, make_config(info["config"]["env"], auto_reset=True), seed=a.seed)
    rng = np.random.default_rng(a.seed)
    obs, hist, es = env.observe(), [], np.ones(1, np.uint8)
    food = poison = 0
    name = os.path.basename(a.run.rstrip("/"))

    def frames():
        nonlocal obs, hist, es, food, poison
        for t in range(a.ticks):
            cur = dict(img=obs["image"], drv=obs["drives"], act=np.zeros((1, 3), np.float32), es=es)
            win = hist + [cur]
            st = lambda k: np.stack([w[k] for w in win], 1)   # noqa: E731
            props = [uni_act(m, st("img"), st("drv"), st("act"), st("es"), bins, rng, on["instinct_m"], on["instinct_tol"],
                             on["instinct_h"], on["instinct_jacobi"], on["instinct_decode"], per_body=True) for m in models]
            mv, _ = pieces_move(models, props, on["piece_rule"], rng)
            hist = (hist + [dict(cur, act=mv)])[-(L - 1):]
            obs, _, done, inf = env.step(mv)
            es = done.astype(np.uint8)
            food += int(inf["success"][0])
            poison += int(inf["gt_hazard_click"][0])
            im = Image.new("RGB", (W, H), BG)
            d = ImageDraw.Draw(im)
            d.text((24, 10), "%s   tick %d / %d" % (name, t + 1, a.ticks), font=font(18, True), fill=TEXT)
            d.text((24, 42), "what the model sees (64 x 64)", font=font(15), fill=MUTED)
            d.text((440, 42), "from above (not shown to the model)", font=font(15), fill=MUTED)
            im.paste(Image.fromarray(obs["image"][0]).resize((EYE, EYE), Image.NEAREST), (24, 66))
            im.paste(Image.fromarray(env.render(0, "top", TOP)), (440, 66))
            d.text((24, 462), "food %d" % food, font=font(20, True), fill=FOOD)
            d.text((150, 462), "poison %d" % poison, font=font(20, True), fill=POISON)
            d.text((300, 465), "hunger %.2f" % float(inf["gt_drives"][0, 0]), font=font(16), fill=MUTED)
            yield im

    write_mp4(a.out, frames(), (W, H))
    print(a.out, "food %d poison %d in %d ticks" % (food, poison, a.ticks), flush=True)
    env.close()
    for m in models:
        m.close()


if __name__ == "__main__":
    main()

"""Frozen sense runs (task 00081): the run's pieces live on in its world, without learning, twice: with the senses as they lived
them, and with --cut's world, where the other sense goes off too (the control). Each sense's return is scored at its first tick
back, from a window that holds nothing of it, as the context is shorter than the drop:
  words from the picture: at the teacher's return, the letters of its sentence after the first, read from the text piece's
    prediction (within a frame each token sees the tokens before it, so the first letter only says that the text is back). The
    rest of a word follows from its first letter, so `direction` and `range` give the first letters of those two words alone;
  the picture from words: at the eyes' return, the view piece's error per pixel, over the view and over the thing the teacher names.
And food and poison per 1k ticks with the eyes closed, and with them open.

    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 ./venv/bin/python eval_sense.py RUN [RUN ...] --ticks 1500
"""
import argparse
import dataclasses
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import agent_lib  # noqa: E402
import cuda_lib  # noqa: E402
from lifetime import ckpt_name, pieces_move  # noqa: E402
from online import DECODE, DEFAULT_ONLINE, uni_act, uni_frame  # noqa: E402
from world import make_config, make_world  # noqa: E402
from world.sense_world import LEVELS, WORDS  # noqa: E402


def read_letters(frame, size):
    """Nearest letter or blank per cell, from a text half [H, H]: [8, 8] indices into [blank] + LEVELS."""
    b = size // 8
    cell = frame.reshape(8, b, 8, b).mean((1, 3))
    lv = np.concatenate([[0], LEVELS]).astype(np.float64)
    return np.abs(cell[..., None] - lv).argmin(-1)


def patch_slots(agents):
    """The tokens that predict each piece's patches: a piece's own, as the summaries between them fall differently in each."""
    return [agent_lib.uni_layout(a.config)["patch"] - 1 for a in agents]


def live(path, ticks, bodies, seed, cut, rule):
    info = json.load(open(os.path.join(path, "info.json")))
    on = dict(DEFAULT_ONLINE, **info["config"]["online"])
    S = int(info.get("pieces", 1))
    agents = [agent_lib.Agent.load(os.path.join(path, ckpt_name("final", k, S))) for k in range(S)]
    models = [cuda_lib.CudaModel(a) for a in agents]
    c = agents[0].config
    L, bins, H = c["ctx"], c["bins"], c["img"]
    pp = patch_slots(agents)
    env_cfg = make_config(info["config"]["env"], auto_reset=True)
    object.__setattr__(env_cfg, "sense", dataclasses.replace(env_cfg.sense, cut=cut))
    env = make_world(bodies, env_cfg, seed=seed)
    rng = np.random.default_rng(seed)
    obs = env.observe()
    hist, es = [], np.ones(bodies, np.uint8)
    was = dict(eyes=env.eyes > 0, quiet=env.quiet > 0)
    r = dict(blind=0, seen=0, food_blind=0, food_seen=0, poison_blind=0, poison_seen=0, mae=[], mae_thing=[], letters=[], sentences=[], direction=[], range=[])
    yy, xx = np.mgrid[:H, :H] - H / 2.0
    for t in range(ticks):
        cur = dict(img=obs["image"], drv=obs["drives"], act=np.zeros((bodies, 3), np.float32), es=es)
        win = hist + [cur]
        st = lambda k: np.stack([w[k] for w in win], 1)   # noqa: E731
        back_e = np.flatnonzero(was["eyes"] & (env.eyes == 0))
        back_q = np.flatnonzero(was["quiet"] & (env.quiet == 0))
        if len(win) == L:
            for piece, idx in ((0, back_e), (1, back_q)):
                if not len(idx):
                    continue
                v = models[piece].forward_uni_step(st("img")[idx], st("drv")[idx].astype(np.float32), st("act")[idx], st("es")[idx], L - 1,
                                                   mode=DECODE["pair"])["values_t"][:, pp[piece]]
                pred = uni_frame(dict(c, img_w=0), v)[..., 0].astype(np.float64)
                real = st("img")[idx, -1, :, H * piece:H * (piece + 1), 0].astype(np.float64)
                if piece == 0:
                    place, rad = env.nearest_xy()
                    for i, b in enumerate(idx):
                        dx = (xx - place[b, 0] + H / 2) % H - H / 2
                        dy = (yy - place[b, 1] + H / 2) % H - H / 2
                        m = np.hypot(dx, dy) <= rad[b]
                        r["mae"].append(np.abs(pred[i] - real[i]).mean() / 255.0)
                        r["mae_thing"].append(np.abs(pred[i] - real[i])[m].mean() / 255.0 if m.any() else np.nan)
                else:
                    for i, b in enumerate(idx):
                        want = read_letters(real[i], H)[:len(WORDS)]
                        got = read_letters(pred[i], H)[:len(WORDS)]
                        mask = want > 0
                        mask[0, 0] = False
                        r["letters"].append((got[mask] == want[mask]).mean())
                        r["sentences"].append(bool((got[mask] == want[mask]).all()))
                        r["direction"].append(bool(got[1, 0] == want[1, 0]))
                        r["range"].append(bool(got[2, 0] == want[2, 0]))
        props = [uni_act(m, st("img"), st("drv"), st("act"), st("es"), bins, rng, on["instinct_m"], on["instinct_tol"],
                         on["instinct_h"], on["instinct_jacobi"], on["instinct_decode"], per_body=True, first="sample", ring=False)
                 for m in models]
        a, _ = pieces_move(models, props, rule, rng, prev=None)
        cur = dict(cur, act=a)
        hist = (hist + [cur])[-(L - 1):]
        blind = env.eyes > 0
        was = dict(eyes=env.eyes > 0, quiet=env.quiet > 0)
        obs, _, done, inf = env.step(a)
        bad = inf["gt_hazard_click"]
        r["blind"] += int(blind.sum())
        r["seen"] += int((~blind).sum())
        r["food_blind"] += int((inf["success"] & blind).sum())
        r["food_seen"] += int((inf["success"] & ~blind).sum())
        r["poison_blind"] += int((bad & blind).sum())
        r["poison_seen"] += int((bad & ~blind).sum())
        es = done.astype(np.uint8)
    for m in models:
        m.close()
    return r


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--ticks", type=int, default=1500)
    ap.add_argument("--bodies", type=int, default=16)
    ap.add_argument("--rule", default="gain")
    ap.add_argument("--seed", type=int, default=1234)
    a = ap.parse_args()
    for path in a.runs:
        for cut in (False, True):
            r = live(path, a.ticks, a.bodies, a.seed, cut, a.rule)
            row = dict(run=os.path.basename(path), cut=cut, mae=float(np.nanmean(r["mae"])) if r["mae"] else None,
                       mae_thing=float(np.nanmean(r["mae_thing"])) if r["mae_thing"] else None, n_view=len(r["mae"]),
                       letters=float(np.mean(r["letters"])) if r["letters"] else None,
                       sentences=float(np.mean(r["sentences"])) if r["sentences"] else None, n_text=len(r["letters"]),
                       direction=float(np.mean(r["direction"])) if r["direction"] else None,
                       range=float(np.mean(r["range"])) if r["range"] else None,
                       food_blind=1000.0 * r["food_blind"] / max(r["blind"], 1), poison_blind=1000.0 * r["poison_blind"] / max(r["blind"], 1),
                       food_seen=1000.0 * r["food_seen"] / max(r["seen"], 1), poison_seen=1000.0 * r["poison_seen"] / max(r["seen"], 1))
            print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()

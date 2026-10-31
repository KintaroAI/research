"""Frozen lifetimes of one-body runs (task 00059): the checkpoints of a run (one model, or the pieces of online.pieces)
act for --ticks ticks in --bodies copies of its world, without learning. Prints food and poison per 1k ticks; the size
of the moves made; how well a move points at the nearest food (cosine, ego view: the cursor at the centre); with
pieces, each piece's share of the wins by the quadrant the food is in, and with --rule random the same pieces with
the winner drawn at random, so that the selection's own part shows; how smooth the path is (the cosine of successive
moves), and on what share of ticks the pieces' predicted hungers differ by more than 0.005. --rule council and veto score every piece's move
with every piece and make the one with the lowest mean (council) or lowest worst (veto) predicted next hunger. --touch
takes every poison touch apart: the hunger before it, the ticks since the last food against all ticks, food on the same
tick, whether the piece that made the move foresaw the rise, whether another piece did, and whether another piece's
move would have missed the poison.

Ground truth (gt_*, the world's cursor and circles) is for scoring only and never reaches the model.

    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 ./venv/bin/python eval_pieces.py RUN [RUN ...] --rule gain
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import agent_lib  # noqa: E402
import cuda_lib  # noqa: E402
from lifetime import cosine, ckpt_name, pieces_move  # noqa: E402
from online import DEFAULT_ONLINE, uni_act  # noqa: E402
from world import make_config, make_world  # noqa: E402
from world.circle_world import _inside, wrap_delta  # noqa: E402


def relative(env, xy):
    """Where the points xy [N, 2] are from the cursor on the torus: (dx, dy) in [-size / 2, size / 2)."""
    size = env.config.size
    return (xy - env.cursor + size / 2) % size - size / 2


def touches_poison(env, moves):
    """[N] bool: the move [N, 3] would end inside a hazard (as CircleWorld.step: the cursor moves first, the hazards
    stay where they were drawn). For scoring only."""
    new = np.mod(env.cursor + np.clip(moves[:, :2], -1, 1) * env.config.max_step_px, float(env.config.size))
    size = float(env.config.size)
    dx = wrap_delta(new[:, None, 0] - env.hz_pos[..., 0], 0.0, size)
    dy = wrap_delta(new[:, None, 1] - env.hz_pos[..., 1], 0.0, size)
    return _inside(dx, dy, env.hz_rad, env.config.hazard_shape == "square").any(1)


BINS = (1, 3, 6, 11, 21, 10 ** 9)                        # ticks since food: 1-2, 3-5, 6-10, 11-20, 21+
FORESEE = 0.1                                            # a rise this large in the predicted next hunger


def report_touches(t, S):
    n = max(len(t["h0"]), 1)
    h0 = np.array(t["h0"])
    print("  %d poison touches (%s). Hunger before: under 0.1 %.2f, 0.1-0.5 %.2f, over 0.5 %.2f; food on the same tick "
          "%.2f" % (len(h0), "per piece" if S > 1 else "one model", (h0 < 0.1).mean(), ((h0 >= 0.1) & (h0 <= 0.5)).mean(),
                    (h0 > 0.5).mean(), np.mean(t["same"])))
    print("  the mover foresaw it %.2f; only another piece did %.2f; nobody %.2f; another piece's move missed it %.2f" % (
        np.mean(t["seen"]), np.mean(t["other"]), 1 - np.mean(t["seen"]) - np.mean(t["other"]), np.mean(t["miss"])))
    b = np.digitize(np.array(t["since"]), BINS) - 1
    a = np.digitize(np.array(t["all_since"]), BINS) - 1
    print("  ticks since food   1-2  3-5 6-10 11-20  21+")
    print("  share of touches  " + " ".join("%4.2f" % ((b == i).sum() / n) for i in range(5)))
    print("  share of ticks    " + " ".join("%4.2f" % ((a == i).mean()) for i in range(5)))


def run(path, ticks, bodies, rule, seed, touch=False):
    info = json.load(open(os.path.join(path, "info.json")))
    on = dict(DEFAULT_ONLINE, **info["config"]["online"])
    S = int(info.get("pieces", 1))
    agents = [agent_lib.Agent.load(os.path.join(path, ckpt_name("final", k, S))) for k in range(S)]
    models = [cuda_lib.CudaModel(a) for a in agents]
    L, bins = agents[0].config["ctx"], agents[0].config["bins"]
    env = make_world(bodies, make_config(info["config"]["env"], auto_reset=True), seed=seed)
    rng = np.random.default_rng(seed)
    use = rule if S > 1 else "mean"
    obs = env.observe()
    hist, es = [], np.ones(bodies, np.uint8)
    food = poison = 0
    size, cos_near, cos_all, smooth = [], [], [], []
    split = 0
    won = np.zeros((4, max(S, 4)))
    near = 16.0
    since = np.full(bodies, 10 ** 6)                         # ticks since each body's last food
    t_ = dict(h0=[], same=[], seen=[], other=[], miss=[], since=[], all_since=[])
    cross = touch
    for t in range(ticks):
        cur = dict(img=obs["image"], drv=obs["drives"], act=np.zeros((bodies, 3), np.float32), es=es)
        win = hist + [cur]
        st = lambda k: np.stack([w[k] for w in win], 1)   # noqa: E731
        props = [uni_act(m, st("img"), st("drv"), st("act"), st("es"), bins, rng, on["instinct_m"], on["instinct_tol"],
                         on["instinct_h"], on["instinct_jacobi"], on["instinct_decode"], per_body=True)
                 for m in models]
        hunger = np.stack([p[3] for p in props])              # [S, N]
        split += (hunger.max(0) - hunger.min(0) > 0.005).sum()
        cand = np.stack([p[0] for p in props], 1)             # [N, S, 3]
        if cross:   # right after each piece's pass: its score of every piece's move, [scorer, N, mover]
            hc = np.stack([m.uni_candidates(cand) for m in models])
            np.testing.assert_allclose(hc[np.arange(S), :, np.arange(S)].T, hunger.T, atol=1e-4)   # its own score again
        a, w = pieces_move(models, props, use, rng)
        if t:
            smooth.append(cosine(a[:, :2].astype(np.float64), last[:, :2].astype(np.float64)).mean())
        last = a
        h_before = obs["drives"][:, 0].copy()
        missed = np.stack([touches_poison(env, cand[:, k]) for k in range(S)], 1) if touch else None
        d = relative(env, env.target)
        dist = np.linalg.norm(d, axis=1)
        q = (d[:, 1] >= 0) * 2 + (d[:, 0] >= 0)                 # the food's quadrant in the ego view, as band_k
        c = cosine(a[:, :2].astype(np.float64), d)
        size.append(np.abs(a[:, :2]).mean())
        cos_all.append(c.mean())
        if (dist <= near).any():
            cos_near.append(c[dist <= near].mean())
        if S > 1 and use != "mean":
            np.add.at(won, (q[dist <= near], w[dist <= near, 0]), 1)
        cur = dict(cur, act=a)
        hist = (hist + [cur])[-(L - 1):]
        obs, _, done, inf = env.step(a)
        food += int(inf["success"].sum())
        poison += int(inf["gt_hazard_click"].sum()) if "gt_hazard_click" in inf else 0
        since += 1
        if touch:
            t_["all_since"] += since.tolist()
            for b in np.flatnonzero(inf["gt_hazard_click"]):
                mv = max(int(w[b, 0]), 0)
                rise = hc[:, b, mv] - h_before[b] > FORESEE   # who predicted the rise, for the move made [scorer]
                t_["h0"].append(float(h_before[b]))
                t_["same"].append(bool(inf["success"][b]))
                t_["seen"].append(bool(rise[mv]))
                t_["other"].append(bool(rise.any() and not rise[mv]))
                t_["miss"].append(bool((~missed[b]).any()))
                t_["since"].append(int(since[b]))
        since[inf["success"]] = 0
        es = done.astype(np.uint8)
    n = ticks * bodies / 1000.0
    print("%-14s %-6s food %5.1f poison %4.1f share %.2f  move size %.3f  smooth %.3f  gain differs %.2f  cos to food "
          "%.3f (within %d px %.3f)" % (os.path.basename(path), use, food / n, poison / n, food / max(food + poison, 1),
                                        np.mean(size), np.mean(smooth), split / (ticks * bodies), np.mean(cos_all), near,
                                        np.mean(cos_near)), flush=True)
    if S > 1 and use != "mean":
        print("  wins by food quadrant, food within %d px (rows TL TR BL BR, columns piece):" % near)
        print(np.array2string(won[:, :S] / np.maximum(won[:, :S].sum(1, keepdims=True), 1), precision=2, floatmode="fixed"))
    if touch:
        report_touches(t_, S)
    for m in models:
        m.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--ticks", type=int, default=1500)
    ap.add_argument("--bodies", type=int, default=16)
    ap.add_argument("--rule", default="gain", help="pieces: mean, max, gain, random, council, veto, sharp or gain_sharp")
    ap.add_argument("--touch", action="store_true", help="take every poison touch apart")
    ap.add_argument("--seed", type=int, default=1234)
    a = ap.parse_args()
    for r in a.runs:
        run(r, a.ticks, a.bodies, a.rule, a.seed, a.touch)


if __name__ == "__main__":
    main()

"""Task 00125: does a DreamerV3 world model keep the poison's look? In the food world food and poison have the same grey
(94 at luma) and differ only by the checker (2 px cells); a world model that blurs the checker can't tell them apart.

For each run (its dreamer/latest.pt), a new life of random moves (seed --world-seed) is observed through the world model
(encoder, posterior, decoder); each object in the real frame is found from its grey (94) and classed by how much of its
area is lit (food about 1, poison about .5). Inside each object: the real and the decoded pixels' spread (std) and their
correlation. A kept checker: the decoded poison's spread near the real one and a high correlation.

    ~/venvs/dreamer/bin/python baselines/dreamer_recon.py RUN_DIR [RUN_DIR ...] --config configs/U_agent.json
        [--dset key=value per run, as RUN_DIR:key=value] [--ticks 400] [--sheet out.png]
"""
import argparse
import os
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import run_dreamer  # noqa: E402

GREY = 94


def load_agent(run_dir, config, dset, dreamer_dir):
    ns = run_dreamer.dreamer_config(dreamer_dir, tempfile.mkdtemp(), 1, 60000, 512, dset)
    import dreamer
    import tools
    import torch
    env = run_dreamer.DreamerVM(config, 1, [], None)
    ns.num_actions = env.action_space.shape[0]
    agent = dreamer.Dreamer(env.observation_space, env.action_space, ns, tools.Logger(ns.logdir, 0), None).to(ns.device)
    ck = torch.load(os.path.join(run_dir, "dreamer", "latest.pt"), map_location=ns.device)
    agent.load_state_dict(ck["agent_state_dict"])
    agent.eval()
    return agent


def life(config, seed, ticks):
    """A new life of uniform random moves: images (T, 64, 64, 1) uint8, drives, prev, actions (action[t] led to obs t)."""
    env = run_dreamer.DreamerVM(config, seed, [], None)
    rng = np.random.default_rng(seed)
    o = env.reset()
    obs = [o]
    acts = [np.zeros(2, np.float32)]
    for _ in range(ticks - 1):
        a = rng.uniform(-1, 1, 2).astype(np.float32)
        o, _, _, _ = env.step(a)
        obs.append(o)
        acts.append(a)
    return {k: np.stack([x[k] for x in obs]) for k in ("image", "drives", "prev")}, np.stack(acts)


def decode(agent, data, acts):
    import torch
    wm = agent._wm
    T = len(acts)
    d = dict(image=data["image"][None], drives=data["drives"][None], prev=data["prev"][None], action=acts[None],
             is_first=np.eye(1, T, dtype=np.float32), is_terminal=np.zeros((1, T), np.float32))
    with torch.no_grad():
        p = wm.preprocess(d)
        embed = wm.encoder(p)
        states, _ = wm.dynamics.observe(embed, p["action"], p["is_first"])
        img = wm.heads["decoder"](wm.dynamics.get_feat(states))["image"].mode()
    return np.clip(img[0, ..., 0].float().cpu().numpy() * 255.0, 0, 255)


def objects(frame):
    """Objects of the real frame (64, 64) uint8: [(mask, lit share)], each a connected blob of the grey after one dilation
    (which joins the checker's cells), eroded once so that the edge doesn't count."""
    lit = np.abs(frame.astype(int) - GREY) <= 1
    dil = lit.copy()
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            dil |= np.roll(np.roll(lit, dy, 0), dx, 1)
    seen = np.zeros_like(dil)
    out = []
    H, W = dil.shape
    for y0, x0 in zip(*np.nonzero(dil)):
        if seen[y0, x0]:
            continue
        comp = np.zeros_like(dil)
        stack = [(y0, x0)]
        seen[y0, x0] = True
        while stack:
            y, x = stack.pop()
            comp[y, x] = True
            for ny, nx in ((y + 1, x), (y - 1, x), (y, x + 1), (y, x - 1)):
                if 0 <= ny < H and 0 <= nx < W and dil[ny, nx] and not seen[ny, nx]:
                    seen[ny, nx] = True
                    stack.append((ny, nx))
        er = comp.copy()
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            er &= np.roll(np.roll(comp, dy, 0), dx, 1)
        if er.sum() >= 40:
            out.append((er, float(lit[er].mean())))
    return out


def score(real, rec):
    rows = {"food": [], "poison": []}
    for t in range(len(real)):
        R, D = real[t].astype(np.float64), rec[t]
        for m, share in objects(real[t]):
            kind = "food" if share > 0.85 else ("poison" if 0.35 < share < 0.65 else None)
            if kind is None:
                continue
            r, d = R[m], D[m]
            c = float(np.corrcoef(r, d)[0, 1]) if r.std() > 0 and d.std() > 0 else np.nan
            rows[kind].append((r.std(), d.std(), r.mean(), d.mean(), c, np.abs(r - d).mean()))
    return {k: np.array(v) for k, v in rows.items()}


def sheet(real, recs, labels, path, n=6):
    from PIL import Image, ImageDraw
    pick = [t for t in range(len(real)) if any(0.35 < s < 0.65 for _, s in objects(real[t]))]
    pick = pick[:: max(1, len(pick) // n)][:n]
    S, g = 3 * 64, 4
    im = Image.new("L", (len(pick) * (S + g), (1 + len(recs)) * (S + 14)), 255)
    dr = ImageDraw.Draw(im)
    for j, t in enumerate(pick):
        for i, fr in enumerate([real[t].astype(np.float64)] + [r[t] for r in recs]):
            tile = Image.fromarray(np.clip(fr, 0, 255).astype(np.uint8)).resize((S, S), Image.NEAREST)
            im.paste(tile, (j * (S + g), i * (S + 14) + 14))
            if j == 0:
                dr.text((2, i * (S + 14)), (["real"] + labels)[i], fill=0)
    im.save(path)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("runs", nargs="+", help="RUN_DIR or RUN_DIR:key=value,key=value (the run's --dset)")
    ap.add_argument("--config", required=True)
    ap.add_argument("--world-seed", type=int, default=101)
    ap.add_argument("--ticks", type=int, default=400)
    ap.add_argument("--sheet")
    ap.add_argument("--dreamer", default=os.path.expanduser("~/src/dreamerv3-torch"))
    a = ap.parse_args()
    config = os.path.abspath(a.config)
    data, acts = life(config, a.world_seed, a.ticks)
    real = data["image"][..., 0]
    recs, labels = [], []
    print("run | kind n | real std  decoded std | real mean  decoded mean | corr | mean abs error")
    for spec in a.runs:
        run, _, ds = spec.partition(":")
        agent = load_agent(run, config, [x for x in ds.split(",") if x], a.dreamer)
        rec = decode(agent, data, acts)
        recs.append(rec)
        labels.append(os.path.basename(run.rstrip("/")))
        sc = score(real, rec)
        for k in ("food", "poison"):
            v = sc[k]
            if len(v):
                print("%s | %s %d | %.1f  %.1f | %.1f  %.1f | %.2f | %.1f" % (labels[-1], k, len(v), v[:, 0].mean(), v[:, 1].mean(),
                      v[:, 2].mean(), v[:, 3].mean(), np.nanmean(v[:, 4]), v[:, 5].mean()), flush=True)
        del agent
    if a.sheet:
        sheet(real, recs, labels, a.sheet)


if __name__ == "__main__":
    main()

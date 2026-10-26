"""The universal model trained offline on a text world (task 00034): the classical reference's data for the unchanged
model. Every update: --batch windows of ctx consecutive pages, each from a random letter offset of the train split, its
own context (the first step starts it); moves and drives 0. The GPU update is the lifetime's (lifetime.py), so the
only difference from a text lifetime is where the windows come from. Checkpoints load in eval_predict.py.

    ./venv/bin/python train_text_offline.py --config configs/T_text_B.json --updates 100000 --seed 1 -o OUT_DIR
"""

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import agent_lib  # noqa: E402
import cuda_lib  # noqa: E402
from online import fill_rollout, load_config  # noqa: E402
from train import git_hash  # noqa: E402
from world import make_config, make_world  # noqa: E402


def windows(world, offsets, L):
    """Rows (online.FIELDS) of windows of L pages starting at the given letter offsets: [B, L, ...]."""
    B, page = len(offsets), world.config.page
    pages = np.stack([world.render(world.page_codes(offsets + k * page)) for k in range(L + 1)], 1)   # [B, L+1, ...]
    z = lambda *s: np.zeros((B, L) + s, np.float32)   # noqa: E731
    es = np.zeros((B, L), np.uint8)
    es[:, 0] = 1
    return dict(img=pages[:, :L], nimg=pages[:, 1:], drv=z(2), ndrv=z(2), pa=z(3), pr=z(), es=es, act=z(3), rew=z(),
                done=np.zeros((B, L), np.uint8), hit=np.zeros((B, L), np.uint8), cmd=z())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", action="append", default=[])
    ap.add_argument("--updates", type=int, default=100000)
    ap.add_argument("--save-every", type=int, default=10000)
    ap.add_argument("--log-every", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("-o", "--out", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config, args.set)
    on = cfg["online"]
    world = make_world(1, make_config(dict(cfg["env"], split="train")), seed=args.seed)
    agent = agent_lib.Agent(cfg["model"], seed=args.seed)
    L, B = agent.config["ctx"], on["batch"]
    ppo = dict(agent_lib.DEFAULT_PPO)
    ppo.update(dict(use_ppo=0, epochs=1, minibatches=1))
    ppo.update(cfg["ppo"])
    gpu = cuda_lib.CudaModel(agent)
    ro = cuda_lib.Rollout(L, B, agent.config["img"], next_frames=not cuda_lib.has("vmc_uni_candidates"))
    rng = np.random.default_rng([args.seed, 34])
    os.makedirs(args.out, exist_ok=True)
    info = dict(command=" ".join(sys.argv), git=git_hash(), config=cfg, seed=args.seed, out_dir=args.out,
                model_config=agent.config, num_params=agent.num_params, ppo=ppo, script="train_text_offline.py",
                start_time=time.strftime("%Y-%m-%d %H:%M:%S"))
    json.dump(info, open(os.path.join(args.out, "info.json"), "w"), indent=2)
    print("params %d  batch %d windows of %d pages" % (agent.num_params, B, L), flush=True)
    mf = open(os.path.join(args.out, "metrics.jsonl"), "w")
    stats, t0 = [], time.time()
    for u in range(1, args.updates + 1):
        fill_rollout(ro, windows(world, rng.integers(0, len(world.codes), B), L), L)
        st = gpu.update_rollout(ro, ppo, agent, sync=False, stats=u % 10 == 0)
        if st is not None:
            stats.append(st)
        if u % args.save_every == 0:
            gpu.sync_to(agent)
            agent.save(os.path.join(args.out, "ckpt_%08d.bin" % u), with_optimizer=False)
        if u % args.log_every == 0:
            row = dict(step=u, time=time.time() - t0)
            row.update({k: float(np.mean([s[k] for s in stats])) for k in ("uni_ce_patch", "uni_ce_move", "uni_ce_body")
                        if stats and k in stats[0]})
            mf.write(json.dumps(row) + "\n")
            mf.flush()
            print("update %7d  patch ce %.4f  %.0f s" % (u, row.get("uni_ce_patch", float("nan")), row["time"]),
                  flush=True)
            stats = []
    gpu.sync_to(agent)
    agent.save(os.path.join(args.out, "ckpt_final.bin"), with_optimizer=False)


if __name__ == "__main__":
    main()

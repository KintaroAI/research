"""One lifetime of the target architecture (../ARCHITECTURE.md, task 00020): the self actor of online.py, learning
on every tick, with parallel bodies early in life. No reward anywhere.

Every tick:
  1. Act. Each active body b = 0..N(tick)-1 reads its own last ctx real states (frames, drives, moves, command) and
     makes the move the model predicts (sampled); with probability eps it babbles instead (dx, dy ~ N(0, babble_sigma),
     a click with probability babble_click). One batched forward pass for all bodies.
  2. Each body's world steps; the transition goes into that body's short-term buffer and, chunk by chunk, through its
     gate into the long-term memory that all bodies share (online.py's GatedMemory and ChunkStore).
  3. Learn: one gradient step on the newest window of every active body plus batch - 1 windows replayed from
     long-term memory.

The same weights drive every body; bodies have no identity input. N(tick) is online.bodies for the first
online.bodies_hold ticks, then halves every online.bodies_halve ticks down to 1; body 0 lives on. With online.bodies=1
this is online.py's self actor with gated memory (the same checkpoints, bit for bit, without surprise gating).

Command (online.goal=1): online.goal_mode = "relief" (vm-00006: its hunger if a hit follows within goal_k steps, else
0) or "delta" (the change of hunger over the next goal_k steps: relief positive, hazards and time negative). When
acting, the command is the current hunger: "relieve all of it".

Exploration: eps falls linearly from online.explore_eps to online.explore_eps_end over online.explore_decay ticks
(explore_eps_end unset: constant).

Outputs (as online.py, so that `online.py --eval RUN` and the analysis scripts read them): info.json, metrics.jsonl
(body 0's hits and hazard shares under online.py's names; all_* over the active bodies), ckpt_NNNNNNNN.bin.

usage:
  ./venv/bin/python lifetime.py --config configs/AHR_gated.json --set model.patch=8 --set model.act_token=1 \\
      --set ppo.act_coef=0.5 --set online.goal_mode=delta --set online.bodies=32 --set online.bodies_hold=10000 \\
      --set online.bodies_halve=5000 --set online.device=cuda --seed 1 -o OUT
"""
import argparse
import dataclasses
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import agent_lib  # noqa: E402
from evaluate import executed_action  # noqa: E402
from online import FIELDS, STAT_KEYS, GatedMemory, fill_rollout, load_config, sample_moves  # noqa: E402
from output_name import next_output  # noqa: E402
from train import git_hash  # noqa: E402
from world import CircleWorld, EnvConfig  # noqa: E402

LIFE_STATS = STAT_KEYS + ("act_drive_mse", "act_frame_mse")


def bodies_at(on, tick):
    """Active bodies at tick (counted from 1)."""
    n0, hold, halve = int(on["bodies"]), int(on["bodies_hold"]), int(on["bodies_halve"])
    if n0 <= 1 or tick <= hold or halve <= 0:
        return max(1, n0)
    return max(1, n0 >> ((tick - hold - 1) // halve + 1))


def eps_at(on, tick):
    e0, e1, k = on["explore_eps"], on["explore_eps_end"], on["explore_decay"]
    if e1 is None or k <= 0:
        return e0
    return e1 if tick - 1 >= k else e0 + (e1 - e0) * (tick - 1) / k


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", action="append", default=[], help="override, e.g. online.bodies=32")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--exp", type=int, default=16)
    ap.add_argument("-o", "--out", default=None)
    ap.add_argument("--init", default=None, help="start from this checkpoint instead of a random model")
    args = ap.parse_args()

    cfg = load_config(args.config, args.set)
    on = cfg["online"]
    if on["actor"] != "self" or on["memory"] != "gated":
        raise ValueError("lifetime.py runs the self actor with gated memory (online.actor=self, online.memory=gated)")
    if on["store_surprise"]:
        raise ValueError("lifetime.py does not measure surprise: online.store_surprise must be 0")
    name = cfg.get("name", os.path.splitext(os.path.basename(args.config))[0])
    out_dir = args.out or next_output(args.exp, "%s_s%d" % (name, args.seed))
    os.makedirs(out_dir, exist_ok=True)
    agent_lib.set_threads(on["threads"])

    N0 = max(1, int(on["bodies"]))
    env_cfg = dataclasses.replace(EnvConfig(**cfg["env"]), auto_reset=True)
    env = CircleWorld(N0, env_cfg, seed=args.seed)
    agent = agent_lib.Agent.load(args.init) if args.init else agent_lib.Agent(cfg["model"], seed=args.seed)
    if on["click_bias"] is not None and not args.init:   # initial click-logit bias, e.g. -2.197 = click p 0.1
        off = {name: o for name, o, _ in agent.param_layout()}["pi_b"]
        agent.params()[off + 2] = on["click_bias"]
        agent.params_changed()
    goal_k = on["goal_k"] if on["goal"] else 0
    if goal_k and not agent.config["use_prev_reward"]:
        raise ValueError("online.goal=1 needs model.use_prev_reward=1 (the scalar token slot carries the goal)")
    ppo = dict(agent_lib.DEFAULT_PPO)
    ppo.update(dict(use_ppo=0, epochs=1, minibatches=1))
    ppo.update(cfg["ppo"])
    if ppo["use_ppo"] or not env_cfg.drives:
        raise ValueError("lifetime.py is reward-free: needs ppo.use_ppo=0 and env.drives=true")
    L, B = agent.config["ctx"], on["batch"]
    lag = goal_k - 1 if goal_k else 0         # the newest window whose hindsight labels are complete
    rng = np.random.default_rng(args.seed)
    gms = [GatedMemory(on, L, env_cfg.size, goal_k, args.seed)]
    gms += [GatedMemory(on, L, env_cfg.size, goal_k, args.seed, store=gms[0].store, rng_key=(b,)) for b in range(1, N0)]
    store = gms[0].store
    gpu, fwd = None, agent.forward
    if on["device"] == "cuda":
        import cuda_lib
        gpu = cuda_lib.CudaModel(agent)
        fwd = gpu.forward
    elif on["device"] != "cpu":
        raise ValueError("online.device must be 'cpu' or 'cuda'")
    rollouts = {}

    def rollout(n):   # one per batch size (the batch grows with the active bodies)
        if n not in rollouts:
            rollouts[n] = cuda_lib.Rollout(L, n, agent.config["img"]) if gpu else agent.new_rollout(L, n)
        return rollouts[n]

    info = dict(command=" ".join(sys.argv), git=git_hash(), config=cfg, seed=args.seed, out_dir=out_dir,
                model_config=agent.config, num_params=agent.num_params, threads=agent_lib.get_threads(),
                env_config=dataclasses.asdict(env_cfg), ppo=ppo, start_time=time.strftime("%Y-%m-%d %H:%M:%S"),
                script="lifetime.py")
    with open(os.path.join(out_dir, "info.json"), "w") as f:
        json.dump(info, f, indent=2)
    print("out: %s\nparams: %d  threads: %d  batch: bodies + %d replayed x %d steps  replay: %d  bodies: %d" % (
        out_dir, agent.num_params, agent_lib.get_threads(), B - 1, L, on["replay"], N0), flush=True)
    agent.save(os.path.join(out_dir, "ckpt_%08d.bin" % 0), with_optimizer=False)

    mf = open(os.path.join(out_dir, "metrics.jsonl"), "w")
    obs = env.observe()
    prev_a = np.zeros((N0, 3), np.float32)
    prev_r = np.zeros(N0, np.float32)
    ep_start = np.ones(N0, np.uint8)

    def new_window():
        return dict(eps=[], stats=[], step_hits=0, hazard=0, hazard_felt=0, capped=0, all_hits=0, all_hazard=0,
                    body_steps=0, babbled=0, poison=0, all_poison=0, t=time.time())
    win = new_window()
    t0 = time.time()
    for tick in range(1, on["steps"] + 1):
        N = bodies_at(on, tick)
        h = obs["drives"][:N, 0]
        cmd = (h if goal_k else np.zeros(N)).astype(np.float32)
        # 1. act: each body's move, predicted from its own last L real states (this one included)
        k = min(gms[0].stm.n, L - 1)
        idx = [gm.stm.last(k) for gm in gms[:N]]
        win_in = [np.stack([np.concatenate([getattr(gm.stm, f)[ix], cur]) for gm, ix, cur in
                            zip(gms[:N], idx, cs)]) for f, cs in
                  (("img", obs["image"][:N, None]), ("drv", obs["drives"][:N, None]), ("pa", prev_a[:N, None]),
                   ("cmd", cmd[:, None]), ("es", ep_start[:N, None]))]
        out = fwd(win_in[0], win_in[1], win_in[2], win_in[3].astype(np.float32), win_in[4].astype(np.uint8))
        a = sample_moves(agent, out, rng)
        bab = rng.random(N) < eps_at(on, tick)
        if bab.any():   # keeps trying: own moves alone drift to never clicking
            nb = np.concatenate([rng.normal(0, on["babble_sigma"], (N, 2)), rng.random((N, 1)) < on["babble_click"]], 1)
            a = np.where(bab[:, None], nb, a).astype(np.float32)
        # 2. the worlds step; each body remembers its transition
        full = np.zeros((N0, 3), np.float32)
        full[:N] = a
        img, drv = obs["image"][:N].copy(), obs["drives"][:N].copy()
        obs, rew, done, inf = env.step(full)
        for b in range(N):
            gms[b].stm.add(img=img[b], nimg=inf["terminal_image"][b], drv=drv[b], ndrv=inf["terminal_drives"][b],
                           pa=prev_a[b], pr=prev_r[b], es=ep_start[b], act=a[b], rew=rew[b], done=done[b],
                           hit=inf["success"][b], cmd=cmd[b])
        hz = inf["gt_in_hazard"][:N] if "gt_in_hazard" in inf else np.zeros(N, bool)     # scoring only
        win["step_hits"] += int(inf["success"][0])
        win["hazard"] += int(hz[0])
        win["hazard_felt"] += int(hz[0] and inf["terminal_drives"][0, 0] - drv[0, 0] > 1.5 * env_cfg.h_rate)
        win["capped"] += int(drv[0, 0] >= 1.0 - 1e-6)
        win["all_hits"] += int(inf["success"][:N].sum())
        if "gt_hazard_click" in inf:    # poison clicks (scoring only)
            win["poison"] += int(inf["gt_hazard_click"][0])
            win["all_poison"] += int(inf["gt_hazard_click"][:N].sum())
        win["all_hazard"] += int(hz.sum())
        win["body_steps"] += N
        win["babbled"] += int(bab.sum())
        if done[0]:
            win["eps"].append((bool(inf["ep_success"][0]), float(inf["ep_length"][0]), float(inf["ep_clicks"][0])))
        prev_a[:N] = executed_action(a)
        prev_r[:N] = (rew[:N] * ppo["reward_scale"]).astype(np.float32)   # stored for completeness; the model zeroes it
        prev_a[:N][done[:N].astype(bool)] = 0.0
        prev_r[:N][done[:N].astype(bool)] = 0.0
        ep_start[:N] = done[:N]
        for b in range(N):
            gms[b].decide(tick)
        # 3. learn from the newest window of every active body plus replayed ones
        if gms[0].stm.n >= L + lag:
            if N == 1:
                rows = gms[0].rows(B, rng, lag)
            else:
                parts = [gm.newest(lag) for gm in gms[:N]]
                if store.n:
                    parts.append(gms[0].replayed(B - 1, rng))
                else:
                    parts.append(gms[0]._from_stm(gms[0].stm.starts(L, B, rng, lag)[1:], lag))
                rows = {f: np.concatenate([p[f] for p in parts]) for f in FIELDS}
            ro = rollout(len(rows["img"]))
            fill_rollout(ro, rows, L)
            zeros = np.zeros(len(rows["img"]), np.float32)
            win["stats"].append(gpu.update_rollout(ro, ppo, agent, sync=False) if gpu else agent.update(ro, zeros, ppo))
        if tick % on["save_every"] == 0:
            if gpu:
                gpu.sync_to(agent)
            agent.save(os.path.join(out_dir, "ckpt_%08d.bin" % tick), with_optimizer=False)
        if tick % on["log_every"] == 0:
            eps, le, bs = win["eps"], on["log_every"], max(1, win["body_steps"])
            row = dict(step=tick, time=time.time() - t0, sps=le / max(1e-9, time.time() - win["t"]), bodies=N,
                       episodes=len(eps), hits=sum(e[0] for e in eps),
                       ep_len=float(np.mean([e[1] for e in eps])) if eps else None,
                       step_hits=win["step_hits"], hazard_frac=win["hazard"] / le,
                       hazard_felt=win["hazard_felt"] / le, hunger_capped=win["capped"] / le, hunger=float(h[0]),
                       all_hits_per_1k=1000.0 * win["all_hits"] / bs, all_hazard_frac=win["all_hazard"] / bs,
                       babbled=win["babbled"] / bs, eps=eps_at(on, tick), poison=win["poison"],
                       all_poison_per_1k=1000.0 * win["all_poison"] / bs)
            if win["stats"]:
                row.update({k: float(np.mean([s[k] for s in win["stats"]])) for k in LIFE_STATS
                            if k in win["stats"][0]})
            row.update(gms[0].metrics())
            mf.write(json.dumps(row) + "\n")
            mf.flush()
            print("tick %7d  %5.1f tps  bodies %2d  hits %3d (all bodies %5.1f per 1k)  pred %.5f  drive %.5f  "
                  "act %.5f/%.5f  in hazard %.3f" % (
                      tick, row["sps"], N, row["step_hits"] if env_cfg.respawn else row["hits"], row["all_hits_per_1k"],
                      row.get("pred_mse", float("nan")), row.get("drive_mse", float("nan")),
                      row.get("act_drive_mse", float("nan")), row.get("act_frame_mse", float("nan")),
                      row["hazard_frac"]), flush=True)
            win = new_window()
    if gpu:
        gpu.sync_to(agent)
    agent.save(os.path.join(out_dir, "ckpt_final.bin"), with_optimizer=gpu is None)
    mf.close()


if __name__ == "__main__":
    main()

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

Surprise gating (online.store_surprise > 0) for the universal model on the GPU (task 00028): a step's surprise is the
largest of its tokens' cross-entropies in the update that trained on it (as the newest window), each divided by its
slot's recent mean (GatedMemory.note_tokens). It is read on the next tick, after acting, when the GPU is idle.

Command (online.goal=1): online.goal_mode = "relief" (vm-00006: its hunger if a hit follows within goal_k steps, else
0) or "delta" (the change of hunger over the next goal_k steps: relief positive, hazards and time negative). When
acting, the command is the current hunger: "relieve all of it".

Pieces (online.pieces = 1 with model.band_n = S >= 2, task 00059): S separate transformers, band_k = 0..S-1, live
one lifetime on one world and one body. Each acts as the universal model does (its own move prediction and instinct);
the body's move comes from theirs by online.piece_rule: mean; max (per axis, the proposal of the largest magnitude);
gain (the whole move of the piece whose instinct predicts the lowest next hunger after it). Every piece's move token
holds the move made. They share the experience
(one memory) and each takes its own gradient step on the same rows; piece k starts from seed + 7919 k. Checkpoints
ckpt_NNNNNNNN_pK.bin per piece.

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
from online import (FIELDS, STAT_KEYS, GatedMemory, dream_inputs, fill_rollout, load_config, sample_moves,  # noqa: E402
                    uni_act)
from output_name import next_output  # noqa: E402
from train import git_hash  # noqa: E402
from world import make_config, make_world  # noqa: E402

LIFE_STATS = STAT_KEYS + ("act_drive_mse", "act_frame_mse", "uni_ce_body", "uni_ce_patch", "uni_ce_move",
                          "uni_mse_body", "uni_mse_patch", "uni_mse_move")


GATE_SHARED = ("ltm_chunks", "ltm_relief_share", "ltm_age_mean", "ltm_age_max", "ltm_label_share")


def gate_metrics(gates):
    """The gates' metrics since the last call (task 00053): counts summed over every body's gate, the shared long-term
    memory's state once, the surprise threshold and means body 0's. One gate: its own metrics."""
    ms = [g.metrics() for g in gates]
    out = dict(ms[0])
    for k in out:
        if k not in GATE_SHARED and not k.startswith("surprise_mean_") and k != "surprise_thr":
            out[k] = sum(m[k] for m in ms)
    return out


def ckpt_name(tick, k, S):
    """ckpt_NNNNNNNN.bin (ckpt_final.bin); with pieces (task 00059) piece k's: ckpt_NNNNNNNN_pK.bin."""
    base = "ckpt_final" if tick == "final" else "ckpt_%08d" % tick
    return base + ("_p%d" % k if S > 1 else "") + ".bin"


def piece_seed(seed, k):
    """Piece k's initial weights (task 00059): its own seed, so the pieces do not start the same."""
    return seed + 7919 * k


def body_move(moves, rule="mean", hunger=None, rng=None):
    """The body's move [N, 3] from the pieces' moves (S of [N, 3]) and which piece made each axis [N, 2] (task 00059).
    mean: dx, dy their mean, the click a majority (winners -1: everyone); max: per axis the proposal of the largest
    magnitude (ties, frequent with binned moves, broken at random by rng), the click a majority; gain: the whole move of the piece with the lowest predicted
    next hunger (hunger [S, N]). One piece: its move unchanged."""
    S, N = len(moves), len(moves[0])
    if S == 1:
        return moves[0], np.zeros((N, 2), int)
    mv = np.stack(moves)                                                      # [S, N, 3]
    click = (mv[:, :, 2].mean(0) > 0.5).astype(np.float32)
    if rule == "mean":
        m, won = mv.mean(0), np.full((N, 2), -1)
    elif rule == "max":
        mag = np.abs(mv[:, :, :2]) + 1e-6 * (rng or np.random.default_rng()).random(mv[:, :, :2].shape)
        won = mag.argmax(0)                                                   # [N, 2]
        m = np.concatenate([np.take_along_axis(mv[:, :, :2], won[None], 0)[0], click[:, None]], 1)
    elif rule == "gain":
        if hunger is None or np.isnan(hunger).any():
            raise ValueError("piece_rule gain needs the instinct (online.instinct_m > 1)")
        w = np.asarray(hunger).argmin(0)                                      # [N]
        m, won = mv[w, np.arange(N)].copy(), np.stack([w, w], 1)
    else:
        raise ValueError("online.piece_rule must be mean, max or gain")
    m[:, 2] = click
    return m.astype(np.float32), won


def cosine(u, v):
    """Row-wise cosine of [N, 2] vectors (0 where either is zero)."""
    d = np.linalg.norm(u, axis=1) * np.linalg.norm(v, axis=1)
    return np.where(d > 0, (u * v).sum(1) / np.maximum(d, 1e-12), 0.0)


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
    name = cfg.get("name", os.path.splitext(os.path.basename(args.config))[0])
    out_dir = args.out or next_output(args.exp, "%s_s%d" % (name, args.seed))
    os.makedirs(out_dir, exist_ok=True)
    agent_lib.set_threads(on["threads"])

    N0 = max(1, int(on["bodies"]))
    env_cfg = make_config(cfg["env"], auto_reset=True)   # CircleWorld, or TextWorld with env.kind = text
    env = make_world(N0, env_cfg, seed=args.seed)
    S = int(cfg["model"].get("band_n", 0)) if on["pieces"] else 1   # task 00059: the pieces of one body
    if on["pieces"] and (S < 2 or args.init or not cfg["model"].get("universal")):
        raise ValueError("online.pieces needs the universal model with model.band_n >= 2 (and no --init)")
    if on["piece_rule"] not in ("mean", "max", "gain") or (on["piece_rule"] == "gain" and on["instinct_m"] <= 1):
        raise ValueError("online.piece_rule: mean, max, or gain (with the instinct, online.instinct_m > 1)")
    if S > 1:
        agents = [agent_lib.Agent(dict(cfg["model"], band_k=k), seed=piece_seed(args.seed, k)) for k in range(S)]
    else:
        agents = [agent_lib.Agent.load(args.init) if args.init else agent_lib.Agent(cfg["model"], seed=args.seed)]
    agent = agents[0]
    pib = {name: (o, n) for name, o, n in agent.param_layout()}["pi_b"]
    if on["click_bias"] is not None and not args.init and pib[1]:   # initial click-logit bias, e.g. -2.197 = p 0.1
        agent.params()[pib[0] + 2] = on["click_bias"]                 # (no policy head, e.g. the universal model: none)
        agent.params_changed()
    goal_k = on["goal_k"] if on["goal"] else 0
    uni = bool(agent.config["universal"])
    if on["sf_windows"] and not (uni and on["device"] == "cuda"):
        raise ValueError("self-forcing (online.sf_windows) needs the universal model on the GPU (online.device=cuda)")
    tok_sur = bool(on["store_surprise"])
    if S > 1 and (on["sf_windows"] or tok_sur):
        raise ValueError("online.pieces runs without self-forcing and surprise gating")
    if tok_sur and not (uni and on["device"] == "cuda"):
        raise ValueError("lifetime.py measures surprise for the universal model on the GPU only (online.device=cuda)")
    if uni and goal_k:
        raise ValueError("the universal model has no command: set online.goal=0")
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
    gms = [GatedMemory(on, L, env_cfg.size, goal_k, args.seed, tokens=tok_sur)]
    gms += [GatedMemory(on, L, env_cfg.size, goal_k, args.seed, store=gms[0].store, rng_key=(b,), tokens=tok_sur)
            for b in range(1, N0)]
    store = gms[0].store
    gpu, fwd, gpus = None, agent.forward, [None] * S
    if on["device"] == "cuda":
        import cuda_lib
        gpus = [cuda_lib.CudaModel(a) for a in agents]
        gpu = gpus[0]
        fwd = gpu.forward
    elif on["device"] != "cpu":
        raise ValueError("online.device must be 'cpu' or 'cuda'")
    umodel = (gpu or agent) if uni else None   # the universal model's forward (and, on the GPU, its fast candidates)
    umodels = [g or a for g, a in zip(gpus, agents)] if uni else None
    if tok_sur and not cuda_lib.has("vmc_uni_token_ce"):
        raise ValueError("surprise gating needs a libvmcuda.so with vmc_uni_token_ce (task 00028): rebuild it")
    nt = agent_lib.uni_layout(agent.config)["nt"] if uni else 0
    kinds = np.array([0] * (nt - 2) + [1, 2]) if uni else None   # what each token predicts: patch (or a summary,
    # task 00047: no target, cross-entropy 0), move, body
    ce_n = prev_N = 0   # bodies whose newest window the last update trained on (their surprise is read next tick)
    rollouts = {}

    def rollout(n):   # one per batch size (the batch grows with the active bodies)
        if n not in rollouts:
            # the universal model needs no next frames; builds before task 00026 still expect them
            if not gpu:
                rollouts[n] = agent.new_rollout(L, n)
            else:
                nf = not (uni and cuda_lib.has("vmc_uni_candidates"))
                rollouts[n] = cuda_lib.Rollout(L, n, agent.config["img"], next_frames=nf)
        return rollouts[n]

    def piece_rollout(k, n):   # pieces on the CPU (task 00059): piece k's own C++ rollout
        if (k, n) not in rollouts:
            rollouts[(k, n)] = agents[k].new_rollout(L, n)
        return rollouts[(k, n)]

    info = dict(command=" ".join(sys.argv), git=git_hash(), config=cfg, seed=args.seed, out_dir=out_dir,
                model_config=agent.config, num_params=agent.num_params, threads=agent_lib.get_threads(), pieces=S,
                env_config=dataclasses.asdict(env_cfg), ppo=ppo, start_time=time.strftime("%Y-%m-%d %H:%M:%S"),
                script="lifetime.py")
    with open(os.path.join(out_dir, "info.json"), "w") as f:
        json.dump(info, f, indent=2)
    print("out: %s\nparams: %d  threads: %d  batch: bodies + %d replayed x %d steps  replay: %d  bodies: %d" % (
        out_dir, agent.num_params, agent_lib.get_threads(), B - 1, L, on["replay"], N0), flush=True)
    for k, a in enumerate(agents):
        a.save(os.path.join(out_dir, ckpt_name(0, k, S)), with_optimizer=False)

    mf = open(os.path.join(out_dir, "metrics.jsonl"), "w")
    obs = env.observe()
    prev_a = np.zeros((N0, 3), np.float32)
    prev_r = np.zeros(N0, np.float32)
    ep_start = np.ones(N0, np.uint8)

    def new_window():
        return dict(eps=[], stats=[], step_hits=0, hazard=0, hazard_felt=0, capped=0, all_hits=0, all_hazard=0,
                    body_steps=0, babbled=0, poison=0, all_poison=0, override=0.0, gain=0.0, sf=0, t=time.time(),
                    prop=np.zeros((S, 2)), agree=np.zeros(S), won=np.zeros((S, 2)), stats_k=[[] for _ in range(S)])
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
                   ("cmd", cmd[:, None]), ("es", ep_start[:N, None]), ("act", np.zeros((N, 1, 3), np.float32)))]
        if uni:   # the universal model: its own move prediction, and the instinct; pieces: each its own, the mean made
            props = [uni_act(um, win_in[0], win_in[1], win_in[5].astype(np.float32), win_in[4].astype(np.uint8),
                             agent.config["bins"], rng, on["instinct_m"], on["instinct_tol"], on["instinct_h"],
                             on["instinct_jacobi"], on["instinct_decode"], per_body=True) for um in umodels]
            a, won = body_move([p[0] for p in props], on["piece_rule"], np.stack([p[3] for p in props]), rng)
            win["override"] += np.mean([p[1] for p in props]) * N
            win["gain"] += np.mean([p[2] for p in props]) * N
            if S > 1:
                for k, p in enumerate(props):
                    win["prop"][k] += np.abs(p[0][:, :2]).sum(0)
                    win["agree"][k] += cosine(p[0][:, :2], a[:, :2]).sum()
                    win["won"][k] += (won == k).sum(0)
        else:
            out = fwd(win_in[0], win_in[1], win_in[2], win_in[3].astype(np.float32), win_in[4].astype(np.uint8))
            a = sample_moves(agent, out, rng)
        if tok_sur and prev_N:   # each body's previous step: its surprise in the last update (the GPU is idle now)
            ce = gpu.uni_token_ce(ce_n) if ce_n else None
            for b in range(prev_N):
                gms[b].note_tokens(None if ce is None else ce[b], kinds)
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
        ce_n, prev_N = 0, N
        if gms[0].stm.n >= L + lag:
            ce_n = N
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
            if S > 1:   # every piece its own gradient step on the same rows (task 00059)
                stats = []
                for k, (g, ag) in enumerate(zip(gpus, agents)):
                    if g:   # a GPU rollout holds only the data: one for all pieces
                        stats.append(g.update_rollout(ro, ppo, ag, sync=False, stats=tick % on["stats_every"] == 0))
                    else:   # a C++ rollout belongs to its agent
                        rk = ro if k == 0 else piece_rollout(k, len(rows["img"]))
                        if k:
                            fill_rollout(rk, rows, L)
                        stats.append(ag.update(rk, zeros, ppo))
                if stats[0] is not None:
                    win["stats"].append({key: float(np.mean([s_[key] for s_ in stats])) for key in stats[0]
                                         if isinstance(stats[0][key], (int, float))})
                    for k, s_ in enumerate(stats):
                        win["stats_k"][k].append(s_)
            elif gpu:
                inputs = None
                if on["sf_windows"]:   # self-forcing (task 00039): replayed windows only, none with an episode start
                    ok = np.flatnonzero(~rows["es"][:, 1:].any(1))
                    ok = ok[ok >= ce_n]
                    idx = rng.choice(ok, min(on["sf_windows"], len(ok)), replace=False)
                    k = rng.integers(1, on["sf_steps"] + 1, len(idx))
                    inputs = dream_inputs(gpu, rows["img"], rows["drv"], rows["act"], rows["es"], idx, k,
                                          on["sf_jacobi"])
                    win["sf"] += len(idx)
                st = gpu.update_rollout(ro, ppo, agent, sync=False, stats=tick % on["stats_every"] == 0,
                                        inputs=inputs)
            else:
                st = agent.update(ro, zeros, ppo)
            if S == 1 and st is not None:
                win["stats"].append(st)
        if tick % on["save_every"] == 0:
            for k, (g, ag) in enumerate(zip(gpus, agents)):
                if g:
                    g.sync_to(ag)
                ag.save(os.path.join(out_dir, ckpt_name(tick, k, S)), with_optimizer=False)
        if tick % on["log_every"] == 0:
            eps, le, bs = win["eps"], on["log_every"], max(1, win["body_steps"])
            row = dict(step=tick, time=time.time() - t0, sps=le / max(1e-9, time.time() - win["t"]), bodies=N,
                       episodes=len(eps), hits=sum(e[0] for e in eps),
                       ep_len=float(np.mean([e[1] for e in eps])) if eps else None,
                       step_hits=win["step_hits"], hazard_frac=win["hazard"] / le,
                       hazard_felt=win["hazard_felt"] / le, hunger_capped=win["capped"] / le, hunger=float(h[0]),
                       all_hits_per_1k=1000.0 * win["all_hits"] / bs, all_hazard_frac=win["all_hazard"] / bs,
                       babbled=win["babbled"] / bs, eps=eps_at(on, tick), poison=win["poison"],
                       all_poison_per_1k=1000.0 * win["all_poison"] / bs, instinct_override=win["override"] / bs,
                       instinct_gain=win["gain"] / bs, sf_dreamed=win["sf"])
            if win["stats"]:
                row.update({k: float(np.mean([s[k] for s in win["stats"]])) for k in LIFE_STATS
                            if k in win["stats"][0]})
            if S > 1:   # per piece: its prediction, its proposals' size and their agreement with the move made
                for k in range(S):
                    if win["stats_k"][k]:
                        row.update({"%s_p%d" % (key, k): float(np.mean([s_[key] for s_ in win["stats_k"][k]]))
                                    for key in ("uni_ce_body", "uni_ce_patch", "uni_ce_move")
                                    if key in win["stats_k"][k][0]})
                    row["prop_dx_p%d" % k], row["prop_dy_p%d" % k] = (win["prop"][k] / bs).tolist()
                    row["agree_p%d" % k] = float(win["agree"][k] / bs)
                    row["won_x_p%d" % k], row["won_y_p%d" % k] = (win["won"][k] / bs).tolist()   # max, gain
            row.update(gate_metrics(gms))
            mf.write(json.dumps(row) + "\n")
            mf.flush()
            print("tick %7d  %5.1f tps  bodies %2d  hits %3d (all bodies %5.1f per 1k)  pred %.5f  drive %.5f  "
                  "act %.5f/%.5f  in hazard %.3f" % (
                      tick, row["sps"], N, row["step_hits"] if env_cfg.respawn else row["hits"], row["all_hits_per_1k"],
                      row.get("pred_mse", float("nan")), row.get("drive_mse", float("nan")),
                      row.get("act_drive_mse", float("nan")), row.get("act_frame_mse", float("nan")),
                      row["hazard_frac"]), flush=True)
            win = new_window()
    for k, (g, ag) in enumerate(zip(gpus, agents)):
        if g:
            g.sync_to(ag)
        ag.save(os.path.join(out_dir, ckpt_name("final", k, S)), with_optimizer=g is None)
    mf.close()


if __name__ == "__main__":
    main()

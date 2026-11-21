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

Clogged acting (online.clog_act = 1, task 00086): each transformer also acts on its own damaged view; the world and memory stay clean.
Delayed copies (env.delays, task 00091): the frame is a composite of the view now and in the past (world/delay_world.py); in training the composite is the
input and the present, tiled, the target (online.delay_targets).
Outage (online.off = q, task 00089): each transformer is off q of the time, its own two-state chain with off spells of online.off_span ticks on average;
an off transformer makes no move and takes no update, and its ring skips the ticks it missed (no compute, gaps in its context, ages in ticks). Memory
keeps which transformers were on at each tick; in an update, a transformer's missed steps are no keys and carry no loss.
World schedule (online.env_sched, task 00094): world settings changed at given ticks, {"key": [[tick, value], ...]}, through
env.reconfigure, before that tick's move; only keys the world lists in RUNTIME (CircleWorld: food_show, inertia, inertia_drag). The
metrics carry env_<key>. The frozen tests and videos take inertia and inertia_drag at their last scheduled value (env_at_end).
Input noise (online.noise_in = q, task 00088): the model acts and trains on frames noised as env.noise does, one draw for all transformers; the world, memory and targets stay clean.
Clogged channels (online.clog = p, task 00085): about p of the patches of each step are damaged in each transformer's input, its own damage, drawn fresh each time a
window is used, some for the whole window and some for one step, black or speckled, while the target stays clean. Memory and acting stay clean.

Hunger ahead (online.future_k = k, task 00082): the hunger the move token learns to predict, and the instinct reads, is
the mean over the next k ticks, labelled in hindsight from real futures (the newest window ends k - 1 ticks back).
A bootstrapped value (online.value_gamma = g, task 00096): in place of that mean, (1 - g) sum_{i<n} g^i h_{j+i} + g^n vh_{j+n}, n =
online.value_n (default future_k) real ticks and then the model's own predicted label of the move taken at tick j + n (the
transformer that made the move; the mean of all if the rule mixes them), stored while acting; the newest window ends n ticks back.
online.value_refresh = 1 (task 00098): the tail is made fresh at every update instead, the mean over every transformer of its
predicted label of the move made at tick j + n (a forward pass without gradient over the windows that start n ticks later).

Command (online.goal=1): online.goal_mode = "relief" (vm-00006: its hunger if a hit follows within goal_k steps, else
0) or "delta" (the change of hunger over the next goal_k steps: relief positive, hazards and time negative). When
acting, the command is the current hunger: "relieve all of it".

Pieces (online.pieces = 1 with model.band_n = S >= 2, task 00059): S separate transformers, band_k = 0..S-1, live
one lifetime on one world and one body. Each acts as the universal model does (its own move prediction and instinct);
the body's move comes from theirs by online.piece_rule: mean; max (per axis, the proposal of the largest magnitude);
gain (the whole move of the piece whose instinct predicts the lowest next hunger after it); random (the whole move of
a piece drawn at random); council, veto (every piece scores every piece's move; the lowest mean, or lowest worst, is
made). Every piece's move token holds the move made. They share the experience
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
from online import (STAT_KEYS, ClogStream, GatedMemory, clog_frames, delay_targets, dream_inputs, noise_frames, fill_rollout, load_config, sample_moves,  # noqa: E402
                    DECODE, NEED_FLOOR, UNI_CHUNK, MealMonitor, meal_target, panel_targets, thought_dream, uni_act, uni_frame)
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
    next hunger (hunger [S, N]); random: the whole move of a piece drawn by rng. One piece: its move unchanged."""
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
    elif rule in ("gain", "random"):
        if rule == "random":
            w = rng.integers(S, size=N)
        elif hunger is None or np.isnan(hunger).any():
            raise ValueError("piece_rule gain needs the instinct (online.instinct_m > 1)")
        else:
            w = np.asarray(hunger).argmin(0)                                  # [N]
        m, won = mv[w, np.arange(N)].copy(), np.stack([w, w], 1)
    else:
        raise ValueError("online.piece_rule must be mean, max, gain or random")
    m[:, 2] = click
    return m.astype(np.float32), won


CLOSE = 0.005   # gain_sharp: predicted hungers this close are a tie (as online.instinct_tol)


TIED = {"in": ("uni_in_w", "uni_in_b"), "inout": ("uni_in_w", "uni_in_b", "uni_out_w", "uni_out_b")}   # online.tie_tables


def table_gap(models, ranges):
    """How far the pieces' tables have moved apart (task 00079): their spread around the mean, relative to the mean's size."""
    p = [m.params() for m in models]
    x = np.stack([np.concatenate([q[o:o + n] for o, n in ranges]) for q in p])
    mean = x.mean(0)
    return float(np.sqrt(((x - mean) ** 2).sum(1).mean() / (mean ** 2).sum()))


def boot_values(models, rows, ctx=None, mean=True, chan=0):
    """Task 00098: each window's fresh bootstrap tails [B, L], the mean over models of the move token's predicted
    label (the next body's hunger slot) of the move made at each step of the bootstrap windows (b_img, b_drv, b_act,
    b_es: the L steps that start n steps after each training window). No gradient: forward passes only. Task 00104
    (b_st, the reads stored while acting, [B, L, models, V]): each model reads its own, and windows longer than the
    context (pairs) go in pieces of ctx steps. mean=False (task 00108): each model's own tails [models, B, L]. chan (task
    00112): the next body's channel read (2: the body label)."""
    B, L = rows["b_img"].shape[:2]
    w = ctx or L
    out = []
    for k, m in enumerate(models):
        st = rows.get("b_st")
        out.append(np.concatenate([np.concatenate([
            m.forward_uni(rows["b_img"][i:i + UNI_CHUNK, t:t + w], rows["b_drv"][i:i + UNI_CHUNK, t:t + w],
                          rows["b_act"][i:i + UNI_CHUNK, t:t + w], rows["b_es"][i:i + UNI_CHUNK, t:t + w], values=False,
                          move_probs=False, next_body=True,
                          state=None if st is None else st[i:i + UNI_CHUNK, t:t + w, k].astype(np.float32))["next_body"][..., chan]
            for t in range(0, L, w)], 1) for i in range(0, B, UNI_CHUNK)]))
    return (np.mean(out, 0) if mean else np.stack(out)).astype(np.float32)


def pieces_move(models, props, rule, rng, prev=None, lam=1.0, temp=0.01):
    """The body's move and who made it from the pieces' props (uni_act's, per_body), right after them: council and veto
    (task 00060) have every piece score every piece's move ([scorer, N, mover], only the move token computed; the GPU's
    uni_candidates) and make the one with the lowest mean or lowest worst predicted next hunger. sharp: the piece with
    the sharpest move prediction makes its move; gain_sharp: the sharpest among those within CLOSE of the lowest
    predicted hunger, so gain decides when something is at stake; gain_stay: as gain, but on a tie the piece that made
    the last move (prev [N]) makes this one. Task 00072: pessimism: as council, the lowest mean + lam x the scorers' spread (disagreement as a warning); rank:
    every piece ranks every move, the lowest sum of ranks (ties at random, within a scorer and between sums; rank_first: a tie between sums goes to the lowest piece, a fixed convention, vm-00063); soft: the proposals' mean, weighted by softmin(predicted hunger / temp), from
    the mean at a high temp to gain at a low one. Else body_move."""
    moves = [p[0] for p in props]
    hunger = np.stack([p[3] for p in props])
    N = len(moves[0])
    if rule == "gain_stay":
        best = hunger.argmin(0)
        if prev is not None:
            stay = hunger[prev, np.arange(N)] <= hunger.min(0) + CLOSE
            best = np.where(stay, prev, best)
        m, _ = body_move(moves, "mean")
        m[:, :2] = np.stack(moves, 1)[np.arange(N), best, :2]
        return m, np.stack([best, best], 1)
    if rule == "soft":
        w = np.exp(-(hunger - hunger.min(0)) / temp)
        w = (w / w.sum(0)).astype(np.float32)                                 # [S, N]
        m, won = body_move(moves, "mean")
        m[:, :2] = (w[:, :, None] * np.stack(moves)[:, :, :2]).sum(0)
        return m, won
    if rule not in ("council", "veto", "sharp", "gain_sharp", "pessimism", "rank", "rank_first"):
        return body_move(moves, rule, hunger, rng)
    cand = np.stack(moves, 1)
    if rule in ("sharp", "gain_sharp"):
        sharp = np.stack([p[4] for p in props])                               # [S, N]
        w = (sharp if rule == "sharp" else np.where(hunger <= hunger.min(0) + CLOSE, sharp, -np.inf)).argmax(0)
    else:
        scores = np.stack([m.uni_candidates(cand) for m in models])
        if rule in ("rank", "rank_first"):
            ranks = (scores + 1e-9 * rng.random(scores.shape)).argsort(2).argsort(2).sum(0)
            w = (ranks if rule == "rank_first" else ranks + rng.random(ranks.shape)).argmin(1)   # the sums are small integers and tie often
        elif rule == "pessimism":
            w = (scores.mean(0) + lam * scores.std(0)).argmin(1)
        else:
            w = (scores.mean(0) if rule == "council" else scores.max(0)).argmin(1)
    m, _ = body_move(moves, "mean")                                           # the click: a majority
    m[:, :2] = cand[np.arange(N), w, :2]
    return m, np.stack([w, w], 1)


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


def env_schedule(env, env_cfg, spec, steps):
    """online.env_sched (task 00094): {"key": [[tick, value], ...]} -> {tick: {key: value}}. Refused unless the world can
    change each key mid-life (env.RUNTIME), each tick is a whole number in 1..steps, increasing per key, and the world's
    config accepts every value."""
    if not spec:
        return {}
    keys = getattr(env, "RUNTIME", ())
    if not isinstance(spec, dict) or not hasattr(env, "reconfigure") or any(k not in keys for k in spec):
        raise ValueError(f"online.env_sched: {{key: [[tick, value], ...]}} with keys this world can change mid-life ({keys}), got {spec!r}")
    out = {}
    for k, pts in spec.items():
        ticks = [p[0] for p in pts]
        if not pts or any(len(p) != 2 or int(p[0]) != p[0] or not 1 <= p[0] <= steps for p in pts) or ticks != sorted(set(ticks)):
            raise ValueError(f"online.env_sched[{k!r}]: [[tick, value], ...], whole ticks in 1..{steps}, increasing, got {pts!r}")
        for t, v in pts:
            dataclasses.replace(env_cfg, **{k: v}).validate()
            out.setdefault(int(t), {})[k] = v
    return out


END_KEYS = ("inertia", "inertia_drag")   # task 00106: the scheduled settings a frozen test takes at their last value


def env_at_end(env, spec):
    """Task 00106: a run's world settings as its life ended them, for the frozen tests and videos: each online.env_sched key
    in END_KEYS at its last scheduled value (others, such as food_show, as the run began, as before)."""
    out = dict(env)
    for k, pts in (spec or {}).items():
        if k in END_KEYS:
            out[k] = pts[-1][1]
    return out


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
    ap.add_argument("--init", default=None, help="start from this checkpoint instead of a random model (with online.pieces"
                    " the stem, e.g. run/ckpt_00010000, of the pieces' _pK.bin)")
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
    if on["pieces"] and (S < 2 or not cfg["model"].get("universal")):
        raise ValueError("online.pieces needs the universal model with model.band_n >= 2")
    together = ("council", "veto", "pessimism", "rank", "rank_first")                     # every piece scores every move
    scored = on["piece_rule"] in ("gain", "gain_sharp", "soft") + together
    if on["piece_rule"] not in ("mean", "max", "random") + ("gain", "gain_sharp", "soft") + together or (scored and on["instinct_m"] <= 1):
        raise ValueError("online.piece_rule: mean, max, random, or gain, gain_sharp, soft, council, veto, pessimism, rank, rank_first (with online.instinct_m > 1)")
    if on["piece_rule"] in together and on["device"] != "cuda":
        raise ValueError("online.piece_rule council, veto, pessimism, rank and rank_first run on the GPU (online.device=cuda)")
    if on["tie_every"] < 1:
        raise ValueError("online.tie_every: at least 1")
    if on["tie_tables"] and (on["tie_tables"] not in TIED or S < 2 or on["device"] != "cuda"):
        raise ValueError("online.tie_tables: in or inout, for online.pieces on the GPU (online.device=cuda)")
    if S > 1:
        agents = [agent_lib.Agent.load("%s_p%d.bin" % (args.init, k)) if args.init   # --init a checkpoint's stem: ckpt_NNNNNNNN
                  else agent_lib.Agent(dict(cfg["model"], band_k=k), seed=piece_seed(args.seed, k)) for k in range(S)]
    else:
        agents = [agent_lib.Agent.load(args.init) if args.init else agent_lib.Agent(cfg["model"], seed=args.seed)]
    agent = agents[0]
    tied = [(o, n) for name, o, n in agent.param_layout() if name in TIED.get(on["tie_tables"], ())]   # task 00076
    last_tie = 0
    for a_ in agents[1:] if tied else ():                                         # from piece 0's tables
        for o, n in tied:
            a_.params()[o:o + n] = agent.params()[o:o + n]
        a_.params_changed()
    pib = {name: (o, n) for name, o, n in agent.param_layout()}["pi_b"]
    if on["click_bias"] is not None and not args.init and pib[1]:   # initial click-logit bias, e.g. -2.197 = p 0.1
        agent.params()[pib[0] + 2] = on["click_bias"]                 # (no policy head, e.g. the universal model: none)
        agent.params_changed()
    goal_k = on["goal_k"] if on["goal"] else 0
    uni = bool(agent.config["universal"])
    if on["sf_windows"] and not (uni and on["device"] == "cuda"):
        raise ValueError("self-forcing (online.sf_windows) needs the universal model on the GPU (online.device=cuda)")
    tok_sur = bool(on["store_surprise"])
    if on["ring"] and not (uni and agent.config["age_key"] and on["device"] == "cuda" and on["instinct_h"] == 1 and
                           (N0 == 1 or not on["bodies_halve"])):
        raise ValueError("online.ring needs model.age_key on the GPU, online.instinct_h=1 and a fixed number of bodies")
    if S > 1 and (on["sf_windows"] or tok_sur):
        raise ValueError("online.pieces runs without self-forcing and surprise gating")
    if tok_sur and not (uni and on["device"] == "cuda"):
        raise ValueError("lifetime.py measures surprise for the universal model on the GPU only (online.device=cuda)")
    if uni and goal_k:
        raise ValueError("the universal model has no command: set online.goal=0")
    future_k = on["future_k"]                 # task 00082: the hunger predicted is the mean over the next future_k ticks
    if future_k and (not uni or on["sf_windows"]):
        raise ValueError("online.future_k needs the universal model, without self-forcing")
    gamma = float(on["value_gamma"] or 0.0)   # task 00096: a bootstrapped value
    value_n = int(on["value_n"] or future_k) if gamma else 0
    if gamma and not (0.0 < gamma < 1.0 and future_k >= 1 and value_n >= 1 and on["instinct_m"] > 1):
        raise ValueError("online.value_gamma needs 0 < gamma < 1, online.future_k >= 1, online.value_n >= 1 and the instinct (online.instinct_m > 1)")
    clog = on["clog"]                         # task 00085: each transformer's inputs damaged, the targets clean
    if clog and (not uni or on["device"] != "cuda" or on["sf_windows"]):
        raise ValueError("online.clog needs the universal model on the GPU, without self-forcing")
    if on["clog_act"] and not (clog and on["ring"]):
        raise ValueError("online.clog_act needs online.clog above 0 and online.ring")
    q_in = on["noise_in"]                     # task 00088: noise on the model's input only, the world, memory and targets clean
    if q_in and (not uni or on["device"] != "cuda" or on["sf_windows"] or clog or not on["ring"]):
        raise ValueError("online.noise_in needs the universal model on the GPU with the ring, without self-forcing or online.clog")
    q_off = on["off"]                         # task 00089: transformers switched off at random
    if q_off and (S < 2 or not uni or on["device"] != "cuda" or not on["ring"] or on["sf_windows"] or on["tie_tables"]
                  or not 0 < q_off < 1 or on["off_span"] < 1):
        raise ValueError("online.off needs 0 < off < 1, several pieces, the universal model on the GPU with the ring, "
                         "without self-forcing or tied tables")
    delayed = bool(getattr(env_cfg, "delays", ()))   # task 00091: inputs the composite, targets the present tiled
    if delayed and (not uni or on["device"] != "cuda" or on["sf_windows"] or clog or q_in or q_off):
        raise ValueError("env.delays needs the universal model on the GPU, without self-forcing, online.clog, online.noise_in or online.off")
    panels = bool(on["panels"])   # task 00115: forecast panels; the world wrapped below, after the checks
    if panels and not (uni and on["device"] == "cuda" and S == 4 and agent.config["band_cols"] == 2 and on["ring"]
                       and agent.config["img"] == 2 * int(getattr(env_cfg, "view", 0) or env_cfg.size) and not agent.config["img_w"]
                       and not (delayed or on["sf_windows"] or clog or q_in or q_off or agent.config.get("state"))
                       and on["panel_decode"] in DECODE and on["panel_lag"] >= 0 and on["panel_noise"] >= 0 and N0 == 1):
        raise ValueError("online.panels needs the universal model on the GPU with the ring, four quarter pieces (band_n 4, band_cols 2), "
                         "model.img twice the view, one body, and none of env.delays, self-forcing, clog, noise_in, off or model.state")
    if panels:
        from world.panel_world import PanelWorld
        env = PanelWorld(env, on["panel_lag"], seed=args.seed)
    refresh = bool(on["value_refresh"])      # task 00098: the value's tail fresh at every update, the mean of the transformers
    if refresh and not (gamma and uni and on["device"] == "cuda" and not (clog or q_in or q_off or delayed)):
        raise ValueError("online.value_refresh needs online.value_gamma and the universal model on the GPU, "
                         "without online.clog, online.noise_in, online.off or env.delays")
    meal = on["value_target"] == "meal"   # task 00108: the meal label, each transformer its own
    if on["value_target"] not in ("hunger", "meal") or (meal and not (refresh and value_n == 1 and not agent.config.get("state"))):
        raise ValueError("online.value_target is 'hunger' or 'meal'; 'meal' needs online.value_refresh, online.value_n = 1 and no model.state")
    if on["need"] and on["instinct_m"] < 2:
        raise ValueError("online.need weighs the instinct's choice: it needs online.instinct_m > 1")
    slabel = on["label"] == "meal"   # task 00109: the stored meal label in the body's own channel
    if on["label"] not in ("", "meal") or bool(agent.config.get("body_label")) != slabel or (slabel and not (
            uni and on["device"] == "cuda" and on["instinct_m"] > 1 and not (gamma or future_k or refresh or on["need"]
                                                                              or agent.config.get("state") or on["off"] or clog
                                                                              or on["noise_in"] or on["sf_windows"] or tok_sur))):
        raise ValueError("online.label = meal goes with model.body_label = 1 (and only with it): the universal model on the GPU with "
                         "the instinct, without value_gamma, future_k, value_refresh, need, model.state, off, clog, noise_in, "
                         "self-forcing or surprise gating")
    dose = bool(on["full_dose"])   # task 00110: a poison costs its full dose at any hunger (the world's hunger_cut)
    if dose and not slabel:
        raise ValueError("online.full_dose goes with online.label = meal")
    lref = bool(on["label_refresh"])   # task 00112: the body label re-labeled at every update, each transformer its own
    if lref and not (slabel and S > 1 and not delayed):
        raise ValueError("online.label_refresh goes with online.label = meal, several transformers (online.pieces) and no env.delays")
    state = bool(agent.config.get("state")) and uni   # task 00104: a state carried across windows; updates on pairs of them
    if state and not (on["device"] == "cuda" and on["ring"] and on["batch"] >= 2 and on["batch"] % 2 == 0 and
                      not (on["sf_windows"] or clog or q_in or q_off or delayed or tok_sur or hasattr(env, "think"))):
        raise ValueError("model.state needs the GPU and the ring, an even online.batch, and none of self-forcing, online.clog, "
                         "online.noise_in, online.off, env.delays, env.thought or surprise gating")
    fast = float(on["fast"] or 0.0)   # task 00105: the flexible layer, W = S + F
    if fast and not (0.0 < fast < 1.0 and on["device"] == "cuda" and uni and not (state or on["sf_windows"] or clog or q_in or q_off or delayed)):
        raise ValueError("online.fast needs 0 < fast < 1, the universal model on the GPU, and none of model.state, self-forcing, "
                         "online.clog, online.noise_in, online.off or env.delays")
    mem_on = bool(on["mem_score"]) and bool(getattr(env_cfg, "view", 0))   # task 00105: memory measured in life
    if on["mem_score"] and not mem_on:
        raise ValueError("online.mem_score needs a world larger than the view (env.view)")
    sched = env_schedule(env, env_cfg, on.get("env_sched"), on["steps"])   # task 00094: {tick: {key: value}}
    off_rng = np.random.default_rng([args.seed, 0x0FF])   # its own: off = 0 draws nothing
    is_off, missed = np.zeros(S, bool), np.zeros(S, int)
    p_back = 1.0 / on["off_span"]
    p_down = q_off / ((1.0 - q_off) * on["off_span"]) if q_off else 0.0
    if goal_k and not agent.config["use_prev_reward"]:
        raise ValueError("online.goal=1 needs model.use_prev_reward=1 (the scalar token slot carries the goal)")
    ppo = dict(agent_lib.DEFAULT_PPO)
    ppo.update(dict(use_ppo=0, epochs=1, minibatches=1))
    ppo.update(cfg["ppo"])
    if ppo["use_ppo"] or not env_cfg.drives:
        raise ValueError("lifetime.py is reward-free: needs ppo.use_ppo=0 and env.drives=true")
    L, B = agent.config["ctx"], on["batch"]
    Lm, Bm = (2 * L, B // 2) if state else (L, B)   # task 00104: memory's chunks are pairs of windows, Bm of them per update
    label_k = goal_k or (value_n + 1 if gamma else future_k)   # the steps a hindsight label looks ahead (value: n real, then the tail)
    lag = label_k - 1 if label_k else 0       # the newest window whose hindsight labels are complete
    if slabel:   # task 00109: step t's stored label is written at t + 1, so the newest window ends a step back
        lag = max(lag, 1)
    if panels:   # task 00115: the targets are panel 1 of the 4 steps after each step
        lag = max(lag, 4)
    rng = np.random.default_rng(args.seed)
    clog_rng = np.random.default_rng([args.seed, 0xC106])   # its own, so that clog = 0 draws what it always drew
    in_rng, act_rng = np.random.default_rng([args.seed, 0x0154]), np.random.default_rng([args.seed, 0x0155])   # task 00088
    noised = lambda r: (noise_frames(r["img"], in_rng, q_in), r["drv"]) if q_in else None   # noqa: E731
    damaged = lambda r: (clog_frames(r["img"], clog_rng, agent.config["patch"], clog, on["clog_long"]), r["drv"]) if clog else None   # noqa: E731
    views = [None] * S                        # task 00086: each transformer's damaged view while acting, its own generator
    if on["clog_act"]:
        ac = agent.config
        views = [ClogStream(np.random.default_rng([args.seed, 0xC107, k]), N0, ac["img"], ac.get("img_w") or ac["img"], ac["patch"], clog, on["clog_long"]) for k in range(S)]
    # the universal model reads no next frames, and with model.gray one channel (task 00062)
    kept = dict(ch=1 if uni and agent.config["gray"] else 3, next_frames=not uni, img_w=agent.config["img_w"], pieces=bool(q_off),
                state=(S, agent.config["patch"] ** 2 * (1 if agent.config["gray"] else 3)) if state else None,
                labels=S if slabel else 0, fut=4 if panels else 0)   # task 00109; task 00115
    gate_on = dict(on, goal_mode="value" if gamma else "future") if future_k else on
    gms = [GatedMemory(gate_on, Lm, agent.config["img"], label_k, args.seed, tokens=tok_sur, **kept)]   # the model's frame (task 00091: not the world's size)
    gms += [GatedMemory(gate_on, Lm, agent.config["img"], label_k, args.seed, store=gms[0].store, rng_key=(b,), tokens=tok_sur, **kept)
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
    if on["ring"]:
        for g in gpus:
            g.uni_ring_reset(N0)
    if tok_sur and not cuda_lib.has("vmc_uni_token_ce"):
        raise ValueError("surprise gating needs a libvmcuda.so with vmc_uni_token_ce (task 00028): rebuild it")
    nt = agent_lib.uni_layout(agent.config)["nt"] if uni else 0
    kinds = np.array([0] * (nt - 2) + [1, 2]) if uni else None   # what each token predicts: patch (or a summary,
    # task 00047: no target, cross-entropy 0), move, body
    if panels:   # task 00115: the tokens that predict each piece's own patches (row-major in its block), and its block's side
        panel_lay = [agent_lib.uni_layout(a_.config)["patch"] - 1 for a_ in agents]
        panel_g = agent.config["img"] // agent.config["patch"] // 2
        panel_rng = np.random.default_rng([args.seed, 0x9A4F])
        panel_strip = []
    ce_n = prev_N = 0   # bodies whose newest window the last update trained on (their surprise is read next tick)
    rollouts = {}

    def rollout(n):   # one per batch size (the batch grows with the active bodies)
        if n not in rollouts:
            # the universal model needs no next frames; builds before task 00026 still expect them
            if not gpu:
                rollouts[n] = agent.new_rollout(L, n)
            else:
                nf = not (uni and cuda_lib.has("vmc_uni_candidates"))
                rollouts[n] = cuda_lib.Rollout(L, n, agent.config["img"], next_frames=nf, img_w=agent.config["img_w"])
        return rollouts[n]

    def pair_rollouts(n):   # task 00104: the first and the next windows of n pairs
        if ("pair", n) not in rollouts:
            rollouts[("pair", n)] = tuple(cuda_lib.Rollout(L, n, agent.config["img"], next_frames=False,
                                                           img_w=agent.config["img_w"]) for _ in range(2))
        return rollouts[("pair", n)]

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
    print("out: %s\nparams: %d  threads: %d  batch: bodies + %d replayed x %d steps%s  replay: %d  bodies: %d" % (
        out_dir, agent.num_params, agent_lib.get_threads(), Bm - 1, Lm, " (pairs of windows: task 00104)" if state else "",
        on["replay"], N0), flush=True)
    for k, a in enumerate(agents):
        a.save(os.path.join(out_dir, ckpt_name(0, k, S)), with_optimizer=False)

    mf = open(os.path.join(out_dir, "metrics.jsonl"), "w")
    if mem_on:   # task 00105: a blind world stepped with the same moves, for scoring only
        from eval_hunger import MemoryScore
        from eval_pieces import relative
        blind = make_world(N0, env_cfg, seed=args.seed + 7777)
        mscore = dict(seen=MemoryScore(N0), blind=MemoryScore(N0))
    mmon = MealMonitor(N0, gamma) if refresh and meal else None   # task 00108: the label's predictions against what happened
    smon = None
    if slabel:   # task 00109: the predictions, and the labels stored in memory, against what happened (no cost for the drift)
        mmon = MealMonitor(N0, on["label_gamma"], drift=False)
        smon = MealMonitor(N0, on["label_gamma"], drift=False)
    thinking = hasattr(env, "think")   # task 00064: the thought half's next frame is the right piece's dream, and waves
    if thinking and env.cfg.mode == "dream" and (S != 2 or not gpu or agent.config["band_cols"] != 1):
        raise ValueError("a dreaming env.thought needs two pieces, left and right (online.pieces, model.band_cols=1), on the GPU")
    obs = env.observe()
    prev_a = np.zeros((N0, 3), np.float32)
    prev_r = np.zeros(N0, np.float32)
    ep_start = np.ones(N0, np.uint8)

    def new_window():
        return dict(eps=[], stats=[], step_hits=0, hazard=0, hazard_felt=0, capped=0, all_hits=0, all_hazard=0,
                    body_steps=0, babbled=0, poison=0, all_poison=0, override=0.0, gain=0.0, sf=0, t=time.time(), vh=0.0, vlab=0.0, vlab_n=0, vboot=0.0, vboot_n=0,
                    prop=np.zeros((S, 2)), agree=np.zeros(S), won=np.zeros((S, 2)), stats_k=[[] for _ in range(S)],
                    smooth=0.0, t_window=0.0, t_act=0.0, t_world=0.0, t_memory=0.0, t_batch=0.0, t_update=0.0, t_think=0.0,
                    on_k=np.zeros(S), st_n=0, st_sum=0.0, st_sq=0.0)
    win = new_window()
    t0 = time.time()
    for tick in range(1, on["steps"] + 1):
        if tick in sched:   # task 00094: before this tick's move, so the frame the world draws next follows the new setting
            env.reconfigure(**sched[tick])
            if mem_on:   # task 00106: the blind world must move as the real one does
                blind.reconfigure(**sched[tick])
        N = bodies_at(on, tick)
        h = obs["drives"][:N, 0]
        cmd = (h if goal_k else np.zeros(N)).astype(np.float32)
        if q_off:   # task 00089: each transformer's chain; if all would be off, the lowest stays on
            r_ = off_rng.random(S)
            is_off = np.where(is_off, r_ >= p_back, r_ < p_down)
            if is_off.all():
                is_off[0] = False
        live = np.flatnonzero(~is_off)
        win["on_k"] += ~is_off
        # 1. act: each body's move, predicted from its own last L real states (this one included)
        k = min(gms[0].stm.n, L - 1)
        idx = [gm.stm.last(k) for gm in gms[:N]]
        t_ = time.time()
        if on["ring"]:   # task 00062: only the new step
            win_in = [obs["image"][:N, None], obs["drives"][:N, None], None, None, ep_start[:N, None],
                      np.zeros((N, 1, 3), np.float32)]
        else:
            win_in = [np.stack([np.concatenate([gm.stm.frames(ix) if f == "img" else getattr(gm.stm, f)[ix], cur])
                                for gm, ix, cur in zip(gms[:N], idx, cs)]) for f, cs in
                      (("img", obs["image"][:N, None]), ("drv", obs["drives"][:N, None]), ("pa", prev_a[:N, None]),
                       ("cmd", cmd[:, None]), ("es", ep_start[:N, None]), ("act", np.zeros((N, 1, 3), np.float32)))]
        win["t_window"] += time.time() - t_
        t_ = time.time()
        if q_in:   # task 00088: one noisy view for every transformer; the ring holds it, memory gets the clean frame
            win_in[0] = noise_frames(win_in[0], act_rng, q_in)
        if uni:   # the universal model: its own move prediction, and the instinct; pieces: each its own, the mean made
            for k in live:   # task 00089: back from an outage: the ticks it missed, nothing computed
                if missed[k]:
                    gpus[k].uni_ring_skip(missed[k])
                    missed[k] = 0
            tol = on["instinct_tol"]   # task 00108, online.need: the label weighted by hunger when choosing
            if on["need"]:
                tol = (on["instinct_tol"] / np.maximum(h, NEED_FLOOR)).astype(np.float32)
            props = [uni_act(umodels[k], views[k](win_in[0][:, 0])[:, None] if views[k] else win_in[0], win_in[1], win_in[5].astype(np.float32), win_in[4].astype(np.uint8),
                             agent.config["bins"], rng, on["instinct_m"], tol, on["instinct_h"],
                             on["instinct_jacobi"], on["instinct_decode"], per_body=True, ring=on["ring"], temp=on["instinct_temp"])
                     for k in live]
            a, won = pieces_move([umodels[k] for k in live], props, on["piece_rule"], rng, lam=on["piece_lambda"], temp=on["piece_temp"])
            if gamma or slabel:   # task 00096: the predicted label of the move taken, by the transformer that made it (a mixed move: their mean)
                hp = np.stack([p[3] for p in props])                                    # [live, N]
                vh = np.where(won[:, 0] >= 0, hp[np.maximum(won[:, 0], 0), np.arange(N)], hp.mean(0)).astype(np.float32)
                win["vh"] += float(vh.sum())
            won = np.where(won >= 0, live[np.maximum(won, 0)], won)   # the pieces' own numbers
            win["override"] += np.mean([p[1] for p in props]) * N
            win["gain"] += np.mean([p[2] for p in props]) * N
            if S > 1:
                for k, p in zip(live, props):
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
        if slabel:   # task 00109: each transformer's predicted label of the move made (after babbling), before the ring takes it
            vhs = np.stack([gpus[k].uni_candidates(np.ascontiguousarray(a[:, None, :], np.float32))[:, 0] for k in range(S)], 1)
        if on["ring"]:
            for k in live:
                gpus[k].uni_ring_move(a)
        if panels:   # task 00115: transformers 1-3's forecasts of their own panels at this step, noised, are next tick's panels 2-4
            t_p = time.time()
            if on["panel_blank"]:
                env.set_forecasts(None)
            else:
                kk = min(gms[0].stm.n, L - 1)
                ix = gms[0].stm.last(kk)
                st_ = gms[0].stm
                w_img = np.concatenate([st_.frames(ix), obs["image"][:1]])[None]
                w_drv = np.concatenate([st_.drv[ix], obs["drives"][:1]])[None].astype(np.float32)
                w_act = np.concatenate([st_.act[ix], a[:1]])[None].astype(np.float32)
                w_es = np.concatenate([st_.es[ix], ep_start[:1]])[None].astype(np.uint8)
                fc = []
                for k in range(3):
                    o = gpus[k].forward_uni_step(w_img, w_drv, w_act, w_es, kk, mode=DECODE[on["panel_decode"]])
                    v = o["values_t"][:, panel_lay[k]]
                    if on["panel_noise"] > 0:
                        v = v + panel_rng.normal(0.0, on["panel_noise"], v.shape)
                    fc.append(uni_frame(agent.config, np.clip(v, 0.0, 1.0), grid=(panel_g, panel_g)))
                env.set_forecasts(np.stack(fc, 1))
            win["t_panels"] = win.get("t_panels", 0.0) + time.time() - t_p
        missed[is_off] += 1
        win["t_act"] += time.time() - t_
        t_ = time.time()
        if tick > 1:   # scoring only: how straight the path is
            win["smooth"] += float(cosine(a[:, :2].astype(np.float64), last_a[:N, :2].astype(np.float64)).sum())
        last_a = a.copy()
        # 2. the worlds step; each body remembers its transition
        full = np.zeros((N0, 3), np.float32)
        full[:N] = a
        img, drv = obs["image"][:N].copy(), obs["drives"][:N].copy()
        if mem_on:   # task 00105: before the step, as eval_hunger.py chain
            xy = full[:, :2].astype(np.float64)
            for name, e in (("seen", env), ("blind", blind)):
                mscore[name].see(e, cosine(xy, relative(e, e.target)))
        obs, rew, done, inf = env.step(full)
        if dose and "hunger_cut" not in inf:
            raise ValueError("online.full_dose needs a world that reports hunger_cut (CircleWorld, task 00110)")
        if mmon is not None:   # task 00108: each body's hunger before and after its move, and the label it predicted for it
            mmon.add(vh, h, inf["terminal_drives"][:N, 0], inf["hunger_cut"][:N] if dose else None)
        if mem_on:
            mscore["seen"].ate(inf["success"])
            mscore["blind"].ate(blind.step(full)[3]["success"])
        win["t_world"] += time.time() - t_
        t_ = time.time()
        on_bits = dict(on=np.uint64(sum(1 << int(k) for k in live))) if q_off else {}   # task 00089
        if state:   # task 00104: each transformer's reads this tick, for memory; body 0's for the state's spread
            reads = np.stack([gpus[k].last_read for k in range(S)], 1)                # [N, S, V]
            win["st_n"] += 1
            win["st_sum"] = win["st_sum"] + reads[0].astype(np.float64)
            win["st_sq"] = win["st_sq"] + reads[0].astype(np.float64) ** 2
        for b in range(N):
            vb = dict(vh=vh[b]) if gamma else (dict(vhs=vhs[b]) if slabel else {})    # task 00096; task 00109
            if dose:   # task 00110
                vb["cut"] = inf["hunger_cut"][b]
            if state:
                vb["st"] = reads[b].astype(np.float16)
            gms[b].stm.add(img=img[b], nimg=inf["terminal_image"][b], drv=drv[b], ndrv=inf["terminal_drives"][b],
                           pa=prev_a[b], pr=prev_r[b], es=ep_start[b], act=a[b], rew=rew[b], done=done[b],
                           hit=inf["success"][b], cmd=cmd[b], **on_bits, **vb)
        if slabel:   # task 00109: the step before the new one gets its stored labels, once; the monitor of what memory holds
            ii = [gms[b].write_label(on["label_gamma"]) for b in range(N)]
            if ii[0] is not None:
                sm_ = [gms[b].stm for b in range(N)]
                smon.add(np.array([m_.lab[i_].mean() for m_, i_ in zip(sm_, ii)]), np.array([m_.drv[i_, 0] for m_, i_ in zip(sm_, ii)]),
                         np.array([m_.ndrv[i_, 0] for m_, i_ in zip(sm_, ii)]),
                         np.array([m_.cut[i_] for m_, i_ in zip(sm_, ii)]) if dose else None)   # task 00110
        if thinking:   # the window ending in the new step: the world's new half, the old thought; moves up to now
            t_think = time.time()
            ix = [gm.stm.last(min(gm.stm.n, L - 1)) for gm in gms[:N]]
            st_ = lambda f, new: np.stack([np.concatenate([gm.stm.frames(i) if f == "img" else getattr(gm.stm, f)[i], new[b:b + 1]])   # noqa: E501,E731
                                           for b, (gm, i) in enumerate(zip(gms[:N], ix))])
            dream = None
            if env.cfg.mode == "dream":
                dream = thought_dream(gpus[-1], st_("img", obs["image"]), st_("drv", obs["drives"]).astype(np.float32),
                                      st_("act", np.zeros((N, 3), np.float32)), st_("es", done.astype(np.uint8)),
                                      on["thought_jacobi"], on["thought_decode"])
            obs = dict(obs, image=env.think(dream))
            win["t_think"] += time.time() - t_think
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
        win["t_memory"] += time.time() - t_
        t_ = time.time()
        if gms[0].stm.n >= Lm + lag:
            ce_n = N
            if N == 1:
                rows = gms[0].rows(Bm, rng, lag)
            else:
                parts = [gm.newest(lag) for gm in gms[:N]]
                if store.n:
                    parts.append(gms[0].replayed(Bm - 1, rng))
                else:
                    parts.append(gms[0]._from_stm(gms[0].stm.starts(Lm, Bm, rng, lag)[1:], lag))
                rows = {f: np.concatenate([p[f] for p in parts]) for f in gms[0].keys}
            if refresh and meal:   # task 00108: each transformer's own tail, and so its own label
                vbs = boot_values(gpus, rows, L if state else None, mean=False)
                labels = [meal_target(rows["pr_real"], rows["tw"], v_) for v_ in vbs]
                rows["pr"] = np.mean(labels, 0).astype(np.float32)
                vb = vbs.mean(0)
                win["vboot"] += float(vb.sum())
                win["vboot_n"] += vb.size
            elif refresh:   # task 00098: the tails from the current transformers, their mean prediction of the move made
                vb = boot_values(gpus, rows, L if state else None)
                rows["pr"] = rows["pr_real"] + rows["tw"] * vb
                win["vboot"] += float(vb.sum())
                win["vboot_n"] += vb.size
            if lref:   # task 00112: each transformer's own label, from its fresh prediction of the next move (label channel)
                vbs = boot_values(gpus, rows, mean=False, chan=2)
                labels = [meal_target(rows["pr_real"], rows["tw"], v_) for v_ in vbs]
                rows["pr"] = np.mean(labels, 0).astype(np.float32)
                win["vboot"] += float(vbs.mean(0).sum())
                win["vboot_n"] += vbs[0].size
            elif slabel:   # task 00109: the stored labels, each transformer its own (the mean in the shared rollout)
                rows["pr"] = rows["lab"].mean(2).astype(np.float32)
            if gamma or slabel:
                win["vlab"] += float(rows["pr"].sum())
                win["vlab_n"] += rows["pr"].size
            if future_k:   # the next body's hunger is only a target: the inputs keep the hunger of each step
                rows["ndrv"] = rows["ndrv"].copy()
                rows["ndrv"][..., 0] = rows["pr"]
            seen_in = None
            if delayed:   # task 00091: the composite as the input, the present tiled as the target
                seen_in = (rows["img"], rows["drv"])
                rows = dict(rows, img=delay_targets(rows["img"]))
            elif panels:   # task 00115: the composite as shown as the input; panel i's target the real frame i ticks after panel 1's
                seen_in = (rows["img"], rows["drv"])
                rows = dict(rows, img=panel_targets(rows["f_img"], L))
            if state:   # task 00104: the pairs' first and next windows
                ro1, ro2 = pair_rollouts(len(rows["img"]))
                fill_rollout(ro1, {k: v[:, :L] for k, v in rows.items()}, L)
                fill_rollout(ro2, {k: v[:, L:] for k, v in rows.items()}, L)
                reads1 = rows["st"][:, :L].astype(np.float32)                         # [pairs, L, S, V]
            else:
                ro = rollout(len(rows["img"]))
                fill_rollout(ro, rows, L)
            zeros = np.zeros(len(rows["img"]), np.float32)
            win["t_batch"] += time.time() - t_
            t_ = time.time()
            if state:   # task 00104: every piece its own update on the same pairs, from its own reads
                stats, who = [], []
                for k, g in enumerate(gpus):
                    stats.append(g.update_pairs(ro1, ro2, reads1[:, :, k], ppo, grad=bool(on["state_grad"]),
                                                stats=tick % on["stats_every"] == 0))
                    who.append(k)
                if stats[0] is not None:
                    win["stats"].append({key: float(np.mean([s_[key] for s_ in stats])) for key in stats[0]
                                         if isinstance(stats[0][key], (int, float))})
                    if S > 1:
                        for k, s_ in zip(who, stats):
                            win["stats_k"][k].append(s_)
                st = None
            elif S > 1:   # every piece its own gradient step on the same rows (task 00059)
                stats, who = [], []   # who: the piece of each stats entry (task 00089: off pieces take no update)
                shared = noised(rows)   # task 00088: one noisy input for all pieces, fresh each update
                for k, (g, ag) in enumerate(zip(gpus, agents)):
                    if is_off[k]:   # task 00089: an off transformer does not learn
                        continue
                    if g:   # a GPU rollout holds only the data: one for all pieces
                        if refresh and meal:   # task 00108: this transformer's own label (set_batch copies at once)
                            ro.ndrv[:, :, 0] = labels[k].T
                        elif slabel:   # task 00109: its own stored labels, the body label's target (the prev_reward slot); task 00112: re-labeled
                            ro.pr[:, :] = (labels[k] if lref else rows["lab"][:, :, k]).T
                        sm = ((rows["on"] >> np.uint64(k)) & np.uint64(1)).astype(np.uint8) if q_off else None   # the steps it saw
                        stats.append(g.update_rollout(ro, ppo, ag, sync=False, stats=tick % on["stats_every"] == 0,
                                                      inputs=seen_in if delayed or panels else shared if q_in else damaged(rows), step_mask=sm))
                        who.append(k)
                    else:   # a C++ rollout belongs to its agent
                        rk = ro if k == 0 else piece_rollout(k, len(rows["img"]))
                        if k:
                            fill_rollout(rk, rows, L)
                        stats.append(ag.update(rk, zeros, ppo))
                if tied and tick - last_tie >= on["tie_every"]:   # task 00079: an average every so many ticks; 1: every update
                    if on["tie_every"] > 1:
                        win["tie_gap"] = table_gap(gpus, tied)
                    cuda_lib.tie(gpus, tied)
                    last_tie = tick
                if stats[0] is not None:
                    win["stats"].append({key: float(np.mean([s_[key] for s_ in stats])) for key in stats[0]
                                         if isinstance(stats[0][key], (int, float))})
                    for k, s_ in zip(who, stats):
                        win["stats_k"][k].append(s_)
            elif gpu:
                inputs = seen_in if delayed else noised(rows) if q_in else damaged(rows)
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
            if fast:   # task 00105: each transformer's fast step, from the gradient of the newest window alone
                if ("new", 1) not in rollouts:
                    rollouts[("new", 1)] = cuda_lib.Rollout(L, 1, agent.config["img"], next_frames=False, img_w=agent.config["img_w"])
                ro_new = rollouts[("new", 1)]
                fill_rollout(ro_new, {k: v[:1] for k, v in rows.items()}, L)
                for g in gpus:
                    g.newest_fast(ro_new, fast, on["fast_k"], ppo)
            if on["profile"] and gpu:   # wait for the updates, so that their time is theirs
                for g in gpus:
                    g.sync()
            win["t_update"] += time.time() - t_
        if panels and (tick - 1) % on["panel_snap"] < 6:   # task 00115: 6 consecutive composites of body 0 every panel_snap ticks (and at birth)
            panel_strip.append(obs["image"][0].copy())
            if len(panel_strip) == 6:
                from PIL import Image
                Image.fromarray(np.concatenate(panel_strip, 1)).save(os.path.join(out_dir, "panels_%07d.png" % (tick - 5)))
                panel_strip = []
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
                       instinct_gain=win["gain"] / bs, sf_dreamed=win["sf"], smooth=win["smooth"] / bs,
                       **{k: 1000.0 * win[k] / le for k in win if k.startswith("t_") and k != "t"})
            if gamma:   # task 00096: the mean predicted label of the moves taken, and the mean label trained on
                row.update(value_pred=win["vh"] / bs, value_label=win["vlab"] / max(1, win["vlab_n"]))
            if refresh:   # task 00098: the mean fresh tail
                row.update(value_boot=win["vboot"] / max(1, win["vboot_n"]))
            if fast:   # task 00105: |F| / |S| over all parameters, and the largest by tensor, the mean over transformers
                rat, worst = [], []
                lay = agent.param_layout()
                for g in gpus:
                    F_, P_ = g.fast(0), g.params()
                    S_ = P_ - F_
                    rat.append(float(np.linalg.norm(F_) / max(np.linalg.norm(S_), 1e-12)))
                    worst.append(max(float(np.linalg.norm(F_[o:o + n]) / max(np.linalg.norm(S_[o:o + n]), 1e-12)) for _, o, n in lay if n))
                row.update(fast_ratio=float(np.mean(rat)), fast_ratio_max=float(np.mean(worst)))
            if mem_on:   # task 00105: the memory measure's sums and counts since the last row, by bin
                row["mem"] = dict(seen=mscore["seen"].take(), blind=mscore["blind"].take())
            if mmon is not None:   # task 00108: predicted and real V by ticks to the next meal (MEAL_BIN_NAMES), since the last row
                row["meal"] = mmon.take()
            if smon is not None:   # task 00109: the labels stored in memory, by ticks to the next meal
                row["meal_stored"] = smon.take()
            if state and win["st_n"]:   # task 00104: how much body 0's reads vary over the window, mean over pieces and values
                mu = win["st_sum"] / win["st_n"]
                row["state_spread"] = float(np.sqrt(np.maximum(win["st_sq"] / win["st_n"] - mu ** 2, 0.0)).mean())
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
            for k_ in (on.get("env_sched") or {}):   # task 00094: the scheduled settings as they stand
                row["env_" + k_] = getattr(env.config, k_)
            if q_off:   # task 00089: the share of ticks each piece was on
                row["off_share"] = 1.0 - float(win["on_k"].mean()) / le
                row.update({"on_p%d" % k: float(win["on_k"][k]) / le for k in range(S)})
            if "tie_gap" in win:
                row["tie_gap"] = win.pop("tie_gap")
            row.update(gate_metrics(gms))
            if hasattr(env, "stats"):
                row.update(env.stats())
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
            if fast:   # task 00105: F alone, so that S = the checkpoint - F can be recovered
                np.save(os.path.join(out_dir, ckpt_name("final", k, S).replace(".bin", "_fast.npy")), g.fast(0))
        ag.save(os.path.join(out_dir, ckpt_name("final", k, S)), with_optimizer=g is None)
    mf.close()


if __name__ == "__main__":
    main()

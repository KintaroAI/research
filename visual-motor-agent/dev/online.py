"""One lifetime: a single agent in a single world, learning on every step. No reward anywhere.

Two ways to act (online.actor):
  planner  vm-00005: the CEM planner on the latent world model (below).
  self     vm-00006, the basic agent (babbles instead with probability explore_eps): the model reads its last ctx
           real states (frames, drives, moves) and predicts
           the next one: next move (policy head), next frame and drives (prediction head, given that move). It makes
           the predicted move (sampled) and learns, every step, from what actually happened: the real next frame and
           drives, and the move it made (ppo.move_coef). With online.goal=1 every token also carries the hunger to be
           relieved within the next goal_k steps: in training the true value from memory (hindsight), when acting
           the current hunger. It uses the scalar token slot (model.use_prev_reward=1), which carries no reward here.

Planner, every step:
  1. Act. The CEM planner of plan.py plans with the current weights. If its plan does not lower the imagined hunger
     below doing nothing by at least `explore_gain` per step (the model sees no way to relieve hunger), or with
     probability `explore_eps`, the agent babbles instead: dx, dy ~ N(0, babble_sigma), click with probability
     babble_click (as the vm-00002 babbler). Early in life the model imagines no relief, so the agent babbles;
     later the planner takes over near circles. There is no phase switch.
  2. The world steps (auto-reset after a hit or 200 steps). The transition goes into a replay memory: a ring buffer
     of the agent's own past `replay` steps.
  3. Learn: one gradient step on `batch` sequences of `ctx` consecutive steps. It always includes the newest sequence
     (ending at this step); the other batch-1 are drawn uniformly from the memory. Losses as in reward-free babbling
     (ppo.use_ppo=0): next frame, next drives, inverse dynamics and latent chains. With model.one_frame=1 every
     prediction uses only the current frame.

The world computes a reward (it always does), but nothing reads it: the model never sees it (use_prev_reward=0),
and the losses never use it (use_ppo=0).

Memory (online.memory):
  stream  vm-00005/6: every step enters the ring buffer; replayed sequences start anywhere in it.
  gated   vm-00007 (GatedMemory): a short-term buffer of the last `stm` steps, and a long-term memory of `replay`
          steps kept as whole chunks of ctx steps. Each chunk enters long-term memory when it leads to hunger relief
          (store_relief), when it was surprising (store_surprise: the top fraction of chunks by the error of the
          model's prediction of the next state: frame, drives and, for the self actor, the move), or else with
          probability store_base. Surprise is measured right after each step, before learning from it, and only
          decides what is stored: the model never sees it.

Outputs (default ~/data/research/visual-motor-agent/exp_NNNNN/RRR_name_sS/):
  info.json; metrics.jsonl (every log_every steps: hits, episode lengths, share of planner steps, losses, steps/s);
  ckpt_NNNNNNNN.bin every save_every steps (weights only); ckpt_final.bin.

Usage:
    ./venv/bin/python online.py --config configs/L_online.json --seed 1 --exp 5
    ./venv/bin/python online.py --config configs/L_online.json --set online.batch=1 --set online.replay=8   # no memory
"""
import argparse
import collections
import dataclasses
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import agent_lib  # noqa: E402
from evaluate import executed_action  # noqa: E402
from output_name import next_output  # noqa: E402
from plan import CEMPlanner  # noqa: E402
from train import git_hash, set_path  # noqa: E402
from world import CircleWorld, EnvConfig  # noqa: E402

DEFAULT_ONLINE = dict(steps=200_000, batch=8, replay=100_000, explore_gain=0.02, explore_eps=0.05,
                      babble_sigma=0.607, babble_click=0.1, save_every=10_000, log_every=1_000, threads=8,
                      planner=None, actor="planner", goal=0, goal_k=12, click_bias=None,
                      memory="stream", stm=256, relief_k=12, store_relief=0, store_surprise=0.0, store_base=1.0,
                      store_warmup=1_000, surprise_window=1_000, device="cpu")
EVAL_SEED = 777_000          # the test episodes of train.py / plan.py


class Replay:
    """Ring buffer of single-world transitions; sequences of L consecutive steps never cross the write head."""

    def __init__(self, cap, img):
        self.cap, self.n, self.w = cap, 0, 0
        self.img = np.zeros((cap, img, img, 3), np.uint8)
        self.nimg = np.zeros((cap, img, img, 3), np.uint8)      # the frame after the step (terminal frame if done)
        self.drv = np.zeros((cap, 2), np.float32)
        self.ndrv = np.zeros((cap, 2), np.float32)
        self.pa = np.zeros((cap, 3), np.float32)
        self.pr = np.zeros(cap, np.float32)
        self.es = np.zeros(cap, np.uint8)
        self.act = np.zeros((cap, 3), np.float32)
        self.rew = np.zeros(cap, np.float32)
        self.done = np.zeros(cap, np.uint8)
        self.hit = np.zeros(cap, np.uint8)
        self.cmd = np.zeros(cap, np.float32)      # self actor with goal: the hunger command used when acting

    def add(self, **kw):
        i = self.w
        for k, v in kw.items():
            getattr(self, k)[i] = v
        self.w = (self.w + 1) % self.cap
        self.n = min(self.n + 1, self.cap)

    def oldest(self):
        return (self.w - self.n) % self.cap

    def last(self, k):
        """Physical indices of the last k stored steps, oldest first."""
        return (self.w - k + np.arange(k)) % self.cap

    def starts(self, L, B, rng, lag=0):
        """Physical start indices of B sequences: the newest one (ending lag steps before the newest step) first,
        then B-1 uniform over the memory."""
        hi = self.n - L - lag
        logical = np.concatenate([[hi], rng.integers(0, hi + 1, B - 1)])
        return (self.oldest() + logical) % self.cap

    def fill(self, ro, starts, L, goal_k=0):
        """Write the sequences into a rollout [T=L, N=len(starts)]. With goal_k, the scalar input of each token is
        its hindsight hunger relief: its hunger if a hit happens within its next goal_k steps, else 0."""
        for t in range(L):
            i = (starts + t) % self.cap
            if goal_k:
                ahead = (i[:, None] + np.arange(goal_k)[None]) % self.cap
                pr = np.where(self.hit[ahead].any(1), self.drv[i, 0], 0.0).astype(np.float32)
            else:
                pr = self.pr[i]
            ro.store(t, self.img[i], self.drv[i], self.pa[i], pr, self.es[i], self.act[i])
            ro.outcome(t, self.rew[i], self.done[i], self.nimg[i], self.ndrv[i])


FIELDS = ("img", "nimg", "drv", "ndrv", "pa", "pr", "es", "act", "rew", "done", "hit", "cmd")
SURPRISE_PARTS = ("frame", "drives", "move")
REASONS = ("dropped", "warmup", "relief", "surprise", "base")


def hindsight_labels(hit, done, value, k):
    """For each step j of a stretch of consecutive steps: value[j] if a hit happens at one of the steps j .. j+k-1 of
    the same episode, else 0. Labels near the end of the stretch need the k-1 steps after it to be included."""
    out = np.zeros(len(hit), np.float32)
    nxt = -1                      # the next hit at or after j within j's episode
    for j in range(len(hit) - 1, -1, -1):
        if done[j]:
            nxt = -1              # j ends its episode: hits after it belong to the next one
        if hit[j]:
            nxt = j
        if nxt >= 0 and nxt - j < k:
            out[j] = value[j]
    return out


def downsample_u8(img, f):
    """Block average of [..., S, S, 3] uint8 frames by f, rounded (the C++ prediction targets)."""
    s = img.shape[-3] // f
    x = img.reshape(img.shape[:-3] + (s, f, s, f, 3)).astype(np.int32).sum(axis=(-4, -2))
    return ((x + f * f // 2) // (f * f)).astype(np.uint8)


def surprise_parts(out, t, img, nimg, ndrv, act, sd, f, changed_weight, move=True):
    """How wrong the prediction at token t of a forward pass (with pred_actions) was about the next state:
      frame   weighted MSE of the predicted next frame; pixels that change between the frames count
              1 + changed_weight, as in training;
      drives  squared error of the predicted next drives;
      move    the part of -log pi(move made) that depends on the move: 0.5 sum ((a - mu) / sd)^2 over dx, dy, plus
              -log p of the click outcome. NaN when move=False (the planner chose the move, nothing predicted it)."""
    nxt = downsample_u8(nimg, f)
    w = np.where((nxt != downsample_u8(img, f)).any(-1, keepdims=True), 1.0 + changed_weight, 1.0)
    e = out["pred_frames"][0, t] - nxt / 255.0
    frame = float((w * e * e).mean())
    drives = float(((out["pred_drives"][0, t] - ndrv) ** 2).sum())
    if not move:
        return frame, drives, float("nan")
    z = (act[:2] - out["mu"][0, t]) / sd
    p = float(np.clip(out["click_p"][0, t], 1e-6, 1 - 1e-6))
    return frame, drives, 0.5 * float((z * z).sum()) - float(np.log(p) if act[2] > 0.5 else np.log1p(-p))


class GatedMemory:
    """online.memory = "gated": a short-term buffer of the recent stream (stm), a long-term memory of whole chunks of
    L consecutive steps, and a gate between them. The stream is cut into chunks (steps 1..L, L+1..2L, ...). The chunk
    that ended D = max(relief_k, goal_k) - 1 steps ago, so that whether relief follows it is known, is written to
    long-term memory when (the first that applies):
      warmup    it ends within the first store_warmup steps;
      relief    store_relief and a hit happens in it or in the relief_k - 1 steps after it (same episode);
      surprise  store_surprise = q > 0 and its surprise (the largest of its steps) is in the top fraction q of the
                last surprise_window chunks;
      base      a draw with probability store_base (1 = store everything: the uniform control).
    A step's surprise is the sum of its surprise_parts, each divided by its mean over the last surprise_window steps.
    Training sequences: the newest one from stm, plus chunks drawn uniformly from long-term memory."""

    def __init__(self, on, L, img, goal_k, seed):
        self.on, self.L, self.goal_k = on, L, goal_k
        self.D = max(on["relief_k"], goal_k, 1) - 1
        if on["stm"] < 2 * L + self.D:
            raise ValueError("online.stm must hold at least 2 ctx + D = %d steps" % (2 * L + self.D))
        self.stm = Replay(on["stm"], img)
        self.cap = on["replay"] // L
        self.ltm = {k: np.zeros((self.cap, L) + getattr(self.stm, k).shape[1:], getattr(self.stm, k).dtype)
                    for k in FIELDS}
        self.ltm["lbl"] = np.zeros((self.cap, L), np.float32)     # hindsight labels (goal), fixed at writing
        self.reason = np.zeros(self.cap, np.uint8)
        self.relief = np.zeros(self.cap, bool)                     # leads to relief, whatever the reason it was stored
        self.n = self.w = 0                                        # chunks stored, write position
        self.sur = np.zeros(on["stm"])                             # surprise of the steps in stm
        self.top = np.zeros(on["stm"], np.int8)                    # and its largest part
        self.recent = [collections.deque(maxlen=on["surprise_window"]) for _ in SURPRISE_PARTS]
        self.hist = collections.deque(maxlen=on["surprise_window"])
        self.rng = np.random.default_rng([seed, 7])                # the gate's own draws
        self.thr = float("nan")
        self.counts = collections.Counter()

    def note(self, parts):
        """The surprise parts of the step just added to stm (NaN: not measured)."""
        z = np.zeros(len(parts))
        for j, (q, d) in enumerate(zip(parts, self.recent)):
            if not np.isnan(q):
                d.append(q)
                z[j] = q / max(float(np.mean(d)), 1e-12)
        i = (self.stm.w - 1) % self.stm.cap
        self.sur[i] = z.sum()
        self.top[i] = int(np.argmax(z))

    def decide(self, step):
        """After step `step` (counted from 1) was added and noted: decide the chunk that ended D steps ago."""
        end, L, on, s = step - self.D, self.L, self.on, self.stm
        if end < L or end % L:
            return
        ext = s.last(L + self.D)                                   # the chunk and the D steps after it
        idx = ext[:L]
        relief = bool(hindsight_labels(s.hit[ext], s.done[ext], np.ones(len(ext)), on["relief_k"])[:L].any())
        j = int(np.argmax(self.sur[idx]))
        sur = float(self.sur[idx][j])
        u = self.rng.random()
        q = on["store_surprise"]
        self.thr = float(np.quantile(self.hist, 1 - q)) if q > 0 and self.hist else float("inf")
        self.hist.append(sur)
        if end <= on["store_warmup"]:
            reason = 1
        elif on["store_relief"] and relief:
            reason = 2
        elif q > 0 and sur >= self.thr:
            reason = 3
            self.counts["top_" + SURPRISE_PARTS[self.top[idx][j]]] += 1
        elif u < on["store_base"]:
            reason = 4
        else:
            reason = 0
        self.counts["chunks"] += 1
        self.counts[REASONS[reason]] += 1
        self.counts["relief_seen"] += relief
        if not reason:
            return
        w = self.w
        for k in FIELDS:
            self.ltm[k][w] = getattr(s, k)[idx]
        if self.goal_k:
            self.ltm["lbl"][w] = hindsight_labels(s.hit[ext], s.done[ext], s.drv[ext, 0], self.goal_k)[:L]
        self.reason[w], self.relief[w] = reason, relief
        self.w = (w + 1) % self.cap
        self.n = min(self.n + 1, self.cap)

    def _from_stm(self, starts, lag):
        s, L = self.stm, self.L
        idx = (starts[:, None] + np.arange(L)[None]) % s.cap
        r = {k: getattr(s, k)[idx] for k in FIELDS}
        if self.goal_k:
            ext = (starts[:, None] + np.arange(L + lag)[None]) % s.cap
            r["pr"] = np.stack([hindsight_labels(s.hit[e], s.done[e], s.drv[e, 0], self.goal_k)[:L] for e in ext])
        return r

    def rows(self, B, rng, lag):
        """B training sequences [B, L, ...]: the newest one (ending lag steps back), then B - 1 chunks drawn uniformly
        from long-term memory (from stm, as the stream memory does, until the first chunk is stored)."""
        s = self.stm
        if not self.n:
            return self._from_stm(s.starts(self.L, B, rng, lag), lag)
        new = self._from_stm(np.array([(s.w - self.L - lag) % s.cap]), lag)
        ci = rng.integers(0, self.n, B - 1)
        old = {k: self.ltm[k][ci] for k in FIELDS}
        if self.goal_k:
            old["pr"] = self.ltm["lbl"][ci]
        return {k: np.concatenate([new[k], old[k]]) for k in FIELDS}

    def metrics(self):
        """Gate counts since the last call, and the state of long-term memory."""
        c, n = self.counts, self.n
        m = dict(chunks=c["chunks"], stored=c["chunks"] - c["dropped"], relief_seen=c["relief_seen"],
                 ltm_chunks=n, ltm_relief_share=float(self.relief[:n].mean()) if n else None,
                 surprise_thr=self.thr if np.isfinite(self.thr) else None)
        m.update({"stored_" + r: c[r] for r in REASONS[1:]})
        m.update({"surprise_top_" + p: c["top_" + p] for p in SURPRISE_PARTS})
        m.update({"surprise_mean_" + p: float(np.mean(d)) if d else None for p, d in zip(SURPRISE_PARTS, self.recent)})
        if self.goal_k:
            m["ltm_label_share"] = float((self.ltm["lbl"][:n] > 0).mean()) if n else None
        self.counts = collections.Counter()
        return m


def fill_rollout(ro, r, L):
    """Write training sequences r[field] [N, L, ...] into a rollout [T=L, N]."""
    for t in range(L):
        ro.store(t, r["img"][:, t], r["drv"][:, t], r["pa"][:, t], r["pr"][:, t], r["es"][:, t], r["act"][:, t])
        ro.outcome(t, r["rew"][:, t], r["done"][:, t], r["nimg"][:, t], r["ndrv"][:, t])


def sample_moves(agent, out, rng):
    """Sample moves [n,3] from the last position of a forward pass: dx, dy ~ N(mu, std), click ~ Bernoulli(p)."""
    off = {name: o for name, o, _ in agent.param_layout()}["logstd"]
    sd = np.exp(np.clip(agent.params()[off:off + 2], -5.0, 1.0))
    mu, p = out["mu"][:, -1], out["click_p"][:, -1]
    n = mu.shape[0]
    return np.concatenate([mu + sd * rng.standard_normal((n, 2)), rng.random((n, 1)) < p[:, None]],
                          1).astype(np.float32)


def eval_self(agent, env_cfg, goal, n=128, seed=EVAL_SEED, rng_seed=0):
    """The self actor on the n test episodes (auto-reset off): the moves it predicts from its last ctx real states."""
    # test episodes: the first hit within 200 steps, also in a continual world (its circles, motion and hazards kept)
    env = CircleWorld(n, dataclasses.replace(env_cfg, auto_reset=False, bonuses=False, end_on_success=True,
                                             respawn=False, max_steps=200), seed=seed)
    rng = np.random.default_rng(rng_seed)
    L = agent.config["ctx"]
    hist = []
    obs = env.observe()
    prev_a = np.zeros((n, 3), np.float32)
    es = np.ones(n, np.uint8)
    active = np.ones(n, bool)
    success = np.zeros(n, bool)
    length = np.zeros(n, int)
    clicks = np.zeros(n, int)
    for _ in range(env_cfg.max_steps):
        cur = dict(img=obs["image"], drv=obs["drives"], pa=prev_a,
                   cmd=(obs["drives"][:, 0] if goal else np.zeros(n)).astype(np.float32), es=es)
        win = hist + [cur]
        out = agent.forward(np.stack([w["img"] for w in win], 1), np.stack([w["drv"] for w in win], 1),
                            np.stack([w["pa"] for w in win], 1), np.stack([w["cmd"] for w in win], 1),
                            np.stack([w["es"] for w in win], 1))
        a = np.where(active[:, None], sample_moves(agent, out, rng), 0.0).astype(np.float32)
        obs, _, done, inf = env.step(a)
        length += active
        clicks += active & inf["click"]
        success |= active & inf["success"]
        active &= ~done
        hist = (hist + [cur])[-(L - 1):] if L > 1 else []
        prev_a = executed_action(a)
        es = np.zeros(n, np.uint8)
        if not active.any():
            break
    return dict(success=float(success.mean()), median_steps=float(np.median(np.where(success, length, 200))),
                median_steps_hit=float(np.median(length[success])) if success.any() else float("nan"),
                clicks_per_episode=float(clicks.mean()), episodes=n,
                steps_success=[int(x) for x in np.where(success, length, -1)])


def eval_forage(agent, env_cfg, goal, n=32, steps=1000, seed=EVAL_SEED, rng_seed=0):
    """A continual world (respawn): hits per 1000 steps of the frozen self actor over n worlds of `steps` steps, and
    the share of steps inside a hazard. Acting window: the last ctx real states, as in online.py."""
    env = CircleWorld(n, dataclasses.replace(env_cfg, auto_reset=False, bonuses=False, max_steps=10 ** 9), seed=seed)
    rng = np.random.default_rng(rng_seed)
    L = agent.config["ctx"]
    hist, obs = [], env.observe()
    prev_a = np.zeros((n, 3), np.float32)
    es = np.ones(n, np.uint8)
    hits, in_hz = np.zeros(n), np.zeros(n)
    for _ in range(steps):
        cur = dict(img=obs["image"], drv=obs["drives"], pa=prev_a,
                   cmd=(obs["drives"][:, 0] if goal else np.zeros(n)).astype(np.float32), es=es)
        win = hist + [cur]
        out = agent.forward(*[np.stack([w[k] for w in win], 1) for k in ("img", "drv", "pa", "cmd", "es")])
        a = sample_moves(agent, out, rng)
        obs, _, _, inf = env.step(a)
        hits += inf["success"]
        in_hz += inf.get("gt_in_hazard", np.zeros(n, bool))      # scoring only
        hist = (hist + [cur])[-(L - 1):] if L > 1 else []
        prev_a = executed_action(a)
        es = np.zeros(n, np.uint8)
    return dict(forage_hits_per_1k=float(hits.mean() * 1000 / steps), forage_hazard_frac=float(in_hz.mean() / steps),
                forage_hunger=float(env.hunger.mean()))


class _GpuEval:
    """An agent for eval_self / eval_forage: the C++ agent's weights and config, the GPU's forward."""

    def __init__(self, agent, gpu):
        self.config, self.params, self.param_layout, self.forward = agent.config, agent.params, agent.param_layout, gpu.forward


def load_config(path, overrides):
    with open(path) as f:
        cfg = json.load(f)
    for ov in overrides or []:
        k, v = ov.split("=", 1)
        set_path(cfg, k, v)
    online = dict(DEFAULT_ONLINE)
    online.update(cfg.get("online", {}))
    cfg["online"] = online
    for k in ("env", "model", "ppo"):
        cfg.setdefault(k, {})
    return cfg


STAT_KEYS = ("pred_mse", "pred_changed_mse", "drive_mse", "inv_mse", "inv_r2", "dyn_mse", "dyn_hunger_mse",
             "dyn_recon_mse", "grad_norm", "move_nll", "click_p", "std_x")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", help="required unless --eval")
    ap.add_argument("--set", action="append", default=[], help="override, e.g. online.batch=1")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--exp", type=int, default=5)
    ap.add_argument("-o", "--out", default=None)
    ap.add_argument("--eval", metavar="RUN_DIR", help="evaluate a self-actor run's checkpoints on the test episodes "
                    "(-> RUN_DIR/eval/ckpt_NNNNNNNN.json) instead of training")
    ap.add_argument("--ckpts", type=int, nargs="*", default=[], help="with --eval: checkpoints (steps)")
    ap.add_argument("--eval-device", default="cpu", choices=["cpu", "cuda"],
                    help="with --eval: run the forward passes on the GPU (libvmcuda.so; same outputs to float precision)")
    ap.add_argument("--init", metavar="CKPT", help="start from these weights instead of a random initialization "
                    "(the model config comes from the checkpoint; fresh optimizer, empty memory)")
    args = ap.parse_args()
    if args.eval:
        with open(os.path.join(args.eval, "info.json")) as f:
            info = json.load(f)
        agent_lib.set_threads(16)
        ec = dict(info["env_config"])
        for k in ("bg_color", "target_color", "cursor_color", "cursor_outline", "hazard_color"):
            if k in ec:
                ec[k] = tuple(ec[k])
        os.makedirs(os.path.join(args.eval, "eval"), exist_ok=True)
        for st in args.ckpts:
            agent = agent_lib.Agent.load(os.path.join(args.eval, "ckpt_%08d.bin" % st))
            ev = agent
            if args.eval_device == "cuda":   # the CPU agent's weights (for sampling) and config, the GPU forward
                import cuda_lib
                ev = _GpuEval(agent, cuda_lib.CudaModel(agent))
            res = eval_self(ev, EnvConfig(**ec), info["config"]["online"]["goal"])
            if ec.get("respawn"):
                res.update(eval_forage(ev, EnvConfig(**ec), info["config"]["online"]["goal"]))
            res.update(run=args.eval, step=st)
            with open(os.path.join(args.eval, "eval", "ckpt_%08d.json" % st), "w") as f:
                json.dump(res, f)
            print("%s %8d  success %.3f  median %5.1f  clicks %5.1f" % (os.path.basename(args.eval.rstrip("/")), st,
                  res["success"], res["median_steps"], res["clicks_per_episode"])
                  + ("  forage: hits/1k %.1f  in hazard %.3f" % (res["forage_hits_per_1k"], res["forage_hazard_frac"])
                     if "forage_hits_per_1k" in res else ""), flush=True)
            agent.close()
        return
    if not args.config:
        ap.error("--config is required")

    cfg = load_config(args.config, args.set)
    on = cfg["online"]
    name = cfg.get("name", os.path.splitext(os.path.basename(args.config))[0])
    out_dir = args.out or next_output(args.exp, "%s_s%d" % (name, args.seed))
    os.makedirs(out_dir, exist_ok=True)
    agent_lib.set_threads(on["threads"])

    env_cfg = dataclasses.replace(EnvConfig(**cfg["env"]), auto_reset=True)
    env = CircleWorld(1, env_cfg, seed=args.seed)
    agent = agent_lib.Agent.load(args.init) if args.init else agent_lib.Agent(cfg["model"], seed=args.seed)
    if on["click_bias"] is not None and not args.init:   # initial click-logit bias, e.g. -2.197 = click p 0.1
        off = {name: o for name, o, _ in agent.param_layout()}["pi_b"]
        agent.params()[off + 2] = on["click_bias"]
        agent.params_changed()
    self_actor = on["actor"] == "self"
    if on["actor"] not in ("planner", "self"):
        raise ValueError("online.actor must be 'planner' or 'self'")
    if agent.config["patch"] and not self_actor:
        raise ValueError("model.patch (patch tokens) supports online.actor=self only: the planner imagines with the "
                         "CNN model's latent dynamics")
    goal_k = on["goal_k"] if (self_actor and on["goal"]) else 0
    if goal_k and not agent.config["use_prev_reward"]:
        raise ValueError("online.goal=1 needs model.use_prev_reward=1 (the scalar token slot carries the goal)")
    ppo = dict(agent_lib.DEFAULT_PPO)
    ppo.update(dict(use_ppo=0, epochs=1, minibatches=1))
    ppo.update(cfg["ppo"])
    if ppo["use_ppo"] or not env_cfg.drives:
        raise ValueError("online.py is reward-free: needs ppo.use_ppo=0 and env.drives=true")
    L, B = agent.config["ctx"], on["batch"]
    lag = goal_k - 1 if goal_k else 0         # the newest window whose hindsight labels are complete
    if on["replay"] < L + lag:
        raise ValueError("online.replay must hold at least ctx + goal_k - 1 = %d steps" % (L + lag))
    rng = np.random.default_rng(args.seed)
    if on["memory"] not in ("stream", "gated"):
        raise ValueError("online.memory must be 'stream' or 'gated'")
    gated = on["memory"] == "gated"
    if gated:
        gm = GatedMemory(on, L, env_cfg.size, goal_k, args.seed)
        mem = gm.stm                 # the acting window and the newest sequence; long-term memory is in gm
        ds = agent.config["img"] // agent.config["pred_img"]
        sd_off = {name: o for name, o, _ in agent.param_layout()}["logstd"]
    else:
        mem = Replay(on["replay"], env_cfg.size)
    gpu, fwd = None, agent.forward
    if on["device"] == "cuda":   # the lifetime on the GPU (tasks 00011, 00012): forward passes and updates
        import cuda_lib
        gpu = cuda_lib.CudaModel(agent)
        fwd = gpu.forward
        ro = cuda_lib.Rollout(L, B, agent.config["img"])
    elif on["device"] != "cpu":
        raise ValueError("online.device must be 'cpu' or 'cuda'")
    else:
        ro = agent.new_rollout(L, B)
    zeros_b = np.zeros(B, np.float32)
    dm = gpu or agent   # the model the planner imagines with
    planner = CEMPlanner(dm, 1, seed=args.seed, **(on["planner"] or {}))
    noop = np.zeros((1, planner.H, 3), np.float32)

    info = dict(command=" ".join(sys.argv), git=git_hash(), config=cfg, seed=args.seed, out_dir=out_dir,
                model_config=agent.config, num_params=agent.num_params, threads=agent_lib.get_threads(),
                env_config=dataclasses.asdict(env_cfg), ppo=ppo, start_time=time.strftime("%Y-%m-%d %H:%M:%S"))
    with open(os.path.join(out_dir, "info.json"), "w") as f:
        json.dump(info, f, indent=2)
    print("out: %s\nparams: %d  threads: %d  batch: %d x %d steps  replay: %d" % (
        out_dir, agent.num_params, agent_lib.get_threads(), B, L, on["replay"]), flush=True)
    agent.save(os.path.join(out_dir, "ckpt_%08d.bin" % 0), with_optimizer=False)

    mf = open(os.path.join(out_dir, "metrics.jsonl"), "w")
    obs = env.observe()
    prev_a = np.zeros((1, 3), np.float32)
    prev_r = np.zeros(1, np.float32)
    ep_start = 1
    win = dict(eps=[], planned=0, gain=[], stats=[], dyn_h1=[], step_hits=0, hazard=0)
    t0 = time.time()
    for step in range(1, on["steps"] + 1):
        h = obs["drives"][:, 0]
        gain = 0.0
        cmd = float(h[0]) if goal_k else 0.0
        if self_actor:
            # 1. act: the move the model predicts from its last L real states (this one included)
            idx = mem.last(min(mem.n, L - 1))
            out = fwd(np.concatenate([mem.img[idx], obs["image"]])[None],
                                np.concatenate([mem.drv[idx], obs["drives"]])[None],
                                np.concatenate([mem.pa[idx], prev_a])[None],
                                np.concatenate([mem.cmd[idx], [cmd]]).astype(np.float32)[None],
                                np.concatenate([mem.es[idx], [ep_start]]).astype(np.uint8)[None])
            a = sample_moves(agent, out, rng)
            if rng.random() < on["explore_eps"]:     # keeps trying: own moves alone drift to never clicking
                a = np.concatenate([rng.normal(0, on["babble_sigma"], (1, 2)), rng.random((1, 1)) < on["babble_click"]],
                                   1).astype(np.float32)
        # 1. act (planner): plan, or babble when the model imagines no way to lower hunger
        elif on["explore_eps"] < 1:          # explore_eps >= 1: a pure babbler, no planning
            a_plan, _ = planner.act(obs["image"], h, np.ones(1, bool))
            plan = np.concatenate([np.clip(planner.plan, -1, 1), (planner.plan_click > 0.5)[..., None]], -1)
            s0 = dm.encode(obs["image"])["vf"]
            hp = dm.imagine(np.repeat(s0, 2, 0), np.repeat(h, 2).astype(np.float32),
                               np.concatenate([plan.astype(np.float32), noop]))["hunger"]
            gain = float((hp[1] - hp[0]).mean())
        if self_actor:
            pass
        elif on["explore_eps"] >= 1 or gain < on["explore_gain"] or rng.random() < on["explore_eps"]:
            a = np.concatenate([rng.normal(0, on["babble_sigma"], (1, 2)), rng.random((1, 1)) < on["babble_click"]],
                               1).astype(np.float32)
        else:
            a = a_plan.astype(np.float32)
            win["planned"] += 1
        win["gain"].append(gain)
        # 2. the world steps; remember the transition
        img, drv = obs["image"][0].copy(), obs["drives"][0].copy()
        obs, rew, done, inf = env.step(a)
        mem.add(img=img, nimg=inf["terminal_image"][0], drv=drv, ndrv=inf["terminal_drives"][0], pa=prev_a[0],
                pr=prev_r[0], es=ep_start, act=a[0], rew=rew[0], done=done[0], hit=inf["success"][0], cmd=cmd)
        win["step_hits"] += int(inf["success"][0])                      # every hit (a continual world has no episodes)
        win["hazard"] += int(inf["gt_in_hazard"][0]) if "gt_in_hazard" in inf else 0   # scoring only
        if done[0]:
            win["eps"].append((bool(inf["ep_success"][0]), float(inf["ep_length"][0]), float(inf["ep_clicks"][0])))
            planner.reset(np.ones(1, bool))
        prev_a = executed_action(a)
        prev_r = (rew * ppo["reward_scale"]).astype(np.float32)   # stored for completeness; the model zeroes it
        if done[0]:
            prev_a[:] = 0.0
            prev_r[:] = 0.0
        ep_start = int(done[0])
        if gated:
            # how wrong the prediction of this step's outcome was (before learning from it); then the gate
            i = mem.last(1)[0]
            if self_actor:     # the window the move was predicted from (this step's token last)
                idx = mem.last(min(mem.n, L))
                po = fwd(mem.img[idx][None], mem.drv[idx][None], mem.pa[idx][None], mem.cmd[idx][None],
                                   mem.es[idx][None], pred_actions=mem.act[idx][None])
                sd = np.exp(np.clip(agent.params()[sd_off:sd_off + 2], -5.0, 1.0))
                parts = surprise_parts(po, len(idx) - 1, mem.img[i], mem.nimg[i], mem.ndrv[i], mem.act[i], sd, ds,
                                       ppo["pred_changed_weight"])
            else:              # one-frame model: the token alone; the planner chose the move
                po = fwd(mem.img[i][None, None], mem.drv[i][None, None], mem.pa[i][None, None],
                                   np.zeros((1, 1), np.float32), mem.es[i][None, None],
                                   pred_actions=mem.act[i][None, None])
                parts = surprise_parts(po, 0, mem.img[i], mem.nimg[i], mem.ndrv[i], mem.act[i], None, ds,
                                       ppo["pred_changed_weight"], move=False)
                if on["explore_eps"] < 1:      # diagnostic: the latent hunger head that the planner imagines with
                    hd = dm.imagine(s0, h.astype(np.float32), a[None])["hunger"][0, 0]
                    win["dyn_h1"].append(float((hd - mem.ndrv[i][0]) ** 2))
            gm.note(parts)
            gm.decide(step)
        # 3. learn from the newest sequence plus replayed ones
        if mem.n >= L + lag:
            if gated:
                fill_rollout(ro, gm.rows(B, rng, lag), L)
            else:
                mem.fill(ro, mem.starts(L, B, rng, lag), L, goal_k)
            win["stats"].append(gpu.update_rollout(ro, ppo, agent, sync=False) if gpu else agent.update(ro, zeros_b, ppo))
        if step % on["save_every"] == 0:
            if gpu:
                gpu.sync_to(agent)
            agent.save(os.path.join(out_dir, "ckpt_%08d.bin" % step), with_optimizer=False)
        if step % on["log_every"] == 0:
            eps = win["eps"]
            row = dict(step=step, time=time.time() - t0, sps=on["log_every"] / max(1e-9, time.time() - win.get("t", t0)),
                       episodes=len(eps), hits=sum(e[0] for e in eps),
                       ep_len=float(np.mean([e[1] for e in eps])) if eps else None,
                       ep_clicks=float(np.mean([e[2] for e in eps])) if eps else None,
                       planned=win["planned"] / on["log_every"], gain=float(np.mean(win["gain"])),
                       step_hits=win["step_hits"], hazard_frac=win["hazard"] / on["log_every"],
                       hunger=float(h[0]))
            if win["stats"]:
                row.update({k: float(np.mean([s[k] for s in win["stats"]])) for k in STAT_KEYS})
            gate = ""
            if gated:
                row.update(gm.metrics())
                if win["dyn_h1"]:
                    row["dyn_h1_mse"] = float(np.mean(win["dyn_h1"]))
                gate = "  stored %3d/%3d (relief %d surprise %d base %d)  ltm %5d  relief share %.2f" % (
                    row["stored"], row["chunks"], row["stored_relief"], row["stored_surprise"], row["stored_base"],
                    row["ltm_chunks"], row["ltm_relief_share"] or 0.0)
                if goal_k:
                    gate += "  labelled %.3f" % (row["ltm_label_share"] or 0.0)
            mf.write(json.dumps(row) + "\n")
            mf.flush()
            print("step %7d  %5.1f sps  episodes %3d  hits %3d  len %5s  planned %.2f  gain %+.3f  pred %.5f  "
                  "inv_r2 %.2f  dyn_h %.5f" % (step, row["sps"], row["episodes"],
                                               row["step_hits"] if env_cfg.respawn else row["hits"],
                                               "%.0f" % row["ep_len"] if eps else "-", row["planned"], row["gain"],
                                               row.get("pred_mse", float("nan")), row.get("inv_r2", float("nan")),
                                               row.get("dyn_hunger_mse", float("nan"))) + gate
                  + ("  in hazard %.3f" % row["hazard_frac"] if env_cfg.hazards else ""), flush=True)
            win = dict(eps=[], planned=0, gain=[], stats=[], dyn_h1=[], t=time.time(), step_hits=0, hazard=0)
    if gpu:
        gpu.sync_to(agent)
    agent.save(os.path.join(out_dir, "ckpt_final.bin"), with_optimizer=gpu is None)   # GPU: the optimizer state stays there
    mf.close()


if __name__ == "__main__":
    main()

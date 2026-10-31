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
                      store_warmup=1_000, surprise_window=1_000, device="cpu",
                      # task 00031: which stored chunk a new one replaces once long-term memory is full
                      ltm_evict="oldest",
                      # task 00020 (lifetime.py; online.py reads goal_mode only): the hindsight command's form, and
                      # parallel bodies with a shrinking schedule, exploration decay
                      goal_mode="relief", bodies=1, bodies_hold=0, bodies_halve=0, explore_eps_end=None,
                      explore_decay=0,
                      # task 00023 (the universal model): the hunger instinct, one-step lookahead over instinct_m
                      # candidate moves (0: the policy alone), keeping the first sample within instinct_tol of the best
                      instinct_m=0, instinct_tol=0.005,
                      # its horizon (task 00029): each candidate held for instinct_h steps, the steps after the first
                      # imagined by the model (patches decoded by instinct_jacobi Jacobi iterations); 1: one step
                      instinct_h=1, instinct_jacobi=2, instinct_decode="mean",
                      # lifetime.py on the GPU: read the update's stats every stats_every ticks (each read waits for
                      # the GPU; the metrics average the samples)
                      stats_every=10,
                      # task 00039 (lifetime.py, the universal model on the GPU): self-forcing: sf_windows replayed
                      # windows per update (0: off) get their last 1..sf_steps steps' inputs (frame, body) from the
                      # model's own predictions (sf_jacobi Jacobi iterations per frame, pair decoding); targets stay real
                      sf_windows=0, sf_steps=3, sf_jacobi=2,
                      # task 00059 (lifetime.py, the universal model with model.band_n = S): 1 = all S pieces in one
                      # lifetime, one world and one body; each piece its own weights; the body's move from theirs by
                      # piece_rule: mean, max (per axis the proposal of the largest magnitude) or gain (the move of the
                      # piece whose instinct predicts the lowest next hunger), random (the move of a random piece),
                      # council or veto (every piece scores every move: the lowest mean, or lowest worst)
                      pieces=0, piece_rule="mean")
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

    def fill(self, ro, starts, L, goal_k=0, goal_mode="relief"):
        """Write the sequences into a rollout [T=L, N=len(starts)]. With goal_k, the scalar input of each token is
        its hindsight command (goal_labels): relief, its hunger if a hit happens within its next goal_k steps, else 0;
        delta, the change of hunger over its next goal_k steps."""
        if goal_k and goal_mode != "relief":
            ext = (starts[:, None] + np.arange(L + goal_k - 1)[None]) % self.cap
            lab = np.stack([goal_labels(self, e, goal_k, goal_mode)[:L] for e in ext])
        for t in range(L):
            i = (starts + t) % self.cap
            if goal_k and goal_mode != "relief":
                pr = lab[:, t]
            elif goal_k:
                ahead = (i[:, None] + np.arange(goal_k)[None]) % self.cap
                pr = np.where(self.hit[ahead].any(1), self.drv[i, 0], 0.0).astype(np.float32)
            else:
                pr = self.pr[i]
            ro.store(t, self.img[i], self.drv[i], self.pa[i], pr, self.es[i], self.act[i])
            ro.outcome(t, self.rew[i], self.done[i], self.nimg[i], self.ndrv[i])


FIELDS = ("img", "nimg", "drv", "ndrv", "pa", "pr", "es", "act", "rew", "done", "hit", "cmd")
SURPRISE_PARTS = ("frame", "drives", "move")
TOKEN_KINDS = ("patch", "move", "body")   # what a universal model's token predicts (GatedMemory.note_tokens)
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


def delta_labels(drv0, ndrv0, done, k):
    """For each step j of a stretch of consecutive steps: hunger_j minus the hunger after its next k steps (the step
    j+k-1's outcome), or after the end of j's episode if it ends sooner. Relief is positive, a rise (a hazard, or
    just time) negative. Labels near the end of the stretch need the k-1 steps after it to be included."""
    n = len(drv0)
    out = np.zeros(n, np.float32)
    end = n - 1                   # the step whose outcome closes j's window
    for j in range(n - 1, -1, -1):
        if done[j]:
            end = j
        e = min(j + k - 1, end)
        out[j] = drv0[j] - ndrv0[e]
        if done[j]:
            end = j
    return out


def goal_labels(m, idx, k, mode):
    """The hindsight command of the steps idx (consecutive, physical indices into memory fields m) of form mode."""
    if mode == "relief":
        return hindsight_labels(m.hit[idx], m.done[idx], m.drv[idx, 0], k)
    if mode == "delta":
        return delta_labels(m.drv[idx, 0], m.ndrv[idx, 0], m.done[idx], k)
    raise ValueError("online.goal_mode must be 'relief' or 'delta'")


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


class ChunkStore:
    """Long-term memory of whole chunks of L steps (replay // L chunks); several GatedMemory gates (one per body,
    lifetime.py) can share one. Once full, a chunk the gate stores replaces (ltm_evict):
      oldest     the oldest stored chunk (a ring: the memory holds the most recent stored chunks);
      reservoir  a random stored chunk, and only with probability cap / k for the k-th stored chunk (reservoir
                 sampling: the memory stays a uniform sample of every chunk the gate ever stored, the oldest included)."""

    def __init__(self, on, L, fields_like):
        if on.get("ltm_evict", "oldest") not in ("oldest", "reservoir"):
            raise ValueError("online.ltm_evict must be 'oldest' or 'reservoir'")
        self.evict = on.get("ltm_evict", "oldest")
        self.cap = on["replay"] // L
        self.ltm = {k: np.zeros((self.cap, L) + getattr(fields_like, k).shape[1:], getattr(fields_like, k).dtype)
                    for k in FIELDS}
        self.ltm["lbl"] = np.zeros((self.cap, L), np.float32)     # hindsight labels (goal), fixed at writing
        self.reason = np.zeros(self.cap, np.uint8)
        self.relief = np.zeros(self.cap, bool)                     # leads to relief, whatever the reason it was stored
        self.end = np.zeros(self.cap, np.int64)                    # the step each chunk ended
        self.n = self.w = 0                                        # chunks stored, write position
        self.k = 0                                                 # chunks the gates chose to store, ever

    def slot(self, rng):
        """Where the next chunk a gate chose goes (None: not kept, reservoir only)."""
        self.k += 1
        if self.n < self.cap or self.evict == "oldest":
            return self.w
        j = int(rng.integers(self.k))
        return j if j < self.cap else None


class GatedMemory:
    """online.memory = "gated": a short-term buffer of the recent stream (stm), a long-term memory of whole chunks of
    L consecutive steps, and a gate between them. The stream is cut into chunks (steps 1..L, L+1..2L, ...). The chunk
    that ended D = max(relief_k, goal_k) - 1 steps ago (at least 1 with token surprise, which is noted a tick late), so
    that whether relief follows it is known, is written to
    long-term memory when (the first that applies):
      warmup    it ends within the first store_warmup steps;
      relief    store_relief and a hit happens in it or in the relief_k - 1 steps after it (same episode);
      surprise  store_surprise = q > 0 and its surprise (the largest of its steps) is in the top fraction q of the
                last surprise_window chunks;
      base      a draw with probability store_base (1 = store everything: the uniform control).
    A step's surprise is the sum of its surprise_parts, each divided by its mean over the last surprise_window steps;
    with tokens (the universal model, note_tokens): the largest of its tokens' cross-entropies, each divided by its
    slot's mean over the last surprise_window steps.
    Training sequences: the newest one from stm, plus chunks drawn uniformly from long-term memory."""

    def __init__(self, on, L, img, goal_k, seed, store=None, rng_key=(), tokens=False):
        self.on, self.L, self.goal_k = on, L, goal_k
        self.goal_mode = on.get("goal_mode", "relief")
        # token surprise is noted one tick late (after the update that trained on the step), so a chunk is decided at
        # least one step after it ends (task 00053)
        self.D = max(on["relief_k"], goal_k, 2 if tokens else 1) - 1
        if on["stm"] < 2 * L + self.D:
            raise ValueError("online.stm must hold at least 2 ctx + D = %d steps" % (2 * L + self.D))
        self.stm = Replay(on["stm"], img)
        self.store = store or ChunkStore(on, L, self.stm)          # long-term memory (shared between bodies or not)
        self.sur = np.zeros(on["stm"])                             # surprise of the steps in stm
        self.top = np.zeros(on["stm"], np.int8)                    # and its largest part
        self.recent = [collections.deque(maxlen=on["surprise_window"]) for _ in SURPRISE_PARTS]
        self.hist = collections.deque(maxlen=on["surprise_window"])
        self.rng = np.random.default_rng([seed, 7, *rng_key])      # the gate's own draws
        self.thr = float("nan")
        self.last_end = 0
        self.counts = collections.Counter()
        self.part_names = TOKEN_KINDS if tokens else SURPRISE_PARTS
        self.tok, self.tok_n, self.kinds = None, 0, None             # tokens: the slots' recent cross-entropies

    # the long-term memory's state, under its earlier names
    ltm = property(lambda self: self.store.ltm)
    reason = property(lambda self: self.store.reason)
    relief = property(lambda self: self.store.relief)
    n = property(lambda self: self.store.n)
    cap = property(lambda self: self.store.cap)

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

    def note_tokens(self, ce, kinds):
        """The universal model (task 00028): the cross-entropies ce [ntok] of the step just added to stm (each token's
        prediction of the next token; None: not measured), kinds [ntok] (index into TOKEN_KINDS). The step's surprise
        is the largest ratio of a token's cross-entropy to its slot's mean over the last surprise_window steps: one
        rule for every token."""
        i = (self.stm.w - 1) % self.stm.cap
        if ce is None:
            self.sur[i], self.top[i] = 0.0, 0
            return
        if self.tok is None:
            self.tok, self.kinds = np.zeros((self.on["surprise_window"], len(ce))), np.asarray(kinds)
        self.tok[self.tok_n % len(self.tok)] = ce
        self.tok_n += 1
        z = ce / np.maximum(self.tok[:min(self.tok_n, len(self.tok))].mean(0), 1e-12)
        j = int(np.argmax(z))
        self.sur[i], self.top[i] = z[j], self.kinds[j]

    def decide(self, step):
        """After step `step` (counted from 1) was added and noted: decide the chunk that ended D steps ago."""
        end, L, on, s = step - self.D, self.L, self.on, self.stm
        self.last_end = end
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
            self.counts["top_" + self.part_names[self.top[idx][j]]] += 1
        elif u < on["store_base"]:
            reason = 4
        else:
            reason = 0
        self.counts["chunks"] += 1
        self.counts[REASONS[reason]] += 1
        self.counts["relief_seen"] += relief
        if not reason:
            return
        st = self.store
        w = st.slot(self.rng)
        if w is None:
            self.counts["evicted_new"] += 1
            return
        for k in FIELDS:
            st.ltm[k][w] = getattr(s, k)[idx]
        if self.goal_k:
            st.ltm["lbl"][w] = goal_labels(s, ext, self.goal_k, self.goal_mode)[:L]
        st.reason[w], st.relief[w], st.end[w] = reason, relief, end
        if st.n < st.cap or st.evict == "oldest":
            st.w = (w + 1) % st.cap
            st.n = min(st.n + 1, st.cap)

    def _from_stm(self, starts, lag):
        s, L = self.stm, self.L
        idx = (starts[:, None] + np.arange(L)[None]) % s.cap
        r = {k: getattr(s, k)[idx] for k in FIELDS}
        if self.goal_k:
            ext = (starts[:, None] + np.arange(L + lag)[None]) % s.cap
            r["pr"] = np.stack([goal_labels(s, e, self.goal_k, self.goal_mode)[:L] for e in ext])
        return r

    def newest(self, lag):
        """The newest training sequence of this memory's stream (ending lag steps back), [1, L, ...]."""
        return self._from_stm(np.array([(self.stm.w - self.L - lag) % self.stm.cap]), lag)

    def replayed(self, n, rng):
        """n chunks drawn uniformly from long-term memory, [n, L, ...]."""
        st = self.store
        ci = rng.integers(0, st.n, n)
        old = {k: st.ltm[k][ci] for k in FIELDS}
        if self.goal_k:
            old["pr"] = st.ltm["lbl"][ci]
        return old

    def rows(self, B, rng, lag):
        """B training sequences [B, L, ...]: the newest one (ending lag steps back), then B - 1 chunks drawn uniformly
        from long-term memory (from stm, as the stream memory does, until the first chunk is stored)."""
        s = self.stm
        if not self.store.n:
            return self._from_stm(s.starts(self.L, B, rng, lag), lag)
        new, old = self.newest(lag), self.replayed(B - 1, rng)
        return {k: np.concatenate([new[k], old[k]]) for k in FIELDS}

    def metrics(self):
        """Gate counts since the last call, and the state of long-term memory."""
        c, n = self.counts, self.store.n
        m = dict(chunks=c["chunks"], stored=c["chunks"] - c["dropped"], relief_seen=c["relief_seen"],
                 ltm_chunks=n, ltm_relief_share=float(self.store.relief[:n].mean()) if n else None,
                 surprise_thr=self.thr if np.isfinite(self.thr) else None, ltm_not_kept=c["evicted_new"])
        if n:   # how far back long-term memory reaches, in steps
            age = self.last_end - self.store.end[:n]
            m.update(ltm_age_mean=float(age.mean()), ltm_age_max=int(age.max()))
        m.update({"stored_" + r: c[r] for r in REASONS[1:]})
        m.update({"surprise_top_" + p: c["top_" + p] for p in self.part_names})
        if self.tok is not None:   # tokens: the mean cross-entropy of each kind's slots
            mean = self.tok[:min(self.tok_n, len(self.tok))].mean(0)
            m.update({"surprise_mean_" + p: float(mean[self.kinds == j].mean()) for j, p in enumerate(TOKEN_KINDS)})
        elif self.part_names is SURPRISE_PARTS:
            m.update({"surprise_mean_" + p: float(np.mean(d)) if d else None
                      for p, d in zip(SURPRISE_PARTS, self.recent)})
        if self.goal_k:
            m["ltm_label_share"] = float((self.store.ltm["lbl"][:n] > 0).mean()) if n else None
        self.counts = collections.Counter()
        return m


def fill_rollout(ro, r, L):
    """Write training sequences r[field] [N, L, ...] into a rollout [T=L, N]."""
    for t in range(L):
        ro.store(t, r["img"][:, t], r["drv"][:, t], r["pa"][:, t], r["pr"][:, t], r["es"][:, t], r["act"][:, t])
        ro.outcome(t, r["rew"][:, t], r["done"][:, t], r["nimg"][:, t], r["ndrv"][:, t])


UNI_CHUNK = 256   # windows per forward pass when acting


def uni_patches(cfg, img):
    """The patch tokens' values [n, patches, V] in [0, 1] of frames [n, H, W, 3] (as the universal model reads them)."""
    n, ps = img.shape[0], cfg["patch"]
    g = cfg["img"] // ps
    x = img.reshape(n, g, ps, g, ps, 3).astype(np.float32) / 255.0
    if cfg.get("gray"):
        x = x[..., :1]
    return x.transpose(0, 1, 3, 2, 4, 5).reshape(n, g * g, -1)


def uni_frame(cfg, pv):
    """Frames [n, H, W, 3] u8 from patch values [n, patches, V] (uni_patches' inverse; gray: the value in every
    channel)."""
    n, ps = pv.shape[0], cfg["patch"]
    g, ch = cfg["img"] // ps, 1 if cfg.get("gray") else 3
    x = np.clip(np.rint(np.asarray(pv, np.float64) * 255.0), 0, 255).astype(np.uint8)
    x = x.reshape(n, g, g, ps, ps, ch).transpose(0, 1, 3, 2, 4, 5).reshape(n, g * ps, g * ps, ch)
    return np.repeat(x, 3, -1) if ch == 1 else x


# imagined pixels: the expected value, the most probable bin's centre, or the most probable two-hot pair (task 00038)
DECODE = dict(mean=0, mode=1, pair=2)


def uni_lookahead(model, img, drv, act, es, cand, H, J=2, decode="mean"):
    """The instinct's horizon (task 00029): each candidate move sequence cand [N, M, H, 3] (or [N, M, 3]: one move held
    for H steps) made from the end of the windows [N, T, ...] (act's last step ignored), and the model's predicted
    hunger after the H-th move [N, M]. The window keeps its last ctx - (H - 1) real steps; each imagined step's body is
    the previous move token's predicted next body, its patches are decoded from the model's own expected values by J
    Jacobi iterations from the current frame (J = patches + 1: exactly sequential decoding). decode: each imagined
    pixel the expected value ("mean"), the most probable bin's centre ("mode", sharp) or the most probable bin and its
    more probable neighbour, weighted ("pair": sharp, but a two-hot value exactly; task 00038); the last two on the GPU
    only."""
    if decode not in DECODE:
        raise ValueError("decode must be one of %s" % ", ".join(DECODE))
    if decode != "mean" and not hasattr(model, "forward_uni_step"):
        raise ValueError("decode='%s' needs the GPU model (CudaModel.forward_uni_step)" % decode)
    c = model.config
    N, T = img.shape[:2]
    M = cand.shape[1]
    cs = np.asarray(cand, np.float32)
    if cs.ndim == 3:
        cs = np.repeat(cs[:, :, None], H, 2)
    R = min(T, c["ctx"] - (H - 1))
    if R < 1:
        raise ValueError("instinct_h must be at most ctx")
    pp = agent_lib.uni_layout(c)["patch"] - 1   # the tokens that predict the patches (task 00047: summaries between)
    rep = lambda x: np.repeat(x[:, T - R:], M, axis=0)   # noqa: E731
    im, dv = rep(img), rep(drv).astype(np.float32)
    im = np.concatenate([im] + [im[:, -1:]] * (H - 1), 1)       # the imagined frames start as the current one
    dv = np.concatenate([dv] + [dv[:, -1:]] * (H - 1), 1)
    ac = np.concatenate([rep(act).astype(np.float32), np.zeros((N * M, H - 1, 3), np.float32)], 1)
    ac[:, R - 1:] = cs.reshape(N * M, H, 3)
    e = np.concatenate([rep(es), np.zeros((N * M, H - 1), np.uint8)], 1)
    e[:, 0] = 1

    def fwd(j, st):   # windows j: the next bodies, and (st >= 0) step st's values
        x = (im[j:j + UNI_CHUNK], dv[j:j + UNI_CHUNK], ac[j:j + UNI_CHUNK], e[j:j + UNI_CHUNK])
        if st < 0:
            return model.forward_uni(*x, values=False, move_probs=False, next_body=True)
        if hasattr(model, "forward_uni_step"):   # the GPU: only step st's values come back
            return model.forward_uni_step(*x, st, mode=DECODE[decode])
        o = model.forward_uni(*x, values=True, move_probs=False, next_body=True)
        return dict(values_t=o["values"][:, st], next_body=o["next_body"])

    def run(st):
        outs = [fwd(j, st) for j in range(0, N * M, UNI_CHUNK)]
        return {k: np.concatenate([o[k] for o in outs]) for k in outs[0]}
    for st in range(R, R + H - 1):
        for _ in range(J):
            out = run(st)
            dv[:, st] = np.clip(out["next_body"][:, st - 1], 0.0, 1.0)
            im[:, st] = uni_frame(c, out["values_t"][:, pp])
    return run(-1)["next_body"][:, -1, 0].reshape(N, M)


def dream_inputs(model, img, drv, act, es, idx, k, J=2, decode="pair"):
    """Self-forcing (task 00039): the token inputs of windows [N, T, ...] with the last k[i] steps of window idx[i]
    replaced by the model's own predictions, in order: each dreamed step's body is the previous step's move token's
    prediction, its frame J Jacobi iterations from the previous frame (J = patches + 1: exactly sequential decoding),
    values decoded `decode` (DECODE). Moves and the other steps stay real. model: a CudaModel (forward_uni_step).
    Returns copies (frames u8, bodies f32) [N, T, ...]."""
    c = model.config
    T = img.shape[1]
    lay = agent_lib.uni_layout(c)
    pp, mv = lay["patch"] - 1, lay["move"]   # the tokens that predict the patches; the move token (the next body)
    im, dv = img.copy(), np.array(drv, np.float32)
    idx, k = np.asarray(idx, int), np.asarray(k, int)
    if not len(idx):
        return im, dv
    if k.min() < 1 or k.max() > T - 1:
        raise ValueError("dreamed steps must be 1 .. T - 1")
    wi, wd = im[idx], dv[idx]
    wa, we = np.ascontiguousarray(act[idx], np.float32), np.ascontiguousarray(es[idx], np.uint8)
    start = T - k
    for s in range(int(start.min()), T):
        on = s >= start
        o = model.forward_uni_step(wi, wd, wa, we, s - 1, mode=DECODE[decode])   # the move token's next body
        wd[on, s] = np.clip(o["values_t"][on, mv, :2], 0.0, 1.0)
        wi[on, s] = wi[on, s - 1]
        for _ in range(J):
            o = model.forward_uni_step(wi, wd, wa, we, s, mode=DECODE[decode])
            wi[on, s] = uni_frame(c, o["values_t"][on][:, pp])
    im[idx], dv[idx] = wi, wd
    return im, dv


def uni_act(model, img, drv, act, es, bins, rng, m=0, tol=0.005, horizon=1, jacobi=2, decode="mean", per_body=False,
            first="sample"):
    """The universal model's moves for N windows [N, T, ...] (act: the moves made, the last step's ignored): sample each
    of (dx+1)/2, (dy+1)/2, click from the bins predicted for the current move; with the instinct (m > 0), sample m
    candidates, read each one's predicted next hunger (the move token's prediction of the next body) and make the
    lowest, keeping the first sample when it is within tol of the lowest. model: an Agent or CudaModel (forward_uni;
    with uni_candidates, the candidates reuse the policy pass instead of a full pass each). horizon > 1: each candidate
    held for that many steps and scored by the predicted hunger after the last (uni_lookahead). Returns the moves [N, 3]
    (dx, dy in [-1, 1], click 0 / 1), the share of bodies where the instinct overrode the first sample, and the mean
    predicted gain; per_body (task 00059): also each body's predicted next hunger after its move [N] (with the
    instinct; without it NaN) and the sharpness of its move prediction [N]: minus the entropy over the bins, dx plus
    dy. first = mode: the first candidate is the most likely bin of each value instead of a sample, so with nothing at
    stake the move is the most likely one."""
    N = img.shape[0]
    a0 = act.copy()
    a0[:, -1] = 0.0
    M = max(1, m)
    fast = m > 1 and horizon == 1 and hasattr(model, "uni_candidates")
    cand, h, sharp = [], [], []
    for i in range(0, N, UNI_CHUNK):   # at most UNI_CHUNK windows per forward pass (bounded memory)
        sl = slice(i, i + UNI_CHUNK)
        probs = model.forward_uni(img[sl], drv[sl], a0[sl], es[sl], values=False, move_probs=True)["move_probs"][:, -1]
        n = probs.shape[0]
        if per_body:
            sharp.append((probs[:, :2] * np.log(np.maximum(probs[:, :2], 1e-12))).sum((1, 2)))
        idx = np.minimum((np.cumsum(probs, -1)[:, None] < rng.random((n, M, 3, 1))).sum(-1), bins - 1)   # [n, M, 3]
        if first == "mode":
            idx[:, 0] = probs.argmax(-1)
        c = idx / (bins - 1)
        cc = np.stack([2 * c[..., 0] - 1, 2 * c[..., 1] - 1, (c[..., 2] > 0.5).astype(np.float64)], -1).astype(np.float32)
        cand.append(cc)
        if fast:   # right after the policy pass of these windows: its keys and values are reused
            h.append(model.uni_candidates(cc))
    cand = np.concatenate(cand)
    if m <= 1:
        return (cand[:, 0], 0.0, 0.0) + ((np.full(N, np.nan), np.concatenate(sharp)) if per_body else ())
    if horizon > 1:
        h = uni_lookahead(model, img, drv, act, es, cand, horizon, jacobi, decode)
    elif fast:
        h = np.concatenate(h)
    else:
        rep = lambda x: np.repeat(x, M, axis=0)   # noqa: E731
        ac = rep(a0)
        ac[:, -1] = cand.reshape(N * M, 3)
        ri, rd, re = rep(img), rep(drv), rep(es)
        h = np.concatenate([model.forward_uni(ri[j:j + UNI_CHUNK], rd[j:j + UNI_CHUNK], ac[j:j + UNI_CHUNK],
                                              re[j:j + UNI_CHUNK], values=False, move_probs=False,
                                              next_body=True)["next_body"][:, -1, 0]
                            for j in range(0, N * M, UNI_CHUNK)]).reshape(N, M)
    best = np.argmin(h, 1)
    pick = np.where(h[:, 0] - h[np.arange(N), best] <= tol, 0, best)
    gain = h[:, 0] - h[np.arange(N), pick]
    out = cand[np.arange(N), pick], float((pick != 0).mean()), float(gain.mean())
    return out + ((h[np.arange(N), pick], np.concatenate(sharp)) if per_body else ())


def sample_moves(agent, out, rng):
    """Sample moves [n,3] from the last position of a forward pass: dx, dy ~ N(mu, std), click ~ Bernoulli(p)."""
    off = {name: o for name, o, _ in agent.param_layout()}["logstd"]
    sd = np.exp(np.clip(agent.params()[off:off + 2], -5.0, 1.0))
    mu, p = out["mu"][:, -1], out["click_p"][:, -1]
    n = mu.shape[0]
    return np.concatenate([mu + sd * rng.standard_normal((n, 2)), rng.random((n, 1)) < p[:, None]],
                          1).astype(np.float32)


def eval_moves(agent, win, rng, instinct=(0, 0.005)):
    """The frozen agent's moves from a window of steps (dicts of img, drv, pa, cmd, es, act; the last is the current
    step): the policy head's sample, or for the universal model its own prediction plus the instinct (m, tol)."""
    st = lambda k: np.stack([w[k] for w in win], 1)   # noqa: E731
    if agent.config.get("universal"):
        return uni_act(agent, st("img"), st("drv"), st("act").astype(np.float32), st("es"),
                       agent.config["bins"], rng, *instinct)[0]
    return sample_moves(agent, agent.forward(st("img"), st("drv"), st("pa"), st("cmd"), st("es")), rng)


def eval_self(agent, env_cfg, goal, n=128, seed=EVAL_SEED, rng_seed=0, instinct=(0, 0.005)):
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
                   cmd=(obs["drives"][:, 0] if goal else np.zeros(n)).astype(np.float32), es=es,
                   act=np.zeros((n, 3), np.float32))
        win = hist + [cur]
        a = np.where(active[:, None], eval_moves(agent, win, rng, instinct), 0.0).astype(np.float32)
        cur["act"] = a
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


def eval_forage(agent, env_cfg, goal, n=32, steps=1000, seed=EVAL_SEED, rng_seed=0, instinct=(0, 0.005)):
    """A continual world (respawn): hits per 1000 steps of the frozen self actor over n worlds of `steps` steps, and
    the share of steps inside a hazard, of hazard steps that raised hunger (none at the cap of 1) and of steps at the
    cap. Acting window: the last ctx real states, as in online.py."""
    env = CircleWorld(n, dataclasses.replace(env_cfg, auto_reset=False, bonuses=False, max_steps=10 ** 9), seed=seed)
    rng = np.random.default_rng(rng_seed)
    L = agent.config["ctx"]
    hist, obs = [], env.observe()
    prev_a = np.zeros((n, 3), np.float32)
    es = np.ones(n, np.uint8)
    hits, in_hz, felt, capped, poison = np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n)
    for _ in range(steps):
        h0 = env.hunger.copy()
        cur = dict(img=obs["image"], drv=obs["drives"], pa=prev_a,
                   cmd=(obs["drives"][:, 0] if goal else np.zeros(n)).astype(np.float32), es=es,
                   act=np.zeros((n, 3), np.float32))
        win = hist + [cur]
        a = eval_moves(agent, win, rng, instinct)
        cur["act"] = a
        obs, _, _, inf = env.step(a)
        poison += inf.get("gt_hazard_click", np.zeros(n, bool))  # scoring only
        hits += inf["success"]
        in_hz += inf.get("gt_in_hazard", np.zeros(n, bool))      # scoring only
        felt += inf.get("gt_in_hazard", np.zeros(n, bool)) & (env.hunger - h0 > 1.5 * env_cfg.h_rate)
        capped += h0 >= 1.0 - 1e-6
        hist = (hist + [cur])[-(L - 1):] if L > 1 else []
        prev_a = executed_action(a)
        es = np.zeros(n, np.uint8)
    return dict(forage_hits_per_1k=float(hits.mean() * 1000 / steps), forage_hazard_frac=float(in_hz.mean() / steps),
                forage_hunger=float(env.hunger.mean()), forage_hazard_felt=float(felt.mean() / steps),
                forage_capped=float(capped.mean() / steps), forage_poison_per_1k=float(poison.mean() * 1000 / steps))


class _GpuEval:
    """An agent for eval_self / eval_forage: the C++ agent's weights and config, the GPU's forward."""

    def __init__(self, agent, gpu):
        self.config, self.params, self.param_layout, self.forward = agent.config, agent.params, agent.param_layout, gpu.forward
        self.forward_uni = gpu.forward_uni
        self.uni_candidates = gpu.uni_candidates
        self.forward_uni_step = gpu.forward_uni_step


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
    ap.add_argument("--eval-instinct", type=int, default=None,
                    help="universal model: the instinct's candidates in the frozen tests (default: the run's; 0: off)")
    ap.add_argument("--eval-tests", default="both", choices=["both", "forage"],
                    help="with --eval: both frozen tests, or the forage test alone (e.g. for a costly instinct)")
    ap.add_argument("--eval-worlds", type=int, default=32, help="with --eval: the forage test's worlds")
    ap.add_argument("--eval-horizon", type=int, default=None,
                    help="universal model: the instinct's horizon in the frozen tests (default: the run's)")
    ap.add_argument("--eval-decode", default=None, choices=["mean", "mode", "pair"],
                    help="universal model: how the lookahead decodes imagined frames (default: the run's)")
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
            ion = dict(DEFAULT_ONLINE, **info["config"]["online"])
            inst = (ion["instinct_m"] if args.eval_instinct is None else args.eval_instinct, ion["instinct_tol"],
                    ion["instinct_h"] if args.eval_horizon is None else args.eval_horizon, ion["instinct_jacobi"],
                    ion["instinct_decode"] if args.eval_decode is None else args.eval_decode)
            res = {} if args.eval_tests == "forage" else eval_self(ev, EnvConfig(**ec), info["config"]["online"]["goal"],
                                                                  instinct=inst)
            if ec.get("respawn"):
                res.update(eval_forage(ev, EnvConfig(**ec), info["config"]["online"]["goal"], n=args.eval_worlds,
                                       instinct=inst))
            res["instinct_m"], res["instinct_h"] = inst[0], inst[2]
            res.update(run=args.eval, step=st)
            tag = ("" if args.eval_instinct is None else "_instinct%d" % args.eval_instinct) + (
                "" if args.eval_horizon is None else "_h%d" % args.eval_horizon) + (
                "" if args.eval_decode is None else "_" + args.eval_decode) + (
                "" if args.eval_tests == "both" else "_forage%d" % args.eval_worlds)
            with open(os.path.join(args.eval, "eval", "ckpt_%08d%s.json" % (st, tag)), "w") as f:
                json.dump(res, f)
            print("%s %8d" % (os.path.basename(args.eval.rstrip("/")), st)
                  + ("  success %.3f  median %5.1f  clicks %5.1f" % (res["success"], res["median_steps"],
                                                                      res["clicks_per_episode"]) if "success" in res else "")
                  + ("  forage: hits/1k %.1f  in hazard %.3f (felt %.3f)  hunger capped %.2f" % (
                      res["forage_hits_per_1k"], res["forage_hazard_frac"], res["forage_hazard_felt"],
                      res["forage_capped"])
                     if "forage_hits_per_1k" in res else "")
                  + ("  poison clicks/1k %.1f" % res["forage_poison_per_1k"] if res.get("forage_poison_per_1k") else "")
                  + ("  instinct %d" % inst[0] if inst[0] else "") + ("  horizon %d" % inst[2] if inst[2] > 1 else ""),
                  flush=True)
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
    pib = {name: (o, n) for name, o, n in agent.param_layout()}["pi_b"]
    if on["click_bias"] is not None and not args.init and pib[1]:   # initial click-logit bias, e.g. -2.197 = p 0.1
        agent.params()[pib[0] + 2] = on["click_bias"]                 # (no policy head, e.g. the universal model: none)
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
    win = dict(eps=[], planned=0, gain=[], stats=[], dyn_h1=[], step_hits=0, hazard=0, hazard_felt=0, capped=0,
               poison=0)
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
        # scoring only: hazard steps whose cost showed in the drives (hunger is capped at 1: at the cap a hazard step
        # changes nothing), and steps at the cap
        dh = float(inf["terminal_drives"][0, 0] - drv[0])
        win["hazard_felt"] += int(bool(inf["gt_in_hazard"][0]) and dh > 1.5 * env_cfg.h_rate) if "gt_in_hazard" in inf else 0
        win["capped"] += int(drv[0] >= 1.0 - 1e-6)
        win["poison"] += int(inf["gt_hazard_click"][0]) if "gt_hazard_click" in inf else 0   # scoring only
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
                mem.fill(ro, mem.starts(L, B, rng, lag), L, goal_k, on["goal_mode"])
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
                       hazard_felt=win["hazard_felt"] / on["log_every"], hunger_capped=win["capped"] / on["log_every"],
                       poison=win["poison"],
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
            win = dict(eps=[], planned=0, gain=[], stats=[], dyn_h1=[], t=time.time(), step_hits=0, hazard=0,
                       hazard_felt=0, capped=0, poison=0)
    if gpu:
        gpu.sync_to(agent)
    agent.save(os.path.join(out_dir, "ckpt_final.bin"), with_optimizer=gpu is None)   # GPU: the optimizer state stays there
    mf.close()


if __name__ == "__main__":
    main()

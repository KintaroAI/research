"""World-model measures that do not depend on the model (task 00132; ARCHITECTURE.md section 15).

A model is read through three calls only:

    reset(n)                      n streams, nothing carried over (the weights stay)
    observe(frame, body, move)    one real tick of every stream: the frame [n, S, S] in grey levels (u8), the body's
                                  values [n, 2], the move [n, 3] made on seeing them
    dream(H, moves=None)          the next H ticks as frames [n, H, S, S] in grey levels and bodies [n, H, 2]; moves
                                  [n, H, 3] are the moves made at the dreamed ticks (None: null moves). It leaves what
                                  the model carries untouched. frames=False: only the bodies are read (the last
                                  tick's frame need not be made).

Its loss, bins, tokens and attention are never read, so models with different parts can be set side by side. Every
number is in the world's own units (grey levels, px, hunger, events) and stands beside floors that need no model:
copy (the last real frame held), the mean frame, a straight line for each thing (through its last two true places,
and, fairer, through its places as the last 2, 4 or 8 frames show them), and the ceilings: the world itself and,
where the world has chance, the world with other dice.

The exam is one world of 16 copies stepped with null moves, 2,300 ticks from birth, its state kept for scoring only:
seed 1001 is the working set, seed 2001 the final set (not to be opened before a final comparison). "Home" is a
life's own world (its seed) from tick 60,000 on. Before a start the model observes the 64 real ticks up to it.

The numbers (the task plan has them in words; they were set before any life was scored):
    next     the first dreamed tick of the regular starts: skill = 1 - the model's summed squared error / the copy's,
             over the frame and by kind of thing
    curve    the same for ticks 1..32 against the copy and the mean frame; the horizon is the number of ticks from
             the first on during which the dream's error stays below both
    hold     the reader (locate) on every thing at every dreamed tick: held (found within 2 px), lost, the distance
    real     the distance between the grey-level distributions of dreamed and real frames; the share of valid frames
             (the count of bright pixels of each zone within the real frames' range)
    events   the court: the predicted change of the hunger as a forecast of hits and misses (area under the curve),
             next tick and within h dreamed ticks; the disc world: the skill and the distance on discs that turn
    warm     next and the next-tick forecast from a reset, shown the last k real ticks
    free     a dream of 200 ticks: the valid frames and the bright counts by zone (what the dream keeps)
Added after the first lives were scored (vm-00119): the decoders instinct and instinct_pair (Pieces), the counts by
zone (zone_counts), the straight lines through the frames as seen (seen_things), the dream after a short warm-up
(the part warmdream), the discs read at tick 4 by whether they turned (after_a_turn).

Usage (from dev/, on a GPU):
    ./venv/bin/python wm_measures.py score RUN --step 60000 [--decode pair] [--set work] [--parts regular events warm]
    ./venv/bin/python wm_measures.py anchors configs/W_discs.json -o OUT        (no GPU: the floors and the ceilings)
    ./venv/bin/python wm_measures.py table OUT WD                               (no GPU: the README's tables)
    ./venv/bin/python wm_measures.py strip RUN --step 60000 -o PNG               (a picture: real ticks and the dreamed ones)
    ./venv/bin/python wm_measures.py figure OUT WD -o PNG                        (the dream's curves of an arm)
score writes RUN/wm/step_<step>_<decode>_<set>.json and .npz (the errors by stretch, for paired comparisons). The
learning curve is the regular part cut to 8 dreamed ticks at every checkpoint, with the next-tick events only:
--horizon 8 --parts regular next --name learn. table reads OUT/<ARM>_s<seed>/wm/ and OUT/anchors_<world>_work.json.
"""
import argparse
import copy
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from world import make_config, make_world  # noqa: E402
from world.circle_world import _draw_dots, wrap_delta  # noqa: E402

WORK_SEED, FINAL_SEED = 1001, 2001
COPIES, TICKS = 16, 2300
STARTS = tuple(range(200, 2076, 125))   # the regular starts of every copy; no start is before the first
WARM, HORIZON = 64, 32                  # real ticks shown before a start; dreamed ticks after it
LEADS = (1, 2, 4, 8, 16)                # the court's event starts: ticks before the event
REACH, HOLD = 8, 2.0                    # the reader looks this far (px); a thing found within HOLD px is held
WARM_K = (1, 2, 3, 4, 6, 8, 16)
HOME_SKIP = 60000
FREE_TICKS, FREE_PER_COPY = 200, 2
LEARN_TICKS = 8                         # the learning curve's dreams at every checkpoint
CHUNK = 256                             # windows per forward pass


# ------------------------------------------------------------------------------------------------- the exam
class Exam:
    """Recorded ticks of a world stepped with null moves: frames [E, T, S, S] (grey levels), body [E, T, 2] and, for
    scoring only, the world's state in `state` (name -> [E, T, ...]). keep: ticks at which a copy of the world is kept
    (for other dice). skip: ticks stepped before the recording starts. starts: the regular starts of every copy."""

    def __init__(self, env, seed=WORK_SEED, copies=COPIES, ticks=TICKS, skip=0, keep=(), starts=STARTS):
        self.env, self.seed, self.skip = dict(env), int(seed), int(skip)
        self.starts, self.first = tuple(starts), int(starts[0])            # the regular starts; no start before the first
        self.cfg = cfg = make_config(self.env, auto_reset=True)
        self.kind = getattr(cfg, "kind", "circle")
        if self.kind not in ("watch", "breakout"):
            raise ValueError("the exam knows the worlds 'watch' and 'breakout', not '%s'" % self.kind)
        w = make_world(copies, cfg, seed=self.seed)
        null = np.zeros((copies, 3))
        for _ in range(self.skip):
            w.step(null)
        self.E, self.T, self.S = E, T, S = int(copies), int(ticks), int(cfg.size)
        self.frames = np.zeros((E, T, S, S), np.uint8)
        self.body = np.zeros((E, T, 2), np.float32)
        self.state, self.worlds = {}, {}
        obs, info = w.observe(), None
        for t in range(T):
            if t:
                obs, _, _, info = w.step(null)
            self.frames[:, t], self.body[:, t] = obs["image"][..., 0], obs["drives"]
            for name, val in self._read(w, info).items():
                val = np.asarray(val)
                self.state.setdefault(name, np.zeros((E, T) + val.shape[1:], val.dtype))[:, t] = val
            if t in keep:
                self.worlds[t] = copy.deepcopy(w)
        if self.kind == "watch":
            self.rad, self.kinds, self.dot_val, self.chk = w.rad.copy(), w.kind.copy(), w.dot_val.copy(), w._chk.copy()
        self.mean = self.frames.mean((0, 1), dtype=np.float64).astype(np.float32)
        z = self.zones()
        c = self.counts(self.frames)                                   # [E, T, Z]
        self.count_lo, self.count_hi = c.min(1), c.max(1)              # [E, Z]: the real frames' range, per copy
        self.zone_names = [n for n, _, _ in z]

    def _read(self, w, info):
        n = w.num_envs
        if self.kind == "watch":
            return dict(pos=w.pos, dots=w.dot_pos, turned=info["gt_bounce"] if info else np.zeros(w.rad.shape, bool))
        f = np.zeros(n, bool)
        return dict(ball=w.ball, live=np.asarray(w.live, bool), paddle=w.paddle,
                    hit=np.asarray(info["gt_paddle_hit"], bool) if info else f,
                    miss=np.asarray(info["gt_hazard_click"], bool) if info else f)

    # ---- the things of the world, for the reader and for the floors
    @property
    def thing_kinds(self):
        return ("food", "poison") if self.kind == "watch" else ("ball", "paddle")

    def _dots_layer(self, e, t):
        """The frames with the dots only [M, S, S]."""
        c = self.cfg
        img = np.full((len(e), self.S, self.S, 3), c.bg, np.uint8)
        if c.dots:
            _draw_dots(img, self.state["dots"][e, t], self.dot_val[e], int(c.dot_size), np.array((c.bg,) * 3, np.uint8))
        return img[..., 0]

    def _court(self):
        c, S = self.cfg, self.S
        img = np.full((S, S), c.bg, np.uint8)
        img[:, :c.wall] = c.wall_grey
        img[:, S - c.wall:] = c.wall_grey
        img[:c.wall, :] = c.wall_grey
        return img

    def things(self, e, t):
        """The things at ticks (e [M], t [M]), each a dict: kind; shape [M, S, S] (the pixels it covers); paintable
        ([S, S]: where it paints when it covers a pixel; a checker disc shows the view's checker); paint (its grey
        level); B [M, S, S] (the real frame without it, the other things in their places); ok [M] (it is there, whole);
        wrap (a torus); axis ("xy", or "x": it moves along its row only); pos [M, 2] (px; for the floors)."""
        e, t = np.asarray(e), np.asarray(t)
        c, S, M = self.cfg, self.S, len(e)
        F = self.frames[e, t]
        out = []
        if self.kind == "watch":
            px = np.arange(S, dtype=np.float64) + 0.5
            pos, rad = self.state["pos"][e, t], self.rad[e]                           # [M, K, 2], [M, K]
            dx = wrap_delta(px[None, None, None, :], pos[..., 0, None, None], float(S))
            dy = wrap_delta(px[None, None, :, None], pos[..., 1, None, None], float(S))
            inside = dx * dx + dy * dy <= (rad ** 2)[..., None, None]                 # [M, K, S, S], as the world draws
            dots = self._dots_layer(e, t)
            for j, k in enumerate(self.kinds):
                paintable = self.chk if k == 1 else np.ones((S, S), bool)
                out.append(dict(kind=self.thing_kinds[int(k)], shape=inside[:, j], paintable=paintable, paint=float(c.grey),
                                B=np.where(inside[:, j] & paintable, dots, F), ok=np.ones(M, bool), wrap=True, axis="xy",
                                pos=pos[:, j]))
            return out
        st = self.state
        court = self._court()
        ball, live, pad = st["ball"][e, t], st["live"][e, t], st["paddle"][e, t]
        bx, by = np.rint(ball[:, 0]).astype(int), np.rint(ball[:, 1]).astype(int)
        px = np.rint(pad).astype(int)
        col, row = np.arange(S)[None, None, :], np.arange(S)[None, :, None]
        pshape = ((row >= c.paddle_y) & (row < c.paddle_y + c.paddle_h) & (col >= px[:, None, None])
                  & (col < px[:, None, None] + c.paddle_w))
        bshape = ((row >= by[:, None, None]) & (row < by[:, None, None] + c.ball) & (col >= bx[:, None, None])
                  & (col < bx[:, None, None] + c.ball) & live[:, None, None])
        whole = live & (bx >= 0) & (bx + c.ball <= S) & (by >= 0) & (by + c.ball <= S)
        with_pad = np.where(pshape, np.uint8(c.paddle_grey), court[None])
        with_ball = np.where(bshape, np.uint8(c.ball_grey), court[None])
        every = np.ones((S, S), bool)
        out.append(dict(kind="ball", shape=bshape, paintable=every, paint=float(c.ball_grey), B=with_pad, ok=whole,
                        wrap=False, axis="xy", pos=ball))
        out.append(dict(kind="paddle", shape=pshape, paintable=every, paint=float(c.paddle_grey), B=with_ball,
                        ok=np.ones(M, bool), wrap=False, axis="x", pos=np.stack([pad, np.zeros(M)], 1)))
        return out

    def compose(self, th):
        """The frame a thing's dict stands for: B with the thing painted in its true place (the real frame, a test)."""
        return np.where(th["shape"] & th["paintable"], np.uint8(th["paint"]), th["B"])

    def kind_masks(self, e, s):
        """Whose pixels a prediction of tick s + 1 is scored on: name -> [M, S, S], the pixels a kind of thing covers at
        tick s or s + 1; 'dots' (the disc world; where no disc is); 'rest'. things: [n, M, S, S], the same per thing."""
        a, b = self.things(e, s), self.things(e, np.asarray(s) + 1)
        per = np.stack([x["shape"] | y["shape"] for x, y in zip(a, b)])
        out = {k: np.zeros(per.shape[1:], bool) for k in self.thing_kinds}
        for th, m in zip(a, per):
            out[th["kind"]] |= m
        used = per.any(0)
        if self.kind == "watch":
            bg = self.cfg.bg
            out["dots"] = ((self._dots_layer(e, s) != bg) | (self._dots_layer(e, np.asarray(s) + 1) != bg)) & ~used
            used = used | out["dots"]
        out["rest"] = ~used
        return out, per

    def positions(self, e, t):
        """[n things, M, 2] px and ok [n, M]; kinds [n]."""
        e, t = np.asarray(e), np.asarray(t)
        if self.kind == "watch":
            p = self.state["pos"][e, t].transpose(1, 0, 2)
            return p, np.ones(p.shape[:2], bool), [self.thing_kinds[int(k)] for k in self.kinds]
        st = self.state
        pad = st["paddle"][e, t]
        p = np.stack([st["ball"][e, t], np.stack([pad, np.zeros(len(e))], 1)])
        return p, np.stack([st["live"][e, t], np.ones(len(e), bool)]), ["ball", "paddle"]

    def apart(self, a, b):
        """The distance in px between positions a and b [..., 2] (on the torus in the disc world)."""
        d = wrap_delta(a, b, float(self.S)) if self.kind == "watch" else a - b
        return np.hypot(d[..., 0], d[..., 1])

    # ---- valid frames: the count of bright pixels of each zone, within the real frames' range
    def zones(self):
        """[(name, mask [S, S], level)]: a pixel above the level is bright."""
        c, S = self.cfg, self.S
        if self.kind == "watch":
            return [("all", np.ones((S, S), bool), (c.bg + c.grey) / 2.0)]
        wall = np.zeros((S, S), bool)
        wall[:, :c.wall] = wall[:, S - c.wall:] = wall[:c.wall, :] = True
        rows = np.zeros((S, S), bool)
        rows[c.paddle_y:c.paddle_y + c.paddle_h] = True
        return [("walls", wall, (c.bg + c.wall_grey) / 2.0), ("paddle rows", rows & ~wall, (c.bg + c.paddle_grey) / 2.0),
                ("court", ~rows & ~wall, (c.bg + c.ball_grey) / 2.0)]

    def counts(self, frames):
        """[..., Z]: the bright pixels of each zone in frames [..., S, S]."""
        return np.stack([((frames > lv) & m).sum((-2, -1)) for _, m, lv in self.zones()], -1)

    def valid(self, frames, e):
        """[M, H]: every zone's count of frames [M, H, S, S] is one the real frames of copy e [M] reach."""
        c = self.counts(frames)
        return ((c >= self.count_lo[e][:, None]) & (c <= self.count_hi[e][:, None])).all(-1)

    # ---- the starts
    def regular(self):
        e = np.repeat(np.arange(self.E), len(self.starts))
        return e, np.tile(np.array(self.starts), self.E)

    def every_tick(self):
        s = np.arange(self.first, self.T - 1)
        return np.repeat(np.arange(self.E), len(s)), np.tile(s, self.E)

    def _in_turn(self, per, n):
        out = [(e, int(p[j])) for j in range(max([len(p) for p in per] + [0])) for e, p in enumerate(per) if j < len(p)][:n]
        return np.array([o[0] for o in out], int), np.array([o[1] for o in out], int)

    def event_starts(self, name, lead, n=64):
        """Up to n starts whose tick s + lead has the event (the court's 'hit' or 'miss'), the events taken in turn
        from the copies; the same events for every lead."""
        ev = self.state[name]
        per = [np.flatnonzero(ev[e]) for e in range(self.E)]
        e, te = self._in_turn([p[p >= self.first + max(LEADS)] for p in per], n)
        return e, te - lead

    def bounce_starts(self, n=128):
        """Starts whose next tick has a bounce (the disc world), taken in turn from the copies."""
        tr = self.state["turned"].any(2)
        per = [np.flatnonzero(tr[e]) - 1 for e in range(self.E)]
        return self._in_turn([p[p >= self.first] for p in per], n)

    def free_starts(self, n=128, step=250):
        """Starts with no bounce from 2 ticks before to 2 after: per copy the first such tick at or after
        the first start + 10 + step j."""
        tr = self.state["turned"].any(2)
        e_, s_ = [], []
        for e in range(self.E):
            for j in range(n // self.E):
                s = self.first + 10 + step * j
                while s + 3 < self.T and tr[e, s - 1:s + 3].any():
                    s += 1
                if s + 3 < self.T:
                    e_.append(e), s_.append(s)
        return np.array(e_, int), np.array(s_, int)


# ------------------------------------------------------------------------------------------------- the reader
def locate(F, th, reach=REACH):
    """Where frames F [M, S, S] have a thing (a dict of Exam.things): the integer shift (dx, dy) [M, 2] of the largest
    gain within reach, and that gain [M]. gain(d) is the sum over the pixels the thing paints when shifted by d of
    abs(F - B) - abs(F - paint): above 0, F is nearer to the thing drawn there than to the frame without it. Among
    equal gains the smallest shift wins. A thing that cannot leave the frame is not looked for outside it."""
    F, B = np.asarray(F, np.float64), th["B"].astype(np.float64)
    M, S = F.shape[0], F.shape[-1]
    G = (np.abs(F - B) - np.abs(F - th["paint"])) * th["paintable"]
    shape = th["shape"].astype(np.float64)
    C = np.fft.irfft2(np.fft.rfft2(G) * np.conj(np.fft.rfft2(shape)), s=(S, S))   # C[m, dy, dx] = sum G(x) shape(x - d)
    d = np.arange(-reach, reach + 1)
    dy, dx = [a.ravel() for a in np.meshgrid(d, d, indexing="ij")]
    if th["axis"] == "x":
        dx, dy = dx[dy == 0], dy[dy == 0]
    order = np.argsort(dx * dx + dy * dy, kind="stable")
    dx, dy = dx[order], dy[order]
    g = np.round(C[:, dy % S, dx % S], 3)
    if not th["wrap"]:
        xs, ys = th["shape"].any(1), th["shape"].any(2)                             # the columns and rows it covers
        x0, x1 = xs.argmax(1), S - 1 - xs[:, ::-1].argmax(1)
        y0, y1 = ys.argmax(1), S - 1 - ys[:, ::-1].argmax(1)
        out = ((x0[:, None] + dx < 0) | (x1[:, None] + dx > S - 1) | (y0[:, None] + dy < 0) | (y1[:, None] + dy > S - 1))
        g = np.where(out, -np.inf, g)
    best = g.argmax(1)
    return np.stack([dx[best], dy[best]], 1), g[np.arange(M), best]


def read_things(exam, e, s, fd, chunk=1024):
    """The reader on dreamed frames fd [M, H, S, S] of stretches (e, s): dist [n things, M, H] (px; nan: the thing is
    not there to read at that tick) and found [n, M, H]; kinds [n]."""
    M, H = fd.shape[:2]
    ee, tt = np.repeat(e, H), (np.asarray(s)[:, None] + 1 + np.arange(H)).ravel()
    F = fd.reshape(M * H, *fd.shape[2:])
    dist = found = kinds = None
    for i in range(0, M * H, chunk):
        ths = exam.things(ee[i:i + chunk], tt[i:i + chunk])
        if dist is None:
            kinds = [th["kind"] for th in ths]
            dist, found = np.full((len(ths), M * H), np.nan), np.zeros((len(ths), M * H), bool)
        for j, th in enumerate(ths):
            sh, gain = locate(F[i:i + chunk], th)
            dist[j, i:i + chunk] = np.where(th["ok"], np.hypot(sh[:, 0], sh[:, 1]), np.nan)
            found[j, i:i + chunk] = th["ok"] & (gain > 0)
    return dist.reshape(-1, M, H), found.reshape(-1, M, H), kinds


def floor_things(exam, e, s, H):
    """The floors' distances to the truth [n things, M, H] (nan: not there): linear (each thing kept moving as it moved
    from tick s - 1 to s) and copy (it stays where it was at s); kinds [n]."""
    e, s = np.asarray(e), np.asarray(s)
    p0, ok0, kinds = exam.positions(e, s - 1)
    p1, ok1, _ = exam.positions(e, s)
    step = wrap_delta(p1, p0, float(exam.S)) if exam.kind == "watch" else p1 - p0
    lin, cop = [], []
    for h in range(1, H + 1):
        p, ok, _ = exam.positions(e, s + h)
        ok = ok & ok0 & ok1
        lin.append(np.where(ok, exam.apart(p1 + h * step, p), np.nan))
        cop.append(np.where(ok, exam.apart(p1, p), np.nan))
    return np.stack(lin, -1), np.stack(cop, -1), kinds


def seen_things(exam, e, s, H, k):
    """A fairer floor (added after the first results): the straight line through each thing's place as it can be read
    off the frames of ticks s - k + 1 and s, not through its true places. In the court a thing is drawn at whole pixels,
    so two frames give its speed to the pixel only; in the disc world a disc's place is read as the centroid of the
    pixels it covers. Returns the distances [n things, M, H] to the true places (nan: not there) and kinds [n]."""
    e, s = np.asarray(e), np.asarray(s)

    def seen(t):
        p, ok, kinds = exam.positions(e, t)
        if exam.kind == "breakout":
            return np.rint(p), ok, kinds
        S, out = exam.S, []
        ang = 2.0 * np.pi * (np.arange(S) + 0.5) / S
        for th in exam.things(e, t):                                              # the centroid on the torus
            m = th["shape"]
            x = np.angle((m.sum(1) * np.exp(1j * ang)).sum(1)) / (2.0 * np.pi) * S
            y = np.angle((m.sum(2) * np.exp(1j * ang)).sum(1)) / (2.0 * np.pi) * S
            out.append(np.stack([np.mod(x, S), np.mod(y, S)], 1))
        return np.stack(out), ok, kinds
    p0, ok0, kinds = seen(s - k + 1)
    p1, ok1, _ = seen(s)
    step = (wrap_delta(p1, p0, float(exam.S)) if exam.kind == "watch" else p1 - p0) / (k - 1)
    out = []
    for h in range(1, H + 1):
        p, ok, _ = exam.positions(e, s + h)
        out.append(np.where(ok & ok0 & ok1, exam.apart(p1 + h * step, p), np.nan))
    return np.stack(out, -1), kinds


SEEN = (2, 4, 8)   # the frames a seen straight line is drawn through


# ------------------------------------------------------------------------------------------------- the models
class Copy:
    """The floor: the last real frame and body, held."""
    name = "copy"

    def reset(self, n):
        self.f = self.b = None

    def observe(self, frame, body, move):
        self.f, self.b = np.asarray(frame), np.asarray(body, np.float32)

    def dream(self, H, moves=None, frames=True):
        return np.repeat(self.f[:, None], H, 1), np.repeat(self.b[:, None], H, 1)


class MeanFrame(Copy):
    """The floor a blurred dream ends at: the mean of the exam's real frames, per pixel; the body held."""
    name = "mean frame"

    def __init__(self, exam):
        self.mean = exam.mean

    def dream(self, H, moves=None, frames=True):
        n = len(self.b)
        return np.broadcast_to(self.mean, (n, H) + self.mean.shape).copy(), np.repeat(self.b[:, None], H, 1)


class TheWorld:
    """The ceiling: the recorded continuation."""
    name = "the world"

    def dream_from(self, exam, e, s, H):
        t = np.asarray(s)[:, None] + 1 + np.arange(H)
        return exam.frames[np.asarray(e)[:, None], t], exam.body[np.asarray(e)[:, None], t]


class OtherDice:
    """The ceiling where the world has chance: the world copied at the start and stepped with another chance. Needs the
    exam's kept worlds (Exam(keep=STARTS)); every copy of a kept world starts at the same tick."""
    name = "other dice"

    def __init__(self, seed=7):
        self.seed = seed

    def dream_from(self, exam, e, s, H):
        e, s = np.asarray(e), np.asarray(s)
        fd, bd = np.zeros((len(e), H, exam.S, exam.S), np.uint8), np.zeros((len(e), H, 2), np.float32)
        for t in np.unique(s):
            w = copy.deepcopy(exam.worlds[int(t)])
            w.rng = np.random.default_rng([self.seed, int(t)])
            null, sel = np.zeros((w.num_envs, 3)), s == t
            for h in range(H):
                obs = w.step(null)[0]
                fd[sel, h], bd[sel, h] = obs["image"][e[sel], ..., 0], obs["drives"][e[sel]]
        return fd, bd


class Pieces:
    """The universal model of a run's checkpoint through the contract: one transformer, or the pieces of
    online.pieces. The window is the last ctx real ticks. A dreamed tick: its body is the mean of the pieces' expected
    next body (the move token's); each piece decodes its own block, the blocks make one frame, and that frame is the
    input of every piece. The decoding starts from the previous frame and is repeated until the frame stops changing,
    which is decoding token by token exactly (every piece's tokens follow the frame's row order, so a token's inputs
    of the same tick come from earlier rows). decode: 'pair' (the most probable bin and its neighbour; sharp), 'mean'
    (the expected value) or 'mode'. passes: the decoding passes of each dreamed tick.

    decode 'instinct' (added after the first results) is the way the agent's instinct and vm-00114's lookahead dream
    (online.uni_lookahead): the expected value, 2 passes and no more, and each piece alone: it replaces its own block
    and sees the rest of the frame as the last real one for the whole dream. The frame given back is the pieces' own
    blocks put together. 'instinct_pair' is the same way with pair (online.instinct_decode = pair)."""

    def __init__(self, run, step, decode="pair"):
        import agent_lib
        import cuda_lib
        from lifetime import ckpt_name
        from online import DECODE, uni_frame
        info = json.load(open(os.path.join(run, "info.json")))
        self.S = int(info.get("pieces", 1))
        self.agents = [agent_lib.Agent.load(os.path.join(run, ckpt_name(step, k, self.S))) for k in range(self.S)]
        self.models = [cuda_lib.CudaModel(a) for a in self.agents]
        self.alone = decode in ("instinct", "instinct_pair")
        self.name, self._frame = "%s %s" % (os.path.basename(run.rstrip("/")), decode), uni_frame
        self.mode = DECODE[dict(instinct="mean", instinct_pair="pair").get(decode, decode)]
        self.L = self.models[0].config["ctx"]
        self.blocks, patches = [], 0
        for m in self.models:
            c = m.config
            lay = agent_lib.uni_layout(c)
            patches += len(lay["patch"])
            if c.get("band_n", 0) >= 2:
                r0, r1, c0, c1 = agent_lib.band_piece(c)
                ps = c["patch"]
                blk = (slice(r0 * ps, (r1 + 1) * ps), slice(c0 * ps, (c1 + 1) * ps), (r1 - r0 + 1, c1 - c0 + 1))
            else:
                blk = (slice(None), slice(None), None)
            self.blocks.append(blk + (lay["patch"] - 1,))   # the tokens that predict the patches
        self.max_passes = patches + 1
        self.params = int(sum(a.num_params for a in self.agents))
        self.passes, self.unsettled = [], 0
        self.guess = None   # a test's hook: frames [n, S, S] every dreamed tick's decoding starts from, not the last frame

    def reset(self, n):
        self.n, self.fr, self.bd, self.mv = int(n), [], [], []

    def observe(self, frame, body, move):
        self.fr.append(np.asarray(frame, np.uint8)), self.bd.append(np.asarray(body, np.float32))
        self.mv.append(np.asarray(move, np.float32))
        del self.fr[:-self.L], self.bd[:-self.L], self.mv[:-self.L]

    def _pass(self, wi, wd, wa):
        """One pass of every piece on windows [n, T, ...]: the last tick's frame as the pieces predict it from the
        window as it is, and the next body."""
        n, T = wi.shape[:2]
        we = np.zeros((n, T), np.uint8)
        new, nb = wi[:, -1].copy(), np.zeros((n, 2), np.float64)
        for m, (rs, cs, grid, pp) in zip(self.models, self.blocks):
            for i in range(0, n, CHUNK):
                o = m.forward_uni_step(wi[i:i + CHUNK], wd[i:i + CHUNK], wa[i:i + CHUNK], we[i:i + CHUNK], T - 1,
                                       mode=self.mode)
                new[i:i + CHUNK, rs, cs] = self._frame(m.config, o["values_t"][:, pp], grid)
                nb[i:i + CHUNK] += o["next_body"][:, T - 1, :2]
        return new, np.clip(nb / len(self.models), 0.0, 1.0).astype(np.float32)

    def _dream_alone(self, wi, wd, wa, mv, H):
        """The instinct's dream: each piece on its own window, its block decoded by 2 passes from the last real frame,
        the other blocks left as the last real frame; the body each piece's own expected next body."""
        n, L = wi.shape[0], self.L
        last = wi[:, -1:].copy()
        fd, bd = np.zeros((n, H) + wi.shape[2:4], np.uint8), np.zeros((n, H, 2), np.float32)
        for m, (rs, cs, grid, pp) in zip(self.models, self.blocks):
            xi, xd, xa = wi.copy(), wd.copy(), wa.copy()
            for h in range(H):
                T = xi.shape[1]
                we = np.zeros((n, T), np.uint8)
                nb = np.concatenate([m.forward_uni_step(xi[i:i + CHUNK], xd[i:i + CHUNK], xa[i:i + CHUNK], we[i:i + CHUNK], T - 1,
                                                        mode=self.mode)["next_body"][:, T - 1, :2] for i in range(0, n, CHUNK)])
                nb = np.clip(nb, 0.0, 1.0)
                bd[:, h] += nb / len(self.models)
                xi = np.concatenate([xi, last], 1)[:, -L:]
                xd = np.concatenate([xd, nb[:, None]], 1)[:, -L:]
                xa = np.concatenate([xa, mv[:, h:h + 1]], 1)[:, -L:]
                T = xi.shape[1]
                we = np.zeros((n, T), np.uint8)
                for _ in range(2):
                    for i in range(0, n, CHUNK):
                        o = m.forward_uni_step(xi[i:i + CHUNK], xd[i:i + CHUNK], xa[i:i + CHUNK], we[i:i + CHUNK], T - 1, mode=self.mode)
                        xi[i:i + CHUNK, -1, rs, cs] = self._frame(m.config, o["values_t"][:, pp], grid)
                fd[:, h, rs, cs] = xi[:, -1, rs, cs, 0]
        self.passes += [2] * H
        return fd, bd

    def dream(self, H, moves=None, frames=True):
        n, L = self.n, self.L
        if not self.fr:
            raise ValueError("dream needs at least one observed tick")
        wi = np.repeat(np.stack(self.fr, 1)[..., None], 3, -1)
        wd, wa = np.stack(self.bd, 1), np.stack(self.mv, 1)
        mv = np.zeros((n, H, 3), np.float32) if moves is None else np.asarray(moves, np.float32)
        if self.alone:
            return self._dream_alone(wi, wd, wa, mv, H)
        nb = self._pass(wi, wd, wa)[1]
        fd = np.zeros((n, H) + wi.shape[2:4], np.uint8)
        bd = np.zeros((n, H, 2), np.float32)
        for h in range(H):
            bd[:, h] = nb
            if not frames and h == H - 1:
                break
            first = wi[:, -1:] if self.guess is None else np.repeat(self.guess[:, None, ..., None], 3, -1)
            wi = np.concatenate([wi, first], 1)[:, -L:]                     # the new tick's frame starts as the last one
            wd = np.concatenate([wd, nb[:, None]], 1)[:, -L:]
            wa = np.concatenate([wa, mv[:, h:h + 1]], 1)[:, -L:]
            k, same = 0, np.zeros(n, bool)
            while k < self.max_passes and not same.all():   # the whole batch every pass: a batch that shrank as its
                new, nb = self._pass(wi, wd, wa)            # windows settled would not repeat the same last bits, and
                same = (new == wi[:, -1]).all((1, 2, 3))    # a value at a rounding edge would flip for ever
                wi[:, -1] = new
                k += 1
            self.passes.append(k)
            self.unsettled += int((~same).sum())
            fd[:, h] = wi[:, -1, ..., 0]
        return fd, bd

    def close(self):
        for m in self.models:
            m.close()
        for a in self.agents:
            a.close()


def dreams(model, exam, e, s, H, warm=WARM, frames=True):
    """The model's dream of H ticks from each stretch (e, s) after `warm` real ticks: frames [M, H, S, S] (None with
    frames=False) and bodies [M, H, 2]. A ceiling reads the exam itself (dream_from)."""
    e, s = np.asarray(e), np.asarray(s)
    if hasattr(model, "dream_from"):
        return model.dream_from(exam, e, s, H)
    if len(s) and s.min() - warm + 1 < 0:
        raise ValueError("a start has fewer than %d ticks before it" % warm)
    fs, bs = [], []
    for i in range(0, len(e), CHUNK):
        ee, ss = e[i:i + CHUNK], s[i:i + CHUNK]
        null = np.zeros((len(ee), 3), np.float32)
        model.reset(len(ee))
        for j in range(warm - 1, -1, -1):
            model.observe(exam.frames[ee, ss - j], exam.body[ee, ss - j], null)
        f, b = model.dream(H, frames=frames) if not frames else model.dream(H)
        fs.append(f), bs.append(b)
    return (np.concatenate(fs) if frames else None), np.concatenate(bs)


def real(exam, e, s, H):
    return TheWorld().dream_from(exam, e, s, H)


# ------------------------------------------------------------------------------------------------- the numbers
def sq(a, b, mask=None):
    """The squared differences in grey levels of frames a and b [..., S, S], summed over the frame (or a mask)."""
    d = (np.asarray(a, np.float32) - np.asarray(b, np.float32)) ** 2
    return (d if mask is None else d * mask).sum((-2, -1), dtype=np.float64)


def skill(err, floor):
    """1 - the summed error / the floor's summed error (nan where the floor has none)."""
    err, floor = np.sum(err, 0), np.sum(floor, 0)
    return np.where(floor > 0, 1.0 - err / np.where(floor > 0, floor, 1.0), np.nan)


def horizon(ok):
    """The number of ticks from the first on during which ok [H] holds (the whole length: it always does)."""
    bad = np.flatnonzero(~np.asarray(ok, bool))
    return int(bad[0]) if len(bad) else len(ok)


def auc(pos, neg):
    """The chance that a positive scores above a negative; ties count half. nan without both."""
    pos, neg = np.asarray(pos, np.float64), np.asarray(neg, np.float64)
    if not len(pos) or not len(neg):
        return float("nan")
    _, inv, cnt = np.unique(np.concatenate([pos, neg]), return_inverse=True, return_counts=True)
    rank = (np.cumsum(cnt) - (cnt - 1) / 2.0)[inv]
    return float((rank[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2.0) / (len(pos) * len(neg)))


def grey_w1(a, b):
    """The mean absolute difference between the sorted grey levels of two sets of frames, in grey levels."""
    ha = np.bincount(np.clip(np.rint(a), 0, 255).astype(np.uint8).ravel(), minlength=256) / a.size
    hb = np.bincount(np.clip(np.rint(b), 0, 255).astype(np.uint8).ravel(), minlength=256) / b.size
    return float(np.abs(np.cumsum(ha) - np.cumsum(hb)).sum())


def rmse(err, pixels):
    return float(np.sqrt(np.sum(err) / max(float(np.sum(pixels)), 1.0)))


def zone_counts(exam, e, fd, fr):
    """By zone and dreamed tick: the share of dreamed frames fd [M, H, S, S] whose count of bright pixels is one the
    same copy's real frames reach, and the mean counts of the dreamed and of the real frames fr (a diagnosis of what
    makes a frame invalid; added after the first lives were scored)."""
    cd, cr = exam.counts(fd), exam.counts(fr)                                       # [M, H, Z]
    ok = (cd >= exam.count_lo[e][:, None]) & (cd <= exam.count_hi[e][:, None])
    return {n: dict(valid=ok[..., z].mean(0).tolist(), count=cd[..., z].mean(0).tolist(), real=cr[..., z].mean(0).tolist())
            for z, n in enumerate(exam.zone_names)}


def next_frame(exam, e, s, f1, fr1):
    """The first dreamed tick f1 [M, S, S] against the real one fr1: the skill over the frame and by kind of thing
    (with the root mean squared errors in grey levels; where the copy has no error the skill is nan)."""
    last = exam.frames[e, s]
    masks, _ = exam.kind_masks(e, s)
    out = dict(skill=float(skill(sq(f1, fr1), sq(last, fr1))), rmse=rmse(sq(f1, fr1), fr1.size),
               copy_rmse=rmse(sq(last, fr1), fr1.size), by={})
    for k, m in masks.items():
        em, ec = sq(f1, fr1, m), sq(last, fr1, m)
        out["by"][k] = dict(skill=float(skill(em, ec)), rmse=rmse(em, m.sum()), copy_rmse=rmse(ec, m.sum()),
                            pixels=float(m.sum() / len(e)))
    return out


def hold_summary(dist, found, kinds):
    """Per kind of thing and dreamed tick: the shares held (found within HOLD px) and lost, the median distance of the
    found, and the horizon (the ticks from the first on with at least half held). found None: a floor, never lost."""
    out = {}
    for k in dict.fromkeys(kinds):
        sel = [i for i, x in enumerate(kinds) if x == k]
        d = dist[sel].reshape(-1, dist.shape[-1])
        there = ~np.isnan(d)
        f = there if found is None else found[sel].reshape(-1, dist.shape[-1]) & there
        n = np.maximum(there.sum(0), 1)
        held = (f & (np.nan_to_num(d, nan=1e9) <= HOLD)).sum(0) / n
        med = [float(np.median(d[:, h][f[:, h]])) if f[:, h].any() else float("nan") for h in range(d.shape[1])]
        out[k] = dict(held=held.tolist(), lost=((there & ~f).sum(0) / n).tolist(), dist=med, horizon=horizon(held >= 0.5),
                      n=int(there[:, 0].sum()))
    return out


def regular_part(model, exam, H=HORIZON, hold=True):
    """Measures 1, 4, 5 and 6 on the regular starts. Returns the summary and the arrays by stretch."""
    e, s = exam.regular()
    fr, _ = real(exam, e, s, H)
    fd, bd = dreams(model, exam, e, s, H)
    fc, _ = dreams(Copy(), exam, e, s, H)
    fm, _ = dreams(MeanFrame(exam), exam, e, s, H)
    err, ec, em = sq(fd, fr), sq(fc, fr), sq(fm, fr)                                    # [M, H]
    out = dict(starts=len(e), horizon_ticks=H, next=next_frame(exam, e, s, fd[:, 0], fr[:, 0]),
               curve=dict(skill_copy=skill(err, ec).tolist(), skill_mean=skill(err, em).tolist(),
                          rmse=np.sqrt(err.mean(0) / exam.S ** 2).tolist(),
                          copy_rmse=np.sqrt(ec.mean(0) / exam.S ** 2).tolist(),
                          mean_rmse=np.sqrt(em.mean(0) / exam.S ** 2).tolist(),
                          horizon=horizon((err.sum(0) < ec.sum(0)) & (err.sum(0) < em.sum(0)))),
               real=dict(w1=[grey_w1(fd[:, h], fr[:, h]) for h in range(H)], valid=exam.valid(fd, e).mean(0).tolist(),
                         zones=zone_counts(exam, e, fd, fr)))
    arrays = dict(err=err.astype(np.float32), err_copy=ec.astype(np.float32), err_mean=em.astype(np.float32),
                  body=bd.astype(np.float32))
    if hold:
        dist, found, kinds = read_things(exam, e, s, fd)
        lin, cop, _ = floor_things(exam, e, s, H)
        out["hold"] = hold_summary(dist, found, kinds)
        out["hold_linear"], out["hold_copy"] = hold_summary(lin, None, kinds), hold_summary(cop, None, kinds)
        out["hold_seen"] = {str(k): hold_summary(seen_things(exam, e, s, H, k)[0], None, kinds) for k in SEEN}
        arrays.update(dist=dist.astype(np.float32), found=found)
    return out, arrays, (e, s, fd, bd)


def court_next_tick(exam, e, s, pred):
    """The next-tick forecast of the court's events from the predicted next hunger pred [M] at starts (e, s): for
    hits the score is the predicted fall, for misses the predicted rise; the area under the curve of the ticks with the
    event against the ticks without, and the summed predicted change / the summed real change on the event's ticks."""
    h0, h1 = exam.body[e, s, 0], exam.body[e, s + 1, 0]
    change, out = pred - h0, {}
    for name, sign in (("hit", -1.0), ("miss", 1.0)):
        ev = exam.state[name][e, s + 1]
        realc = (h1 - h0)[ev].sum()
        out[name] = dict(auc=auc(sign * change[ev], sign * change[~ev]), events=int(ev.sum()),
                         size=float(change[ev].sum() / realc) if ev.any() and realc != 0 else float("nan"))
    return out


def jump(h0, bd, sign, h):
    """The largest fall (sign -1) or rise (+1) between two successive ticks of the dreamed hunger bd [M, H'] within
    its first h ticks, from the real hunger h0 [M]."""
    seq = np.concatenate([h0[:, None], bd[:, :h]], 1)
    return (sign * np.diff(seq, axis=1)).max(1)


def events_part(model, exam, reg=None, within=True, keep=None):
    """Measure 2. The court: next tick over every tick, and (within) within h ticks (event starts with the lead h
    against the regular starts without the event within h ticks; reg: their dreamed bodies, from regular_part). The disc
    world: the next-frame skill and the distance on the discs that turn against discs in free flight, beside linear."""
    if exam.kind == "breakout":
        e, s = exam.every_tick()
        _, b = dreams(model, exam, e, s, 1, frames=False)
        out = dict(next=court_next_tick(exam, e, s, b[:, 0, 0]))
        if keep is not None:                                                        # the forecasts by tick, for later readings
            keep.update(next_e=e.astype(np.int16), next_s=s.astype(np.int16), next_hunger=b[:, 0, 0].astype(np.float32))
        if not within:
            return out
        out["within"] = {}
        re_, rs_ = exam.regular()
        rb = dreams(model, exam, re_, rs_, max(LEADS))[1] if reg is None else reg
        for name, sign in (("hit", -1.0), ("miss", 1.0)):
            out["within"][name] = {}
            for d in LEADS:
                ee, es = exam.event_starts(name, d)
                _, eb = dreams(model, exam, ee, es, d, frames=False)                   # the last tick's frame is not needed
                later = np.stack([exam.state[name][re_, rs_ + j] for j in range(1, d + 1)]).any(0)
                pos = jump(exam.body[ee, es, 0], eb[..., 0], sign, d)
                neg = jump(exam.body[re_, rs_, 0], rb[..., 0], sign, d)[~later]
                out["within"][name][str(d)] = dict(auc=auc(pos, neg), events=len(ee), without=int((~later).sum()))
        return out
    out = {}
    for name, (e, s) in (("bounce", exam.bounce_starts()), ("free", exam.free_starts())):
        fr, _ = real(exam, e, s, 1)
        fd, _ = dreams(model, exam, e, s, 1)
        last = exam.frames[e, s]
        _, per = exam.kind_masks(e, s)                                              # [n, M, S, S]
        dist, found, kinds = read_things(exam, e, s, fd)
        lin, cop, _ = floor_things(exam, e, s, 1)
        turned = exam.state["turned"][e, s + 1].T                                   # [n, M]
        use = turned if name == "bounce" else np.ones_like(turned)
        out[name] = dict(starts=len(e))
        for k in dict.fromkeys(kinds):
            sel = np.array([x == k for x in kinds])[:, None] & use                   # [n, M]: the things scored
            m = per & sel[..., None, None]
            em = sum(sq(fd[:, 0], fr[:, 0], m[j]) for j in range(len(kinds)))
            ec = sum(sq(last, fr[:, 0], m[j]) for j in range(len(kinds)))
            d_, f_ = dist[..., 0][sel], found[..., 0][sel]
            out[name][k] = dict(things=int(sel.sum()), skill=float(skill(em, ec)), held=float((f_ & (d_ <= HOLD)).mean()),
                                lost=float((~f_).mean()), dist=float(np.mean(d_[f_])) if f_.any() else float("nan"),
                                linear_dist=float(lin[..., 0][sel].mean()), linear_held=float((lin[..., 0][sel] <= HOLD).mean()),
                                copy_dist=float(cop[..., 0][sel].mean()), copy_held=float((cop[..., 0][sel] <= HOLD).mean()))
    return out


def warm_part(model, exam):
    """Measure 9: from a reset, shown the last k real ticks: the next-frame skill of the regular starts and, in the
    court, the next-tick forecast of the event starts with lead 1 against the regular starts."""
    e, s = exam.regular()
    fr, _ = real(exam, e, s, 1)
    last = exam.frames[e, s]
    out = {}
    for k in WARM_K:
        fd, bd = dreams(model, exam, e, s, 1, warm=k)
        out[str(k)] = dict(skill=float(skill(sq(fd[:, 0], fr[:, 0]), sq(last, fr[:, 0]))))
        if exam.kind == "breakout":
            for name, sign in (("hit", -1.0), ("miss", 1.0)):
                ee, es = exam.event_starts(name, 1)
                _, eb = dreams(model, exam, ee, es, 1, warm=k, frames=False)
                none = ~exam.state[name][e, s + 1]
                out[str(k)][name] = auc(jump(exam.body[ee, es, 0], eb[..., 0], sign, 1),
                                        jump(exam.body[e, s, 0], bd[..., 0], sign, 1)[none])
    return out


def warm_dream_part(model, exam, ks=(2, 4, 8), H=8):
    """Added after the first results: does the dream use the whole window? From a reset, shown the last k real ticks,
    the regular starts are dreamed for H ticks: the pixel skill and the share of each kind of thing held, by tick."""
    e, s = exam.regular()
    fr, _ = real(exam, e, s, H)
    ec = sq(np.repeat(exam.frames[e, s][:, None], H, 1), fr)
    out = {}
    for k in ks:
        fd, _ = dreams(model, exam, e, s, H, warm=k)
        dist, found, kinds = read_things(exam, e, s, fd)
        hold = hold_summary(dist, found, kinds)
        out[str(k)] = dict(skill=skill(sq(fd, fr), ec).tolist(), held={a: v["held"] for a, v in hold.items()},
                           horizon={a: v["horizon"] for a, v in hold.items()})
    return out


def free_part(model, exam):
    """Measure 6: a dream of FREE_TICKS ticks from the first FREE_PER_COPY regular starts of every copy: the first
    tick with an invalid frame (FREE_TICKS + 1: none), its median; the share of valid frames by tick."""
    e, s = exam.regular()
    keep = np.tile(np.arange(len(exam.starts)) < FREE_PER_COPY, exam.E)
    e, s = e[keep], s[keep]
    fd, _ = dreams(model, exam, e, s, FREE_TICKS)
    fr, _ = real(exam, e, s, FREE_TICKS)
    ok = exam.valid(fd, e)
    first = np.where(ok.all(1), FREE_TICKS + 1, (~ok).argmax(1) + 1)
    return dict(starts=len(e), first_invalid=float(np.median(first)), valid=ok.mean(0).tolist(),
                zones=zone_counts(exam, e, fd, fr))   # by zone: what the dream keeps (the ball is the court's zone)


def score(run, step, decode="pair", which="work", parts=("regular", "events", "warm"), H=HORIZON, out_dir=None, exam=None,
          name=None):
    """The measures of one checkpoint on one exam set. Writes (and adds to) RUN/wm/step_<step>_<decode>_<which>.json and
    .npz; returns the summary. exam: a ready one (a test's small one). name: the set's name in the file, if not `which`
    (the learning curve's shorter run on the working set is kept apart as 'learn')."""
    info = json.load(open(os.path.join(run, "info.json")))
    env = info["config"]["env"]
    t0 = time.time()
    if exam is not None:
        pass
    elif which == "home":
        exam = Exam(env, seed=info["seed"], copies=1, skip=HOME_SKIP)
    else:
        exam = Exam(env, seed=dict(work=WORK_SEED, final=FINAL_SEED)[which])
    t_exam = time.time() - t0
    model = Pieces(run, step, decode)
    out_dir = out_dir or os.path.join(run, "wm")
    os.makedirs(out_dir, exist_ok=True)
    base = os.path.join(out_dir, "step_%08d_%s_%s" % (step, decode, name or which))
    res = json.load(open(base + ".json")) if os.path.exists(base + ".json") else {}
    res.update(run=os.path.basename(run.rstrip("/")), seed=info["seed"], step=step, decode=decode, set=which,
               world=exam.kind, params=model.params)
    reg, sec = None, res.setdefault("seconds", {})
    sec["exam"] = round(t_exam, 1)
    un = res["unsettled"] = res["unsettled"] if isinstance(res.get("unsettled"), dict) else {}
    for part in parts:
        t0, model.passes, model.unsettled = time.time(), [], 0
        if part == "regular":
            res["regular"], arrays, (_, _, _, bd) = regular_part(model, exam, H)
            reg = bd
            np.savez_compressed(base + ".npz", **arrays)
            res["regular"]["passes"] = float(np.mean(model.passes))
        elif part in ("events", "next"):   # next: the next tick only (the court's dreams toward an event left out)
            kept = {}
            res["events"] = events_part(model, exam, reg if reg is not None and reg.shape[1] >= max(LEADS) else None,
                                        within=part == "events", keep=kept)
            if kept:
                np.savez_compressed(base + "_events.npz", **kept)
        elif part == "warm":
            res["warm"] = warm_part(model, exam)
        elif part == "warmdream":
            res["warm_dream"] = warm_dream_part(model, exam)
        elif part == "free":
            res["free"] = free_part(model, exam)
        else:
            raise ValueError("unknown part '%s'" % part)
        sec[part] = round(time.time() - t0, 1)
        un[part] = int(model.unsettled)   # frames whose decoding did not settle within the passes allowed
    model.close()
    with open(base + ".json.tmp", "w") as f:
        json.dump(res, f)
    os.replace(base + ".json.tmp", base + ".json")   # in one step: a reader never meets half a file
    return res


def anchors(env, which="work", H=HORIZON, exam=None):
    """The floors and the ceilings on an exam set, with the same numbers as a model's: copy, the mean frame, the world
    itself and, in the court, other dice (exam: a ready one, with its worlds kept at the regular starts)."""
    if exam is None:
        exam = Exam(env, seed=dict(work=WORK_SEED, final=FINAL_SEED)[which],
                    keep=STARTS if make_config(env).kind == "breakout" else ())
    out = dict(world=exam.kind, set=which, starts=len(exam.starts) * exam.E,
               facts=dict(ticks=exam.T, copies=exam.E, zones=exam.zone_names, count_lo=exam.count_lo.tolist(),
                          count_hi=exam.count_hi.tolist(), count_mean=exam.counts(exam.frames).mean((0, 1)).tolist()))
    if exam.kind == "breakout":
        out["facts"].update(hits=int(exam.state["hit"][:, exam.first:].sum()), misses=int(exam.state["miss"][:, exam.first:].sum()))
    else:
        out["facts"].update(bounce_ticks=int(exam.state["turned"].any(2)[:, exam.first:].sum()))
    models = [Copy(), MeanFrame(exam), TheWorld()] + ([OtherDice()] if exam.kind == "breakout" else [])
    for m in models:
        r, _, (_, _, _, bd) = regular_part(m, exam, H)
        if not (exam.kind == "breakout" and isinstance(m, OtherDice)):   # the event starts have no kept worlds
            r["events"] = events_part(m, exam, bd)
        out[m.name] = r
    return out


# ------------------------------------------------------------------------------------------------- the tables
REPORT = (1, 2, 4, 8, 16, 32)   # the dreamed ticks shown


def _stat(v):
    """The mean of the seeds' values and its 95% half-width (nan with fewer than 2)."""
    from analyze_misses import T95
    v = np.asarray([x for x in v if x is not None and x == x], np.float64)
    if len(v) < 2:
        return (float(v.mean()) if len(v) else float("nan")), float("nan")
    return float(v.mean()), float(T95[len(v)] * v.std(ddof=1) / np.sqrt(len(v)))


def _get(r, *path):
    """r[path[0]][path[1]]..., nan where a key or a tick is not there."""
    for k in path:
        if isinstance(k, int):
            r = r[k] if isinstance(r, list) and k < len(r) else None
        else:
            r = r[k] if isinstance(r, dict) and k in r else None
        if r is None:
            return float("nan")
    return r


def _cut(r, H):
    """A summary of a full run as its first H dreamed ticks alone would give it: the horizons at most H."""
    r = copy.deepcopy(r)
    g = r["regular"]
    g["curve"]["horizon"] = min(g["curve"]["horizon"], H)
    for k in g.get("hold", {}):
        g["hold"][k]["horizon"] = min(g["hold"][k]["horizon"], H)
    return r


def load(out, arm, step, decode, which, seeds):
    """seed -> the summary of that life's checkpoint, where it is scored."""
    res = {}
    for sd in seeds:
        p = os.path.join(out, "%s_s%d" % (arm, sd), "wm", "step_%08d_%s_%s.json" % (step, decode, which))
        if os.path.exists(p):
            res[sd] = json.load(open(p))
    return res


def after_a_turn(out, arm, seeds, step=60000, early=2, at=4, exam=None):
    """Added after the results (the disc world): a bounce moves a disc only about .4 px off its line at the next tick,
    too little to show what the model makes of it. Here the regular starts' discs are split by when they first turn in
    the dreamed stretch: within the first `early` ticks, or not within the first `at`; and the share held at tick `at` is
    read for the dream (`pair`, from the saved distances) and for each disc kept moving as it last moved.
    Returns rows (kind, group, discs, the dream's held shares by seed, linear's held share)."""
    P = load(out, arm, step, "pair", "work", seeds)
    exam = exam or Exam(json.load(open(os.path.join(out, "%s_s%d" % (arm, next(iter(P))), "info.json")))["config"]["env"])
    e, s = exam.regular()
    turned = np.stack([exam.state["turned"][e, s + h] for h in range(1, at + 1)], -1)      # [M, K, at]
    first = np.where(turned.any(-1), turned.argmax(-1) + 1, 10 ** 6).T                      # [K, M]: the first tick it turns
    lin, _, kinds = floor_things(exam, e, s, at)
    rows = []
    for k in dict.fromkeys(kinds):
        sel = np.array([x == k for x in kinds])[:, None]
        for label, grp in (("turns within the first %d ticks" % early, first <= early), ("does not turn in the first %d" % at, first > at)):
            m = sel & grp
            held = []
            for sd in sorted(P):
                z = np.load(os.path.join(out, "%s_s%d" % (arm, sd), "wm", "step_%08d_pair_work.npz" % step))
                held.append(float((z["found"][..., at - 1] & (z["dist"][..., at - 1] <= HOLD))[m].mean()))
            rows.append((k, label, int(m.sum()), held, float((lin[..., at - 1] <= HOLD)[m].mean())))
    return rows


def tables(out, arm, seeds, step=60000, steps=(0, 10000, 20000, 30000, 40000, 50000, 60000), exam=None):
    """The README's tables of one arm (a world): the board with the floors and the ceilings, then each measure. exam: the
    exam the lives were scored on, if not the working set (a test's small one)."""
    from analyze_lives import f1, f2, f3, table

    def pm(rs, *path, fmt=f3):
        m, h = _stat([_get(r, *path) for r in rs.values()])
        return fmt(m) + (" ± " + fmt(h) if h == h else "")

    def pdiff(a, b, fmt=f3):
        """The cells of a paired difference a - b by seed; dashes with fewer than 2 seeds that have both."""
        from analyze_lives import diff
        ok = [x == x and y == y for x, y in zip(a, b)]
        return diff(a, b, fmt) if sum(ok) > 1 else ["–", "–"]

    def one(r, *path, fmt=f3, top=None):
        v = _get(r, *path)
        return ("%d+" % v) if top is not None and v == top else fmt(v)

    P, Mn = load(out, arm, step, "pair", "work", seeds), load(out, arm, step, "mean", "work", seeds)
    In, Ip = load(out, arm, step, "instinct", "work", seeds), load(out, arm, step, "instinct_pair", "work", seeds)
    if not P:
        raise SystemExit("no scored lives of %s under %s" % (arm, out))
    world = next(iter(P.values()))["world"]
    A = json.load(open(os.path.join(out, "anchors_%s_work.json" % world)))
    court = world == "breakout"
    kinds = ("ball", "paddle") if court else ("food", "poison")
    H = len(_get(next(iter(P.values())), "regular", "curve", "skill_copy"))
    hs = [h for h in REPORT if h <= H]
    models = [("the four transformers, `pair`", "pair", P), ("the same weights, `mean`", "mean", Mn),
              ("the same weights, as the instinct dreams (`instinct`; added after the first results)", "instinct", In),
              ("the same weights, the instinct's way with `pair` (`instinct_pair`; added after the first results)", "instinct_pair", Ip)]
    models = [m for m in models if m[2]]
    floors = [("copy", "copy"), ("the mean frame", "mean frame"), ("the world itself", "the world")] + \
        ([("the world, other dice", "other dice")] if court else [])
    print("## %s: %d seeds at %dk (%s); the working set: %d starts, %s\n" % (
        arm, len(P), step // 1000, ", ".join(str(s) for s in sorted(P)), A["starts"],
        "%d hits and %d misses" % (A["facts"]["hits"], A["facts"]["misses"]) if court else "%d ticks with a bounce" % A["facts"]["bounce_ticks"]))

    # ---- the board
    head = ["Predictor", "Next-frame skill"] + (["Hit forecast, next tick", "Miss forecast, next tick"] if court else
                                              ["Skill on discs that turn: food", "poison"])
    head += ["Dream horizon by pixels, ticks"] + ["Horizon of the %s, ticks" % k for k in kinds] + ["Valid frames at 8 ticks"]
    ev = (lambda k: ("events", "next", k, "auc")) if court else (lambda k: ("events", "bounce", k, "skill"))
    evk = ("hit", "miss") if court else kinds
    body = []
    for name, tag, rs in models:
        body.append([name, pm(rs, "regular", "next", "skill")] + [pm(rs, *ev(k)) if tag == "pair" else "–" for k in evk]
                    + [pm(rs, "regular", "curve", "horizon", fmt=f1)]
                    + [pm(rs, "regular", "hold", k, "horizon", fmt=f1) for k in kinds] + [pm(rs, "regular", "real", "valid", 7)])
    for name, key in floors:
        r = A[key]
        body.append([name, one(r, "next", "skill")] + [one(r, *ev(k)) if "events" in r else "–" for k in evk]
                    + [one(r, "curve", "horizon", fmt=f1, top=H)] + [one(r, "hold", k, "horizon", fmt=f1, top=H) for k in kinds]
                    + [one(r, "real", "valid", 7)])
    lin, cop, seen = A["copy"]["hold_linear"], A["copy"]["hold_copy"], A["copy"].get("hold_seen", {})
    for k_ in sorted(seen, key=int):   # added after the first results: what a straight line can do from the frames
        body.append(["a straight line through the last %s frames as seen" % k_] + ["–"] * (1 + len(evk) + 1)
                    + [one(seen[k_], k, "horizon", fmt=f1, top=H) for k in kinds] + ["–"])
    body.append(["a straight line through the last 2 true places"] + ["–"] * (1 + len(evk) + 1)
                + [one(lin, k, "horizon", fmt=f1, top=H) for k in kinds] + ["–"])
    print("### The board\n")
    table(head, body)

    # ---- next frame by kind of thing
    by = list(_get(next(iter(P.values())), "regular", "next", "by"))
    head = ["Pixels of", "Pixels per frame", "Copy's error, grey levels (root mean square)"]
    for _, tag, rs in models:
        head += ["Skill, `%s`" % tag, "Error, `%s`" % tag]
    body = [["the whole frame", "4096", f2(A["copy"]["next"]["copy_rmse"])]
            + sum([[pm(rs, "regular", "next", "skill"), pm(rs, "regular", "next", "rmse", fmt=f2)] for _, _, rs in models], [])]
    for k in by:
        body.append([k if k == "rest" else "the " + k, f1(_get(A["copy"], "next", "by", k, "pixels")),
                     f2(_get(A["copy"], "next", "by", k, "copy_rmse"))]
                    + sum([[pm(rs, "regular", "next", "by", k, "skill"), pm(rs, "regular", "next", "by", k, "rmse", fmt=f2)]
                           for _, _, rs in models], []))
    print("### The next frame, by kind of thing\n")
    table(head, body)

    # ---- the dream curve
    head = ["Dreamed tick"] + [str(h) for h in hs]
    body = []
    for _, tag, rs in models:
        body.append(["Skill against the copy, `%s`" % tag] + [pm(rs, "regular", "curve", "skill_copy", h - 1) for h in hs])
        body.append(["Skill against the mean frame, `%s`" % tag] + [pm(rs, "regular", "curve", "skill_mean", h - 1) for h in hs])
        body.append(["Error in grey levels, `%s`" % tag] + [pm(rs, "regular", "curve", "rmse", h - 1, fmt=f2) for h in hs])
    body.append(["Error of the copy"] + [f2(A["copy"]["curve"]["copy_rmse"][h - 1]) for h in hs])
    body.append(["Error of the mean frame"] + [f2(A["copy"]["curve"]["mean_rmse"][h - 1]) for h in hs])
    if court:
        body.append(["Error of the world with other dice"] + [f2(A["other dice"]["curve"]["rmse"][h - 1]) for h in hs])
    print("### The dream, tick by tick (pixels)\n")
    table(head, body)

    # ---- object hold
    head = ["Thing", "Read on"] + ["Tick %d" % h for h in hs]
    body = []
    for k in kinds:
        for _, tag, rs in models:
            body.append([k, "held, `%s`" % tag] + [pm(rs, "regular", "hold", k, "held", h - 1) for h in hs])
            body.append([k, "lost, `%s`" % tag] + [pm(rs, "regular", "hold", k, "lost", h - 1) for h in hs])
            body.append([k, "px off where found (median), `%s`" % tag] + [pm(rs, "regular", "hold", k, "dist", h - 1, fmt=f1) for h in hs])
        for k_ in sorted(seen, key=int):
            body.append([k, "held, a straight line through the last %s frames as seen" % k_] + [f3(seen[k_][k]["held"][h - 1]) for h in hs])
        body.append([k, "held, a straight line through the last 2 true places"] + [f3(lin[k]["held"][h - 1]) for h in hs])
        body.append([k, "held, left where it was"] + [f3(cop[k]["held"][h - 1]) for h in hs])
        if court:
            body.append([k, "held, the world with other dice"] + [f3(A["other dice"]["hold"][k]["held"][h - 1]) for h in hs])
    print("### The things in the dream (held: found within %g px of the true place)\n" % HOLD)
    table(head, body)

    # ---- realism
    head = ["Dreamed tick"] + [str(h) for h in hs]
    body = []
    for _, tag, rs in models:
        body.append(["Valid frames, `%s`" % tag] + [pm(rs, "regular", "real", "valid", h - 1) for h in hs])
        body.append(["Grey levels off the real distribution, `%s`" % tag] + [pm(rs, "regular", "real", "w1", h - 1, fmt=f2) for h in hs])
    body.append(["Grey levels off, the mean frame"] + [f2(A["mean frame"]["real"]["w1"][h - 1]) for h in hs])
    print("### Is a dreamed frame a possible frame?\n")
    table(head, body)
    Z = {sd: r for sd, r in P.items() if _get(r, "regular", "real", "zones") == _get(r, "regular", "real", "zones")}
    if Z:   # by zone (added after the first results): what makes a frame invalid
        names = {"all": "the frame", "court": "the court (the ball)", "paddle rows": "the paddle's rows", "walls": "the walls"}
        body = []
        for n in A["facts"]["zones"]:
            body.append([names[n], "valid, `pair`"] + [pm(Z, "regular", "real", "zones", n, "valid", h - 1) for h in hs])
            body.append([names[n], "bright pixels, `pair`"] + [pm(Z, "regular", "real", "zones", n, "count", h - 1, fmt=f1) for h in hs])
            if Mn:
                body.append([names[n], "bright pixels, `mean`"] + [pm(Mn, "regular", "real", "zones", n, "count", h - 1, fmt=f1) for h in hs])
            body.append([names[n], "bright pixels, the real frames"] + [f1(_get(next(iter(Z.values())), "regular", "real", "zones", n, "real", h - 1)) for h in hs])
        print("By zone (added after the first results): the share of dreamed frames whose count of bright pixels is one the real frames reach, and the counts.\n")
        table(["Zone", "Read"] + ["Tick %d" % h for h in hs], body)

    # ---- the decoders of the same weights, paired by seed
    others = [(tag, rs) for _, tag, rs in models[1:]]
    both = sorted(set(P).intersection(*[set(rs) for _, rs in others])) if others else []
    if len(both) > 1:
        rows = [("Next-frame skill", ("regular", "next", "skill"), f3), ("Skill against the copy at tick 8", ("regular", "curve", "skill_copy", 7), f3),
                ("Dream horizon by pixels, ticks", ("regular", "curve", "horizon"), f1), ("Valid frames at 8 ticks", ("regular", "real", "valid", 7), f3),
                ("Grey levels off the real distribution at 8 ticks", ("regular", "real", "w1", 7), f2)]
        for k in kinds:
            rows += [("The %s held at tick 2" % k, ("regular", "hold", k, "held", 1), f3), ("The %s held at tick 8" % k, ("regular", "hold", k, "held", 7), f3),
                     ("Horizon of the %s, ticks" % k, ("regular", "hold", k, "horizon"), f1)]
        body = []
        for label, path, fmt in rows:
            pv = [_get(P[sd], *path) for sd in both]
            row = [label, pm({sd: P[sd] for sd in both}, *path, fmt=fmt)]
            for _, rs in others:
                row += [pm({sd: rs[sd] for sd in both}, *path, fmt=fmt)] + pdiff([_get(rs[sd], *path) for sd in both], pv, fmt)
            body.append(row)
        print("### The same weights decoded %d ways (paired by seed, %d seeds)\n" % (1 + len(others), len(both)))
        table(["Number", "`pair`"] + sum([["`%s`" % tag, "`%s` − `pair`" % tag, "Seeds with `%s` above" % tag] for tag, _ in others], []), body)

    # ---- events
    print("### Events\n")
    if court:
        head = ["Event", "Events in the set", "Area under the curve, next tick", "Predicted size ÷ real"] + \
            ["Within %d dreamed tick%s" % (d, "" if d == 1 else "s") for d in LEADS]
        body = []
        for k in ("hit", "miss"):
            body.append([k, str(A["the world"]["events"]["next"][k]["events"]), pm(P, "events", "next", k, "auc"),
                         pm(P, "events", "next", k, "size")] + [pm(P, "events", "within", k, str(d), "auc") for d in LEADS])
        table(head, body)
    else:
        head = ["Starts", "Discs", "Scored", "Skill on their pixels", "Held", "Lost", "px off where found", "px off, kept moving as it last moved",
                "px off, left where it was"]
        body = []
        for nm, label in (("bounce", "the next tick has a bounce (the discs that turn)"), ("free", "free flight")):
            for k in kinds:
                rs = P
                body.append([label, k, str(_get(A["copy"], "events", nm, k, "things")), pm(rs, "events", nm, k, "skill"),
                             pm(rs, "events", nm, k, "held"), pm(rs, "events", nm, k, "lost"), pm(rs, "events", nm, k, "dist", fmt=f2),
                             f2(_get(A["copy"], "events", nm, k, "linear_dist")), f2(_get(A["copy"], "events", nm, k, "copy_dist"))])
        table(head, body)

        rows = after_a_turn(out, arm, sorted(P), step, exam=exam)
        print("Added after the results: the discs of the regular starts by when they first turn in the dreamed stretch, and the share held at tick 4.\n")
        table(["Kind", "In the dreamed stretch the disc", "Discs", "Held at tick 4, the dream (`pair`)", "Held at tick 4, kept moving as it last moved"],
              [[k, label, str(n), "%s ± %s" % tuple(f3(v) for v in _stat(held)), f3(lin)] for k, label, n, held, lin in rows])

    # ---- warm-up
    if any("warm" in r for r in P.values()):
        head = ["Real ticks shown"] + [str(k) for k in WARM_K]
        body = [["Next-frame skill"] + [pm(P, "warm", str(k), "skill") for k in WARM_K]]
        if court:
            body += [["Hit forecast"] + [pm(P, "warm", str(k), "hit") for k in WARM_K],
                     ["Miss forecast"] + [pm(P, "warm", str(k), "miss") for k in WARM_K]]
        print("### Warm-up: from a reset, shown the last k real ticks (`pair`)\n")
        table(head, body)

    # ---- the dream after a short warm-up (added after the first results)
    Wd = {sd: r for sd, r in P.items() if "warm_dream" in r}
    if Wd:
        ks = sorted(next(iter(Wd.values()))["warm_dream"], key=int)
        ticks = [h for h in (1, 2, 4, 8) if h <= len(_get(next(iter(Wd.values())), "warm_dream", ks[0], "skill"))]
        body = []
        for k_ in ks:
            body.append(["%s real ticks shown" % k_, "skill against the copy"] + [pm(Wd, "warm_dream", k_, "skill", h - 1) for h in ticks])
            for k in kinds:
                body.append(["%s real ticks shown" % k_, "the %s held" % k] + [pm(Wd, "warm_dream", k_, "held", k, h - 1) for h in ticks])
        print("### The dream after a short warm-up (`pair`; added after the first results)\n")
        table(["From a reset", "Read"] + ["Tick %d" % h for h in ticks], body)

    # ---- learning (the last checkpoint is not scored twice: its first 8 dreamed ticks are in the full run)
    L = {st: load(out, arm, st, "pair", "learn", seeds) for st in steps}
    if not L[step]:
        L[step] = {sd: _cut(r, LEARN_TICKS) for sd, r in P.items()}
    if all(L.values()):
        head = ["Ticks lived"] + ["%dk" % (st // 1000) for st in steps]
        body = [["Next-frame skill"] + [pm(L[st], "regular", "next", "skill") for st in steps],
                ["Skill at the 8th dreamed tick"] + [pm(L[st], "regular", "curve", "skill_copy", 7) for st in steps],
                ["Dream horizon by pixels (of 8)"] + [pm(L[st], "regular", "curve", "horizon", fmt=f1) for st in steps]]
        body += [["Horizon of the %s (of 8)" % k] + [pm(L[st], "regular", "hold", k, "horizon", fmt=f1) for st in steps] for k in kinds]
        if court:
            body += [["Hit forecast, next tick"] + [pm(L[st], "events", "next", "hit", "auc") for st in steps],
                     ["Miss forecast, next tick"] + [pm(L[st], "events", "next", "miss", "auc") for st in steps]]
        print("### Learning: the same numbers by checkpoint (`pair`)\n")
        table(head, body)
        last = L[steps[-1]]
        first = []
        for sd in sorted(last):
            end = _get(last[sd], "regular", "next", "skill")
            first.append(next((st for st in steps if sd in L[st] and _get(L[st][sd], "regular", "next", "skill") >= end - 0.1 * abs(end)), steps[-1]))
        print("The first checkpoint with the next-frame skill within a tenth of its value at %dk, by seed: %s.\n" % (
            steps[-1] // 1000, ", ".join("%dk" % (x // 1000) for x in first)))

    # ---- home
    Hm = load(out, arm, step, "pair", "home", seeds)
    if Hm:
        both = sorted(set(Hm) & set(P))
        head = ["Exam", "Starts per life", "Next-frame skill", "Skill at tick 8", "Dream horizon by pixels"] + ["Horizon of the %s" % k for k in kinds]
        body = []
        for label, rs in (("away: the working set", {s: P[s] for s in both}), ("home: the life's own world, later", {s: Hm[s] for s in both})):
            body.append([label, str(_get(next(iter(rs.values())), "regular", "starts")), pm(rs, "regular", "next", "skill"),
                         pm(rs, "regular", "curve", "skill_copy", 7), pm(rs, "regular", "curve", "horizon", fmt=f1)]
                        + [pm(rs, "regular", "hold", k, "horizon", fmt=f1) for k in kinds])
        print("### Home and away (`pair`)\n")
        table(head, body)
        if len(both) > 1:
            for label, path, fmt in (("next-frame skill", ("regular", "next", "skill"), f3),
                                     ("skill at tick 8", ("regular", "curve", "skill_copy", 7), f3),
                                     ("dream horizon by pixels", ("regular", "curve", "horizon"), f1)):
                d = pdiff([_get(Hm[s], *path) for s in both], [_get(P[s], *path) for s in both], fmt)
                print("Home − away, %s: %s, %s." % (label, d[0], d[1]))
            print()

    # ---- a long dream
    Fr = {sd: r for sd, r in P.items() if "free" in r}
    if Fr:
        ts = [t for t in (1, 8, 32, 64, 100, 200) if t <= FREE_TICKS]
        body = [["Valid frames"] + [pm(Fr, "free", "valid", t - 1) for t in ts]]
        for z, n in enumerate(A["facts"]["zones"]):
            label = {"all": "the frame", "court": "the court (the ball)", "paddle rows": "the paddle's rows", "walls": "the walls"}[n]
            body.append(["Bright pixels of %s (the real frames: %s)" % (label, f1(A["facts"]["count_mean"][z]))]
                        + [pm(Fr, "free", "zones", n, "count", t - 1, fmt=f1) for t in ts])
        print("### A dream of %d ticks (`pair`; seeds %s; %d starts a life)\n" % (
            FREE_TICKS, ", ".join(str(s) for s in sorted(Fr)), _get(next(iter(Fr.values())), "free", "starts")))
        table(["Dreamed tick"] + [str(t) for t in ts], body)
        print("The first invalid frame comes at tick %s (the median of a life's dreams, by seed; %d: none in the dream).\n" % (
            ", ".join(f1(_get(r, "free", "first_invalid")) for _, r in sorted(Fr.items())), FREE_TICKS + 1))

    # ---- by seed
    head = ["Seed", "Next-frame skill"] + (["Hit forecast", "Miss forecast"] if court else ["Skill on turning food", "poison"]) + \
        ["Dream horizon by pixels"] + ["Horizon of the %s" % k for k in kinds] + ["Valid at 8 ticks", "Decoding passes a tick"]
    body = [[str(sd), f3(_get(r, "regular", "next", "skill"))] + [f3(_get(r, *ev(k))) for k in evk]
            + [str(_get(r, "regular", "curve", "horizon"))] + [str(_get(r, "regular", "hold", k, "horizon")) for k in kinds]
            + [f3(_get(r, "regular", "real", "valid", 7)), f1(_get(r, "regular", "passes"))] for sd, r in sorted(P.items())]
    print("### By seed (`pair`)\n")
    table(head, body)
    un = sum(sum(r["unsettled"].values()) if isinstance(r.get("unsettled"), dict) else r.get("unsettled", 0)
             for r in list(P.values()) + list(Mn.values()))
    print("Frames whose decoding did not settle: %d.\n" % un)


def figure(out, arm, seeds, path, step=60000):
    """A README figure of one arm's dreams at a checkpoint, the seeds' mean with its 95% band, one colour a decoder:
    the pixel skill against the copy by dreamed tick (beside the mean frame), and for each kind of thing the share held
    (beside the thing kept moving as it last moved)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from readme_images import COLORS, GRID, INK, MUTED_INK, SURFACE, save
    runs = [(d, load(out, arm, step, d, "work", seeds)) for d in ("pair", "mean", "instinct", "instinct_pair")]
    runs = [(d, rs, c) for (d, rs), c in zip(runs, COLORS) if rs]          # a decoder keeps its colour in every panel
    world = next(iter(runs[0][1].values()))["world"]
    A = json.load(open(os.path.join(out, "anchors_%s_work.json" % world)))
    kinds = ("ball", "paddle") if world == "breakout" else ("food", "poison")
    H = len(_get(next(iter(runs[0][1].values())), "regular", "curve", "skill_copy"))
    x = np.arange(1, H + 1)

    def band(rs, *path):
        v = np.array([_get(r, *path) for r in rs.values()], np.float64)
        st = [_stat(v[:, h]) for h in range(v.shape[1])]
        return np.array([a for a, _ in st]), np.nan_to_num(np.array([b for _, b in st]))

    def ends(ax, items, lo, hi):                                          # direct labels at the lines' ends, kept apart
        gap, place = (hi - lo) * 0.075, []
        for name, y in sorted(items, key=lambda t: t[1]):
            place.append((name, min(max(y, place[-1][1] + gap) if place else max(y, lo), hi)))
        for name, y in place:
            ax.annotate(name, (H, y), xytext=(5, 0), textcoords="offset points", va="center", fontsize=8, color=MUTED_INK,
                        annotation_clip=False)

    fig, axs = plt.subplots(1, 3, figsize=(9.2, 3.5), dpi=100, facecolor=SURFACE)
    ax, least, last = axs[0], -0.25, []
    for d, rs, col in runs:
        m, h = band(rs, "regular", "curve", "skill_copy")
        ax.fill_between(x, m - h, m + h, color=col, alpha=0.12, lw=0)
        ax.plot(x, m, color=col, lw=2, label=d, solid_capstyle="round")
        last.append((d, m[-1]))
        least = min(least, float(m.min()))
    lo = max(-1.0, float(np.floor(least * 4) / 4))                        # the models' whole curves, down to -1 at most
    mf = np.array(A["mean frame"]["curve"]["skill_copy"])
    ax.plot(x[mf >= lo], mf[mf >= lo], color=MUTED_INK, lw=1.2, ls=(0, (4, 2)), label="a floor that needs no model")
    ax.axhline(0, color=INK, lw=1)
    ax.set_ylim(lo, 1.02)
    ax.set_title("Pixels: skill against the copy", fontsize=10, color=INK, loc="left")
    ends(ax, last + [("mean frame", mf[-1])], lo, 1.02)
    lin, seen = A["copy"]["hold_linear"], A["copy"].get("hold_seen", {})
    for ax, k in zip(axs[1:], kinds):
        for d, rs, col in runs:
            m, h = band(rs, "regular", "hold", k, "held")
            ax.fill_between(x, m - h, m + h, color=col, alpha=0.12, lw=0)
            ax.plot(x, m, color=col, lw=2, solid_capstyle="round")
        marks = []
        for k_, ls in (("2", (0, (1, 2))), ("8", (0, (4, 2)))):          # straight lines through the frames as seen
            if k_ in seen:
                ax.plot(x, seen[k_][k]["held"][:H], color=MUTED_INK, lw=1.2, ls=ls)
                marks.append(("line, %s frames" % k_, seen[k_][k]["held"][H - 1]))
        if not marks:
            ax.plot(x, lin[k]["held"][:H], color=MUTED_INK, lw=1.2, ls=(0, (4, 2)))
            marks.append(("moved on\nas before", lin[k]["held"][H - 1]))
        ax.set_ylim(-0.02, 1.02)
        ax.set_title("The %s held (within %g px)" % (k, HOLD), fontsize=10, color=INK, loc="left")
        ends(ax, marks, -0.02, 1.02)                                      # the decoders: the legend
    for ax in axs:
        ax.set_facecolor(SURFACE)
        ax.set_xscale("log", base=2)
        ax.set_xticks([t for t in REPORT if t <= H])
        ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: "%g" % v))
        ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        ax.set_xlabel("dreamed tick", fontsize=9, color=MUTED_INK)
        ax.grid(axis="y", color=GRID, lw=1)
        ax.set_axisbelow(True)
        for k in ("top", "right", "left"):
            ax.spines[k].set_visible(False)
        ax.spines["bottom"].set_color(GRID)
        ax.tick_params(colors=MUTED_INK, labelsize=9, length=0)
        ax.set_xlim(1, H)
    fig.legend(*axs[0].get_legend_handles_labels(), frameon=False, fontsize=8, labelcolor=MUTED_INK, ncol=5, loc="lower center",
               handlelength=2.0, columnspacing=1.4)
    fig.tight_layout(w_pad=3.2, rect=(0, 0.07, 0.97, 1))
    save(fig, path)


def strip(run, step, path, copies=(0, 1), start=1200, ticks=(1, 2, 4, 8, 16, 32), scale=2, exam=None):
    """A picture of dreams for a README: for each stretch (a copy of the working set at tick `start`) a row of real
    frames (the last real tick, then `ticks` later), and under it the same ticks as the model dreams them with pair and
    with mean. Returns the rows' next-frame errors for the caption."""
    from PIL import Image, ImageDraw, ImageFont
    from readme_images import FONT, MUTED_INK, SURFACE, fit
    info = json.load(open(os.path.join(run, "info.json")))
    exam = exam or Exam(info["config"]["env"])
    e, s_ = np.array(copies), np.full(len(copies), start)
    H = max(ticks)
    fr, _ = real(exam, e, s_, H)
    rows = [("the world", fr)]
    for dec in ("pair", "mean"):
        m = Pieces(run, step, dec)
        rows.append(("the dream, `%s`" % dec, dreams(m, exam, e, s_, H)[0]))
        m.close()
    side, gap, pad, lab = exam.S * scale, 6, 6, 18
    n = 1 + len(ticks)
    font = ImageFont.truetype(FONT, 13)
    im = Image.new("RGB", (n * side + (n - 1) * gap + 2 * pad, len(copies) * len(rows) * (side + lab) + 2 * pad), SURFACE)
    d = ImageDraw.Draw(im)

    def tile(f):
        return Image.fromarray(np.asarray(f, np.uint8), "L").resize((side, side), Image.NEAREST)
    y = pad
    for j in range(len(copies)):
        for name, f in rows:
            d.text((pad, y), "%s (stretch %d)" % (name.replace("`", ""), j + 1), font=font, fill=MUTED_INK)
            im.paste(tile(exam.frames[e[j], s_[j]]), (pad, y + lab))
            for k, t in enumerate(ticks):
                x = pad + (k + 1) * (side + gap)
                d.text((x + side - 34, y), "+%d" % t, font=font, fill=MUTED_INK)
                im.paste(tile(f[j, t - 1]), (x, y + lab))
            y += side + lab
    fit(im, path)
    return im.size


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("score")
    a.add_argument("run")
    a.add_argument("--step", type=int, nargs="+", required=True)
    a.add_argument("--decode", nargs="+", default=["pair"])
    a.add_argument("--set", dest="which", nargs="+", default=["work"], choices=["work", "home", "final"])
    a.add_argument("--parts", nargs="+", default=["regular", "events", "warm"])
    a.add_argument("--horizon", type=int, default=HORIZON)
    a.add_argument("--name", default=None, help="the set's name in the file (learn: the checkpoints' shorter run)")
    a = sub.add_parser("anchors")
    a.add_argument("config")
    a.add_argument("-o", "--out", required=True)
    a.add_argument("--set", dest="which", default="work", choices=["work", "final"])
    a = sub.add_parser("figure")
    a.add_argument("out")
    a.add_argument("arm")
    a.add_argument("-o", "--path", required=True)
    a.add_argument("--seeds", type=int, nargs="+", default=list(range(1, 13)))
    a.add_argument("--step", type=int, default=60000)
    a = sub.add_parser("strip")
    a.add_argument("run")
    a.add_argument("--step", type=int, default=60000)
    a.add_argument("-o", "--out", required=True)
    a = sub.add_parser("table")
    a.add_argument("out")
    a.add_argument("arm")
    a.add_argument("--seeds", type=int, nargs="+", default=list(range(1, 13)))
    a.add_argument("--step", type=int, default=60000)
    args = ap.parse_args()
    if args.cmd == "table":
        return tables(args.out, args.arm, args.seeds, args.step)
    if args.cmd == "figure":
        return figure(args.out, args.arm, args.seeds, args.path, args.step)
    if args.cmd == "strip":
        return print("wrote", args.out, strip(args.run, args.step, args.out))
    if args.cmd == "score":
        for st in args.step:
            for dec in args.decode:
                for which in args.which:
                    r = score(args.run, st, dec, which, args.parts, args.horizon, name=args.name)
                    g = r.get("regular", {})
                    print("%-10s %6d %-5s %-5s  next-frame skill %s  horizon %s  passes %s  seconds %s" % (
                        r["run"], st, dec, which, "%.3f" % g["next"]["skill"] if g else "-",
                        g["curve"]["horizon"] if g else "-", "%.1f" % g["passes"] if g else "-", r["seconds"]), flush=True)
    elif args.cmd == "anchors":
        env = json.load(open(args.config))["env"]
        r = anchors(env, args.which)
        os.makedirs(args.out, exist_ok=True)
        p = os.path.join(args.out, "anchors_%s_%s.json" % (r["world"], args.which))
        with open(p, "w") as f:
            json.dump(r, f)
        print("wrote", p)


if __name__ == "__main__":
    main()

"""SenseWorld (task 00081): two senses side by side, one wide frame [view | text]. The view is the wrapped world's. The text is a
separate sense, not drawn into the picture: a scripted teacher writes what is nearest, as words of grey letters ("food", "left",
"near"), one letter to a block of size / 8 pixels, as the text world's level encoding. Each sense goes off by itself for a while:
the eyes close, or the teacher falls silent, for longer than the model's context, so that what was there has left its window.
Moves, hunger and touches are the world's. Nothing here is read by the model other than as pixels.
"""
import dataclasses

import numpy as np

WORDS = (("food", "poison"), ("left", "right", "up", "down"), ("near", "far"))
LETTERS = sorted(set("".join(w for ws in WORDS for w in ws)))
LEVELS = np.array([int(round(255 * (k + 1) / len(LETTERS))) for k in range(len(LETTERS))], np.uint8)   # blank is 0
NEAR = 16.0                                                                                               # px


@dataclasses.dataclass
class SenseConfig:
    teacher: str = "true"         # true: what is nearest; shuffle: a random sentence each time the true one changes (control); off
    gap_min: int = 60             # ticks with both senses on, uniform in min..max
    gap_max: int = 140
    drop_min: int = 10            # ticks one sense is off
    drop_max: int = 16
    cut: bool = False             # for scoring: while either sense is off the other is off too

    def validate(self):
        if self.teacher not in ("true", "shuffle", "off"):
            raise ValueError("sense.teacher must be true, shuffle or off")
        if not (1 <= self.gap_min <= self.gap_max and 1 <= self.drop_min <= self.drop_max):
            raise ValueError("sense: gaps and drops of at least 1 tick, min <= max")
        return self


def sentence_frames(size):
    """All sentences as grey text frames, [kinds, directions, ranges, size, size] u8: a word to a row."""
    b = size // 8
    if max(len(w) for ws in WORDS for w in ws) > 8:
        raise ValueError("words longer than a row")
    out = np.zeros(tuple(len(w) for w in WORDS) + (size, size), np.uint8)
    for ix in np.ndindex(out.shape[:3]):
        for row, (ws, i) in enumerate(zip(WORDS, ix)):
            for col, ch in enumerate(ws[i]):
                out[ix][row * b:(row + 1) * b, col * b:(col + 1) * b] = LEVELS[LETTERS.index(ch)]
    return out


class SenseWorld:
    """A world's frames with a text sense to their right."""

    def __init__(self, num_envs, world, config, seed=0):
        self.num_envs, self.world, self.cfg = num_envs, world, config.validate()
        self.config = world.config
        self.size = world.config.size
        self.rng = np.random.default_rng([seed, 81])
        self.table = sentence_frames(self.size)
        self.eyes = np.zeros(num_envs, int)                       # ticks the eyes stay closed
        self.quiet = np.zeros(num_envs, int)                      # ticks the teacher stays silent
        self.gap = self._draw(config.gap_min, config.gap_max)
        self.said = self._nearest()
        self.shown = self.said.copy()
        self.reset_stats()

    def __getattr__(self, name):                                  # scoring reads the world's cursor and circles
        if name == "world":
            raise AttributeError(name)
        return getattr(self.world, name)

    def _draw(self, lo, hi):
        return self.rng.integers(lo, hi + 1, self.num_envs)

    def relative(self, xy):
        """Where points [N, K, 2] are from the cursor on the torus."""
        return (xy - self.world.cursor[:, None] + self.size / 2) % self.size - self.size / 2

    def _nearest(self):
        """[N, 3] word indices: the kind, direction and range of the thing nearest the cursor, in the ego view."""
        w, N = self.world, self.num_envs
        pts = np.concatenate([w.target[:, None], w.hz_pos], 1)
        d = self.relative(pts)
        dist = np.hypot(d[..., 0], d[..., 1])
        k = dist.argmin(1)
        dx, dy = d[np.arange(N), k, 0], d[np.arange(N), k, 1]
        side = np.where(np.abs(dx) >= np.abs(dy), np.where(dx < 0, 0, 1), np.where(dy < 0, 2, 3))
        return np.stack([(k > 0).astype(int), side, (dist[np.arange(N), k] > NEAR).astype(int)], 1)

    def nearest_xy(self):
        """Scoring: [N, 2] the nearest thing's place relative to the cursor, [N] its radius."""
        w, N = self.world, self.num_envs
        d = self.relative(np.concatenate([w.target[:, None], w.hz_pos], 1))
        k = np.hypot(d[..., 0], d[..., 1]).argmin(1)
        r = np.concatenate([w.radius[:, None], w.hz_rad], 1)
        return d[np.arange(N), k], r[np.arange(N), k]

    def reset_stats(self):
        self.st = dict(blind=0, food_blind=0, poison_blind=0, quiet=0, seen=0, food_seen=0, poison_seen=0)

    def stats(self):
        """Per 1k ticks with the eyes closed, and with them open; the shares of ticks."""
        s, n = self.st, max(self.st["blind"] + self.st["seen"], 1)
        self.reset_stats()
        return dict(sense_blind=s["blind"] / n, sense_quiet=s["quiet"] / n, food_blind=1000.0 * s["food_blind"] / max(s["blind"], 1),
                    poison_blind=1000.0 * s["poison_blind"] / max(s["blind"], 1), food_seen=1000.0 * s["food_seen"] / max(s["seen"], 1),
                    poison_seen=1000.0 * s["poison_seen"] / max(s["seen"], 1))

    def frame(self, view):
        v = view.copy()
        c = self.cfg
        i = self.shown
        text = self.table[i[:, 0], i[:, 1], i[:, 2]]
        off_text = self.quiet > 0
        off_view = self.eyes > 0
        if c.cut:
            off_text, off_view = off_text | off_view, off_text | off_view
        if c.teacher == "off":
            off_text = np.ones(self.num_envs, bool)
        text = np.where(off_text[:, None, None], 0, text)
        v[off_view] = 0
        return np.concatenate([v, np.repeat(text[..., None], 3, -1)], 2)

    def observe(self):
        o = dict(self.world.observe())
        o["image"] = self.frame(o["image"])
        return o

    def _tick(self):
        """The senses' clocks: a sense that is off counts down; after the gap one goes off, either at even odds."""
        self.eyes = np.maximum(self.eyes - 1, 0)
        self.quiet = np.maximum(self.quiet - 1, 0)
        both = (self.eyes == 0) & (self.quiet == 0)
        self.gap = self.gap - both
        due = both & (self.gap <= 0)
        if due.any():
            c = self.cfg
            d, pick = self._draw(c.drop_min, c.drop_max), self.rng.random(self.num_envs) < 0.5
            self.eyes = np.where(due & pick, d, self.eyes)
            self.quiet = np.where(due & ~pick, d, self.quiet)
            self.gap = np.where(due, self._draw(c.gap_min, c.gap_max), self.gap)

    def step(self, actions):
        blind = self.eyes > 0
        obs, rew, done, inf = self.world.step(actions)
        s = self.st
        hit = inf["success"]
        bad = inf["gt_hazard_click"] if "gt_hazard_click" in inf else np.zeros(self.num_envs, bool)
        s["blind"] += int(blind.sum())
        s["seen"] += int((~blind).sum())
        s["quiet"] += int((self.quiet > 0).sum())
        s["food_blind"] += int((hit & blind).sum())
        s["poison_blind"] += int((bad & blind).sum())
        s["food_seen"] += int((hit & ~blind).sum())
        s["poison_seen"] += int((bad & ~blind).sum())
        self._tick()
        new = self._nearest()
        if self.cfg.teacher == "shuffle":
            changed = (new != self.said).any(1)
            rnd = np.stack([self.rng.integers(0, len(w), self.num_envs) for w in WORDS], 1)
            self.shown = np.where(changed[:, None], rnd, self.shown)
        else:
            self.shown = new
        self.said = new
        return dict(obs, image=self.frame(obs["image"])), rew, done, inf

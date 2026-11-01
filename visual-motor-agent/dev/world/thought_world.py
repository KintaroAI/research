"""ThoughtWorld (task 00064): a world and a grey thought field side by side, one wide frame [world | thought], with the
wrapped world's interface. Moves, hunger and touches are the world's. The thought field's next frame is what the caller gives
think(), the model's own dream of it, plus waves: so a prediction becomes its reality, and only waves bring anything new.

Waves start when thought goes quiet (its dream barely differs from it): at a random point a ring spreads; a value it passes goes
up, then down, then back, and overlapping rings add. A wave leaves a small bump where it started, so a value can settle a little
differently. Small areas can also change at random. Nothing here is read by the model other than as pixels.
"""
import dataclasses

import numpy as np


@dataclasses.dataclass
class ThoughtConfig:
    mode: str = "dream"           # dream; frozen: thought never changes; noise: a new random field every tick (controls)
    cell: int = 8                 # thought at birth: random grey on a grid of this many pixels, interpolated
    quiet: float = 0.002          # thought is quiet when its dream moves a pixel by less than this on average (0..1)
    p_wave: float = 0.2           # while quiet: a new wave per tick with this probability
    p_area: float = 0.05          # and a small area changes
    max_waves: int = 4
    amp: float = 0.15             # a wave's height at its start
    speed: float = 1.5            # pixels per tick
    width: float = 3.0            # pixels
    falloff: float = 24.0         # pixels: a wave's height halves at this distance
    settle: float = 0.02          # the bump a wave leaves
    area: int = 8                 # pixels
    area_amp: float = 0.1

    def validate(self):
        if self.mode not in ("dream", "frozen", "noise"):
            raise ValueError("thought.mode must be dream, frozen or noise")
        return self


def smooth_noise(rng, n, size, cell):
    """Random grey fields [n, size, size] in [0, 1]: a coarse random grid, bilinearly interpolated."""
    k = size // cell + 2
    g = rng.random((n, k, k))
    x = np.arange(size) / cell
    i, f = x.astype(int), x - np.floor(x)
    a = g[:, i[:, None], i[None, :]]
    b = g[:, i[:, None], i[None, :] + 1]
    c = g[:, i[:, None] + 1, i[None, :]]
    e = g[:, i[:, None] + 1, i[None, :] + 1]
    fy, fx = f[:, None], f[None, :]
    return a * (1 - fy) * (1 - fx) + b * (1 - fy) * fx + c * fy * (1 - fx) + e * fy * fx


class ThoughtWorld:
    """A world's frames with a thought field to their right. think(dream) makes the next thought; until it is called the
    frames show the last one."""

    def __init__(self, num_envs, world, config, seed=0):
        self.num_envs, self.world, self.cfg = num_envs, world, config.validate()
        self.config = world.config
        self.size = world.config.size
        self.rng = np.random.default_rng([seed, 64])
        self.T = smooth_noise(self.rng, num_envs, self.size, config.cell)
        self.waves = [[] for _ in range(num_envs)]               # per body: [y, x, age]
        y, x = np.mgrid[:self.size, :self.size]
        self.yy, self.xx = y.astype(np.float64), x.astype(np.float64)
        self.reset_stats()

    def reset_stats(self):
        self.st = dict(change=0.0, quiet=0, waves=0, areas=0, ticks=0)

    def stats(self):
        """Per body-tick since the last call: the mean change of a pixel, the share of quiet ticks, waves and areas started."""
        s, n = self.st, max(self.st["ticks"], 1)
        self.reset_stats()
        out = dict(thought_change=s["change"] / n, thought_quiet=s["quiet"] / n, thought_waves=s["waves"] / n,
                   thought_areas=s["areas"] / n, thought_extreme=float((np.abs(self.T - 0.5) > 0.45).mean()))
        if hasattr(self.world, "stats"):   # a world with trials (task 00065)
            out.update(self.world.stats())
        return out

    def thought_frames(self):
        return np.repeat(np.clip(np.rint(self.T * 255.0), 0, 255).astype(np.uint8)[..., None], 3, -1)

    def frame(self, world_image):
        return np.concatenate([world_image, self.thought_frames()], 2)

    def observe(self):
        o = dict(self.world.observe())
        o["image"] = self.frame(o["image"])
        return o

    def step(self, actions):
        obs, rew, done, inf = self.world.step(actions)
        self._world_image = obs["image"]
        return dict(obs, image=self.frame(obs["image"])), rew, done, inf

    def wave(self, y, x, age):
        """A wave's shift [size, size]: a ring at distance speed * age, up on its way in, down on its way out."""
        c = self.cfg
        r = np.hypot(self.yy - y, self.xx - x)
        u = (r - c.speed * age) / c.width
        return c.amp * (2.0 ** (-r / c.falloff)) * (-u) * np.exp(0.5 - 0.5 * u * u)

    def think(self, dream=None):
        """The next thought from dream [N, size, size, 3] u8 (None: as it is) and the waves; returns the frames."""
        c = self.cfg
        if c.mode == "frozen":
            self.st["ticks"] += self.num_envs
            return self.frame(self._world_image)
        if c.mode == "noise":
            new = smooth_noise(self.rng, self.num_envs, self.size, c.cell)
            self.st["change"] += float(np.abs(new - self.T).mean((1, 2)).sum())
            self.st["ticks"] += self.num_envs
            self.T = new
            return self.frame(self._world_image)
        d = self.T if dream is None else np.asarray(dream)[..., 0] / 255.0
        moved = np.abs(d - self.T).mean((1, 2))
        new = d.copy()
        for b in range(self.num_envs):
            if moved[b] < c.quiet:
                self.st["quiet"] += 1
                if len(self.waves[b]) < c.max_waves and self.rng.random() < c.p_wave:
                    y, x = self.rng.random(2) * self.size
                    self.waves[b].append([y, x, 0])
                    self.st["waves"] += 1
                    bump = np.exp(-0.5 * (np.hypot(self.yy - y, self.xx - x) / (2 * c.width)) ** 2)
                    new[b] += c.settle * self.rng.choice([-1.0, 1.0]) * bump
                if self.rng.random() < c.p_area:
                    y, x = self.rng.integers(0, self.size - c.area + 1, 2)
                    new[b, y:y + c.area, x:x + c.area] += self.rng.normal(0, c.area_amp)
                    self.st["areas"] += 1
            keep = []
            for w in self.waves[b]:                              # each wave moves on: its change this tick
                new[b] += self.wave(w[0], w[1], w[2] + 1) - self.wave(w[0], w[1], w[2])
                w[2] += 1
                if c.speed * w[2] < 1.5 * self.size + 3 * c.width:
                    keep.append(w)
            self.waves[b] = keep
        new = np.clip(new, 0.0, 1.0)
        self.st["change"] += float(np.abs(new - self.T).mean((1, 2)).sum())
        self.st["ticks"] += self.num_envs
        self.T = new
        return self.frame(self._world_image)

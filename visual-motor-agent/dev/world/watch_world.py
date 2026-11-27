"""Task 00132: a world to watch. Food and poison discs move on a torus and bounce off each other. Nothing is eaten, there
is no body in the frame, and the move does nothing: the model only observes.

The looks are the food world's (CircleWorld with gray, a checker poison, dots): food a solid disc, poison a disc that
shows the view's checkerboard, both the same grey on the same background, small grey dots moving under them.

- Each disc keeps its speed for ever: food `food_speed` px a tick, poison `poison_speed` (slower by default).
- Discs never overlap. Two that meet are pushed apart along the line of their centres, and each one that was moving
  toward the other has its velocity mirrored off that line, so its speed is unchanged. This is not an exchange of
  momentum: with equal masses the speeds would mix, and food and poison would stop differing in speed.
- Dots move on straight lines under everything and touch nothing.
- After the first frame nothing is left to chance: the world is fully determined by its state.

The body's values are constants (0): lifetime.py needs `drives`. The gt_* keys are for scoring only.
"""
import dataclasses
from typing import Any, Dict, Optional

import numpy as np

from .circle_world import _checker, _draw_dots, wrap_delta


@dataclasses.dataclass
class WatchConfig:
    kind: str = "watch"
    size: int = 64                      # the world (and frame) side in px; a torus
    foods: int = 2
    poisons: int = 2                    # 2 + 2 discs: a disc flies about 13 to 20 ticks between bounces (2 + 4: about 8 to 13)
    food_speed: float = 1.5             # px per tick, kept through every bounce
    poison_speed: float = 0.75
    food_radius: tuple = (8.0, 11.0)    # each disc's radius ~ U[min, max]; the default agent's world has the same
    poison_radius: tuple = (5.0, 8.0)
    checker_cell: int = 2               # the side in px of the poison's checker cells, aligned to the view grid
    dots: int = 8
    dot_size: int = 2
    dot_speed: tuple = (0.5, 2.0)       # px per tick ~ U[min, max] per dot, in a uniform direction
    dot_grey: tuple = (60.0, 180.0)     # grey level ~ U[min, max] per dot
    bg: int = 21                        # grey levels: the food world's background and circles under gray
    grey: int = 94
    # read by the agent loops, as in EnvConfig
    drives: bool = True
    respawn: bool = True
    auto_reset: bool = True
    gray: bool = True
    h_rate: float = 0.0

    def __post_init__(self) -> None:
        for name in ("food_radius", "poison_radius", "dot_speed", "dot_grey"):   # lists from JSON
            setattr(self, name, tuple(float(v) for v in getattr(self, name)))

    def validate(self) -> "WatchConfig":
        if self.kind != "watch":
            raise ValueError("WatchConfig.kind must be 'watch'")
        if int(self.size) != self.size or self.size < 16:
            raise ValueError("size must be an integer >= 16")
        if min(self.foods, self.poisons, self.dots) < 0 or self.foods + self.poisons < 1:
            raise ValueError("foods, poisons and dots >= 0, and at least one disc")
        for name in ("food_radius", "poison_radius", "dot_speed", "dot_grey"):
            lo, hi = getattr(self, name)
            if not 0 < lo <= hi:
                raise ValueError(f"{name} must be (min, max) with 0 < min <= max")
        r_min = min(([self.food_radius[0]] if self.foods else []) + ([self.poison_radius[0]] if self.poisons else []))
        if not (0 <= self.food_speed < r_min and 0 <= self.poison_speed < r_min):
            raise ValueError("speeds must be >= 0 and below the smallest radius a tick (or two discs can pass through each other)")
        area = np.pi * (self.foods * self.food_radius[1] ** 2 + self.poisons * self.poison_radius[1] ** 2)
        if area > 0.4 * self.size ** 2:
            raise ValueError("the discs at their largest must cover at most 40% of the world (or they cannot be placed apart)")
        if self.checker_cell < 1 or self.dot_size < 1 or not (0 <= self.bg <= 255 and 0 <= self.grey <= 255 and self.bg != self.grey):
            raise ValueError("checker_cell and dot_size >= 1; bg and grey different grey levels in 0..255")
        return self


class WatchWorld:
    """N worlds. pos [N, K, 2] the discs' centres (foods first, then poisons), vel [N, K, 2] px per tick, rad [N, K]."""

    def __init__(self, num_envs: int, config: Optional[WatchConfig] = None, seed: int = 0):
        if int(num_envs) < 1:
            raise ValueError("num_envs must be >= 1")
        self.num_envs = n = int(num_envs)
        self.config = c = dataclasses.replace(config if config is not None else WatchConfig()).validate()
        rng = np.random.default_rng(seed)   # used at birth only: afterwards the world is determined
        self.t = 0
        k = c.foods + c.poisons
        self.kind = np.repeat([0, 1], [c.foods, c.poisons])               # 0 food, 1 poison
        lo = np.where(self.kind == 0, c.food_radius[0], c.poison_radius[0])
        hi = np.where(self.kind == 0, c.food_radius[1], c.poison_radius[1])
        self.rad = lo + (hi - lo) * rng.random((n, k))
        self.pos = np.zeros((n, k, 2))
        size = float(c.size)
        for e in range(n):   # one at a time, clear of the ones before it
            for j in range(k):
                for _ in range(100000):
                    p = size * rng.random(2)
                    d = wrap_delta(p[None], self.pos[e, :j], size)
                    if j == 0 or np.all(np.hypot(d[:, 0], d[:, 1]) > self.rad[e, j] + self.rad[e, :j] + 1.0):
                        break
                else:
                    raise ValueError("could not place the discs apart")
                self.pos[e, j] = p
        ang = 2.0 * np.pi * rng.random((n, k))
        speed = np.where(self.kind == 0, c.food_speed, c.poison_speed)
        self.vel = speed[None, :, None] * np.stack([np.cos(ang), np.sin(ang)], -1)
        d = c.dots
        self.dot_pos = size * rng.random((n, d, 2))
        ang = 2.0 * np.pi * rng.random((n, d))
        sp = c.dot_speed[0] + (c.dot_speed[1] - c.dot_speed[0]) * rng.random((n, d))
        self.dot_vel = sp[..., None] * np.stack([np.cos(ang), np.sin(ang)], -1)
        self.dot_val = c.dot_grey[0] + (c.dot_grey[1] - c.dot_grey[0]) * rng.random((n, d))
        self.bounces = np.zeros(n, np.int64)
        self._chk = _checker(c.size, c.checker_cell)
        self._frame = None

    # ------------------------------------------------------------------ the discs
    def _bounce(self) -> np.ndarray:
        """After the discs moved: overlapping pairs are pushed apart, half the overlap each, and each disc that was moving
        toward the other mirrors its velocity off the line of centres. Returns [N, K] bool: whose velocity changed."""
        size = float(self.config.size)
        n, k = self.rad.shape
        turned = np.zeros((n, k), bool)
        d = wrap_delta(self.pos[:, :, None, :], self.pos[:, None, :, :], size)
        over = np.triu(np.hypot(d[..., 0], d[..., 1]) < self.rad[:, :, None] + self.rad[:, None, :], 1)
        for e in np.flatnonzero(over.any((1, 2))):
            pos, vel, rad = self.pos[e], self.vel[e], self.rad[e]
            for _ in range(64):   # a push can make a new overlap (a squeeze takes a few rounds)
                again = False
                for a in range(k):
                    for b in range(a + 1, k):
                        dd = wrap_delta(pos[a], pos[b], size)          # from b to a
                        dist = float(np.hypot(dd[0], dd[1]))
                        gap = rad[a] + rad[b] - dist
                        if gap <= 0.0:
                            continue
                        again = True
                        nrm = dd / dist if dist > 0.0 else np.array((1.0, 0.0))
                        pos[a] += nrm * (0.5 * gap + 1e-9)
                        pos[b] -= nrm * (0.5 * gap + 1e-9)
                        va, vb = float(vel[a] @ nrm), float(vel[b] @ nrm)
                        if va < 0.0:   # a was moving toward b
                            vel[a] -= 2.0 * va * nrm
                            turned[e, a] = True
                        if vb > 0.0:   # b was moving toward a
                            vel[b] -= 2.0 * vb * nrm
                            turned[e, b] = True
                if not again:
                    break
            self.pos[e] = np.mod(pos, size)
        return turned

    # ------------------------------------------------------------------ frames and body
    def _frames(self) -> np.ndarray:
        c = self.config
        n, s = self.num_envs, int(c.size)
        px = np.arange(s, dtype=np.float64) + 0.5
        dx = wrap_delta(px[None, None, None, :], self.pos[..., 0, None, None], float(s))
        dy = wrap_delta(px[None, None, :, None], self.pos[..., 1, None, None], float(s))
        inside = dx * dx + dy * dy <= (self.rad ** 2)[..., None, None]                    # [N, K, S, S]
        food = inside[:, self.kind == 0].any(1)
        poison = inside[:, self.kind == 1].any(1) & self._chk[None]
        img = np.full((n, s, s, 3), c.bg, np.uint8)
        img[food | poison] = c.grey
        if c.dots:   # on background pixels only: under the discs, and showing through the poison's gaps
            _draw_dots(img, self.dot_pos, self.dot_val, int(c.dot_size), np.array((c.bg,) * 3, np.uint8))
        return img

    def _drives(self) -> np.ndarray:
        return np.zeros((self.num_envs, 2), np.float32)

    # ------------------------------------------------------------------ CircleWorld's interface
    def observe(self) -> Dict[str, np.ndarray]:
        if self._frame is None:
            self._frame = self._frames()
        return {"image": self._frame.copy(), "drives": self._drives()}

    def reset(self, env_ids=None) -> Dict[str, np.ndarray]:
        return self.observe()

    def step(self, actions: np.ndarray):
        n = self.num_envs
        if np.shape(actions) != (n, 3):
            raise ValueError(f"expected actions of shape ({n}, 3), got {np.shape(actions)}")
        size = float(self.config.size)
        self.pos = np.mod(self.pos + self.vel, size)
        turned = self._bounce()
        self.dot_pos = np.mod(self.dot_pos + self.dot_vel, size)
        self.t += 1
        self.bounces += turned.any(1)
        self._frame = None
        obs = self.observe()
        f = np.zeros(n, bool)
        nan = np.full(n, np.nan)
        info = dict(terminal_image=obs["image"], terminal_drives=obs["drives"], hunger_cut=np.zeros(n), success=f.copy(),
                    timeout=f.copy(), click=f.copy(), ep_length=nan, ep_return=nan.copy(), ep_success=nan.copy(),
                    ep_clicks=nan.copy(),
                    # --- evaluation-only ground truth: NEVER a model input ---
                    gt_drives=np.zeros((n, 2)), gt_pos=self.pos.copy(), gt_vel=self.vel.copy(), gt_rad=self.rad.copy(),
                    gt_kind=np.broadcast_to(self.kind, self.rad.shape).copy(), gt_bounce=turned, gt_dots=self.dot_pos.copy())
        return obs, np.zeros(n, np.float32), f.copy(), info

    def get_state(self) -> Dict[str, Any]:
        return dict(t=self.t, pos=self.pos.copy(), vel=self.vel.copy(), rad=self.rad.copy(), dot_pos=self.dot_pos.copy(),
                    dot_vel=self.dot_vel.copy(), dot_val=self.dot_val.copy(), bounces=self.bounces.copy())

    def set_state(self, state: Dict[str, Any]) -> None:
        for name in ("pos", "vel", "rad", "dot_pos", "dot_vel", "dot_val", "bounces"):
            arr = np.array(state[name])
            if arr.shape != getattr(self, name).shape:
                raise ValueError(f"set_state: {name} has shape {arr.shape}, this world's is {getattr(self, name).shape}")
            setattr(self, name, arr)
        self.t = int(state["t"])
        self._frame = None

    def stats(self) -> Dict[str, Any]:
        return dict(wa_bounces=int(self.bounces.sum()))

    def close(self) -> None:
        pass

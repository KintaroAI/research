"""Task 00111 (vm-00103): a Breakout court for the default agent, the model unchanged; only the wiring is new.

The frame is the whole court, size x size grey pixels: walls on the left, right and top, a paddle near the bottom, one
small ball. The move's dx pushes the paddle (dx * paddle_speed px, clipped to the court); its dy and click do nothing.
The ball moves ball_speed px a tick and bounces off the walls; off the paddle it leaves at an angle set by where it hit.

The body is CircleWorld's: hunger and restlessness.
- hunger drifts +h_rate a tick;
- a hit is a meal: hunger falls by h_relief (to 0);
- a miss (the ball passes below the paddle) is a poison: hunger +miss_add, the part the cap at 1 cuts off is
  info["hunger_cut"] (task 00110);
- restlessness: +b_rate a tick, relieved by the paddle's movement within a movement budget, as CircleWorld's.

After a miss the ball is gone for `pause` ticks, then comes back under the top wall at a random place, falling at a
random angle within 45 degrees of vertical. The world never resets. gt_* keys are for scoring only.
"""
import dataclasses
from typing import Any, Dict, Optional

import numpy as np


@dataclasses.dataclass
class BreakoutConfig:
    kind: str = "breakout"
    size: int = 64                 # the court (and frame) side in px
    wall: int = 2                  # wall width in px (left, right, top)
    paddle_w: int = 12
    paddle_h: int = 2
    paddle_y: int = 60             # the paddle's top row
    paddle_speed: float = 4.5      # px per tick at dx = 1
    ball: int = 2                  # the ball's side in px
    ball_speed: float = 1.5        # px per tick
    max_angle: float = 60.0        # degrees from vertical off the paddle's ends
    serve_angle: float = 45.0      # a new ball falls within this many degrees of vertical
    pause: int = 8                 # ticks without a ball after a miss
    h_rate: float = 0.001
    h_relief: float = 1.0          # a hit
    miss_add: float = 0.5          # a miss
    b_rate: float = 0.02           # restlessness, as CircleWorld's
    b_relief: float = 0.1
    b_init_min: float = 0.0
    b_init_max: float = 0.5
    move_budget_px: float = 200.0
    bg: int = 20                   # grey levels
    wall_grey: int = 90
    paddle_grey: int = 200
    ball_grey: int = 255
    # read by the agent loops, as in EnvConfig
    drives: bool = True
    respawn: bool = True
    auto_reset: bool = True
    gray: bool = True

    def validate(self) -> "BreakoutConfig":
        if self.kind != "breakout":
            raise ValueError("BreakoutConfig.kind must be 'breakout'")
        inner = self.size - 2 * self.wall
        if self.size < 16 or self.wall < 1 or self.ball < 1 or self.paddle_h < 1:
            raise ValueError("size >= 16; wall, ball and paddle_h >= 1")
        if not self.ball < self.paddle_w <= inner:
            raise ValueError("the paddle must be wider than the ball and fit the court")
        if not self.wall + 4 * self.ball < self.paddle_y <= self.size - self.paddle_h:
            raise ValueError("paddle_y must leave room for the ball above it and the paddle inside the frame")
        if self.ball_speed <= 0 or self.paddle_speed <= 0 or self.ball_speed >= self.paddle_h + self.ball:
            raise ValueError("speeds must be positive, and the ball slower than paddle_h + ball a tick (or it can pass through)")
        if not 0 < self.max_angle < 90 or not 0 <= self.serve_angle < 90 or self.pause < 0:
            raise ValueError("angles in (0, 90), serve_angle in [0, 90), pause >= 0")
        return self


class BreakoutWorld:
    """N courts. ball [N, 2] (x, y) of the ball's top-left corner, vel [N, 2] px per tick, paddle [N] its left edge."""

    def __init__(self, num_envs: int, config: Optional[BreakoutConfig] = None, seed: int = 0):
        if int(num_envs) < 1:
            raise ValueError("num_envs must be >= 1")
        self.num_envs = n = int(num_envs)
        self.config = c = dataclasses.replace(config if config is not None else BreakoutConfig()).validate()
        self.rng = np.random.default_rng(seed)
        self.t = 0
        self.lo, self.hi = float(c.wall), float(c.size - c.wall)        # the court's inner x range (and top at y = wall)
        self.paddle = np.full(n, (c.size - c.paddle_w) / 2.0)
        self.ball = np.zeros((n, 2))
        self.vel = np.zeros((n, 2))
        self.live = np.zeros(n, bool)
        self.wait = np.zeros(n, np.int64)                               # ticks until the next serve
        self.hunger = np.full(n, 0.5)
        self.restlessness = c.b_init_min + (c.b_init_max - c.b_init_min) * self.rng.random(n)
        self.cum_move = np.zeros(n)
        self.hits = np.zeros(n, np.int64)
        self.misses = np.zeros(n, np.int64)
        self._serve(np.arange(n))
        self._frame = None

    # ------------------------------------------------------------------ the ball
    def _serve(self, idx) -> None:
        c = self.config
        idx = np.asarray(idx)
        if idx.size == 0:
            return
        u = self.rng.random((idx.size, 2))
        self.ball[idx, 0] = self.lo + 2 + u[:, 0] * (self.hi - self.lo - c.ball - 4)
        self.ball[idx, 1] = c.wall + 1.0
        a = np.radians((2 * u[:, 1] - 1) * c.serve_angle)
        self.vel[idx] = np.stack([c.ball_speed * np.sin(a), c.ball_speed * np.cos(a)], 1)   # falling: vy > 0
        self.live[idx] = True

    def _move_ball(self):
        """One tick of the live balls: walls, the top, the paddle. Returns (hit, miss) [N] bool."""
        c = self.config
        n = self.num_envs
        hit, miss = np.zeros(n, bool), np.zeros(n, bool)
        live = self.live
        p = self.ball + self.vel
        x_hi = self.hi - c.ball
        left, right = live & (p[:, 0] < self.lo), live & (p[:, 0] > x_hi)
        p[left, 0] = 2 * self.lo - p[left, 0]
        p[right, 0] = 2 * x_hi - p[right, 0]
        self.vel[left | right, 0] *= -1
        top = live & (p[:, 1] < c.wall)
        p[top, 1] = 2 * c.wall - p[top, 1]
        self.vel[top, 1] *= -1
        # the paddle: the ball's bottom crosses the paddle's top row while falling, and the two overlap in x
        bottom_old, bottom_new = self.ball[:, 1] + c.ball, p[:, 1] + c.ball
        cross = live & (self.vel[:, 1] > 0) & (bottom_old <= c.paddle_y) & (bottom_new > c.paddle_y)
        over = (p[:, 0] + c.ball > self.paddle) & (p[:, 0] < self.paddle + c.paddle_w)
        hit = cross & over
        if hit.any():
            off = ((p[hit, 0] + c.ball / 2.0) - (self.paddle[hit] + c.paddle_w / 2.0)) / ((c.paddle_w + c.ball) / 2.0)
            a = np.radians(np.clip(off, -1.0, 1.0) * c.max_angle)
            self.vel[hit] = np.stack([c.ball_speed * np.sin(a), -c.ball_speed * np.cos(a)], 1)
            p[hit, 1] = c.paddle_y - c.ball - (bottom_new[hit] - c.paddle_y)   # reflected above the paddle
        miss = live & ~hit & (p[:, 1] >= c.size)
        self.ball = np.where(live[:, None], p, self.ball)
        self.live = live & ~miss
        self.wait[miss] = c.pause
        return hit, miss

    # ------------------------------------------------------------------ frames and body
    def _frames(self) -> np.ndarray:
        c = self.config
        n, s = self.num_envs, c.size
        img = np.full((n, s, s), c.bg, np.uint8)
        img[:, :, :c.wall] = c.wall_grey
        img[:, :, s - c.wall:] = c.wall_grey
        img[:, :c.wall, :] = c.wall_grey
        px = np.rint(self.paddle).astype(int)
        bx, by = np.rint(self.ball[:, 0]).astype(int), np.rint(self.ball[:, 1]).astype(int)
        for b in range(n):
            img[b, c.paddle_y:c.paddle_y + c.paddle_h, px[b]:px[b] + c.paddle_w] = c.paddle_grey
            if self.live[b]:
                y0, x0 = max(by[b], 0), max(bx[b], 0)
                img[b, y0:min(by[b] + c.ball, s), x0:min(bx[b] + c.ball, s)] = c.ball_grey
        return np.repeat(img[..., None], 3, axis=3)

    def _drives(self) -> np.ndarray:
        if self.config.drives:
            return np.stack([self.hunger, self.restlessness], 1).astype(np.float32)
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
        c = self.config
        a = np.nan_to_num(np.clip(np.asarray(actions, np.float64), -1.0, 1.0))
        old = self.paddle.copy()
        self.paddle = np.clip(self.paddle + a[:, 0] * c.paddle_speed, self.lo, self.hi - c.paddle_w)
        disp = np.abs(self.paddle - old)
        hit, miss = self._move_ball()
        gone = ~self.live & ~miss                 # balls lost on an earlier tick
        self.wait[gone] -= 1
        self._serve(np.flatnonzero(gone & (self.wait <= 0)))   # the miss's frame and pause - 1 more show no ball
        self.t += 1
        self.hits += hit
        self.misses += miss
        h_ = self.hunger + c.h_rate - c.h_relief * hit
        cut = np.maximum(h_ - 1.0, 0.0)
        self.hunger = np.clip(h_, 0.0, 1.0)
        h_ = self.hunger + c.miss_add * miss
        cut = cut + np.maximum(h_ - 1.0, 0.0)
        self.hunger = np.clip(h_, 0.0, 1.0)
        self.cum_move += disp
        factor = np.maximum(0.0, 1.0 - self.cum_move / c.move_budget_px)
        self.restlessness = np.clip(self.restlessness + c.b_rate - c.b_relief * (disp / c.paddle_speed) * factor, 0.0, 1.0)
        self._frame = None
        obs = self.observe()
        f = np.zeros(n, bool)
        nan = np.full(n, np.nan)
        info = dict(terminal_image=obs["image"], terminal_drives=obs["drives"], hunger_cut=cut, success=hit, timeout=f.copy(),
                    click=f.copy(), ep_length=nan, ep_return=nan.copy(), ep_success=nan.copy(), ep_clicks=nan.copy(),
                    gt_hazard_click=miss,
                    # --- evaluation-only ground truth: NEVER a model input ---
                    gt_drives=np.stack([self.hunger, self.restlessness], 1), gt_ball=self.ball.copy(), gt_ball_vel=self.vel.copy(),
                    gt_ball_live=self.live.copy(), gt_paddle=self.paddle.copy())
        return obs, np.zeros(n, np.float32), f.copy(), info

    def stats(self) -> Dict[str, Any]:
        return dict(bo_hits=int(self.hits.sum()), bo_misses=int(self.misses.sum()))

    def close(self) -> None:
        pass

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

Stage 2 (task 00117, vm-00104), all off by default (stage 1 unchanged):
- paddle_relief: the food of a paddle hit (stage 1: 1, to 0; stage 2: 0, no meal);
- bricks = 1: brick_rows x brick_cols bricks of 3 types (food ranks 0, 1, 2 feeding brick_food[rank]), each look in grey
  distinct (solid, checker, hollow); which look feeds how much rotates by brick_shift (look = (rank + shift) % 3). A ball
  meeting a brick removes it and bounces; when the last is gone a new layout comes in which no slot keeps its type (a
  derangement, the same count of each type). The serve comes from just below the bricks.
- info["hunger_cut"] is signed: the clip's correction (unclipped - clipped hunger), positive at the cap, negative at the
  floor (a meal the floor cut short). Earlier label code adds it only on steps whose hunger did not fall.
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
    paddle_relief: float = 1.0     # a paddle hit's food (task 00117: 0 in stage 2)
    bricks: int = 0                # task 00117: 1 = a wall of bricks
    brick_rows: int = 4
    brick_cols: int = 6
    brick_h: int = 3               # px; the width is inner / cols - 1 (a 1 px gap after each brick)
    brick_top: int = 10            # the top row's y
    brick_food: tuple = (0.1, 0.25, 0.5)   # food by rank
    brick_shift: int = 0           # look = (rank + shift) % 3: which look feeds how much
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
        if self.bricks:
            n = self.brick_rows * self.brick_cols
            if len(self.brick_food) != 3 or n % 3 or inner % self.brick_cols or inner // self.brick_cols < 4:
                raise ValueError("bricks: 3 food values, rows x cols a multiple of 3, cols dividing the court's inner width")
            if self.brick_h < 2 or self.ball_speed >= self.brick_h + self.ball:
                raise ValueError("brick_h >= 2 and the ball slower than brick_h + ball a tick (or it can pass through a brick)")
            if not self.wall + self.ball < self.brick_top or self.brick_top + self.brick_rows * (self.brick_h + 1) + 4 * self.ball >= self.paddle_y:
                raise ValueError("the bricks must leave room above them and above the paddle")
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
        self.food_seen = 0.0
        if c.bricks:   # task 00117: a separate stream, so stage 1's draws are unchanged
            self.brng = np.random.default_rng([seed, 0xB1])
            k = c.brick_rows * c.brick_cols
            self.bw = int(self.hi - self.lo) // c.brick_cols - 1      # pitch inner / cols, a 1 px gap after each
            col, row = np.arange(k) % c.brick_cols, np.arange(k) // c.brick_cols
            self.bx = (self.lo + col * (self.bw + 1)).astype(int)
            self.by = c.brick_top + row * (c.brick_h + 1)
            self.layout = np.stack([self.brng.permutation(np.repeat(np.arange(3), k // 3)) for _ in range(n)])
            self.alive = np.ones((n, k), bool)
            self.tiles = self._tiles()
            self.bricks_hit = np.zeros((n, 3), np.int64)
            self.clears = np.zeros(n, np.int64)
            self.pref_hit = self.pref_stand = 0.0
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
        self.ball[idx, 1] = (c.brick_top + c.brick_rows * (c.brick_h + 1) + 1.0) if c.bricks else c.wall + 1.0
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
        self.brick_now = np.full(n, -1)
        self.brick_food_now = np.zeros(n)
        if c.bricks:
            self._bricks(p, live)
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

    def _bricks(self, p, live):
        """Task 00117: balls meeting standing bricks remove them and bounce once (off the top or bottom if the ball was
        within the brick's columns before the move, else off its side); food, the preference sums, refills."""
        c = self.config
        b = c.ball
        over = (live[:, None] & self.alive & (p[:, :1] < self.bx + self.bw) & (p[:, :1] + b > self.bx)
                & (p[:, 1:] < self.by + c.brick_h) & (p[:, 1:] + b > self.by))
        for e in np.flatnonzero(over.any(1)):
            ks = np.flatnonzero(over[e])
            ranks = self.layout[e, ks]
            stand = self.layout[e, self.alive[e]].mean()
            k = ks[0]
            if self.ball[e, 0] < self.bx[k] + self.bw and self.ball[e, 0] + b > self.bx[k]:
                p[e, 1] = self.ball[e, 1]
                self.vel[e, 1] *= -1
            else:
                p[e, 0] = self.ball[e, 0]
                self.vel[e, 0] *= -1
            self.alive[e, ks] = False
            self.brick_now[e] = int(ranks.max())
            self.brick_food_now[e] = float(np.sum(np.asarray(c.brick_food)[ranks]))
            np.add.at(self.bricks_hit[e], ranks, 1)
            self.pref_hit += float(ranks.sum())
            self.pref_stand += stand * len(ks)
            if not self.alive[e].any():
                self._refill(e)

    def _refill(self, e):
        """A new full layout in which no slot keeps its type, with the same count of each type: of the slots that held
        type 0, a get type 1 and the rest type 2; of type 1's, 8 - a (per third) get 0, the rest 2; of type 2's, a get 0, the
        rest 1 (the only counts that keep each type's total)."""
        old = self.layout[e]
        m = len(old) // 3
        a = int(self.brng.integers(0, m + 1))
        new = np.empty_like(old)
        for t, (first, n_first, second) in enumerate(((1, a, 2), (0, m - a, 2), (0, a, 1))):
            slots = self.brng.permutation(np.flatnonzero(old == t))
            new[slots[:n_first]], new[slots[n_first:]] = first, second
        self.layout[e] = new
        self.alive[e] = True
        self.clears[e] += 1

    def _tiles(self):
        """Each look's brick image (h x w grey): solid, checker, hollow."""
        c = self.config
        h, w = c.brick_h, self.bw
        y, x = np.mgrid[:h, :w]
        solid = np.full((h, w), 120, np.uint8)
        checker = np.where((x + y) % 2 == 0, 230, 60).astype(np.uint8)
        hollow = np.where((y == 0) | (y == h - 1) | (x == 0) | (x == w - 1), 230, c.bg).astype(np.uint8)
        return (solid, checker, hollow)

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
            if c.bricks:
                for k in np.flatnonzero(self.alive[b]):
                    img[b, self.by[k]:self.by[k] + c.brick_h, self.bx[k]:self.bx[k] + self.bw] = self.tiles[(self.layout[b, k] + c.brick_shift) % 3]
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
        food = c.h_relief * c.paddle_relief * hit + self.brick_food_now
        self.food_seen += float(food.sum())
        h_ = self.hunger + c.h_rate - food
        self.hunger = np.clip(h_, 0.0, 1.0)
        cut = h_ - self.hunger                    # signed (task 00117): + at the cap, - at the floor
        h_ = self.hunger + c.miss_add * miss
        self.hunger = np.clip(h_, 0.0, 1.0)
        cut = cut + h_ - self.hunger
        self.cum_move += disp
        factor = np.maximum(0.0, 1.0 - self.cum_move / c.move_budget_px)
        self.restlessness = np.clip(self.restlessness + c.b_rate - c.b_relief * (disp / c.paddle_speed) * factor, 0.0, 1.0)
        self._frame = None
        obs = self.observe()
        f = np.zeros(n, bool)
        nan = np.full(n, np.nan)
        info = dict(terminal_image=obs["image"], terminal_drives=obs["drives"], hunger_cut=cut, success=food > 0, timeout=f.copy(),
                    click=f.copy(), ep_length=nan, ep_return=nan.copy(), ep_success=nan.copy(), ep_clicks=nan.copy(),
                    gt_hazard_click=miss,
                    # --- evaluation-only ground truth: NEVER a model input ---
                    gt_drives=np.stack([self.hunger, self.restlessness], 1), gt_ball=self.ball.copy(), gt_ball_vel=self.vel.copy(),
                    gt_ball_live=self.live.copy(), gt_paddle=self.paddle.copy(), gt_paddle_hit=hit)
        if c.bricks:
            info["gt_bricks"] = np.where(self.alive, self.layout, -1)
            info["gt_brick_hit"] = self.brick_now.copy()
        return obs, np.zeros(n, np.float32), f.copy(), info

    def stats(self) -> Dict[str, Any]:
        out = dict(bo_hits=int(self.hits.sum()), bo_misses=int(self.misses.sum()))
        if self.config.bricks:
            t = self.bricks_hit.sum(0)
            out.update(bo_bricks=int(t.sum()), bo_brick_t0=int(t[0]), bo_brick_t1=int(t[1]), bo_brick_t2=int(t[2]),
                       bo_food=self.food_seen, bo_clears=int(self.clears.sum()), bo_pref_hit=self.pref_hit, bo_pref_stand=self.pref_stand)
        return out

    def close(self) -> None:
        pass

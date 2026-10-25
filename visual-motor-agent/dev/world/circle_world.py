"""CircleWorld: a vectorized, deterministic "click the red circle" simulator.

Phase 0 environment of the visual-motor project (plan sections 3, 4 and 6).
The world is a ``size x size`` RGB screen with a uniform dark background, one
solid red circle and a custom-drawn crosshair cursor.  The agent sees only the
rendered image (plus two internal drive values when ``drives=True``) and acts
with continuous pointer commands ``a = (dx, dy, click)``.  Success is a click
while the cursor lies inside the circle.

Coordinates
-----------
Continuous logical coordinates; x grows to the right and y grows downward.
``image[row=y, col=x]``.  Pixel (col j, row i) is sampled at its center
(j + 0.5, i + 0.5).  The cursor always lies in [0, size - 1e-4] on both axes
and its hotspot pixel is (floor(x), floor(y)).

All simulation math is float64.  ``browser_demo/circle_world.js`` implements
the same math; ``check_js_equivalence.py`` checks that its renders match
pixel for pixel and that its step outputs match.

Step order (per env)
--------------------
1. Move:   cursor = clip(cursor + [dx, dy] * max_step_px, 0, size - 1e-4);
           disp = actual displacement (0 when pushing into an edge).
2. Hit:    inside = (x-cx)^2 + (y-cy)^2 <= r^2 at the new position, and the
           target is alive; success = click & inside; empty = click & ~success.
3. Reward: success_reward * success - step_cost
           + movement / empty-click bonuses        (bonuses=True)
           - hunger_cost * hunger_before_step      (drives=True)
4. Update counters and drives; distractor dots move (distractors > 0).  Drives are always simulated, but they appear
   in the observation only when drives=True.
5. A success kills the target: it is no longer drawn and cannot be hit.
6. done = (success & end_on_success) | (t >= max_steps).
7. ``info`` is built from the post-step state, BEFORE any auto-reset.
8. With auto_reset, done envs start a new episode and their returned
   observation is that episode's first frame.

Information hygiene
-------------------
The observation is exactly ``{'image', 'drives'}``.  ``info`` also carries
evaluation-only ground truth under keys prefixed ``gt_`` (cursor and target
geometry, hit test, raw drives).

    !!! NEVER feed any ``gt_*`` value, or anything derived from one, to a
    !!! model input, an auxiliary loss or a reward.  These keys exist only
    !!! for metrics, plots and debugging.

Usage
-----
    from world import CircleWorld, EnvConfig
    env = CircleWorld(128, EnvConfig(bonuses=True), seed=0)
    obs = env.reset()                      # {'image': uint8 (N,64,64,3), 'drives': float32 (N,2)}
    obs, reward, done, info = env.step(actions)   # actions: float (N,3) = [dx, dy, click]

Benchmark (from dev/):  python -m world.circle_world --bench
"""

from __future__ import annotations

import argparse
import dataclasses
import functools
import math
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np

RGB = Tuple[int, int, int]

#: The cursor is clamped to [0, size - CURSOR_EDGE_EPS] on both axes.
CURSOR_EDGE_EPS = 1e-4

_COLOR_FIELDS = ("bg_color", "target_color", "cursor_color", "cursor_outline", "hazard_color")


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class EnvConfig:
    """World parameters.  Distances are in logical pixels of the viewport."""

    # Geometry and episode structure
    size: int = 64                      # viewport / observation side (px)
    max_steps: int = 200                # episode time limit (actions)
    max_step_px: float = 4.0            # displacement for |dx| = 1 (per axis)
    radius_min: float = 12.0            # target radius ~ U[radius_min, radius_max]
    radius_max: float = 16.0
    cursor_start_outside: bool = True   # resample the initial cursor until outside
    end_on_success: bool = True         # success ends the episode (v0 choice)
    auto_reset: bool = True             # step() resets done envs itself

    # Colors (RGB, 0..255)
    bg_color: RGB = (20, 20, 30)
    target_color: RGB = (220, 40, 40)
    cursor_color: RGB = (255, 255, 255)
    cursor_outline: RGB = (0, 0, 0)

    # External reward
    success_reward: float = 10.0
    step_cost: float = 0.001

    # Bounded exploration bonuses (plan section 6); budgets do not regenerate
    bonuses: bool = False
    move_bonus: float = 0.01            # per full-length step, scaled by disp/max_step_px
    move_budget_px: float = 200.0       # bonus factor max(0, 1 - cum_move / budget)
    click_bonus: float = 0.02           # per empty click
    click_budget: int = 10              # bonus factor max(0, 1 - empty_clicks / budget)

    # Internal drives: always simulated, exposed in obs['drives'] only if drives=True
    drives: bool = False
    h_init_min: float = 0.0             # hunger at reset ~ U[h_init_min, h_init_max]
    h_init_max: float = 0.5
    h_rate: float = 0.005               # hunger growth per step
    h_relief: float = 1.0               # hunger reduction on success
    hunger_cost: float = 0.0            # reward -= hunger_cost * hunger (drives=True only)
    b_init_min: float = 0.0             # restlessness at reset ~ U[b_init_min, b_init_max]
    b_init_max: float = 0.5
    b_rate: float = 0.02                # restlessness growth per step
    b_relief: float = 0.1               # relief per full step, scaled by the move-budget factor

    # Distractors (vm-00010): grey dots that move on their own, drawn under the circle and the cursor. They never
    # affect the task. Their random draws use separate generators, so the episodes are the same with or without them.
    distractors: int = 0                # dots per env (0 = none: the world is unchanged)
    dot_size: int = 2                   # dot side (px)
    dot_brightness_min: float = 60.0    # grey level ~ U[min, max] per dot (0..255)
    dot_brightness_max: float = 180.0
    dot_speed_min: float = 0.5          # px per step ~ U[min, max] in a uniform direction; they bounce off the walls
    dot_speed_max: float = 2.0

    # A richer world (vm-00011). All off by default: the world is unchanged. Their random draws use separate
    # generators, so the first circle, the cursor and the drives of every episode are the same with or without them.
    respawn: bool = False               # a hit circle is replaced at once by a new one, never on the cursor (with
                                        # end_on_success=False and a large max_steps: a continual world)
    target_speed: float = 0.0           # px per step of the circle in a uniform direction; it bounces off the walls
    hazards: int = 0                    # blue circles, moving and bouncing, that add to hunger while the cursor is
                                        # inside one; drawn under the circle
    hazard_radius_min: float = 5.0
    hazard_radius_max: float = 8.0
    hazard_speed: float = 0.75          # px per step
    hazard_cost: float = 0.02           # hunger added per step inside a hazard
    hazard_click: float = -1.0          # >= 0: a click inside a hazard sets hunger to this (poison; after any relief
                                        # from the circle). The hazard stays. < 0: off
    hazard_color: RGB = (50, 90, 230)
    wrap: bool = False                  # the screen is a torus: the cursor, circles and dots wrap around the edges
                                        # (no walls); hit tests use the wrap-around distance
    egocentric: bool = False            # (needs wrap) the view is centered on the cursor: moving shifts the scene,
                                        # the cursor stays drawn at the center

    def __post_init__(self) -> None:
        # Colors loaded from JSON arrive as lists; store tuples (hashable, comparable).
        for name in _COLOR_FIELDS:
            value = getattr(self, name)
            if isinstance(value, (list, tuple, np.ndarray)):
                setattr(self, name, tuple(int(v) if float(v).is_integer() else v for v in value))

    def validate(self) -> "EnvConfig":
        """Raise ValueError if the configuration is inconsistent; return self."""
        if int(self.size) != self.size or self.size < 1:
            raise ValueError(f"size must be a positive integer, got {self.size!r}")
        if int(self.max_steps) != self.max_steps or self.max_steps < 1:
            raise ValueError(f"max_steps must be a positive integer, got {self.max_steps!r}")
        floats = ("max_step_px", "radius_min", "radius_max", "success_reward", "step_cost",
                  "move_bonus", "move_budget_px", "click_bonus", "click_budget",
                  "h_init_min", "h_init_max", "h_rate", "h_relief", "hunger_cost",
                  "b_init_min", "b_init_max", "b_rate", "b_relief", "dot_brightness_min",
                  "dot_brightness_max", "dot_speed_min", "dot_speed_max", "target_speed",
                  "hazard_radius_min", "hazard_radius_max", "hazard_speed", "hazard_cost", "hazard_click")
        for name in floats:
            if not math.isfinite(float(getattr(self, name))):
                raise ValueError(f"{name} must be finite, got {getattr(self, name)!r}")
        if not self.max_step_px > 0:
            raise ValueError("max_step_px must be > 0")
        if not 0 < self.radius_min <= self.radius_max <= self.size / 2:
            raise ValueError(f"need 0 < radius_min <= radius_max <= size/2, got "
                             f"[{self.radius_min}, {self.radius_max}] for size {self.size}")
        if not (self.move_budget_px > 0 and self.click_budget > 0):
            raise ValueError("move_budget_px and click_budget must be > 0")
        if not 0 <= self.h_init_min <= self.h_init_max <= 1:
            raise ValueError("need 0 <= h_init_min <= h_init_max <= 1")
        if not 0 <= self.b_init_min <= self.b_init_max <= 1:
            raise ValueError("need 0 <= b_init_min <= b_init_max <= 1")
        if int(self.distractors) != self.distractors or self.distractors < 0:
            raise ValueError(f"distractors must be an integer >= 0, got {self.distractors!r}")
        if int(self.dot_size) != self.dot_size or self.dot_size < 1 or (self.distractors and self.dot_size > self.size):
            raise ValueError(f"dot_size must be an integer in 1..size, got {self.dot_size!r}")
        if not 0 <= self.dot_brightness_min <= self.dot_brightness_max <= 255:
            raise ValueError("need 0 <= dot_brightness_min <= dot_brightness_max <= 255")
        if not 0 <= self.dot_speed_min <= self.dot_speed_max:
            raise ValueError("need 0 <= dot_speed_min <= dot_speed_max")
        if not (self.target_speed >= 0 and self.hazard_speed >= 0 and self.hazard_cost >= 0):
            raise ValueError("target_speed, hazard_speed and hazard_cost must be >= 0")
        if self.hazard_click > 1:
            raise ValueError("hazard_click must be <= 1 (or < 0: off)")
        if int(self.hazards) != self.hazards or self.hazards < 0:
            raise ValueError(f"hazards must be an integer >= 0, got {self.hazards!r}")
        if self.egocentric and not self.wrap:
            raise ValueError("egocentric needs wrap")
        if self.hazards and not 0 < self.hazard_radius_min <= self.hazard_radius_max <= self.size / 2:
            raise ValueError("need 0 < hazard_radius_min <= hazard_radius_max <= size/2")
        for name in _COLOR_FIELDS:
            c = getattr(self, name)
            if len(c) != 3 or any(int(v) != v or not 0 <= v <= 255 for v in c):
                raise ValueError(f"{name} must be 3 integers in 0..255, got {c!r}")
        return self

    def to_dict(self) -> Dict[str, Any]:
        """JSON-friendly dict (colors as lists)."""
        d = dataclasses.asdict(self)
        for name in _COLOR_FIELDS:
            d[name] = [int(v) for v in d[name]]
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "EnvConfig":
        """Inverse of to_dict(); unknown keys raise ValueError."""
        known = {f.name for f in dataclasses.fields(cls)}
        unknown = set(d) - known
        if unknown:
            raise ValueError(f"unknown EnvConfig keys: {sorted(unknown)}")
        kwargs = dict(d)
        for name in _COLOR_FIELDS:
            if name in kwargs:
                kwargs[name] = tuple(int(v) for v in kwargs[name])
        return cls(**kwargs)

    def max_episode_bonus(self) -> float:
        """Analytic upper bound on the total exploration bonus of one episode.

        Movement: sum_k (d_k / s) * f(c_k), with f(c) = max(0, 1 - c/B) and
        s = max_step_px, is a left Riemann sum of the decreasing f with step
        widths d_k <= sqrt(2) * s (a diagonal step).  It exceeds the integral
        B/2 by at most one step, d_max * f(0), so the movement bonus is at
        most move_bonus * (B / (2s) + sqrt(2)).
        Clicks: the sum over k < click_budget of click_bonus * (1 - k / click_budget).
        Both bounds hold however long the episode is and however the agent waits.
        """
        if not self.bonuses:
            return 0.0
        move = self.move_bonus * (self.move_budget_px / (2.0 * self.max_step_px) + math.sqrt(2.0))
        n = math.ceil(self.click_budget)
        click = self.click_bonus * (n - n * (n - 1) / (2.0 * self.click_budget))
        return move + click


# =============================================================================
# Rendering (pure functions)
# =============================================================================

def _cursor_offsets() -> Tuple[np.ndarray, np.ndarray]:
    """(dx, dy) offsets of the cursor sprite relative to its hotspot pixel.

    White set W: a 7-px "+" crosshair, {(0,0)} U {(+-k,0), (0,+-k) : k=1..3}.
    Outline O: (3x3 dilation of W) minus W, i.e. a 1-px black border.
    """
    white = {(0, 0)}
    for k in (1, 2, 3):
        white |= {(k, 0), (-k, 0), (0, k), (0, -k)}
    dilated = {(x + a, y + b) for (x, y) in white for a in (-1, 0, 1) for b in (-1, 0, 1)}
    return (np.array(sorted(white), dtype=np.int64),
            np.array(sorted(dilated - white), dtype=np.int64))


#: (13, 2) and (32, 2) int arrays of (dx, dy) sprite offsets.
CURSOR_WHITE_OFFSETS, CURSOR_OUTLINE_OFFSETS = _cursor_offsets()


def _rgb(color: Sequence[int]) -> RGB:
    return (int(color[0]), int(color[1]), int(color[2]))


@functools.lru_cache(maxsize=32)
def _row_bank(size: int, bg: RGB, fg: RGB) -> np.ndarray:
    """Read-only view ``bank[count, size - start]``: one image row (size*3 bytes).

    The row is background except for ``count`` foreground pixels starting at
    column ``start``.  Backing store: for each count c, a 3*size-pixel strip
    (size bg, c fg, rest bg); a row is the size-pixel window at offset
    ``size - start``.  Memory is O(size^2) and the row gather is a contiguous copy.
    """
    strip = np.empty((size + 1, 3 * size, 3), dtype=np.uint8)
    strip[:] = bg
    for c in range(1, size + 1):
        strip[c, size:size + c] = fg
    return np.lib.stride_tricks.as_strided(
        strip, shape=(size + 1, 2 * size + 1, 3 * size),
        strides=(strip.strides[0], 3, 1), writeable=False)


def _circle_spans(cx: np.ndarray, cy: np.ndarray, radius: np.ndarray,
                  alive: np.ndarray, size: int) -> Tuple[np.ndarray, np.ndarray]:
    """Per image row, the run of pixels whose centers satisfy the circle test.

    The exact per-pixel predicate (the JS renderer uses the same expression) is
        dx*dx + dy*dy <= r*r,   dx = (j + 0.5) - cx,   dy = (i + 0.5) - cy
    evaluated in float64.
    Because IEEE rounding is monotone, the pixels satisfying it in a row form
    one contiguous run.  The run's ends are estimated with the closed-form chord
    and then decided *exactly* by evaluating the predicate on the three
    candidate columns around each estimate.  The estimate is off by far less
    than one pixel (~1e-7 px at worst), so the true ends are always among the
    candidates.  Returns (start, count), int64 arrays of shape (n, size).
    """
    centers = np.arange(size, dtype=np.float64) + 0.5
    ddy = centers[None, :] - cy[:, None]                      # (n, size), one per row
    ddy2 = ddy * ddy
    r2 = (radius * radius)[:, None]
    half = np.sqrt(np.maximum(r2 - ddy2, 0.0))                 # approximate half-chord
    c = cx[:, None]
    lo = np.clip(np.ceil(c - half - 0.5), 0, size - 1).astype(np.int64)
    hi = np.clip(np.floor(c + half - 0.5), 0, size - 1).astype(np.int64)

    cand = np.stack((lo - 1, lo, lo + 1, hi + 1, hi))         # (5, n, size)
    d = (cand + 0.5) - c
    ok = (d * d + ddy2 <= r2) & (cand >= 0) & (cand < size)
    s0, s1, s2, e0, e1 = ok

    start = np.where(s0, lo - 1, np.where(s1, lo, lo + 1))
    end = np.where(e0, hi + 1, np.where(e1, hi, hi - 1))
    nonempty = (s0 | s1 | s2) & alive[:, None]
    count = np.where(nonempty, np.maximum(end - start + 1, 0), 0)
    start = np.where(nonempty, start, 0)
    return start, count


def _draw_cursor(image: np.ndarray, cursor: np.ndarray, outline: RGB, color: RGB) -> None:
    """Draw the crosshair sprite in place (outline first, then white), clipped."""
    n, size = image.shape[0], image.shape[1]
    if n == 0:
        return
    # Clipping before floor keeps far-away cursors invisible and the ints small.
    hx = np.floor(np.clip(cursor[:, 0], -16.0, size + 16.0)).astype(np.int64)
    hy = np.floor(np.clip(cursor[:, 1], -16.0, size + 16.0)).astype(np.int64)
    flat = image.reshape(-1, 3)
    base = (np.arange(n, dtype=np.int64) * (size * size))[:, None]
    for offsets, rgb in ((CURSOR_OUTLINE_OFFSETS, outline), (CURSOR_WHITE_OFFSETS, color)):
        px = hx[:, None] + offsets[:, 0]
        py = hy[:, None] + offsets[:, 1]
        ok = (px >= 0) & (px < size) & (py >= 0) & (py < size)
        flat[(base + py * size + px)[ok]] = np.asarray(rgb, dtype=np.uint8)


def render_state(cursor_xy: np.ndarray, target_xy: np.ndarray, radius: np.ndarray,
                 target_alive: np.ndarray, config: EnvConfig, dots: Optional[Tuple[np.ndarray, np.ndarray]] = None,
                 hazards: Optional[Tuple[np.ndarray, np.ndarray]] = None) -> np.ndarray:
    """Render N world states to uint8 images of shape (N, size, size, 3).

    Pure function of (cursor hotspot pixel, set of circle pixels, colors).

    Args:
        cursor_xy: (N, 2) cursor positions (x, y).
        target_xy: (N, 2) circle centers (x, y).
        radius: (N,) circle radii.
        target_alive: (N,) bool; dead targets are not drawn.
        config: EnvConfig providing ``size`` and the colors.

    Layers: background; then the circle, at pixels with
    (j+0.5-cx)^2 + (i+0.5-cy)^2 <= r^2 where alive; then the distractor dots
    (``dots`` = (positions (N,K,2), grey levels (N,K))), dot_size squares at
    (floor(x), floor(y)) drawn on background pixels only (under the circle);
    then the cursor sprite (black outline, then the white "+") at hotspot
    (floor(x), floor(y)).  Hazards (``hazards`` = (centers (N,H,2), radii (N,H))) are painted in hazard_color on the
    background pixels after the circle (so under it) and before the dots, with the circle's pixel test.
    """
    if config.wrap:
        return _render_wrap(cursor_xy, target_xy, radius, target_alive, config, dots, hazards)
    size = int(config.size)
    cursor = np.asarray(cursor_xy, dtype=np.float64).reshape(-1, 2)
    target = np.asarray(target_xy, dtype=np.float64).reshape(-1, 2)
    radius = np.asarray(radius, dtype=np.float64).reshape(-1)
    alive = np.asarray(target_alive, dtype=bool).reshape(-1)
    n = cursor.shape[0]
    if not target.shape[0] == radius.shape[0] == alive.shape[0] == n:
        raise ValueError("render_state: inconsistent batch sizes "
                         f"{cursor.shape}, {target.shape}, {radius.shape}, {alive.shape}")
    start, count = _circle_spans(target[:, 0], target[:, 1], radius, alive, size)
    bank = _row_bank(size, _rgb(config.bg_color), _rgb(config.target_color))
    image = bank[count, size - start].reshape(n, size, size, 3)
    if hazards is not None and hazards[0].shape[1] > 0:
        _paint_circles(image, hazards[0], hazards[1], _rgb(config.hazard_color), _rgb(config.bg_color))
    if dots is not None and dots[0].shape[1] > 0:
        _draw_dots(image, dots[0], dots[1], int(config.dot_size), _rgb(config.bg_color))
    _draw_cursor(image, cursor, _rgb(config.cursor_outline), _rgb(config.cursor_color))
    return image


def wrap_delta(a: np.ndarray, b: np.ndarray, size: float) -> np.ndarray:
    """a - b on a circle of circumference size, in [-size/2, size/2)."""
    return np.mod(a - b + 0.5 * size, size) - 0.5 * size


def _render_wrap(cursor_xy, target_xy, radius, target_alive, config, dots=None, hazards=None) -> np.ndarray:
    """render_state for the wrap-around world: circles drawn with the wrap-around pixel test, the view shifted so the
    cursor sits at the center if egocentric. Same layers as render_state."""
    size = int(config.size)
    S = float(size)
    cursor = np.asarray(cursor_xy, dtype=np.float64).reshape(-1, 2)
    target = np.asarray(target_xy, dtype=np.float64).reshape(-1, 2)
    radius = np.asarray(radius, dtype=np.float64).reshape(-1)
    alive = np.asarray(target_alive, dtype=bool).reshape(-1)
    n = cursor.shape[0]
    shift = (0.5 * S - cursor) if config.egocentric else np.zeros_like(cursor)          # world -> view
    bg = np.array(_rgb(config.bg_color), np.uint8)
    image = np.empty((n, size, size, 3), np.uint8)
    image[:] = bg
    c = np.arange(size, dtype=np.float64) + 0.5

    def mask(centers, radii):            # centers (n, K, 2) in world coordinates, radii (n, K) -> (n, S, S)
        v = np.mod(centers + shift[:, None], S)
        dx = wrap_delta(c[None, None, None, :], v[..., 0, None, None], S)
        dy = wrap_delta(c[None, None, :, None], v[..., 1, None, None], S)
        return (dx * dx + dy * dy <= (radii * radii)[..., None, None]).any(1)

    tmask = mask(target[:, None], radius[:, None]) & alive[:, None, None]
    image[tmask] = np.array(_rgb(config.target_color), np.uint8)
    if hazards is not None and hazards[0].shape[1] > 0:
        hmask = mask(hazards[0], hazards[1]) & ~tmask
        image[hmask] = np.array(_rgb(config.hazard_color), np.uint8)
    if dots is not None and dots[0].shape[1] > 0:
        _draw_dots(image, np.mod(dots[0] + shift[:, None], S), dots[1], int(config.dot_size), _rgb(config.bg_color))
    _draw_cursor(image, np.mod(cursor + shift, S), _rgb(config.cursor_outline), _rgb(config.cursor_color))
    return image


def _paint_circles(image: np.ndarray, centers: np.ndarray, radii: np.ndarray, color: RGB, bg: RGB) -> None:
    """Paint circles (centers (N,H,2), radii (N,H)) in color on the background pixels of image (N,S,S,3), in place."""
    size = image.shape[1]
    c = np.arange(size, dtype=np.float64) + 0.5
    dx = c[None, None, None, :] - centers[..., 0, None, None]                 # (N,H,1,S)
    dy = c[None, None, :, None] - centers[..., 1, None, None]                 # (N,H,S,1)
    inside = (dx * dx + dy * dy <= (radii * radii)[..., None, None]).any(1)   # (N,S,S)
    on_bg = np.all(image == np.array(bg, np.uint8), axis=-1)
    image[inside & on_bg] = np.array(color, np.uint8)


def _draw_dots(image: np.ndarray, pos: np.ndarray, val: np.ndarray, ds: int, bg: np.ndarray) -> None:
    """Grey squares of side ds at floor(pos) on the background pixels of image (N,S,S,3), in place."""
    n, k = val.shape
    size = image.shape[1]
    x0 = np.clip(np.floor(pos[..., 0]).astype(np.int64), 0, size - ds)
    y0 = np.clip(np.floor(pos[..., 1]).astype(np.int64), 0, size - ds)
    off = np.arange(ds)
    ys = (y0[..., None, None] + off[:, None]).repeat(ds, -1)          # (N,K,ds,ds)
    xs = (x0[..., None, None] + off[None, :]).repeat(ds, -2)
    ns = np.broadcast_to(np.arange(n)[:, None, None, None], ys.shape)
    grey = np.broadcast_to(np.round(val).astype(np.uint8)[..., None, None], ys.shape)
    on_bg = np.all(image[ns, ys, xs] == bg, axis=-1)
    image[ns[on_bg], ys[on_bg], xs[on_bg]] = grey[on_bg][:, None]


def sanitize_actions(actions: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Map raw (N, 3) actions to (move (N, 2) float64 in [-1, 1], click (N,) bool).

    NaN becomes 0, dx and dy are clipped to [-1, 1] (inf included), and
    click = actions[:, 2] > 0.5.
    """
    a = np.asarray(actions, dtype=np.float64)
    if a.ndim != 2 or a.shape[1] != 3:
        raise ValueError(f"actions must have shape (N, 3), got {a.shape}")
    a = np.where(np.isnan(a), 0.0, a)
    return np.clip(a[:, :2], -1.0, 1.0), a[:, 2] > 0.5


# =============================================================================
# RNG state <-> numpy encoding
# =============================================================================

_MASK64 = (1 << 64) - 1


def _encode_rng(gen: np.random.Generator) -> list:
    st = gen.bit_generator.state
    s, inc = int(st["state"]["state"]), int(st["state"]["inc"])
    return [s >> 64, s & _MASK64, inc >> 64, inc & _MASK64,
            int(st["has_uint32"]), int(st["uinteger"])]


def _decode_rng(row: np.ndarray) -> dict:
    v = [int(x) for x in row]
    return {"bit_generator": "PCG64",
            "state": {"state": (v[0] << 64) | v[1], "inc": (v[2] << 64) | v[3]},
            "has_uint32": v[4], "uinteger": v[5]}


# =============================================================================
# Vectorized simulator
# =============================================================================

class CircleWorld:
    """N independent CircleWorld episodes, simulated in lockstep with numpy.

    Each env has its own PCG64 generator (spawned from ``SeedSequence(seed)``)
    that is used only at resets, so env i's episodes depend only on ``seed``
    and env i's own actions.

    Public state arrays (read them freely; change them only via set_state):
        cursor (N,2), target (N,2), radius (N,), target_alive (N,) bool,
        t (N,) int64, hunger, restlessness, cum_move (N,), empty_clicks,
        clicks (N,) int64, ep_return, path_length, init_dist, bonus_total (N,).
    Floats are float64.
    """

    #: name -> (per-env shape, dtype) of every state array except the RNG states.
    STATE_SPEC: Dict[str, Tuple[Tuple[int, ...], Any]] = {
        "cursor": ((2,), np.float64),
        "target": ((2,), np.float64),
        "radius": ((), np.float64),
        "target_alive": ((), np.bool_),
        "t": ((), np.int64),
        "hunger": ((), np.float64),
        "restlessness": ((), np.float64),
        "cum_move": ((), np.float64),
        "empty_clicks": ((), np.int64),
        "clicks": ((), np.int64),
        "ep_return": ((), np.float64),
        "path_length": ((), np.float64),
        "init_dist": ((), np.float64),
        "bonus_total": ((), np.float64),
    }

    def __init__(self, num_envs: int, config: Optional[EnvConfig] = None, seed: int = 0):
        """Create N envs and start the first episode in each.

        Args:
            num_envs: number of parallel envs N (>= 1).
            config: EnvConfig (copied; defaults to EnvConfig()).
            seed: int (or int sequence) for np.random.SeedSequence.
        """
        if int(num_envs) < 1:
            raise ValueError("num_envs must be >= 1")
        self.num_envs = int(num_envs)
        self.config = dataclasses.replace(config if config is not None else EnvConfig()).validate()
        self.seed = seed
        self._rngs = [np.random.Generator(np.random.PCG64(s))
                      for s in np.random.SeedSequence(seed).spawn(self.num_envs)]
        for name, (shape, dtype) in self.STATE_SPEC.items():
            setattr(self, name, np.zeros((self.num_envs,) + shape, dtype=dtype))
        k = int(self.config.distractors)
        self._dot_rngs = ([np.random.Generator(np.random.PCG64(s))
                           for s in np.random.SeedSequence(seed, spawn_key=(0xD075,)).spawn(self.num_envs)]
                          if k else [])
        self.dot_pos = np.zeros((self.num_envs, k, 2))   # top-left corner (x, y), float px
        self.dot_vel = np.zeros((self.num_envs, k, 2))   # px per step
        self.dot_val = np.zeros((self.num_envs, k))      # grey level
        c = self.config   # the richer world (vm-00011): own generators for respawns / target motion and for hazards
        self._spawn_rngs = ([np.random.Generator(np.random.PCG64(s))
                             for s in np.random.SeedSequence(seed, spawn_key=(0x5BA4,)).spawn(self.num_envs)]
                            if (c.respawn or c.target_speed > 0) else [])
        h = int(c.hazards)
        self._hz_rngs = ([np.random.Generator(np.random.PCG64(s))
                          for s in np.random.SeedSequence(seed, spawn_key=(0x4A2D,)).spawn(self.num_envs)]
                         if h else [])
        self.target_vel = np.zeros((self.num_envs, 2))
        self.hz_pos = np.zeros((self.num_envs, h, 2))
        self.hz_vel = np.zeros((self.num_envs, h, 2))
        self.hz_rad = np.zeros((self.num_envs, h))
        self.ep_hits = np.zeros(self.num_envs, np.int64)
        self._reset_envs(np.arange(self.num_envs))

    # ------------------------------------------------------------------ resets
    def reset(self, env_ids: Optional[Sequence[int]] = None) -> Dict[str, np.ndarray]:
        """Start new episodes and return the observation of all envs.

        Args:
            env_ids: envs to reset, as indices or an (N,) bool mask such as
                ``done`` (useful with auto_reset=False).  None resets all envs.
        """
        if env_ids is None:
            idx = np.arange(self.num_envs)
        else:
            ids = np.asarray(env_ids)
            idx = np.flatnonzero(ids) if ids.dtype == np.bool_ else ids.astype(np.int64).reshape(-1)
        self._reset_envs(idx)
        return self.observe()

    def set_radius_range(self, radius_min: float, radius_max: float) -> None:
        """Change the target radius range used by future resets (curriculum)."""
        self.config = dataclasses.replace(
            self.config, radius_min=float(radius_min), radius_max=float(radius_max)).validate()

    def _reset_envs(self, idx: np.ndarray) -> None:
        """Draw new episodes for envs ``idx``.

        Per-env draw order: 5 uniforms (radius, center x, center y, hunger,
        restlessness), then cursor (x, y) pairs until one is accepted.
        """
        if idx.size == 0:
            return
        cfg = self.config
        size = float(cfg.size)
        rngs = self._rngs
        u = np.array([rngs[i].random(5) for i in idx])
        r = cfg.radius_min + (cfg.radius_max - cfg.radius_min) * u[:, 0]
        cx = r + (size - 2.0 * r) * u[:, 1]
        cy = r + (size - 2.0 * r) * u[:, 2]
        hunger = cfg.h_init_min + (cfg.h_init_max - cfg.h_init_min) * u[:, 3]
        restless = cfg.b_init_min + (cfg.b_init_max - cfg.b_init_min) * u[:, 4]

        # Cursor ~ U[0, size)^2.  Rejection loop, vectorized over the pending
        # envs; each env draws only from its own generator.
        cursor = np.empty((idx.size, 2))
        pending = np.arange(idx.size)
        while pending.size:
            cand = np.array([rngs[idx[k]].random(2) for k in pending]) * size
            cand = np.minimum(cand, size - CURSOR_EDGE_EPS)  # same range as step()
            if cfg.cursor_start_outside:
                ddx = cand[:, 0] - cx[pending]
                ddy = cand[:, 1] - cy[pending]
                ok = ddx * ddx + ddy * ddy > r[pending] * r[pending]
            else:
                ok = np.ones(pending.size, dtype=bool)
            cursor[pending[ok]] = cand[ok]
            pending = pending[~ok]

        dist = np.sqrt((cursor[:, 0] - cx) ** 2 + (cursor[:, 1] - cy) ** 2)
        self.cursor[idx] = cursor
        self.target[idx] = np.stack((cx, cy), axis=1)
        self.radius[idx] = r
        self.target_alive[idx] = True
        self.hunger[idx] = hunger
        self.restlessness[idx] = restless
        self.init_dist[idx] = np.maximum(0.0, dist - r)
        for name in ("t", "cum_move", "empty_clicks", "clicks", "ep_return",
                     "path_length", "bonus_total"):
            getattr(self, name)[idx] = 0
        k = int(cfg.distractors)
        if k:   # per env from its own dot generator: K positions, directions, speeds, grey levels
            for i in idx:
                u = self._dot_rngs[i].random((k, 5))
                ang = 2.0 * np.pi * u[:, 2]
                speed = cfg.dot_speed_min + (cfg.dot_speed_max - cfg.dot_speed_min) * u[:, 3]
                self.dot_pos[i] = u[:, :2] * (size - cfg.dot_size)
                self.dot_vel[i] = np.stack((np.cos(ang), np.sin(ang)), axis=1) * speed[:, None]
                self.dot_val[i] = cfg.dot_brightness_min + (cfg.dot_brightness_max - cfg.dot_brightness_min) * u[:, 4]
        self.ep_hits[idx] = 0
        if cfg.target_speed > 0:
            for i in idx:
                ang = 2.0 * np.pi * self._spawn_rngs[i].random()
                self.target_vel[i] = cfg.target_speed * np.array([np.cos(ang), np.sin(ang)])
        h = int(cfg.hazards)
        if h:   # per env from its own hazard generator: H radii, centers, directions
            for i in idx:
                u = self._hz_rngs[i].random((h, 4))
                r = cfg.hazard_radius_min + (cfg.hazard_radius_max - cfg.hazard_radius_min) * u[:, 0]
                self.hz_rad[i] = r
                self.hz_pos[i] = np.stack((r + (size - 2 * r) * u[:, 1], r + (size - 2 * r) * u[:, 2]), axis=1)
                ang = 2.0 * np.pi * u[:, 3]
                self.hz_vel[i] = cfg.hazard_speed * np.stack((np.cos(ang), np.sin(ang)), axis=1)

    def _respawn(self, idx: np.ndarray) -> None:
        """New circles for envs idx (their own spawn generators), resampled until the cursor is outside."""
        cfg = self.config
        size = float(cfg.size)
        for i in idx:
            g = self._spawn_rngs[i]
            for _ in range(1000):
                u = g.random(3)
                r = cfg.radius_min + (cfg.radius_max - cfg.radius_min) * u[0]
                if cfg.wrap:   # anywhere on the torus
                    cx, cy = size * u[1], size * u[2]
                    ddx, ddy = wrap_delta(self.cursor[i, 0], cx, size), wrap_delta(self.cursor[i, 1], cy, size)
                else:
                    cx, cy = r + (size - 2.0 * r) * u[1], r + (size - 2.0 * r) * u[2]
                    ddx, ddy = self.cursor[i, 0] - cx, self.cursor[i, 1] - cy
                if ddx * ddx + ddy * ddy > r * r:
                    break
            self.target[i] = (cx, cy)
            self.radius[i] = r
            if cfg.target_speed > 0:
                ang = 2.0 * np.pi * g.random()
                self.target_vel[i] = cfg.target_speed * np.array([np.cos(ang), np.sin(ang)])

    # ------------------------------------------------------------ observation
    def observe(self) -> Dict[str, np.ndarray]:
        """Observation of the current state: {'image': (N,S,S,3) uint8, 'drives': (N,2) float32}."""
        return {"image": self._render(), "drives": self._drive_obs()}

    def _render(self, idx: Optional[np.ndarray] = None) -> np.ndarray:
        if idx is None:
            return render_state(self.cursor, self.target, self.radius, self.target_alive, self.config,
                                (self.dot_pos, self.dot_val), (self.hz_pos, self.hz_rad))
        return render_state(self.cursor[idx], self.target[idx], self.radius[idx],
                            self.target_alive[idx], self.config, (self.dot_pos[idx], self.dot_val[idx]),
                            (self.hz_pos[idx], self.hz_rad[idx]))

    def _drive_obs(self) -> np.ndarray:
        if self.config.drives:
            return np.stack((self.hunger, self.restlessness), axis=1).astype(np.float32)
        return np.zeros((self.num_envs, 2), dtype=np.float32)

    # -------------------------------------------------------------- dynamics
    def step(self, actions: np.ndarray):
        """Advance every env by one action.

        Args:
            actions: float array (N, 3) of [dx, dy, click].  NaN -> 0, dx and
                dy clipped to [-1, 1], click = actions[:, 2] > 0.5.

        Returns:
            (obs, reward float32 (N,), done bool (N,), info).  See the module
            docstring for the order of operations; ``info`` keys:
              terminal_image (N,S,S,3) uint8   post-step frame before auto-reset
              terminal_drives (N,2) float32    post-step drives (zeros unless drives=True)
              success, timeout, click, empty_click, first   (N,) bool
              bonus (N,) float64               exploration-bonus part of reward
              ep_length, ep_return, ep_success, ep_clicks, ep_empty_clicks,
              ep_path_length, ep_init_dist, ep_bonus_total
                                               (N,) float64, NaN where not done
              gt_cursor, gt_target (N,2), gt_radius (N,) float64,
              gt_target_alive, gt_inside (N,) bool, gt_drives (N,2) float64
              gt_dots (N,K,2) float64          (distractors > 0) dot positions
                                               EVALUATION ONLY, never model input
        """
        cfg = self.config
        n = self.num_envs
        move, click = sanitize_actions(actions)
        if move.shape[0] != n:
            raise ValueError(f"expected actions of shape ({n}, 3), got {np.shape(actions)}")

        # 1. Move; displacement is the actual (clamped) motion.
        old = self.cursor
        if cfg.wrap:   # a torus: no walls
            new = np.mod(old + move * cfg.max_step_px, float(cfg.size))
            delta = move * cfg.max_step_px
        else:
            new = np.clip(old + move * cfg.max_step_px, 0.0, cfg.size - CURSOR_EDGE_EPS)
            delta = new - old
        disp = np.sqrt(delta[:, 0] * delta[:, 0] + delta[:, 1] * delta[:, 1])
        self.cursor = new

        # 2. Hit test at the new continuous position.
        if cfg.wrap:
            ddx = wrap_delta(new[:, 0], self.target[:, 0], float(cfg.size))
            ddy = wrap_delta(new[:, 1], self.target[:, 1], float(cfg.size))
        else:
            ddx = new[:, 0] - self.target[:, 0]
            ddy = new[:, 1] - self.target[:, 1]
        inside = (ddx * ddx + ddy * ddy <= self.radius * self.radius) & self.target_alive
        success = click & inside
        empty = click & ~success
        if cfg.hazards:   # inside any hazard, at the new position
            hdx = new[:, None, 0] - self.hz_pos[..., 0]
            hdy = new[:, None, 1] - self.hz_pos[..., 1]
            if cfg.wrap:
                hdx, hdy = wrap_delta(hdx, 0.0, float(cfg.size)), wrap_delta(hdy, 0.0, float(cfg.size))
            in_hazard = (hdx * hdx + hdy * hdy <= self.hz_rad * self.hz_rad).any(1)
        else:
            in_hazard = np.zeros(n, dtype=bool)

        # 3. Reward.  Budget factors use the counters from BEFORE this step.
        move_factor = np.maximum(0.0, 1.0 - self.cum_move / cfg.move_budget_px)
        reward = cfg.success_reward * success - cfg.step_cost
        if cfg.bonuses:
            click_factor = np.maximum(0.0, 1.0 - self.empty_clicks / cfg.click_budget)
            bonus = (cfg.move_bonus * (disp / cfg.max_step_px) * move_factor
                     + np.where(empty, cfg.click_bonus * click_factor, 0.0))
            reward = reward + bonus
        else:
            bonus = np.zeros(n)
        if cfg.drives:
            reward = reward - cfg.hunger_cost * self.hunger

        # 4. Counters and drives.
        self.cum_move += disp
        self.empty_clicks += empty
        self.clicks += click
        self.path_length += disp
        self.hunger = np.clip(self.hunger + cfg.h_rate - cfg.h_relief * success + cfg.hazard_cost * in_hazard, 0.0, 1.0)
        poison = click & in_hazard if cfg.hazard_click >= 0 else None
        if poison is not None:   # a click inside a hazard: hunger is set (it wins over relief from the circle)
            self.hunger = np.where(poison, cfg.hazard_click, self.hunger)
        self.ep_hits += success
        self.restlessness = np.clip(
            self.restlessness + cfg.b_rate - cfg.b_relief * (disp / cfg.max_step_px) * move_factor,
            0.0, 1.0)
        self.t += 1
        self.ep_return += reward
        self.bonus_total += bonus
        if cfg.distractors and cfg.wrap:   # dots drift around the torus
            self.dot_pos = np.mod(self.dot_pos + self.dot_vel, float(cfg.size))
        elif cfg.distractors:   # dots move and bounce off the walls
            hi = float(cfg.size - cfg.dot_size)
            p = self.dot_pos + self.dot_vel
            low, high = p < 0.0, p > hi
            p = np.where(low, -p, np.where(high, 2.0 * hi - p, p))
            self.dot_vel = np.where(low | high, -self.dot_vel, self.dot_vel)
            self.dot_pos = np.clip(p, 0.0, hi)
        if cfg.target_speed > 0 and cfg.wrap:
            self.target = np.mod(self.target + self.target_vel, float(cfg.size))
        elif cfg.target_speed > 0:   # the circle moves and bounces off the walls (its center stays r from them)
            lo_, hi_ = self.radius[:, None], cfg.size - self.radius[:, None]
            p = self.target + self.target_vel
            low, high = p < lo_, p > hi_
            p = np.where(low, 2.0 * lo_ - p, np.where(high, 2.0 * hi_ - p, p))
            self.target_vel = np.where(low | high, -self.target_vel, self.target_vel)
            self.target = np.clip(p, lo_, hi_)
        if cfg.hazards and cfg.wrap:
            self.hz_pos = np.mod(self.hz_pos + self.hz_vel, float(cfg.size))
        elif cfg.hazards:
            lo_, hi_ = self.hz_rad[..., None], cfg.size - self.hz_rad[..., None]
            p = self.hz_pos + self.hz_vel
            low, high = p < lo_, p > hi_
            p = np.where(low, 2.0 * lo_ - p, np.where(high, 2.0 * hi_ - p, p))
            self.hz_vel = np.where(low | high, -self.hz_vel, self.hz_vel)
            self.hz_pos = np.clip(p, lo_, hi_)

        # 5. A hit target disappears, or is replaced at once (respawn).
        if cfg.respawn:
            if success.any():
                self._respawn(np.flatnonzero(success))
        else:
            self.target_alive &= ~success

        # 6. Termination.
        done = (success & cfg.end_on_success) | (self.t >= cfg.max_steps)
        timeout = done & ~success

        # 7. Info from the post-step, pre-reset state.
        terminal_image = self._render()
        nan = np.nan
        info: Dict[str, np.ndarray] = {
            "terminal_image": terminal_image,
            "terminal_drives": self._drive_obs(),
            "success": success,
            "timeout": timeout,
            "click": click,
            "empty_click": empty,
            "first": np.zeros(n, dtype=bool),
            "bonus": bonus,
            "ep_length": np.where(done, self.t, nan),
            "ep_return": np.where(done, self.ep_return, nan),
            "ep_success": np.where(done, ((self.ep_hits > 0) if cfg.respawn else ~self.target_alive).astype(np.float64),
                                   nan),
            "ep_clicks": np.where(done, self.clicks, nan),
            "ep_empty_clicks": np.where(done, self.empty_clicks, nan),
            "ep_path_length": np.where(done, self.path_length, nan),
            "ep_init_dist": np.where(done, self.init_dist, nan),
            "ep_bonus_total": np.where(done, self.bonus_total, nan),
            # --- evaluation-only ground truth: NEVER a model input ---
            "gt_cursor": self.cursor.copy(),
            "gt_target": self.target.copy(),
            "gt_radius": self.radius.copy(),
            "gt_target_alive": self.target_alive.copy(),
            "gt_inside": inside,
            "gt_drives": np.stack((self.hunger, self.restlessness), axis=1),
        }
        if cfg.distractors:
            info["gt_dots"] = self.dot_pos.copy()
        if cfg.respawn:
            info["ep_hits"] = np.where(done, self.ep_hits, nan)
        if cfg.hazards:
            info["gt_hazards"] = self.hz_pos.copy()
            info["gt_in_hazard"] = in_hazard
            if poison is not None:
                info["gt_hazard_click"] = poison

        # 8. Auto-reset: done envs return the first frame of their next episode.
        image = terminal_image.copy()
        if cfg.auto_reset and done.any():
            idx = np.flatnonzero(done)
            self._reset_envs(idx)
            image[idx] = self._render(idx)
            info["first"] = done.copy()
        obs = {"image": image, "drives": self._drive_obs()}

        # 9.
        return obs, reward.astype(np.float32), done, info

    # ------------------------------------------------------------ save/restore
    def get_state(self) -> Dict[str, np.ndarray]:
        """Copy of the full simulator state as a dict of numpy arrays.

        Keys: STATE_SPEC names plus ``rng_state``, a (N, 6) uint64 encoding of
        each env's PCG64 state.  ``set_state(get_state())`` restores exactly.
        """
        state = {name: getattr(self, name).copy() for name in self.STATE_SPEC}
        state["rng_state"] = np.array([_encode_rng(g) for g in self._rngs], dtype=np.uint64)
        if self.config.distractors:
            for name in ("dot_pos", "dot_vel", "dot_val"):
                state[name] = getattr(self, name).copy()
            state["dot_rng_state"] = np.array([_encode_rng(g) for g in self._dot_rngs], dtype=np.uint64)
        for name, on in (("target_vel", self.config.target_speed > 0), ("hz_pos", self.config.hazards),
                         ("hz_vel", self.config.hazards), ("hz_rad", self.config.hazards),
                         ("ep_hits", self.config.respawn)):
            if on:
                state[name] = getattr(self, name).copy()
        if self._spawn_rngs:
            state["spawn_rng_state"] = np.array([_encode_rng(g) for g in self._spawn_rngs], dtype=np.uint64)
        if self._hz_rngs:
            state["hz_rng_state"] = np.array([_encode_rng(g) for g in self._hz_rngs], dtype=np.uint64)
        return state

    def set_state(self, state: Dict[str, Any]) -> None:
        """Overwrite state arrays from ``state`` (a full or partial get_state() dict).

        Only the keys present are changed, which lets tests and counterfactual
        analyses place the cursor and target exactly.  Arrays must have shape
        (N, ...) as in STATE_SPEC; unknown keys raise KeyError.  Call
        observe() afterwards for the matching observation.
        """
        n = self.num_envs
        for key, value in state.items():
            if key == "rng_state":
                rows = np.asarray(value, dtype=np.uint64)
                if rows.shape != (n, 6):
                    raise ValueError(f"rng_state must have shape ({n}, 6), got {rows.shape}")
                for gen, row in zip(self._rngs, rows):
                    gen.bit_generator.state = _decode_rng(row)
                continue
            if key in ("dot_pos", "dot_vel", "dot_val", "dot_rng_state") and self.config.distractors:
                if key == "dot_rng_state":
                    for gen, row in zip(self._dot_rngs, np.asarray(value, dtype=np.uint64)):
                        gen.bit_generator.state = _decode_rng(row)
                else:
                    arr = np.array(value, dtype=np.float64)
                    if arr.shape != getattr(self, key).shape:
                        raise ValueError(f"state[{key!r}] must have shape {getattr(self, key).shape}")
                    setattr(self, key, arr)
                continue
            if key in ("spawn_rng_state", "hz_rng_state"):
                for gen, row in zip(self._spawn_rngs if key == "spawn_rng_state" else self._hz_rngs,
                                    np.asarray(value, dtype=np.uint64)):
                    gen.bit_generator.state = _decode_rng(row)
                continue
            if key in ("target_vel", "hz_pos", "hz_vel", "hz_rad", "ep_hits"):
                arr = np.array(value, dtype=getattr(self, key).dtype)
                if arr.shape != getattr(self, key).shape:
                    raise ValueError(f"state[{key!r}] must have shape {getattr(self, key).shape}")
                setattr(self, key, arr)
                continue
            if key not in self.STATE_SPEC:
                raise KeyError(f"unknown state key {key!r}")
            shape, dtype = self.STATE_SPEC[key]
            arr = np.array(value, dtype=dtype)  # copy
            if arr.shape != (n,) + shape:
                raise ValueError(f"state[{key!r}] must have shape {(n,) + shape}, got {arr.shape}")
            if np.issubdtype(arr.dtype, np.floating) and not np.all(np.isfinite(arr)):
                raise ValueError(f"state[{key!r}] must be finite")
            setattr(self, key, arr)


# =============================================================================
# Benchmark CLI
# =============================================================================

def _bench(num_envs: int, steps: int, seed: int) -> None:
    cfg = EnvConfig()
    env = CircleWorld(num_envs, cfg, seed=seed)
    rng = np.random.default_rng(seed)
    actions = rng.uniform(-1.0, 1.0, size=(steps, num_envs, 3))  # click p = 0.25
    actions[:, :, 2] = rng.random((steps, num_envs))             # click p = 0.5
    env.reset()
    for a in actions[:50]:                                        # warm-up
        env.step(a)
    episodes = 0
    t0 = time.perf_counter()
    for a in actions:
        obs, reward, done, info = env.step(a)   # keep results alive, as a training loop does
        episodes += int(done.sum())
    dt = time.perf_counter() - t0

    t1 = time.perf_counter()
    reps = max(1, steps // 4)
    for _ in range(reps):
        render_state(env.cursor, env.target, env.radius, env.target_alive, cfg)
    dr = (time.perf_counter() - t1) / reps

    print(f"CircleWorld bench: N={num_envs}, {steps} steps of random actions "
          f"(dx,dy ~ U[-1,1], click p=0.5), size={cfg.size}")
    print(f"  step():   {1e3 * dt / steps:.3f} ms/step  ->  {steps / dt:,.0f} steps/s  "
          f"=  {steps * num_envs / dt:,.0f} env-steps/s   ({episodes} episodes finished)")
    print(f"  render:   {1e3 * dr:.3f} ms per batch of {num_envs} frames")


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="CircleWorld simulator utilities.")
    parser.add_argument("--bench", action="store_true", help="time step() with random actions")
    parser.add_argument("--num-envs", type=int, default=128)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    if args.bench:
        _bench(args.num_envs, args.steps, args.seed)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

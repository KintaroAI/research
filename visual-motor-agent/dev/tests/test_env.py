"""CircleWorld mechanics: shapes, determinism, movement, hits, resets, rendering."""

import numpy as np
import pytest

from world import CircleWorld, EnvConfig, render_state
from world.circle_world import CURSOR_EDGE_EPS, CURSOR_OUTLINE_OFFSETS, CURSOR_WHITE_OFFSETS

CFG = EnvConfig()
BG = list(CFG.bg_color)
RED = list(CFG.target_color)
WHITE = list(CFG.cursor_color)
BLACK = list(CFG.cursor_outline)
EDGE = CFG.size - CURSOR_EDGE_EPS

INFO_SPEC = {  # key -> (shape after N, dtype)
    "terminal_image": ((64, 64, 3), np.uint8),
    "terminal_drives": ((2,), np.float32),
    "success": ((), np.bool_), "timeout": ((), np.bool_), "click": ((), np.bool_),
    "empty_click": ((), np.bool_), "first": ((), np.bool_),
    "bonus": ((), np.float64),
    "ep_length": ((), np.float64), "ep_return": ((), np.float64),
    "ep_success": ((), np.float64), "ep_clicks": ((), np.float64),
    "ep_empty_clicks": ((), np.float64), "ep_path_length": ((), np.float64),
    "ep_init_dist": ((), np.float64), "ep_bonus_total": ((), np.float64),
    "gt_cursor": ((2,), np.float64), "gt_target": ((2,), np.float64),
    "gt_radius": ((), np.float64), "gt_target_alive": ((), np.bool_),
    "gt_inside": ((), np.bool_), "gt_drives": ((2,), np.float64),
}


def place(env, cursor, target=(48.0, 48.0), radius=10.0, alive=True, **extra):
    """Put every env of ``env`` into the same exact state."""
    n = env.num_envs
    state = {"cursor": np.tile(np.asarray(cursor, float), (n, 1)),
             "target": np.tile(np.asarray(target, float), (n, 1)),
             "radius": np.full(n, float(radius)), "target_alive": np.full(n, alive)}
    for key, value in extra.items():
        state[key] = np.full(n, value)
    env.set_state(state)


def red_count(images):
    return np.all(images == np.array(RED, np.uint8), axis=-1).sum(axis=(-2, -1))


def act(*a):
    return np.array([a], dtype=float)


def reference_render(cursor, target, radius, alive, cfg):
    """Slow, obviously-correct per-pixel renderer (the spec, literally)."""
    s = cfg.size
    out = np.empty((len(radius), s, s, 3), np.uint8)
    for e in range(len(radius)):
        img = out[e]
        img[:] = cfg.bg_color
        cx, cy, r = float(target[e][0]), float(target[e][1]), float(radius[e])
        if alive[e]:
            for i in range(s):
                for j in range(s):
                    dx, dy = (j + 0.5) - cx, (i + 0.5) - cy
                    if dx * dx + dy * dy <= r * r:
                        img[i, j] = cfg.target_color
        hx, hy = int(np.floor(cursor[e][0])), int(np.floor(cursor[e][1]))
        for offsets, color in ((CURSOR_OUTLINE_OFFSETS, cfg.cursor_outline),
                               (CURSOR_WHITE_OFFSETS, cfg.cursor_color)):
            for dx, dy in offsets:
                if 0 <= hx + dx < s and 0 <= hy + dy < s:
                    img[hy + dy, hx + dx] = color
    return out


# --------------------------------------------------------------------------- shapes

def test_shapes_and_dtypes():
    n = 5
    env = CircleWorld(n, EnvConfig(), seed=0)
    obs = env.reset()
    assert set(obs) == {"image", "drives"}
    assert obs["image"].shape == (n, 64, 64, 3) and obs["image"].dtype == np.uint8
    assert obs["drives"].shape == (n, 2) and obs["drives"].dtype == np.float32
    assert obs["image"].flags.c_contiguous

    obs, reward, done, info = env.step(np.zeros((n, 3)))
    assert obs["image"].shape == (n, 64, 64, 3) and obs["image"].dtype == np.uint8
    assert obs["drives"].shape == (n, 2) and obs["drives"].dtype == np.float32
    assert reward.shape == (n,) and reward.dtype == np.float32
    assert done.shape == (n,) and done.dtype == np.bool_
    assert set(info) == set(INFO_SPEC)
    for key, (shape, dtype) in INFO_SPEC.items():
        assert info[key].shape == (n,) + shape, key
        assert info[key].dtype == dtype, key


def test_bad_action_shape_raises():
    env = CircleWorld(3, seed=0)
    with pytest.raises(ValueError):
        env.step(np.zeros((2, 3)))
    with pytest.raises(ValueError):
        env.step(np.zeros((3, 2)))


# ---------------------------------------------------------------------- determinism

def _rollout(seed, steps=500, n=6, actions_seed=99):
    env = CircleWorld(n, EnvConfig(max_steps=40, bonuses=True, drives=True, hunger_cost=0.01), seed=seed)
    rng = np.random.default_rng(actions_seed)
    out = [env.reset()]
    for _ in range(steps):
        a = rng.uniform(-1, 1, (n, 3))
        a[:, :2] = np.clip(a[:, :2] * 2, -1, 1)
        out.append(env.step(a))
    return out


def test_same_seed_same_trajectories():
    a, b = _rollout(seed=5), _rollout(seed=5)
    np.testing.assert_array_equal(a[0]["image"], b[0]["image"])
    resets = 0
    for (oa, ra, da, ia), (ob, rb, db, ib) in zip(a[1:], b[1:]):
        np.testing.assert_array_equal(oa["image"], ob["image"])
        np.testing.assert_array_equal(oa["drives"], ob["drives"])
        np.testing.assert_array_equal(ra, rb)
        np.testing.assert_array_equal(da, db)
        assert set(ia) == set(ib)
        for key in ia:
            np.testing.assert_array_equal(ia[key], ib[key], err_msg=key)  # NaN-aware
        resets += int(da.sum())
    assert resets > 50  # the comparison covered many auto-resets


def test_different_seeds_differ():
    a, b = _rollout(seed=5, steps=50), _rollout(seed=6, steps=50)
    assert not np.array_equal(a[0]["image"], b[0]["image"])
    assert not all(np.array_equal(x[1], y[1]) for x, y in zip(a[1:], b[1:]))


def test_env_independent_of_other_envs():
    """Env i's episodes depend only on the seed and env i's own actions."""
    cfg = EnvConfig(max_steps=25)
    wa, wb = CircleWorld(4, cfg, seed=7), CircleWorld(4, cfg, seed=7)
    rng = np.random.default_rng(0)
    for _ in range(300):
        a, b = rng.uniform(-1, 1, (4, 3)), rng.uniform(-1, 1, (4, 3))
        b[2] = a[2]  # env 2 gets the same actions, the others differ
        oa, ra, da, ia = wa.step(a)
        ob, rb, db, ib = wb.step(b)
        np.testing.assert_array_equal(oa["image"][2], ob["image"][2])
        assert ra[2] == rb[2] and da[2] == db[2]
        np.testing.assert_array_equal(ia["gt_target"][2], ib["gt_target"][2])


def test_get_set_state_roundtrip_reproduces_future():
    env = CircleWorld(8, EnvConfig(max_steps=30, bonuses=True, drives=True), seed=1)
    rng = np.random.default_rng(3)
    for _ in range(45):
        env.step(rng.uniform(-1, 1, (8, 3)))
    saved = env.get_state()
    actions = rng.uniform(-1, 1, (80, 8, 3))
    first = [env.step(a) for a in actions]
    env.set_state(saved)
    second = [env.step(a) for a in actions]
    assert sum(int(f[2].sum()) for f in first) > 5  # includes RNG-driven resets
    for (o1, r1, d1, i1), (o2, r2, d2, i2) in zip(first, second):
        np.testing.assert_array_equal(o1["image"], o2["image"])
        np.testing.assert_array_equal(r1, r2)
        np.testing.assert_array_equal(i1["gt_target"], i2["gt_target"])
    # get_state returns copies
    s = env.get_state()
    s["cursor"][:] = -5
    assert np.all(env.cursor >= 0)


def test_set_state_validation():
    env = CircleWorld(2, seed=0)
    with pytest.raises(KeyError):
        env.set_state({"bogus": np.zeros(2)})
    with pytest.raises(ValueError):
        env.set_state({"cursor": np.zeros((3, 2))})
    with pytest.raises(ValueError):
        env.set_state({"radius": np.array([1.0, np.nan])})


# ------------------------------------------------------------------------- movement

def test_full_step_moves_exactly_max_step_px():
    env = CircleWorld(1, seed=0)
    place(env, (20.0, 30.0))
    _, _, _, info = env.step(act(1, 0, 0))
    assert info["gt_cursor"][0].tolist() == [24.0, 30.0]
    _, _, _, info = env.step(act(0.5, -1, 0))
    assert info["gt_cursor"][0].tolist() == [26.0, 26.0]
    _, _, _, info = env.step(act(-0.25, 0.75, 0))
    assert info["gt_cursor"][0].tolist() == [25.0, 29.0]
    assert env.path_length[0] == pytest.approx(4 + np.hypot(2, 4) + np.hypot(1, 3))


@pytest.mark.parametrize("start, move, expected", [
    ((1.0, 30.0), (-1, 0), (0.0, 30.0)),
    ((62.0, 30.0), (1, 0), (EDGE, 30.0)),
    ((30.0, 1.5), (0, -1), (30.0, 0.0)),
    ((30.0, 61.0), (0, 1), (30.0, EDGE)),
    ((0.5, 63.5), (-1, 1), (0.0, EDGE)),
])
def test_clamping_at_edges(start, move, expected):
    env = CircleWorld(1, seed=0)
    place(env, start, target=(32.0, 32.0), radius=5.0)
    _, _, _, info = env.step(act(move[0], move[1], 0))
    assert tuple(info["gt_cursor"][0]) == expected
    path = env.path_length[0]
    for _ in range(3):  # keep pushing: no displacement at all
        _, _, _, info = env.step(act(move[0], move[1], 0))
        assert tuple(info["gt_cursor"][0]) == expected
    assert env.path_length[0] == path


def test_nan_and_inf_actions():
    env = CircleWorld(1, seed=0)
    place(env, (30.0, 30.0), target=(10.0, 10.0), radius=5.0)
    _, _, _, info = env.step(act(np.nan, np.nan, np.nan))
    assert info["gt_cursor"][0].tolist() == [30.0, 30.0] and not info["click"][0]
    _, _, _, info = env.step(act(np.nan, 1.0, np.nan))
    assert info["gt_cursor"][0].tolist() == [30.0, 34.0]
    _, _, _, info = env.step(act(np.inf, -np.inf, np.inf))
    assert info["gt_cursor"][0].tolist() == [34.0, 30.0] and info["click"][0]
    _, _, _, info = env.step(act(7.0, -3.0, 0.5))  # out-of-range values clip; 0.5 is no click
    assert info["gt_cursor"][0].tolist() == [38.0, 26.0] and not info["click"][0]


# ------------------------------------------------------------------- hits and resets

def test_success_kills_target_then_resets():
    cfg = EnvConfig()
    env = CircleWorld(1, cfg, seed=3)
    place(env, (48.0, 48.0), target=(48.0, 48.0), radius=12.0)
    assert red_count(env.observe()["image"])[0] > 0
    obs, reward, done, info = env.step(act(0, 0, 1))
    assert info["success"][0] and info["gt_inside"][0] and done[0] and not info["timeout"][0]
    assert reward[0] >= np.float32(cfg.success_reward - cfg.step_cost)
    assert not info["gt_target_alive"][0]
    assert red_count(info["terminal_image"])[0] == 0      # circle gone in the terminal frame
    assert red_count(obs["image"])[0] > 0                 # new episode's first frame
    assert info["first"][0]
    assert info["ep_success"][0] == 1.0 and info["ep_length"][0] == 1
    assert info["ep_clicks"][0] == 1 and info["ep_empty_clicks"][0] == 0
    assert env.t[0] == 0 and env.target_alive[0]


def test_click_outside_is_not_success():
    env = CircleWorld(1, seed=0)
    place(env, (5.0, 5.0), target=(40.0, 40.0), radius=12.0)
    obs, reward, done, info = env.step(act(0, 0, 1))
    assert not info["success"][0] and info["empty_click"][0] and info["click"][0]
    assert not done[0] and not info["first"][0]
    assert reward[0] == np.float32(-CFG.step_cost)
    assert red_count(obs["image"])[0] > 0


def test_click_on_exact_boundary_counts():
    env = CircleWorld(1, seed=0)
    place(env, (40.0, 30.0), target=(30.0, 30.0), radius=10.0)   # d^2 == r^2
    _, _, done, info = env.step(act(0, 0, 1))
    assert info["success"][0] and done[0]


def test_dead_target_cannot_be_hit_and_is_not_drawn():
    env = CircleWorld(1, EnvConfig(end_on_success=False, max_steps=5), seed=0)
    place(env, (30.0, 30.0), target=(30.0, 30.0), radius=10.0)
    _, reward, done, info = env.step(act(0, 0, 1))
    assert info["success"][0] and not done[0]          # episode continues
    assert red_count(info["terminal_image"])[0] == 0
    _, reward, done, info = env.step(act(0, 0, 1))
    assert not info["success"][0] and info["empty_click"][0] and not info["gt_inside"][0]
    for _ in range(3):
        _, _, done, info = env.step(act(0, 0, 0))
    assert done[0] and info["timeout"][0] and info["ep_success"][0] == 1.0


def test_timeout_at_exactly_max_steps():
    cfg = EnvConfig()
    env = CircleWorld(3, cfg, seed=0)
    place(env, (2.0, 2.0), target=(48.0, 48.0), radius=10.0)
    for _ in range(cfg.max_steps - 1):
        _, _, done, info = env.step(np.zeros((3, 3)))
        assert not done.any() and not info["first"].any()
        assert np.isnan(info["ep_length"]).all()
    obs, reward, done, info = env.step(np.zeros((3, 3)))
    assert done.all() and info["timeout"].all() and not info["success"].any()
    assert np.all(info["ep_length"] == cfg.max_steps)
    assert np.all(info["ep_success"] == 0.0)
    assert info["ep_return"] == pytest.approx(np.full(3, -cfg.step_cost * cfg.max_steps))
    assert info["first"].all() and np.all(env.t == 0)


def test_auto_reset_first_flag_and_terminal_image_consistency():
    env = CircleWorld(16, EnvConfig(max_steps=15), seed=2)
    rng = np.random.default_rng(0)
    n_done = 0
    for _ in range(200):
        obs, reward, done, info = env.step(rng.uniform(-1, 1, (16, 3)))
        np.testing.assert_array_equal(info["first"], done)
        keep = ~done
        np.testing.assert_array_equal(info["terminal_image"][keep], obs["image"][keep])
        np.testing.assert_array_equal(info["terminal_drives"][keep], obs["drives"][keep])
        assert np.isnan(info["ep_return"][keep]).all() and np.isfinite(info["ep_return"][done]).all()
        np.testing.assert_array_equal(env.t[done], 0)
        n_done += int(done.sum())
    assert n_done > 100


def test_no_auto_reset_leaves_terminal_state():
    env = CircleWorld(1, EnvConfig(auto_reset=False), seed=0)
    place(env, (30.0, 30.0), target=(30.0, 30.0), radius=10.0)
    obs, _, done, info = env.step(act(0, 0, 1))
    assert done[0] and not info["first"][0]
    np.testing.assert_array_equal(obs["image"], info["terminal_image"])
    assert red_count(obs["image"])[0] == 0 and env.t[0] == 1
    obs = env.reset(env_ids=[0])
    assert red_count(obs["image"])[0] > 0 and env.t[0] == 0


def test_partial_reset_only_touches_selected_envs():
    env = CircleWorld(4, seed=0)
    before = env.get_state()
    env.reset(env_ids=[1, 3])
    after = env.get_state()
    for i in (0, 2):
        np.testing.assert_array_equal(before["target"][i], after["target"][i])
    for i in (1, 3):
        assert not np.array_equal(before["target"][i], after["target"][i])
    env.reset(env_ids=np.array([False, False, True, False]))   # bool mask, e.g. `done`
    final = env.get_state()
    assert not np.array_equal(after["target"][2], final["target"][2])
    for i in (0, 1, 3):
        np.testing.assert_array_equal(after["target"][i], final["target"][i])


@pytest.mark.parametrize("rmin, rmax", [(12.0, 16.0), (4.0, 8.0), (32.0, 32.0)])
def test_reset_geometry_over_10k_resets(rmin, rmax):
    cfg = EnvConfig(radius_min=rmin, radius_max=rmax)
    env = CircleWorld(10_000, cfg, seed=11)
    r = env.radius
    assert np.all((r >= rmin) & (r <= rmax))
    # cursor strictly outside the circle, inside the viewport
    d2 = ((env.cursor - env.target) ** 2).sum(axis=1)
    assert np.all(d2 > r * r)
    assert np.all((env.cursor >= 0) & (env.cursor <= cfg.size - CURSOR_EDGE_EPS))
    # circle fully inside the viewport
    assert np.all(env.target - r[:, None] >= 0) and np.all(env.target + r[:, None] <= cfg.size)
    assert np.allclose(env.init_dist, np.sqrt(d2) - r)
    assert np.all(env.target_alive) and np.all(env.t == 0)


def test_set_radius_range_affects_future_resets():
    env = CircleWorld(500, seed=0)
    assert env.radius.min() >= 12
    env.set_radius_range(4, 8)
    assert env.radius.min() >= 12            # current episodes unchanged
    env.reset()
    assert env.radius.min() >= 4 and env.radius.max() <= 8
    with pytest.raises(ValueError):
        env.set_radius_range(10, 40)         # larger than size / 2
    with pytest.raises(ValueError):
        env.set_radius_range(8, 4)


def test_cursor_start_inside_allowed_when_disabled():
    env = CircleWorld(5000, EnvConfig(cursor_start_outside=False), seed=0)
    d2 = ((env.cursor - env.target) ** 2).sum(axis=1)
    assert np.any(d2 <= env.radius ** 2)


# ------------------------------------------------------------------------- rendering

def test_exact_pixels_hand_built_state():
    img = render_state([[20.3, 30.7]], [[45.0, 45.0]], [10.0], [True], CFG)[0]
    hx, hy = 20, 30
    assert img[hy, hx].tolist() == WHITE                         # hotspot
    for k in (1, 2, 3):                                           # the 7-px "+"
        for x, y in ((hx + k, hy), (hx - k, hy), (hx, hy + k), (hx, hy - k)):
            assert img[y, x].tolist() == WHITE
    for x, y in ((hx + 4, hy), (hx - 4, hy), (hx, hy + 4), (hx, hy - 4),   # arm ends
                 (hx + 1, hy + 1), (hx - 1, hy - 1), (hx + 3, hy - 1), (hx - 1, hy + 3)):
        assert img[y, x].tolist() == BLACK
    assert img[hy + 2, hx + 2].tolist() == BG                     # between the arms
    assert img[45, 45].tolist() == RED and img[44, 44].tolist() == RED   # circle center
    assert img[63, 63].tolist() == BG and img[0, 63].tolist() == BG      # far corners
    colors = img.reshape(-1, 3).tolist()
    assert colors.count(WHITE) == 13 and colors.count(BLACK) == 32


def test_cursor_is_drawn_over_the_circle():
    img = render_state([[44.5, 45.2]], [[45.0, 45.0]], [12.0], [True], CFG)[0]
    assert img[45, 44].tolist() == WHITE and img[45, 48].tolist() == BLACK
    assert img[45, 49].tolist() == RED
    no_cursor_pixels = red_count(img[None])[0]
    assert no_cursor_pixels == red_count(render_state([[1.0, 1.0]], [[45.0, 45.0]], [12.0],
                                                      [True], CFG))[0] - 45


def test_sprite_clipped_at_viewport_corner():
    img = render_state([[0.0, 0.0]], [[40.0, 40.0]], [5.0], [True], CFG)[0]
    assert img[0, 0].tolist() == WHITE and img[0, 3].tolist() == WHITE
    assert img[0, 4].tolist() == BLACK and img[1, 1].tolist() == BLACK


def test_dead_target_not_drawn():
    img = render_state([[1.0, 1.0]], [[40.0, 40.0]], [12.0], [False], CFG)
    assert red_count(img)[0] == 0


def test_render_matches_reference_on_random_and_boundary_states():
    rng = np.random.default_rng(0)
    for size in (5, 17, 64):
        cfg = EnvConfig(size=size, radius_min=1.0, radius_max=size / 2)
        n = 40
        cursor = rng.uniform(-5, size + 5, (n, 2))
        target = rng.uniform(-size / 2, 1.5 * size, (n, 2))
        radius = rng.uniform(0, size, n)
        # boundary cases: half-pixel centers with radii hitting pixel centers exactly
        target[:15] = np.round(target[:15] * 2) / 2
        radius[:15] = rng.choice([0.0, 0.5, 1.0, np.sqrt(2), 2.5, 5.0, 13.0], 15)
        alive = rng.random(n) < 0.9
        fast = render_state(cursor, target, radius, alive, cfg)
        np.testing.assert_array_equal(fast, reference_render(cursor, target, radius, alive, cfg))


def test_render_is_vectorized_consistently():
    env = CircleWorld(32, seed=4)
    batch = env.observe()["image"]
    for i in (0, 7, 31):
        single = render_state(env.cursor[i:i + 1], env.target[i:i + 1], env.radius[i:i + 1],
                              env.target_alive[i:i + 1], env.config)
        np.testing.assert_array_equal(batch[i], single[0])

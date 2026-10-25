"""The richer world of vm-00011: respawning, moving circles and hazards (all off by default)."""
import dataclasses

import numpy as np

from world.circle_world import CircleWorld, EnvConfig

RICH = EnvConfig(drives=True, respawn=True, end_on_success=False, max_steps=10 ** 6, target_speed=0.8, hazards=2)


def random_actions(rng, n, p_click=0.2):
    return np.concatenate([rng.uniform(-1, 1, (n, 2)), rng.random((n, 1)) < p_click], 1)


def test_episodes_start_the_same_with_or_without_the_new_features():
    a, b = CircleWorld(16, EnvConfig(drives=True), seed=3), CircleWorld(16, RICH, seed=3)
    for k in ("cursor", "target", "radius", "hunger", "restlessness"):
        np.testing.assert_array_equal(getattr(a, k), getattr(b, k))
    assert set(a.get_state()) < set(b.get_state())
    for k in ("target_vel", "hz_pos", "spawn_rng_state", "hz_rng_state"):
        assert k not in a.get_state()


def test_respawn_replaces_a_hit_circle_away_from_the_cursor():
    env = CircleWorld(64, RICH, seed=1)
    env.set_state(dict(cursor=env.target.copy(), hz_pos=np.full_like(env.hz_pos, -100.0)))   # hazards out of reach
    old = env.target.copy()
    obs, rew, done, info = env.step(np.tile([0.0, 0.0, 1.0], (64, 1)))
    assert info["success"].all() and not done.any() and env.target_alive.all()
    d = np.linalg.norm(env.cursor - env.target, axis=1)
    assert (d > env.radius).all()                              # the new circle never starts on the cursor
    assert (np.abs(env.target - old).sum(1) > 0).all()
    assert (env.ep_hits == 1).all()
    red = (obs["image"] == np.array(RICH.target_color, np.uint8)).all(-1).sum((1, 2))
    assert (red > 0).all()                                     # and it is drawn


def test_moving_circle_keeps_its_speed_and_stays_inside():
    cfg = dataclasses.replace(RICH, hazards=0)
    env = CircleWorld(32, cfg, seed=2)
    rng = np.random.default_rng(0)
    for _ in range(500):
        prev = env.target.copy()
        env.step(random_actions(rng, 32, p_click=0.0))
        r = env.radius[:, None]
        assert ((env.target >= r - 1e-9) & (env.target <= cfg.size - r + 1e-9)).all()
        step = np.linalg.norm(env.target - prev, axis=1)
        assert (step <= cfg.target_speed + 1e-9).all()
    np.testing.assert_allclose(np.linalg.norm(env.target_vel, axis=1), cfg.target_speed)


def test_hazards_raise_hunger_and_are_drawn_under_the_circle():
    cfg = dataclasses.replace(RICH, target_speed=0.0, hazard_speed=0.0, h_rate=0.0, hazard_cost=0.05)
    env = CircleWorld(8, cfg, seed=4)
    env.set_state(dict(hunger=np.full(8, 0.3), target=np.full((8, 2), 50.0), radius=np.full(8, 6.0),
                       hz_pos=np.tile([[16.0, 16.0], [40.0, 16.0]], (8, 1, 1)), hz_rad=np.full((8, 2), 6.0),
                       cursor=np.tile([16.2, 16.2], (8, 1))))
    obs, _, _, info = env.step(np.zeros((8, 3)))
    assert info["gt_in_hazard"].all()
    np.testing.assert_allclose(env.hunger, 0.35)
    img = obs["image"]
    assert (img[:, 16, 40] == np.array(cfg.hazard_color, np.uint8)).all()      # hazard pixel (row y=16, col x=40)
    assert (img[:, 50, 50] == np.array(cfg.target_color, np.uint8)).all()
    env.set_state(dict(cursor=np.tile([30.0, 40.0], (8, 1))))
    _, _, _, info = env.step(np.zeros((8, 3)))
    assert not info["gt_in_hazard"].any()
    np.testing.assert_allclose(env.hunger, 0.35)


def test_poison_click_sets_hunger_and_food_still_relieves():
    """env.hazard_click (vm-00017): a click inside a hazard sets hunger; a click in the circle relieves it; the poison
    wins inside both; contact alone costs hazard_cost; the hazard stays."""
    cfg = dataclasses.replace(RICH, target_speed=0.0, hazard_speed=0.0, h_rate=0.001, hazard_cost=0.0,
                              hazard_click=1.0)
    env = CircleWorld(4, cfg, seed=5)
    base = dict(hunger=np.full(4, 0.3), target=np.full((4, 2), 50.0), radius=np.full(4, 6.0),
                hz_pos=np.tile([[16.0, 16.0], [40.0, 16.0]], (4, 1, 1)), hz_rad=np.full((4, 2), 6.0))
    # cursors: in the hazard, in the circle, on the background, in the hazard without a click
    cur = np.array([[16.2, 16.2], [50.0, 50.0], [30.0, 40.0], [16.2, 16.2]])
    env.set_state(dict(base, cursor=cur))
    _, _, _, info = env.step(np.array([[0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 0]], float))
    np.testing.assert_allclose(env.hunger, [1.0, 0.0, 0.301, 0.301])
    np.testing.assert_array_equal(info["gt_hazard_click"], [True, False, False, False])
    np.testing.assert_array_equal(info["success"], [False, True, False, False])
    np.testing.assert_allclose(env.hz_pos, base["hz_pos"])                     # the hazards stay
    env.set_state(dict(base, cursor=np.tile([50.0, 50.0], (4, 1)), hz_pos=np.tile([[50.0, 50.0], [40.0, 16.0]],
                                                                                    (4, 1, 1))))
    _, _, _, info = env.step(np.tile([0.0, 0.0, 1.0], (4, 1)))
    assert info["success"].all() and info["gt_hazard_click"].all()
    np.testing.assert_allclose(env.hunger, 1.0)                                # inside both: the poison wins


def test_poison_click_off_by_default():
    cfg = dataclasses.replace(RICH, hazard_cost=0.0)
    a = CircleWorld(32, cfg, seed=6)
    b = CircleWorld(32, dataclasses.replace(cfg, hazard_click=-1.0), seed=6)
    rng = np.random.default_rng(1)
    for _ in range(200):
        act = random_actions(rng, 32, p_click=0.5)
        ra, rb = a.step(act), b.step(act)
        np.testing.assert_array_equal(ra[0]["drives"], rb[0]["drives"])
        assert "gt_hazard_click" not in ra[3]


def test_state_round_trip_with_every_feature():
    env = CircleWorld(8, RICH, seed=5)
    rng = np.random.default_rng(1)
    for _ in range(30):
        env.step(random_actions(rng, 8))
    st = env.get_state()
    acts = [random_actions(rng, 8, p_click=0.5) for _ in range(40)]
    first = [env.step(a) for a in acts]
    env.set_state(st)
    second = [env.step(a) for a in acts]
    for (o1, r1, d1, i1), (o2, r2, d2, i2) in zip(first, second):
        np.testing.assert_array_equal(o1["image"], o2["image"])
        np.testing.assert_array_equal(r1, r2)
        np.testing.assert_array_equal(i1["success"], i2["success"])


def test_ground_truth_keys_only_when_enabled():
    rng = np.random.default_rng(2)
    _, _, _, info = CircleWorld(4, EnvConfig(), seed=0).step(random_actions(rng, 4))
    assert "gt_hazards" not in info and "ep_hits" not in info
    _, _, _, info = CircleWorld(4, RICH, seed=0).step(random_actions(rng, 4))
    assert "gt_hazards" in info and "ep_hits" in info


TORUS = dataclasses.replace(RICH, wrap=True)


def test_torus_wraps_the_cursor_and_hits_across_the_edge():
    cfg = dataclasses.replace(TORUS, hazards=0, target_speed=0.0)
    env = CircleWorld(4, cfg, seed=0)
    env.set_state(dict(cursor=np.tile([63.0, 10.0], (4, 1)), target=np.tile([1.0, 10.0], (4, 1)), radius=np.full(4, 3.0)))
    _, _, _, info = env.step(np.tile([0.5, 0.0, 1.0], (4, 1)))       # +2 px: wraps to x = 1, inside the circle
    np.testing.assert_allclose(env.cursor[:, 0], 1.0)
    assert info["success"].all()
    env.set_state(dict(cursor=np.tile([62.5, 10.0], (4, 1)), target=np.tile([1.0, 10.0], (4, 1)), radius=np.full(4, 3.0)))
    _, _, _, info = env.step(np.tile([0.0, 0.0, 1.0], (4, 1)))       # 2.5 px away across the edge
    assert info["success"].all()


def test_torus_draws_a_circle_across_the_edge():
    cfg = dataclasses.replace(TORUS, hazards=0)
    env = CircleWorld(1, cfg, seed=0)
    env.set_state(dict(target=np.array([[0.5, 30.0]]), radius=np.array([4.0]), cursor=np.array([[40.0, 50.0]])))
    img = env.observe()["image"][0]
    red = (img == np.array(cfg.target_color, np.uint8)).all(-1)
    assert red[30, 0] and red[30, 63] and red[30, 3] and red[30, 61] and not red[30, 60] and not red[30, 10]


def test_egocentric_view_keeps_the_cursor_centered_and_moves_the_scene():
    cfg = dataclasses.replace(TORUS, hazards=0, egocentric=True, target_speed=0.0)
    env = CircleWorld(1, cfg, seed=1)
    env.set_state(dict(cursor=np.array([[10.0, 20.0]]), target=np.array([[30.0, 20.0]]), radius=np.array([3.0])))
    img0 = env.observe()["image"][0]
    obs, _, _, _ = env.step(np.array([[1.0, 0.0, 0.0]]))             # 4 px to the right
    img1 = obs["image"][0]
    white = np.array(cfg.cursor_color, np.uint8)
    assert (img0[32, 32] == white).all() and (img1[32, 32] == white).all()
    red0 = np.argwhere((img0 == np.array(cfg.target_color, np.uint8)).all(-1))
    red1 = np.argwhere((img1 == np.array(cfg.target_color, np.uint8)).all(-1))
    np.testing.assert_array_equal(red1.mean(0), red0.mean(0) - [0, 4])   # the circle moved 4 px left in the view


def test_torus_state_round_trip():
    env = CircleWorld(6, dataclasses.replace(TORUS, egocentric=True, distractors=3), seed=2)
    rng = np.random.default_rng(3)
    for _ in range(20):
        env.step(random_actions(rng, 6))
    st = env.get_state()
    acts = [random_actions(rng, 6, p_click=0.5) for _ in range(30)]
    first = [env.step(a)[0]["image"] for a in acts]
    env.set_state(st)
    second = [env.step(a)[0]["image"] for a in acts]
    for a, b in zip(first, second):
        np.testing.assert_array_equal(a, b)

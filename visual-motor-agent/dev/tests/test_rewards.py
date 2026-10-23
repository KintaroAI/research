"""Reward accounting and the anti-exploit rules of plan section 6.

Unless stated otherwise the world runs with bonuses=True, drives=True and
hunger_cost=0.01, the configuration in which the most reward terms are active.
"""

import dataclasses

import numpy as np
import pytest

from world import CircleWorld, EnvConfig
from world.circle_world import CURSOR_EDGE_EPS

CFG = EnvConfig(bonuses=True, drives=True, hunger_cost=0.01)
EDGE = CFG.size - CURSOR_EDGE_EPS


def make(n=1, **overrides):
    cfg = dataclasses.replace(CFG, **overrides)
    return CircleWorld(n, cfg, seed=0), cfg


def place(env, cursor, target=(48.0, 48.0), radius=8.0, alive=True, **extra):
    n = env.num_envs
    state = {"cursor": np.tile(np.asarray(cursor, float), (n, 1)),
             "target": np.tile(np.asarray(target, float), (n, 1)),
             "radius": np.full(n, float(radius)), "target_alive": np.full(n, alive)}
    for key, value in extra.items():
        state[key] = np.full(n, value)
    env.set_state(state)


def act(*a):
    return np.array([a], dtype=float)


def zigzag(k, diagonal=False, click=0.0):
    """Full-magnitude back-and-forth step k (never touches an edge from the center)."""
    s = 1.0 if k % 2 == 0 else -1.0
    return [s, s if diagonal else 0.0, click]


# ------------------------------------------------------------------- inactivity

def test_standing_still_earns_nothing():
    env, cfg = make(n=16)
    env.reset()
    for _ in range(2 * cfg.max_steps):          # two full episodes, across a reset
        _, reward, _, info = env.step(np.zeros((16, 3)))
        assert np.all(reward <= 0)
        assert np.all(reward < 0)                # step cost is always paid
        assert np.all(info["bonus"] == 0)


# --------------------------------------------------------------- movement bonus

@pytest.mark.parametrize("steps, max_steps", [(200, 200), (2000, 10_000)])
def test_oscillation_bonus_is_bounded(steps, max_steps):
    env, cfg = make(max_steps=max_steps, auto_reset=False)
    place(env, (32.0, 32.0), alive=False)
    total = 0.0
    for k in range(steps):
        _, _, _, info = env.step(act(*zigzag(k)))
        total += info["bonus"][0]
    integral = cfg.move_bonus * cfg.move_budget_px / cfg.max_step_px / 2
    assert total <= integral + cfg.move_bonus          # analytic bound + one-step slack
    assert total >= integral                           # ...and the bonus is really paid
    assert env.cum_move[0] == steps * cfg.max_step_px  # every step moved the full 4 px


@pytest.mark.parametrize("start, move", [
    ((0.0, 30.0), (-1, 0)), ((EDGE, 30.0), (1, 0)),
    ((30.0, 0.0), (0, -1)), ((30.0, EDGE), (0, 1)),
    ((0.0, 0.0), (-1, -1)), ((EDGE, EDGE), (1, 1)),
])
def test_pushing_against_an_edge_earns_nothing(start, move):
    env, _ = make()
    place(env, start, target=(32.0, 32.0), radius=6.0)
    for _ in range(100):
        _, reward, _, info = env.step(act(move[0], move[1], 0))
        assert info["bonus"][0] == 0.0 and reward[0] < 0
    assert env.cum_move[0] == 0.0


# ------------------------------------------------------------------ click bonus

def test_repeated_empty_clicks_are_bounded():
    env, cfg = make()
    place(env, (5.0, 5.0), target=(48.0, 48.0))
    total = 0.0
    for _ in range(200):
        _, _, _, info = env.step(act(0, 0, 1))
        assert info["empty_click"][0]
        total += info["bonus"][0]
    bound = cfg.click_bonus * (cfg.click_budget + 1) / 2
    assert total <= bound + 1e-12
    assert total == pytest.approx(bound)   # the first click_budget clicks pay, linearly decaying


# --------------------------------------------------------------------- budgets

def test_waiting_does_not_restore_budgets():
    env, cfg = make()
    place(env, (32.0, 32.0), target=(10.0, 54.0), radius=6.0)
    for k in range(25):                       # 100 px of motion: half the movement budget
        env.step(act(*zigzag(k)))
    for _ in range(5):                        # half the click budget
        env.step(act(0, 0, 1))
    snapshot = env.get_state()

    def bonus_of(action, wait):
        env.set_state(snapshot)
        for _ in range(wait):
            env.step(act(0, 0, 0))
        return env.step(act(*action))[3]["bonus"][0]

    move_now, move_later = bonus_of((1, 0, 0), 0), bonus_of((1, 0, 0), 100)
    click_now, click_later = bonus_of((0, 0, 1), 0), bonus_of((0, 0, 1), 100)
    assert move_now == pytest.approx(cfg.move_bonus * 0.5)
    assert click_now == pytest.approx(cfg.click_bonus * 0.5)
    assert move_later == move_now and click_later == click_now


def test_bonus_budgets_reset_only_with_a_new_episode():
    env, cfg = make(max_steps=60)
    place(env, (32.0, 32.0), target=(10.0, 54.0), radius=6.0)
    for k in range(cfg.max_steps):            # exhaust the movement budget, then time out
        _, _, done, info = env.step(act(*zigzag(k)))
    assert done[0] and info["first"][0]
    assert env.cum_move[0] == 0 and env.empty_clicks[0] == 0


# ------------------------------------------------------------- reward balance

def _farm_bonus(n_envs, strategy, cfg):
    """Run one full episode per env with a bonus-farming strategy (target dead)."""
    env = CircleWorld(n_envs, cfg, seed=1)
    place(env, (32.0, 32.0), alive=False, hunger=0.0)
    rng = np.random.default_rng(0)
    for k in range(cfg.max_steps):
        if strategy == "random":
            a = np.ones((n_envs, 3))
            a[:, :2] = rng.choice([-1.0, 1.0], (n_envs, 2))
        else:
            a = np.tile(zigzag(k, diagonal=(strategy == "diagonal"), click=1.0), (n_envs, 1))
        _, _, done, info = env.step(a)
    assert done.all() and info["timeout"].all()
    return info["ep_bonus_total"], info["ep_return"]


@pytest.mark.parametrize("strategy", ["axis", "diagonal", "random"])
def test_worst_case_bonus_per_episode_is_below_one(strategy):
    bonus, _ = _farm_bonus(32, strategy, CFG)
    assert np.all(bonus < 1.0)
    assert np.all(bonus <= CFG.max_episode_bonus() + 1e-12)
    assert CFG.max_episode_bonus() < 1.0
    assert np.all(bonus > 0.3)          # the strategies do collect nearly all of it


@pytest.mark.parametrize("drives", [True, False])
def test_timeout_never_beats_an_immediate_success(drives):
    cfg = dataclasses.replace(CFG, drives=drives)
    best_timeout = max(_farm_bonus(32, s, cfg)[1].max() for s in ("axis", "diagonal", "random"))

    # random-policy timeouts, too
    env = CircleWorld(64, cfg, seed=2)
    rng = np.random.default_rng(1)
    for _ in range(3 * cfg.max_steps):
        _, _, done, info = env.step(rng.uniform(-1, 1, (64, 3)))
        timeouts = done & info["timeout"]
        if timeouts.any():
            best_timeout = max(best_timeout, info["ep_return"][timeouts].max())

    env = CircleWorld(1, cfg, seed=0)
    place(env, (48.0, 48.0), target=(48.0, 48.0), hunger=1.0)   # worst-case hunger cost
    _, _, done, info = env.step(act(0, 0, 1))
    assert done[0] and info["success"][0]
    assert best_timeout < info["ep_return"][0]
    # analytic version: best possible timeout < worst possible one-step success
    worst_success = cfg.success_reward - cfg.step_cost - (cfg.hunger_cost if drives else 0.0)
    assert cfg.max_episode_bonus() - cfg.max_steps * cfg.step_cost < worst_success


# ------------------------------------------------------------ feature toggles

def test_bonuses_off_pays_no_bonus():
    cfg = EnvConfig(bonuses=False, drives=True, hunger_cost=0.01)
    env = CircleWorld(32, cfg, seed=0)
    rng = np.random.default_rng(0)
    for _ in range(400):
        hunger_before = env.hunger.copy()
        _, reward, done, info = env.step(rng.uniform(-1, 1, (32, 3)))
        assert np.all(info["bonus"] == 0)
        assert np.all(info["ep_bonus_total"][done] == 0)
        expected = cfg.success_reward * info["success"] - cfg.step_cost - cfg.hunger_cost * hunger_before
        np.testing.assert_array_equal(reward, expected.astype(np.float32))


def test_drives_off_hides_drives_and_has_no_hunger_cost():
    cfg = EnvConfig(bonuses=False, drives=False, hunger_cost=0.5)
    env = CircleWorld(32, cfg, seed=0)
    rng = np.random.default_rng(0)
    obs = env.reset()
    assert np.all(obs["drives"] == 0)
    for _ in range(400):
        obs, reward, done, info = env.step(rng.uniform(-1, 1, (32, 3)))
        assert np.all(obs["drives"] == 0) and np.all(info["terminal_drives"] == 0)
        expected = np.where(info["success"], cfg.success_reward - cfg.step_cost, -cfg.step_cost)
        np.testing.assert_array_equal(reward, expected.astype(np.float32))
    assert np.any(info["gt_drives"] > 0)   # drives are still simulated internally


# ---------------------------------------------------------------------- drives

def test_hunger_rises_resets_on_success_and_is_clamped():
    env, cfg = make(auto_reset=False)
    place(env, (5.0, 5.0), hunger=0.2)
    for k in range(1, 21):
        obs, _, _, info = env.step(act(1, 1, 0) if k % 2 else act(-1, -1, 0))  # motion: no relief
        assert obs["drives"][0, 0] == pytest.approx(0.2 + k * cfg.h_rate, abs=1e-6)
    place(env, (5.0, 5.0), hunger=0.999, t=0)
    for _ in range(3):
        _, _, _, info = env.step(act(0, 0, 1))                                   # empty clicks: no relief
        assert info["gt_drives"][0, 0] == 1.0
    place(env, (48.0, 48.0), target=(48.0, 48.0), hunger=0.8, t=0)
    _, _, done, info = env.step(act(0, 0, 1))
    assert info["success"][0] and info["terminal_drives"][0, 0] == 0.0
    place(env, (48.0, 48.0), target=(48.0, 48.0), hunger=0.0, t=0, target_alive=True)
    _, _, _, info = env.step(act(0, 0, 1))
    assert info["gt_drives"][0, 0] == 0.0                                        # clamped at 0


def test_restlessness_rises_when_still_and_falls_when_moving():
    env, cfg = make(auto_reset=False)
    place(env, (20.0, 20.0), target=(54.0, 54.0), radius=6.0, restlessness=0.5)
    _, _, _, info = env.step(act(0, 0, 0))
    assert info["gt_drives"][0, 1] == pytest.approx(0.5 + cfg.b_rate)
    _, _, _, info = env.step(act(1, 0, 0))     # full step, untouched budget
    assert info["gt_drives"][0, 1] == pytest.approx(0.5 + 2 * cfg.b_rate - cfg.b_relief)
    _, _, _, info = env.step(act(0.5, 0, 0))   # half step, budget factor 1 - 4/200
    expected = 0.5 + 3 * cfg.b_rate - cfg.b_relief - cfg.b_relief * 0.5 * (1 - 4 / cfg.move_budget_px)
    assert info["gt_drives"][0, 1] == pytest.approx(expected)
    # once the movement budget is spent, motion no longer relieves restlessness
    place(env, (20.0, 20.0), restlessness=0.5, cum_move=cfg.move_budget_px, t=0)
    _, _, _, info = env.step(act(1, 0, 0))
    assert info["gt_drives"][0, 1] == pytest.approx(0.5 + cfg.b_rate)
    # clamped to [0, 1]
    place(env, (20.0, 20.0), restlessness=0.05, cum_move=0.0, t=0)
    _, _, _, info = env.step(act(1, 1, 0))
    assert info["gt_drives"][0, 1] == 0.0
    place(env, (20.0, 20.0), restlessness=0.99, t=0)
    _, _, _, info = env.step(act(0, 0, 0))
    assert info["gt_drives"][0, 1] == 1.0


def test_hunger_cost_uses_pre_step_hunger():
    env, cfg = make(auto_reset=False)
    place(env, (5.0, 5.0), hunger=0.6)
    _, reward, _, _ = env.step(act(0, 0, 0))
    assert reward[0] == np.float32(-cfg.step_cost - cfg.hunger_cost * 0.6)

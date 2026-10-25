"""lifetime.py (task 00020): the signed hindsight command, the body schedule, the shared long-term memory, and a short
lifetime with several bodies. (One body against online.py, bit for bit, is checked by running both: see the task.)"""
import json
import os
import subprocess
import sys

import numpy as np
import pytest

DEV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, DEV)

import online  # noqa: E402
from lifetime import bodies_at, eps_at  # noqa: E402


def test_delta_labels_by_hand():
    drv0 = np.array([0.50, 0.51, 0.52, 0.10, 0.11])
    ndrv0 = np.array([0.51, 0.52, 0.10, 0.11, 0.12])        # a hit at step 2 (hunger 0.52 -> 0.10)
    done = np.zeros(5, np.uint8)
    lab = online.delta_labels(drv0, ndrv0, done, 2)          # hunger now minus the hunger after the next 2 steps
    np.testing.assert_allclose(lab, [0.50 - 0.52, 0.51 - 0.10, 0.52 - 0.11, 0.10 - 0.12, 0.11 - 0.12], atol=1e-6)
    done[1] = 1                                              # step 1 ends its episode: step 0's window stops there
    lab = online.delta_labels(drv0, ndrv0, done, 3)
    np.testing.assert_allclose(lab, [0.50 - 0.52, 0.51 - 0.52, 0.52 - 0.12, 0.10 - 0.12, 0.11 - 0.12], atol=1e-6)


def test_delta_labels_mark_hazards_and_relief():
    """A hazard stretch gives negative labels, relief positive ones, the steady rise slightly negative."""
    rise = np.full(30, 0.001)
    rise[10:15] += 0.02                                      # in a hazard at steps 10-14
    h = 0.3 + np.concatenate([[0.0], np.cumsum(rise)])
    drv0, ndrv0 = h[:-1].copy(), h[1:].copy()
    ndrv0[24:] -= 0.3                                        # a hit at step 24
    lab = online.delta_labels(drv0, ndrv0, np.zeros(30, np.uint8), 4)
    assert (lab[7:15] < -0.01).all() and (lab[21:25] > 0.2).all() and np.allclose(lab[0:5], -0.004)


def test_relief_mode_unchanged():
    class M:
        pass
    m = M()
    m.hit = np.array([0, 0, 1, 0, 0], np.uint8)
    m.done = np.zeros(5, np.uint8)
    m.drv = np.array([[0.4, 0], [0.5, 0], [0.6, 0], [0.1, 0], [0.2, 0]], np.float32)
    idx = np.arange(5)
    np.testing.assert_array_equal(online.goal_labels(m, idx, 2, "relief"),
                                  online.hindsight_labels(m.hit, m.done, m.drv[:, 0], 2))
    with pytest.raises(ValueError):
        online.goal_labels(m, idx, 2, "nonsense")


def test_bodies_schedule():
    on = dict(bodies=32, bodies_hold=100, bodies_halve=50)
    got = [bodies_at(on, t) for t in (1, 100, 101, 150, 151, 200, 201, 351, 10_000)]
    assert got == [32, 32, 16, 16, 8, 8, 4, 1, 1]
    assert bodies_at(dict(bodies=1, bodies_hold=0, bodies_halve=0), 5) == 1
    assert bodies_at(dict(bodies=8, bodies_hold=0, bodies_halve=0), 10 ** 6) == 8     # no halving: N0 throughout
    on = dict(explore_eps=0.2, explore_eps_end=0.05, explore_decay=100)
    assert eps_at(on, 1) == 0.2 and abs(eps_at(on, 51) - 0.125) < 1e-12 and eps_at(on, 500) == 0.05
    assert eps_at(dict(explore_eps=0.2, explore_eps_end=None, explore_decay=0), 500) == 0.2


def test_shared_memory_keeps_bodies_apart():
    """Bodies write chunks into one long-term memory; each chunk and each newest window comes from one body."""
    on = dict(online.DEFAULT_ONLINE, stm=64, replay=8 * 40, store_base=1.0, store_warmup=0)
    L, k = 8, 4
    gms = [online.GatedMemory(on, L, 8, k, seed=1)]
    gms += [online.GatedMemory(on, L, 8, k, seed=1, store=gms[0].store, rng_key=(b,)) for b in (1, 2)]
    for step in range(1, 61):
        for b, gm in enumerate(gms):
            gm.stm.add(img=np.zeros((8, 8, 3), np.uint8), nimg=np.zeros((8, 8, 3), np.uint8),
                       drv=np.array([0.5, b], np.float32), ndrv=np.array([0.5, b], np.float32), pa=np.zeros(3),
                       pr=0.0, es=0, act=np.zeros(3), rew=0.0, done=0, hit=int(step % 13 == 0), cmd=0.0)
            gm.decide(step)
    st = gms[0].store
    D = max(on["relief_k"], k) - 1                           # a chunk is decided D steps after it ends
    assert st.n == 3 * ((60 - D) // L)
    ids = st.ltm["drv"][:st.n, :, 1]
    assert (ids == ids[:, :1]).all() and set(ids[:, 0]) == {0.0, 1.0, 2.0}
    for b, gm in enumerate(gms):
        assert (gm.newest(k - 1)["drv"][..., 1] == b).all()
    assert gms[0].replayed(5, np.random.default_rng(0))["drv"].shape == (5, L, 2)


@pytest.mark.parametrize("goal_mode", ["relief", "delta"])
def test_short_lifetime_with_bodies(tmp_path, goal_mode):
    out = str(tmp_path / "run")
    cmd = [sys.executable, os.path.join(DEV, "lifetime.py"), "--config", os.path.join(DEV, "configs", "AHR_gated.json"),
           "--set", "model.patch=16", "--set", "model.act_token=1", "--set", "model.d_model=32", "--set", "model.n_head=2",
           "--set", "model.d_ff=64", "--set", "model.pred_hidden=32", "--set", "model.pi_hidden=16",
           "--set", "ppo.act_coef=0.5", "--set", "ppo.pi_detach=1", "--set", "online.goal_mode=" + goal_mode,
           "--set", "online.bodies=4", "--set", "online.bodies_hold=40", "--set", "online.bodies_halve=20",
           "--set", "online.explore_eps_end=0.05", "--set", "online.explore_decay=60",
           "--set", "online.steps=100", "--set", "online.log_every=20", "--set", "online.save_every=100",
           "--set", "online.threads=2", "--set", "online.stm=64", "--set", "online.replay=4000",
           "--set", "env.respawn=true", "--set", "env.end_on_success=false", "--set", "env.max_steps=1000000000",
           "--set", "env.wrap=true", "--set", "env.hazards=2", "--seed", "2", "-o", out]
    subprocess.run(cmd, check=True, cwd=DEV, capture_output=True)
    rows = [json.loads(line) for line in open(os.path.join(out, "metrics.jsonl"))]
    assert [r["bodies"] for r in rows] == [4, 4, 2, 1, 1]
    assert all(r["act_frame_mse"] > 0 for r in rows[1:]) and abs(rows[-1]["eps"] - 0.05) < 1e-9
    assert os.path.exists(os.path.join(out, "ckpt_final.bin"))


@pytest.mark.parametrize("instinct", [0, 4])
def test_short_lifetime_universal(tmp_path, instinct):
    """The universal model (task 00023) lives a short lifetime with two bodies, with and without the instinct, and its
    frozen tests run."""
    out = str(tmp_path / "run")
    cmd = [sys.executable, os.path.join(DEV, "lifetime.py"), "--config", os.path.join(DEV, "configs", "AHR_gated.json"),
           "--set", "model.patch=16", "--set", "model.universal=1", "--set", "model.bins=8", "--set", "model.d_model=32",
           "--set", "model.n_head=2", "--set", "model.d_ff=64", "--set", "online.goal=0", "--set", "online.explore_eps=0",
           "--set", "online.instinct_m=%d" % instinct, "--set", "online.bodies=2", "--set", "online.bodies_hold=30",
           "--set", "online.bodies_halve=20", "--set", "online.steps=60", "--set", "online.log_every=30",
           "--set", "online.save_every=60", "--set", "online.threads=2", "--set", "online.stm=64",
           "--set", "online.replay=4000", "--set", "env.respawn=true", "--set", "env.end_on_success=false",
           "--set", "env.max_steps=1000000000", "--set", "env.wrap=true", "--set", "env.hazards=2",
           "--set", "env.hazard_cost=0", "--set", "env.hazard_click_add=0.5", "--seed", "3", "-o", out]
    subprocess.run(cmd, check=True, cwd=DEV, capture_output=True)
    rows = [json.loads(line) for line in open(os.path.join(out, "metrics.jsonl"))]
    assert [r["bodies"] for r in rows] == [2, 1]
    assert all(r["uni_ce_patch"] > 0 and r["uni_ce_body"] > 0 for r in rows)
    assert "instinct_override" in rows[-1]
    if instinct == 0:
        assert all(r["instinct_override"] == 0 for r in rows)
    import agent_lib   # nothing (e.g. online.click_bias, which needs a policy head) altered the newborn weights
    born = agent_lib.Agent.load(os.path.join(out, "ckpt_00000000.bin"))
    fresh = agent_lib.Agent(born.config, seed=3)
    np.testing.assert_array_equal(born.params(), fresh.params())
    ev = subprocess.run([sys.executable, os.path.join(DEV, "online.py"), "--eval", out, "--ckpts", "60"], cwd=DEV,
                        capture_output=True, text=True, check=True)
    assert "success" in ev.stdout and "forage: hits/1k" in ev.stdout


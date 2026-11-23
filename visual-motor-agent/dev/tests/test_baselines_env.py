"""Task 00120: the baselines' environment gives the default agent's input and the meal label's immediate part."""
import json
import os
import sys

import numpy as np

DEV = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, DEV)

import online  # noqa: E402
from baselines.vm_env import POISON_RISE, VMEnv, label_reward  # noqa: E402

CONFIGS = [os.path.join(DEV, "configs", c) for c in ("U_breakout.json", "U_agent.json")]


def test_poison_rise_matches_online():
    assert POISON_RISE == online.POISON_RISE


def test_reward_is_the_labels_immediate_part():
    for h0, h1, cut in [(0.5, 0.2, 0.0), (0.5, 0.501, 0.0), (0.5, 1.0, 0.3), (1.0, 1.0, 0.5), (0.3, 0.305, 0.0), (0.0, 0.0, -0.2)]:
        t = online.stored_labels(h0, h1, False, np.array([0.5]), 0.0, cut=cut)[0]
        assert abs(label_reward(h0, h1, cut) - (1.0 - 2.0 * float(t))) < 1e-6, (h0, h1, cut)


def test_random_lives_and_input(tmp_path):
    rng = np.random.default_rng(0)
    for c in CONFIGS:
        env = VMEnv(c, seed=3, out_dir=str(tmp_path / os.path.basename(c)), every=500)
        o = env.reset()
        assert set(o) == {"image", "drives", "prev"}
        assert o["image"].shape == (64, 64, 1) and o["image"].dtype == np.uint8 and o["drives"].shape == (2,)
        rs, rows = [], []
        for t in range(1500):
            mv = rng.uniform(-1, 1, 2)
            h0 = float(env.obs["drives"][0, 0])
            o, r, info, row = env.step(mv)
            cut = float(info["hunger_cut"][0]) if "hunger_cut" in info else 0.0
            t_ = online.stored_labels(h0, float(info["terminal_drives"][0, 0]), False, np.array([0.5]), 0.0, cut=cut)[0]
            assert abs(r - (1.0 - 2.0 * float(t_))) < 1e-6
            assert np.allclose(o["prev"], np.clip(mv, -1, 1))
            rs.append(r)
            if row:
                rows.append(row)
        assert len(rows) == 3 and rows[-1]["step"] == 1500
        assert sum(x["step_hits"] for x in rows) == sum(1 for r in rs if r == 1.0)
        lines = [json.loads(x) for x in open(tmp_path / os.path.basename(c) / "metrics.jsonl")]
        assert [x["step"] for x in lines] == [500, 1000, 1500]
        if "breakout" in c:
            assert "bo_hits" in lines[-1]


def test_vec_env_rows(tmp_path):
    from baselines.vm_env import VecVMEnv
    rng = np.random.default_rng(1)
    env = VecVMEnv(CONFIGS[0], seed=2, n=4, out_dir=str(tmp_path), every=250)
    o = env.reset()
    assert o["image"].shape == (4, 64, 64, 1) and o["drives"].shape == (4, 2) and o["prev"].shape == (4, 2)
    meals = 0
    for t in range(500):
        o, r, info, row = env.step(rng.uniform(-1, 1, (4, 2)))
        meals += int((r == 1.0).sum())
        assert np.array_equal(info["success"].astype(bool), r == 1.0)
    rows = [json.loads(x) for x in open(tmp_path / "metrics.jsonl")]
    assert [x["step"] for x in rows] == [1000, 2000] and rows[0]["bodies"] == 4
    assert abs(sum(x["step_hits"] for x in rows) * 4 - meals) < 1e-9

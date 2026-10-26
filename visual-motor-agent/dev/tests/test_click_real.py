"""The click test on real windows (task 00027, analyze_hazards.py click-real): the world copies do not disturb the real
run, the true click effects follow the world's rules, the predictions read the model as the instinct does, and a short
test runs end to end."""
import json
import os
import subprocess
import sys

import numpy as np
import pytest

DEV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, DEV)

import agent_lib  # noqa: E402
from analyze_hazards import (branch_hunger, click_real, click_truth, patch_values, place_click_states,  # noqa: E402
                             summarize_click_real)
from world import CircleWorld, EnvConfig  # noqa: E402

with open(os.path.join(DEV, "configs", "U_universal.json")) as f:
    U_ENV = json.load(f)["env"]
SMALL_UNI = dict(img=32, c1=4, c2=8, c3=8, feat=16, d_model=16, n_layer=2, n_head=4, d_ff=32, ctx=8, pred_img=16,
                 pc1=4, pc2=8, pred_hidden=16, use_prev_reward=1, logstd_init=-0.3, patch=8, universal=1, bins=8,
                 gray=1)


def random_moves(rng, n, p_click=0.5):
    return np.concatenate([rng.uniform(-1, 1, (n, 2)), rng.random((n, 1)) < p_click], 1).astype(np.float32)


def test_world_copies_leave_the_real_run():
    """click_truth's branch on a copy of the world, then the real step, gives the same run as plain steps (respawn,
    drifting hazards, poison, dots, torus)."""
    cfg = EnvConfig(**dict(U_ENV, h_rate=0.02))
    a, b = CircleWorld(16, cfg, seed=3), CircleWorld(16, cfg, seed=3)
    rng = np.random.default_rng(0)
    for _ in range(60):
        mv = random_moves(rng, 16)
        oa, ia = click_truth(a, mv)[:2]
        ob, _, _, ib = b.step(mv)
        np.testing.assert_array_equal(oa["image"], ob["image"])
        np.testing.assert_array_equal(ia["success"], ib["success"])
    sa, sb = a.get_state(), b.get_state()
    assert sa.keys() == sb.keys()
    for k in sa:
        np.testing.assert_array_equal(sa[k], sb[k], err_msg=k)


@pytest.mark.parametrize("click", [0, 1])
def test_true_effects_follow_the_rules(click):
    """Placed states with hunger 0.3: a click lands on food (relief h_relief, down to 0), on poison
    (+hazard_click_add) or on nothing (no effect), whether the real move clicks or not."""
    cfg = EnvConfig(**U_ENV)
    n = 300
    env = CircleWorld(n, cfg, seed=5)
    rng = np.random.default_rng(1)
    kind, cur, keep = place_click_states(env, cfg, rng, n)
    env.set_state(dict(cursor=cur, hunger=np.full(n, 0.3)))
    mv = np.zeros((n, 3), np.float32)
    mv[:, 2] = click
    _, _, got, true = click_truth(env, mv)
    assert keep.mean() > 0.9
    np.testing.assert_array_equal(got[keep], kind[keep])
    h1 = 0.3 + cfg.h_rate   # the next hunger without a click
    np.testing.assert_allclose(true[keep & (kind == 0)], max(h1 - cfg.h_relief, 0.0) - h1, atol=1e-9)
    np.testing.assert_allclose(true[keep & (kind == 1)], cfg.hazard_click_add, atol=1e-9)
    np.testing.assert_allclose(true[keep & (kind == 2)], 0.0, atol=1e-12)


def windows(cfg, S, T, seed):
    rng = np.random.default_rng(seed)
    img = rng.integers(0, 256, (S, T, cfg["img"], cfg["img"], 3), dtype=np.uint8)
    drv = rng.random((S, T, 2)).astype(np.float32)
    act = np.concatenate([rng.uniform(-1, 1, (S, T, 2)), rng.random((S, T, 1)) < 0.5], -1).astype(np.float32)
    es = np.zeros((S, T), np.uint8)
    es[:, 0] = 1
    cand = np.stack([random_moves(rng, S), random_moves(rng, S)], 1)
    return img, drv, act, es, cand


def test_branch_hunger_reads_the_move_token():
    """Each candidate's predicted next hunger is the full pass's with that candidate as the window's last move, also
    for a window cut to its last k steps (the context restarting there), and the values come from the window with the
    last move zeroed."""
    agent = agent_lib.Agent(SMALL_UNI, seed=2)
    img, drv, act, es, cand = windows(agent.config, 3, 8, 4)
    for k in (1, 3, 8):
        sl = slice(8 - k, 8)
        esk = es[:, sl].copy()
        esk[:, 0] = 1
        h, vals = branch_hunger(agent, img[:, sl], drv[:, sl], act[:, sl], esk, cand, values=True)
        for j in range(2):
            a2 = act[:, sl].copy()
            a2[:, -1] = cand[:, j]
            nb = agent.forward_uni(img[:, sl], drv[:, sl], a2, esk, values=False, move_probs=False, next_body=True)
            np.testing.assert_allclose(h[:, j], nb["next_body"][:, -1, 0], atol=1e-6)
        a0 = act[:, sl].copy()
        a0[:, -1] = 0
        np.testing.assert_allclose(vals, agent.forward_uni(img[:, sl], drv[:, sl], a0, esk)["values"], atol=1e-6)
        if k < 8:   # the cut window equals the full one with the context restarting at the cut
            esf = es.copy()
            esf[:, 8 - k] = 1
            hf = branch_hunger(agent, img, drv, act, esf, cand)[0]
            np.testing.assert_allclose(h, hf, atol=1e-5)
    agent.close()


def test_branch_hunger_gpu_fast_path():
    """On the GPU the candidates reuse the forward pass (also one that returned all values) and match the C++ model."""
    cuda_lib = pytest.importorskip("cuda_lib")
    if not cuda_lib.available() or not cuda_lib.has("vmc_uni_candidates"):
        pytest.skip("libvmcuda.so with vmc_uni_candidates not available")
    agent = agent_lib.Agent(SMALL_UNI, seed=2)
    m = cuda_lib.CudaModel(agent)
    img, drv, act, es, cand = windows(agent.config, 5, 4, 6)
    h_gpu, v_gpu = branch_hunger(m, img, drv, act, es, cand, values=True)
    h_cpu, v_cpu = branch_hunger(agent, img, drv, act, es, cand, values=True)
    np.testing.assert_allclose(h_gpu, h_cpu, atol=2e-5)
    np.testing.assert_allclose(v_gpu, v_cpu, atol=2e-5)
    m.close()
    agent.close()


def test_patch_values_match_the_reference():
    torch = pytest.importorskip("torch")  # noqa: F841
    from reference import torch_reference as ref
    for gray in (0, 1):
        cfg = dict(SMALL_UNI, gray=gray)
        img = np.random.default_rng(gray).integers(0, 256, (2, 1, 32, 32, 3), dtype=np.uint8)
        v, _ = ref.uni_values(cfg, img, np.zeros((2, 1, 2)), np.zeros((2, 1, 3)))
        np.testing.assert_allclose(patch_values(cfg, img[:, 0]), v[:, 0, 1:-1], atol=1e-7)


def test_summary_by_hand():
    kind = np.array([0, 0, 0, 1, 1, 2, 2, 3])
    true = np.array([-0.3, -0.3, 0.0, 0.5, 0.5, 0.0, 0.0, 0.2])
    pred = np.array([-0.2, 0.0, -0.1, 0.3, -0.01, 0.01, 0.2, 0.0])
    r = dict(kind=kind, true=true, seen_food=np.zeros(8, bool), seen_poison=np.array([0, 0, 0, 1, 0, 0, 0, 0], bool),
             pred_1=pred, act_err_1=np.full(8, 0.1), patch_1=np.full(8, 0.01), changed_1=np.full(8, np.nan))
    s = summarize_click_real(r, [1], 1, 8, 0, 2, 1)
    f, p, z = s["hist"]["1"]["food"], s["hist"]["1"]["poison"], s["hist"]["1"]["nothing"]
    assert (f["n"], f["n_felt"], f["right"]) == (3, 2, 0.5)          # -0.2 right, 0.0 not; the unfelt case not counted
    assert abs(f["predicted"] - (-0.1)) < 1e-12
    assert (p["n"], p["right"]) == (2, 0.5) and z["flat"] == 0.5 and s["both"] == 1
    assert s["poison_seen"]["right"] == 1.0 and s["poison_unseen"]["right"] == 0.0
    assert s["food_per_1k"] == 250.0


def test_click_real_end_to_end(tmp_path):
    """A short universal lifetime in the default world, then a short click-real test of its checkpoint."""
    out = str(tmp_path / "run")
    sets = ["model.img=32", "model.patch=8", "model.d_model=16", "model.n_head=2", "model.d_ff=32", "model.bins=8",
            "env.size=32", "online.steps=40", "online.log_every=20", "online.save_every=40", "online.threads=2",
            "online.stm=64", "online.replay=2000", "online.instinct_m=4", "online.device=cpu", "online.store_warmup=20"]
    cmd = [sys.executable, os.path.join(DEV, "lifetime.py"), "--config", os.path.join(DEV, "configs", "U_universal.json"),
           "--seed", "2", "-o", out] + sum([["--set", s] for s in sets], [])
    subprocess.run(cmd, check=True, cwd=DEV, capture_output=True)
    r = click_real(out, 40, n=6, steps=12)
    assert sorted(r["hist"]) == ["1", "2", "4", "8"]
    for h in r["hist"].values():
        assert sum(h[k]["n"] for k in ("food", "poison", "nothing")) + r["both"] == 6 * 12
        assert h["patch_mse"] > 0 and h["next_hunger_mae"] >= 0
    json.dumps(r)
    r0 = click_real(out, 40, n=6, steps=12, instinct=0)
    assert r0["instinct_m"] == 0 and r["instinct_m"] == 4

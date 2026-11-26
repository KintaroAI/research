"""Task 00128: the move token as a policy-gradient actor. CudaModel.set_move_weight: the move's loss row of each step times
its weight, plus ent times the gradient of minus the entropy of the move's bins; against the PyTorch reference. A weight
of 1 and no entropy is the old loss exactly; the next batch clears it. uni_act(pg=True): the move made is the first
sample and each body's baseline is the mean label of its candidates. lifetime.py: a short life with online.actor_pg stores
the advantages and logs them; the refusals.

Requires libvmcuda.so, a GPU and torch (CPU) for the GPU tests.
"""
import json
import os
import subprocess
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import agent_lib  # noqa: E402
import cuda_lib  # noqa: E402
import online  # noqa: E402

gpu = pytest.mark.skipif(not cuda_lib.available() or not cuda_lib.has("vmc_set_move_weight"),
                         reason="needs libvmcuda.so with vmc_set_move_weight and a GPU")
DEV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")


def life(tmp_path, *sets):
    cmd = [sys.executable, os.path.join(DEV, "lifetime.py"), "--config", os.path.join(DEV, "configs", "U_agent.json"),
           "--set", "online.steps=60", "--set", "online.log_every=30", "--set", "online.save_every=100000",
           "--set", "online.piece_rule=random", "--seed", "1", "-o", str(tmp_path / "run")]
    for s in sets:
        cmd += ["--set", s]
    return subprocess.run(cmd, cwd=DEV, capture_output=True, text=True)


@gpu
def test_a_life_with_the_actor(tmp_path):
    r = life(tmp_path, "online.actor_pg=1")
    assert r.returncode == 0, r.stderr[-2000:]
    rows = [json.loads(line) for line in open(tmp_path / "run" / "metrics.jsonl")]
    assert len(rows) == 2
    for row in rows:
        assert row["instinct_override"] == 0.0                         # the move made is always the actor's sample
        assert row["pg_scale"] > 0 and 0.0 < row["pg_pos"] < 1.0 and row["move_ent"] > 0


@gpu
@pytest.mark.parametrize("bad", [["online.instinct_h=4", "online.ring=0"],
                                 ["online.label=", "model.body_label=0", "online.full_dose=0", "online.label_refresh=0"],
                                 ["online.pg_clip=0"]], ids=["horizon", "no_label", "no_clip"])
def test_the_refusals(tmp_path, bad):
    r = life(tmp_path, "online.actor_pg=1", *bad)
    assert r.returncode != 0 and "online.actor_pg goes with" in r.stderr, r.stderr[-1000:]


def cfgs():
    from test_outage import BAND, CHAN, UNI as U
    return [U, BAND, dict(CHAN, body_label=1)]


IDS = ["plain", "piece", "piece_label"]


def weights(S, T, seed=3):
    rng = np.random.default_rng(seed)
    w = rng.uniform(-2.0, 2.0, (S, T)).astype(np.float32)
    w[0, 0] = 0.0
    return w


def gpu_grads(agent, b, mw=None, ent=0.0, mask=None):
    from test_outage import PC
    m = cuda_lib.CudaModel(agent)
    m.set_batch(*b)
    if mask is not None:
        m.set_step_mask(mask)
    if mw is not None:
        m.set_move_weight(mw, ent)
    st, g = m.loss_and_grads(PC), m.grads().copy()
    m.close()
    return st, g


@gpu
@pytest.mark.parametrize("case", ["weight", "entropy", "both"])
@pytest.mark.parametrize("k", range(3), ids=IDS)
def test_the_move_weight_matches_the_reference(k, case):
    pytest.importorskip("torch")
    from test_cuda_model import _ref_flat_grads, compare
    from test_outage import batches
    agent = agent_lib.Agent(cfgs()[k], seed=4)
    b, batch = batches(agent)
    S, T = b[0].shape[:2]
    w = weights(S, T) if case != "entropy" else np.ones((S, T), np.float32)
    ent = 0.0 if case == "weight" else 0.5
    _, g = gpu_grads(agent, b, w, ent)
    _, want = _ref_flat_grads(agent, dict(batch, move_weight=w, move_ent=ent))
    compare(agent, want, g, 1e-4)
    _, g0 = _ref_flat_grads(agent, batch)
    assert np.abs(want - g0).max() > 1e-3 * np.abs(g0).max()          # the weight mattered
    agent.close()


@gpu
def test_with_the_step_mask():
    pytest.importorskip("torch")
    from test_cuda_model import _ref_flat_grads, compare
    from test_outage import a_mask, batches
    agent = agent_lib.Agent(cfgs()[1], seed=4)
    b, batch = batches(agent)
    S, T = b[0].shape[:2]
    w, mask = weights(S, T), a_mask(S, T)
    _, g = gpu_grads(agent, b, w, 0.3, mask)
    _, want = _ref_flat_grads(agent, dict(batch, move_weight=w, move_ent=0.3, step_mask=mask.astype(bool)))
    compare(agent, want, g, 1e-4)
    agent.close()


@gpu
def test_weight_one_without_entropy_is_the_old_loss():
    from test_outage import batches
    agent = agent_lib.Agent(cfgs()[2], seed=4)
    b, _ = batches(agent)
    st0, g0 = gpu_grads(agent, b)
    st1, g1 = gpu_grads(agent, b, np.ones(b[0].shape[:2], np.float32), 0.0)
    assert abs(st0["loss"] - st1["loss"]) < 1e-6 * abs(st0["loss"])
    np.testing.assert_allclose(g1, g0, rtol=1e-5, atol=1e-6 * np.abs(g0).max())   # the GPU's atomic sums: not bitwise
    agent.close()


@gpu
def test_the_next_batch_clears_it_and_the_reported_loss_is_unweighted():
    from test_outage import PC, batches
    agent = agent_lib.Agent(cfgs()[0], seed=4)
    b, _ = batches(agent)
    m = cuda_lib.CudaModel(agent)
    m.set_batch(*b)
    l0, g0 = m.loss_and_grads(PC)["loss"], m.grads().copy()
    m.set_move_weight(weights(*b[0].shape[:2]), 0.5)
    l1, g1 = m.loss_and_grads(PC)["loss"], m.grads().copy()
    m.set_batch(*b)
    l2, g2 = m.loss_and_grads(PC)["loss"], m.grads().copy()
    assert abs(l1 - l0) < 1e-6 * abs(l0) and abs(l2 - l0) < 1e-6 * abs(l0)
    tol = 1e-6 * np.abs(g0).max()
    assert np.abs(g1 - g0).max() > 1e3 * tol
    np.testing.assert_allclose(g2, g0, rtol=1e-5, atol=tol)
    with pytest.raises(RuntimeError):
        m.set_move_weight(np.ones((2, 3), np.float32))                  # not the batch's shape
    m.close()
    agent.close()


def test_uni_act_pg_makes_the_first_sample_and_returns_the_baseline():
    from test_lookahead import SMALL_UNI
    agent = agent_lib.Agent(dict(SMALL_UNI, gray=1, in_bins=1, uni_chan=1, body_label=1), seed=5)
    c = agent.config
    rng = np.random.default_rng(0)
    N, T = 3, c["ctx"]
    img = rng.integers(0, 256, (N, T, c["img"], c["img"], 3)).astype(np.uint8)
    drv = rng.random((N, T, 2)).astype(np.float32)
    act = np.zeros((N, T, 3), np.float32)
    es = np.zeros((N, T), np.uint8)
    out = online.uni_act(agent, img, drv, act, es, c["bins"], np.random.default_rng(7), 6, tol=0.0, per_body=True, pg=True)
    move, over, _, h_made, _, base = out
    assert over == 0.0
    # the same draws by hand: 6 candidates at temperature 1, the first is the move made; the baseline their mean label
    rng2 = np.random.default_rng(7)
    probs = agent.forward_uni(img, drv, act, es, values=False, move_probs=True)["move_probs"][:, -1]
    idx = np.minimum((np.cumsum(probs, -1)[:, None] < rng2.random((N, 6, 3, 1))).sum(-1), c["bins"] - 1)
    cc = idx / (c["bins"] - 1)
    cand = np.stack([2 * cc[..., 0] - 1, 2 * cc[..., 1] - 1, (cc[..., 2] > 0.5).astype(np.float64)], -1).astype(np.float32)
    h = []
    for j in range(6):
        a_ = act.copy()
        a_[:, -1] = cand[:, j]
        h.append(agent.forward_uni(img, drv, a_, es, values=False, move_probs=False, next_body=True)["next_body"][:, -1, 2])
    h = np.stack(h, 1)
    np.testing.assert_array_equal(move, cand[:, 0])
    np.testing.assert_allclose(h_made, h[:, 0], atol=1e-5)
    np.testing.assert_allclose(base, h.mean(1), atol=1e-5)
    agent.close()

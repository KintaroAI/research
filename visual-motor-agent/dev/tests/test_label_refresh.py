"""The body label re-labeled at every update (task 00112, online.label_refresh with the stored label's rule): memory keeps
the facts and one step past each chunk; at each update every transformer's label of step t is the stored label's rule with
its current prediction for the move made at t + 1 (the bootstrap window, one step later). Without the option nothing changes."""
import json
import os
import subprocess
import sys

import numpy as np
import pytest

DEV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, DEV)

import cuda_lib  # noqa: E402
from online import DEFAULT_ONLINE, GatedMemory, meal_parts, meal_target, stored_labels  # noqa: E402

HAVE_GPU = cuda_lib.available()
CFG = os.path.join(DEV, "configs", "U_view.json")
G = 0.95


def test_the_parts_give_the_stored_label():
    # a meal, drift, a poison below the cap, a poison at the cap (its dose cut off), drift at the cap, an episode's end
    h0 = np.array([0.6, 0.0, 0.001, 0.502, 1.0, 1.0, 0.3, 0.3])
    h1 = np.array([0.0, 0.001, 0.502, 1.0, 1.0, 1.0, 0.301, 0.302])
    cut = np.array([0.0, 0.0, 0.0, 0.003, 0.501, 0.001, 0.0, 0.0])
    done = np.array([0, 0, 0, 0, 0, 0, 1, 0])
    real, tw = meal_parts(h0, h1, done, G, drift=False, cut=cut)
    for tail in (0.0, 0.3, 0.5, 0.9, 1.0):
        for t in range(len(h0) - 1):   # the last step of a stretch has no next move in it
            want = stored_labels(h0[t], h1[t], done[t], [tail], G, cut[t])[0]
            assert meal_target(real[t], tw[t], tail) == pytest.approx(want, abs=1e-6), (t, tail)
    np.testing.assert_allclose(1 - 2 * meal_target(real[4], tw[4], 0.5), -0.501, atol=1e-6)   # a poison at the cap: its full dose
    r0, w0 = meal_parts(h0, h1, done, G, drift=False)                     # without the cut: task 00109's rule
    assert r0[4] == 0.0 and r0[3] == pytest.approx(-0.498)


def stream(on, h, cut, S=3, L=8, steps=None):
    gm = GatedMemory(on, L, 8, 0, seed=1, labels=S)
    rng = np.random.default_rng(2)
    vhs = rng.random((len(h) - 1, S)).astype(np.float32)
    for t in range(len(h) - 1):
        extra = dict(cut=cut[t]) if "cut" in gm.stm.fields else {}
        gm.stm.add(img=np.full((8, 8, 3), t % 256, np.uint8), drv=np.array([h[t], 0], np.float32),
                   ndrv=np.array([h[t + 1], 0], np.float32), pa=np.zeros(3, np.float32), pr=0.0, es=0,
                   act=np.array([t / 1000, 0, 0], np.float32), rew=0.0, done=0, hit=0, cmd=0.0, vhs=vhs[t], **extra)
        gm.write_label(on["label_gamma"])
        gm.note(np.zeros(3))
        gm.decide(t + 1)
    return gm, vhs


BASE = dict(DEFAULT_ONLINE, memory="gated", label="meal", label_gamma=G, stm=256, replay=1600, store_warmup=10_000, full_dose=1)


def hunger(n=90):
    h = np.zeros(n, np.float32)
    for t in range(1, n):   # drift, a poison every 9 steps, a meal every 23
        h[t] = 0.0 if t % 23 == 0 else min(1.0, h[t - 1] + 0.001 + (0.5 if t % 9 == 4 else 0.0))
    cut = np.zeros(n, np.float32)
    for t in range(n - 1):
        if t % 23 != 22:
            cut[t] = max(0.0, h[t] + 0.001 + (0.5 if (t + 1) % 9 == 4 else 0.0) - 1.0)
    return h, cut


def check_rows(r, S=3):
    """Every window: the bootstrap window is the window one step later, and the label made from each transformer's
    prediction for the next move (here the predictions stored while acting, vhs) is the label stored once."""
    np.testing.assert_array_equal(r["b_act"][:, :-1], r["act"][:, 1:])
    np.testing.assert_array_equal(r["b_drv"][:, :-1], r["drv"][:, 1:])
    np.testing.assert_array_equal(r["b_img"][:, :-1], r["img"][:, 1:])
    for k in range(S):
        lab = meal_target(r["pr_real"][:, :-1], r["tw"][:, :-1], r["vhs"][:, 1:, k])
        np.testing.assert_allclose(lab, r["lab"][:, :-1, k], atol=1e-6)
    assert (r["tw"][:, -1] > 0).any()   # the window's last step has its next move too (one step past the chunk)


def test_memory_gives_the_parts_and_the_bootstrap_windows():
    h, cut = hunger()
    gm, vhs = stream(dict(BASE, label_refresh=1), h, cut)
    assert gm.refresh and gm.lrefresh and gm.vn == 1 and gm.store.extra == 1
    assert {"pr_real", "tw", "b_img", "b_drv", "b_act", "b_es"} <= set(gm.keys)
    assert gm.store.n > 3
    rng = np.random.default_rng(5)
    check_rows(gm.newest(1))
    check_rows(gm.replayed(6, rng))
    check_rows(gm._from_stm(gm.stm.starts(8, 5, rng, 1), 1))
    with pytest.raises(ValueError):
        gm.newest(0)   # the newest window needs its next step


def test_without_the_option_memory_is_unchanged():
    h, cut = hunger()
    gm, _ = stream(BASE, h, cut)
    assert not gm.refresh and gm.vn == 0 and gm.store.extra == 0
    assert "pr_real" not in gm.keys and "b_img" not in gm.keys


def run_life(tmp_path, sets):
    cmd = [sys.executable, os.path.join(DEV, "lifetime.py"), "--config", CFG, "-o", str(tmp_path / "run")] + sum([["--set", s] for s in sets], [])
    env = dict(os.environ)
    env.pop("CUDA_MPS_PIPE_DIRECTORY", None)
    return subprocess.run(cmd, cwd=DEV, capture_output=True, text=True, env=env)


def test_the_option_needs_the_stored_label(tmp_path):
    r = run_life(tmp_path, ["online.steps=10", "online.label_refresh=1"])
    assert r.returncode != 0 and "label_refresh" in r.stderr


@pytest.mark.skipif(not HAVE_GPU, reason="needs libvmcuda.so and a GPU")
def test_a_short_life_and_its_value_test(tmp_path):
    out = str(tmp_path / "run")
    r = run_life(tmp_path, ["online.steps=80", "online.log_every=40", "online.save_every=80", "online.label=meal", "online.label_gamma=0.95",
                            "model.body_label=1", "online.full_dose=1", "online.label_refresh=1", "env.inertia=4"])
    assert r.returncode == 0, r.stderr[-2000:]
    rows = [json.loads(line) for line in open(os.path.join(out, "metrics.jsonl"))]
    assert rows[-1]["step"] == 80 and "meal" in rows[-1] and "meal_stored" in rows[-1]
    assert json.load(open(os.path.join(out, "info.json")))["config"]["online"]["label_refresh"] == 1
    env = dict(os.environ)
    env.pop("CUDA_MPS_PIPE_DIRECTORY", None)
    r = subprocess.run([sys.executable, "eval_hunger.py", "value", out, "--ticks", "200", "--bodies", "4"], cwd=DEV,
                       capture_output=True, text=True, env=env)
    assert r.returncode == 0, r.stderr[-2000:]
    row = json.loads(r.stdout.strip().splitlines()[-1])
    assert row["mode"] == "value" and "meal" in row

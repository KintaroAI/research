"""A poison costs its full dose at any hunger (task 00110, online.full_dose with the stored label): the world reports the
hunger the clip at 1 cut off (info hunger_cut), memory keeps it (field cut), and the stored label, the monitors and the
frozen value test count it in the rise. What the model senses is unchanged; without the option nothing changes."""
import json
import os
import subprocess
import sys

import numpy as np
import pytest

DEV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, DEV)

import cuda_lib  # noqa: E402
from online import DEFAULT_ONLINE, GatedMemory, MealMonitor, stored_labels  # noqa: E402
from world import make_config, make_world  # noqa: E402

HAVE_GPU = cuda_lib.available()
CFG = os.path.join(DEV, "configs", "U_view.json")


def V(t):
    return 1.0 - 2.0 * np.asarray(t, np.float64)


def walk(h_start, steps=600, n=32, seed=3):
    """A random walk of n bodies in U_view's world from hunger h_start: per step the hunger before and after, the cut, the
    meals and the poisons."""
    cfg = make_config(json.load(open(CFG))["env"])
    assert cfg.hazard_cost == 0 and cfg.turn_cost == 0 and cfg.hazard_click_add == 0.5
    env = make_world(n, cfg, seed=seed)
    env.observe()
    env.hunger = np.full(n, h_start, np.float64)
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(steps):
        h0 = env.hunger.copy()
        _, _, _, inf = env.step(np.concatenate([rng.uniform(-1, 1, (n, 2)), np.zeros((n, 1))], 1).astype(np.float32))
        out.append((h0, inf["terminal_drives"][:, 0].astype(np.float64), inf["hunger_cut"], inf["success"], inf["gt_hazard_click"]))
    return cfg, out


def test_the_world_reports_what_the_cap_cut_off():
    cfg, out = walk(0.95)
    full = 0
    for h0, h1, cut, ate, poison in out:
        assert (cut >= 0).all()
        np.testing.assert_allclose(cut[h1 < 1.0], 0.0)                     # below the cap nothing is cut
        no_meal = ~ate
        want = h0 + cfg.h_rate + cfg.hazard_click_add * poison             # what hunger would be without the cap
        np.testing.assert_allclose((h1 + cut)[no_meal], want[no_meal], atol=1e-6)
        full += int((poison & (h0 >= 1.0 - 1e-9) & no_meal).sum())
        if (poison & (h0 >= 1.0 - 1e-9) & no_meal).any():                  # a poison at the cap: its whole dose was cut
            np.testing.assert_allclose(cut[poison & (h0 >= 1.0 - 1e-9) & no_meal], cfg.hazard_click_add + cfg.h_rate, atol=1e-6)
    assert full > 0, "the walk never touched a poison at the cap"


def test_below_the_cap_the_world_is_unchanged():
    _, out = walk(0.0, steps=300)
    for h0, h1, cut, ate, poison in out:
        lo = h0 + 0.501 < 1.0
        np.testing.assert_allclose(cut[lo], 0.0)


def test_the_label_counts_the_full_dose():
    nxt = np.array([0.0, 0.25, 0.5])
    below = stored_labels(0.2, 0.701, 0, nxt, 0.95)                        # a poison below the cap
    at_cap = stored_labels(1.0, 1.0, 0, nxt, 0.95, cut=0.501)              # the same poison at the cap, its dose cut off
    np.testing.assert_allclose(at_cap, below, atol=1e-6)
    np.testing.assert_allclose(stored_labels(1.0, 1.0, 0, nxt, 0.95), stored_labels(1.0, 1.0, 0, nxt, 0.95, cut=0.0))
    np.testing.assert_allclose(V(stored_labels(1.0, 1.0, 0, nxt, 0.95, cut=0.001)), [0.95, 0.475, 0.0], atol=1e-6)   # drift: free
    np.testing.assert_allclose(V(stored_labels(0.6, 0.0, 0, nxt, 0.95, cut=0.0)), [1, 1, 1])                       # a meal: 1


def test_the_monitor_counts_it_too():
    m, m0 = MealMonitor(1, 0.95, drift=False), MealMonitor(1, 0.95, drift=False)
    for h0, h1, c in ((1.0, 1.0, 0.501), (1.0, 0.0, 0.0)):                 # a poison at the cap, then a meal
        m.add(np.array([0.5]), np.array([h0]), np.array([h1]), np.array([c]))
        m0.add(np.array([0.5]), np.array([h0]), np.array([h1]))
    r, r0 = m.take(), m0.take()
    assert r["real"][1] == pytest.approx(-0.501 + 0.95) and r0["real"][1] == pytest.approx(0.95)   # the step before the meal


def stream(on, h, cut, S=3, L=8):
    gm = GatedMemory(on, L, 8, 0, seed=1, labels=S)
    vhs = np.random.default_rng(2).random((len(h) - 1, S)).astype(np.float32)
    first = {}
    for t in range(len(h) - 1):
        extra = dict(cut=cut[t]) if "cut" in gm.stm.fields else {}
        gm.stm.add(img=np.full((8, 8, 3), t % 256, np.uint8), drv=np.array([h[t], 0], np.float32),
                   ndrv=np.array([h[t + 1], 0], np.float32), pa=np.zeros(3, np.float32), pr=0.0, es=0,
                   act=np.array([t / 1000, 0, 0], np.float32), rew=0.0, done=0, hit=0, cmd=0.0, vhs=vhs[t], **extra)
        i = gm.write_label(on["label_gamma"])
        if i is not None:
            first[t - 1] = gm.stm.lab[i].copy()
        gm.note(np.zeros(3))
        gm.decide(t + 1)
    return gm, vhs, first


def test_memory_keeps_the_cut_only_with_the_option():
    base = dict(DEFAULT_ONLINE, memory="gated", label="meal", label_gamma=0.95, stm=256, replay=1600, store_warmup=10_000)
    h = np.ones(60, np.float32)                                           # at the cap all the time
    cut = np.where(np.arange(60) % 7 == 3, 0.501, 0.001).astype(np.float32)   # a poison every 7 steps
    gm, vhs, first = stream(dict(base, full_dose=1), h, cut)
    assert "cut" in gm.stm.fields and "cut" in gm.keys
    for t in range(len(h) - 3):
        np.testing.assert_allclose(first[t], stored_labels(h[t], h[t + 1], 0, vhs[t + 1], 0.95, cut[t]), atol=1e-7)
    free = V(stored_labels(h[3], h[4], 0, vhs[4], 0.95))                  # the poison step had the cap hidden it
    np.testing.assert_allclose(V(first[3]), np.maximum(free - 0.501, -1.0), atol=1e-6)   # it costs its full dose
    gm0, vhs0, first0 = stream(base, h, cut)
    assert "cut" not in gm0.stm.fields                                    # without the option the layout is task 00109's
    for t in range(len(h) - 3):
        np.testing.assert_allclose(first0[t], stored_labels(h[t], h[t + 1], 0, vhs0[t + 1], 0.95), atol=1e-7)


def test_the_option_needs_the_stored_label(tmp_path):
    cmd = [sys.executable, os.path.join(DEV, "lifetime.py"), "--config", CFG, "-o", str(tmp_path / "run"),
           "--set", "online.steps=10", "--set", "online.full_dose=1"]
    r = subprocess.run(cmd, cwd=DEV, capture_output=True, text=True)
    assert r.returncode != 0 and "full_dose" in r.stderr


@pytest.mark.skipif(not HAVE_GPU, reason="needs libvmcuda.so and a GPU")
def test_a_short_life_and_its_value_test(tmp_path):
    out = str(tmp_path / "run")
    sets = ["online.steps=80", "online.log_every=40", "online.save_every=80", "online.label=meal", "online.label_gamma=0.95",
            "model.body_label=1", "online.full_dose=1", "env.inertia=4"]
    cmd = [sys.executable, os.path.join(DEV, "lifetime.py"), "--config", CFG, "--seed", "2", "-o", out] + sum([["--set", s] for s in sets], [])
    env = dict(os.environ)
    env.pop("CUDA_MPS_PIPE_DIRECTORY", None)
    r = subprocess.run(cmd, cwd=DEV, capture_output=True, text=True, env=env)
    assert r.returncode == 0, r.stderr[-2000:]
    rows = [json.loads(line) for line in open(os.path.join(out, "metrics.jsonl"))]
    assert rows[-1]["step"] == 80 and "meal_stored" in rows[-1]
    assert json.load(open(os.path.join(out, "info.json")))["config"]["online"]["full_dose"] == 1
    r = subprocess.run([sys.executable, "eval_hunger.py", "value", out, "--ticks", "200", "--bodies", "4"], cwd=DEV,
                       capture_output=True, text=True, env=env)
    assert r.returncode == 0, r.stderr[-2000:]
    row = json.loads(r.stdout.strip().splitlines()[-1])
    assert row["mode"] == "value" and "meal" in row

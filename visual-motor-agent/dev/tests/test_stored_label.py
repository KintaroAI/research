"""The stored meal label (task 00109, online.label = meal, with model.body_label): each step's label is written once, one tick
later, from what happened (a meal 1; a poison's rise a cost; no cost for the drift) and each transformer's own prediction for
the next step's move, and never re-labeled. The owner's mechanism on a table, the label's arithmetic, memory writing each
label once and handing it out unchanged, the options' checks, and a short life on the GPU."""
import json
import os
import subprocess
import sys

import numpy as np
import pytest

DEV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, DEV)

import cuda_lib  # noqa: E402
from online import DEFAULT_ONLINE, GatedMemory, MEAL_BIN_NAMES, stored_labels  # noqa: E402

HAVE_GPU = cuda_lib.available()


def V(t):
    return 1.0 - 2.0 * np.asarray(t, np.float64)


def test_the_labels_arithmetic():
    nxt = np.array([0.0, 0.25, 0.5])                                       # three transformers' predictions (T): V 1, .5, 0
    np.testing.assert_allclose(V(stored_labels(0.2, 0.0, 0, nxt, 0.95)), [1, 1, 1])            # a meal: 1, whatever follows
    np.testing.assert_allclose(V(stored_labels(0.2, 0.201, 0, nxt, 0.95)), [0.95, 0.475, 0.0], atol=1e-6)   # no drift cost
    np.testing.assert_allclose(V(stored_labels(0.2, 0.701, 0, nxt, 0.95)), [0.449, -0.026, -0.501], atol=1e-6)   # poison
    np.testing.assert_allclose(V(stored_labels(0.2, 0.201, 1, nxt, 0.95)), [0, 0, 0], atol=1e-6)   # the episode ended: no tail
    np.testing.assert_allclose(V(stored_labels(0.9, 1.0, 0, np.array([1.0]), 0.95)), [-1.0], atol=1e-6)   # clipped at -1


def test_the_owners_mechanism_on_a_table():
    """Runs of N ticks ending in a meal. In each run every step's label is stored once from the table as it is then (the
    prediction for the next step's move); between runs the table learns the newest stored labels. After the first run only
    the meal's 1 is stored; after the second 0.95 before it; then 0.9025 - the front moves back one step per run, and an
    earlier run's labels are never changed."""
    N, gamma = 6, 0.95
    h0 = 0.2 + 0.001 * np.arange(N)
    h1 = h0 + 0.001
    h1[-1] = 0.0                                                          # the last move eats
    table = np.full(N + 1, 0.5)                                           # T for each state's move; nothing known: V 0
    runs = []
    for r in range(N):
        lab = np.array([stored_labels(h0[t], h1[t], 0, table[t + 1:t + 2], gamma)[0] for t in range(N)])
        runs.append(lab.copy())
        table[:N] = lab                                                   # the model learns its memory's newest labels
    first = V(runs[0])
    assert first[-1] == 1.0 and np.allclose(first[:-1], 0.0)            # only the meal's 1
    second = V(runs[1])
    assert second[-2] == pytest.approx(0.95) and np.allclose(second[:-2], 0.0)
    third = V(runs[2])
    assert third[-3] == pytest.approx(0.9025) and np.allclose(third[:-3], 0.0)
    np.testing.assert_allclose(V(runs[-1]), 0.95 ** np.arange(N)[::-1], atol=1e-6)


def stream(on, h, S=3, L=8, seed=2):
    """A gated memory fed the hunger series h with each step's predictions vhs [S] (random), writing labels as the life does."""
    gm = GatedMemory(on, L, 8, 0, seed=1, labels=S)
    T = len(h) - 1
    vhs = np.random.default_rng(seed).random((T, S)).astype(np.float32)
    first = {}
    for t in range(T):
        gm.stm.add(img=np.full((8, 8, 3), t % 256, np.uint8), drv=np.array([h[t], 0], np.float32),
                   ndrv=np.array([h[t + 1], 0], np.float32), pa=np.zeros(3, np.float32), pr=0.0, es=0,
                   act=np.array([t / 1000, 0, 0], np.float32), rew=0.0, done=0, hit=0, cmd=0.0, vhs=vhs[t])
        i = gm.write_label(on["label_gamma"])
        if i is not None:
            first[t - 1] = gm.stm.lab[i].copy()                           # step t - 1's label, the moment it is written
        gm.note(np.zeros(3))
        gm.decide(t + 1)
    return gm, vhs, first


def test_memory_writes_each_label_once_and_hands_it_out_unchanged():
    on = dict(DEFAULT_ONLINE, memory="gated", label="meal", label_gamma=0.95, stm=256, replay=1600, store_warmup=10_000)
    rng = np.random.default_rng(1)
    h = [0.3]
    for _ in range(400):
        e = rng.random()
        h.append(0.0 if e < 0.05 else min(1.0, h[-1] + (0.5 if e < 0.08 else 0.001)))
    h = np.array(h, np.float32)
    gm, vhs, first = stream(on, h)
    for t in range(len(h) - 2):                                           # each written from the next step's predictions
        np.testing.assert_allclose(first[t], stored_labels(h[t], h[t + 1], 0, vhs[t + 1], 0.95), atol=1e-7)
    assert gm.store.n > 10 and "lab" in gm.keys
    rows = gm.rows(16, np.random.default_rng(4), 1)
    steps = np.rint(rows["act"][..., 0] * 1000).astype(int)
    want = np.stack([np.stack([first[t] for t in row]) for row in steps])
    np.testing.assert_array_equal(rows["lab"], want)                      # the windows carry the labels as first written
    assert steps[0, -1] == len(h) - 3                                     # the newest window ends a step back


@pytest.mark.parametrize("bad", [["online.label=meal"], ["model.body_label=1"],
                                 ["online.label=meal", "model.body_label=1", "online.value_gamma=0.9", "online.future_k=4"]],
                         ids=["no_body_label", "no_label", "with_the_hunger_value"])
def test_the_options_go_together(tmp_path, bad):
    cmd = [sys.executable, os.path.join(DEV, "lifetime.py"), "--config", os.path.join(DEV, "configs", "U_view.json"),
           "-o", str(tmp_path / "run"), "--set", "online.steps=10"] + sum([["--set", s] for s in bad], [])
    r = subprocess.run(cmd, cwd=DEV, capture_output=True, text=True)
    assert r.returncode != 0 and ("online.label" in r.stderr or "invalid model config" in r.stderr)


@pytest.mark.skipif(not HAVE_GPU, reason="needs libvmcuda.so and a GPU")
def test_a_short_life(tmp_path):
    out = str(tmp_path / "run")
    sets = ["online.steps=80", "online.log_every=40", "online.save_every=80", "online.label=meal", "online.label_gamma=0.95",
            "model.body_label=1"]
    cmd = [sys.executable, os.path.join(DEV, "lifetime.py"), "--config", os.path.join(DEV, "configs", "U_view.json"),
           "--seed", "2", "-o", out] + sum([["--set", s] for s in sets], [])
    env = dict(os.environ)
    env.pop("CUDA_MPS_PIPE_DIRECTORY", None)
    r = subprocess.run(cmd, cwd=DEV, capture_output=True, text=True, env=env)
    assert r.returncode == 0, r.stderr[-2000:]
    rows = [json.loads(line) for line in open(os.path.join(out, "metrics.jsonl"))]
    assert rows[-1]["step"] == 80
    for k in ("meal", "meal_stored"):
        assert all(len(rows[-1][k][f]) == len(MEAL_BIN_NAMES) for f in ("pred", "real", "n"))
    info = json.load(open(os.path.join(out, "info.json")))
    assert info["config"]["model"]["body_label"] == 1 and info["config"]["online"]["label"] == "meal"

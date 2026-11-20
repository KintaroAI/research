"""The instinct's candidates at a temperature (task 00113, online.instinct_temp): the first candidate is drawn at
temperature 1 as before, the other m - 1 are redrawn from p^(1/T) renormalized; T = 1 repeats the earlier draws exactly."""
import json
import os
import subprocess
import sys

import numpy as np
import pytest

DEV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, DEV)

import cuda_lib  # noqa: E402
from online import DEFAULT_ONLINE, uni_act  # noqa: E402

HAVE_GPU = cuda_lib.available()
CFG = os.path.join(DEV, "configs", "U_view.json")
BINS, N, M = 32, 64, 16


class Fake:
    """A stand-in model: sharp move predictions (90% in one bin), and a label for each candidate that is flat (score
    None) or given by a function of the candidate. It records the candidates it was asked to score."""

    def __init__(self, score=None, top=0.9):
        self.config = dict(body_label=1)
        p = np.full((N, 3, BINS), (1 - top) / (BINS - 1))
        p[:, :, 20] = top
        self.p, self.score, self.seen = p, score, []

    def forward_uni(self, img, drv, act, es, values=False, move_probs=False, next_body=False):
        if move_probs:
            return dict(move_probs=self.p[:len(img), None].repeat(img.shape[1], 1))
        c = act[:, -1]
        self.seen.append(c.copy())
        v = np.zeros(len(c)) if self.score is None else self.score(c)
        nb = np.zeros((len(c), act.shape[1], 3), np.float32)
        nb[:, -1, 2] = v
        return dict(next_body=nb)


def call(model, seed, temp=None, m=M):
    z = lambda *s: np.zeros(s, np.float32)   # noqa: E731
    kw = {} if temp is None else dict(temp=temp)
    return uni_act(model, z(N, 1, 8, 8, 1), z(N, 1, 2), z(N, 1, 3), np.zeros((N, 1), np.uint8), BINS, np.random.default_rng(seed),
                   m, 0.002, **kw)


def old_first(model, seed):
    """The first candidate as the code before task 00113 drew it."""
    rng = np.random.default_rng(seed)
    idx = np.minimum((np.cumsum(model.p, -1)[:, None] < rng.random((N, M, 3, 1))).sum(-1), BINS - 1)
    c = idx[:, 0] / (BINS - 1)
    return np.stack([2 * c[..., 0] - 1, 2 * c[..., 1] - 1, (c[..., 2] > 0.5).astype(np.float64)], -1).astype(np.float32)


def distinct(cands):
    c = np.round(cands.reshape(N, M, 3)[..., :2], 4)
    return np.mean([len({tuple(x) for x in row}) for row in c])


def test_temperature_1_is_the_old_draw():
    for seed in range(3):
        a1, *_ = call(Fake(), seed, 1.0)
        a0, *_ = call(Fake(), seed)
        np.testing.assert_array_equal(a1, a0)
        np.testing.assert_array_equal(a1, old_first(Fake(), seed))   # nothing at stake: the first candidate is made


def test_a_higher_temperature_varies_the_others_and_keeps_the_first():
    f1, f2 = Fake(), Fake()
    a1, *_ = call(f1, 7, 1.0)
    a2, *_ = call(f2, 7, 2.0)
    np.testing.assert_array_equal(a1, a2)                                # the first candidate is the same draw
    c1, c2 = np.concatenate(f1.seen), np.concatenate(f2.seen)
    np.testing.assert_array_equal(c1.reshape(N, M, 3)[:, 0], c2.reshape(N, M, 3)[:, 0])
    d1, d2 = distinct(c1), distinct(c2)
    assert d2 > d1 + 3, (d1, d2)                                         # T = 2: far more varied candidates
    off = (np.abs(c2.reshape(N, M, 3)[:, 1:, 0] - (2 * 20 / 31 - 1)) > 1e-6).mean()
    assert 0.5 < off < 0.8, off                                          # about two thirds of the redrawn ones leave the top bin


def test_the_instinct_finds_the_better_move_more_often():
    target = -1.0                                                       # the label is best far left, away from the model's bin
    score = lambda c: np.abs(c[:, 0] - target)   # noqa: E731
    for_t = {}
    for t in (1.0, 2.0):
        a, override, gain = call(Fake(score), 3, t)
        for_t[t] = (np.abs(a[:, 0] - target).mean(), override)
    assert for_t[2.0][0] < for_t[1.0][0] and for_t[2.0][1] > for_t[1.0][1]


def test_the_default_is_off():
    assert DEFAULT_ONLINE["instinct_temp"] == 1.0


@pytest.mark.skipif(not HAVE_GPU, reason="needs libvmcuda.so and a GPU")
def test_a_short_life_and_its_frozen_test(tmp_path):
    out = str(tmp_path / "run")
    sets = ["online.steps=80", "online.log_every=40", "online.save_every=80", "online.label=meal", "online.label_gamma=0.95",
            "model.body_label=1", "online.full_dose=1", "env.inertia=4", "online.instinct_temp=2"]
    env = dict(os.environ)
    env.pop("CUDA_MPS_PIPE_DIRECTORY", None)
    r = subprocess.run([sys.executable, "lifetime.py", "--config", CFG, "-o", out] + sum([["--set", s] for s in sets], []), cwd=DEV,
                       capture_output=True, text=True, env=env)
    assert r.returncode == 0, r.stderr[-2000:]
    assert json.load(open(os.path.join(out, "info.json")))["config"]["online"]["instinct_temp"] == 2
    r = subprocess.run([sys.executable, "eval_candidates.py", out], cwd=DEV, capture_output=True, text=True, env=env)
    assert r.returncode == 0, r.stderr[-2000:]
    row = json.loads(r.stdout.strip().splitlines()[-1])
    assert row["mode"] == "candidates" and row["distinct"] > 1

"""SenseWorld (task 00081): the view half is the world's; the text half is the sentence; each sense goes off by itself, for longer
than the context, never both; the control's words mean nothing; a short lifetime and its frozen test.
"""
import json
import os
import subprocess
import sys

import numpy as np
import pytest

DEV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, DEV)

import cuda_lib  # noqa: E402
from world import make_config, make_world  # noqa: E402
from world.sense_world import LETTERS, LEVELS, WORDS, sentence_frames  # noqa: E402

ENV = json.load(open(os.path.join(DEV, "configs", "U_sense.json")))["env"]


def worlds(sense=None, n=4, seed=1):
    plain = {k: v for k, v in ENV.items() if k != "sense"}
    return make_world(n, make_config(dict(ENV, sense=sense or {}), auto_reset=True), seed=seed), make_world(n, make_config(plain, auto_reset=True), seed=seed)


def moves(rng, n=4):
    return rng.uniform(-1, 1, (n, 3)).astype(np.float32)


def test_letters_fit_the_bins():
    assert len(LETTERS) + 1 <= 32 and len(set(LEVELS.tolist())) == len(LETTERS) and LEVELS.min() > 0
    assert np.diff(np.sort(LEVELS)).min() >= 255 / 32


def test_sentence_frames():
    t = sentence_frames(64)
    assert t.shape == (2, 4, 2, 64, 64)
    f = t[0, 1, 0]                                      # food right near
    for row, word in enumerate(("food", "right", "near")):
        for col, ch in enumerate(word):
            assert (f[row * 8:(row + 1) * 8, col * 8:(col + 1) * 8] == LEVELS[LETTERS.index(ch)]).all()
        assert (f[row * 8:(row + 1) * 8, len(word) * 8:] == 0).all()
    assert (f[24:] == 0).all()


def test_the_view_is_the_world_until_the_eyes_close():
    s, w = worlds()
    rng = np.random.default_rng(0)
    assert s.observe()["image"].shape == (4, 64, 128, 3)
    closed = 0
    for _ in range(400):
        m = moves(rng)
        a, _, _, _ = s.step(m)
        b, _, _, _ = w.step(m)
        np.testing.assert_array_equal(a["drives"], b["drives"])
        for i in range(4):
            if s.eyes[i] > 0:
                closed += 1
                assert (a["image"][i, :, :64] == 0).all()
            else:
                np.testing.assert_array_equal(a["image"][i, :, :64], b["image"][i])
    assert closed > 0


def test_the_text_says_what_is_nearest():
    s, _ = worlds()
    rng = np.random.default_rng(1)
    for _ in range(100):
        a, _, _, _ = s.step(moves(rng))
        d, _ = s.nearest_xy()
        for i in range(4):
            if s.quiet[i] > 0:
                assert (a["image"][i, :, 64:] == 0).all()
                continue
            k, side, far = s.shown[i]
            np.testing.assert_array_equal(a["image"][i, :, 64:, 0], s.table[k, side, far])
            dx, dy = d[i]
            want = (0 if dx < 0 else 1) if abs(dx) >= abs(dy) else (2 if dy < 0 else 3)
            assert side == want and far == int(np.hypot(dx, dy) > 16)
            kind = np.hypot(*((s.world.target[i] - s.world.cursor[i] + 32) % 64 - 32))
            assert (k == 0) == bool(abs(kind - np.hypot(dx, dy)) < 1e-9)


def test_drops_are_long_and_never_both():
    s, _ = worlds(n=8)
    rng = np.random.default_rng(2)
    runs_e, runs_q, cur_e, cur_q = [], [], np.zeros(8, int), np.zeros(8, int)
    for _ in range(3000):
        s.step(moves(rng, 8))
        assert not ((s.eyes > 0) & (s.quiet > 0)).any()
        for cur, arr, runs in ((cur_e, s.eyes, runs_e), (cur_q, s.quiet, runs_q)):
            ended = (cur > 0) & (arr == 0)
            runs += cur[ended].tolist()
            cur[:] = np.where(arr > 0, cur + 1, 0)
    assert len(runs_e) > 10 and len(runs_q) > 10
    assert min(runs_e + runs_q) >= 9 and max(runs_e + runs_q) <= 16          # the longest context is 8


def test_cut_turns_both_off():
    s, _ = worlds(dict(cut=True), n=8)
    rng = np.random.default_rng(3)
    seen = 0
    for _ in range(600):
        a, _, _, _ = s.step(moves(rng, 8))
        off = (s.eyes > 0) | (s.quiet > 0)
        seen += off.sum()
        assert (a["image"][off] == 0).all()
    assert seen > 0


def test_shuffled_words_mean_nothing():
    s, _ = worlds(dict(teacher="shuffle"), n=8)
    rng = np.random.default_rng(4)
    same = []
    for _ in range(1500):
        s.step(moves(rng, 8))
        same.append((s.shown == s.said).all(1).mean())
    assert 0.0 < np.mean(same) < 0.2                                          # 1 in 16 by chance


def test_teacher_off():
    s, _ = worlds(dict(teacher="off"))
    a, _, _, _ = s.step(moves(np.random.default_rng(5)))
    assert (a["image"][:, :, 64:] == 0).all()


def test_invalid():
    with pytest.raises(ValueError):
        make_config(dict(ENV, sense=dict(teacher="no")))
    with pytest.raises(ValueError):
        make_config(dict(ENV, sense=dict(drop_min=0)))


@pytest.mark.skipif(not cuda_lib.available(), reason="needs libvmcuda.so and a GPU")
def test_each_piece_is_read_from_its_own_slots():
    """Piece 1's patches follow the left half's summaries: piece 0's slots read its odd rows from the wrong tokens."""
    from types import SimpleNamespace

    import agent_lib
    from eval_sense import patch_slots
    c = json.load(open(os.path.join(DEV, "configs", "U_sense.json")))["model"]
    agents = [SimpleNamespace(config=dict(c, band_k=k)) for k in range(2)]
    pp = patch_slots(agents)
    assert (pp[0] != pp[1]).any()
    for a, p in zip(agents, pp):
        assert (agent_lib.uni_layout(a.config)["kind"][p + 1] == 1).all()


def test_a_lifetime_and_its_frozen_test(tmp_path):
    out = str(tmp_path / "x")
    cmd = [sys.executable, "lifetime.py", "--config", "configs/U_sense.json", "-o", out, "--set", "online.steps=40", "--set", "online.save_every=40",
           "--set", "online.log_every=20", "--set", "online.store_warmup=8", "--set", "env.sense.gap_min=2", "--set", "env.sense.gap_max=4"]
    r = subprocess.run(cmd, cwd=DEV, capture_output=True, text=True, timeout=280)
    assert r.returncode == 0, r.stderr[-800:]
    rows = [json.loads(line) for line in open(os.path.join(out, "metrics.jsonl"))]
    assert "food_blind" in rows[-1] and rows[-1]["sense_blind"] > 0
    e = subprocess.run([sys.executable, "eval_sense.py", out, "--ticks", "60", "--bodies", "2"], cwd=DEV, capture_output=True, text=True, timeout=280)
    assert e.returncode == 0, e.stderr[-800:]
    a, b = [json.loads(line) for line in e.stdout.strip().split("\n")]
    assert (a["cut"], b["cut"]) == (False, True) and a["n_view"] + a["n_text"] > 0

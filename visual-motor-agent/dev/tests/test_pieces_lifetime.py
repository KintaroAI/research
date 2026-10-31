"""Pieces of one body (task 00059, online.pieces with model.band_n): the body's move from the pieces' moves; short
lifetimes on the CPU and the GPU with two and four pieces: every piece saves its checkpoints, learns, and starts
from its own weights; invalid settings.

Requires libvmagent.so in the venv; the GPU runs libvmcuda.so and a GPU.
"""
import json
import os
import subprocess
import sys

import numpy as np
import pytest

DEV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, DEV)

import agent_lib  # noqa: E402
import cuda_lib  # noqa: E402
from lifetime import body_move, cosine, piece_seed, pieces_move  # noqa: E402

HAVE_GPU = cuda_lib.available()
TINY = ["--set", "env.size=32", "--set", "model.img=32", "--set", "model.d_model=16", "--set", "model.d_ff=32",
        "--set", "model.n_layer=1", "--set", "model.bins=8", "--set", "model.uni_pyr=2", "--set", "model.sum_thumb=1",
        "--set", "online.instinct_m=4", "--set", "online.steps=24", "--set", "online.save_every=24",
        "--set", "online.log_every=12", "--set", "online.store_warmup=8", "--set", "online.threads=2",
        "--set", "env.radius_min=4", "--set", "env.radius_max=6"]


A = np.array([[1.0, -1.0, 1.0], [0.5, 0.0, 0.0]], np.float32)
B = np.array([[-1.0, -1.0, 1.0], [0.5, 1.0, 1.0]], np.float32)
C = np.array([[0.0, 0.5, 0.0], [-0.75, 0.5, 1.0]], np.float32)


def test_body_move_mean():
    m, won = body_move([A, B, C])
    np.testing.assert_allclose(m[:, :2], [[0.0, -0.5], [0.0833333, 0.5]], atol=1e-6)
    assert m[:, 2].tolist() == [1.0, 1.0] and (won == -1).all()               # the click: a majority
    assert body_move([A])[0] is A


def test_body_move_max_per_axis():
    rng = np.random.default_rng(0)
    seen = set()
    for _ in range(50):
        m, won = body_move([A, B, C], "max", rng=rng)
        assert abs(m[0, 0]) == 1.0 and m[0, 1] == -1.0 and m[1, :2].tolist() == [-0.75, 1.0]
        assert won[1].tolist() == [2, 1] and won[0, 0] in (0, 1) and won[0, 1] in (0, 1)
        np.testing.assert_allclose(m[0, 0], [A, B][won[0, 0]][0, 0])
        seen.add(tuple(won[0]))
    assert len(seen) == 4                                                     # ties: at random


def test_body_move_random():
    rng = np.random.default_rng(0)
    seen = set()
    for _ in range(50):
        m, won = body_move([A, B, C], "random", rng=rng)
        assert won[0, 0] == won[0, 1] and won[1, 0] == won[1, 1]
        for b in range(2):
            np.testing.assert_allclose(m[b, :2], [A, B, C][won[b, 0]][b, :2])
        seen.add(won[0, 0])
    assert seen == {0, 1, 2}


class Scorer:
    """A piece for pieces_move: uni_candidates gives its fixed scores of the moves."""

    def __init__(self, scores):
        self.scores = np.array(scores, float)

    def uni_candidates(self, cand):
        assert cand.shape == self.scores.shape + (3,)
        return self.scores


def test_council_and_veto():
    props = [(A, 0, 0, np.zeros(2)), (B, 0, 0, np.zeros(2)), (C, 0, 0, np.zeros(2))]
    s = [Scorer([[0.1, 0.9, 0.3], [0.5, 0.2, 0.2]]), Scorer([[0.1, 0.1, 0.3], [0.5, 0.2, 0.2]]),
         Scorer([[0.7, 0.2, 0.3], [0.5, 0.2, 0.6]])]
    m, won = pieces_move(s, props, "council", np.random.default_rng(0))     # means [.3, .4, .3] and [.5, .2, .33]
    assert won[:, 0].tolist() == [0, 1]
    np.testing.assert_allclose(m[:, :2], [A[0, :2], B[1, :2]])
    m, won = pieces_move(s, props, "veto", np.random.default_rng(0))        # worst [.7, .9, .3] and [.5, .2, .6]
    assert won[:, 0].tolist() == [2, 1]
    np.testing.assert_allclose(m[:, :2], [C[0, :2], B[1, :2]])
    assert m[:, 2].tolist() == [1.0, 1.0]


def test_body_move_gain():
    h = np.array([[0.5, 0.2], [0.4, 0.3], [0.6, 0.1]])                       # [S, N] predicted next hunger
    m, won = body_move([A, B, C], "gain", h)
    np.testing.assert_allclose(m, [[-1.0, -1.0, 1.0], [-0.75, 0.5, 1.0]])   # B's whole move, then C's
    assert won.tolist() == [[1, 1], [2, 2]]
    with pytest.raises(ValueError):
        body_move([A, B, C], "gain", np.full((3, 2), np.nan))
    np.testing.assert_allclose(cosine(np.array([[1.0, 0.0], [0.0, 0.0]]), np.array([[2.0, 0.0], [1.0, 1.0]])),
                               [1.0, 0.0])


def run(tmp_path, dev, extra):
    out = str(tmp_path / ("run_" + dev))
    cmd = [sys.executable, "lifetime.py", "--config", "configs/U_touch_ego_c2.json", "--seed", "3", "-o", out,
           "--set", "online.device=" + dev] + TINY + extra
    r = subprocess.run(cmd, cwd=DEV, capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stderr[-3000:]
    return out


@pytest.mark.parametrize("dev", ["cpu", "cuda"])
@pytest.mark.parametrize("S,cols,rule", [(2, 1, "mean"), (4, 2, "mean"), (4, 2, "max"), (2, 1, "gain"), (4, 2, "random"),
                                         (4, 2, "council"), (2, 1, "veto")],
                         ids=["halves", "quadrants", "max", "gain", "random", "council", "veto"])
def test_pieces_learn_from_their_own_start(tmp_path, dev, S, cols, rule):
    if dev == "cuda" and not HAVE_GPU or dev == "cpu" and rule in ("council", "veto"):
        pytest.skip("needs libvmcuda.so and a GPU")
    out = run(tmp_path, dev, ["--set", "model.band_n=%d" % S, "--set", "model.band_cols=%d" % cols,
                              "--set", "online.pieces=1", "--set", "online.piece_rule=" + rule])
    assert json.load(open(os.path.join(out, "info.json")))["pieces"] == S
    p0 = []
    for k in range(S):
        a0 = agent_lib.Agent.load(os.path.join(out, "ckpt_00000000_p%d.bin" % k))
        a1 = agent_lib.Agent.load(os.path.join(out, "ckpt_00000024_p%d.bin" % k))
        assert a0.config["band_k"] == k and a0.config["band_n"] == S
        assert not np.array_equal(a0.params(), a1.params())                   # it learned
        fresh = agent_lib.Agent(a0.config, seed=piece_seed(3, k))             # from its own seed
        assert np.array_equal(fresh.params(), a0.params())
        p0.append(a0.params().copy())
    assert not np.array_equal(p0[0][:1000], p0[1][:1000])                    # not the same start
    rows = [json.loads(line) for line in open(os.path.join(out, "metrics.jsonl"))]
    assert all("agree_p%d" % k in rows[-1] and "prop_dx_p%d" % k in rows[-1] for k in range(S))
    won = np.array([[r["won_x_p%d" % k] for k in range(S)] for r in rows])  # each axis made by one piece, or none
    np.testing.assert_allclose(won.sum(1), 0.0 if rule == "mean" else 1.0, atol=1e-9)
    if dev == "cuda":
        assert all("uni_ce_patch_p%d" % k in rows[-1] for k in range(S))


@pytest.mark.skipif(not HAVE_GPU, reason="needs libvmcuda.so and a GPU")
@pytest.mark.parametrize("S,cols", [(4, 2), (0, 0)], ids=["quadrants", "one_model"])
def test_a_lifetime_on_the_ring(tmp_path, S, cols):
    extra = ["--set", "model.band_n=%d" % S, "--set", "model.band_cols=%d" % cols, "--set", "online.pieces=1",
             "--set", "online.piece_rule=gain"] if S else []
    out = run(tmp_path, "cuda", extra + ["--set", "model.age_key=1", "--set", "online.ring=1", "--set", "online.profile=1"])
    a = agent_lib.Agent.load(os.path.join(out, "ckpt_final_p0.bin" if S else "ckpt_final.bin"))
    assert a.config["age_key"] == 1
    rows = [json.loads(line) for line in open(os.path.join(out, "metrics.jsonl"))]
    assert rows[-1]["t_act"] > 0 and rows[-1]["t_update"] > 0


@pytest.mark.parametrize("extra", [[], ["--set", "model.universal=0"], ["--set", "online.sf_windows=2"],
                                   ["--set", "online.piece_rule=sum"], ["--set", "online.ring=1"],
                                   ["--set", "online.piece_rule=council", "--set", "online.device=cpu"],
                                   ["--set", "online.piece_rule=gain", "--set", "online.instinct_m=0"]],
                         ids=["no_band_n", "not_universal", "self_forcing", "rule", "ring_without_age_key", "council_on_cpu",
                              "gain_without_instinct"])
def test_invalid_pieces(tmp_path, extra):
    base = ["--set", "online.pieces=1"] + (["--set", "model.band_n=2", "--set", "model.band_cols=1"] if extra else [])
    cmd = [sys.executable, "lifetime.py", "--config", "configs/U_touch_ego_c2.json", "-o", str(tmp_path / "x")] + \
        TINY + base + extra
    r = subprocess.run(cmd, cwd=DEV, capture_output=True, text=True, timeout=300)
    assert r.returncode != 0 and "ValueError" in r.stderr

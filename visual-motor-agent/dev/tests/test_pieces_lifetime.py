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
from lifetime import cosine, mean_move, piece_seed  # noqa: E402

HAVE_GPU = cuda_lib.available()
TINY = ["--set", "env.size=32", "--set", "model.img=32", "--set", "model.d_model=16", "--set", "model.d_ff=32",
        "--set", "model.n_layer=1", "--set", "model.bins=8", "--set", "model.uni_pyr=2", "--set", "model.sum_thumb=1",
        "--set", "online.instinct_m=4", "--set", "online.steps=24", "--set", "online.save_every=24",
        "--set", "online.log_every=12", "--set", "online.store_warmup=8", "--set", "online.threads=2",
        "--set", "env.radius_min=4", "--set", "env.radius_max=6"]


def test_mean_move():
    a = np.array([[1.0, -1.0, 1.0], [0.5, 0.0, 0.0]], np.float32)
    b = np.array([[-1.0, -1.0, 1.0], [0.5, 1.0, 1.0]], np.float32)
    c = np.array([[0.0, 0.5, 0.0], [0.5, 0.5, 1.0]], np.float32)
    m = mean_move([a, b, c])
    np.testing.assert_allclose(m[:, :2], [[0.0, -0.5], [0.5, 0.5]], atol=1e-7)
    assert m[:, 2].tolist() == [1.0, 1.0]                                    # the click: a majority
    assert mean_move([a]) is a
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
@pytest.mark.parametrize("S,cols", [(2, 1), (4, 2)], ids=["halves", "quadrants"])
def test_pieces_learn_from_their_own_start(tmp_path, dev, S, cols):
    if dev == "cuda" and not HAVE_GPU:
        pytest.skip("needs libvmcuda.so and a GPU")
    out = run(tmp_path, dev, ["--set", "model.band_n=%d" % S, "--set", "model.band_cols=%d" % cols,
                              "--set", "online.pieces=1"])
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
    if dev == "cuda":
        assert all("uni_ce_patch_p%d" % k in rows[-1] for k in range(S))


@pytest.mark.parametrize("extra", [[], ["--set", "model.universal=0"], ["--set", "online.sf_windows=2"]],
                         ids=["no_band_n", "not_universal", "self_forcing"])
def test_invalid_pieces(tmp_path, extra):
    base = ["--set", "online.pieces=1"] + (["--set", "model.band_n=2", "--set", "model.band_cols=1"] if extra else [])
    cmd = [sys.executable, "lifetime.py", "--config", "configs/U_touch_ego_c2.json", "-o", str(tmp_path / "x")] + \
        TINY + base + extra
    r = subprocess.run(cmd, cwd=DEV, capture_output=True, text=True, timeout=300)
    assert r.returncode != 0 and "ValueError" in r.stderr

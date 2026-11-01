"""One table for all pieces (task 00076, online.tie_tables, vmc_tie): only the named ranges are copied or averaged; a
lifetime's pieces keep equal tables and different other weights; invalid settings.

Requires libvmagent.so; the GPU tests libvmcuda.so and a GPU.
"""
import os
import subprocess
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import agent_lib  # noqa: E402
import cuda_lib  # noqa: E402
from lifetime import TIED  # noqa: E402
from test_model_reference import SMALL  # noqa: E402
from test_pieces_lifetime import TINY  # noqa: E402

DEV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
gpu = pytest.mark.skipif(not cuda_lib.available(), reason="needs libvmcuda.so and a GPU")
PIECE = dict(SMALL, patch=8, universal=1, bins=8, gray=1, in_bins=1, age_key=1, uni_pyr=2, sum_thumb=1, band_n=4, band_cols=2)


def parts(n=3):
    agents = [agent_lib.Agent(dict(PIECE, band_k=k), seed=10 + k) for k in range(n)]
    return agents, [cuda_lib.CudaModel(a) for a in agents]


def ranges(agent, names):
    lay = {n: (o, s) for n, o, s in agent.param_layout()}
    return [lay[n] for n in names]


@gpu
@pytest.mark.parametrize("mean", [0, 1], ids=["copy", "mean"])
def test_tie_changes_only_the_range(mean):
    agents, models = parts()
    before = [m.params().copy() for m in models]
    (o, s), = ranges(agents[0], ["uni_in_w"])
    cuda_lib.tie(models, [(o, s)], mean)
    want = np.mean([b[o:o + s] for b in before], 0) if mean else before[0][o:o + s]
    for m, b in zip(models, before):
        p = m.params()
        np.testing.assert_allclose(p[o:o + s], want, atol=1e-5)
        np.testing.assert_array_equal(np.delete(p, np.arange(o, o + s)), np.delete(b, np.arange(o, o + s)))
    for m, a in zip(models, agents):
        m.close()
        a.close()


@gpu
def test_tie_refuses_a_wrong_range():
    agents, models = parts(2)
    with pytest.raises(RuntimeError):
        cuda_lib.tie(models, [(models[0].n_params - 1, 2)], 1)
    for m, a in zip(models, agents):
        m.close()
        a.close()


def lifetime(tmp_path, extra):
    out = str(tmp_path / "x")
    cmd = [sys.executable, "lifetime.py", "--config", "configs/U_default.json", "-o", out] + TINY + extra
    return out, subprocess.run(cmd, cwd=DEV, capture_output=True, text=True, timeout=300)


@gpu
@pytest.mark.parametrize("mode", ["in", "inout"])
def test_a_lifetime_keeps_the_tables_equal(tmp_path, mode):
    out, r = lifetime(tmp_path, ["--set", "online.tie_tables=" + mode])
    assert r.returncode == 0, r.stderr[-800:]
    a = [agent_lib.Agent.load(os.path.join(out, "ckpt_00000024_p%d.bin" % k)) for k in range(4)]
    a0 = [agent_lib.Agent.load(os.path.join(out, "ckpt_00000000_p%d.bin" % k)) for k in range(4)]
    tied = ranges(a[0], TIED[mode])
    for x in a[1:]:
        for o, s in tied:
            np.testing.assert_array_equal(x.params()[o:o + s], a[0].params()[o:o + s])
    for o, s in tied:
        assert not np.array_equal(a[0].params()[o:o + s], a0[0].params()[o:o + s])   # it learned
    other = ranges(a[0], ["uni_slot", "wpe"] + ([] if mode == "inout" else ["uni_out_w"]))
    for o, s in other:
        assert not np.array_equal(a[1].params()[o:o + s], a[0].params()[o:o + s])


@pytest.mark.parametrize("extra", [["--set", "online.tie_tables=all"], ["--set", "online.tie_tables=in", "--set", "online.device=cpu", "--set", "online.ring=0", "--set", "online.piece_rule=gain"],
                                   ["--set", "online.tie_tables=in", "--set", "online.pieces=0"]], ids=["unknown", "cpu", "one_model"])
def test_invalid(tmp_path, extra):
    _, r = lifetime(tmp_path, extra)
    assert r.returncode != 0 and "ValueError" in r.stderr

"""Self-forcing (task 00039): dreamed token inputs for the universal model's update (online.dream_inputs) and a short
lifetime with it. The GPU's loss with other inputs against the reference: tests/test_cuda_model.py."""
import json
import os
import subprocess
import sys

import numpy as np
import pytest

DEV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, DEV)

import agent_lib  # noqa: E402
import online  # noqa: E402
from test_lookahead import SMALL_UNI, windows  # noqa: E402

cuda_lib = pytest.importorskip("cuda_lib")
if not cuda_lib.available() or not cuda_lib.has("vmc_forward_uni_step"):
    pytest.skip("libvmcuda.so with vmc_forward_uni_step not available", allow_module_level=True)


@pytest.fixture(scope="module", params=[dict(in_bins=0), dict(in_bins=1), dict(in_bins=1, attn_win=1, uni_sum=2)],
                ids=["linear", "in_bins", "in_bins_hier"])
def model(request):
    agent = agent_lib.Agent(dict(SMALL_UNI, gray=1, **request.param), seed=5)
    m = cuda_lib.CudaModel(agent)
    yield m
    m.close()
    agent.close()


def test_real_steps_and_moves_untouched(model):
    c = model.config
    img, drv, act, es, _ = windows(c, 5, c["ctx"], 1, 2)
    act0 = act.copy()
    im, dv = online.dream_inputs(model, img, drv, act, es, [1, 3], [2, 5], J=2)
    np.testing.assert_array_equal(act, act0)
    for i in (0, 2, 4):                                            # not dreamed
        np.testing.assert_array_equal(im[i], img[i])
        np.testing.assert_array_equal(dv[i], drv[i])
    for i, k in ((1, 2), (3, 5)):
        T = c["ctx"]
        np.testing.assert_array_equal(im[i, :T - k], img[i, :T - k])
        np.testing.assert_array_equal(dv[i, :T - k], drv[i, :T - k])
        assert not np.array_equal(im[i, T - k:], img[i, T - k:]) and not np.allclose(dv[i, T - k:], drv[i, T - k:])
    with pytest.raises(ValueError):
        online.dream_inputs(model, img, drv, act, es, [0], [c["ctx"]])


def test_full_jacobi_is_sequential_decoding(model):
    """J = patches + 1: each dreamed frame equals decoding patch by patch; each body is the previous step's move
    token's pair-decoded prediction."""
    c = model.config
    P = (c["img"] // c["patch"]) ** 2
    lay = agent_lib.uni_layout(c)
    pp, mv = lay["patch"] - 1, lay["move"]                         # the patches' predictors, the move token
    T = c["ctx"]
    img, drv, act, es, _ = windows(c, 2, T, 1, 3)
    im, dv = online.dream_inputs(model, img, drv, act, es, [0, 1], [3, 1], J=P + 1)
    for i, k in ((0, 3), (1, 1)):
        wi, wd = img[i:i + 1].copy(), drv[i:i + 1].astype(np.float32)
        wa, we = act[i:i + 1], es[i:i + 1]
        for s in range(T - k, T):
            o = model.forward_uni_step(wi, wd, wa, we, s - 1, mode=2)
            wd[0, s] = np.clip(o["values_t"][0, mv, :2], 0, 1)
            wi[0, s] = wi[0, s - 1]
            for p in range(P):                                     # patch p given the decoded patches before it
                o = model.forward_uni_step(wi, wd, wa, we, s, mode=2)
                pv = online.uni_patches(c, wi[:, s])
                pv[0, p] = o["values_t"][0, pp[p]]
                wi[:, s] = online.uni_frame(c, pv)
        np.testing.assert_array_equal(im[i], wi[0])
        np.testing.assert_allclose(dv[i], wd[0], atol=1e-6)


def test_short_lifetime_with_self_forcing(tmp_path):
    """A short universal lifetime with self-forcing runs and reports the dreamed windows; the CPU refuses it."""
    if not cuda_lib.has("vmc_set_uni_inputs"):
        pytest.skip("libvmcuda.so without vmc_set_uni_inputs")
    out = str(tmp_path / "run")
    sets = ["model.img=32", "env.size=32", "model.d_model=16", "model.n_head=2", "model.d_ff=32", "model.bins=8",
            "online.steps=120", "online.log_every=60", "online.save_every=120", "online.stm=64",
            "online.replay=2000", "online.instinct_m=4", "online.sf_windows=3", "online.sf_steps=3"]
    cmd = [sys.executable, os.path.join(DEV, "lifetime.py"), "--config", os.path.join(DEV, "configs", "U_universal.json"),
           "--seed", "4", "-o", out] + sum([["--set", s] for s in sets], [])
    subprocess.run(cmd, check=True, cwd=DEV, capture_output=True)
    rows = [json.loads(line) for line in open(os.path.join(out, "metrics.jsonl"))]
    assert rows[-1]["sf_dreamed"] > 0 and all(r["uni_ce_patch"] > 0 for r in rows)
    bad = subprocess.run(cmd + ["--set", "online.device=cpu"], cwd=DEV, capture_output=True, text=True)
    assert bad.returncode != 0 and "self-forcing" in bad.stderr

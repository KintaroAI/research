"""Where information can flow in the universal model (task 00053): attention is causal, so it flows only forward.
Changing the last patch of a step leaves every earlier token unchanged; changing the first patch reaches the move's
predictor (slot nt - 2) of the same step, also through the pyramid's summaries with a local window; an episode start
cuts the window. The C++ forward, and the GPU's when present. And a GPU allocation that cannot succeed aborts the
process with a message instead of running on (or hanging).

Requires libvmagent.so; the GPU cases libvmcuda.so and a GPU.
"""
import os
import subprocess
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import agent_lib  # noqa: E402
from test_model_reference import SMALL  # noqa: E402

try:
    import cuda_lib
    HAVE_GPU = cuda_lib.available()
except Exception:   # no library or no device
    HAVE_GPU = False

UNI = dict(SMALL, patch=8, universal=1, bins=8, in_bins=1, gray=1)   # 32 x 32 frames: a 4 x 4 grid
CFGS = {"full": UNI, "pyr_win": dict(UNI, attn_win=1, uni_pyr=2), "pyr_win_thumb": dict(UNI, attn_win=1, uni_pyr=2,
                                                                                         sum_thumb=1)}
S, T = 2, 3


def batch(seed=0):
    rng = np.random.default_rng(seed)
    img = rng.integers(0, 256, (S, T, 32, 32, 3), dtype=np.uint8)
    return img, rng.random((S, T, 2)).astype(np.float32), rng.uniform(-1, 1, (S, T, 3)).astype(np.float32)


def models(cfg):
    agent = agent_lib.Agent(cfg, seed=5)
    out = [("cpu", agent)]
    if HAVE_GPU:
        out.append(("gpu", cuda_lib.CudaModel(agent)))
    return out


def values(model, img, drv, act, es):
    return model.forward_uni(img, drv, act, es, values=True, move_probs=False)["values"]   # [S, T, nt, V]


@pytest.mark.parametrize("name", list(CFGS))
def test_a_later_patch_does_not_reach_earlier_tokens(name):
    cfg = CFGS[name]
    lay = agent_lib.uni_layout(cfg)
    last = lay["patch"][-1]                       # the slot of patch (3, 3)
    img, drv, act = batch()
    es = np.zeros((S, T), np.uint8)
    img2 = img.copy()
    img2[:, 1, 24:32, 24:32] = 255 - img2[:, 1, 24:32, 24:32]   # step 1, the last patch
    for dev, m in models(cfg):
        a, b = values(m, img, drv, act, es), values(m, img2, drv, act, es)
        assert np.array_equal(a[:, 0], b[:, 0]), dev                   # the step before
        assert np.array_equal(a[:, 1, :last], b[:, 1, :last]), dev     # this step, every token before it
        assert np.abs(a[:, 1, lay["nt"] - 2] - b[:, 1, lay["nt"] - 2]).max() > 1e-6, dev   # the move's predictor
        assert np.abs(a[:, 2] - b[:, 2]).max() > 1e-6, dev             # and the next step


@pytest.mark.parametrize("name", list(CFGS))
def test_the_first_patch_reaches_the_move_predictor(name):
    cfg = CFGS[name]
    lay = agent_lib.uni_layout(cfg)
    img, drv, act = batch(1)
    es = np.zeros((S, T), np.uint8)
    img2 = img.copy()
    img2[:, 2, 0:8, 0:8] = 255 - img2[:, 2, 0:8, 0:8]           # the last step, patch (0, 0)
    for dev, m in models(cfg):
        a, b = values(m, img, drv, act, es), values(m, img2, drv, act, es)
        assert np.abs(a[:, 2, lay["nt"] - 2] - b[:, 2, lay["nt"] - 2]).max() > 1e-6, dev


@pytest.mark.parametrize("name", list(CFGS))
def test_an_episode_start_cuts_the_window(name):
    cfg = CFGS[name]
    img, drv, act = batch(2)
    es = np.zeros((S, T), np.uint8)
    es[:, 2] = 1
    img2, drv2, act2 = img.copy(), drv.copy(), act.copy()
    img2[:, :2] = 255 - img2[:, :2]
    drv2[:, :2] = 1 - drv2[:, :2]
    act2[:, :2] = -act2[:, :2]
    for dev, m in models(cfg):
        a, b = values(m, img, drv, act, es), values(m, img2, drv2, act2, es)
        assert np.array_equal(a[:, 2], b[:, 2]), dev
        assert np.abs(a[:, 1] - b[:, 1]).max() > 1e-6, dev


ABORT = """
import sys, numpy as np
sys.path.insert(0, %r)
import agent_lib, cuda_lib
cfg = dict(%r)
gpu = cuda_lib.CudaModel(agent_lib.Agent(cfg, seed=1))
n = 8000                                    # its two-hot inputs alone: 8000 * 8 * 66 * 2048 floats, 35 GB
img = np.zeros((n, 8, 64, 64, 3), np.uint8)
gpu.forward_uni(img, np.zeros((n, 8, 2), np.float32), np.zeros((n, 8, 3), np.float32), None, values=False)
print("returned")
"""


@pytest.mark.skipif(not HAVE_GPU, reason="needs libvmcuda.so and a GPU")
def test_an_impossible_allocation_aborts_with_a_message():
    if cuda_lib.device_free() < 4e9:   # it allocates ~2 GB before the buffer that cannot fit
        pytest.skip("the GPU is busy")
    cfg = dict(SMALL, img=64, patch=8, universal=1, bins=32, gray=1, in_bins=1, ctx=8)
    code = ABORT % (os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."), cfg)
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=300)
    assert r.returncode != 0
    assert "returned" not in r.stdout
    assert "out of device memory" in r.stderr and "aborting" in r.stderr   # not run on (the old builds did)

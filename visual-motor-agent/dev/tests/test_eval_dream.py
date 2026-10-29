"""eval_dream.dream (task 00039's dream measure): each dreamed step's body is the last move token's prediction and its
frame the fixed point of decoding, which equals decoding patch by patch. Checked with and without summary tokens
(task 00047), whose slots sit between the patches. Requires libvmcuda.so and a GPU."""
import os
import sys

import numpy as np
import pytest

DEV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, DEV)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import agent_lib  # noqa: E402
import online  # noqa: E402
from test_lookahead import SMALL_UNI, windows  # noqa: E402

cuda_lib = pytest.importorskip("cuda_lib")
if not cuda_lib.available() or not cuda_lib.has("vmc_forward_uni_step"):
    pytest.skip("libvmcuda.so with vmc_forward_uni_step not available", allow_module_level=True)
import eval_dream  # noqa: E402


@pytest.mark.parametrize("extra", [dict(), dict(uni_sum=2), dict(uni_sum=1, attn_win=1), dict(uni_pyr=2, attn_win=1),
                                   dict(uni_pyr=2, attn_win=1, sum_thumb=1)],
                         ids=["plain", "R2", "W1_R1", "W1_pyr2", "W1_pyr2_thumb"])
def test_dream_step_is_sequential_decoding(extra):
    agent = agent_lib.Agent(dict(SMALL_UNI, gray=1, in_bins=1, **extra), seed=5)
    m = cuda_lib.CudaModel(agent)
    c = m.config
    L = c["ctx"]
    img, drv, act, _, _ = windows(c, 2, L + 1, 1, 4)
    fd, bd = eval_dream.dream(m, img[:, :L], drv[:, :L], act, 1, "mean")
    lay = agent_lib.uni_layout(c)
    wi, wd, wa = img[:, :L].copy(), drv[:, :L].copy(), np.ascontiguousarray(act[:, :L])
    we = np.zeros((2, L), np.uint8)
    o = m.forward_uni_step(wi, wd, wa, we, L - 1)
    body = np.clip(o["values_t"][:, lay["move"], :2], 0, 1)                 # the move token: the next body
    wi = np.concatenate([wi[:, 1:], wi[:, -1:]], 1)
    wd = np.concatenate([wd[:, 1:], body[:, None]], 1)
    wa = np.ascontiguousarray(np.concatenate([wa[:, 1:], act[:, L:L + 1]], 1))
    for p, slot in enumerate(lay["patch"]):                                  # patch p from the token before its slot
        o = m.forward_uni_step(wi, wd, wa, we, L - 1)
        pv = online.uni_patches(c, wi[:, -1])
        pv[:, p] = o["values_t"][:, slot - 1]
        wi[:, -1] = online.uni_frame(c, pv)
    np.testing.assert_array_equal(fd[:, 0], wi[:, -1])
    np.testing.assert_allclose(bd[:, 0], body, atol=1e-6)
    m.close()
    agent.close()

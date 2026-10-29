"""Tokens from the summaries alone (task 00054, model.sum_only): the patches only make the thumbnails. The layout; the
text world at twice the scale is the normal one upscaled; with the same parameters, the normal text model on a frame
and the summary-only model on its 2x upscale compute the same thing (C++ and GPU); invalid configs; checkpoints.

Requires libvmagent.so; the GPU case libvmcuda.so and a GPU.
"""
import json
import os
import sys

import numpy as np
import pytest

DEV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, DEV)

import agent_lib  # noqa: E402
from reference import torch_reference as ref  # noqa: E402
from world import make_config, make_world  # noqa: E402

try:
    import cuda_lib
    HAVE_GPU = cuda_lib.available()
except Exception:
    HAVE_GPU = False


def config(name):
    with open(os.path.join(DEV, "configs", name)) as f:
        return json.load(f)


SMALL = dict(d_model=32, n_layer=2, n_head=4, d_ff=64, ctx=4)
A = {**agent_lib.DEFAULT_MODEL, **config("T_text_B.json")["model"], **SMALL}    # 16 x 16 page, letters 2 x 2
B = {**agent_lib.DEFAULT_MODEL, **config("T_text_B2.json")["model"], **SMALL}   # 32 x 32 page, letters 4 x 4


def test_layout_is_body_the_summaries_row_by_row_and_the_move():
    lay, t = agent_lib.uni_layout(B), ref.uni_slot_table(B)
    assert lay["nt"] == agent_lib.uni_layout(A)["nt"] == 66
    assert len(lay["patch"]) == 0 and len(lay["summary"]) == 64
    assert t["kind"][0] == 0 and t["kind"][-1] == 3 and (t["kind"][1:-1] == 2).all()
    assert (t["level"][1:-1] == 1).all()
    assert list(zip(t["r"][1:-1], t["c"][1:-1])) == [(r, c) for r in range(8) for c in range(8)]


def test_the_scaled_world_is_the_normal_one_upscaled():
    wa = make_world(3, make_config(config("T_text_B.json")["env"], auto_reset=True), seed=4)
    wb = make_world(3, make_config(config("T_text_B2.json")["env"], auto_reset=True), seed=4)
    for _ in range(3):
        fa, fb = wa.observe()["image"], wb.observe()["image"]
        assert np.array_equal(fb, fa.repeat(2, 1).repeat(2, 2))
        z = np.zeros((3, 3), np.float32)
        wa.step(z)
        wb.step(z)


def frames(seed=0, S=2, T=4):
    """Text-like frames: 8 x 8 letters of 2 x 2 pixels, grey levels of a 65-letter alphabet."""
    rng = np.random.default_rng(seed)
    lv = np.round(rng.integers(0, 65, (S, T, 8, 8)) * 255 / 64).astype(np.uint8)
    img = np.repeat(lv[..., None], 3, -1).repeat(2, 2).repeat(2, 3)
    return img, np.zeros((S, T, 2), np.float32), rng.uniform(-1, 1, (S, T, 3)).astype(np.float32)


@pytest.mark.parametrize("dev", ["cpu", "gpu"])
def test_same_parameters_same_outputs_on_the_upscaled_frame(dev):
    if dev == "gpu" and not HAVE_GPU:
        pytest.skip("needs libvmcuda.so and a GPU")
    a, b = agent_lib.Agent(A, seed=2), agent_lib.Agent(B, seed=9)
    assert a.num_params == b.num_params
    b.params()[:] = a.params()
    b.params_changed()
    ma, mb = (a, b) if dev == "cpu" else (cuda_lib.CudaModel(a), cuda_lib.CudaModel(b))
    img, drv, act = frames()
    es = np.zeros(img.shape[:2], np.uint8)
    oa = ma.forward_uni(img, drv, act, es, values=True, move_probs=True)
    ob = mb.forward_uni(img.repeat(2, 2).repeat(2, 3), drv, act, es, values=True, move_probs=True)
    np.testing.assert_allclose(ob["values"], oa["values"], atol=2e-5)
    np.testing.assert_allclose(ob["move_probs"], oa["move_probs"], atol=2e-5)


@pytest.mark.parametrize("bad", [dict(sum_thumb=0), dict(uni_pyr=0, sum_thumb=0), dict(attn_win=1),
                                 dict(sum_only=2)], ids=["no_thumbs", "no_pyramid", "window", "two"])
def test_invalid_configs(bad):
    with pytest.raises(ValueError):
        agent_lib.Agent(dict(B, **bad))


def test_checkpoint_keeps_the_flag(tmp_path):
    a = agent_lib.Agent(B, seed=1)
    a.save(str(tmp_path / "c.bin"))
    assert agent_lib.Agent.load(str(tmp_path / "c.bin")).config["sum_only"] == 1

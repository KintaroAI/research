"""Task 00091: delayed copies of the view. The composite's quarters are the frames 0, d1, d2, d3 ticks back, everything
else is the plain world's, the targets are the present tiled, and a short life learns on it."""
import json
import os
import subprocess
import sys

import numpy as np
import pytest

DEV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, DEV)

from online import delay_targets  # noqa: E402
from world import make_config, make_world  # noqa: E402

ENV = json.load(open(os.path.join(DEV, "configs", "U_default.json")))["env"]
DELAYS = [0, 1, 4, 16]


def moves(n, rng):
    return np.concatenate([rng.uniform(-1, 1, (n, 2)), np.zeros((n, 1))], 1).astype(np.float32)


def test_the_quarters_are_the_past_and_the_rest_is_the_world():
    plain = make_world(3, make_config(ENV, auto_reset=True), seed=5)
    late = make_world(3, make_config(dict(ENV, delays=DELAYS), auto_reset=True), seed=5)
    frames = [plain.observe()["image"]]
    comp = late.observe()["image"]
    H = frames[0].shape[1]
    assert comp.shape[1:3] == (2 * H, 2 * H)
    rng = np.random.default_rng(1)
    for t in range(1, 40):
        a = moves(3, rng)
        op, _, _, ip = plain.step(a)
        ol, _, _, il = late.step(a)
        frames.append(op["image"])
        quarters = [ol["image"][:, :H, :H], ol["image"][:, :H, H:], ol["image"][:, H:, :H], ol["image"][:, H:, H:]]
        for q, d in zip(quarters, DELAYS):
            np.testing.assert_array_equal(q, frames[max(t - d, 0)], err_msg="tick %d delay %d" % (t, d))
        np.testing.assert_array_equal(op["drives"], ol["drives"])
        np.testing.assert_array_equal(ip["success"], il["success"])
        np.testing.assert_array_equal(plain.cursor, late.cursor)
        np.testing.assert_array_equal(il["terminal_image"], ol["image"])


def test_the_targets_are_the_present_tiled():
    rng = np.random.default_rng(2)
    comp = rng.integers(0, 256, (2, 3, 8, 8, 3), dtype=np.uint8)
    t = delay_targets(comp)
    for a in (0, 4):
        for b in (0, 4):
            np.testing.assert_array_equal(t[..., a:a + 4, b:b + 4, :], comp[..., :4, :4, :])


@pytest.mark.parametrize("bad", [[1, 2, 3, 4], [0, 1, 2], [0, -1, 2, 3], [0, 1.5, 2, 3]])
def test_bad_delays_are_refused(bad):
    with pytest.raises(ValueError):
        make_world(1, make_config(dict(ENV, delays=bad)))


def test_a_life_on_delayed_copies(tmp_path):
    import cuda_lib
    if not cuda_lib.available():
        pytest.skip("needs libvmcuda.so and a GPU")
    out = str(tmp_path / "life")
    cmd = [sys.executable, "lifetime.py", "--config", "configs/U_default.json", "--seed", "3", "-o", out,
           "--set", "model.img=128", "--set", "model.uni_pyr=4", "--set", "model.band_n=16", "--set", "online.future_k=4",
           "--set", "env.delays=[0,1,4,16]", "--set", "online.steps=200", "--set", "online.log_every=100",
           "--set", "online.save_every=200"]
    r = subprocess.run(cmd, cwd=DEV, capture_output=True, text=True, timeout=900)
    assert r.returncode == 0, r.stderr[-3000:]
    rows = [json.loads(line) for line in open(os.path.join(out, "metrics.jsonl"))]
    assert all(rows[-1]["uni_ce_patch_p%d" % k] > 0 for k in range(16))
    r = subprocess.run(cmd + ["--set", "online.noise_in=0.1"], cwd=DEV, capture_output=True, text=True, timeout=300)
    assert r.returncode != 0 and "env.delays needs" in r.stderr

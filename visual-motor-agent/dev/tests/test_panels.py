"""Forecast panels (task 00115): the composite the body builds (world/panel_world.py), the lag, the targets (online.panel_targets),
memory's windows with the steps after them (GatedMemory(fut=4)), and a short GPU life."""
import json
import os
import subprocess
import sys

import numpy as np
import pytest

DEV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, DEV)

import cuda_lib  # noqa: E402
from online import DEFAULT_ONLINE, GatedMemory, panel_one, panel_targets  # noqa: E402
from world.panel_world import PanelWorld  # noqa: E402

HAVE_GPU = cuda_lib.available()
CFG = os.path.join(DEV, "configs", "U_view.json")


class Fake:
    """A world whose view at tick t is filled with t, and whose hunger is t / 100."""

    def __init__(self, n=2, h=4):
        self.n, self.h, self.t = n, h, 0

    def _obs(self):
        return dict(image=np.full((self.n, self.h, self.h, 3), self.t, np.uint8), drives=np.full((self.n, 2), self.t / 100, np.float32))

    def observe(self):
        return self._obs()

    def step(self, a):
        self.t += 1
        o = self._obs()
        return o, np.zeros(self.n), np.zeros(self.n, bool), dict(terminal_drives=o["drives"], hunger_cut=np.full(self.n, self.t, np.float32))


def test_composite_and_forecasts():
    w = PanelWorld(Fake(), lag=0)
    o = w.observe()
    assert o["image"].shape == (2, 8, 8, 3) and (o["image"][:, :4, :4] == 0).all()
    f = np.stack([np.full((2, 4, 4, 3), v, np.uint8) for v in (201, 202, 203)], 1)
    w.set_forecasts(f)
    o, _, _, inf = w.step(None)
    im = o["image"]
    assert (im[:, :4, :4] == 1).all() and (im[:, :4, 4:] == 201).all() and (im[:, 4:, :4] == 202).all() and (im[:, 4:, 4:] == 203).all()
    assert (inf["terminal_image"] == im).all()


def test_lag_delays_frame_and_body():
    w = PanelWorld(Fake(), lag=2)
    w.observe()
    seen = []
    for _ in range(5):
        o, _, _, inf = w.step(None)
        seen.append((int(o["image"][0, 0, 0, 0]), float(o["drives"][0, 0]), float(inf["terminal_drives"][0, 0]), float(inf["hunger_cut"][0])))
    # tick t (1..5) shows the world's frame of t - 2 (the first frame stands in before), its body too, and the cut of step t - 2
    assert [s[0] for s in seen] == [0, 0, 1, 2, 3]
    assert np.allclose([s[1] for s in seen], [0, 0, .01, .02, .03])
    assert [s[3] for s in seen] == [0, 0, 1, 2, 3]


def test_panel_targets():
    n, L, h = 2, 8, 4
    f = np.stack([np.full((n, h, h, 3), j, np.uint8) for j in range(L + 4)], 1)
    t = panel_targets(f, L)
    assert t.shape == (n, L, 2 * h, 2 * h, 3)
    for j in range(L):
        assert (t[:, j, :h, :h] == j + 1).all() and (t[:, j, :h, h:] == j + 2).all()
        assert (t[:, j, h:, :h] == j + 3).all() and (t[:, j, h:, h:] == j + 4).all()
    with pytest.raises(ValueError):
        panel_targets(f[:, :L + 3], L)


def test_memory_windows_carry_the_steps_after():
    on = dict(DEFAULT_ONLINE, stm=64, replay=64, relief_k=12, store_warmup=10 ** 6)
    L, img = 4, 8
    gm = GatedMemory(on, L, img, 0, 1, ch=1, next_frames=False, fut=4)
    assert gm.store.extra == 4 and "f_img" in gm.keys
    for t in range(40):
        fr = np.zeros((img, img, 3), np.uint8)
        fr[:img // 2, :img // 2] = t          # panel 1 holds the tick
        fr[img // 2:, img // 2:] = 255        # the rest is not panel 1
        gm.stm.add(img=fr, drv=np.zeros(2, np.float32), ndrv=np.zeros(2, np.float32), pa=np.zeros(3, np.float32), pr=0.0, es=0,
                   act=np.zeros(3, np.float32), rew=0.0, done=False, hit=False, cmd=0.0)
        gm.decide(t + 1)
    assert gm.store.n > 0
    for r in (gm.newest(4), gm.replayed(3, np.random.default_rng(0))):
        f = r["f_img"]
        assert f.shape[1:] == (L + 4, img // 2, img // 2, 3)
        p1 = panel_one(r["img"])[..., 0, 0, 0]
        assert (f[:, :L, 0, 0, 0] == p1).all()                                  # the window's own steps
        assert (np.diff(f[:, :, 0, 0, 0].astype(int), axis=1) == 1).all()       # and the 4 after, consecutive
    with pytest.raises(ValueError):
        gm.newest(3)


@pytest.mark.skipif(not HAVE_GPU, reason="needs libvmcuda.so and a GPU")
@pytest.mark.parametrize("extra", [[], ["online.panel_blank=1"], ["online.panel_lag=2", "online.panel_noise=0.1"]])
def test_a_short_life(tmp_path, extra):
    out = str(tmp_path / "run")
    sets = ["online.steps=60", "online.log_every=30", "online.save_every=60", "online.label=meal", "online.label_gamma=0.95",
            "model.body_label=1", "online.full_dose=1", "env.inertia=4", "online.label_refresh=1", "online.instinct_temp=2",
            "model.n_layer=2", "model.img=128", "online.panels=1", "online.replay=2000", "online.stm=256"] + extra
    env = dict(os.environ)
    env.pop("CUDA_MPS_PIPE_DIRECTORY", None)
    r = subprocess.run([sys.executable, "lifetime.py", "--config", CFG, "-o", out] + sum([["--set", s] for s in sets], []), cwd=DEV,
                       capture_output=True, text=True, env=env)
    assert r.returncode == 0, r.stderr[-3000:]
    rows = [json.loads(x) for x in open(os.path.join(out, "metrics.jsonl"))]
    assert rows[-1]["step"] == 60 and "uni_ce_patch_p3" in rows[-1]
    assert os.path.exists(os.path.join(out, "panels_0000001.png"))

"""Forecast panels (task 00115): the frame is a 2 x 2 composite of four panels the size of the world's view. Panel 1 (top
left) is the view as sensed, lag ticks late; panels 2, 3 and 4 (top right, bottom left, bottom right) are what the caller
set with set_forecasts before the step (the model's own forecasts, noised), noise until it first does. With a lag the
body is sensed as late as the frame: the drives, the terminal drives and the hunger the cap cut off are delayed with it.
Everything else (scoring, the world's state) is the wrapped world's own."""
import numpy as np


class PanelWorld:
    def __init__(self, world, lag=0, seed=0):
        self.world, self.lag = world, int(lag)
        self.rng = np.random.default_rng([seed, 0x9A4E])
        self.hist = []          # (image, drives) as the world reported them, newest last; lag + 1 of them
        self.cut_hist = []      # the hunger cut off, by tick, newest last
        self.fore = None        # [N, 3, h, w, 3] u8: panels 2..4

    def __getattr__(self, name):   # scoring reads the world's cursor and circles
        return getattr(self.world, name)

    def _blank(self, img):
        n, h, w = img.shape[:3]
        g = self.rng.integers(0, 256, (n, 3, h, w, 1), dtype=np.uint8)
        return np.repeat(g, 3, -1)

    def set_forecasts(self, frames):
        """frames [N, 3, h, w, 3] u8: next tick's panels 2, 3, 4 (None: noise, drawn anew each tick)."""
        self.fore = None if frames is None else np.ascontiguousarray(frames, np.uint8)

    def _push(self, img, drv):
        if not self.hist:
            self.hist = [(img, drv)] * (self.lag + 1)
        else:
            self.hist = self.hist[1:] + [(img, drv)]

    def compose(self, img):
        f = self.fore if self.fore is not None else self._blank(img)
        top = np.concatenate([img, f[:, 0]], 2)
        bottom = np.concatenate([f[:, 1], f[:, 2]], 2)
        return np.concatenate([top, bottom], 1)

    def observe(self):
        obs = self.world.observe()
        if not self.hist:
            self._push(obs["image"], obs["drives"])
        img, drv = self.hist[0]
        return dict(obs, image=self.compose(img), drives=drv)

    def step(self, actions):
        if not self.hist:
            self.observe()
        obs, rew, done, inf = self.world.step(actions)
        self._push(obs["image"], obs["drives"])
        cut = inf.get("hunger_cut")
        if cut is not None:
            self.cut_hist = (self.cut_hist or [np.zeros_like(cut)] * (self.lag + 1))[1:] + [cut]
        img, drv = self.hist[0]
        frame = self.compose(img)
        inf = dict(inf, terminal_image=frame, terminal_drives=drv)
        if cut is not None:
            inf["hunger_cut"] = self.cut_hist[0]
        return dict(obs, image=frame, drives=drv), rew, done, inf

"""Delayed copies of the view (task 00091): each frame is a 2 x 2 composite of the world's view now (top-left) and
delays[1], delays[2], delays[3] ticks ago (top-right, bottom-left, bottom-right). Before tick d the first frame stands
in. Everything but the image is the wrapped world's own; the frame is twice the world's size on each side."""
import numpy as np


class DelayWorld:
    def __init__(self, world, delays):
        self.world, self.delays = world, tuple(int(d) for d in delays)
        self.hist = []                                                    # the world's own frames, newest last

    def __getattr__(self, name):                                          # scoring reads the world's cursor and circles
        return getattr(self.world, name)

    def _push(self, frame):
        if not self.hist:
            self.hist = [frame] * (max(self.delays) + 1)
        else:
            self.hist = self.hist[1:] + [frame]

    def compose(self):
        """The composite [N, 2H, 2W, C] of the frames in hist."""
        f = [self.hist[-1 - d] for d in self.delays]
        return np.concatenate([np.concatenate(f[:2], 2), np.concatenate(f[2:], 2)], 1)

    def observe(self):
        obs = self.world.observe()
        if not self.hist:
            self._push(obs["image"])
        return dict(obs, image=self.compose())

    def step(self, actions):
        if not self.hist:
            self.observe()
        obs, rew, done, inf = self.world.step(actions)
        self._push(obs["image"])
        img = self.compose()
        inf = dict(inf, terminal_image=img)                               # the frame after the step, as the model sees it
        return dict(obs, image=img), rew, done, inf

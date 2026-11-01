"""CueWorld (task 00065): the delayed-cue world, a test a window of ticks cannot pass. A trial: a sample flashes, a block solid or
checker at the top left of the view; a delay with no circles; then one food and one poison on opposite sides of the cursor. The
food looks like the sample, the poison the other way, so the choice is a coin to anything that no longer holds the sample. A touch of
either, or a timeout, ends the trial. Food and poison act as in CircleWorld; only their drawing swaps. Nothing here is read by the
model other than as pixels.
"""
import dataclasses

import numpy as np

from .circle_world import CircleWorld, _checker, _rgb


@dataclasses.dataclass
class CueConfig:
    flash: int = 2                # ticks the sample shows
    delay_min: int = 20           # ticks without circles, uniform in min..max
    delay_max: int = 30
    timeout: int = 40             # ticks to touch one
    dist: float = 20.0            # px from the cursor to each circle's centre
    hold: bool = False            # the control: the sample stays on through delay and choice
    block: int = 16               # the sample's side in px, at the top left of the view

    def validate(self):
        if not (self.flash >= 1 and 0 <= self.delay_min <= self.delay_max and self.timeout >= 1 and self.block >= 1):
            raise ValueError("cue: flash and timeout >= 1, 0 <= delay_min <= delay_max, block >= 1")
        return self


class CueWorld(CircleWorld):
    """Per body: phase 0 the flash, 1 the delay, 2 the choice; left, its ticks left; sample, False solid or True checker."""

    def __init__(self, num_envs, config, seed=0):
        c = config
        if not (c.wrap and c.touch and c.respawn and c.gray and c.egocentric and c.hazards == 1 and c.hazard_pattern == "checker"
                and c.hazard_speed == 0 and c.target_speed == 0 and not c.no_overlap):
            raise ValueError("env.cue needs the egocentric gray torus with touch and respawn, one still checker hazard, "
                             "still food, and no no_overlap")
        if c.cue.dist <= max(c.radius_max, c.hazard_radius_max):
            raise ValueError("cue.dist must keep the cursor outside both circles")
        super().__init__(num_envs, config, seed=seed)
        self.cue = c.cue
        self.crng = np.random.default_rng([seed, 65])
        self.phase = np.zeros(num_envs, int)
        self.left = np.zeros(num_envs, int)
        self.sample = np.zeros(num_envs, bool)
        self.swap = self.sample                                   # a checker sample: the food drawn checker, the poison solid
        self.reset_stats()
        self._new_trial(np.arange(num_envs))

    def reset_stats(self):
        self.st = dict(trials=np.zeros(2), food=np.zeros(2), poison=np.zeros(2), timeout=np.zeros(2))

    def stats(self):
        """Since the last call: finished trials; of those with a touch the share of food, in all and by sample (None if none)."""
        s = self.st
        self.reset_stats()
        share = lambda f, p: float(f / (f + p)) if f + p else None   # noqa: E731
        return dict(cue_trials=int(s["trials"].sum()), cue_correct=share(s["food"].sum(), s["poison"].sum()),
                    cue_correct_solid=share(s["food"][0], s["poison"][0]),
                    cue_correct_checker=share(s["food"][1], s["poison"][1]),
                    cue_touch=float((s["food"] + s["poison"]).sum() / max(s["trials"].sum(), 1)))

    def _hide(self, idx):
        self.target_alive[idx] = False
        self.hz_rad[idx] = 0.0

    def _new_trial(self, idx):
        self.sample[idx] = self.crng.random(len(idx)) < 0.5
        self.phase[idx] = 0
        self.left[idx] = self.cue.flash
        self._hide(idx)

    def _show(self, idx):
        c, S = self.config, float(self.config.size)
        a = 2.0 * np.pi * self.crng.random(len(idx))
        v = self.cue.dist * np.stack([np.cos(a), np.sin(a)], 1)
        self.target[idx] = np.mod(self.cursor[idx] + v, S)
        self.hz_pos[idx, 0] = np.mod(self.cursor[idx] - v, S)
        self.radius[idx] = self.crng.uniform(c.radius_min, c.radius_max, len(idx))
        self.hz_rad[idx, 0] = self.crng.uniform(c.hazard_radius_min, c.hazard_radius_max, len(idx))
        self.target_alive[idx] = True
        self.phase[idx] = 2
        self.left[idx] = self.cue.timeout

    def _settle(self, success, poison):
        in_choice = self.phase == 2
        s = self.sample.astype(int)
        food, bad = in_choice & success, in_choice & poison
        self.left -= 1
        over = in_choice & (food | bad | (self.left <= 0))
        np.add.at(self.st["food"], s[food], 1)
        np.add.at(self.st["poison"], s[bad & ~food], 1)
        np.add.at(self.st["timeout"], s[over & ~food & ~bad], 1)
        np.add.at(self.st["trials"], s[over], 1)
        flashed = np.flatnonzero((self.phase == 0) & (self.left <= 0))
        self.phase[flashed] = 1
        self.left[flashed] = self.crng.integers(self.cue.delay_min, self.cue.delay_max + 1, len(flashed))
        waited = np.flatnonzero((self.phase == 1) & (self.left <= 0))
        if len(waited):
            self._show(waited)
        if over.any():
            self._new_trial(np.flatnonzero(over))

    def _render(self, idx=None):
        im = super()._render(idx)
        ids = np.arange(self.num_envs) if idx is None else np.asarray(idx)
        b, c = self.cue.block, self.config
        r, g, bl = _rgb(c.target_color)
        v = (299 * r + 587 * g + 114 * bl + 500) // 1000
        r, g, bl = _rgb(c.bg_color)
        bg = (299 * r + 587 * g + 114 * bl + 500) // 1000
        chk = _checker(c.size, c.hazard_checker_cell)[:b, :b]
        for k, i in enumerate(ids):
            if self.phase[i] == 0 or self.cue.hold:
                im[k, :b, :b] = np.where(chk | ~self.sample[i], v, bg)[..., None]
        return im

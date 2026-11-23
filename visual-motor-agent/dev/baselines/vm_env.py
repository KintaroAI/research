"""Task 00120: one body of our worlds as a plain RL environment, for the learned baselines (DreamerV3, PPO).

The same world, input and signal as the default agent's:
- the world from a config's `env` block through `world.make_world` (one body, auto-reset, the seed as our lives');
- the input: the frame's grey channel (64 x 64 x 1 uint8), the drives (2 values) and the previous executed move (dx, dy);
  no gt_* key and no world state;
- the action: the move (dx, dy) in [-1, 1]^2, click 0;
- the reward: the meal label's immediate part (`online.stored_labels`, full dose): +1 on a tick whose hunger fell,
  minus the hunger a poison added (a rise, with the cap's cut, above POISON_RISE), else 0;
- one life: no episode ever ends.

`Recorder` writes our metrics.jsonl rows every `every` ticks (step, step_hits, poison, hunger, reward, and the world's
stats, e.g. Breakout's bo_*), from the world's info, for scoring only.
"""
import json
import os

import numpy as np

POISON_RISE = 0.01   # online.POISON_RISE (task 00109); tested equal in tests/test_baselines_env.py


def load_env_block(path, sets=()):
    """The config's env block with `env.key=value` overrides (JSON values)."""
    cfg = json.load(open(path))
    env = dict(cfg["env"])
    for s in sets:
        k, v = s.split("=", 1)
        if not k.startswith("env."):
            raise ValueError("only env.* settings: %s" % s)
        try:
            env[k[4:]] = json.loads(v)
        except json.JSONDecodeError:
            env[k[4:]] = v
    return env


def label_reward(h0, h1, cut):
    """The meal label's immediate part (online.stored_labels with size 0 and the full dose)."""
    rise = float(h1) - float(h0)
    if rise < 0:
        return 1.0
    rise += float(cut)
    return -rise if rise > POISON_RISE else 0.0


class Recorder:
    """metrics.jsonl rows every `every` ticks, as lifetime.py writes them (the keys the analyses read)."""

    def __init__(self, out_dir, every=1000):
        self.out = out_dir
        self.every = every
        self.tick = 0
        self.win = dict(step_hits=0, poison=0, reward=0.0)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
            open(os.path.join(out_dir, "metrics.jsonl"), "w").close()

    def add(self, env, info, reward, hunger):
        self.tick += 1
        self.win["step_hits"] += int(info["success"][0])
        if "gt_hazard_click" in info:
            self.win["poison"] += int(info["gt_hazard_click"][0])
        self.win["reward"] += float(reward)
        if self.tick % self.every == 0:
            row = dict(step=self.tick, hunger=float(hunger), **self.win)
            if hasattr(env, "stats"):
                row.update(env.stats())
            if self.out:
                with open(os.path.join(self.out, "metrics.jsonl"), "a") as f:
                    f.write(json.dumps(row) + "\n")
            self.win = dict(step_hits=0, poison=0, reward=0.0)
            return row
        return None


class VMEnv:
    """One body of a world. obs: dict(image (64, 64, 1) uint8, drives (2,) float32, prev (2,) float32)."""

    def __init__(self, config, seed, sets=(), out_dir=None, every=1000):
        import sys
        dev = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if dev not in sys.path:
            sys.path.insert(0, dev)
        from world import make_config, make_world
        self.env = make_world(1, make_config(load_env_block(config, sets), auto_reset=True), seed=seed)
        self.rec = Recorder(out_dir, every)
        self.prev = np.zeros(2, np.float32)
        self.obs = None

    def _pack(self, o):
        return dict(image=np.ascontiguousarray(o["image"][0, :, :, :1]), drives=o["drives"][0].astype(np.float32),
                    prev=self.prev.copy())

    def reset(self):
        self.prev[:] = 0.0
        self.obs = self.env.reset()
        return self._pack(self.obs)

    def step(self, move):
        """move: (2,) floats; returns obs, reward, info (the world's, for scoring only), row (a metrics row or None)."""
        m = np.nan_to_num(np.asarray(move, np.float64).reshape(-1)[:2])
        a = np.zeros((1, 3), np.float32)
        a[0, :2] = np.clip(m, -1.0, 1.0)
        h0 = float(self.obs["drives"][0, 0])
        obs, _, done, info = self.env.step(a)
        h1 = float(info["terminal_drives"][0, 0]) if "terminal_drives" in info else float(obs["drives"][0, 0])
        cut = float(info["hunger_cut"][0]) if "hunger_cut" in info else 0.0
        r = label_reward(h0, h1, cut)
        self.prev = a[0, :2].copy()
        if bool(np.asarray(done).reshape(-1)[0]):   # never in our continual worlds (respawn); as lifetime.py
            self.prev[:] = 0.0
        self.obs = obs
        row = self.rec.add(self.env, info, r, obs["drives"][0, 0])
        return self._pack(obs), r, info, row


class VecVMEnv:
    """n bodies of one world (PPO's parallel form): obs arrays with a leading n, rewards (n,). Rows every `every` ticks of
    one body (every x n samples): step = samples so far, bodies = n, step_hits / poison / reward per body (summed over the
    bodies / n, so per 1k ticks of one body), hunger the bodies' mean, the world's cumulative stats summed over the bodies."""

    def __init__(self, config, seed, n, sets=(), out_dir=None, every=1000):
        import sys
        dev = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if dev not in sys.path:
            sys.path.insert(0, dev)
        from world import make_config, make_world
        self.n = n
        self.env = make_world(n, make_config(load_env_block(config, sets), auto_reset=True), seed=seed)
        self.prev = np.zeros((n, 2), np.float32)
        self.out, self.every, self.tick = out_dir, every, 0
        self.win = dict(step_hits=0, poison=0, reward=0.0)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
            open(os.path.join(out_dir, "metrics.jsonl"), "w").close()
        self.obs = None

    def _pack(self, o):
        return dict(image=np.ascontiguousarray(o["image"][:, :, :, :1]), drives=o["drives"].astype(np.float32),
                    prev=self.prev.copy())

    def reset(self):
        self.prev[:] = 0.0
        self.obs = self.env.reset()
        return self._pack(self.obs)

    def step(self, moves):
        a = np.zeros((self.n, 3), np.float32)
        a[:, :2] = np.clip(np.nan_to_num(np.asarray(moves, np.float64)[:, :2]), -1.0, 1.0)
        h0 = self.obs["drives"][:, 0].astype(np.float64)
        obs, _, done, info = self.env.step(a)
        h1 = info["terminal_drives"][:, 0] if "terminal_drives" in info else obs["drives"][:, 0]
        cut = info["hunger_cut"] if "hunger_cut" in info else np.zeros(self.n)
        r = np.array([label_reward(h0[i], h1[i], cut[i]) for i in range(self.n)], np.float32)
        self.prev = a[:, :2].copy()
        self.prev[np.asarray(done, bool).reshape(-1)] = 0.0
        self.obs = obs
        self.tick += 1
        self.win["step_hits"] += int(np.sum(info["success"]))
        if "gt_hazard_click" in info:
            self.win["poison"] += int(np.sum(info["gt_hazard_click"]))
        self.win["reward"] += float(r.sum())
        row = None
        if self.tick % self.every == 0:
            row = dict(step=self.tick * self.n, bodies=self.n, hunger=float(np.mean(obs["drives"][:, 0])),
                       step_hits=self.win["step_hits"] / self.n, poison=self.win["poison"] / self.n, reward=self.win["reward"] / self.n)
            if hasattr(self.env, "stats"):
                row.update(self.env.stats())
            if self.out:
                with open(os.path.join(self.out, "metrics.jsonl"), "a") as f:
                    f.write(json.dumps(row) + "\n")
            self.win = dict(step_hits=0, poison=0, reward=0.0)
        return self._pack(obs), r, info, row

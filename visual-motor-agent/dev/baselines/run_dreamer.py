"""Task 00120: DreamerV3 (the PyTorch port NM512/dreamerv3-torch, cloned to ~/src/dreamerv3-torch) in one body of our
worlds for one life, through baselines/vm_env.VMEnv (the default agent's input and the meal label's immediate part).

Run with ~/venvs/dreamer/bin/python (CUDA torch; gym is a shim over gymnasium). Writes our metrics.jsonl rows to --out,
Dreamer's own logs and checkpoint to --out/dreamer.

    ~/venvs/dreamer/bin/python baselines/run_dreamer.py --config configs/U_breakout.json --seed 1 --out DIR [--steps 60000]
        [--train-ratio 512] [--set env.key=value] [--dset key=value (a Dreamer setting, YAML value)]
"""
import argparse
import datetime
import os
import pathlib
import sys
import time
import uuid

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from vm_env import VMEnv  # noqa: E402


class DreamerVM:
    """VMEnv in the port's env API: dict obs with is_first / is_terminal, a Box action, an id; never done."""

    def __init__(self, config, seed, sets, out_dir):
        import gym
        self.env = VMEnv(config, seed, sets, out_dir)
        self.observation_space = gym.spaces.Dict(dict(
            image=gym.spaces.Box(0, 255, (64, 64, 1), dtype=np.uint8),
            drives=gym.spaces.Box(-np.inf, np.inf, (2,), dtype=np.float32),
            prev=gym.spaces.Box(-1.0, 1.0, (2,), dtype=np.float32)))
        self.action_space = gym.spaces.Box(-1.0, 1.0, (2,), dtype=np.float32)
        self.id = "%s-%s" % (datetime.datetime.now().strftime("%Y%m%dT%H%M%S"), uuid.uuid4().hex)
        self.t0 = time.time()

    def reset(self):
        o = self.env.reset()
        return dict(o, is_first=True, is_terminal=False)

    def step(self, action):
        if isinstance(action, dict):
            action = action["action"]
        o, r, info, row = self.env.step(action)
        if row is not None:
            print("tick %6d  %.1f tps  hits %d  poison %d  reward %.1f  hunger %.3f%s" % (
                row["step"], row["step"] / (time.time() - self.t0), row["step_hits"], row["poison"], row["reward"],
                row["hunger"], "  bo_hits %d bo_misses %d" % (row["bo_hits"], row["bo_misses"]) if "bo_hits" in row else ""),
                flush=True)
        return dict(o, is_first=False, is_terminal=False), r, False, {}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=60000)
    ap.add_argument("--train-ratio", type=float, default=512)
    ap.add_argument("--set", action="append", default=[])
    ap.add_argument("--dset", action="append", default=[])
    ap.add_argument("--dreamer", default=os.path.expanduser("~/src/dreamerv3-torch"))
    a = ap.parse_args()
    a.config = os.path.abspath(a.config)
    sys.path.insert(0, a.dreamer)
    from ruamel.yaml import YAML
    import dreamer
    import tools

    y = YAML(typ="safe")
    base = y.load((pathlib.Path(a.dreamer) / "configs.yaml").read_text())["defaults"]
    # the port runs chunks of eval_every steps after the prefill while its step < steps + eval_every: with eval_every equal
    # to the prefill (2500) and steps = budget - 2500 the life is exactly the budget (prefill included)
    pre = int(base["prefill"])
    over = dict(task="vm_world", steps=a.steps - pre, envs=1, action_repeat=1, time_limit=int(1e9), train_ratio=a.train_ratio,
                eval_episode_num=0, eval_every=pre, log_every=int(1e4), video_pred_log=False, grayscale=True,
                size=[64, 64], seed=a.seed, logdir=os.path.join(a.out, "dreamer"), compile=False,
                encoder=dict(base["encoder"], mlp_keys="drives|prev", cnn_keys="image"),
                decoder=dict(base["decoder"], mlp_keys="drives|prev", cnn_keys="image"))
    for s in a.dset:
        k, v = s.split("=", 1)
        over[k] = y.load(v)
    base.update(over)
    if a.steps % pre:
        raise ValueError("--steps must be a multiple of the prefill (%d)" % pre)
    ns = argparse.Namespace(**{k: tools.args_type(v)(v) for k, v in base.items()})
    os.makedirs(a.out, exist_ok=True)
    made = []

    def make_env(config, mode, id):
        env = DreamerVM(a.config, a.seed, a.set, a.out if mode == "train" else None)
        made.append(env)
        return env

    dreamer.make_env = make_env
    dreamer.main(ns)


if __name__ == "__main__":
    main()

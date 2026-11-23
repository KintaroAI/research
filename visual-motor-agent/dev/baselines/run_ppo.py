"""Task 00120: PPO (CleanRL-style, continuous actions, the Nature CNN) in our worlds, through baselines/vm_env.VecVMEnv:
the default agent's input (grey frame, drives, previous move) and the meal label's immediate part as the reward.

Run with ~/venvs/vm-gpu/bin/python (CUDA torch). --bodies 1 is our budget's form (PP), --bodies 16 the usual parallel
form (PL). Rows (metrics.jsonl) every 1k ticks of one body; `step` counts samples (ticks x bodies).

    ~/venvs/vm-gpu/bin/python baselines/run_ppo.py --config configs/U_breakout.json --seed 1 --out DIR
        [--bodies 1] [--steps 60000 (samples)] [--set env.key=value]
"""
import argparse
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from vm_env import VecVMEnv  # noqa: E402


def layer(m, std=np.sqrt(2), bias=0.0):
    nn.init.orthogonal_(m.weight, std)
    nn.init.constant_(m.bias, bias)
    return m


class Net(nn.Module):
    """The Nature CNN on the 64 x 64 grey frame; drives and previous move joined after it; a Gaussian policy (2 values,
    state-independent log std) and a value."""

    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(layer(nn.Conv2d(1, 32, 8, 4)), nn.ReLU(), layer(nn.Conv2d(32, 64, 4, 2)), nn.ReLU(),
                                 layer(nn.Conv2d(64, 64, 3, 1)), nn.ReLU(), nn.Flatten())
        self.body = nn.Sequential(layer(nn.Linear(64 * 4 * 4 + 4, 512)), nn.ReLU())
        self.mu = layer(nn.Linear(512, 2), 0.01)
        self.v = layer(nn.Linear(512, 1), 1.0)
        self.logstd = nn.Parameter(torch.zeros(2))

    def forward(self, img, vec):
        x = self.cnn(img.float().permute(0, 3, 1, 2) / 255.0)
        return self.body(torch.cat([x, vec], 1))

    def dist_value(self, img, vec):
        h = self(img, vec)
        mu = self.mu(h)
        return torch.distributions.Normal(mu, self.logstd.expand_as(mu).exp()), self.v(h).squeeze(-1)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--bodies", type=int, default=1)
    ap.add_argument("--steps", type=int, default=60000, help="samples: ticks x bodies")
    ap.add_argument("--rollout", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--minibatches", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2.5e-4)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--lam", type=float, default=0.95)
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--ent", type=float, default=0.0)
    ap.add_argument("--vf", type=float, default=0.5)
    ap.add_argument("--set", action="append", default=[])
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    n, T = a.bodies, a.rollout
    env = VecVMEnv(os.path.abspath(a.config), a.seed, n, a.set, a.out)
    dev = torch.device(a.device)
    net = Net().to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=a.lr, eps=1e-5)
    o = env.reset()
    buf = dict(img=torch.zeros((T, n, 64, 64, 1), dtype=torch.uint8, device=dev), vec=torch.zeros((T, n, 4), device=dev),
               act=torch.zeros((T, n, 2), device=dev), logp=torch.zeros((T, n), device=dev), rew=torch.zeros((T, n), device=dev),
               val=torch.zeros((T, n), device=dev))
    updates = a.steps // (T * n)
    t0 = time.time()
    for u in range(updates):
        for g in opt.param_groups:
            g["lr"] = a.lr * (1.0 - u / updates)
        for t in range(T):
            img = torch.as_tensor(o["image"], device=dev)
            vec = torch.as_tensor(np.concatenate([o["drives"], o["prev"]], 1), device=dev)
            with torch.no_grad():
                d, v = net.dist_value(img, vec)
                act = d.sample()
            buf["img"][t], buf["vec"][t], buf["act"][t] = img, vec, act
            buf["logp"][t], buf["val"][t] = d.log_prob(act).sum(-1), v
            o, r, _, row = env.step(act.cpu().numpy())
            buf["rew"][t] = torch.as_tensor(r, device=dev)
            if row is not None and n * env.tick % (10 * 1000 * n) == 0:
                print("samples %8d  %.0f sps  hits/1k %.1f  poison/1k %.1f  hunger %.3f%s" % (
                    row["step"], row["step"] / (time.time() - t0), row["step_hits"], row["poison"], row["hunger"],
                    "  bo_hits %d bo_misses %d" % (row["bo_hits"], row["bo_misses"]) if "bo_hits" in row else ""), flush=True)
        with torch.no_grad():   # one life, no episode end: bootstrap from the next state's value
            img = torch.as_tensor(o["image"], device=dev)
            vec = torch.as_tensor(np.concatenate([o["drives"], o["prev"]], 1), device=dev)
            _, nv = net.dist_value(img, vec)
            adv = torch.zeros((T, n), device=dev)
            last = torch.zeros(n, device=dev)
            for t in reversed(range(T)):
                nxt = nv if t == T - 1 else buf["val"][t + 1]
                delta = buf["rew"][t] + a.gamma * nxt - buf["val"][t]
                last = delta + a.gamma * a.lam * last
                adv[t] = last
            ret = adv + buf["val"]
        b = {k: v.reshape((T * n,) + v.shape[2:]) for k, v in buf.items()}
        badv, bret = adv.reshape(-1), ret.reshape(-1)
        mb = T * n // a.minibatches
        for _ in range(a.epochs):
            idx = torch.randperm(T * n, device=dev)
            for s in range(0, T * n, mb):
                j = idx[s:s + mb]
                d, v = net.dist_value(b["img"][j], b["vec"][j])
                logp = d.log_prob(b["act"][j]).sum(-1)
                ratio = (logp - b["logp"][j]).exp()
                ad = badv[j]
                ad = (ad - ad.mean()) / (ad.std() + 1e-8)
                pl = torch.max(-ad * ratio, -ad * ratio.clamp(1 - a.clip, 1 + a.clip)).mean()
                vl = 0.5 * ((v - bret[j]) ** 2).mean()
                loss = pl + a.vf * vl - a.ent * d.entropy().sum(-1).mean()
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), 0.5)
                opt.step()
    torch.save(net.state_dict(), os.path.join(a.out, "ppo_final.pt"))
    print("done %d samples in %.0f s" % (updates * T * n, time.time() - t0), flush=True)


if __name__ == "__main__":
    main()

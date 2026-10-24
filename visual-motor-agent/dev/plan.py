"""Act without reward: model-predictive control with the latent world model learned by babbling.

At every step the planner encodes the current frame into the latent s (the encoder's visual feature) and samples
`samples` action sequences of length `horizon`. It imagines their outcomes with the latent dynamics (transition +
hunger head, the predicted hunger fed back; vm_imagine) and scores each sequence by its summed predicted hunger.
The cross-entropy method refits the sampling distribution (Gaussian dx, dy; Bernoulli click) to the best
`elite` fraction for `iters` rounds, then executes the first action of the mean plan. The plan is shifted by
one step to warm-start the next decision.

Nothing here uses reward: no value function, no policy, no world state. The only goal is lower predicted hunger.

Episodes: the first --episodes of the training-time evaluation set (seed 777000, radius from the run; --env-seed
picks another set), hunger observed (env.drives=True), success = a click inside the circle within 200 steps.
median_steps counts a miss as 200 steps; median_steps_hit is over successful episodes only (the definition of
train.py's evaluation).

--explore-gain / --explore-eps: online.py's acting rule (vm-00005) with frozen weights. The agent babbles instead
(dx, dy ~ N(0, 0.607), click with probability 0.1) when the plan does not lower the imagined hunger below doing
nothing by at least explore_gain per step, or with probability explore_eps.

Usage:
    ./venv/bin/python plan.py RUN_DIR [--ckpt ckpt_final.bin] [--episodes 128] [-o result.json]
    ./venv/bin/python plan.py RUN_DIR --policy random      # same episodes, random actions (baseline)
    ./venv/bin/python plan.py RUN_DIR --ckpts 0 262144 1048576   # several checkpoints -> RUN_DIR/plan/ckpt_*.json
    ./venv/bin/python plan.py RUN_DIR --explore-gain 0.02 --explore-eps 0.05   # online.py's acting rule, frozen
"""
import argparse
import dataclasses
import json
import os
import subprocess
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import agent_lib  # noqa: E402
from record_episodes import env_config_from_info  # noqa: E402
from world import CircleWorld  # noqa: E402
from world.record import EpisodeRecorder  # noqa: E402

EVAL_SEED = 777_000          # train.py's evaluation set


def colored_noise(rng, beta, shape):
    """Standard normal samples whose power spectrum along the last axis falls as 1/f^beta (iCEM, Pinneri et al.
    2020). beta = 0 is white noise; beta = 2 gives persistent, random-walk-like sequences that travel further."""
    if beta == 0:
        return rng.standard_normal(shape)
    T = shape[-1]
    f = np.fft.rfftfreq(T)
    f[0] = f[1]                           # the mean gets the lowest frequency's power
    scale = f ** (-beta / 2)
    spec = (rng.standard_normal(shape[:-1] + f.shape) + 1j * rng.standard_normal(shape[:-1] + f.shape)) * scale
    w = np.full(f.size, 4.0)              # per-element variance of irfft: (s_0^2 + 4 sum s_k^2 [+ s_nyq^2]) / T^2
    w[0] = 1.0
    if T % 2 == 0:
        w[-1] = 1.0
    return np.fft.irfft(spec, n=T, axis=-1) * (T / np.sqrt((w * scale ** 2).sum()))


class CEMPlanner:
    """Cross-entropy-method MPC over n environments at once (one planning problem per environment)."""

    def __init__(self, agent, n, horizon=12, samples=256, iters=4, elite=0.1, sigma0=0.6, click0=0.2,
                 alpha=0.8, sigma_min=0.05, beta=0.0, seed=0, random_cost=False):
        self.agent, self.n, self.H, self.N, self.iters = agent, n, horizon, samples, iters
        self.beta = beta                      # noise color of the sampled moves (0 = white)
        self.K = max(2, int(round(elite * samples)))
        self.sigma0, self.click0, self.alpha, self.sigma_min = sigma0, click0, alpha, sigma_min
        self.rng = np.random.default_rng(seed)
        self.random_cost = random_cost        # control: same sampling and refitting, costs carry no information
        self.mu = np.zeros((n, horizon, 2))
        self.pi = np.full((n, horizon), click0)
        self.plan = np.zeros((n, horizon, 2))      # the mean moves of the latest plan (before the shift)
        self.plan_click = np.zeros((n, horizon))   # and its click probabilities

    def reset(self, mask):
        """Forget the warm-start plan of the envs in mask (e.g. after their episode ended)."""
        self.mu[mask] = 0.0
        self.pi[mask] = self.click0

    def act(self, images, hunger, active):
        """Plan for the active envs; returns actions [n, 3] (dx, dy, click 0/1) and the best plan's predicted
        hunger [n, H] (diagnostics)."""
        idx = np.flatnonzero(active)
        actions = np.zeros((self.n, 3), np.float32)
        best_h = np.full((self.n, self.H), np.nan)
        if idx.size == 0:
            return actions, best_h
        m, N, H, K, a = idx.size, self.N, self.H, self.K, self.alpha
        if not self.random_cost:
            s0 = self.agent.encode(images[idx])["vf"]
            S0 = np.repeat(s0, N, axis=0)
            H0 = np.repeat(hunger[idx].astype(np.float32), N)
        mu, pi = self.mu[idx], self.pi[idx]
        sig = np.full((m, H, 2), self.sigma0)
        for _ in range(self.iters):
            noise = colored_noise(self.rng, self.beta, (m, N, 2, H)).transpose(0, 1, 3, 2)
            dxy = np.clip(mu[:, None] + sig[:, None] * noise, -1, 1)
            clk = (self.rng.random((m, N, H)) < pi[:, None]).astype(np.float64)
            acts = np.concatenate([dxy, clk[..., None]], -1).reshape(m * N, H, 3).astype(np.float32)
            hun = (self.rng.random((m, N, H)) if self.random_cost else
                   self.agent.imagine(S0, H0, acts)["hunger"].reshape(m, N, H))
            cost = hun.sum(-1)
            el = np.argsort(cost, axis=1)[:, :K]
            e_dxy = np.take_along_axis(dxy, el[:, :, None, None], 1)
            e_clk = np.take_along_axis(clk, el[:, :, None], 1)
            mu = a * e_dxy.mean(1) + (1 - a) * mu
            sig = np.maximum(a * e_dxy.std(1) + (1 - a) * sig, self.sigma_min)
            pi = a * e_clk.mean(1) + (1 - a) * pi
            best_h[idx] = hun[np.arange(m), el[:, 0]]
        actions[idx, :2] = mu[:, 0]
        actions[idx, 2] = (pi[:, 0] > 0.5).astype(np.float32)
        self.plan[idx] = mu
        self.plan_click[idx] = pi
        self.mu[idx] = np.concatenate([mu[:, 1:], np.zeros((m, 1, 2))], 1)          # warm start: shift the plan
        self.pi[idx] = np.concatenate([pi[:, 1:], np.full((m, 1), self.click0)], 1)
        return actions, best_h


def explore_policy(planner, gain, eps, rng, stats, sigma=0.607, click=0.1):
    """online.py's acting rule with frozen weights: plan, but babble (dx, dy ~ N(0, sigma), click with probability
    `click`) when the plan's imagined hunger is not below doing nothing by `gain` per step (None: never), or with
    probability `eps`. Counts the planned steps in stats."""
    noop = np.zeros((1, planner.H, 3), np.float32)

    def policy(images, hunger, active):
        actions, _ = planner.act(images, hunger, active)
        idx = np.flatnonzero(active)
        m = idx.size
        if m == 0:
            return actions
        babble = rng.random(m) < eps
        if gain is not None:
            plan = np.concatenate([np.clip(planner.plan[idx], -1, 1), (planner.plan_click[idx] > 0.5)[..., None]],
                                  -1).astype(np.float32)
            s0 = planner.agent.encode(images[idx])["vf"]
            hp = planner.agent.imagine(np.concatenate([s0, s0]), np.tile(hunger[idx], 2).astype(np.float32),
                                       np.concatenate([plan, np.repeat(noop, m, 0)]))["hunger"]
            babble |= (hp[m:] - hp[:m]).mean(1) < gain          # doing nothing minus the plan, per step
        nb = int(babble.sum())
        if nb:
            actions[idx[babble]] = np.concatenate([rng.normal(0, sigma, (nb, 2)), rng.random((nb, 1)) < click],
                                                  1).astype(np.float32)
        stats["planned"] += m - nb
        stats["steps"] += m
        return actions
    return policy


def run_episodes(policy, cfg, n, seed=EVAL_SEED, record=False):
    """One episode in each of n envs (auto_reset off). policy(images, hunger, active) -> actions [n, 3]."""
    env = CircleWorld(n, cfg, seed=seed)
    obs = env.observe()
    recs = [EpisodeRecorder(cfg, i) for i in range(n)] if record else None
    if record:
        for r in recs:
            r.begin(env)
    active = np.ones(n, bool)
    length = np.zeros(n, int)
    success = np.zeros(n, bool)
    clicks = np.zeros(n, int)
    for _ in range(cfg.max_steps):
        a = policy(obs["image"], obs["drives"][:, 0], active)
        a = np.where(active[:, None], a, 0.0)
        obs, _, done, inf = env.step(a)
        if record:
            for i in np.flatnonzero(active):
                recs[i].record(a, np.zeros(n), done, inf)
        length += active
        clicks += active & inf["click"]
        success |= active & inf["success"]
        active &= ~done
        if not active.any():
            break
    out = dict(success=float(success.mean()), median_steps=float(np.median(np.where(success, length, cfg.max_steps))),
               median_steps_hit=float(np.median(length[success])) if success.any() else float("nan"),
               mean_steps=float(length.mean()), clicks_per_episode=float(clicks.mean()), episodes=int(n),
               steps_success=[int(x) for x in np.where(success, length, -1)])
    if record:
        out["recordings"] = recs
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir")
    ap.add_argument("--ckpt", default="ckpt_final.bin")
    ap.add_argument("--ckpts", type=int, nargs="*", help="evaluate these checkpoints (env steps); results go to "
                    "RUN_DIR/plan/ckpt_NNNNNNNN.json")
    ap.add_argument("--policy", choices=["cem", "random"], default="cem")
    ap.add_argument("--random-cost", action="store_true", help="control: the planner's costs are random numbers "
                    "(same sampling, refitting and smoothing; no model information)")
    ap.add_argument("--episodes", type=int, default=128)
    ap.add_argument("--horizon", type=int, default=12)
    ap.add_argument("--samples", type=int, default=256)
    ap.add_argument("--iters", type=int, default=4)
    ap.add_argument("--sigma0", type=float, default=0.6, help="initial std of the sampled moves")
    ap.add_argument("--beta", type=float, default=0.0, help="noise color of the sampled moves: 0 white, 2 persistent")
    ap.add_argument("--click-p", type=float, default=0.5, help="random policy: click probability")
    ap.add_argument("--seed", type=int, default=0, help="planner / random-policy sampling seed")
    ap.add_argument("--explore-gain", type=float, default=None, help="online.py's gate: babble when the plan "
                    "lowers the imagined hunger below doing nothing by less than this per step")
    ap.add_argument("--explore-eps", type=float, default=0.0, help="babble with this probability")
    ap.add_argument("--env-seed", type=int, default=EVAL_SEED, help="episodes from another seed")
    ap.add_argument("--threads", type=int, default=64)
    ap.add_argument("-o", "--out", default=None)
    args = ap.parse_args()
    agent_lib.set_threads(args.threads)
    with open(os.path.join(args.run_dir, "info.json")) as f:
        info = json.load(f)
    cfg = dataclasses.replace(env_config_from_info(info), bonuses=False, auto_reset=False, drives=True)
    if args.ckpts is not None:
        os.makedirs(os.path.join(args.run_dir, "plan"), exist_ok=True)
        for st in args.ckpts:
            ck = "ckpt_%08d.bin" % st
            sub = [sys.executable, os.path.abspath(__file__), args.run_dir, "--ckpt", ck, "--episodes", str(args.episodes),
                   "--horizon", str(args.horizon), "--samples", str(args.samples), "--iters", str(args.iters),
                   "--sigma0", str(args.sigma0), "--beta", str(args.beta), "--seed", str(args.seed), "--threads", str(args.threads),
                   "--explore-eps", str(args.explore_eps), "--env-seed", str(args.env_seed),
                   "-o", os.path.join(args.run_dir, "plan", "ckpt_%08d.json" % st)]
            if args.explore_gain is not None:
                sub += ["--explore-gain", str(args.explore_gain)]
            subprocess.run(sub, check=True)
        return
    t0 = time.time()
    stats = dict(planned=0, steps=0)
    if args.policy == "cem":
        agent = agent_lib.Agent.load(os.path.join(args.run_dir, args.ckpt))
        planner = CEMPlanner(agent, args.episodes, horizon=args.horizon, samples=args.samples, iters=args.iters,
                             sigma0=args.sigma0, beta=args.beta, seed=args.seed, random_cost=args.random_cost)
        if args.explore_gain is None and args.explore_eps == 0:
            policy = lambda img, h, act: planner.act(img, h, act)[0]  # noqa: E731
        else:
            policy = explore_policy(planner, args.explore_gain, args.explore_eps,
                                    np.random.default_rng(args.seed + 1), stats)
        res = run_episodes(policy, cfg, args.episodes, seed=args.env_seed)
    else:
        rng = np.random.default_rng(args.seed)

        def policy(img, h, act):
            n = img.shape[0]
            return np.concatenate([rng.normal(0, 0.607, (n, 2)), (rng.random((n, 1)) < args.click_p)], 1)
        res = run_episodes(policy, cfg, args.episodes, seed=args.env_seed)
    res.update(run=args.run_dir, ckpt=args.ckpt if args.policy == "cem" else None, policy=args.policy,
               random_cost=args.random_cost,
               horizon=args.horizon, samples=args.samples, iters=args.iters, sigma0=args.sigma0,
               beta=args.beta, explore_gain=args.explore_gain, explore_eps=args.explore_eps, env_seed=args.env_seed,
               planned=stats["planned"] / stats["steps"] if stats["steps"] else None,
               seconds=round(time.time() - t0, 1))
    print(json.dumps({k: v for k, v in res.items() if k != "steps_success"}))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()

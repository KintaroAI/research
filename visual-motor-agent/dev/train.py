"""PPO training loop: Python world (world/CircleWorld) + C++ agent (libvmagent.so via agent_lib).

The world is simulated here; every learning step (acting, rollout memory, GAE, PPO, prediction losses,
AdamW) happens in C++. The agent receives only rendered pixels, its two drive values and its own
previous action — never target/cursor coordinates (those appear in `info["gt_*"]` for evaluation only).

Usage:
    ./venv/bin/python train.py --config configs/A_baseline.json --seed 1 --total-steps 2000000
    ./venv/bin/python train.py --config configs/C_prediction.json --set ppo.pred_coef=0.5 -o /path/to/out
    ./venv/bin/python train.py --config configs/A_baseline.json --init /path/to/ckpt.bin   # transfer
    ./venv/bin/python train.py --config configs/P_dyn.json --init BABBLER/ckpt.bin --set train.actor=planner

train.actor=planner (reward-free runs only: ppo.use_ppo=0, env.drives=true): the CEM planner of plan.py, using the
current weights, picks every action instead of the policy; the model trains on that experience with the same
prediction losses. train.planner holds CEMPlanner settings.

Outputs (default ~/data/research/visual-motor-agent/exp_NNNNN/RRR_name_sS/): info.json, metrics.jsonl
(one row per iteration), eval.jsonl, ckpt_latest.bin, ckpt_final.bin.
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
from evaluate import executed_action, run_episodes  # noqa: E402
from output_name import next_output  # noqa: E402
from world import CircleWorld, EnvConfig  # noqa: E402

DEFAULT_TRAIN = dict(num_envs=128, rollout=128, total_steps=2_000_000, threads=0, lr_anneal=True,
                     eval_every=5, eval_episodes=256, eval_seed=777_000, save_every=25,
                     milestones=[0, 131072, 262144, 524288, 1048576, 2097152, 4194304],
                     curriculum=None, curriculum_threshold=0.9, click_bias=None, actor="policy", planner=None)


def set_path(cfg, dotted, value):
    keys = dotted.split(".")
    d = cfg
    for k in keys[:-1]:
        d = d.setdefault(k, {})
    try:
        value = json.loads(value)
    except (json.JSONDecodeError, TypeError):
        pass
    d[keys[-1]] = value


def load_config(path, overrides):
    with open(path) as f:
        cfg = json.load(f)
    for ov in overrides or []:
        k, v = ov.split("=", 1)
        set_path(cfg, k, v)
    train = dict(DEFAULT_TRAIN)
    train.update(cfg.get("train", {}))
    cfg["train"] = train
    cfg.setdefault("env", {})
    cfg.setdefault("model", {})
    cfg.setdefault("ppo", {})
    return cfg


def git_hash():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=os.path.dirname(__file__),
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


def summarize_episodes(eps):
    """eps: list of dicts with success, length, clicks, ret, bonus. Returns aggregate metrics."""
    if not eps:
        return {}
    succ = np.array([e["success"] for e in eps], bool)
    length = np.array([e["length"] for e in eps], float)
    clicks = np.array([e["clicks"] for e in eps], float)
    out = dict(train_episodes=len(eps), train_success=float(succ.mean()),
               train_len_mean=float(length.mean()),
               train_return=float(np.mean([e["ret"] for e in eps])),
               train_bonus=float(np.mean([e["bonus"] for e in eps])),
               train_clicks_per_ep=float(clicks.mean()))
    if succ.any():
        out["train_steps_to_success_median"] = float(np.median(length[succ]))
        out["train_clicks_per_success"] = float(clicks.sum() / succ.sum())
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", action="append", default=[], help="override, e.g. ppo.pred_coef=0.5")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--total-steps", type=int, default=None)
    ap.add_argument("--exp", type=int, default=1, help="experiment number for the default output dir")
    ap.add_argument("-o", "--out", default=None)
    ap.add_argument("--init", default=None, help="initialize weights from a checkpoint (optimizer state reset)")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config, args.set)
    tr = cfg["train"]
    if args.total_steps is not None:
        tr["total_steps"] = args.total_steps
    name = cfg.get("name", os.path.splitext(os.path.basename(args.config))[0])
    out_dir = args.out or next_output(args.exp, "%s_s%d" % (name, args.seed))
    os.makedirs(out_dir, exist_ok=True)
    if tr["threads"]:
        agent_lib.set_threads(tr["threads"])

    N, T = tr["num_envs"], tr["rollout"]
    env_cfg = EnvConfig(**cfg["env"])
    stages = tr["curriculum"] or []
    stage = 0
    if stages:
        env_cfg = dataclasses.replace(env_cfg, radius_min=stages[0][0], radius_max=stages[0][1])
    env = CircleWorld(N, env_cfg, seed=args.seed)

    if args.init:
        agent = agent_lib.Agent.load(args.init)
        # fresh optimizer state: reload params into a new model with the same config
        fresh = agent_lib.Agent(agent.config, seed=args.seed)
        fresh.params()[:] = agent.params()
        fresh.params_changed()
        agent.close()
        agent = fresh
    else:
        agent = agent_lib.Agent(cfg["model"], seed=args.seed)
        if tr["click_bias"] is not None:   # initial click-logit bias; with ppo.use_ppo=0 the policy stays frozen
            off = {name: o for name, o, _ in agent.param_layout()}["pi_b"]
            agent.params()[off + 2] = tr["click_bias"]
            agent.params_changed()
    ppo = dict(agent_lib.DEFAULT_PPO)
    ppo.update(cfg["ppo"])
    rscale = ppo["reward_scale"]
    planner = None
    if tr["actor"] == "planner":
        if ppo["use_ppo"] or not env_cfg.drives:
            raise ValueError("train.actor=planner needs ppo.use_ppo=0 and env.drives=true")
        from plan import CEMPlanner
        planner = CEMPlanner(agent, N, seed=args.seed, **(tr["planner"] or {}))
    elif tr["actor"] != "policy":
        raise ValueError("train.actor must be 'policy' or 'planner', got %r" % tr["actor"])

    info = dict(command=" ".join(sys.argv), git=git_hash(), config=cfg, seed=args.seed, out_dir=out_dir,
                model_config=agent.config, num_params=agent.num_params, threads=agent_lib.get_threads(),
                env_config=dataclasses.asdict(env_cfg), ppo=ppo, start_time=time.strftime("%Y-%m-%d %H:%M:%S"))
    with open(os.path.join(out_dir, "info.json"), "w") as f:
        json.dump(info, f, indent=2)
    if not args.quiet:
        print("out: %s\nparams: %d  threads: %d  envs: %d  rollout: %d  total: %d" % (
            out_dir, agent.num_params, agent_lib.get_threads(), N, T, tr["total_steps"]))

    ctx = agent.new_context(N, seed=args.seed * 1000 + 1)
    ro = agent.new_rollout(T, N)
    obs = env.reset()
    ep_start = np.ones(N, np.uint8)
    prev_a = np.zeros((N, 3), np.float32)
    prev_r = np.zeros(N, np.float32)
    n_iters = max(1, tr["total_steps"] // (N * T))
    mf = open(os.path.join(out_dir, "metrics.jsonl"), "w")
    ef = open(os.path.join(out_dir, "eval.jsonl"), "w")
    t_start = time.time()
    steps = 0
    first_success_step = None
    milestones = sorted(tr["milestones"] or [])
    if milestones and milestones[0] == 0:
        agent.save(os.path.join(out_dir, "ckpt_%08d.bin" % 0), with_optimizer=False)
        milestones.pop(0)

    def do_eval(it):
        ecfg = dataclasses.replace(env_cfg, bonuses=False)
        row = dict(iter=it, steps=steps, time=time.time() - t_start, stage=stage,
                   radius=[ecfg.radius_min, ecfg.radius_max])
        for det in (False, True):
            m = run_episodes(agent, ecfg, tr["eval_episodes"], seed=tr["eval_seed"], deterministic=det,
                             reward_scale=rscale)
            row.update({("det_" if det else "") + k: v for k, v in m.items()})
        ef.write(json.dumps(row) + "\n")
        ef.flush()
        return row

    for it in range(n_iters):
        lr = ppo["lr"] * (1.0 - it / n_iters) if tr["lr_anneal"] else ppo["lr"]
        t0 = time.time()
        eps = []
        t_env = 0.0
        for t in range(T):
            if planner is not None:
                a_plan = planner.act(obs["image"], obs["drives"][:, 0], np.ones(N, bool))[0]
            out = agent.act(ctx, obs["image"], obs["drives"], prev_a, prev_r, ep_start=ep_start, rollout=ro, t=t)
            a = out["action"]
            if planner is not None:     # the rollout keeps the tokens; the stored action becomes the planner's
                a = a_plan
                ro.set_action(t, a)
            te = time.time()
            obs, rew, done, inf = env.step(a)
            t_env += time.time() - te
            ro.outcome(t, rew, done, inf["terminal_image"], inf["terminal_drives"])
            first = inf["first"].astype(bool)
            if planner is not None:
                planner.reset(first)
            prev_a = executed_action(a)
            prev_a[first] = 0.0
            prev_r = (rew * rscale).astype(np.float32)
            prev_r[first] = 0.0
            ep_start = first.astype(np.uint8)
            if done.any():
                for i in np.nonzero(done)[0]:
                    eps.append(dict(success=bool(inf["ep_success"][i]), length=float(inf["ep_length"][i]),
                                    clicks=float(inf["ep_clicks"][i]), ret=float(inf["ep_return"][i]),
                                    bonus=float(inf["ep_bonus_total"][i])))
                    if first_success_step is None and inf["ep_success"][i]:
                        first_success_step = steps + t * N + int(i)
        steps += N * T
        last_value = agent.act(ctx, obs["image"], obs["drives"], prev_a, prev_r, ep_start=ep_start, peek=True)["value"]
        t1 = time.time()
        st = agent.update(ro, last_value, dict(ppo, lr=lr))
        t2 = time.time()
        row = dict(iter=it, steps=steps, time=t2 - t_start, lr=lr, fps=N * T / (t2 - t0),
                   t_rollout=t1 - t0, t_env=t_env, t_update=t2 - t1, first_success_step=first_success_step)
        row.update(summarize_episodes(eps))
        row.update({k: (float(v) if isinstance(v, float) else v) for k, v in st.items()})
        mf.write(json.dumps(row) + "\n")
        mf.flush()
        ev = None
        if tr["eval_every"] and (it % tr["eval_every"] == 0 or it == n_iters - 1):
            ev = do_eval(it)
            if stages and stage + 1 < len(stages) and ev["success"] >= tr["curriculum_threshold"]:
                stage += 1
                env.set_radius_range(*stages[stage])
                env_cfg = dataclasses.replace(env_cfg, radius_min=stages[stage][0], radius_max=stages[stage][1])
        if not args.quiet:
            msg = ("it %4d  steps %8d  fps %5.0f  succ %.3f  len %5.1f  ret %6.2f  ent %5.2f  kl %.4f  "
                   "clip %.3f  ev %.2f  pred %.5f  chg %.4f/%.4f  std %.2f  clk %.2f") % (
                it, steps, row["fps"], row.get("train_success", float("nan")), row.get("train_len_mean", float("nan")),
                row.get("train_return", float("nan")), st["entropy"], st["approx_kl"], st["clipfrac"],
                st["explained_var"], st["pred_mse"], st["pred_changed_mse"], st["copy_changed_mse"], st["std_x"],
                st["click_p"])
            if ev is not None:
                msg += "  | EVAL succ %.3f det %.3f med %.1f clk/s %.1f" % (
                    ev["success"], ev["det_success"], ev.get("steps_to_success_median", float("nan")),
                    ev.get("clicks_per_success", float("nan")))
            print(msg, flush=True)
        if tr["save_every"] and (it % tr["save_every"] == 0 or it == n_iters - 1):
            agent.save(os.path.join(out_dir, "ckpt_latest.bin"))
        while milestones and steps >= milestones[0]:
            agent.save(os.path.join(out_dir, "ckpt_%08d.bin" % milestones.pop(0)), with_optimizer=False)
    agent.save(os.path.join(out_dir, "ckpt_final.bin"))
    mf.close()
    ef.close()
    info["end_time"] = time.strftime("%Y-%m-%d %H:%M:%S")
    info["wall_time_s"] = time.time() - t_start
    info["first_success_step"] = first_success_step
    with open(os.path.join(out_dir, "info.json"), "w") as f:
        json.dump(info, f, indent=2)


if __name__ == "__main__":
    main()

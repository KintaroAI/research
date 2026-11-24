# Task 00120: Calibrate — Learned Baselines on the Same Pixels and Ticks (experiment vm-00108)

**Date:** 2026-11-23
**Status:** Complete (vm-00108, 2026-11-24). The owner, 2026-11-23: "yes write the task and queue it after 107", on the proposal: "Calibrate first. No learned baseline has run since the first few experiments; later worlds are scored against scripted walkers. Run the PPO agent already in the codebase and one off-the-shelf world-model agent such as DreamerV3 on the food world and Breakout, with the same pixels and tick budget. That shows whether the 'no reward' machinery buys sample efficiency or costs it."

## Context

- **Since vm-00001 every result is scored against scripted walkers** (random, still, tracker, aimer, blind walks) and against our own earlier arms. They give a floor and some ceilings, but not what an ordinary learner gets from the same experience. So .68 hit share in Breakout (vm-00103) or 72 food per 1k (vm-00101) can't be called fast or slow.
- **Our agent is not reward-free.** The default agent (`configs/U_agent.json`) learns a bootstrapped meal label (a meal 1, a poison its full dose, γ .95) as a body value of its one next-token predictor, and the instinct picks among sampled moves by that predicted label. The question is whether this architecture (one predictor, one life, one update a tick, the instinct) learns faster per tick than standard RL given **the same signal**.
- **Two references:**
  - **PPO:** the standard model-free learner. At our budget it is expected to be poor. That's a property of PPO, so it also runs long enough to reach our level: the ticks it needs is the sample-efficiency figure.
  - **DreamerV3:** the standard world-model learner, built for sample efficiency (it also learns a world model from pixels and acts in imagination).

## Fairness: the same world, input, signal and ticks

- **Worlds** (unchanged, through `world.make_world`):
  - **W1, Breakout stage 1:** `configs/U_breakout.json`, a paddle hit is a meal. Learning is unambiguous and quick here (vm-00103: random .24, ours .68 at 50–60k).
  - **W2, the food world:** `configs/U_agent.json`'s world (128 world, 64 view, egocentric, inertia 4). Ours: vm-00101's TT.
- **Input, as our model gets it:** the 64 × 64 grey frame (uint8), the drives (hunger, and the second drive value as the world gives it), and the previous move. No `gt_*` keys and no world state.
- **Action:** the move (dx, dy) in [−1, 1]², continuous; click 0 (the touch world and Breakout don't use it; Breakout ignores dy, as for our agent).
- **The reward is the label's immediate part, in the same units:**
  - +1 on a tick whose hunger fell (a meal);
  - minus the hunger a poison added (a rise above .01, `POISON_RISE`), with the full dose (`hunger_cut`, as `full_dose`);
  - 0 otherwise (no drift cost).
  - The baselines discount with their own defaults (Dreamer's γ .997, PPO's .99): that's their off-the-shelf form; our .95 is noted beside it.
- **One life, no episodes:** one world, one body, 60k ticks, no resets, as our lives. Dreamer is told no episode ever ends.
- **Seeds:** 6 per baseline and world, the world seeds 1–6 (the same worlds as our seeds 1–6).

## The arms (vm-00108)

| Arm | Learner | Ticks | Bodies |
|---|---|---|---|
| DR | DreamerV3, PyTorch port (`NM512/dreamerv3-torch`, commit 6ef8646), its default settings for 64 × 64 pixels (the drives and previous move as vector inputs, reconstructed too), train ratio 512 (its pixel-control setting; Atari 100k uses 1024, `TR=1024` in the Makefile) | 60k | 1 |
| PP | PPO (`baselines/run_ppo.py`, CleanRL-style: the Nature CNN, drives and previous move joined after it, a Gaussian policy; rollout 128, 4 epochs, 4 minibatches, lr 2.5e-4 annealed, γ .99, λ .95). `train.py`'s PPO is tied to `CircleWorld`'s episodes and its old CNN agent, so it isn't reused | 60k | 1 |
| PL | PPO, long: its usual parallel form | 2M | 16 |

- **JAX can't run here** (this CPU has no AVX), hence the PyTorch port. Its venv `~/venvs/dreamer` reuses `~/venvs/vm-gpu`'s CUDA torch (a .pth), adds ruamel.yaml, tensorboard and gymnasium, and a `gym` shim over gymnasium (the port imports `gym`; old gym breaks under numpy 2).
- **One life in the port:** it runs chunks of steps after a 2,500-tick random prefill (counted in the 60k); the life never ends (no `done`), so the replay holds one growing episode.
- **Compute is reported too:** wall-clock and GPU, since Dreamer spends far more compute per tick than our agent (about 2k replayed steps a tick against our 8 windows of 8).

## Measures (the same as our runs)

- **W1:** paddle hit share by 10k lived (and misses per 1k), against vm-00103's BD (12 seeds; paired by seed for seeds 1–6) and the random paddle.
- **W2:** food and poison per 1k by 10k lived, against vm-00101's TT (paired by seed, seeds 1–6) and the blind walks.
- **PL:** the ticks it needs to reach our 50–60k level (hit share, food per 1k), by seed.
- **Learning curves** in one figure per world: ours, DR, PP, PL (its first 60k on the same axis, and its whole run on its own).

## The readings, set before the results

1. **Plumbing:** the wrapper's tests pass (the reward equals the label's immediate part on recorded lives; the input carries no `gt_*` key); every life runs its ticks.
2. **Against PPO at our budget:** our 40–60k measure against PP's, paired by seed (expected: ours far ahead).
3. **Against Dreamer at our budget:** our 40–60k measure against DR's, paired by seed. Either direction is the answer.
4. **PPO's sample cost:** PL's ticks to our level (descriptive; "not reached in 2M" is a result).

## Out of scope here (follow-up)

- **The in-house ablation:** our same transformer with a standard actor-critic on top (PPO on its tokens in the CUDA path) instead of the label and the instinct. It isolates the action-selection machinery from the architecture, which Dreamer against ours does not (architecture, model size, replay ratio and acting rule all differ). A separate task once vm-00108 shows where we stand.

## Implementation

- [x] `dev/baselines/vm_env.py`: a single-body gym-style wrapper around `make_world` (obs: image, drives, previous move; the reward above; no episode end), and `tests/test_baselines_env.py` (4 tests: the reward equals `stored_labels`' immediate part on random lives in both worlds; the input keys; the rows; `VecVMEnv` for PPO's bodies).
- [x] DreamerV3: the port cloned to `~/src/dreamerv3-torch` at a pinned commit, its own venv `~/venvs/dreamer` (above); a small launcher `dev/baselines/run_dreamer.py` registering the wrapper as its env and writing our `metrics.jsonl` rows (every 1k ticks: `step`, `step_hits`, `poison`, `hunger`, and Breakout's `bo_*` counters).
- [x] PPO: `dev/baselines/run_ppo.py` (CleanRL-style, torch, the same metrics rows); 1 body for PP, 16 for PL.
- [x] Pilots under load (vm-00106's 24 frozen lives on both cards): PL on Breakout 930 samples/s, its hit share already above random at 160k samples; Dreamer about 0.8 updates/s, too loaded to time, so `make pilot` times it on free cards first.
- [x] `00108-learned-baselines/` (README, Makefile); queued after vm-00107.
- [x] The runs, analysis, figures, write-up (vm-00108).

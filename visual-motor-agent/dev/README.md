# visual-motor-agent / dev

A pixels-to-mouse agent. The **world** is Python/numpy (`world/`); the **model** is C++
(`src/vm_agent.cpp`, built as `libvmagent.so`). They meet in `agent_lib.py` (ctypes).

```
              Python                                       C++ (libvmagent.so)
 ┌───────────────────────────────┐   images u8, drives,   ┌──────────────────────────────────────┐
 │ world/CircleWorld (N envs)     │   prev action (+rew)   │ CNN encoder → token → causal          │
 │  render, move/click, hit test, │ ─────────────────────▶ │ Transformer (KV cache) → heads        │
 │  drives, bounded bonuses       │ ◀───────────────────── │ sample (dx,dy) ~ N, click ~ Bern      │
 │ train.py: loop, logging, eval  │   actions, logp, V     │ rollout memory → GAE → PPO + pred.    │
 └───────────────────────────────┘   reward, done, next    │ losses → manual backward → AdamW      │
                                     frame (terminal)       └──────────────────────────────────────┘
```

The model never sees ground truth: no cursor/target coordinates, distances, DOM or labels.
`info["gt_*"]` and `CircleWorld.get_state()` exist for scoring only.

## Setup

```bash
make setup        # venv: numpy, pillow, matplotlib, pytest + CPU torch (reference tests only)
make              # build libvmagent.so
make test         # C++ unit tests
make pytest       # Python tests: world, rewards, leaks, JS equivalence, PyTorch gradient equivalence
make bench        # throughput (acting + PPO update)
```

## Model (default `DEFAULT_MODEL` in agent_lib.py, ~1.28M parameters)

| Part | Configuration |
|---|---|
| Encoder | 64×64×3 (+2 coordinate channels if `coord=1`) → conv k4s2 16 → 32 → 32 (ReLU) → FC 128 (ReLU) |
| Token | [visual 128, drives 2, previous executed action 3, previous reward 1 (zeroed unless `use_prev_reward`)] → linear 128 + learned position |
| Transformer | 2 pre-LN blocks, d=128, 4 heads, FF 512 (GELU), causal and segment-masked; context 32 |
| Policy | mean (dx, dy) = tanh(W z), state-independent log σ (init −0.5), click logit; samples are stored unclipped and the world clips |
| Value | linear on z |
| Prediction | [z_t, a_t] → MLP 256 → 32×8×8 (+ tiled latent + coordinates if `coord=1`) → deconv 16×16×16 → deconv 3×32×32. `pred_residual=1` predicts the change (next = current + output); otherwise sigmoid pixels. With `pred_skip=1`, the encoder's conv-3 map (8×8) and tiled action enter deconv 1, and the conv-2 map (16×16) and tiled action enter deconv 2. Next drives come from the MLP layer. |
| Inverse dynamics | `inv_hidden>0`: MLP([v_t, v_{t+1}]) → executed (dx, dy) on consecutive visual features of one episode (`ppo.inv_coef`) |

**Context.** A context starts at every episode start and every 32 steps. A training sequence is
exactly one such block, so the KV-cached acting path and the batched training path compute
bitwise-identical outputs.

**Losses.**
- PPO clipped surrogate with normalized advantages.
- 0.5·vf·(V−R)².
- −ent_coef·H, where H is the Gaussian plus Bernoulli entropy.
- `pred_coef` · weighted MSE of the next frame. The weight is 1 + `pred_changed_weight` on pixels
  that differ between the two consecutive rendered frames.
- `drive_coef` · MSE of the next drives.

The prediction gradient into the shared trunk is scaled by `pred_backbone_scale`. Prediction
accuracy is never rewarded.

**Optimizer.** AdamW (β=0.9/0.999, ε=1e-5), global grad-norm clip 0.5, linear LR anneal.

## Training

```bash
./venv/bin/python train.py --config configs/A_baseline.json --seed 1 --total-steps 1000000 --set train.threads=64
./venv/bin/python run_queue.py queue.txt --exp 1 --slots 2     # job queue, 2 x 64 threads
```

A queue file has one job per line, `<run_name> <train.py args...>`, with `#` comments. Queue files
are local and gitignored. The experiment Makefiles record the exact commands.

Configs: `configs/A..F_*.json` follow the plan's matrix (§9). `*_w10` uses the changed-pixel
weighted prediction loss, `*_small` uses radius 4–8 targets, and `P_*` are reward-free babbling
runs (`ppo.use_ppo=0`: the policy stays at its random initialization).

Per-run outputs go to `~/data/research/visual-motor-agent/exp_NNNNN/RRR_name/`:
- `info.json`
- `metrics.jsonl` (per iteration: episode stats, PPO stats, prediction MSE against copy-last-frame)
- `eval.jsonl` (fixed eval seeds, bonuses off, stochastic + deterministic)
- `ckpt_latest.bin`, `ckpt_final.bin`, and milestone checkpoints `ckpt_000NNNNN.bin`

## Analysis

```bash
./venv/bin/python evaluate.py RUN_DIR                    # task metrics on fresh seeds
./venv/bin/python analyze_dynamics.py RUN_DIR --ckpt ckpt_00524288.bin
./venv/bin/python analyze_dynamics.py --untrained        # random-init control
./venv/bin/python plot_runs.py RUN_A RUN_C -o curves.png --labels A C
```

`dynamics_over_training.py RUN` runs this on every milestone checkpoint;
`compare_dynamics.py RUN1 RUN2 ... -o fig.png` overlays runs. `record_episodes.py RUN` writes GIFs,
browser-replay JSON and `[current | predicted | actual]` imagination strips.

`make_video.py --run RUN --run2 RUN_FROM_BABBLER --babbler BABBLER -o DIR` writes MP4s and
screenshot sheets of the agent at every numbered checkpoint. Each checkpoint replays the same start
states, with the cursor path and clicks drawn on top. For a fine-grained time-lapse, train with
dense checkpoints, e.g. `--set 'train.milestones=[0,16384,32768,...]'`. One PPO iteration is
16,384 steps, so the policy changes no more often than that.

`analyze_dynamics.py` asks whether the model knows what its actions do. It restores the world to
the same state and compares predicted and true outcomes of counterfactual actions:
- action retrieval accuracy (chance 1/9)
- cursor displacement read off the predicted frames
- circle-vanishes-on-click with the cursor inside vs outside
- predicted hunger relief
- linear probes of cursor and target position from z

## Performance (this machine: 128 vCPUs, SSE4.2, no GPU)

| | 64 threads |
|---|---|
| act, 128 envs | ~3.7 ms/step |
| CircleWorld.step, 128 envs | ~1.4 ms |
| PPO update, 4 epochs × 16k transitions | ~5.5 s |
| end to end | ~1.7–2.0k env-steps/s per run (two runs in parallel) |

## Files

| File | Role |
|---|---|
| `src/vm_agent.cpp`, `src/vm_agent.h` | model, acting, PPO, AdamW, checkpoints, C API |
| `src/test_vm_agent.cpp`, `src/bench_vm_agent.cpp` | C++ tests / benchmark (`#define TESTING` + include) |
| `agent_lib.py` | ctypes wrapper |
| `world/` | CircleWorld, recorder, browser demo |
| `train.py`, `run_queue.py`, `output_name.py` | training |
| `evaluate.py`, `analyze_dynamics.py`, `dynamics_over_training.py`, `compare_dynamics.py`, `plot_runs.py`, `record_episodes.py`, `make_video.py` | evaluation and analysis |
| `reference/torch_reference.py`, `tests/` | PyTorch reference and tests |

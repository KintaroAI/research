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
screenshot sheets of the agent at every numbered checkpoint. Each view plays continuously, as in
training: after every hit or timeout, a new random circle and cursor appear. The cursor path and
clicks are drawn on top. Charts come from a fixed set of 128 test episodes per checkpoint. For a fine-grained time-lapse, train with
dense checkpoints, e.g. `--set 'train.milestones=[0,16384,32768,...]'`. One PPO iteration is
16,384 steps, so the policy changes no more often than that.

`analyze_dynamics.py` asks whether the model knows what its actions do. It restores the world to
the same state and compares predicted and true outcomes of counterfactual actions:
- action retrieval accuracy (chance 1/9)
- cursor displacement read off the predicted frames
- circle-vanishes-on-click with the cursor inside vs outside
- predicted hunger relief
- linear probes of cursor and target position from z

## Latent dynamics and planning (vm-00002)

`model.dyn_hidden > 0` adds a world model on the encoder's visual feature v (128-d):
- a transition `v' = v + MLP([v, a])`;
- a hunger head `h' = sigmoid(MLP([v, h, a]))`;
- a coarse decoder `MLP(v) -> (pred_img/2)^2 x 3` frame.

It is trained during babbling on chains of up to `ppo.dyn_horizon` imagined steps. The losses
(`ppo.dyn_coef`, `dyn_h_coef`, `dyn_x_coef`) compare the chains with the encoder's latents of the
real future frames, the real next hunger and the real next frames. `ppo.dyn_h_horizon` limits which chain steps train
the hunger head (0 = all; 1 = only the first step, from the real latent). Chains stop at a terminal
step, so the imagined hunger *after* a successful click is never trained. After a failed click the
episode goes on, so that part is learned.

The heads never change the encoder (v is detached). With gradients into the encoder, the features
collapsed: 32 of 128 dims active after 65k steps, and the cursor was no longer decodable.

`train.click_bias` sets the initial click-logit bias of a fresh agent. During babbling the policy
stays frozen, so −2.197 means click probability 0.1.

```bash
./venv/bin/python train.py --config configs/P_dyn.json --exp 2 --set train.threads=64
./venv/bin/python plan.py RUN_DIR --ckpt ckpt_01048576.bin            # CEM planner on 128 test episodes
./venv/bin/python plan.py RUN_DIR --ckpts 0 262144 1048576            # -> RUN_DIR/plan/ckpt_*.json
./venv/bin/python plan.py RUN_DIR --policy random --click-p 0.1       # baseline on the same episodes
./venv/bin/python analyze_imagination.py RUN_DIR --ckpt ckpt_01048576.bin   # imagined vs real rollouts
```

`plan.py` acts without reward: at every step it imagines sampled action sequences with the latent
model and picks the one with the lowest summed predicted hunger (cross-entropy method). `--beta`
colors the sampling noise (iCEM); `--random-cost` is the control with uninformative costs.
`make_video.py --planner RUN` adds part 4, the planner at each checkpoint with its planned moves
drawn.

`analyze_imagination.py` reads the imagined cursor by retrieval: the nearest latent among frames of
the same circle with the cursor on a 2-px grid. A linear probe misreads imagined latents that leave
the manifold of real ones. It also runs four tests:
- **approach:** a straight path to the circle, then a click;
- **edge:** imagined relief of one click at a signed distance from the circle's edge;
- **straight paths:** imagined vs true displacement;
- **over the circle:** one imagined move with the cursor on the background vs over the circle.

`analyze_planner.py RUN --ckpt CK` traces the planner on the test episodes: success by starting
distance, and per failed episode the closest approach, clicks outside the circle and the relief its
plan expected.

**Learning from its own plans (vm-00004).** With `train.actor=planner`, the planner picks every
action of a reward-free run (`ppo.use_ppo=0`, drives on), using the current weights. The model
trains on that experience with the usual prediction losses; `vm_rollout_set_action` stores the
planner's action in place of the policy's. Run it from a babbler with `--init`.

```bash
./venv/bin/python train.py --config configs/P_dyn.json --init BABBLER/ckpt_02097152.bin --exp 4 \
    --set train.actor=planner --set train.eval_every=0
```

## One lifetime: a single world, learning on every step (vm-00005)

`online.py` runs one agent in one world and does a gradient step after every step:
- **What it learns from:** its newest 8 steps, plus `online.batch − 1` sequences replayed from a
  memory of its own past (`online.replay` steps).
- **How it acts:** the CEM planner, with babbling with probability `online.explore_eps`.
  `explore_eps=1` gives a pure online babbler.
- **What it never uses:** reward. The losses are the reward-free ones (`ppo.use_ppo=0`).
- **One frame:** `model.one_frame=1` makes every token attend only to itself, so each prediction uses
  only the current frame. Training sequences stay `ctx` long, for the inverse-dynamics and
  latent-chain targets.
- **Training batches:** `vm_rollout_store` fills a training rollout from stored experience.

```bash
./venv/bin/python online.py --config configs/L_online.json --seed 1 --exp 5
./venv/bin/python online.py --config configs/L_online.json --set online.batch=1 --set online.replay=8  # no memory
./venv/bin/python plan.py RUN_DIR --ckpts 10000 50000 200000      # the planner of lifetime checkpoints
./venv/bin/python plan.py RUN_DIR --ckpt ckpt_00200000.bin --explore-gain 0.02 --explore-eps 0.05  # + the acting rule
./venv/bin/python online.py --config configs/L_online.json --seed 101 --init RUN_DIR/ckpt_00200000.bin \
    --set online.steps=5000 [--set ppo.lr=0]      # continue from saved weights, learning or frozen
```

**Frozen checkpoints understate an online learner.** In vm-00005, L's frozen planner misses 1–19%
of the episodes, while the learning agent misses almost none from the same weights. It corrects its
errors within steps of making them. So measure with learning on (`online.py --init`) when the
question is the agent's competence. `plan.py --explore-gain/--explore-eps` adds online.py's
acting rule (babble when the plan promises too little relief, or at random) to the frozen test, and
`--env-seed` picks other episodes. `--init` starts with a fresh optimizer and an empty memory.

Per step on a quiet machine: the planner takes about 40 ms (4-way parallel). The update on 8 sequences of 8
steps takes 36–52 ms (8–32 threads). The basic agent (`actor=self`, one window) takes about 23 ms (43 steps/s)
with 8 threads.

**Gated memory (vm-00007, `online.memory=gated`).** Experience enters long-term memory by salience:
- **Two memories:** a short-term buffer of the last `online.stm` steps, and a long-term memory of
  `online.replay` steps kept as whole chunks of `ctx` steps.
- **The gate:** each chunk is decided `relief_k − 1` steps after it ends. It is stored:
  - in full during the first `store_warmup` steps;
  - when it leads to relief (`store_relief`: a hit in it or in the next `relief_k − 1` steps);
  - when it is in the top `store_surprise` fraction of recent chunks by surprise;
  - otherwise with probability `store_base`.
- **Surprise:** the error of the model's prediction of the next state, each part divided by its
  recent mean and summed: frame, drives and, for the self actor, the move. It is measured before the
  model learns from the step.
- **Training:** the newest sequence plus `batch − 1` chunks drawn uniformly from long-term memory.
- **The default `memory=stream`** is the vm-00005/6 ring buffer (bitwise unchanged).

```bash
./venv/bin/python online.py --config configs/AHR_gated.json --set online.store_relief=1 --set online.store_base=0.1
```

**Why the self actor does or does not aim** (`analyze_self_actor.py`, vm-00006/7; the true positions
are used for scoring only):
- `aim`: on the test episodes, the cosine between the predicted mean move and the direction to the
  circle (away from the walls), and the share of steps at a wall.
- `probe`: linear R² of the cursor, the circle and the direction between them, from z and from v.
- `relief`: how well the moves of the hindsight-labelled steps point at the circle (random
  babbling, or an agent replayed as it lives online).
- `inputs`: how much the move changes with the image, the command and the drives, and the norm of v.
- `momentum`: whether the move follows the previous one.

```bash
./venv/bin/python analyze_self_actor.py aim RUN_DIR [RUN_DIR ...] --step 50000
```

**The policy head (vm-00008):**
- `ppo.pi_detach=1`: the policy head's gradient stops at its input z. The move loss then trains the
  head (and the move std) only, never the Transformer or the encoder.
- `model.pi_hidden=H`: the head gets one ReLU hidden layer, π = W2 relu(W1 z + b1) + b2. With 0 (the
  default) it is linear, and the layout and old checkpoints are unchanged.
- `ppo.pi_own_clip=1`: the policy and value heads' gradient is clipped by its own norm and left out of
  the rest's. Without it, a detached head still scales every other update through the shared
  `max_grad_norm` clip.
- With a detached head with a hidden layer and inverse dynamics, the basic agent aims (vm-00008 DMI).

```bash
./venv/bin/python online.py --config configs/AHR_gated.json --set ppo.pi_detach=1 --set model.pi_hidden=128
```

**Patch tokens and moving dots (vm-00010):**
- `model.patch=16`: no CNN. Each step is 16 patch tokens (a 4×4 grid of 16×16 px: one shared linear
  embedding of the raw pixels, plus a learned position vector that starts as a 2D sine-cosine code), plus
  one token each for the drives, the previous move and the command, with type and time embeddings.
  - Attention is block-causal over the window.
  - Each patch token predicts the change of its cell of the next frame, and the drives token the next
    drives, from [its output, the executed move]. The move token's output is the policy.
  - The self actor only: the model acts through `vm_forward` on its window, so `vm_act`, `vm_encode` and
    the planner refuse patch mode. `agent.attention(...)` returns the attention weights of one sequence.
  - Other sizes (vm-00014): `model.patch=8` (64 patches of 8×8, 67 tokens per step, 536 per window of 8) and
    `model.patch=64` (the whole frame as one token). On the GPU, windows longer than 32 tokens use batched cuBLAS
    attention: the P8 update takes 6.7 ms, the forward 0.5 ms (P16: 2.4 and 0.4 ms).
- `env.distractors=8`: grey 2×2-px dots (brightness 60–180) moving at 0.5–2 px per step, bouncing off the
  walls, drawn under the circle and the cursor. They never affect the task, and their random streams are
  separate, so the episodes are identical with and without them. Positions: `info["gt_dots"]` (scoring
  only).
- `analyze_patch.py coupling`: when the cursor or a dot crosses from patch A to B, B's attention on A
  relative to the other patches, against the same ratio for an unrelated patch C looking at A (otherwise
  a salient A looks like coupling). `analyze_patch.py sensitivity`: where the predicted next frame changes
  when the move is reversed (near the cursor, near the dots). `analyze_patch.py tokens`: ridge probes from
  each kind of token output (move, drives, command, patches; via the PyTorch reference). With a detached
  policy head, no loss trains the move token's output: in vm-00010 it holds little, while the patch tokens
  side by side hold nearly everything, already at birth.

```bash
./venv/bin/python online.py --config configs/AHR_gated.json --set ppo.pi_detach=1 --set model.patch=16 \
    --set env.distractors=8 --set online.threads=8
./venv/bin/python analyze_patch.py coupling RUN_DIR --step 50000
```

## Performance (this machine: 128 vCPUs, SSE4.2; an RTX 3090 since 2026-10-24)

**The update on the GPU (task 00011, `libvmcuda.so`).** A CUDA port of the basic agent's update (the CNN model
with coord, residual frames and U-Net skips; frame, drives and move losses; clip and AdamW). Its gradients match
the C++ model's to about 1e-7 (`tests/test_cuda_model.py`). Run a lifetime with `--set online.device=cuda`: the
update and the forward passes run on the GPU (below).

| vm-00008 D-U, sequences × 8 steps | C++, 16 threads | C++, 64 threads | CUDA (with the batch upload) |
|---|---:|---:|---:|
| 8 (the online update) | 31.3 ms | 37.0 ms | 1.8 ms (2.1) |
| 32 | 73.4 ms | 52.9 ms | 4.2 ms (4.6) |
| 128 | 263.6 ms | 141.0 ms | 14.4 ms (16.0) |

**The forward pass too (task 00012).** With `online.device=cuda` the lifetime acts, computes the surprise and learns
on the GPU; the weights come back to the C++ agent only for checkpoints. The basic agent's online loop: 24 steps/s on
the CPU (16 threads), 62 with the GPU update alone, **202** with both (two lifetimes sharing the GPU: 137 each). A 50k
lifetime takes 6 minutes. Reruns of vm-00008 D-U learn as the CPU runs did (test success, hit rate and probes in the
same range; task 00012), but not bitwise. Inverse dynamics is supported too (task 00013), so DMI runs on the GPU.
The planner agent's latent dynamics and patch tokens are supported too (task 00015): the planner agent L runs at 42 steps/s per lifetime with 12 lifetimes sharing the GPU (3.5 on the loaded CPU). Not supported: PPO (only `train.py` uses it).

**Many lifetimes at once: the CUDA multi-process service.** Start `nvidia-cuda-mps-control -d` with `CUDA_MPS_PIPE_DIRECTORY` set (the same in the runs' environment). Without it, the processes take turns on the GPU; with it, their small kernels run at once. Twelve basic-agent lifetimes: 20 steps/s each without, 69 with.


**The GPU, first check (2026-10-24), before the port.** One update of vm-00008's DMI model (forward, backward and AdamW;
1.35M parameters), timed on a quiet machine. The C++ column uses 16 or 64 threads. The GPU column runs the
PyTorch reference in fp32 on the RTX 3090, with the frames downsampled on the GPU (the reference's NumPy
downsampling alone took 80% of the time). CUDA PyTorch is in `~/venvs/vm-gpu`; the project venv keeps CPU
torch.

| Sequences × 8 steps | C++, 16 threads | C++, 64 threads | GPU (PyTorch reference) | GPU busy |
|---|---:|---:|---:|---:|
| 8 (the online update) | 31.3 ms | 30.9 ms | 32.4 ms | 5.3 ms |
| 32 | 72.4 ms | 52.4 ms | 56.1 ms | – |
| 128 | 263.2 ms | 142.8 ms | 97.4 ms | 30.0 ms |

- At the online batch size the GPU computes for 5 ms of the 32. The rest is Python and the launches of about
  500 kernels per update. TF32 changes nothing.
- A GPU version pays only without that overhead (CUDA graphs, or a fused CUDA implementation): about 6× at the
  online batch size, judging by the busy time. The other way is to batch more, for example many lifetimes in
  one update.
- The basic agent's online loop on the quiet machine: 11 steps/s at 4 threads, 22 at 16. The update takes
  85% of the time.


**Small updates spread frames over threads.** When a minibatch has fewer sequences than threads, the
update and the batched forward split each sequence into per-frame work. That covers the convolutional
encoder and the frame decoder, the bulk of the compute: n_seq × ctx items over all threads. The
Transformer and the heads run one thread per sequence. This works for any context length. Gradients
match the PyTorch reference on both paths, and the result is deterministic for a given thread count.
Large minibatches keep one thread per sequence.

| One online step (8-step window), quiet machine | Before | After |
|---|---:|---:|
| Basic agent, 8 threads: forward + update | 6.2 + 43.3 ms | 1.6 + 21.5 ms |
| Basic agent, window of 32, 16 threads | 25.2 + 122.3 ms | 5.3 + 37.3 ms |
| Update on 8 windows, 32 threads | 54 ms | 36 ms |

`VM_AGENT_LIB=/path/to/lib.so` makes `agent_lib` load another build. Use it to test a new build while
experiments run on the installed one.

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
| `online.py` | one lifetime: single world, a gradient step after every step, replay memory |
| `analyze_self_actor.py` | the self actor: aiming, walls, feature probes, the aiming signal of labelled steps, input sensitivity |
| `plan.py`, `analyze_imagination.py`, `analyze_planner.py` | acting without reward (CEM planner on the latent model); imagination accuracy; planner failures |
| `reference/torch_reference.py`, `tests/` | PyTorch reference and tests |

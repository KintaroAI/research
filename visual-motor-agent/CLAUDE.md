# Visual–Motor Agent

A small, randomly initialized agent learns to move a cursor and click a red circle from pixels only.
The question: what does it learn about its own actions without being reinforced for it, and does
that help it learn the task? Plan: `../visual_motor_learning_plan.md`.

## Repository Structure

- **`dev/`**: active development (work here)
  - `src/vm_agent.cpp`, `src/vm_agent.h`: the model, in C++. Covers the CNN encoder, causal
    Transformer, policy/value/prediction heads, manual backward, acting with KV cache, rollout
    memory, GAE, PPO, AdamW and checkpoints. Built as `libvmagent.so`.
  - `agent_lib.py`: the ctypes wrapper (numpy pointers in, no copies).
  - `world/`: the Python world. `circle_world.py` is the vectorized numpy simulator; `record.py`
    records episodes; `browser_demo/` is the HTML/JS replica (play + replay) and is checked
    pixel-exact against Python by `check_js_equivalence.py`.
  - `train.py`: PPO loop (Python world + C++ agent). `run_queue.py`: job queue with 2 slots.
    `evaluate.py`: task metrics.
  - `analyze_dynamics.py`: counterfactual tests of the learned dynamics (plus
    `dynamics_over_training.py` and `compare_dynamics.py`). `plot_runs.py`: learning curves.
    `record_episodes.py`: GIFs, replay JSON and imagination strips. `make_video.py`: MP4s and
    screenshot sheets across checkpoints (needs dense checkpoints via `train.milestones`).
  - `plan.py`: CEM planner on the latent dynamics (acting without reward). `analyze_imagination.py`:
    imagined vs real rollouts, read out by retrieval. `analyze_planner.py`: where the planner fails.
    `train.py --set train.actor=planner`: the model learns from the planner's own experience.
  - `online.py`: one lifetime in one world, with a gradient step after every step from its own
    (replayed) experience. `model.one_frame=1`: each prediction uses only the current frame.
    `online.memory=gated` (vm-00007): 8-step chunks enter long-term memory by relief and surprise
    (`GatedMemory`). The default `stream` is the plain ring buffer, unchanged.
    `analyze_self_actor.py`: why the self actor does or does not aim (aim, probe, relief, inputs,
    momentum). `ppo.pi_detach=1`: the move loss trains the policy head only (vm-00008);
    `model.pi_hidden`: a hidden layer in the policy head.
  - `model.patch=16` (vm-00010): patch tokens instead of the CNN. Each step is 16 patch tokens plus one
    token each for drives, previous move and command; each predicts its own part of the next state.
    `env.distractors=K`: grey dots that move on their own (separate random streams: the episodes stay
    the same). `analyze_patch.py`: coupling (attention between patches), action sensitivity and probes
    per kind of token.
  - `src/vm_cuda.cu`, `src/vm_cuda.h`, `cuda_lib.py` (tasks 00011, 00012): the basic agent's forward pass and
    update on the GPU (`make libvmcuda.so`, `online.device=cuda`).
  - `reference/torch_reference.py`: float64 PyTorch re-implementation, used by the tests (and by
    `analyze_patch.py tokens` for all token outputs).
  - `configs/*.json`: runs from the plan's experiment matrix (A–F) plus reward-free babbling (P_*).
- **`NNNNN-experiment-name/`**: experiment snapshots (README.md + Makefile only; the git tag
  `exp/vm-NNNNN` preserves the source).
- **`tasks/`**: numbered task plans (write one before implementing a plan).

## Build & Test (from dev/)

```bash
make setup          # venv: numpy, pillow, matplotlib, pytest, CPU torch (torch: tests only)
make                # libvmagent.so
make test           # C++ unit tests (GEMM, acting == training bitwise, determinism, checkpoints)
make pytest         # world/reward/leak tests, JS equivalence, PyTorch gradient equivalence
make bench          # acting / update throughput
make libvmcuda.so   # optional: the GPU update (nvcc, cuBLAS); tests/test_cuda_model.py skips without it
```

## Run

```bash
./venv/bin/python train.py --config configs/A_baseline.json --seed 1 --total-steps 1000000 --set train.threads=64
./venv/bin/python analyze_dynamics.py RUN_DIR --ckpt ckpt_final.bin
./venv/bin/python plot_runs.py RUN1 RUN2 -o curves.png --labels A C
```

Outputs go to `~/data/research/visual-motor-agent/exp_NNNNN/RRR_name_sS/`, not into the repo.
`output_name.py` picks the next run number.

## Key Design Decisions

- **The agent runs on the CPU** (QEMU VM, 128 vCPUs, SSE4.2 only): C++17 with OpenMP and a hand-written
  SSE GEMM. Since 2026-10-24 the VM also has an RTX 3090 (24 GB, sm_86, driver 595, nvcc 12.0).
  - **The basic agent's lifetime can run on it** (tasks 00011, 00012): `src/vm_cuda.cu` → `make libvmcuda.so`,
    wrapped by `cuda_lib.py`; `online.py --set online.device=cuda`. Forward passes and updates run on the GPU;
    the weights come back to the C++ agent only for checkpoints. The update takes 1.8 ms against 31 ms in C++,
    and the online loop runs 202 steps/s against 24 (a 50k lifetime in 6 minutes). It supports the CNN model
    with coord, residual frames and U-Net skips, and the frame, drives and move losses only (not PPO, inverse
    or latent dynamics, patch tokens). GPU lifetimes learn like CPU ones but are not bitwise reproducible.
  - **Keep it equivalent:** `tests/test_cuda_model.py` checks the loss and every gradient against
    `vm_debug_grads` (about 1e-7) and the forward against `vm_forward`. Re-run it after touching either side. The same parameter layout is shared,
    so the CUDA code's `ParamId` order must follow `vm_agent.cpp`'s.
  - CUDA PyTorch (for experiments only) is in `~/venvs/vm-gpu`; the project venv keeps CPU torch.
  - Use 64 threads per run. Acting is about 2× faster than with 128, and two runs fit on the
    machine at once.
  - Expect about 1.7–2.0k env-steps/s per run with 4 PPO epochs.
- **Information hygiene.** The model receives only: the image (uint8), the drives (zeros unless
  `env.drives`), the previous executed action, and optionally the previous reward. `info["gt_*"]`
  and the env state are for evaluation and scoring only.
- **Context = block.** Contexts restart at episode starts and every `ctx` (32) steps. Training
  sequences are exactly those blocks, so acting and training give bitwise-identical outputs (tested).
- **Every gradient is verified** against the PyTorch reference (`tests/test_model_reference.py`,
  about 1e-7 relative error per tensor). Re-run it after touching any op.
- **Prediction is auxiliary only.** It is never rewarded. Plain pixel MSE from a global latent
  ignores the 1%-of-pixels cursor. The recipe that learns action consequences reward-free (exp
  vm-00001) is:
  - `model.coord=1`, `model.pred_residual=1`, `model.pred_skip=1`
  - `ppo.pred_changed_weight=10`
  - enough updates: `model.ctx=8` with 32 minibatches, lr 1e-3
  - optionally the inverse-dynamics head: `model.inv_hidden=128`, `ppo.inv_coef=0.1`, about 2–4×
    faster. It predicts the own executed (dx, dy) from consecutive *visual features*, never from
    tokens, which contain the previous action.
  See `configs/P_babble_inv.json`.
- **Latent dynamics heads (`model.dyn_hidden`) must not train the encoder.** Matching imagined to
  encoded latents with gradients into the encoder collapses the features. Even a decoder on real
  latents alone shrank them until the cursor was no longer decodable (exp vm-00002). The heads
  read a detached v.
- **The self actor (`online.actor=self`) needs replay.** With the newest window only (batch 1), its
  encoder features blow up (norm 10²–10⁵ by 100k steps in 5 of 6 runs, vm-00006), and the drives
  and command vanish from the tokens. There is no weight decay (`ppo.weight_decay=0`). Even with
  replay, its features lose the cursor over training (vm-00007); check with
  `analyze_self_actor.py probe`. The cause is the move loss: detach it (`ppo.pi_detach=1`). To aim, the
  head needs a hidden layer (`model.pi_hidden=128`), so the hindsight command can steer it: with inverse
  dynamics the basic agent then reaches test success 0.63–0.82 at 50k (vm-00008 DMI).
- **A detached head still shares the gradient-norm clip** with the rest (`ppo.max_grad_norm`): its large
  move-loss gradient scales every other update down. `ppo.pi_own_clip=1` clips the heads by their own norm
  (found in vm-00009: a detached move loss slowed the planner agent's early learning).
- **Frozen checkpoints understate an online learner** (vm-00005). L's frozen planner misses 1–19%
  of episodes, while the same weights, still learning, miss almost none: errors are corrected
  within steps. To measure the agent, continue it with learning on (`online.py --init CKPT`; lr 0
  is the frozen control). `plan.py --explore-gain/--explore-eps` adds online.py's acting rule.
- **Run names:** `train.py --set name=X` overrides the config's name. Experiment Makefiles use
  it so their analysis globs find the runs.
- **Checkpoint header.** Config ints sit in slots 2–19 and n_params/has_opt/step in 20–22.
  Extension fields go at 32+ (`pred_residual` 32, `pred_skip` 33, `inv_hidden` 34, `dyn_hidden` 35,
  `one_frame` 36, `pi_hidden` 37, `patch` 38). Never append into slot 20.
- **Patch tokens act through `vm_forward`** on the context window (the basic agent already does), so
  they need no KV cache: `vm_act` and `vm_encode` refuse patch mode, and so does the planner (it imagines
  with the CNN model's latent dynamics). Attention is block-causal: a token sees all tokens of its own
  step and of earlier steps.
- `make` builds `libvmagent.so` under a temporary name and renames it, so running trainers are not
  disturbed by a rebuild.
- **Small updates are frame-parallel.** With fewer sequences than threads, `loss_and_grads_frames`
  and `vm_forward` spread per-frame encoder and decoder work over threads, for any `ctx`. Larger
  minibatches use one thread per sequence. Both paths are covered by the reference gradient tests
  (run at 4 and 16 threads).
- **Testing a new build while experiments run:** build it to another path, run
  `VM_AGENT_LIB=that.so pytest`, and only then `make`.

## Conventions

- No co-authored-by lines in commit messages.
- Don't kill runs with `pkill -f <pattern>`: the pattern also matches the calling shell. Kill by PID.
- Experiment READMEs are technical reports (hypothesis, method, results, analysis).
- Large logs and checkpoints stay in `~/data/...`; READMEs quote key numbers only.

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
    momentum).
  - `reference/torch_reference.py`: float64 PyTorch re-implementation, used only by tests.
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

- **No GPU on this machine** (QEMU VM, 128 vCPUs, SSE4.2 only). Everything is CPU C++17 with
  OpenMP and a hand-written SSE GEMM.
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
  `analyze_self_actor.py probe`.
- **Frozen checkpoints understate an online learner** (vm-00005). L's frozen planner misses 1–19%
  of episodes, while the same weights, still learning, miss almost none: errors are corrected
  within steps. To measure the agent, continue it with learning on (`online.py --init CKPT`; lr 0
  is the frozen control). `plan.py --explore-gain/--explore-eps` adds online.py's acting rule.
- **Run names:** `train.py --set name=X` overrides the config's name. Experiment Makefiles use
  it so their analysis globs find the runs.
- **Checkpoint header.** Config ints sit in slots 2–19 and n_params/has_opt/step in 20–22.
  Extension fields go at 32+ (`pred_residual` 32, `pred_skip` 33, `inv_hidden` 34, `dyn_hidden` 35,
  `one_frame` 36). Never append into slot 20.
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

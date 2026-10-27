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
    For the universal model: `forage_video.py` (a frozen agent foraging, a sheet every 100 steps) and `dream_video.py`
    (forage, then the model dreams on its own predictions; `--decode pair` by default).
  - `plan.py`: CEM planner on the latent dynamics (acting without reward). `analyze_imagination.py`:
    imagined vs real rollouts, read out by retrieval. `analyze_planner.py`: where the planner fails.
    `train.py --set train.actor=planner`: the model learns from the planner's own experience.
  - `online.py`: one lifetime in one world, with a gradient step after every step from its own
    (replayed) experience. `model.one_frame=1`: each prediction uses only the current frame.
    `online.memory=gated` (vm-00007): 8-step chunks enter long-term memory by relief and surprise
    (`GatedMemory`). The default `stream` is the plain ring buffer, unchanged.
    `analyze_hazards.py` (vm-00012): whether a planner agent's hunger model imagines the hazards' cost; `contact`
    also probes patch-token models (per kind of token, and the token of the patch under the cursor).
    `analyze_self_actor.py`: why the self actor does or does not aim (aim, probe, relief, inputs,
    momentum; `--device cuda` runs its forward passes on the GPU, 30× faster). `ppo.pi_detach=1`: the move loss
    trains the policy head only (vm-00008);
    `model.pi_hidden`: a hidden layer in the policy head.
  - `model.patch=16` (vm-00010): patch tokens instead of the CNN. Each step is 16 patch tokens plus one
    token each for drives, previous move and command; each predicts its own part of the next state.
    `env.distractors=K`: grey dots that move on their own (separate random streams: the episodes stay
    the same). vm-00011: `env.respawn` (with `end_on_success=false`, a large `max_steps`: a world that never
    resets), `env.target_speed`, `env.hazards`, `env.wrap` (a torus) and `env.egocentric` (the view centered
    on the cursor). In a continual world with walls the basic agent starves at a wall: use `wrap`.
    Hunger is capped at 1: at the cap a hazard step leaves no trace; use slow hunger (`env.h_rate=0.001`, vm-00015).
    `env.hazard_click=1` (vm-00017): a click inside a hazard sets hunger to 1 (poison); `analyze_hazards.py click`
    tests whether a model predicts what each kind of click does (placed one-step windows); `click-real` (task 00027,
    universal model) tests it on real windows: the frozen agent lives as in the forage test, each chosen move is tried
    with and without a click on copies of the world, and the prediction is read from its last 1, 2, 4 and ctx steps.
    `env.touch` (task 00042, needs `respawn`): food and poison act on contact, no click, and a touched poison is
    replaced away from the cursor (`configs/U_touch.json`, smaller food; a random walk touches 4.7 food and 11.8 poison
    per 1k).
  - **The universal model** (task 00023, `../ARCHITECTURE.md`): `model.universal=1` with `model.patch`: one token
    format, fully causal next-token prediction, one head, two-hot cross-entropy over `model.bins`; `model.gray=1` reads one
    channel per patch (task 00024). Start from `configs/U_universal.json` (grayscale world, checkerboard poison, dots,
    the instinct; `lifetime.py`). Its forward is `Agent.forward_uni` / `CudaModel.forward_uni`; `vm_forward` refuses it.
    `online.click_bias` is skipped for models without a policy head (it used to write into the next parameter).
    Surprise gating (task 00028): `online.store_surprise` with the universal model on the GPU; a step's surprise is its
    largest token cross-entropy relative to that slot's recent mean (`GatedMemory.note_tokens`, read one tick late from
    `CudaModel.uni_token_ce`, after acting). Once long-term memory is full, `online.ltm_evict` (task 00031) says what a
    stored chunk replaces: `oldest` (the ring, default) or `reservoir` (a uniform sample of the life); vm-00025: in a
    life longer than the memory, surprise gating with the ring is the rule that avoids traps. The text world (task 00032):
    `env.kind = text` (`world/text_world.py`, made by `world.make_config` / `make_world`, which `lifetime.py` uses):
    Tiny Shakespeare as grey pages, configs `T_text_A.json` (64 letters in one 8x8 patch token) and `T_text_B.json`
    (one 2x2 letter per token), `bins = 256`, `pred_img = 8` (unused by the universal model but validated: it must
    divide `img`). `eval_predict.py` scores any world's held-out prediction (nats per value; text: per letter);
    `reference/char_gpt.py` (run with `~/venvs/vm-gpu/bin/python`) is the classical character-level GPT on the same
    windows. `model.in_bins = 1` (task 00036, vm-00030): bin-embedding input, `W_in` d x (V * bins) over each used value's
    two-hot bins (init 0.02), in C++, CUDA and the reference; header slot 43. Set in every universal config since task 00037 (vm-00031: with surprise-gated memory no 500k-tick life was trapped). The corpus is read from `../llm-fundamentals/dev/data/tinyshakespeare/input.txt` (`./venv/bin/python prepare_tinyshakespeare.py` there downloads it). The instinct's horizon (task 00029): `online.instinct_h` (1: one step)
    holds each candidate for H steps, imagined by the model (`online.uni_lookahead`, Jacobi decoding); it costs
    (H − 1)·J + 1 forward passes per tick. `online.instinct_decode=mode` (GPU) decodes imagined pixels by their most probable
    bin instead of the mean (its bin centres are not the world's grey levels: use `pair`, task 00038, the most probable
    bin and its more probable neighbour, weighted: sharp, and exact for two-hot values); Self-forcing (task 00039, lifetime.py on the GPU): `online.sf_windows` replayed windows per update
    get their last 1..`sf_steps` steps' inputs (frame and body) from the model's own predictions (`online.dream_inputs`:
    `sf_jacobi` Jacobi iterations, pair decoding; moves stay real), the targets stay real (`CudaModel.set_uni_inputs`);
    8 windows of up to 3 steps cost ~70% of the speed (29 ticks/s per lifetime). `eval_dream.py` measures the dream:
    the frozen agent forages, then the model dreams 16 steps with the real moves that followed (pixels exactly right,
    cursor lost / distance, hunger error per dream step). `reference/text_dream.py` (task 00040, `~/venvs/vm-gpu`): the text world's dream is generation;
    the universal model and the classical GPT continue the same val contexts, greedy and sampled, scored by both (a letter's
    probability: the mean of its glyph values' log-probabilities; eval_predict's `nll_glyph` sums them, a temperature of 1 / V). imagined frames lose the moving cursor either way (task 00029). `online.py --eval-tests forage --eval-worlds 16 --eval-horizon 4` keeps
    the frozen test affordable.
    Speed (task 00026): the instinct's candidates reuse the policy pass's keys and values (`CudaModel.uni_candidates`),
    the universal model's batches carry no next frames, and `lifetime.py` reads the update's stats every
    `online.stats_every` ticks: 178 ticks/s for a gray lifetime with the instinct alone on the RTX 4090.
  - `lifetime.py` (task 00020, `../ARCHITECTURE.md`): the self actor with N parallel bodies early in life
    (`online.bodies`, `bodies_hold`, `bodies_halve`; one shared long-term memory, `ChunkStore`), the signed hindsight
    command (`online.goal_mode=delta`) and exploration decay. With one body it reproduces `online.py` bit for bit.
    `model.act_token=1` (patch models): a learned query token per step that the policy reads, trained by its own
    predictions of the next drives and the coarse next frame (`ppo.act_coef`). `analyze_patch.py`: coupling (attention between patches), action sensitivity and probes
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
  SSE GEMM. Since 2026-10-24 the VM also has an RTX 3090 (24 GB, sm_86, driver 595, nvcc 12.0), and since
  2026-10-25 an RTX 4090 (24 GB, sm_89). `libvmcuda.so` carries native code for both.
  - **Pick a GPU with `CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=N`.** CUDA numbers devices fastest-first by
    default, so without the order variable `CUDA_VISIBLE_DEVICES=0` is the 4090, while `nvidia-smi`'s GPU 0 is the
    3090. The 4090 is about 1.5× faster (8×8-patch update with the act token 3.8 ms against 5.6); three 8×8-patch lifetimes per card at
    once run at 113 ticks/s each on the 4090 and 74 on the 3090. One MPS daemon serves both cards.
  - **Profiling:** Nsight Compute works for this user (`NVreg_RestrictProfilingToAdminUsers=0` in
    `/etc/modprobe.d/nvidia-profiling.conf`): `ncu --metrics gpu__time_duration.sum --csv ./venv/bin/python SCRIPT`.
  - **The basic agent's lifetime can run on it** (tasks 00011, 00012): `src/vm_cuda.cu` → `make libvmcuda.so`,
    wrapped by `cuda_lib.py`; `online.py --set online.device=cuda`. Forward passes and updates run on the GPU;
    the weights come back to the C++ agent only for checkpoints. The update takes 1.8 ms against 31 ms in C++,
    and the online loop runs 202 steps/s against 24 (a 50k lifetime in 6 minutes). It supports the CNN model
    with coord, residual frames and U-Net skips; the frame, drives, move, inverse-dynamics and latent-dynamics
    losses, the planner's `encode`/`imagine`/`decode`, and patch tokens (tasks 00013, 00015): DMI, the planner
    agent L and vm-00010's patch-token agents run there. Windows longer than 32 tokens (patch tokens: `model.patch=8`
    gives 536 per window) use batched cuBLAS attention, any length (task 00018). Not PPO (only `train.py` uses it). GPU lifetimes are not
    bitwise reproducible against CPU ones. `online.py --eval RUN --eval-device cuda` runs the frozen tests'
    forward passes on the GPU (about 10 s per checkpoint).
  - **Many lifetimes on one GPU: start the CUDA multi-process service** (`nvidia-cuda-mps-control -d`, with
    `CUDA_MPS_PIPE_DIRECTORY` set the same for the daemon and the runs). Twelve basic-agent lifetimes then run at
    69 steps/s each instead of 20.
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
  `one_frame` 36, `pi_hidden` 37, `patch` 38, `act_token` 39). Never append into slot 20.
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

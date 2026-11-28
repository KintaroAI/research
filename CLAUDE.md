# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Research repository for AGI development (KintaroAI): small hands-on experiments on how a system can learn to predict, remember and act, from GPT-2 in raw CUDA to self-organizing maps and an agent that learns online from pixels.

There are four topics. Each is a directory with its own `README.md` (overview, current state, progress table), numbered experiments, `tasks/` and `dev/`. Every topic except `llm-fundamentals` has its own `CLAUDE.md`: read it before working in that topic. This file holds what is common, the state of the whole repository, and the `llm-fundamentals` build notes.

### Topics and state (2026-11-27)

| Topic | What it is | State | Latest experiment |
|---|---|---|---|
| `visual-motor-agent/` | One universal next-token model (pixels, body, moves) living one life and learning online; C++/CUDA model, Python worlds | **Active** since 2026-10-23 | vm-00125 (2026-11-27, a look on the checkpoints of five lives: a dream that rolls dice serves the court's ball, 3 to 20 balls in 32 long dreams where the sharp decoding serves none, and at a temperature of .7 at little cost for the lives that lived with the owner's mix); it and all before it written up and tagged |
| `thalamus-sorter/` | Topographic maps from temporal correlation; clusters wired to columns, a feedback loop, a forage benchmark | Parked | ts-00028 (2026-11-14); before it ts-00027 (April) |
| `column/` | A soft winner-take-all cell with Hebbian learning and no backprop (a cortical minicolumn model) | Parked since 2026-03-18 | 00016 |
| `llm-fundamentals/` | GPT-2 training in raw CUDA/C: banded sparsity, grokking, sequential evaluation, fact memorization | Parked since 2026-11-12 | 00021 |

### Current focus (2026-11-27)

- **The default agent** of `visual-motor-agent` is `dev/configs/U_agent.json` (vm-00101's TT, since 2026-11-22). New experiments in the food worlds start from it.
- **By the owner's decisions of 2026-11-26: the world model first, and watching first.** Its prediction, its dream and its memory are to be improved before anything else, and for now without the model's own movement or decisions: it observes worlds that move by themselves (task 00132: `configs/W_discs.json`, moving food and poison discs; `configs/W_breakout.json`, the court played by a script, with its score). Two models are compared by measures that do not depend on their parts, never by the loss: `visual-motor-agent/dev/wm_measures.py` (three calls: `reset`, `observe`, `dream`; `visual-motor-agent/ARCHITECTURE.md` section 15).
- **The first table of those measures is vm-00119.** The model that only watches predicts the scripted Breakout court well (next-frame skill .667, a hit forecast with an area under the curve of .996) and moving discs half as well (.337); its dream holds the ball for 9.1 ticks and a disc for 3.0 to 3.9, where a straight line through the last two frames holds a disc for 10; decoded with the expected value it loses the ball by the fourth tick (vm-00119). Dream with `pair`; vm-00114's lookahead dreamed with the expected value. Next, for the owner's word: the changes of section 15 (the discs held in the dream, new discs in the watched world, the window and then memory), each judged on that table.
- **Waiting for that:** the acting rule (task 00123), that is, whether a trained actor should replace the instinct that picks the best of 16 sampled moves. vm-00114 (lookahead in the model's dream) closed negative, and so did vm-00115 (a policy-gradient actor: it narrows to a near-fixed move within 20k ticks and learns neither world). vm-00116 measured why the trained agent misses in Breakout: its label knows where the paddle should be but barely orders the candidate moves, and the ring is not the cause. Action repeat (vm-00118, `online.act_every = 4`: a move decided every 4 ticks and held between) closed negative too: the label orders held moves better (.69 against .62), but one decision in 4 ticks corrects less than four and the play falls by .065 ± .053. What is left of the task are the arms that teach the label what a move does (an advantage-style target; labels for moves not made).
- **One set of weights in two worlds** (vm-00117, the first step of challenge 1 of `visual-motor-agent/ARCHITECTURE.md` section 13): the weights of a life in the food world start a life in Breakout and the other way round. Transfer is one-sided (a food-world past helps in Breakout, a Breakout past hurts in the food world), and the old world's play is at an untrained model's level after 10k ticks of the new one. The follow-up named beforehand is rehearsal (one process, one memory across both worlds); it waits for the world model too.
- **One transformer predicts better than the four (vm-00120, 2026-11-27; the owner's step back, task 00133).** vm-00119's lives and measures with one transformer on the whole 64 x 64 frame (`configs/W1_discs.json`, `configs/W1_breakout.json`; 2.17M parameters against 8.64M), paired by seed: in the court next-frame skill .759 against .667 (+.092 ± .010) and the ball held for 10.8 ticks against 9.1; in the disc world the next frame level on new discs and a food disc held at tick 4 in .666 of the dreams against .438. The four lose at the borders between the quarters in the court and everywhere in the disc world; they forecast a hit a little better (−.040 ± .035 8 ticks ahead). Acting was not run (the four were adopted for it, vm-00051). Next: the changes of section 15 with one transformer as the watching model; the first, the discs held in the dream (self-forcing, which `lifetime.py` refuses with the four and runs with one), is done (vm-00121, below).
- **Trained on its own dreamed frames (vm-00121, 2026-11-27; task 00134).** Self-forcing with the one watching transformer: of the 7 replayed windows of an update, 2 get the inputs of their last 1 to 3 steps (SD3, SB3) or 1 to 7 steps (SD7, SB7) from the model's own predictions; against vm-00120's lives by seed (seeds 1–6; SB7 seeds 1–2). **It works where the life knows the things:** in the court, with up to 3 steps, the ball is held for 16.2 dreamed ticks against 10.8 (+5.3 ± 2.9, 6 of 6 seeds), past a straight line through the window's frames (11), and the next frame pays nothing. On the disc world's exam, whose discs and dots the life has never seen, nothing is gained (a food disc at tick 4 in .622 of the dreams against .625) and up to 7 steps harm (at tick 8 .157 against .280); the life's own discs are dreamed better in both arms. New in `wm_measures.py`: the part `course` (a dreamed disc falls behind on its course). Next, for the owner's word: the disc world with variety (discs and dots replaced now and then), with and without up to 3 dreamed steps, on 1 to 2 seeds an arm.
- **A longer life (vm-00122, 2026-11-27; task 00135; a look on one seed an arm).** The owner: "how about we run 200k runs for both worlds to see if it improves over time?" The one watching transformer for 200k ticks in both watched worlds, with and without up to 3 dreamed steps, scored every 20k ticks. **What the life knows keeps improving to the end, what it has never seen does not:** the court's ball is held for 16 dreamed ticks at 200k against 9 at 60k (26 against 15 with self-forcing, which keeps its gain: about 65k ticks with it do what 200k do without); the life's own discs are dreamed better and better (the food held at tick 8 in .516 of the dreams against .266); on new discs nothing moves (the food held at tick 4 .725 against .711) and next-frame skill falls a little, so the gap between own and new discs widens. Read one life's curve with care: two runs of the same seed differ at 60k by up to 2 ticks of the ball's horizon. New: `dev/wm_curves.py` (a life's numbers by its ticks), `wm_measures.home_skip` (a life's home set starts after its last tick). Next, for the owner's word: the disc world with variety (discs and dots replaced now and then), with and without up to 3 dreamed steps, one seed an arm.
- **Its own prediction mixed into every frame (vm-00123, 2026-11-27; task 00136; the owner's idea; a look on one seed an arm).** `online.mix = .5`: each pixel of every frame the watching transformer sees comes from its own sharp one-step prediction of that frame with probability .5; the targets and memory are the real frames, and the mix is made again at every update (the owner: "replay - mix should be re-made each replay"). **It gives the court's dream what self-forcing gives it, harms the disc world's, and does not mend a long dream.** The court at 60k: the ball held for 15 dreamed ticks against 9 and 9 without self-forcing and 15 and 17 with it, next-frame skill .742 against .750 and .775. The discs: behind the plain life (the food held at tick 4 on own discs .555 against .781, on new ones .564 against .711). In the free dream of 200 ticks the court life with the mix keeps a ball in about one dream in eight at tick 200 (the others in one in two), and no dream serves a new one. The noise it adds is the model's own error: 0.05% of the lent pixels in the court at 60k, 2.42% in the disc world. A life with the mix costs about four times a plain one.
- **What the mix is made of (vm-00124, 2026-11-27; task 00137; the owner's word; a look on one seed an arm).** The same mix with .7 of the pixels from the prediction (`online.mix = .7`), and with .5 from the prediction and .2 a random grey (`online.mix_noise = .2`, new). **Neither does better than .5, and both pay more.** The court at 60k: the ball held for 12 and 11 dreamed ticks against 15, next-frame skill .696 and .656 against .742; the life with random grey draws no ball at 20k and is still climbing at 60k. The discs: every mix behind the plain life (the food held at tick 4 on own discs .555, .516 and .367 against .781). In the free dream of 200 ticks every mix loses the court's ball sooner than the lives without one, random grey soonest, and no dream serves a new ball; the noisier makes sometimes draw a ball again that was gone for 3 to 5 ticks (a look by a scratch count), never one that has left the window of 8 ticks. Why no ball is served (a scratch test on the 60k checkpoints): every life expects the serve when it is due and not where (the serve's column is random), and `pair` draws each pixel's most probable value, which is empty. Next, for the owner's word: a dream that draws (a sampled decoding) and the disc world with variety.
- **A dream that rolls dice (vm-00125, 2026-11-27; task 00138; the owner's word: "yes - lets do that. can temperature help?"; no new life: a look).** A new decoding of the dream picks one of the humps of each pixel's predicted chances by their weights raised to 1 / temperature (`wm_measures.py score --decode drawp0.7`: a hump weighed by its tallest bin, as `pair` ranks values; `draw0.7`: by its chance), tried on the 60k checkpoints of the five court lives. **It serves:** with `pair` no life serves a ball in its 32 long dreams, with dice at T 1 3 to 20 (the world 12), and a ball is in the court at tick 200 in .719 to .969 of the dreams against .000 to .500. **A temperature below 1 helps the lives that lived with the owner's mix** (at .7 the life with the .5 mix serves 23 balls and holds the short dreams' ball for 14 ticks against 15); the lives without a mix serve few or none below 1. Shown one pixel of a served ball the lives with the mix dream a ball in .84 and .86 of the cases, the plain ones in none (a scratch test): the dice give a pixel, and the mix has taught the model to make a ball of it. **The cost:** a dream with dice is one of the possible futures and scores lower against the world's own continuation (at T 1 the ball held 6 to 11 ticks against 9 to 15): dice for a long dream, `pair` for the exam. Found on the way: `pair` is not the model's likeliest value; it leans toward grey levels that sit on a bin (the ball's white), which is part of what holds the ball in its dreams. Next, for the owner's word: a court life trained on frames drawn with dice.
- **Where the agent stands, its reference numbers and known limits:** `visual-motor-agent/README.md`, section Current state.
- **Hardware:** a QEMU VM with 128 vCPUs (no AVX), 251 GB RAM, an RTX 3090 (GPU 0) and an RTX 4090 (GPU 1) under `CUDA_DEVICE_ORDER=PCI_BUS_ID`.

## Repository Structure

- **`<topic>/README.md`** — the topic's overview, current state and Progress table (one row per experiment)
- **`<topic>/CLAUDE.md`** — how to work in the topic: build, run, options, pitfalls (for `llm-fundamentals`: this file)
- **`<topic>/dev/`** — active development directory (work here)
- **`<topic>/tasks/NNNNN-name.md`** — task plans, written before implementing
- **`<topic>/NNNNN-experiment-name/`** — one directory per experiment: `README.md` (the technical report), `Makefile` (reproduces it) and, where used, `images/` with small PNGs for the README. No copies of the source: the git tag preserves it (`exp/NNNNN` for `llm-fundamentals`, `exp/vm-NNNNN`, `exp/ts-NNNNN`, `c/NNNNN`). Some early experiments also hold scripts.
- **`_templates/`** — README and Makefile templates for experiments and topics
- **`RESEARCH_PLAN.md`** — the routed-prediction plan of March 2026 (thalamocortical modules against recurrent baselines); its phases have not been run as written
- **`visual_motor_learning_plan.md`**, **`visual_motor_learning_transformer.md`** — the starting plans of `visual-motor-agent`; its current design and research plan are in `visual-motor-agent/ARCHITECTURE.md`

Experiments follow the naming convention `NNNNN-short-descriptive-name` (5-digit zero-padded sequential, per topic). The workflow (plan, run, write up, tag) is `llm-fundamentals/dev/EXPERIMENT_PROTOCOL.md`; `column/EXPERIMENT_PROTOCOL.md` is that topic's copy.

## Keeping the Docs Current

The READMEs and CLAUDE.md files are part of the work. On 2026-11-25 the `visual-motor-agent` table was 45 experiments behind and this file described one topic of four. Update them in the same change as the work, without being asked:

1. **A new experiment directory** (planned or running): add its row to the topic README's Progress table, with the status Planned or Running and what is being run.
2. **A write-up:** complete the row: what was run (arms, seeds, ticks), the status, and the key result with its headline numbers, taken from the experiment README's conclusions. One row per experiment directory, in order.
3. **The topic README's "Current state" section:** revise it when a write-up changes what it says: the default agent or config, the reference numbers, a limit found or lifted, what is running. Put the new date in its heading.
4. **The topic `CLAUDE.md`:** every new option, tool, config, default or pitfall, at the place the next session will look for it.
5. **This file and the root `README.md`:** the topic table (state, latest experiment) and "Current focus", whenever a topic's latest experiment, its state (active, parked), the default agent or the hardware changes. The two tables carry the same facts: change them together, with the date.

Rules for all of them:

- Take numbers from the experiment's own README, never from memory. Interim numbers of running jobs stay out.
- No literal pipe characters inside table cells (write words or `abs(d)`).
- Before finishing a write-up, check that no row is missing. The two counts must be equal:
  ```bash
  ls -d visual-motor-agent/[0-9]*/ | wc -l; grep -c '^| \[\?0' visual-motor-agent/README.md
  ```
- If you find these files behind the repository (an experiment without a row, a stale date or state), fix that first and say so.

## llm-fundamentals: Build & Test Commands

All commands run from `llm-fundamentals/dev/`:

```bash
# First-time setup
make setup          # Creates venv, installs Python deps (tiktoken, torch, etc.)
make data           # Downloads & tokenizes TinyStories dataset
make model          # Generates model.bin via create_model.py

# Build
make all            # Builds train and generate executables
make train          # Build training binary only
make generate       # Build inference binary only

# Test
make test           # Build and run the original fp32 correctness test
make test_all       # Build and run all 5 test suites
make test_banded_sparsity   # Build single test
./test_banded_sparsity      # Run single test after building

# Training
./train                           # Dense training (default)
./train -1 256 -2 256             # Banded sparsity on FC1+FC2
./train -p 4                      # Loss only at position 4 (for grokking)
./train -e model.bin -c out.bin   # Custom checkpoint in/out

# Inference
./generate -e checkpoint.bin -n 256 -p "Once upon a time"
```

Tests require `gpt2_124M.bin` and `gpt2_124M_debug_state.bin` (generated by `python train_gpt2.py --model gpt2 --write_tensors 1 --num_iterations 0`).

The fact-memory experiments (00020, 00021) are PyTorch code in `llm-fundamentals/dev/facts/` (`facts_lib.py`, `train.py`, `analyze.py`).

## llm-fundamentals: Architecture

**Tech stack:** Python (PyTorch, tiktoken) for data prep and model creation; C/CUDA with cuBLAS for training and inference.

### CUDA Source (`llm-fundamentals/dev/src/`)

- **`train_gpt2_fp32.cu`** — Complete GPT-2 training: forward, backward, AdamW update, banded sparsity masks, task-position loss masking, checkpoint save/load, snapshot system for visualization. Uses `#ifndef TESTING` guard around `main()`.
- **`generate.cu`** — Inference-only forward pass with token sampling. Also has `#ifndef TESTING` guard.
- **`test_*.cu`** — Test suites use `#define TESTING` + `#include "train_gpt2_fp32.cu"` (or `generate.cu`) pattern to access internals without compiling `main()`.
- **`llmc/`** — Headers: `utils.h` (file I/O), `dataloader.h` (binary token loading), `tokenizer.h` (token decoding), `rand.h` (RNG).

### Checkpoint Format

Binary files with 256-int header (magic `20240326`, version `3`) followed by fp32 parameters. 16 parameter tensors stored contiguously: wte, wpe, ln1w, ln1b, qkvw, qkvb, attprojw, attprojb, ln2w, ln2b, fcw, fcb, fcprojw, fcprojb, lnfw, lnfb. Weight tying: wte serves as both input embedding and output projection.

### Key Features

- **Banded sparsity:** FC1/FC2 weight masks restrict connections to a band pattern. Masks applied during backward (gradient masking) and after optimizer update (weight enforcement). Controlled by `-1`/`-2` bandwidth flags (0 = dense).
- **Task-position masking:** `task_position` field restricts loss/gradients to a single sequence position per sample. Used for grokking experiments where only the answer token matters.
- **Memory-efficient backward:** Only 3 gradient activation buffers reused across layers, vs 21 forward activation buffers.

## Conventions

- No co-authored-by lines in commit messages
- Few seeds first (the owner, 2026-11-27): a new idea is tried on 1 to 2 seeds an arm to see whether it works at all; runs of many seeds wait until something is worth confirming, and for the owner's word (`visual-motor-agent/CLAUDE.md`, Conventions)
- No CI/CD or linting — pure research workflow
- Experiment READMEs serve as technical reports (hypothesis, method, results, analysis)
- Write a numbered task plan in the topic's `tasks/` before implementing
- Experiment outputs, checkpoints and logs go to `~/data/research/<topic>/`, never into the repo; READMEs quote key numbers only
- `llm-fundamentals` build tool is `nvcc` with flags `-O3 -lcublas -lm`

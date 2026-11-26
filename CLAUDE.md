# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Research repository for AGI development (KintaroAI): small hands-on experiments on how a system can learn to predict, remember and act, from GPT-2 in raw CUDA to self-organizing maps and an agent that learns online from pixels.

There are four topics. Each is a directory with its own `README.md` (overview, current state, progress table), numbered experiments, `tasks/` and `dev/`. Every topic except `llm-fundamentals` has its own `CLAUDE.md`: read it before working in that topic. This file holds what is common, the state of the whole repository, and the `llm-fundamentals` build notes.

### Topics and state (2026-11-25)

| Topic | What it is | State | Latest experiment |
|---|---|---|---|
| `visual-motor-agent/` | One universal next-token model (pixels, body, moves) living one life and learning online; C++/CUDA model, Python worlds | **Active** since 2026-10-23 | vm-00115, running since 2026-11-25; vm-00114 and all before it are written up |
| `thalamus-sorter/` | Topographic maps from temporal correlation; clusters wired to columns, a feedback loop, a forage benchmark | Parked | ts-00028 (2026-11-14); before it ts-00027 (April) |
| `column/` | A soft winner-take-all cell with Hebbian learning and no backprop (a cortical minicolumn model) | Parked since 2026-03-18 | 00016 |
| `llm-fundamentals/` | GPT-2 training in raw CUDA/C: banded sparsity, grokking, sequential evaluation, fact memorization | Parked since 2026-11-12 | 00021 |

### Current focus (2026-11-25)

- **The default agent** of `visual-motor-agent` is `dev/configs/U_agent.json` (vm-00101's TT, since 2026-11-22). New experiments in the food worlds start from it.
- **Being worked on:** the acting rule (task 00123), that is, whether a trained actor should replace the instinct that picks the best of 16 sampled moves. vm-00114 (lookahead in the model's dream) closed negative; vm-00115 (a policy-gradient actor) is running.
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
- No CI/CD or linting — pure research workflow
- Experiment READMEs serve as technical reports (hypothesis, method, results, analysis)
- Write a numbered task plan in the topic's `tasks/` before implementing
- Experiment outputs, checkpoints and logs go to `~/data/research/<topic>/`, never into the repo; READMEs quote key numbers only
- `llm-fundamentals` build tool is `nvcc` with flags `-O3 -lcublas -lm`

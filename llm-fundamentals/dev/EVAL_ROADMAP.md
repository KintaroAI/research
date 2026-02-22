# Evaluation Roadmap

Development plan for rigorous generalization testing beyond train/val loss curves.
Each item is scoped for implementation in llm.c/CUDA with minimal dependencies.

---

## Phase 1: Cross-Domain Generalization (near-term)

We already have train/val/test splits on TinyStories. The next step is evaluating on
*different distributions* than the training data.

### 1a. Held-out domain perplexity -- DONE

Train on TinyStories, evaluate perplexity on different text distributions using
existing eval-only mode (`-n 0 -v 1 -j <heldout.bin>`). No C changes needed.

Domains: TinyShakespeare (~300K tokens), WikiText-2 test (~240K tokens).

**Data:** `python prepare_heldout.py` produces `data/heldout/shakespeare.bin`
and `data/heldout/wikitext2.bin`.

**See:** [EVAL.md](EVAL.md) Phase 1a for the full evaluation protocol.

### 1b. Compression ratio metric

Measure bits-per-byte on held-out text: `loss * tokens / bytes`. This normalizes across
tokenizers and gives a compression-theoretic view of generalization ("Language Modeling
Is Compression", Delétang et al. 2024).

**Implementation:** Store byte counts alongside token counts in the dataloader header
(or compute from the raw text file). Report bits-per-byte in addition to cross-entropy loss.

---

## Phase 2: Synthetic Algorithmic Tasks (medium-term)

Controlled tasks where ground truth is known exactly. These isolate generalization
from the noise of natural language. Very relevant for testing banded sparsity, sort layer,
and other architectural changes.

### 2a. Modular arithmetic (grokking) -- DONE

`gen_modular_data.py` generates binary data for 4 operations (add, sub, mul, sq_sum)
with 50/50 train/val split. Experiments 00010 and 00012 validated grokking on addition
and multiplication. The sequential continual-learning protocol (train all 4 tasks in
sequence, evaluate retention) is now the primary eval for architectural comparisons.

**See:** [EVAL.md](EVAL.md) Phase 2a for the full sequential evaluation protocol.

### 2b. Formal language tasks -- DONE

`gen_formal_data.py` generates binary data for 4 tasks (parity, copy, reverse,
majority) with exhaustive enumeration and 50/50 train/val split. The `-P` flag
enables multi-position loss for copy/reverse. Experiment 00015 validated standalone
learnability: copy, reverse, and majority are learnable; parity-8 does not grok
in 50k steps. The sequential eval protocol is documented in EVAL.md Phase 2b.

**See:** [EVAL.md](EVAL.md) Phase 2b for the full sequential evaluation protocol.

### 2c. In-context learning probes

Generate synthetic function-learning tasks:
- Prompt contains input-output examples of a linear function
- Model must predict the output for a new input
- Vary the number of in-context examples (0-shot, 1-shot, few-shot)

**Implementation:** More involved data generator. Measure accuracy as a function of
number of in-context examples. Even small transformers (12-layer GPT-2) show ICL
emergence — this tests whether architecture changes affect that capability.

---

## Phase 3: Scaling Laws (medium-term)

### 3a. Compute-optimal scaling curves -- PARTIAL

`create_model.py` now supports arbitrary configs (num_layers, num_heads, channels).
Still need: the actual scaling grid runs and power-law fitting.

Train a series of models varying:
- Parameters: 10M, 30M, 85M, 124M
- Data: 10M, 50M, 200M, 900M tokens (subsample TinyStories)
- Compute budget: fixed FLOPs, vary the param/data tradeoff

Fit Chinchilla-style power laws: `L(N, D) = A/N^α + B/D^β + E`

**Remaining:** Script to run the grid and collect final val losses.
Curve fitting in Python (scipy or manual least-squares).

### 3b. Per-architecture scaling comparison

For each architectural variant (dense, banded, sort layer), fit separate scaling curves.
If a variant has a better scaling exponent, it will dominate at larger scale even if it
looks worse at small scale.

**Implementation:** Reuse the grid from 3a for each variant. Plot and compare scaling
exponents. This is the most principled way to evaluate architecture changes.

---

## Phase 4: Standard Benchmarks (longer-term)

### 4a. TinyBenchmarks

Subsample MMLU, ARC, HellaSwag to ~100 examples each (per the TinyBenchmarks paper,
Polo et al. 2024). Our codebase already has an EvalLoader for HellaSwag-style
multiple-choice — extend it for other benchmarks.

**Implementation:** Python script to download and subsample benchmarks into our eval
data format. Add a `-h <file>` style flag (already exists for HellaSwag in upstream
llm.c). Report accuracy on each mini-benchmark.

### 4b. Perplexity benchmarks

Standard perplexity evaluation on:
- WikiText-103
- LAMBADA (last-word prediction)
- Penn Treebank

**Implementation:** Data download scripts producing .bin files. Evaluate with the
existing test-set evaluation loop.

---

## Priority Order

| Priority | Item | Status | Why | Effort |
|----------|------|--------|-----|--------|
| 1 | 2a. Modular arithmetic | **Done** | Direct grokking measurement, sequential eval protocol | Small |
| 2 | 1a. Held-out domain perplexity | **Done** | Easy win, just need more eval data | Small |
| 3 | 2b. Formal language tasks | **Done** | Controlled generalization measurement | Small |
| 4 | 3a. Scaling curves | Partial | Most principled architecture comparison | Medium |
| 5 | 1b. Compression ratio | Not started | Novel metric, minimal code | Small |
| 6 | 2c. ICL probes | Not started | Tests emergent capability | Medium |
| 7 | 3b. Per-architecture scaling | Not started | Requires 3a first | Medium |
| 8 | 4a. TinyBenchmarks | Not started | Standard comparison | Medium |
| 9 | 4b. Perplexity benchmarks | Not started | Standard comparison | Small |

---

## Principles

- **Synthetic tasks first.** They give cleaner signal than natural language for
  understanding architectural changes. Save NL benchmarks for final validation.
- **Always compare at matched compute.** Don't compare models that trained for different
  amounts of compute. Use matched-train-loss comparisons (already in compare_runs.py).
- **Multi-seed everything.** 3 seeds minimum (protocol in GENERALIZATION_PROTOCOL.md).
- **Stdlib-only Python.** Keep tooling dependencies minimal — no frameworks for eval
  scripts, just data generators producing .bin files.
- **Log everything.** New eval types should log to the same log file format so
  compare_runs.py can parse them.

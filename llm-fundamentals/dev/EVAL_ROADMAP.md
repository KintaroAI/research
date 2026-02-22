# Evaluation Roadmap

Development plan for rigorous generalization testing beyond train/val loss curves.
Each item is scoped for implementation in llm.c/CUDA with minimal dependencies.

---

## Phase 1: Cross-Domain Generalization (near-term)

We already have train/val/test splits on TinyStories. The next step is evaluating on
*different distributions* than the training data.

### 1a. Held-out domain perplexity

Train on TinyStories, evaluate perplexity on:
- TinyShakespeare (already have the data)
- A subset of OpenWebText or FineWeb (different register/domain)
- Simple Wikipedia or similar constrained text

**Implementation:** Add a `-z <pattern>` flag for "eval-only data" that runs a forward
pass over the given data at the end of training (like `-x` but for arbitrary held-out sets).
Generalize the test eval loop to accept multiple eval sets. Log as `s:STEP dom:NAME dml:LOSS`.

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

### 2b. Formal language tasks

- **Parity:** sequence of 0/1 tokens, predict parity of the sequence
- **Copy/reverse:** input sequence followed by separator, model must reproduce/reverse
- **Majority:** predict the most common token in a sequence

**Implementation:** Same approach — Python data generator, .bin output, task-position
masking for the answer token(s). These test whether architectural changes affect the
model's ability to learn specific algorithmic primitives.

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
| 2 | 1a. Held-out domain perplexity | Not started | Easy win, just need more eval data | Small |
| 3 | 2b. Formal language tasks | Not started | Controlled generalization measurement | Small |
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

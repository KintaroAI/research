# LLM Fundamentals

> Understanding the basics of Large Language Models through hands-on implementation.

## Objective

Build foundational understanding of transformer-based language models by implementing, training, and experimenting with them from scratch.

## Hypothesis

Hands-on implementation provides deeper understanding than reading papers alone. Starting with simple models and progressively adding complexity reveals which components are essential.

## Success Criteria

- [ ] Successfully train a GPT-style model locally
- [ ] Understand every component: embeddings, attention, FFN, layer norm
- [ ] Be able to modify and experiment with the architecture
- [ ] Document learnings that connect to our natural intelligence research

## Architecture: GPT-2 (Decoder-Only Transformer)

All experiments use a GPT-2-style decoder-only transformer. The architecture is defined in `dev/create_model.py` and executed in `dev/src/train_gpt2_fp32.cu`.

### Data Flow

```
Input token IDs: [t₀, t₁, ..., t_{T-1}]    shape: (B, T)
        │
        ▼
┌─────────────────────────────────────────┐
│  Token Embedding (wte)     (V, C)       │  look up each token → dense vector
│  + Position Embedding (wpe) (T, C)      │  add learned position encoding
│  = encoded                 (B, T, C)    │
└─────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────┐
│  Transformer Block ×L                   │  repeated L times
│  ┌───────────────────────────────────┐  │
│  │  LayerNorm (ln1)                  │  │  normalize activations
│  │  ▼                                │  │
│  │  Multi-Head Self-Attention        │  │
│  │  ├─ Linear → Q, K, V   (C→3C)    │  │  project to queries, keys, values
│  │  ├─ Split into NH heads (C/NH=HS) │  │  each head has dimension HS
│  │  ├─ Attention: softmax(Q·Kᵀ/√HS) │  │  compute attention weights (causal)
│  │  ├─ Weighted sum: att @ V         │  │  aggregate values
│  │  └─ Linear projection  (C→C)     │  │  project back
│  │  + residual connection ──────────►│  │  add input back (gradient highway)
│  │  ▼                                │  │
│  │  LayerNorm (ln2)                  │  │  normalize again
│  │  ▼                                │  │
│  │  Feed-Forward Network (MLP)       │  │
│  │  ├─ Linear   (C → 4C)            │  │  expand
│  │  ├─ GELU activation               │  │  non-linearity
│  │  └─ Linear   (4C → C)            │  │  compress back
│  │  + residual connection ──────────►│  │  add input back
│  └───────────────────────────────────┘  │
└─────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────┐
│  Final LayerNorm (lnf)     (B, T, C)    │
│  ▼                                      │
│  Output Projection (wte)   (B, T, V)    │  reuses token embedding (weight tying)
│  ▼                                      │
│  Softmax → next-token probabilities     │
└─────────────────────────────────────────┘
```

### Configurable Hyperparameters

| Symbol | Parameter | What it controls | Effect of increasing |
|--------|-----------|-----------------|---------------------|
| `C` | `n_embd` (channels) | Width of all hidden representations | More expressive per-token features. Params scale as O(C²) per layer. |
| `L` | `n_layer` | Depth — number of transformer blocks | More sequential computation steps. Each layer can refine representations further. Params scale linearly with L. |
| `NH` | `n_head` | Number of attention heads | More independent attention patterns. Each head sees C/NH dimensions. Total attention compute stays the same. |
| `T` | `block_size` (max_seq_len) | Maximum sequence length | Longer context window. Attention cost scales as O(T²). Position embeddings grow as T×C. |
| `V` | `vocab_size` | Number of distinct tokens | Larger vocabulary. Embedding table grows as V×C. |

**Derived quantities:**
- **Head dimension** `HS = C / NH` — how many features each attention head can attend to
- **FFN hidden dim** = `4 × C` — the MLP expansion factor (hardcoded to 4×)
- **Total parameters** ≈ `V×C + T×C + L×(12C² + 13C)` (dominated by the 12C² from attention + MLP weights per layer)

### What Affects What

**Capacity / Expressiveness:**
- `C` (width) is the primary capacity dial. Doubling C roughly 4× the parameters.
- `L` (depth) adds compositional ability — more layers = more sequential reasoning steps.
- `NH` (heads) controls diversity of attention patterns at fixed compute. More heads = more independent "feature detectors" but each is narrower (smaller HS).

**Compute / Memory:**
- Attention: O(B × NH × T² × HS) = O(B × T² × C) — quadratic in sequence length
- MLP: O(B × T × 8C²) per layer — quadratic in width
- Total: roughly O(B × L × T × C × (T + 8C))

**Training Dynamics:**
- Deeper models (large L) are harder to train — residual connections and layer norm are essential for gradient flow
- Wider models (large C) are easier to train but more memory-hungry
- More heads at fixed C means smaller HS, which can hurt if HS becomes too small for meaningful attention patterns

### Weight Tying

The output projection matrix that maps hidden states → logits over vocabulary **reuses the token embedding matrix** (`wte`). This means the model uses the same representation for "what does this token look like as input" and "how likely is this token as output." This reduces parameters by V×C and provides a useful inductive bias.

```python
self.lm_head.weight = self.transformer.wte.weight  # weight tying
```

In the CUDA code, this means the forward pass uses `params.wte` for both `encoder_forward` (input) and the final `matmul_forward` (output logits).

### Configurations Used

| Config | Layers | Heads | Embed | Seq Len | Vocab | Params | Use |
|--------|--------|-------|-------|---------|-------|--------|-----|
| Default (dev/) | 4 | 4 | 128 | 256 | 50,257 | ~7M | TinyStories baseline |
| Experiment 10 | 2 | 4 | 128 | 8 | 101 | ~400K | Modular arithmetic (grokking) |

### 16 Parameter Tensors (in checkpoint order)

| # | Name | Shape | Purpose |
|---|------|-------|---------|
| 1 | `wte` | (V_padded, C) | Token embeddings (also used as output projection) |
| 2 | `wpe` | (T, C) | Position embeddings |
| 3 | `ln1w` | (L, C) | Pre-attention LayerNorm weight (per layer) |
| 4 | `ln1b` | (L, C) | Pre-attention LayerNorm bias |
| 5 | `qkvw` | (L, 3C, C) | Attention QKV projection weight |
| 6 | `qkvb` | (L, 3C) | Attention QKV projection bias |
| 7 | `attprojw` | (L, C, C) | Attention output projection weight |
| 8 | `attprojb` | (L, C) | Attention output projection bias |
| 9 | `ln2w` | (L, C) | Pre-MLP LayerNorm weight |
| 10 | `ln2b` | (L, C) | Pre-MLP LayerNorm bias |
| 11 | `fcw` | (L, 4C, C) | MLP first linear weight (expand) |
| 12 | `fcb` | (L, 4C) | MLP first linear bias |
| 13 | `fcprojw` | (L, C, 4C) | MLP second linear weight (compress) |
| 14 | `fcprojb` | (L, C) | MLP second linear bias |
| 15 | `lnfw` | (C) | Final LayerNorm weight |
| 16 | `lnfb` | (C) | Final LayerNorm bias |

---

## Background

### Prior Work
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — Vaswani et al., 2017 (original transformer)
- [GPT-2](https://openai.com/research/better-language-models) — Radford et al., 2019
- [nanoGPT](https://github.com/karpathy/nanoGPT) — Karpathy's minimal GPT implementation

### Key Concepts
- **Tokenization**: Converting text to integers
- **Embeddings**: Dense vector representations of tokens
- **Self-Attention**: Computing relationships between tokens (O(n²) complexity)
- **Causal Masking**: Preventing tokens from seeing the future
- **Layer Normalization**: Stabilizing training
- **Residual Connections**: Enabling gradient flow

## Setup

### Requirements
- Hardware: GPU recommended (RTX 4090 available)
- Python 3.12+
- PyTorch 2.x with CUDA

### Installation
```bash
pip install torch numpy
```

## Progress

**State (2026-11-25):** parked since 00021 (2026-11-12). 22 experiment directories, 00001 to 00021 (two share the number 00005). The CUDA trainer is in `dev/` (build notes: `../CLAUDE.md`), the fact-memory code in `dev/facts/`.

| Experiment | Status | Key Finding |
|------------|--------|-------------|
| [00001-karpathy-gpt-local](./00001-karpathy-gpt-local/) | Complete | Char-level GPT (210K params) learns Shakespeare structure in ~3 min on RTX 4090 |
| [00002-llama2c-shakespeare](./00002-llama2c-shakespeare/) | Complete | 870K param Llama2 with custom BPE reaches 1.79 train loss; 1401 tok/s C inference |
| [00003-llmc-full-c](./00003-llmc-full-c/) | Complete | Pure C training pipeline; GPT-2 124M finetune val loss 5.33 to 4.29 in 40 steps |
| [00004-cuda-probe](./00004-cuda-probe/) | Complete | RTX 4090 verified: 128 SMs, 23.5 GB VRAM, ~255B elem/s throughput |
| [00005-from-scratch-cuda](./00005-from-scratch-cuda/) | Complete | 7.3M param model trained from scratch in C/CUDA (581K tok/s) |
| [00005-small-model-gpu](./00005-small-model-gpu/) | No status in its README | The same 7.3M param model with a patched llm.c trainer (`-e` flag): loss 10.93 to 8.84 in one epoch (~300 steps), output still random |
| [00006-tinystories-e2e](./00006-tinystories-e2e/) | Complete | Self-contained CUDA trainer on TinyStories; loss ~10.5 to 4.5 after 10K steps |
| [00007-banded-sparsity](./00007-banded-sparsity/) | Complete | Diagonal band on FC1 maintains quality at ~44% density (bandwidth 256-512) |
| [00008-banded-tinystories](./00008-banded-tinystories/) | Complete | Banded FC1 sweet spot at 25-45% density; below 15% fails |
| [00009-banded-fc-comparison](./00009-banded-fc-comparison/) | Complete | FC2 sparsity improves val loss (-2.4% at BW=256); FC1 hurts even at BW=1024 |
| [00010-delayed-generalization](./00010-delayed-generalization/) | Complete | Grokking reproduced on mod 97 with answer-only loss masking (-p 4) |
| [00011-sort-layer](./00011-sort-layer/) | Complete | Learnable correlation-based blending gives +1.1% quality on TinyStories; +40% overhead |
| [00012-grokking-conditions](./00012-grokking-conditions/) | Complete | No delayed generalization; -p 4 dominates; wd=0 memorizes, wd=1 generalizes instantly |
| [00013-wandb-integration](./00013-wandb-integration/) | Complete | Python wrapper logs all training metrics to W&B in real-time; zero C code changes |
| [00014-sequential-modular-eval](./00014-sequential-modular-eval/) | Complete | Sequential eval baseline: dense model shows total catastrophic forgetting (loss ~10 vs random 4.58) |
| [00015-formal-language-tasks](./00015-formal-language-tasks/) | Complete | Formal language tasks for the sequential eval: copy-8, reverse-8 and majority-9 learn (val loss 0.03–0.07, 0.03–0.06, ~0.0001); parity-8 stays at the coin-flip 0.693 in 50k steps |
| [00016-held-out-domain-perplexity](./00016-held-out-domain-perplexity/) | Complete | Held-out domain perplexity with the existing eval-only mode: a TinyStories model at 5k steps scores perplexity 7.7 in domain, 2,237 on TinyShakespeare, 10,030 on WikiText-2 |
| [00017-embedding-blend-layer](./00017-embedding-blend-layer/) | Complete | Position-based embedding blend layer (a causal 1D convolution after the embedding): neutral to slightly negative at 51M params (+0.005 val loss at 50k steps); learns a bigram-like kernel; <1% overhead |
| [00018-hebbian-embedding-pull](./00018-hebbian-embedding-pull/) | Complete | Hebbian pull on the embedding matrix (`-H 4 -u 1e-5`) hurts val loss by +0.066 at 50k steps, throughout: a non-learnable pull works against gradient descent |
| [00019-three-way-comparison](./00019-three-way-comparison/) | Complete | Baseline, blend-G8 and Hebbian-H4 back to back: blend −0.018 val loss (1.304 vs 1.322), Hebbian +0.037; the ranking against 00017 is sensitive to run conditions. Follow-ups: a Hebbian window of 8 is a sharp sweet spot at 512-dim only (gone at 768-dim; at 1024-dim it diverges by 50k steps); at 355M the blend's lead was a missing LR warmup (with warmup the baseline reaches 1.240 at 20k) |
| [00020-fact-memorization](./00020-fact-memorization/) | Complete | Invented facts in a TinyStories stream, five sizes (0.4M to 85M non-embedding params), two seeds: showings needed fall from a median of 21 to 2 (GPT-2 small); every size forgets within a few thousand steps; relearning takes about half the showings of a fresh fact; recall only in the trained wording, never in reverse |
| [00021-fact-retention](./00021-fact-retention/) | Complete | What keeps a fact (sizes L and G, two seeds): spacing does not help under Adam (8 showings massed in one step are kept longest, 10 steps apart worst); a lower or decaying learning rate slows forgetting, and only the decaying rate in G holds more facts at the end (3,006 against 1,874); four training wordings give no recall in a new one |

## Improvement Ideas: Encouraging Delayed Generalization

Based on [Generalization Beyond Overfitting on Small Algorithmic Datasets](https://mathai-iclr.github.io/papers/papers/MATHAI_29_paper.pdf) (Power et al., ICLR 2021 MathAI Workshop):

- **Weight decay (AdamW, wd=1)** — the single most effective technique; dramatically improves data efficiency and generalization speed
- **Weight decay towards initialization** — variant where weights are regularized towards their initial values rather than the origin
- **Dropout (residual dropout = 0.1)** — helps but less impactful than weight decay
- **Gaussian weight noise (σ = 0.01)** — injecting noise into parameters during forward pass provides regularization
- **Increasing training data fraction** — more data consistently reduces the optimization steps needed to generalize (roughly exponential relationship)
- **Longer optimization budgets** — delayed generalization can take 100–1000× more steps than memorization; training longer is essential
- **Learning rate tuning** — suboptimal hyperparameters severely limit generalization
- **SGD-based optimizers (Adam/AdamW) over full-batch methods** — mini-batch stochasticity helps find generalizing solutions

## Open Questions

- How does attention complexity relate to thalamic sorting?
- Can we visualize attention patterns as "correlation maps"?
- What's the minimum viable architecture for coherent text?
- Can we observe delayed generalization in our small-scale experiments?

## References

- Karpathy, A. "Let's build GPT: from scratch, in code, spelled out." YouTube, 2023.
- Vaswani, A., et al. "Attention Is All You Need." NeurIPS, 2017.

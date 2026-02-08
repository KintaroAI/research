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

| Experiment | Date | Status | Key Finding |
|------------|------|--------|-------------|
| [00001-karpathy-gpt-local](./00001-karpathy-gpt-local/) | 2026-02-03 | 🔬 In Progress | Baseline implementation |

## Improvement Ideas: Encouraging Generalization (Grokking)

Based on [Grokking: Generalization Beyond Overfitting on Small Algorithmic Datasets](https://mathai-iclr.github.io/papers/papers/MATHAI_29_paper.pdf) (Power et al., ICLR 2021 MathAI Workshop):

- **Weight decay (AdamW, wd=1)** — the single most effective technique; dramatically improves data efficiency and generalization speed
- **Weight decay towards initialization** — variant where weights are regularized towards their initial values rather than the origin
- **Dropout (residual dropout = 0.1)** — helps but less impactful than weight decay
- **Gaussian weight noise (σ = 0.01)** — injecting noise into parameters during forward pass provides regularization
- **Increasing training data fraction** — more data consistently reduces the optimization steps needed to generalize (roughly exponential relationship)
- **Longer optimization budgets** — grokking can take 100–1000× more steps than memorization; training longer is essential
- **Learning rate tuning** — suboptimal hyperparameters severely limit generalization
- **SGD-based optimizers (Adam/AdamW) over full-batch methods** — mini-batch stochasticity helps find generalizing solutions

## Open Questions

- How does attention complexity relate to thalamic sorting?
- Can we visualize attention patterns as "correlation maps"?
- What's the minimum viable architecture for coherent text?
- Can we observe grokking in our small-scale experiments?

## References

- Karpathy, A. "Let's build GPT: from scratch, in code, spelled out." YouTube, 2023.
- Vaswani, A., et al. "Attention Is All You Need." NeurIPS, 2017.

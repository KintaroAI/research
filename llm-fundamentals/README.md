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

## Open Questions

- How does attention complexity relate to thalamic sorting?
- Can we visualize attention patterns as "correlation maps"?
- What's the minimum viable architecture for coherent text?

## References

- Karpathy, A. "Let's build GPT: from scratch, in code, spelled out." YouTube, 2023.
- Vaswani, A., et al. "Attention Is All You Need." NeurIPS, 2017.

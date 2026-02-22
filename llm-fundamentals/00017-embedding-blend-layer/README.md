# Experiment: Position-Based Embedding Blend Layer

**Date:** 2026-02-22
**Status:** Complete
**W&B:** [50m-baseline](https://wandb.ai/kintaroai-dot-com/gpt2-cuda/runs/vjfdbl48) | [50m-blend-G8](https://wandb.ai/kintaroai-dot-com/gpt2-cuda/runs/qypsrz8u)

## Goal

Test whether a learnable position-based embedding blend layer — a causal 1D
convolution applied once after `encoder_forward` — improves language modeling
on TinyStories. The layer blends nearby token embeddings using learned distance
weights and a learned mixing strength, with a residual connection.

## Hypothesis

Tokens close together in context carry related information. A position-based
blend layer that lets nearby embeddings "pull" toward each other before entering
the transformer stack should provide useful local context that complements
attention's global context. This could improve val loss compared to a baseline
without the layer.

## Method

### Architecture

A single blend layer inserted after `encoder_forward` (wte + wpe) and before
the first transformer block. No changes to the transformer layers themselves.

**Math:**
```
w = softmax(w_raw[0..W-1])          // learned distance weights
alpha = sigmoid(alpha_raw)           // learned mixing strength

blend[t,c] = Σ_{d=0}^{min(W-1, t)} w[d] * encoded[t-d, c]    // causal weighted avg
out[t,c]   = (1 - alpha) * encoded[t,c] + alpha * blend[t,c]  // residual mix
```

- Causal: position t only sees positions <= t
- W = 8 (window size, set via `-G 8`)
- Init: w_raw = 0 (uniform softmax), alpha_raw = -2.0 (sigmoid = 0.12, near-identity)

**Implementation:**
- Two CUDA kernels: `embed_blend_forward_kernel` and `embed_blend_backward_kernel`
- One warp per (b, t) position, shared memory for softmax weights
- Backward uses atomicAdd for gradient scatter across overlapping windows
- Softmax/sigmoid computed on CPU, weights copied to GPU scratch buffer
- Sidecar checkpoint (`.blend` file) for blend parameters
- Separate AdamW optimizer at 10x learning rate, no weight decay
- CLI flag: `-G <window>` (0 = disabled)

### Training setup

| Parameter | Value |
|-----------|-------|
| Model | 8-layer, 8-head, 512-dim GPT-2 (51M params) |
| Data | TinyStories (906M train tokens, 19M val tokens) |
| Batch size | 8 |
| Seq length | 512 |
| Steps | 50,000 |
| Learning rate | 3e-4 (main), 3e-3 (blend params) |
| Checkpoint | every 10k steps |
| Seed | 0 |

Two runs: baseline (no blend) and blend with `-G 8`, same initial weights
(`model_50m.bin`), sequential on a single GPU.

## Results

### Val loss comparison

| Step | Baseline | Blend G8 | Delta |
|------|----------|----------|-------|
| 0 | 10.943 | 10.945 | +0.002 |
| 5000 | 1.856 | 1.945 | +0.089 |
| 10000 | 1.620 | 1.671 | +0.051 |
| 15000 | 1.521 | 1.556 | +0.035 |
| 20000 | 1.460 | 1.480 | +0.020 |
| 25000 | 1.417 | 1.430 | +0.014 |
| 30000 | 1.381 | 1.391 | +0.011 |
| 35000 | 1.354 | 1.364 | +0.011 |
| 40000 | 1.330 | 1.337 | +0.008 |
| 45000 | 1.310 | 1.316 | +0.006 |
| **50000** | **1.297** | **1.302** | **+0.005** |

Blend is consistently behind baseline throughout training. The gap narrows from
+0.089 at 5k steps to +0.005 at 50k steps but never closes.

### Throughput

| Run | tok/s | ms/step |
|-----|-------|---------|
| Baseline | 120,018 | 34.1 |
| Blend G8 | 119,355 | 34.3 |

Negligible overhead (~0.6% slower) — the blend kernels are very lightweight
compared to attention and MLP.

### Learned parameters (at 50k steps)

| Parameter | Init | Final |
|-----------|------|-------|
| alpha | 0.12 | **0.37** |
| w[0] (self) | 0.125 | **0.015** |
| w[1] (t-1) | 0.125 | **0.718** |
| w[2] (t-2) | 0.125 | **0.179** |
| w[3] (t-3) | 0.125 | **0.053** |
| w[4] (t-4) | 0.125 | 0.018 |
| w[5] (t-5) | 0.125 | 0.007 |
| w[6] (t-6) | 0.125 | 0.005 |
| w[7] (t-7) | 0.125 | 0.005 |

Key observations:
- Alpha tripled from 0.12 to 0.37 — the model uses the layer, not ignoring it
- w[1] (nearest neighbor) dominates at 72% — the model primarily blends with
  the immediately preceding token
- w[0] (self) collapsed to 1.5% — redundant with the (1-alpha) residual path
- w[2] gets 18%, w[3] gets 5%, everything else near zero
- The learned kernel is effectively a bigram smoother: 63% original + 27% t-1 + 7% t-2

## Analysis

The blend layer learns a meaningful, interpretable pattern but does not improve
val loss on TinyStories. Several factors explain this:

1. **Redundancy with attention.** The first transformer layer's self-attention
   already computes arbitrary weighted averages over all preceding positions.
   A fixed position-based blend before attention provides information that
   attention can learn to extract itself. The blend layer is strictly less
   expressive than a single attention head.

2. **The learned pattern is a bigram smoother.** The dominant behavior — 72%
   weight on t-1 — is equivalent to adding a fraction of the previous token's
   embedding. Attention heads in layer 0 can (and likely do) learn this pattern
   with content-dependent gating, making the position-only version redundant.

3. **Additional parameters slow early training.** The blend layer starts with
   alpha = 0.12, which perturbs embeddings from step 0. The model must learn
   blend weights AND adjust downstream layers to the perturbed input, creating
   a slight optimization burden that shows up as the early gap (+0.089 at 5k).

4. **Near-zero overhead validates the implementation.** The <1% throughput cost
   confirms the kernels are efficient and the layer doesn't bottleneck training.

5. **Alpha not collapsing to zero is informative.** The model chose alpha = 0.37
   rather than driving it to zero, meaning the blend provides signal that the
   model uses. But this signal is apparently already available through attention,
   so the overall effect is neutral.

## Conclusions

- Position-based embedding blend layer implemented and validated end-to-end
  (forward, backward, AdamW, checkpoint save/load, inference)
- On TinyStories at 51M params, blend `-G 8` is neutral to slightly negative
  (+0.005 val loss at 50k steps)
- The layer learns an interpretable bigram-like kernel (72% nearest neighbor)
- Near-zero throughput overhead (<1%)
- The information provided by position-only blending is redundant with what
  attention already learns

## Next Steps

- [ ] Test on grokking tasks where local positional structure matters more
- [ ] Try content-based blend (using dot-product similarity instead of position)
- [ ] Experiment with per-layer blend (applied after each transformer block)
- [ ] Test with larger window sizes (G=16, G=32) on longer sequences
- [ ] Try on formal language tasks where positional patterns are more structured

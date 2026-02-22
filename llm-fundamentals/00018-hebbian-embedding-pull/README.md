# Experiment: Hebbian Embedding Pull

**Date:** 2026-02-22
**Status:** Complete
**Source:** *tagged on completion as `exp/00018`*
**W&B:** [hebbian-pull-H4](https://wandb.ai/kintaroai-dot-com/gpt2-cuda/runs/p219fk1x)

## Goal

Test whether a Hebbian co-occurrence force on the wte embedding matrix —
pulling embeddings of nearby tokens toward each other before each forward pass —
improves language modeling on TinyStories.

## Hypothesis

Backpropagation naturally clusters embeddings of tokens that appear in similar
contexts. A local Hebbian force that directly pulls co-occurring tokens'
embeddings toward each other should accelerate this clustering, helping
convergence early. However, it might over-cluster later, preventing the model
from learning fine-grained distinctions between tokens.

## Method

### Mechanism

```
For each position (b, t) in the batch:
  For each neighbor at distance d in [1, min(W, t)]:
    if token_t != token_{t-d}:
      delta = (eps / d) * (wte[token_{t-d}] - wte[token_t])
      wte[token_t]   += delta     // pull self toward neighbor
      wte[token_{t-d}] -= delta   // pull neighbor toward self (symmetric)
```

- **Symmetric:** both tokens in a pair move toward their midpoint
- **Distance-weighted:** 1/d decay — nearer tokens pull harder
- **Skip same-token:** pulling a vector toward itself is a no-op
- **Applied before `encoder_forward`:** model trains on pulled embeddings
- **Training only:** not applied during val/inference
- **No learned parameters** — just hyperparameters (eps, window)
- **No new activation buffers** — modifies wte in-place
- **Weight tying note:** wte == lm_head, so pulling embeddings together also
  makes output logits for co-occurring tokens more similar

### Implementation

- Single CUDA kernel: `embed_pull_kernel` — one thread per (b, t, c) element
- Uses `atomicAdd` for concurrent writes to shared wte rows
- Called before `gpt2_forward` in the training loop only
- Inputs copied to GPU before pull (redundant copy inside `gpt2_forward` is harmless)
- CLI flags: `-H <int>` (window), `-u <float>` (epsilon)

### Training setup

| Parameter | Value |
|-----------|-------|
| Model | 8-layer, 8-head, 512-dim GPT-2 (51M params) |
| Data | TinyStories (906M train tokens, 19M val tokens) |
| Batch size | 8 |
| Seq length | 512 |
| Steps | 50,000 |
| Learning rate | 3e-4 |
| Hebbian window | 4 (`-H 4`) |
| Hebbian epsilon | 1e-5 (`-u 1e-5`) |
| Seed | 0 |

Same initial weights (`model_50m.bin`) and data as experiment 00017 baseline.

## Results

### Val loss comparison

| Step | Baseline | Hebbian H4 | Delta |
|------|----------|------------|-------|
| 0 | 10.943 | 10.943 | +0.000 |
| 5000 | 1.856 | 2.011 | +0.155 |
| 10000 | 1.620 | 1.731 | +0.111 |
| 15000 | 1.521 | 1.604 | +0.084 |
| 20000 | 1.460 | 1.531 | +0.071 |
| 25000 | 1.417 | 1.484 | +0.067 |
| 30000 | 1.381 | 1.443 | +0.062 |
| 35000 | 1.354 | 1.417 | +0.063 |
| 40000 | 1.330 | 1.387 | +0.057 |
| 45000 | 1.310 | 1.378 | +0.068 |
| **50000** | **1.297** | **1.363** | **+0.066** |

Hebbian pull is consistently behind baseline throughout training. Unlike the
blend layer (exp 00017) where the gap narrowed over time, the Hebbian pull gap
stays roughly constant at +0.06–0.07 from step 20k onward.

### Throughput

| Run | tok/s | ms/step |
|-----|-------|---------|
| Baseline | 120,018 | 34.1 |
| Hebbian H4 | 119,848 | 34.2 |

Negligible overhead (~0.1% slower) — the pull kernel is very lightweight.

## Analysis

The Hebbian pull hurts val loss by a consistent margin (+0.066 at 50k steps),
worse than the blend layer's +0.005. Several factors explain this:

1. **Over-smoothing the embedding space.** The pull force moves co-occurring
   tokens' embeddings toward each other every step. Over 50k steps, this
   accumulates into significant smoothing that prevents the model from
   maintaining fine-grained distinctions. Tokens that co-occur frequently
   (e.g., common words like "the", "a", "was") get pulled toward many
   different neighbors, creating a muddy average embedding.

2. **Interference with backpropagation.** The pull modifies wte *before* the
   forward pass, but backpropagation's gradient update to wte happens *after*.
   These two forces may conflict: backprop tries to push embeddings apart for
   discrimination, while Hebbian pull pushes them together for co-occurrence.
   The net effect is wasted gradient signal.

3. **The gap stabilizes rather than closing.** The blend layer's gap shrank
   from +0.089 to +0.005 over 50k steps as the model learned to compensate.
   The Hebbian pull gap stays flat at ~+0.066, suggesting the model cannot
   compensate because the pull continuously distorts wte every step. The
   damage is ongoing, not a one-time perturbation.

4. **Weight tying amplifies the effect.** Since wte == lm_head, pulling
   embeddings together also makes the output logit distribution flatter for
   co-occurring tokens, directly harming the model's ability to distinguish
   between likely next tokens.

5. **No learned parameters means no escape.** Unlike the blend layer which
   could theoretically learn alpha → 0 to disable itself, the Hebbian pull
   has no learnable gate. The model cannot turn it off when it becomes
   counterproductive.

## Conclusions

- Hebbian embedding pull (`-H 4 -u 1e-5`) hurts val loss by +0.066 at 50k steps
- The effect is consistently negative — no early benefit followed by late harm
  as hypothesized; it simply hurts throughout
- Near-zero throughput overhead (~0.1%)
- The persistent, non-learnable nature of the pull prevents the model from
  compensating, unlike the blend layer which partially adapted
- Direct weight modification outside the computation graph appears fundamentally
  at odds with gradient-based optimization on this task

## Next Steps

- [ ] Try much smaller epsilon (1e-7, 1e-8) to see if a gentler pull helps
- [ ] Try decaying epsilon over training (strong early, zero late)
- [ ] Try applying pull only for first N steps then stopping
- [ ] Try content-based pull (similarity-gated) instead of pure co-occurrence

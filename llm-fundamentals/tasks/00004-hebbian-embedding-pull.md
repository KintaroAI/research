# Task 00004: Hebbian Embedding Pull

**Date:** 2026-02-22
**Status:** Complete

## Context

Experiment 00017 showed that the learnable blend layer (position-based causal
convolution on activations) is neutral on TinyStories — it learns an interpretable
bigram kernel but doesn't improve val loss, likely because attention already
captures the same information.

This task takes a fundamentally different approach: instead of blending
**activations**, we directly modify the **wte weight matrix** based on token
co-occurrence. The idea is that backpropagation naturally clusters embeddings of
tokens that appear in similar contexts, and we can accelerate this by applying a
local Hebbian force: tokens that appear near each other in the input get their
embeddings pulled toward each other.

This force is **independent of backpropagation** — no gradients, no learned
parameters, just a hyperparameter-controlled direct weight modification. The
hypothesis: it helps convergence early (faster clustering) but might over-cluster
and hurt later (preventing fine distinctions).

## Mechanism

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

## Design

- **No learned parameters** — just hyperparameters (eps, window)
- **No new activation buffers** — modifies wte in-place
- **No checkpoint/sidecar** — the pull is transient, effects accumulate in wte
- **No backward changes** — not part of computation graph
- **No changes to generate.cu** — inference-only, no pulling
- **Weight tying note:** wte == lm_head, so pulling embeddings together also
  makes output logits for co-occurring tokens more similar

## Changes — `src/train_gpt2_fp32.cu`

### 1. Globals (near line 84, after `EMBED_BLEND_WINDOW`)

```c
int EMBED_PULL_WINDOW = 0;      // 0 = disabled
float EMBED_PULL_EPS = 1e-5f;   // pull strength
```

### 2. Kernel + launcher (near line 132, after `apply_mask_kernel`)

Simple flat kernel — one thread per (b, t, c) element:

```c
__global__ void embed_pull_kernel(float* wte, const int* inputs,
                                  float eps, int B, int T, int C, int W) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * T * C) return;
    int bt = idx / C;
    int c  = idx % C;
    int t  = bt % T;
    int self_tok = inputs[bt];
    for (int d = 1; d <= W && d <= t; d++) {
        int nbr_tok = inputs[bt - d];
        if (nbr_tok == self_tok) continue;
        float delta = (eps / (float)d) * (wte[nbr_tok * C + c] - wte[self_tok * C + c]);
        atomicAdd(&wte[self_tok * C + c], delta);
        atomicAdd(&wte[nbr_tok * C + c], -delta);
    }
}
```

Launcher: grid = CEIL_DIV(B*T*C, 256), block = 256.

### 3. Training loop integration (line ~3107)

Apply pull OUTSIDE `gpt2_forward`, only during training, before the forward call:

```c
dataloader_next_batch(&train_loader);
if (EMBED_PULL_WINDOW > 0) {
    cudaCheck(cudaMemcpy(model.inputs, train_loader.inputs,
              B * T * sizeof(int), cudaMemcpyHostToDevice));
    embed_pull(model.params.wte, model.inputs, EMBED_PULL_EPS,
              B, T, C, EMBED_PULL_WINDOW);
}
gpt2_forward(&model, train_loader.inputs, train_loader.targets, B, T);
```

`model.inputs` is allocated by the val forward call at step 0, which runs
before the training forward. The redundant memcpy inside `gpt2_forward` is
harmless (same data).

### 4. CLI flags

- `-H <int>` — EMBED_PULL_WINDOW (H for Hebbian)
- `-u <float>` — EMBED_PULL_EPS

Add to `error_usage`, arg parsing (line ~2891), and parameter print table.

## Files

| File | Action |
|------|--------|
| `src/train_gpt2_fp32.cu` | Add kernel, globals, CLI, training loop integration |

## Verification

1. `make all` compiles
2. `./train -H 4 -u 1e-5 -n 50` — loss decreases normally
3. `./train -H 4 -u 1e-3 -n 20` — large eps to confirm pull has visible effect
4. 50k-step comparison: baseline vs `-H 4 -u 1e-5` on 50M model

# Task 00003: Position-Based Embedding Blend Layer

**Date:** 2026-02-22
**Status:** Done

## Context

Add a learnable layer right after `encoder_forward` that blends nearby token
embeddings based on their relative position. The idea: tokens close together in
context should be able to "pull" their embeddings toward each other. This is a
position-only causal 1D convolution with a residual connection, where the blend
weights and mixing strength are learned through backpropagation.

Unlike the existing sort layer (content-based similarity, applied per-layer after
MLP), this layer uses only positional distance and is applied once, right after
the embedding lookup (wte + wpe).

## Math

```
w = softmax(w_raw[0..W-1])          // learned distance weights
alpha = sigmoid(alpha_raw)           // learned mixing strength

// For each position t, channel c:
blend[t,c] = Σ_{d=0}^{min(W-1, t)} w[d] * encoded[t-d, c]    // causal weighted avg
out[t,c]   = (1 - alpha) * encoded[t,c] + alpha * blend[t,c]
```

- `w[0]` = self-weight (distance 0), `w[1]` = nearest neighbor, etc.
- Causal: position t only sees positions ≤ t
- At init, alpha_raw = -2.0 → alpha ≈ 0.12, so the layer is near-identity
- w_raw initialized uniform → equal weight to all distances

**Backward:**
1. `d_encoded[t,c] += (1-alpha) * dout[t,c]`  (residual path)
2. `d_encoded[t-d,c] += alpha * w[d] * dout[t,c]`  (blend path, scatter)
3. `d_w_raw` via softmax backward from `d_w[d] = alpha * Σ_{t,c} encoded[t-d,c] * dout[t,c]`
4. `d_alpha_raw = sigmoid_deriv * Σ_{t,c} (blend[t,c] - encoded[t,c]) * dout[t,c]`

## Design Decisions

- **Scope: once after embeddings, not per-layer.** The user's intent is to modify
  embeddings before they enter the transformer stack. Per-layer blending is what
  the sort layer already does.
- **Position-only first.** Content-based similarity can be added later behind a
  separate flag. Position is more valuable since similarity already emerges from
  attention.
- **Separate from sort layer.** Different flag (`-G`), different parameter storage,
  can be used independently or together.
- **Sidecar checkpoint.** Like sort layer, blend params saved as `.blend` file
  alongside the main checkpoint to preserve format compatibility.

## Changes

### 1. `src/train_gpt2_fp32.cu`

**New global variable:**
```c
int EMBED_BLEND_WINDOW = 0;   // 0 = disabled
```

**New kernels (2 functions):**

`embed_blend_forward_kernel` — one warp per (b, t) position:
- Read softmax weights `w[W]` and `alpha` from args (precomputed on CPU)
- Compute causal weighted average over window
- Write `out[t] = (1-alpha)*inp[t] + alpha*blend[t]`
- Much simpler than sort layer — no dot products, no per-position attention

`embed_blend_backward_kernel` — one warp per (b, t) position:
- Residual gradient: `dinp[t] += (1-alpha) * dout[t]`
- Blend scatter: for each d in window, `dinp[t-d] += alpha * w[d] * dout[t]`
  (atomicAdd for overlapping writes across warps)
- Parameter gradients: accumulate `d_w[d]` and `d_alpha_raw` via atomicAdd

**New activation buffer:**
- `acts.embed_blend` — (B, T, C), conditionally allocated when EMBED_BLEND_WINDOW > 0
- Add to `ActivationTensors` struct and `fill_in_act_sizes`

**New parameter storage (sidecar, like sort layer):**
- `embed_blend_params_memory` — W+1 floats on GPU: w_raw[0..W-1], alpha_raw
- `embed_blend_grads_memory` — W+1 floats
- `embed_blend_m_memory`, `embed_blend_v_memory` — W+1 floats each (AdamW)
- Add fields to `GPT2` struct
- `init_embed_blend()` — allocate, init w_raw=0 (uniform softmax), alpha_raw=-2.0

**Forward pass integration (after encoder_forward, before layer loop):**
```c
if (EMBED_BLEND_WINDOW > 0) {
    // Compute w = softmax(w_raw) and alpha = sigmoid(alpha_raw) on CPU
    // Call embed_blend_forward(acts.embed_blend, acts.encoded, w, alpha, B, T, C, W)
    // Layer 0 residual = acts.embed_blend instead of acts.encoded
}
```

**Backward pass integration (after layer loop unwinds to layer 0):**
```c
if (EMBED_BLEND_WINDOW > 0) {
    // dresidual holds grad w.r.t. embed_blend
    // Copy to scratch, zero dresidual
    // embed_blend_backward accumulates grad w.r.t. encoded into dresidual
    // Then encoder_backward uses dresidual
}
```

**AdamW update:** Add embed_blend params to the update step (like sort layer).

**Checkpoint save/load:** Save/load `.blend` sidecar file (header: [window, W+1 params]).

**CLI flag:** `-G <window>` to set EMBED_BLEND_WINDOW.

### 2. `src/generate.cu`

Add embed_blend forward pass for inference (same pattern as sort layer in generate).

## Files

| File | Action |
|------|--------|
| `llm-fundamentals/dev/src/train_gpt2_fp32.cu` | Add blend kernels, params, forward/backward/update integration |
| `llm-fundamentals/dev/src/generate.cu` | Add blend forward for inference |

## Verification

1. **Build:** `make all` compiles without errors
2. **Identity test:** Train with `-G 8` for a few steps — loss should decrease
   normally (layer starts near-identity with alpha ≈ 0.12)
3. **Gradient check:** Compare numerical vs analytical gradients for w_raw and
   alpha_raw on a small model (manual spot-check or add to test suite)
4. **Parameter evolution:** Print blend params periodically during training to
   verify they're being updated (alpha should move from init, w_raw should
   develop non-uniform pattern favoring nearby positions)
5. **Comparison:** Train TinyStories with and without `-G 8`, compare val loss
   curves at matched compute

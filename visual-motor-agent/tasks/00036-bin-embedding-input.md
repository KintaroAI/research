# Task 00036: Bin-Embedding Input for the Universal Model (experiment vm-00030)

**Date:** 2026-10-26
**Status:** Complete (results in vm-00030)

## Context

- vm-00029: the universal model's gap to a classical GPT on text is its input map. `W_in` is linear, so a value
  enters as value × one vector; an embedding table closes the gap, and so does the universal version, **bin
  embeddings**: each value split two-hot over the output's bins, each (value slot, bin) with its own learned
  embedding (1.63 nats per letter against 1.93–2.00; with a small initialisation 1.57, better than GPT's 1.62).
- For letters on bin centres it is a lookup table; for pixels and hunger a learned piecewise-linear function of the
  value; `W_in` is the special case of embeddings that grow linearly with the bin.

## Design (`model.in_bins`, default 0 = unchanged)

- **The input as a wider linear map:** each token's used values are expanded two-hot over the K bins (`bins`, the
  output's), unused values 0: a vector of V · K, with at most two non-zeros per value. `W_in` becomes d × (V · K); the
  token's embedding is `W_in · twohot(values) + b`, i.e. the sum over its values of (1 − w) E[v, lo] + w E[v, lo + 1].
  Forward and backward are the existing linear map's, on the CPU and the GPU; only the expansion is new.
- **Initialisation:** `W_in` at 0.02 (GPT-2's scale, the same as the slot and step embeddings; vm-00029's R8), instead
  of √(3 / V).
- Every place the input map is used: the CPU sequence forward and backward (`vm_agent.cpp`), the GPU's `run_uni` and
  the instinct's candidates (`vm_cuda.cu`), the PyTorch reference (`torch_reference.forward_uni`), the config
  (`VMConfig.in_bins`, header slot 43: 0 in older checkpoints).
- Cost: V · K · d parameters (vision: 64 · 32 · 128 = 262k; text arm B: 4 · 256 · 128 = 131k); the expanded input is
  rows × V · K floats (vision, a training batch: 4,224 × 2,048).

## Verification

- The PyTorch reference: its two-hot input equals a hand computation (the sum of the bin embeddings).
- C++ against the reference: forward values and gradients with `in_bins` (rgb and gray), as the existing universal
  tests do; the GPU against the C++ model (forward, update, candidates); old checkpoints still load (in_bins 0).

## Experiment vm-00030

1. The text lifetime, arm B with `in_bins` (3 seeds, 100k ticks) against vm-00026's B (1.80) and the classical GPT
   (1.615).
2. The circle world: the default universal lifetime (vm-00021's configuration, 100k ticks, 3 seeds) with `in_bins`
   against vm-00022's U (the same without): the frozen click test on real windows and forage test, online clicks, the
   patch and body cross-entropies.

## Results (vm-00030)

- Implemented and tested (the reference, CPU, GPU; mutation check on the GPU expansion; the full suite: 1,069 passed).
- Text lifetime: best 1.72–1.75 nats per letter (linear input: 1.80; the classical GPT on the same data: 1.70–1.72),
  then overfitting on the replayed corpus.
- Circle world: patch and body cross-entropy lower in all seeds, the click test better in two of three; a narrower
  policy (move cross-entropy 0.17–0.23 against 0.39–0.67), fewer poison clicks.

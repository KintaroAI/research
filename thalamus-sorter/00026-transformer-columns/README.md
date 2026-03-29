# ts-00026: Transformer Columns

**Date:** 2026-03-29
**Status:** In progress
**Source:** `exp/ts-00026`
**Depends on:** ts-00025 (lateral connections, visual field, blocks, cluster cap)

## Goal

Replace ConscienceColumn with a transformer-based column that learns temporal patterns via self-supervised next-frame prediction. Each column is its own tiny transformer — no shared weights.

## Motivation

ConscienceColumn collapses the signal window to a temporal mean, losing all temporal structure. The conscience threshold prevents winner collapse but causes artificial winner rotation that disrupts motor output. A transformer column can:

1. **Attend over time** — learn "input 5 spiked 3 ticks ago AND input 12 is high now → output 2"
2. **Self-supervised training** — predict next input frame, no labels needed
3. **Natural surprise signal** — prediction error measures novelty
4. **No collapse hack** — prediction pressure drives category differentiation naturally

## Architecture

```
Per column (m independent instances, batched for GPU):

Input: (window, max_inputs) — time series of wired neuron signals
  ↓
Encoder: 1-layer transformer, self-attention over time steps
  ↓ hidden state h (from last time step or pooled)
  ↓
Head 1 (predict): h → (max_inputs,)  — predict next input frame
Head 2 (categorize): h · category_embeddings → softmax → (n_outputs,)

Loss: MSE(predicted_frame, actual_next_frame)
Surprise: prediction error magnitude
```

### Key design decisions

- **One transformer per column** — each column sees different neurons, needs its own temporal model
- **Time as sequence** — tokens are time steps, features are input neuron values
- **Category embeddings** — n_outputs learned vectors, categorization via dot product with hidden state
- **Online SGD** — one forward+backward per tick on the latest window
- **~2K params per column** — 1 layer, 2 heads, dim=max_inputs

### Interface

Same as ColumnBase: `wire()`, `unwire()`, `tick()`, `get_outputs()`, `save()`, `load_state()`.
Drop-in replacement via `--column-type transformer`.

## Comparison

| | Conscience | Transformer |
|---|---|---|
| Temporal | Mean only (lost) | Full attention over window |
| Training | Hebbian (no backprop) | SGD (backprop) |
| Categories | Spatial pattern match | Predictively useful patterns |
| Surprise | None | Prediction error |
| Collapse prevention | Theta threshold | Prediction pressure |
| Compute | ~1.5ms/tick | TBD |

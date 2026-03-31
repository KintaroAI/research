# Task 00001: PredictiveColumn — Self-Supervised Prediction Column

**Date:** 2026-03-29
**Status:** In progress

## Context

ConscienceColumn collapses temporal signal to a mean, losing temporal structure.
Conscience threshold prevents winner collapse but causes artificial rotation
disrupting motor output. PredictiveColumn uses a per-column temporal encoder
(1-layer causal transformer) that learns categories via next-state prediction.
Categories forced to carry predictive meaning through bottleneck — no conscience hack.

## Deliverables

1. `PredictiveColumn(ColumnBase)` class in `column_manager.py`
2. `elif column_type == 'predictive':` branch in `cluster_manager.py`
3. CLI args in `main.py`: `--column-n-heads`, `--column-lambda-sharp`, `--column-lambda-balance`

## Architecture

- Per-column 1-layer causal transformer encoder → summary z
- Prototype head: softmax(z · category_embs / temp) → category probs
- Prediction from category bottleneck: z_q = Σ p_i * c_i → predict next frame
- Balance regularizer: sharp (low per-sample entropy) + balance (uniform batch usage)
- ~5.7K params per column, ~1.1M total for m=200

## Verification

1. Smoke test: `--column-type predictive --signal-source forage -f 100`
2. Output shape: get_outputs() returns (m, n_outputs) float32 in [0,1], sums to ~1
3. Prediction loss decreasing over time
4. Surprise signal: get_surprise() returns (m,) float32
5. Category spread: balance loss prevents collapse to single output

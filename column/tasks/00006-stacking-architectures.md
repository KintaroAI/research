# 00006 — Stacking Architectures for Non-Linear Patterns

**Date:** 2026-03-16
**Status:** In Progress

## Context

A single SoftWTACell does linear separation via prototype matching. Stacking
cells in specific configurations can break this barrier. Four architectures
to explore, starting with residual stacking.

## Option 1: Residual Stacking (implementing)

Each layer categorizes the *residual* — what the previous layer got wrong.
Combined output (w1, w2, w3) gives m^L effective categories from L layers
of m outputs each.

```
x → Cell 1 → w1, residual = x - proto1[w1]
      residual → Cell 2 → w2, residual2 = residual - proto2[w2]
            residual2 → Cell 3 → w3
```

**Test:** 16 clusters in 16D. Single 4-output cell can't separate all 16.
Three stacked 4-output cells (4³=64 effective) should.

## Option 2: Multi-Scale Temporal

Multiple cells on same input with different `streaming_decay`. Fast cell
captures rapid changes, slow cell captures trends.

**Test:** signal with 4 slow trends × 2 fast sub-patterns = 8 categories.
No single time scale can separate all 8.

## Option 3: Recurrent Ring (Attractor Dynamics)

Cells in a loop — output feeds into next cell. Iterate until stable.
Different inputs converge to different attractors. Could learn non-convex
categories from dynamics rather than single-shot matching.

## Option 4: Receptive Field Tiling

Layer 1: multiple cells on input subsets (local features).
Layer 2: one cell on concatenated Layer 1 outputs (combinations).
Detects conjunctions that no single cell can.

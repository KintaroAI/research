# 00005 — Stacked Cell Architectures

**Date:** 2026-03-15
**Status:** In Progress

## Context

A single SoftWTACell has fundamental limits: it can only do linear separation via
prototype matching. Stacking multiple cells unlocks compositional and hierarchical
learning. Three architecture ideas explored below.

## Option 1: Factored Categories (parallel cells on input)

Input has two independent factors (e.g., channels 0-2 = "color", 3-5 = "shape").
Total 3×3 = 9 combinations.

- **Single cell (3 outputs):** learns one factor, ignores the other
- **Two parallel cells (3 each):** each specializes on one factor, joint output covers all 9
- **Test:** joint NMI of two cells vs single cell against 9-way ground truth
- **Open question:** do cells naturally specialize without channel masking?

**Why:** cleanest scientific test of parallel decomposition

## Option 2: Hierarchical Temporal (sequential cells) ← IMPLEMENTING

Movement patterns in 2D: steady drift, oscillation, zigzag, spiral. Each is a
*sequence* of instantaneous directions.

- **Cell 1** (temporal, raw positions) → categorizes instantaneous direction per frame
- **Cell 2** (temporal, Cell 1 output history) → categorizes the sequence of directions

Neither cell alone solves it:
- Cell 1 only sees local direction, not the pattern
- Cell 2 can't work on raw noisy positions — needs intermediate categorization

**Why:** mirrors cortical layer architecture, builds on 3D movement benchmark,
tests whether cells can learn from probability outputs

## Option 3: Compositional Logic (multi-layer)

Two inputs each encode a number (0-3). Task: categorize sum mod 4.

- `(2,1)` and `(1,2)` → same output, `(2,2)` → different
- Non-convex boundaries — single prototype cell can't carve this
- **Three cells:** A categorizes input 1, B categorizes input 2, C sees both outputs
  and learns the sum relationship

**Why:** connects to grokking work in llm-fundamentals, tests compositional reasoning

## Deliverables

- [x] Task file with all three options documented
- [ ] Option 2: `benchmark_hierarchy.py` with hierarchical temporal benchmark
- [ ] Option 1: factored categories benchmark (future)
- [ ] Option 3: compositional logic benchmark (future)

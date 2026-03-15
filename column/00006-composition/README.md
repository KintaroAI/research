# 00006 — Compositional Logic (Negative Result)

**Status:** Complete
**Source:** `exp/00006` (`b709b89`)

## Goal

Test whether three cells with multiplicative interaction (outer product) can learn
relational tasks between two input streams: same/different, proximity, sum mod 4.

## Hypothesis

The outer product of Cell A and Cell B outputs creates a feature space where
relational categories (same vs different, sum classes) have distinct centroids,
enabling Cell C to separate them via prototype matching.

## Method

**Architecture:**
```
input_a (8D) → Cell A (4 outputs) ─┐
                                    ├─ outer product (16D) → Cell C → category
input_b (8D) → Cell B (4 outputs) ─┘
```

**No backpropagation** — all cells use local Hebbian learning only.

**Data:** Numbers 0-3 encoded as 8D Gaussian clusters (noise=0.3). Balanced
sampling (50% same, 50% different). Phased training: Cell A/B pretrained for
5k frames before Cell C starts.

**Three tasks:**
1. Same/different: is a == b? (2 categories)
2. Proximity: is |a-b| mod 4 ≤ 1? (2 categories)
3. Sum mod 4: (a+b) % 4 (4 categories)

**Three architectures compared:**
- 3cell_outer: Cell A ⊗ Cell B → Cell C
- 3cell_concat: Cell A || Cell B → Cell C
- single: raw concatenated input → single cell

## Results

```
task                     3cell_outer    3cell_concat          single
----------------------------------------------------------------
same_diff                      0.000           0.000           0.000
proximity                      0.000           0.002           0.000
sum_mod4                       0.057           0.034           0.057
```

All approaches at chance level across all tasks.

## Analysis — Why It Fails

**This is a fundamental limitation of prototype-based competitive learning
on relational tasks**, not a tuning issue.

### The centroid problem

For same/different: in the outer product space, "same" pairs activate diagonal
positions {(0,0), (1,1), (2,2), (3,3)} while "different" pairs activate
off-diagonal positions. In theory, these have different centroids.

In practice, Cell A/B outputs are soft probabilities (NMI ~0.7), so the outer
product is noisy. The centroid difference between "same" and "different" in 16D
is ~0.12 per position — smaller than the noise from imperfect categorization.

### The balanced residue class problem

For sum mod 4: each residue class {(a,b) : (a+b) mod 4 = k} contains exactly
one pair per row and column (it's a Latin square). This means:
- All class centroids are identical in any factored representation
- Verified: even with perfect one-hot Cell A/B outputs, Cell C gets NMI=0

This is a mathematical property of modular arithmetic, not a limitation of the
cell count or architecture.

### What this reveals

Prototype-based cells with Hebbian learning can solve:
- **Clustering** (exp 00001) — grouping similar inputs ✓
- **Temporal patterns** (exp 00002, 00004) — co-variation structure ✓
- **Hierarchical decomposition** (exp 00005) — transition patterns ✓

But they cannot solve:
- **Relational composition** — "is a the same as b?" requires comparing two
  representations, not matching against a stored prototype
- **Modular arithmetic** — the combinatorial structure defeats centroid-based methods

This suggests that compositional reasoning requires either:
- A different similarity measure (not dot product / cosine)
- An explicit comparison operation between cell outputs
- Or gradient-based learning to discover the right features

# 00015 — Recurrent Ring (Negative Result)

**Status:** Complete
**Source:** `c/00015` (`df2b42c`)

## Goal

Test whether cells connected in a ring with recurrent feedback can learn
non-convex categories through attractor dynamics.

## Hypothesis

Iterating through multiple cycles should refine the categorization — initial
noisy assignments get corrected by feedback. Non-convex clusters (2 sub-clusters
per category) should benefit from recurrent processing.

## Method

**Architecture:**
```
x → Cell A (n→m) → Cell B (m→m) → Cell C (m→m) → feedback to Cell A → ...
```
Cell A receives `0.7 * raw_input + 0.3 * winner_prototype_feedback` on recurrent
passes. Tested with 2 and 3 cells, 1/3/5 cycles.

**Two test datasets:**
1. Standard convex: 8 Gaussian clusters in 16D
2. Non-convex: 4 categories, each with 2 sub-clusters on opposite sides

## Results

### Convex clusters (8 clusters, 16D)

| Architecture | NMI |
|---|---|
| **single cell (m=8)** | **0.957** |
| ring 2 cells, 3 cycles | 0.862 |
| ring 3 cells, 5 cycles | 0.808 |

### Non-convex clusters (4 categories, 2 sub-clusters each)

| Architecture | NMI |
|---|---|
| **single cell (m=4)** | **0.950** |
| single cell (m=8) | 0.812 |
| ring 3 cells, 3 cycles (m=4) | 0.800 |
| ring 2 cells, 5 cycles (m=4) | 0.613 |

## Analysis

**The ring does not beat single cells on either task.** This is a negative result.

**On convex clusters:** the single cell already achieves 0.957 — the ring only adds
noise through the recurrent pathway. Each cycle introduces additional competitive
learning updates that can disturb already-correct assignments.

**On non-convex clusters:** surprisingly, the single cell with m=4 gets 0.950 —
it handles non-convex categories well because the prototype can sit at the centroid
of the two sub-clusters and still win for both (the sub-clusters are close enough).
The ring's feedback mechanism doesn't help because the problem is already solvable.

**Why the ring hurts:** the recurrent pathway makes cells learn from their own outputs
(probabilities as inputs) rather than raw data. This creates a self-reinforcing loop
where early mistakes get amplified. The feedback signal (winner prototype) biases
Cell A's input toward what it already believes, rather than correcting errors.

**The attractor idea needs a different mechanism.** Simple recurrence through competitive
cells doesn't create useful attractors. Would need: explicit energy function, symmetric
connections (Hopfield-style), or a comparison between forward and feedback that gates
the update rather than blending the input.

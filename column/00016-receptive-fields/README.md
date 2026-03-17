# 00016 — Receptive Field Tiling

**Status:** Complete
**Source:** `c/00016` (`a9d2c7e`)

## Goal

Test whether local feature cells (receptive fields) + a combination cell can
detect feature conjunctions that a single cell on the full input cannot.

## Method

**Architecture:**
```
x[0:g]   → Cell 1 (g→k) ─┐
x[g:2g]  → Cell 2 (g→k)  ├─ concat (n_features×k) → Cell C (→m) → category
...                        │
x[(f-1)g:fg] → Cell f    ─┘
```

**Data:** Input split into groups. Each group independently takes one of 2 patterns.
Category = combination of feature 0 × feature 1 (4 categories). Requires seeing
both local features to determine the category.

## Results

### Test 1: 16 inputs, 4 groups of 4

| Architecture | NMI |
|---|---|
| single cell (m=4) | 0.578 |
| **single cell (m=8)** | **0.993** |
| RF 4×2 → 8 | 0.855 |
| RF 4×4 → 4 | 0.000 (collapse) |

### Test 2: 32 inputs, 4 groups of 8

| Architecture | NMI |
|---|---|
| single cell (m=4) | 0.405 |
| single cell (m=8) | 0.462 |
| RF 4×2 → 4 | 0.000 (collapse) |

### Test 3: 32 inputs, 8 groups of 4

| Architecture | NMI |
|---|---|
| single cell (m=4) | 0.500 |
| RF 8×2 → 4 | 0.000 (collapse) |
| **RF 8×4 → 4** | **0.993** |

## Analysis

**RF wins decisively on Test 3.** With 32 inputs split into 8 groups, the single cell
(NMI=0.500) can't separate all conjunction categories. RF 8×4→4 (NMI=0.993) succeeds
because each local cell has an easy task (4D input, 2 patterns → 4 outputs) and the
combination cell sees a rich 32D representation of local features.

**But RF often collapses.** Many configurations get NMI=0.000 — the combination cell
fails to learn from concatenated Layer 1 outputs. This happens when:
- Local cells have too many outputs for their input size (4 outputs on 4D input
  with only 2 patterns → 2 dead outputs producing noise)
- Layer 2 input dimension is too large relative to training data

**Single cell wins when it has enough outputs.** On 16D input (Test 1), single m=8
already gets 0.993 — it can see all 16 inputs at once and has enough prototypes.

**RF shines at high input dimensionality.** The 32D single cell maxes out at 0.50
because 32D is hard to cluster with 4 prototypes. The RF architecture decomposes
the problem: each local cell handles a manageable 4D subspace, and the combination
cell works in the local-feature space.

**The right local output count matters.** RF 8×4→4 works, RF 8×2→4 collapses.
The local cells need enough outputs (4) to represent the local patterns cleanly.
With only 2 outputs, they sometimes assign both patterns to the same output.

## Commands

```bash
cd dev
python benchmark_receptive.py --frames 20000 -o $(python output_name.py 16 receptive)
```

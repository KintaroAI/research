# 00005 — Hierarchical Cell Stack

**Status:** Complete
**Source:** `exp/00005` (`ea1da96`)

## Goal

Test whether two stacked SoftWTACells can learn movement *patterns* that neither
cell can learn alone, demonstrating the value of hierarchical composition.

## Hypothesis

A two-cell hierarchy (Cell 1 → transition matrix → Cell 2) will categorize movement
patterns with higher NMI than either cell operating alone.

## Method

**Movement patterns** in 2D (4 categories):
- **Steady:** constant direction throughout
- **Oscillate:** flip direction every segment (180°)
- **Zigzag:** rotate 90° every segment
- **Circle:** rotate smoothly through all directions

Each trajectory = 12 segments × 10 timesteps. 4 base directions, random starting position.

**Architecture:**
```
raw positions (2, T_seg)  →  Cell 1 (temporal correlation, 4 outputs)
                                     ↓ winner per segment
                              transition matrix (4×4, flattened to 16)
                                     ↓
                              Cell 2 (instantaneous, 4 outputs)  →  pattern category
```

**Why transition matrix?** It's direction-invariant: "steady-north" and "steady-east"
both produce diagonal transition matrices. "Oscillate" produces anti-diagonal.
Covariance of probability outputs failed (v1/v2) because it doesn't capture sequential
structure — only which dimensions co-vary, not the order of changes.

**Why neither cell alone works:**
- Cell 1 alone sees the full trajectory's covariance — but steady and oscillate have
  similar long-run covariance (same axes active, just different temporal patterns)
- A single cell on raw positions can't decompose direction + pattern
- The hierarchy separates the problem: Cell 1 extracts direction, Cell 2 extracts
  temporal structure from the direction sequence

**Parameters:** 8000 samples, 12 segments × 10 timesteps, 4 base directions.

**Commands:**
```bash
cd dev
python benchmark_hierarchy.py --samples 8000 -o $(python output_name.py 5 hierarchy)
```

## Results

```
metric                     hierarchy    cell1_only    cell2_only
----------------------------------------------------------------
cell1_dir_nmi                  0.667             —             —
winner_entropy                 0.757         0.978         0.824
usage_gini                     0.373         0.129         0.359
confidence_gap                 0.779         0.863         0.805
prototype_spread               0.866         0.766         0.556
nmi                            0.565         0.328         0.218
purity                         0.513         0.526         0.555
consistency                    0.752         0.443         0.472
```

## Analysis

**Hierarchy wins.** NMI=0.565 vs cell1_only=0.328 (+72%) vs cell2_only=0.218 (+159%).
Neither cell alone can distinguish movement patterns — the hierarchy is required.

**Transition matrix is the key insight.** Earlier attempts using covariance of Cell 1's
probability outputs (v1, v2) failed because covariance captures co-variation structure,
not sequential change patterns. The transition matrix directly encodes "given the current
direction, what's the next direction?" — which is exactly what distinguishes steady
(stay same) from oscillate (flip) from zigzag (rotate 90°) from circle (rotate 45°).

**Cell 1 direction NMI=0.667.** Reasonable but imperfect — some direction errors propagate
to Cell 2. The hierarchy's pattern NMI (0.565) is bounded by Cell 1's accuracy.

**Consistency is highest for hierarchy.** 0.752 vs 0.443/0.472 for individual cells.
The same pattern reliably gets the same Cell 2 output — even though the starting
direction varies.

**Room for improvement:**
- Cell 1 direction accuracy (0.667) limits the pipeline — better Cell 1 = better patterns
- More training data would help Cell 2 see more (pattern × start_dir) combinations
- The transition matrix is a hand-crafted intermediate — future work could learn
  the inter-cell representation

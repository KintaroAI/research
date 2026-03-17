# 00013 — Residual Stacking

**Status:** Complete

## Goal

Stack SoftWTACells in a residual chain where each layer categorizes what the
previous layer missed. Test whether this breaks the single-cell category limit.

## Hypothesis

A 3-layer residual stack of 4-output cells (4³=64 effective categories) will
separate 16 clusters that a single 4-output cell cannot.

## Method

**Architecture:**
```
x → Cell 1 (n→4) → winner w1, residual = x - proto1[w1]
    → Cell 2 (n→4) → winner w2, residual2 = residual - proto2[w2]
      → Cell 3 (n→4) → winner w3
```
Combined label = w1 × 16 + w2 × 4 + w3 (base-4 encoding).

## Results

### 16 clusters in 16D (20k frames)

| Architecture | Effective categories | NMI |
|---|---|---|
| single cell (m=4) | 4 | 0.325 |
| single cell (m=8) | 8 | 0.825 |
| **single cell (m=16)** | 16 | **0.952** |
| residual 2×4 | 16 | 0.419 |
| residual 3×4 | 64 | 0.452 |
| residual 4×4 | 256 | 0.422 |

Per-layer NMI (cumulative): 0.577 → 0.745 → 0.780 → 0.763

### 8 clusters in 16D (20k frames)

| Architecture | Effective categories | NMI |
|---|---|---|
| single cell (m=4) | 4 | 0.711 |
| **residual 2×4** | 16 | **0.822** |
| residual 3×4 | 64 | 0.736 |
| single cell (m=8) | 8 | 0.957 |

## Analysis

**Residual stacking extends capacity.** On 8 clusters, residual 2×4 (NMI=0.822) beats
single m=4 (NMI=0.711) by 16% — the second layer captures within-cluster variation
that the first layer missed.

**Each layer adds information.** Per-layer NMI increases: 0.577 → 0.745 → 0.780.
Layer 2 refines Layer 1's coarse categorization by separating the residuals.

**But a single cell with enough outputs wins.** Single m=16 (NMI=0.952) dominates
residual 3×4 (NMI=0.452) on 16 clusters. The residual decomposition creates many
effective categories but they don't align cleanly with ground truth because:
- Layer 1 makes imperfect splits — residuals carry noise from wrong assignments
- Error compounds: a wrong Layer 1 winner produces a misleading residual for Layer 2
- The tuple encoding (w1, w2, w3) treats each combination as distinct, fragmenting
  true clusters across multiple tuples

**Diminishing returns after Layer 2.** Adding Layer 3 and 4 barely helps (0.745 → 0.780
→ 0.763). Once the residual is mostly noise from assignment errors, additional layers
can't extract meaningful structure.

**When residual stacking IS useful:** when you're capacity-constrained (can only afford
m=4 outputs per cell) but need finer separation. Two layers doubles effective categories
with genuine improvement.

## Commands

```bash
cd dev
python benchmark_residual.py --frames 20000 -o $(python output_name.py 13 residual)
```

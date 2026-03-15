# 00007 — Composition via Wiring Operations

**Status:** Complete
**Source:** `exp/00007` (`7699067`)

## Goal

Overcome the compositional logic limitations found in exp 00006 by using
task-appropriate wiring operations between cells, without adding backpropagation.

## Hypothesis

The right fixed (non-learned) wiring operation between cells transforms a relational
task into a clustering task that prototype cells can solve. Specifically:
- **Circular convolution** for sum mod 4
- **Comparison statistics** for same/different

## Method

**Architecture:**
```
input_a (8D) → Cell A (4 outputs) ─┐
                                    ├─ [wiring op] → Cell C → category
input_b (8D) → Cell B (4 outputs) ─┘
```

**New wiring operations:**

| Wiring | Formula | Dimension | Task |
|---|---|---|---|
| `circ_conv` | `conv[k] = Σ_j p_a[j] · p_b[(k-j) mod 4]` | 4D | Sum mod 4 |
| `compare` | `[cos_sim, L2_dist, max_prod, entropy_diff]` | 4D | Same/different |

**Why they work:**
- Circular convolution computes the exact probability distribution over (a+b) mod 4.
  Each sum class produces a peaked vector at a distinct position.
- Comparison statistics are value-independent: cosine similarity between p_a and p_b
  is high for same, low for different, regardless of WHICH value matches.

**Why exp 00006 wiring failed:**
- Outer product: 16D space where sum classes are equidistant (no clustering)
- Concatenation: same centroids for all relational categories
- Absolute difference: normalization strips magnitude (same=near-zero, direction lost)

**Phased training:** Cell A/B pretrained for 5k frames, then Cell C trained.

## Results (5 seeds, eval on last 3k frames)

### Sum mod 4

| Architecture | NMI (mean ± std) |
|---|---|
| **circ_conv** | **0.377 ± 0.139** |
| single cell | 0.237 ± 0.169 |
| outer product (exp 06) | ~0.06 |

### Same/Different (noise=0.05)

| Architecture | NMI (mean ± std) |
|---|---|
| **compare** | **0.236 ± 0.013** |
| single cell | 0.007 ± 0.007 |
| outer product (exp 06) | ~0.00 |

## Analysis

**Wiring rescues composition.** The right inter-cell operation transforms relational
tasks from unsolvable (NMI ≈ 0) to partially solved:
- circ_conv: 60% improvement over single cell on sum mod 4
- compare: 34x improvement on same/different (0.236 vs 0.007)

**The pattern across all experiments:**

| Experiment | Wiring operation | What it computes |
|---|---|---|
| 00005 hierarchy | Transition matrix | Temporal sequence structure |
| 00007 sum mod 4 | Circular convolution | Modular addition of distributions |
| 00007 same/diff | Comparison statistics | Value-independent similarity |

All are **fixed, non-learned transforms**. The cells do learned clustering;
the wiring determines what computation is possible. This mirrors cortical
architecture where inter-area connectivity is genetically determined while
intra-area weights are learned.

**Limitations remain.** NMI=0.24–0.38 is well below the 0.96+ achieved on
clustering and temporal tasks. The noise sensitivity is high — Cell A/B errors
propagate through the wiring. Improving Cell A/B accuracy (via lower noise or
better hyperparameters) directly improves downstream performance.

**Variance is significant** for circ_conv (std=0.14), reflecting the sensitivity
of competitive learning to initialization. Compare wiring is more stable (std=0.01)
because the comparison statistics are inherently smoother.

## Commands

```bash
cd dev
python benchmark_composition.py --noise 0.1 -o $(python output_name.py 7 composition)
```

# 00014 — Multi-Scale Temporal

**Status:** Complete

## Goal

Test whether two streaming cells at different time scales can jointly separate
patterns that no single time scale can distinguish.

## Hypothesis

A fast cell (low decay) captures rapid sub-pattern changes. A slow cell (high
decay) captures persistent trends. Combined output separates all categories.

## Method

**Data:** 4 slow trends × 2 fast sub-patterns = 8 combined categories.
- Slow: sinusoidal co-variation along a direction, period ~100 frames
- Fast: random co-variation along a different direction, changes every 5 frames
- Both produce actual temporal variance (not DC bias)

**Architecture:**
```
x(t) → Cell_fast  (16→2, decay=0.3)  → fast winner
x(t) → Cell_slow  (16→4, decay=0.98) → slow winner
Combined: slow * 2 + fast → 8 effective categories
```

## Results

### Single cell at various decays (best decay for each task)

| Decay | vs slow | vs fast | vs combined |
|---|---|---|---|
| 0.3 | 0.002 | 0.278 | 0.103 |
| 0.5 | 0.002 | 0.284 | 0.115 |
| 0.95 | 0.133 | 0.041 | 0.094 |
| 0.98 | — | — | — |

No single decay captures both. Fast decays see fast patterns, slow decays see slow ones.

### Multi-scale tuning (30k frames, eval on last 10k)

| Fast decay | Slow decay | Fast NMI | Slow NMI | Combined NMI |
|---|---|---|---|---|
| 0.3 | 0.95 | 0.277 | 0.130 | 0.190 |
| **0.3** | **0.98** | **0.277** | **0.272** | **0.276** |
| 0.3 | 0.99 | 0.277 | 0.096 | 0.172 |
| 0.5 | 0.95 | 0.291 | 0.097 | 0.175 |

### Summary

| Architecture | Combined NMI (8 categories) |
|---|---|
| Best single cell (decay=0.5) | 0.115 |
| **Multi-scale (fast=0.3, slow=0.98)** | **0.276** |

**2.4x improvement** from multi-scale over best single cell.

## Analysis

**The cells specialize by timescale.** Fast cell (decay=0.3) gets NMI=0.277 on fast
labels but 0.002 on slow — it can't see slow patterns. Slow cell (decay=0.98) gets
NMI=0.272 on slow labels but ignores fast changes. Together they cover both.

**Decay=0.98 is the sweet spot for slow patterns.** The slow sinusoidal signal has
period ~100 frames. Decay=0.98 gives an effective window of ~50 frames — enough to
capture the slow co-variation. Higher decay (0.99+) makes the window too long,
mixing multiple slow blocks.

**Combined NMI is moderate (0.276) not high.** Both cells are imperfect (~0.27 each),
and the combined tuple encoding multiplies their errors. Still, it clearly beats any
single-scale approach.

**This is biologically plausible.** Different cortical layers have different temporal
integration constants — fast layers for sensory processing, slow layers for contextual
stability. The multi-scale approach mirrors this.

## Commands

```bash
cd dev
python benchmark_multiscale.py --frames 20000 -o $(python output_name.py 14 multiscale)
```

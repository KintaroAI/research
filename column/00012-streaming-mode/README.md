# 00012 — Streaming Temporal Mode

**Status:** Complete
**Source:** `exp/00012` (`80ba95a`)

## Goal

Add an O(mn) streaming alternative to the O(n²T) correlation mode.
Instead of computing a full covariance matrix from a trace, maintain
running EMA of projection variance per prototype.

## Method

**New mode: `temporal_mode='streaming'`**

Each step, for input x (n,):
1. Project onto each prototype: `proj = prototypes @ x` — O(mn)
2. Update running mean: `proj_mean = d * proj_mean + (1-d) * proj`
3. Update running variance: `proj_var = d * proj_var + (1-d) * (proj - proj_mean)²`
4. Similarity = `proj_var` (high variance = active direction)
5. Update: Oja's rule instead of power iteration — O(n) per winner

**Key parameter: `streaming_decay`** controls EMA memory length.
Lower decay = shorter memory = faster response to cluster changes.

## Results

### Performance (us/step, CPU)

| Config | Instant | Correlation | Streaming |
|---|---|---|---|
| 16×4 | 183 | 247 | 188 |
| 64×16 | 194 | 1,784 | 192 |
| 256×32 | 188 | — | 199 |

**Streaming matches instantaneous speed at all sizes.** The 64×16 bottleneck
(1,784us → 192us) is a **9x speedup** over correlation.

### Separation quality (co-variation clusters, 16 inputs, 4 clusters)

**Streaming needs temporal continuity** — consecutive samples from the same
source. With i.i.d. random cluster assignment (block=1), NMI≈0.

| Block size | Streaming NMI | Correlation NMI (T=block) |
|---|---|---|
| 1 | 0.006 | — |
| 5 | 0.094 | 0.845 |
| 10 | 0.196 | 0.891 |
| 20 | 0.349 | 0.666 |
| 50 | 0.567 | — |

### Streaming decay tuning (block_size=10)

| Decay | NMI |
|---|---|
| 0.99 | 0.000 |
| 0.95 | 0.103 |
| 0.90 | 0.182 |
| 0.80 | 0.326 |
| 0.70 | 0.520 |
| 0.50 | 0.598 |

Lower decay = shorter memory = better for short blocks. At decay=0.5 with
block_size=10, NMI=0.598 — approaching correlation's 0.891 for T=10.

## Analysis

**Speed vs quality tradeoff:**
- Correlation: best NMI (0.85–0.89) but O(n²T), unusable at n>30
- Streaming: O(mn), same speed as instantaneous, but needs temporal continuity
  and optimal decay tuning to approach correlation quality

**Streaming requires temporal autocorrelation.** In real-world use (sensors,
motor control, perception), consecutive inputs ARE from the same source —
this is a natural property of physical systems. The i.i.d. benchmark is
the adversarial case.

**decay=0.5 is a good default** for block sizes around 10. It gives the EMA
an effective window of ~2 samples (1/(1-0.5)), which captures the current
cluster's variance without mixing in old clusters.

**When to use which mode:**
- Instantaneous: no temporal structure needed
- Correlation: offline or batched temporal analysis, n ≤ ~30
- Streaming: online real-time temporal analysis at any n

## Commands

```bash
cd dev
make test  # 25 tests including streaming
```

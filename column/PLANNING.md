## Initial analysis

The requirements describe a **cortical minicolumn** — a competitive self-organizing module. It's essentially the core properties of a Kohonen/SOM-style unit but with sharper constraints (probability output, anti-collapse, stability/plasticity balance).

## What maps to existing literature

- **Classic SOM / competitive learning** covers requirements 1-8 well
- **Growing Neural Gas** handles adaptation (req 10) better than fixed-topology SOMs
- **Soft-WTA with softmax** covers "probability output with one winner" (req 3, 7) — not standard SOM but straightforward to add
- **Adaptive Resonance Theory (ART)** — Grossberg's answer to the stability-plasticity dilemma (req 9-11). Vigilance parameter controls when to create new categories vs absorb into existing ones. Biologically motivated, local learning, but finicky to tune.

## The hard part: stability vs plasticity (req 9+10+11)

This is the classic unsolved tension. Approaches worth considering:

- **ART-style vigilance** — match quality gate that decides "update existing prototype" vs "recruit new unit." Principled but introduces a sensitive hyperparameter.
- **Frequency-gated lr** — units that fire often reduce their learning rate (stable). Rare-firing units stay plastic. Similar to the KNN-driven adaptive params idea from thalamus-sorter ts-00015.
- **Match-based learning** — only update the winner if match quality exceeds a threshold. Poor matches trigger recruitment of a dormant unit instead.

## Temporal context connection

The optional temporal context (req 1: input as `(n, T)` matrix) is exactly what thalamus-sorter does with the derivative correlation buffer. The machinery for "distinguish both-silent from both-co-varying" already exists there. Could reuse the MSE/correlation/derivative-correlation similarity measures.

## Core architecture

A **soft-WTA cell** with these components:

1. **Prototypes**: each of `m` output units has a prototype vector (like SOM centroids)
2. **Competition**: similarity → softmax → probability output with temperature control
3. **Winner update**: winner's prototype moves toward input (Hebbian pull)
4. **Anti-collapse via usage counter**: per-unit frequency tracker gates plasticity — frequently-winning units learn slower, giving other units a chance
5. **Match threshold for stability/plasticity**: poor matches recruit a dormant unit rather than forcing assimilation into existing categories
6. **Dynamic inputs**: `extend_inputs()` / `remove_inputs()` for live channel add/remove

This is simple, local, biologically plausible, and naturally covers req 1-8. The match threshold + usage gating addresses 9-11.

## Temporal modes — algorithm comparison

Three temporal modes are implemented. The choice depends on input dimensionality,
whether data arrives as traces or individual samples, and latency requirements.

### Mode 1: Instantaneous (`temporal_mode=None`)

Input `(n,)` single vector. No temporal structure.

**Similarity:** cosine (dot product of normalized input and prototypes)
```
sim_i = normalize(x) · proto_i        — O(mn) total
```
**Update:** Hebbian pull — move winner toward normalized input
```
proto[w] += lr * (x_norm - proto[w])   — O(n)
```
**Pros:** simplest, fastest (~190us), works when input is already a meaningful feature vector
**Cons:** blind to temporal structure, can't distinguish co-varying from co-silent channels

### Mode 2: Correlation (`temporal_mode='correlation'`)

Input `(n, T)` trace. Computes full covariance matrix.

**Similarity:** variance of input projected onto each prototype direction
```
C = x_centered @ x_centered.T / (T-1)   — O(n²T) covariance
sim_i = proto_i @ C @ proto_i            — O(mn²) total
```
**Update:** power iteration — rotate winner toward dominant eigenvector of C
```
target = C @ proto[w]                    — O(n²)
proto[w] += lr * (normalize(target) - proto[w])
```
**Pros:** exact covariance computation, best quality at low n
**Cons:** O(n²T + mn²) per step, covariance is rank-deficient when T < n (hurts quality),
memory O(n²) for covariance matrix, requires full trace buffer

### Mode 3: Streaming (`temporal_mode='streaming'`)

Input `(n,)` single sample OR `(n, T)` trace. Avoids full covariance.

**Key insight:** `proto @ C @ proto = variance(proto · x over time)`. We don't need
the n×n covariance — just the scalar projection variance per prototype.

**Similarity for (n, T) trace:**
```
proj = prototypes @ x                   — O(mnT) projection
sim_i = var(proj_i over T)              — O(mT) variance
```
**Similarity for (n,) single sample:** EMA of projection mean and variance
```
proj = prototypes @ x                   — O(mn)
proj_mean = decay * proj_mean + (1-decay) * proj
proj_var  = decay * proj_var  + (1-decay) * (proj - proj_mean)²
sim = proj_var
```
**Update for (n, T):** efficient power iteration without full C
```
inner = x_centered.T @ proto[w]         — O(nT)
target = x_centered @ inner / (T-1)     — O(nT), NOT O(n²)
```
**Update for (n,):** Oja's rule (streaming equivalent of power iteration)
```
proj = proto[w] · x
proto[w] += lr * proj * (x - proj * proto[w])   — O(n)
```

**Pros:**
- O(mnT) for traces, O(mn) for single samples — never builds n×n matrix
- Handles both (n,) and (n, T) transparently
- No rank-deficiency: works correctly even when n >> T
- Scales to any n (256×32 at 784us)

**Cons:**
- For (n,) single samples, needs temporal continuity (consecutive inputs from
  same source) — EMA mixes clusters with i.i.d. input
- Slightly lower quality than correlation at small n where covariance is well-conditioned
- `streaming_decay` parameter controls memory length — needs tuning for the
  temporal block structure of the input

### When to use which

| Condition | Recommended mode |
|---|---|
| No temporal structure | Instantaneous |
| n ≤ T (covariance well-conditioned) | Correlation or Streaming |
| n > T (covariance rank-deficient) | **Streaming** |
| Real-time single samples | Streaming with (n,) |
| Large n (>30) | **Streaming** (correlation too slow) |
| Offline batch analysis | Correlation (exact) |

### Benchmark results (exp 00012)

| Benchmark | Correlation | Streaming |
|---|---|---|
| n=16 T=10 NMI | 0.989 | 0.848 |
| n=64 T=10 NMI | 0.579 | **0.804** |
| n=16 T=10 speed | 258 us | 271 us |
| n=64 T=10 speed | 829 us | **561 us** |
| n=256 T=10 speed | ~28,000 us | **784 us** |

## Future exploration: input-order invariance

When composing multiple cells (exp 00005–00007), the wiring operation between cells
determines what relationships are learnable. An open question: can the architecture
be made **input-order invariant** so that feeding (a, b) produces the same output as
(b, a)?

This matters for commutative relationships (sum, equality, proximity) where order
shouldn't affect the result. Current wiring operations have mixed behavior:
- Circular convolution: naturally commutative (a+b = b+a)
- Comparison statistics: naturally commutative (cos(a,b) = cos(b,a))
- Outer product: NOT commutative (a⊗b ≠ b⊗a)
- Concatenation: NOT commutative

A general approach: use **symmetric wiring** — operations that are invariant to
input order by construction. For two inputs, this means: `f(a,b) = f(b,a)`.
Candidates: element-wise max/min of cell outputs, sorted concatenation,
sum of outputs, or symmetric outer product `(a⊗b + b⊗a) / 2`.

For N>2 inputs, this generalizes to permutation-invariant aggregation (sum, max,
attention-weighted mean). Worth exploring if the cell is used as a building block
in larger architectures where input ordering shouldn't be a degree of freedom.

## Future exploration: noise tolerance per channel

Exp 00009 revealed that the cell is fragile to noisy input channels. Just 2 noisy
channels out of 16 (12%) at high amplitude collapse separation. Root cause: input
normalization maps the vector to a unit sphere, so high-amplitude noise in a few
channels hijacks the direction, drowning out signal channels.

The cell should be **tolerant to noise in individual channels** — a broken or
irrelevant sensor shouldn't destroy categorization from the remaining channels.
This is important for real-world inputs where not all channels are reliable.

Approaches to explore:
- **Per-channel learned gating:** each prototype stores a weight mask indicating
  which channels matter. Similarity = weighted dot product. Noisy channels get
  down-weighted over time. Biologically plausible (dendritic gating).
- **Robust similarity measure:** replace cosine similarity with something less
  sensitive to outlier channels (e.g., trimmed mean of per-channel similarities,
  or median-based distance).
- **Input normalization per channel** instead of per vector: normalize each
  channel to zero-mean unit-variance over a running window, so no single channel
  can dominate magnitude.
- **Attention-weighted input:** a lightweight pre-processing step that scores
  each channel's usefulness (e.g., by variance or predictability) and scales
  the input before prototype matching.

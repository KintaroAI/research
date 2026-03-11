# Experiment: Word2vec-Style Drift

**Date:** 2026-03-10
**Status:** In Progress
**Source:** *tagged on completion as `exp/ts-00005`*

## Goal

Replace centroid-averaging update with word2vec-style skip-gram + negative sampling. Key properties borrowed from word2vec:
1. **Pairwise updates** (not centroid averaging)
2. **Sigmoid self-regulation** (focus learning on "mistakes")
3. **Explicit negative sampling** (push dissimilar pairs apart)
4. **Per-dimension coefficients** (each dim updated differently based on peer)

## Motivation

The centroid approach (exp 00002-00003) averages K neighbors' positions and moves toward the centroid. This:
- Updates all dimensions by the same scalar
- Has no explicit repulsion (relies on LayerNorm to prevent collapse)
- Collapses to lines in 2D with small K

Word2vec (Skip-gram) learns embeddings via pairwise pull/push with sigmoid scaling. The sigmoid provides natural curriculum: strong gradients for "wrong" pairs, saturation for "correct" pairs. Explicit negative sampling provides repulsion without relying on normalization.

## Method

### Update Rule

Each tick, for every neuron:

**Positive (1 random neighbor from top-K):**
```python
delta = pos_j - pos_i                           # toward peer
dist² = sum(delta²)
σ_pos = sigmoid(dist²)                          # large when far → needs pull
pos_i += lr * σ_pos * delta                     # pull toward positive peer
```

**Negative (k_neg random neurons):**
```python
delta = pos_neg - pos_i                          # toward negative peer
dist² = sum(delta²)
σ_neg = sigmoid(-dist²)                         # large when close → needs push
pos_i -= lr * σ_neg * delta                     # push away from negative peer
```

### Key Differences from Word2vec

Original word2vec uses **dot product** and updates in the **direction of** the peer vector (`pos_j`). This optimizes angular alignment, not spatial proximity. Our adaptation:
- Uses **Euclidean distance** in the sigmoid instead of dot product
- Updates **toward/away from** peer (`pos_j - pos_i`) instead of **in the direction of** peer (`pos_j`)
- This preserves the spatial update semantics needed for topographic maps

First attempt with dot product + direction-of-peer collapsed into binary clusters (all embeddings aligned/anti-aligned). Euclidean + toward/away works correctly.

### Parameters
- `k`: positive neighbors (precomputed top-K, same as exp 00002-00003)
- `k_neg`: negative samples per positive per tick (default 5)
- `lr`: learning rate
- `dims`: embedding dimensionality

## Log

### Test 1: Dot product version (original word2vec math), D2, K=11, 30k ticks

Collapsed into two clusters (black/white binary split). Dot product + direction-of-peer update optimizes angular alignment, not spatial proximity. Combined with LayerNorm, creates a degenerate solution.

### Test 2: Euclidean version, D2, K=11, k_neg=5, 30k ticks

Some spatial structure forming — dark/light clustering with coherence. No line collapse (unlike centroid drift at D2 K11). The explicit negative sampling prevents the degenerate solutions that centroid averaging falls into.

### Test 3: Euclidean version, D16, K=25, k_neg=5, 30k ticks

Clear K reconstruction. Comparable to centroid drift (exp 00003) at similar tick count.

### Test 4: Dot product, no LayerNorm, small uniform init, D2, K=11, k_neg=5, 30k ticks

Removed LayerNorm entirely, switched to word2vec-style uniform init `[-0.5/dims, 0.5/dims]`. `std=0.54, mean_norm=0.001` — sigmoid self-regulates scale without normalization. Some spatial structure forming, no binary collapse. **LayerNorm was fighting the sigmoid dynamics** in test 1 — without it, dot product word2vec works.

### Test 5: Dot product, no LayerNorm, D16, K=25, k_neg=5, 30k ticks

`std=0.37` stable. K shape vaguely visible but noisy. The binary positive/negative selection (top-K = always pull, random = always push) is too coarse — no correlation signal modulating push/pull strength. Random negatives sometimes push away true neighbors.

### Test 6: Dot product similarity mode, D2, P=10, sigma=5, threshold=0, 10k ticks

Merged temporal correlation (random sampling, RBF similarity) with word2vec (sigmoid, dot product, no normalization). `tick_similarity()`: random peers, Gaussian RBF on grid coords decides attract/repel, sigmoid of `signal * dot` scales magnitude, update in direction of peer vector. **Positions blow up** — `std` grows from 0.14 to 6.9. With `threshold=0`, RBF similarity is always ≥ 0, so every peer is attractive. No repulsion at all.

### Test 7: Dot product similarity mode, D2, P=10, sigma=5, threshold=0.2, 10k ticks

Threshold=0.2 gives repulsion for distant peers. `std=1.0` — scale controlled. But still radial blow-out pattern, no K structure. Threshold controls magnitude but doesn't fix the dot-product angular dynamics.

### Test 8: Dot product similarity mode, D16, P=10, sigma=5, threshold=0.2, 10k ticks

Some angular/geometric patterns forming but not a clean K. `std=3.6` — growing again (threshold calibrated for D2).

### Test 9–10: Lower lr, push capping

Tried lr=0.005 (test 9) and push capping where push is skipped if accumulated push exceeds pull (test 10). Both control scale but still produce radial patterns, no topographic structure.

### PCA analysis: Dot product word2vec vs ContinuousDrift at 10k ticks

Dot product word2vec D16: variance perfectly uniform across all 16 PCs — each explains ~6.3%, PC0+PC1 = 13%. **No structure has emerged.** ContinuousDrift D16: PC0+PC1 = 63.5% at 10k ticks. The dot-product update (`+= lr * sig * pos_j`) adds bits of random peer vectors, keeping everything uniformly random. **The dot-product math doesn't create spatial structure — it optimizes angular alignment which doesn't translate to topographic ordering.**

### Test 11–13: Angular rendering

Implemented `render_angular()` that normalizes vectors to unit length before Voronoi assignment (only direction matters, not magnitude). Tested skipgram D2 and D16 with angular rendering. Results similar to Euclidean rendering — no K structure. Confirmed: **not a visualization problem, the dot-product learning itself doesn't produce structure.**

### Test 14–15: Euclidean sigmoid (toward/away from peer), D2, K=11, k_neg=5

Switched from dot-product to Euclidean updates: `σ(dist²)` for pull strength (large when far), `σ(-dist²)` for push strength (large when close), update direction is `(pos_j - pos_i)` (toward/away). **K clearly emerging at 10k ticks.** No line collapse at D2 K11.

### Test 16: Euclidean sigmoid, D2, K=11, k_neg=5, 30k ticks

Clear K. `std=4.0, mean_norm=6.5` — positions growing and center drifting, but topographic structure is solid. **No line collapse** — the explicit negative push prevents the degenerate solutions centroid averaging produces at D2 K11.

### Test 17: Euclidean sigmoid, D16, K=25, k_neg=5, 30k ticks

Clear K reconstruction. `std=0.71` — much more stable than D2. Higher dimensions absorb push/pull energy better. Quality comparable to centroid drift (exp 00003) at similar parameters.

### Test 18: Euclidean sigmoid, D16, K=25, k_neg=5, 100k ticks

Clean K. `std=0.94` (slow growth from 0.71 at 30k), `mean_norm=2.5`. Structure solid at longer runs.

### Test 19–23: Two-vector dot product (W/C separation like real word2vec)

Implemented dual vector sets: W (positions, random init) and C (contexts, zero init). Updates always cross-set: `W[i] += σ(-W·C) * C[j]`, `C[j] += σ(-W·C) * W[i]`. Never W→W or C→C.

**Results:**
- D16 K25 30k (test 19): some dark/light clustering but no K. PCA: W PC0=37.4%, C PC0=41.8% — **creates structure** (vs 6.3% uniform with single-vector). But PC1 only 4.5% — **1D structure only**.
- D2 K11 30k (test 20-21): overflows at lr=0.05, line/stripe patterns at lr=0.005. Consistent with 1D structure.
- D16 angular render (test 22): same clustering, no improvement.
- 40px D16 (test 23): same pattern at smaller scale.

**PCA correlation analysis:** Neither W nor C PCs correlate with grid coordinates (max r=0.16). The 37% PC0 variance is non-spatial (likely pixel intensity clustering). Dual-vector dot product finds some structure but **doesn't encode spatial layout** — it separates dark from light without learning where things are.

**Scale self-regulation:** Dot product sigmoid DOES self-regulate (`std=0.4-1.6`), unlike Euclidean which grows (`std=4.0+`). The sigmoid thermostat works: large dot products → σ saturates → updates shrink. Euclidean `σ(dist²)` doesn't saturate the same way.

### Test 24: Euclidean similarity mode, D16, P=10, sigma=5, threshold=0, 30k ticks

```
venv/bin/python main.py word2vec --mode similarity -W 80 -H 80 --dims 16 --sigma 5 --P 10 --lr 0.05 --threshold 0.0 -i K_80_g.png -f 30000
```

Collapsed to `std=0.0`. With threshold=0, RBF similarity is always ≥ 0, so signal is always positive → only pull, no push → everything converges to a point.

### Test 25: Euclidean similarity mode, D16, P=10, sigma=5, threshold=0.1, 30k ticks

```
venv/bin/python main.py word2vec --mode similarity -W 80 -H 80 --dims 16 --sigma 5 --P 10 --lr 0.05 --threshold 0.1 -i K_80_g.png -f 30000 --save-every 300 -o output_5_test25
```

**Clear K!** `std=1.6`, `mean_norm≈0` (well centered). No precomputed top-K — just random peers with Gaussian RBF similarity deciding attract/repel, Euclidean sigmoid updates. This is the merge: random sampling from exp 00004 + sigmoid self-regulation from word2vec + Euclidean spatial updates. Threshold=0.1 gives enough repulsion from distant peers to prevent collapse while nearby peers (sim > 0.1) pull together.

## Findings So Far

1. **Dot product + LayerNorm collapses**: original word2vec math with LayerNorm creates binary clusters. LayerNorm forces unit variance which fights the sigmoid's self-regulation. Without LayerNorm, dot product works.
2. **Sigmoid self-regulates scale (dot product only)**: dot product embeddings stabilize at `std ≈ 0.4-1.6` — sigmoid saturates for large dot products, creating natural thermostat. Euclidean `σ(dist²)` doesn't self-regulate — positions grow unbounded.
3. **No line collapse at D2 K11**: explicit negative push prevents the degenerate solutions centroid averaging produces.
4. **Binary positive/negative selection is too coarse**: using top-K membership to decide attract vs repel ignores the actual similarity signal. Random negatives can accidentally push away true neighbors.
5. **Dot product doesn't create spatial structure**: single-vector PCA uniform (6.3% per PC). Dual-vector better (PC0=37%) but only 1D and not correlated with grid position (max r=0.16). The additive update (`+= sig * pos_j`) doesn't translate neighbor relationships into spatial layout. Angular rendering confirms it's not a display issue.
6. **Dual-vector (W/C) creates structure but not spatial**: two vector sets break the feedback loop (PC0 jumps from 6% to 37%), but the structure is non-spatial (dark/light intensity separation, not x/y position). Dot product optimizes angular alignment — "similar things point the same way" — not spatial proximity.
7. **Euclidean sigmoid works**: switching to `σ(dist²)` with toward/away updates (`pos[j] - pos[i]`) produces clear K at D2 and D16. The interpolative update creates spatial convergence that additive dot product cannot.
8. **D16 more stable than D2**: `std=0.71` at D16 vs `std=4.0` at D2 after 30k ticks. Higher dimensions distribute push/pull energy across more axes.
9. **Dot product for future multi-modal**: dot product can encode multiple independent relationships per dimension (spatial, auditory, visual) where Euclidean cannot. For single-modality spatial sorting, Euclidean is correct. For multi-context neurons, dot product's richer encoding is needed — but requires solving the training and visualization problems.

## Dot Product vs Euclidean: Position vs Meaning

Euclidean updates optimize **spatial proximity** — neurons move toward/away from peers. This is what we need for placing neurons on a 2D grid. But position is a **lossy projection of meaning**.

Consider context-dependent neurons: a neuron processes sound when eyes are closed, so it correlates with auditory neurons in one context and visual neurons in another. With Euclidean embeddings, it gets placed in between — losing both relationships. With dot product embeddings, different dimensions can encode different contexts: "close to auditory cluster along dims 0-3, close to visual cluster along dims 8-11." The full relationship structure lives in the high-dimensional embedding.

This parallels real cortex — V1 neurons encode retinotopy, orientation, ocular dominance, color, spatial frequency simultaneously. You can make a separate map for each property, but the actual embedding (full response profile) is high-dimensional.

**Path forward:**
1. **Euclidean for spatial sorting** — the current task, placing neurons on a grid
2. **Dot product for learning rich embeddings** — capture multi-context correlations where dims specialize for different modalities/contexts
3. **Visualization becomes projection** — choose which aspect to display. The same embedding could show the "auditory map" or the "visual map" depending on which dimensions you project

The hard problem is (3): a 2D grid can only show one organizing principle at a time. But the embedding preserves all relationships — visualization is just a window into one aspect.

For single-modality sorting (current experiments), Euclidean is correct. For multi-modal/multi-context sorting, dot product's ability to encode independent properties per dimension becomes essential. The sigmoid self-regulation works with both — the choice of Euclidean vs dot product is about what information the embeddings encode, not how learning is regulated.

## Why Single-Vector Dot Product Failed

The update `pos[i] += sig * pos[j]` is **additive** — it adds the peer's direction on top of the current vector. Compare with Euclidean `pos[i] += sig * (pos[j] - pos[i])` which is **interpolative** — it pulls toward the peer.

With a single vector set, additive updates create feedback loops: A chases B, B chases C, C chases A. Everyone accumulates bits of random directions → uniform noise (PCA: 6.3% per PC across all 16 dims).

Real word2vec solves this with **two separate vector sets** per word:
- **W (syn0)**: word vectors — "what am I?", randomly initialized
- **C (syn1neg)**: context vectors — "what appears near me?", initialized to zeros

Updates are always cross-set: `W[i] += sig * C[j]`, `C[j] += sig * W[i]`. Never W→W or C→C. This breaks the feedback loop — W[A] learns from C[B], which was shaped by other W vectors (D, E...), providing indirect stable signal instead of circular self-reinforcement.

The roles are architecturally distinct from the start (different arrays, different init, different roles in the training step). The center word indexes W, context/negative words index C.

For our neurons: W[i] = position embedding ("where should I be"), C[i] = context embedding ("how I influence neighbors"). W gets pulled toward C of neighbors, C gets pulled toward W of neighbors. If training creates structure, PCA on W (or W+C averaged) should recover 2D spatial layout.

## Two-Stage Pipeline: Dot Product → Euclidean

Dot product encodes **how similar things are** (relationship). Euclidean encodes **where things are** (position). These are complementary:

1. **Stage 1 — Discover similarity** (dot product): train W/C dual vectors. `W[i]·C[j]` approximates pairwise similarity. Can handle multi-modal input — different dims specialize for different modalities.
2. **Stage 2 — Spatial layout** (Euclidean): extract top-K neighbors from the dot product similarity matrix, feed to Euclidean solver (ContinuousDrift or skipgram mode). Turns abstract similarity into concrete 2D positions.

For multi-modal: dot product stage learns from sound, vision, touch separately. Euclidean stage produces spatial layout from the combined similarity. Multiple 2D maps possible by projecting different dimension subsets.

Biologically plausible: thalamus receives multi-modal input (stage 1: learn correlations) → projects to cortex (stage 2: spatial layout).

### Pipeline validation: 40px, D16, K=25, 30k+60k ticks

**Does W·C correlate with spatial similarity?**

Sampled 100k random pairs after 30k ticks of dual dot product training. Binned by grid distance:

| Grid distance | Mean W·C | Mean RBF (σ=5) |
|--------------|----------|-----------------|
| 0–1.5 | +153 | 0.975 |
| 1.5–3 | +141 | 0.897 |
| 3–5 | -86 | 0.746 |
| 5–8 | -359 | 0.450 |
| 8–12 | -517 | 0.152 |
| 12+ | -450 | ~0 |

Overall correlation r=0.10 (weak due to most pairs being far apart), but **rank ordering is correct**: nearby = positive dot product, far = negative. The crossover at dist≈3 means the dot product learned a neighborhood radius.

**Top-K overlap:** 19/25 (76%) of learned neighbors match ground truth spatial neighbors. The dot product correctly identifies most nearest neighbors despite never seeing explicit spatial coordinates.

**Pipeline test (40px):**

Stage 1: dual dot product 30k ticks → extract top-K from W·C matrix.
Stage 2: ContinuousDrift with learned top-K, 60k ticks.

Result: recognizable K but wobbly ("drunken K") — the 24% wrong neighbors introduce noise. Baseline with ground truth top-K at same 60k ticks: clean sharp K. The pipeline works but quality degrades from imperfect neighbor discovery.

**Magnitudes are large** (W·C ranges from -500 to +150) — this is why D2 dual overflowed. The dot product scale grows unbounded because the sigmoid thermostat only controls individual vector magnitudes, not their products.

## Next Steps

- [x] Test Euclidean sigmoid at longer runs (100k+) — check convergence and stability
- [x] Two-vector dot product: implement W/C separation like real word2vec
- [x] Merge Euclidean sigmoid + similarity mode: random sampling with RBF-driven attract/repel (no precomputed top-K)
- [x] Validate two-stage pipeline: dot product W·C correlates with spatial similarity (r=0.10, 76% top-K overlap)
- [ ] Compare convergence speed vs centroid drift at matched parameters
- [ ] Grid search: k_neg ratio, lr, dims
- [ ] Test at 160px and 1024px scale
- [ ] Address position growth / center drift in Euclidean mode
- [ ] Multi-modal: dot product embeddings with context-dependent correlations, per-dimension projection for visualization

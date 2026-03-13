# Thalamus Sorter — Design Principles

## What this system does

Discovers topographic spatial organization from temporal correlations alone. Neurons that fire together end up as spatial neighbors — the thalamic principle. No supervision, no precomputed neighborhoods, no ground truth layout.

## Architecture

Two-stage pipeline:

1. **Signal → Neighbor discovery**: Random anchor sampling, MSE/derivative-correlation scoring against random candidates from a rolling temporal buffer. Outputs variable-length "sentences" of correlated neurons.
2. **Sentences → Embeddings**: Dual-vector dot product skip-gram (W and C vectors), sliding window pairs, negative sampling. Outputs continuous D-dimensional embeddings that capture correlation structure.

Rendering (PCA/UMAP projection + Voronoi quantization) is purely for visualization — the embeddings are the real output.

### Why dot product (not Euclidean)

The correlation mode uses dot product with dual vectors (W, C) rather than Euclidean distance on a single position vector. This is deliberate:

- **Dot product captures richer structure.** Angular relationships in high-D can encode nuances beyond simple spatial proximity — channel identity, correlation strength, multi-scale neighborhoods. Euclidean mode forces everything into a spatial distance metric, which may discard useful information.
- **Dual vectors (W, C) prevent feedback loops.** Single-vector dot product collapses (ts-00005). Separate W and C vectors break the symmetry — W encodes "what I am", C encodes "what my neighbors look like".
- **Periodic L2 normalization** (`--normalize-every 100`) prevents magnitude blow-up. The dot product sigmoid naturally self-regulates, but explicit normalization keeps gradients stable long-term.
- **Spatial layout is derived, not learned directly.** The embeddings capture correlation structure; PCA/UMAP then projects to 2D for spatial rendering. This separation means the embeddings can encode structure that doesn't map to a flat 2D grid (e.g., channel clusters with internal spatial organization).

The Euclidean mode (`tick_euclidean`) exists as an alternative for precomputed-neighbor scenarios but is not used in the correlation pipeline.

## What works

- **Dual-vector dot product skip-gram** with periodic L2 normalization — captures correlation structure including nuances beyond pure spatial proximity
- **Sliding window pairs** from variable-length sentences: transitive inference (B correlated with A and C → B-C pair) gives 10-20x more training signal than anchor-only pairs
- **MSE-based scoring** from real image saccades: 97-98% K-neighbor precision with proper sigma/threshold
- **L2 normalization** on W and C vectors: prevents magnitude blow-up in dot product mode
- **D=8-16** embedding dims: D<3 collapses, D=8 is the practical minimum, D=16 helps for complex spatial distributions
- **k_sample proportional to N**: maintains constant sampling fraction (~3%) regardless of grid size
- **sigma proportional to grid size**: sigma ~ grid_size/10 keeps correlation reach constant

## What doesn't work

- **D=2 continuous drift** without external signal: collapses to 1D
- **Single-vector dot product** (no W/C separation): feedback loop, uniform PCA, no spatial structure
- **Low threshold / high sigma** in correlation scoring: too many false positives, noisy sentences
- **Fixed k_sample at larger grids**: k_sample=200 at 160x160 gives 65% dead anchors
- **Gaussian noise signals with T<200**: |r| < 0.005, below noise floor
- **T=200 buffer at >500k ticks**: signal churn destabilizes embeddings; T=1000 is the stable default

## Constraints to preserve

### Biological plausibility
- **Hebbian principle**: only pairwise correlations, no global optimization
- **Local random sampling**: each neuron samples random peers, not all-to-all
- **No global operations** except periodic L2 normalization (debatable; could be local normalization)
- **Derivative correlation** preferred over raw MSE: neural systems respond to changes, not absolute levels. Dead neurons (constant signal) naturally get zero contribution

### Signal processing
- **Rolling saccade buffer**: (n, T) float32 buffer refreshed by random walk over source image. T=1000 balances stability and refresh rate
- **Mean subtraction**: per-neuron, per-tick. Essential for MSE scoring to work
- **Threshold precision > recall**: high threshold (few but correct neighbors) beats low threshold (many but noisy). Skip-gram learner tolerates missing pairs but not false pairs

### Scaling rules
- `k_sample ~ 0.03 * n` (3% sampling fraction)
- `sigma ~ grid_size / 10`
- Dead anchor rate should be 10-15%. If >50%, k_sample too low. If <5%, k_sample wastefully high
- `--max-hit-ratio 0.1` as safety net: filters anchors correlated with everything (global signals like brightness flicker)

## Parameters and tradeoffs

### Anchor batch size (`batch=256`)

Number of random anchor neurons sampled per tick. Each anchor independently probes k_sample candidates. Currently fixed at 256.

- More anchors = more pairs/tick = faster convergence, but linear GPU cost
- 256 is a reasonable GPU batch size; not yet tested as a tuning lever

### k_sample (candidate pool per anchor)

Number of random candidates each anchor compares against. **Must scale linearly with n** to maintain a constant sampling fraction.

| Grid | n | k_sample | Fraction | Dead anchor rate |
|------|---|----------|----------|-----------------|
| 80x80 gray | 6,400 | 200 | 3.1% | ~15% |
| 160x160 gray | 25,600 | 800 | 3.1% | ~14% |
| 80x80 RGB | 19,200 | 600 | 3.1% | varies |
| 80x80 RGBG | 25,600 | 800 | 3.1% | varies |

**Why linear scaling**: Each anchor needs to probe enough of the population to find its true neighbors. If the fraction drops, most anchors find zero neighbors (dead ticks). At 160x160 with k_sample=200, 65% of anchors are dead (ts-00012). Restoring 3.1% fraction fixes it.

**Multi-channel note**: With RGB, ~2/3 of sampled candidates are cross-channel and will never pass MSE threshold — effectively wasting sampling budget. The useful same-channel sampling is k_sample/channels. This doesn't break convergence (saccades RGB produces same pair rate as grayscale) but means RGB is inherently less efficient per sample.

**TODO**: Auto-adaptive k_sample — track dead anchor rate, double k_sample if >15%, halve if <5%.

### MSE threshold

Maximum MSE for a candidate to be accepted as a neighbor. Controls precision vs pair volume.

- **Too strict** (0.02 on garden.png): 74% near-pixel hit rate but only 244 pairs/tick — pair starvation
- **Too loose** (0.08): 52k pairs/tick but 0.4% precision — noise overwhelms signal
- **Sweet spot**: depends on image. Saccades.png works at 0.02; garden.png needs 0.04-0.05

The skip-gram sliding window is noise-tolerant (ts-00009: transitive pairs help even with false positives), so erring toward more pairs at lower precision is generally better than pair starvation.

**Tradeoff**: `precision × pairs/tick` — want both high enough. If pairs/tick < 1000, the learner is starving regardless of precision.

### Saccade step size

Pixels the random walk moves per frame in the source image.

- **Large step (50)**: covers more of the image → diverse training signal, but weakens local temporal correlation between neighboring pixels
- **Small step (5)**: stronger local correlations → more pairs pass threshold, but explores the image slowly (may not see the full image in T=1000 frames)

**Image-dependent**: Smooth images (saccades.png: warm browns) tolerate large steps. High-diversity images (garden.png: colorful flowers) need small steps to maintain neighbor correlation above threshold.

| Image | Step | Near MSE | Near hit% | Pairs/tick |
|-------|------|----------|-----------|-----------|
| garden | 50 | 0.018 | 47% | 81 |
| garden | 5 | 0.015 | 74% | 244 |

### Embedding dimensions (D)

Number of dimensions in the W and C vectors.

- **D < 3**: collapses, insufficient capacity
- **D = 8**: practical minimum. Encodes spatial structure well for simple layouts. All grayscale experiments use D=8
- **D = 16**: helps for complex spatial distributions. Garden.png R channel improved 23→51% <5px going 8→16D
- **D > 16**: not yet tested, likely diminishing returns

**Multi-channel tradeoff**: With RGB, the embedding must encode both channel identity and within-channel spatial position. More dims give more room for both. 8D works when each channel has simple spatial structure; 16D needed for complex channels.

### Buffer size (T)

Number of temporal frames in the rolling signal buffer.

- **T=200**: fast refresh, good for early convergence, but noisy MSE at >500k ticks
- **T=1000**: stable default, balances refresh rate and MSE reliability
- **T=2000**: best local structure (98% <5px) but non-planar global layout

**Tradeoff**: longer buffer = more stable MSE estimates = better discrimination, but slower adaptation to signal changes and more memory (n × T float32).

### Learning rate (lr=0.001)

Skip-gram update step size. Currently constant throughout training.

- All successful runs use lr=0.001
- No schedule tested yet — potential improvement for late-stage refinement
- Too high: oscillation, structure doesn't stabilize
- Too low: slow convergence

### Negative sampling (k_neg=5)

Number of random negative samples per positive pair. Push-away force prevents embedding collapse.

- k_neg=5 used across all experiments, not heavily tuned
- Lower: faster per tick but may not prevent collapse
- Higher: stronger repulsion, may over-separate

### Normalization interval (normalize_every=100)

How often to L2-normalize W and C vectors.

- Prevents magnitude blow-up in dot product mode
- Too frequent: constrains learning, may slow convergence
- Too infrequent: magnitudes grow, sigmoid saturates, gradients vanish
- 100 works well empirically

### Sliding window size (window=5)

Context window for generating skip-gram pairs from sentences.

- Each sentence = [anchor, neighbor₁, neighbor₂, ...]
- Window=5: each token pairs with up to 5 tokens on each side
- Larger window = more transitive pairs = more training signal per sentence
- Too large: distant transitive pairs add noise
- 5 is the default word2vec window, works well here

## Multi-channel behavior (ts-00013)

When multiple signal channels (R, G, B) are fed as independent neurons:
- **Channel identity always dominates** spatial proximity in embeddings, regardless of inter-channel correlation
- High inter-channel correlation (saccades.png, r=0.95): instant channel separation, no spatial mixing
- Low inter-channel correlation (garden.png, r=0.48-0.74): delayed but still total channel separation by 500k ticks
- Within-channel spatial sorting quality varies by channel complexity (simple spatial structure sorts faster)
- Cross-channel pixel proximity is never learned — the solver treats each channel as an independent spatial map

## Open questions

- Can we force cross-channel spatial structure (e.g., shared spatial dimensions + channel-specific dimensions)?
- Optimal learning rate schedule? All runs use constant lr=0.001
- Auto-adaptive k_sample based on dead anchor rate?
- Scaling beyond 160x160? Estimated need: k_sample ~ 0.03n, training time ~ O(n * ticks)

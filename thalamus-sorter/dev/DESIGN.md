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

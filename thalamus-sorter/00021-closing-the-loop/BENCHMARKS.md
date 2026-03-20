# ts-00021 Feedback Loop Benchmarks

## Benchmark 1: Shuffled Feedback Control

**Effort:** minimal (one CLI flag)
**Priority:** first — strongest null hypothesis test

Add `--feedback-shuffle`. Each tick, write a random permutation of column outputs
to feedback rows instead of the real mapping. Same value distribution, destroyed
temporal coupling.

**What it proves:** If real feedback beats shuffled, the temporal structure matters.

**Metric:** Compare sensory KNN spatial accuracy and column confidence:
- real feedback vs shuffled vs no feedback
- Pass: spatial_accuracy(real) - spatial_accuracy(shuffled) > 0.1 at convergence

## Benchmark 2: Feedback Neuron Embedding Quality

**Effort:** post-hoc analysis script
**Priority:** second — pure measurement, no training changes

After training, analyze feedback neuron positions in embedding space:
1. For feedback neuron f_i from column c output j, cosine similarity between
   f_i's embedding and cluster c's sensory centroid. High = feedback embeds near
   its driving cluster.
2. KNN composition: fraction of each feedback neuron's KNN neighbors that are
   (a) other feedback from same column, (b) sensory from driving cluster,
   (c) unrelated. Informative pattern: (b) >> (c).
3. Mantel correlation between feedback-neuron pairwise distance matrix and
   centroid pairwise distance matrix. Tests whether feedback spatial structure
   mirrors cluster proximity.

**Pass criteria:**
- Mean feedback-sensory cosine similarity > 0.3
- Mantel correlation > 0.2

## Benchmark 3: Frozen Columns

**Effort:** one attribute in ColumnManager
**Priority:** third

Add `--feedback-frozen`. Columns produce outputs but skip Hebbian update. Tests
whether column learning contributes or any nonlinear projection suffices.

**Three conditions:** (a) learned, (b) frozen, (c) no feedback.
- (a) >> (b) ≈ (c): column learning essential
- (a) ≈ (b) >> (c): any projection helps
- (a) ≈ (b) ≈ (c): feedback is noise

## Benchmark 4: Mutual Information with Future Input

**Effort:** post-hoc analysis on signal buffer
**Priority:** fourth

After warmup, sample consecutive tick pairs (t, t+1). Measure correlation between
feedback fb_t and sensory change delta = s_{t+1} - s_t. Use top-5 PCs to avoid
dimensionality issues.

**What it proves:** Feedback carries predictive information about future sensory state.

**Pass:** Mean absolute correlation between top-5 PCs of delta and top-5 PCs of
fb_t > 0.1 (real) vs < 0.05 (shuffled control).

## Benchmark 5: Synthetic Hierarchical Signal

**Effort:** new signal generator + analysis
**Priority:** last — deepest test, most code

Generate source image with known group structure: spatially non-adjacent pixel
regions that co-activate with temporal delay. Single neuron can't detect the
conjunction; column over the right cluster can.

Measure group-coherence score = fraction of feedback neuron KNN neighbors sharing
same source group. Random = 1/n_groups, perfect = 1.0.

**Pass:** Group coherence > 2x random baseline after 10k ticks.

## Known Risks

- **Signal scale:** Column outputs are softmax probabilities [0,1]. Temporal
  derivatives may have very low variance, causing tick_correlation to never find
  feedback neurons as neighbors. Check pair counts early.
- **Bootstrap:** At tick 1, clusters are garbage → columns are garbage → feedback
  is random. Sensory signal breaks symmetry. Monitor whether feedback embeddings
  differentiate only after clusters stabilize (~1-2k ticks).
- **Sampling budget:** With K << n_sensory, few anchors per tick land on feedback
  neurons. May need longer runs or separate anchor budget.

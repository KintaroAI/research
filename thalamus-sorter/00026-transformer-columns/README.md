# ts-00026: Transformer Columns

**Date:** 2026-03-29
**Status:** In progress
**Source:** `exp/ts-00026`
**Depends on:** ts-00025 (lateral connections, visual field, blocks, cluster cap)

## Goal

Replace ConscienceColumn with a transformer-based column that learns temporal patterns via self-supervised next-frame prediction. Each column is its own tiny transformer — no shared weights.

## Motivation

ConscienceColumn collapses the signal window to a temporal mean, losing all temporal structure. The conscience threshold prevents winner collapse but causes artificial winner rotation that disrupts motor output. A transformer column can:

1. **Attend over time** — learn "input 5 spiked 3 ticks ago AND input 12 is high now → output 2"
2. **Self-supervised training** — predict next input frame, no labels needed
3. **Natural surprise signal** — prediction error measures novelty
4. **No collapse hack** — prediction pressure drives category differentiation naturally

## Architecture: PredictiveColumn

```
Per column (m independent instances, batched via leading m dimension):

1. TEMPORAL ENCODER — 1-layer causal transformer
   Input: (window, d_model) signal trace, d_model = max_inputs
   Positional embeddings (learnable, per-column)
   Pre-LN attention (n_heads, causal mask) + residual
   Pre-LN FFN (GELU, 4x expansion) + residual
   Output: z = last hidden state (d_model,)

2. PROTOTYPE HEAD — map summary to categories
   m learned category embeddings c_1..c_k (m, n_outputs, d_model)
   sim = z · c_k, p = softmax(sim / temperature)

3. FUTURE PREDICTION — from category bottleneck
   z_q = Σ p_i * c_i (category-weighted state)
   predicted_next = z_q @ W_pred + b_pred
   Loss: MSE(prev_prediction, actual current frame)
   Surprise: per-column prediction error

4. BALANCE REGULARIZER
   L_sharp: minimize per-sample entropy (clear winners)
   L_balance: maximize batch-average entropy (use all categories)
```

### Why predict from category bottleneck (not raw z)

If prediction comes from z directly, categories float freely — nothing forces them
to carry predictive information. By predicting from z_q = Σ p_i * c_i, categories
MUST encode "what happens next" to minimize prediction loss. Each category becomes
a distinct "next-frame template."

### Key design decisions

- **Independent weights per column** — each column sees different neurons with different temporal patterns. All weights have leading `m` dimension, all linear ops via `torch.bmm`. ~5.7K params/column, ~1.1M total for m=200.
- **1-layer causal transformer** — window is small (4-10 steps), one layer is sufficient. Causal masking preserves temporal order.
- **Learnable positional embeddings** — (m, window, d_model), independent per column.
- **Category embeddings separate from encoder** — (m, n_outputs, d_model) learned parameters. Dot product + softmax for categorization.
- **Feature masking** — empty slots (slot_map=-1) zeroed by `_gather_input()`. MSE loss masked to valid slots only.
- **Three loss terms** — L_pred (MSE, main driver) + lambda_sharp * L_sharp + lambda_balance * L_balance
- **Online training** — one forward+backward per tick. Adam optimizer, gradient clipping at 1.0.
- **Cross-tick prediction** — store prediction, compare against actual input next tick. No wasted observations.

### Interface

Same as ColumnBase: `wire()`, `unwire()`, `tick()`, `get_outputs()`, `save()`, `load_state()`.
Plus `get_surprise()` → (m,) prediction error per column.
Drop-in replacement via `--column-type predictive`.

### Parameter count (d_model=30, d_ff=120, n_outputs=4, window=10)

| Component | Params |
|-----------|--------|
| pos_emb | 300 |
| W_qkv + W_proj | 3,600 |
| ln1, ln2 (gain+bias) | 120 |
| W_fc1, b_fc1 | 3,720 |
| W_fc2, b_fc2 | 3,630 |
| W_pred, b_pred | 930 |
| cat_embs | 120 |
| **Total per column** | **~12.4K** |
| **Total m=200** | **~2.5M** |

### CLI

```bash
python main.py word2vec --column-type predictive \
    --column-outputs 4 --column-max-inputs 30 --column-window 10 \
    --column-lr 0.001 --column-temperature 0.5 \
    --column-n-heads 2 --column-lambda-sharp 0.01 --column-lambda-balance 0.1 \
    --signal-source forage ...
```

New args: `--column-n-heads`, `--column-lambda-sharp`, `--column-lambda-balance`

## Comparison

| | Conscience | Predictive | Recon |
|---|---|---|---|
| Temporal | Mean only (lost) | Full attention over window | Full attention (context) |
| Training | Hebbian (no backprop) | SGD (backprop) | SGD (backprop) |
| Loss | None (similarity match) | Next-frame prediction | Spatial reconstruction |
| Categories | Spatial pattern match | Temporal dynamics | Spatial via bottleneck |
| Surprise | None | Prediction error | Reconstruction error |
| Collapse prevention | Theta threshold | Entropy reg + theta | Entropy reg + theta |
| Compute (m=1000) | ~15ms/tick | ~15ms/tick | ~15ms/tick |

## Implementation

- `column_manager.py`:
  - `ColumnBase` — shared output rotation (conscience theta, tiredness, WTA modes)
  - `PredictiveColumn(ColumnBase)` — temporal encoder + next-frame prediction bottleneck
  - `ReconColumn(ColumnBase)` — temporal encoder + spatial reconstruction bottleneck
- `cluster_manager.py`: `predictive` and `recon` branches
- `main.py`: CLI args for all shared (alpha, wta, tiredness) and per-type params

## Results

### Pattern classification (single column, 10k train, 1k eval)

| | Conscience | Predictive α=0 | Recon α=0 |
|---|---|---|---|
| Unique outputs | 2/4 (37-42%) | 2/4 (31-35%) | **3/4 (76-96%)** |

Recon dominates spatial pattern tasks — reconstruction bottleneck forces distinct categories.

### XOR (single + chain, 10k train, 1k eval, α=0)

| | Conscience | Predictive | Recon |
|---|---|---|---|
| Single AND | 100% | 75% | **100%** |
| Single OR | 50% | 75% | **100%** |
| Chain XOR | **100%** | 100%* (sep=0.001) | 75% |

### Forage benchmark (1M ticks, m=1000, mi=8, cap=8, α=0.01, confidence WTA)

| Metric | Conscience | Predictive | Recon |
|--------|-----------|------------|-------|
| **Collections** | **1206** | 281 | 103 |
| Dense phase | **125** | 101 | 20 |
| Sparse phase | **1081** | 180 | 83 |
| Clusters alive | 433/1000 | 447/1000 | 439/1000 |
| Cluster stability | 0.288 | 0.979 | **0.995** |
| Total jumps | 33.4M | 131K | 79K |
| **Feature correlations:** | | | |
| pos_x | 0.772 | 0.886 | 0.870 |
| pos_y | 0.894 | 0.932 | **0.941** |
| target_x | 0.860 | 0.959 | **0.974** |
| dir_xn | 0.895 | 0.969 | **0.978** |
| hunger | 0.823 | **0.915** | 0.913 |
| restless | 0.928 | 0.972 | **0.997** |

**Key finding:** Predictive columns learn significantly better representations
(feature correlations 0.88-0.97 vs 0.77-0.93) but collect 4x less food.
The difference is exploration: conscience has massive cluster churn (33M jumps,
stability 0.29) driving random exploration, while predictive is very stable
(131K jumps, stability 0.98) — the agent doesn't explore enough to use its
better representations.

**Why conscience explores more:** The 33 jumps/tick aren't from genuine embedding
instability — they're an artifact of conscience rotation cascading through the
feedback loop. Theta forces winner rotation → feedback neurons carry different
signals → embedding correlations shift → cluster reassignment. The "exploration"
that drives food collection is mechanical perturbation from conscience, not
intelligent behavior. Predictive columns produce stable outputs → stable feedback
→ frozen embeddings → no churn → no exploration.

**Implication:** Neither pure conscience (random exploration, weak representations)
nor pure predictive (good representations, no exploration) is sufficient. Possible
directions:

1. **Dual-column hybrid:** conscience column explores (drives motor via feedback
   perturbation), predictive/recon column memorizes (learns state representations).
   Hunger modulates the blend — hungry → conscience drives (explore), satiated →
   predictive drives (exploit known locations).
2. **Surprise-driven exploration:** use prediction error as an exploration signal.
   High surprise → increase motor noise or override with random action. Low
   surprise → trust the learned policy. This makes exploration intentional rather
   than mechanical.
3. **Column stacking:** conscience output feeds into predictive column as input.
   Conscience provides the rotation signal that predictive uses as temporal context
   for categorization.
4. **Signal-dependent frequency modulation:** instead of passing raw signal values
   to neurons, modulate a carrier sine wave's frequency based on signal strength.
   Zero signal → rare sporadic spikes (low frequency). Strong signal → rapid
   firing (high frequency). This gives derivative-correlation something to work
   with even for near-zero signals (sporadic spikes still produce temporal
   variance), and naturally encodes magnitude as firing rate — closer to
   biological rate coding. The thalamus layer would convert continuous values
   to spike-like temporal patterns before they reach the embedding/column system.
5. **Global neuromodulator (dopamine-like):** on food collection, broadcast a
   scalar reward to all columns. Positive reward: freeze current outputs briefly
   (what you did worked — hold it) or apply eligibility traces (ColumnManager
   already has `--eligibility --trace-decay`). Negative (hunger rising): increase
   exploration. Each column accumulates a trace of recent winner directions;
   reward triggers the trace to update prototypes. Like dopamine — "whatever you
   were just doing, do more of that." The traces decay so only recent actions
   get credit. Already partly implemented, just needs wiring to hunger events.
6. **Prediction error as system-wide signal:** columns that predict their inputs
   well are "understanding their corner." Columns with chronic high error are
   miswired or haven't learned. Could drive rewiring — high-error columns steal
   neurons from low-error ones, or get reassigned to different clusters. Makes
   prediction error useful beyond just column training.
6. **Energy minimization / spike efficiency:** neurons "want" to fire less.
   Connections to neurons that fire right after you = good (predicted the next
   spike, efficient). Connections to uncorrelated neurons = wasted energy.
   Conscience rotation approximates this: rotating outputs = "spending energy
   trying all categories." A column that settles = energy-efficient. The problem
   is conscience forces rotation even when settling is correct. True energy
   minimization would let columns settle when they've found a good category and
   only rotate when prediction error is high.
7. **Global-to-local sorting transition:** early on, global skip-gram discovers
   coarse structure (which neurons correlate at all). Over time, decay global
   anchor rate (`k_sample` or `anchor_batches`) and increase knn2-driven local
   refinement (friend-of-friend-of-friend). Each cluster becomes its own local
   optimizer, recruiting based on neighborhood rather than global correlation.
   Naturally leads to parallelism — clusters run independently once global
   topology stabilizes. Implementation: time-varying schedule on anchor_batches
   (decay) + knn2 neighborhood depth (expand). Could also increase
   `cluster_split_every` over time to let clusters stabilize.
8. **Detector columns (not classifiers):** current columns are classifiers —
   forced to output a category every tick. Instead: each output is an independent
   detector with a learned template. Accumulates match evidence over the window.
   Fires when match crosses threshold. Below threshold → silence. Multiple
   detectors can fire simultaneously, or none. No softmax, no forced competition.
   Closer to biological neurons (threshold, accumulate, fire, refractory).
9. **Self-supervised optimization without reward:** the model should become smarter
   by optimizing an intrinsic objective — reducing surprise in predicting the
   future. No external reward needed. Reward (food, hunger) can incentivize
   direction but the core learning should be: "I want to predict what happens
   next, and I restructure myself to predict better." This is the free energy
   principle — organisms minimize prediction error about their sensory inputs.
   The column's job isn't to categorize — it's to build an internal model that
   predicts its inputs. Categories emerge as a side effect of compression
   (you need categories to predict efficiently through the bottleneck). Reward
   then just biases WHICH predictions matter more (hunger-related predictions
   get amplified, irrelevant ones get suppressed).

## Recent progress

### Unified TransformerColumn

PredictiveColumn and ReconColumn refactored into a single `TransformerColumn`
base with `loss_mode='predictive'|'recon'`. Thin wrappers for backward compat.
Removes ~340 lines of duplication.

### Anti-collapse fixes (from code review)

1. **Cosine logits** — normalize z and cat_embs before dot product. Routing by
   direction, not magnitude. Prevents one category dominating via embedding norm.
2. **Normalized bottleneck values** — `z_q = p @ c_normalized`, not raw cat_embs.
3. **EMA balance loss** — per-column category usage tracked across many ticks
   (decay=0.9), not noisy per-window estimate. Live gradient (not detached).
4. **Orthogonality loss** — `||C_n @ C_n^T - I||^2` pushes categories apart.
5. **Temperature 2.0** (was 0.2/0.5). Prevents early softmax saturation.
6. **L_sharp disabled** (was 0.01). Entropy minimization actively causes collapse.
7. **Unrotated prediction path** — rotation only on external outputs, not on
   stored predictions. Prevents train/output mismatch.
8. **Forward rerun after optimizer.step()** — consistent post-update outputs.

### ConsciencePredictiveColumn (hybrid)

Combines conscience state dynamics with predictive validation:

- **State head**: transformer encoder → state descriptor (normalized z + scaled
  input), cosine similarity to prototypes, conscience rotation
- **Per-category predictors** (W_pred_bank): each category has its own transition
  model predicting next frame. Soft mixture weighted by state probabilities.
- **Nudge loss**: per-category prediction errors build corrective target q that
  pushes state toward categories that both match now AND predict well:
  `q = softmax(log(p) - β * relative_error)`
- **Reconstruction anchor** (L_now): bottleneck reconstruction of current frame
  keeps categories grounded in spatial patterns
- **Hebbian prototype pull**: winner prototype moves toward state (fast, local),
  while encoder trains via gradient (slow, global)

Key tuning parameters:
- `--column-lambda-now` — reconstruction vs prediction balance (default 0.25, try 1.0)
- `--column-lambda-nudge` — predictive validation strength (default 0.10)
- `--column-alpha` — conscience rotation rate
- `temperature` — 1.5 default for hybrid (between conscience's 0.2 and pure predictive's 2.0)

### Forage environment changes

- Spasm decay to zero (no floor) — model must learn to drive itself
- Motor threshold 0.01 (was 0.3) — any motor signal contributes
- Muscle tiredness disabled (tire_rate=0) — motors can push indefinitely
- Motor-suppressed spasms — strong motor output reduces random walk by 90%
- Ground texture for visual field (behind --forage-visual-field)
- Clock neurons (behind --forage-clocks)

### Fast cluster init

- Replaced k-means with random assignment (O(n) vs O(n*m*iters))
- Vectorized knn2 init (no O(m²) loop)
- Skip initial neuron wiring (--skip-init-wiring) for large m
- m=10K init now instant. m=100K possible but slow per-tick.

### Forage: hunger-modulated alpha (1M ticks, conscience, m=1000)

| Metric | Fixed α=0.01 | Hunger × α |
|--------|-------------|------------|
| Collections | **1206** | 928 |
| pos_x | 0.772 | **0.916** |
| target_x | 0.860 | **0.978** |
| dir_xn | 0.895 | **0.986** |
| hunger | 0.823 | **0.929** |
| restless | 0.928 | **0.983** |

Same tradeoff: reducing rotation (α→0 when fed) improves representations but
reduces exploration and food collection. The exploration-exploitation tension
is consistent across all approaches.

### Layers benchmark (100k ticks, m=100, mi=8, cap=8, α=0.01, confidence WTA)

| Metric | Conscience | Predictive |
|--------|-----------|------------|
| V1 clusters | 38 | 92 |
| V2 clusters | **53** | 8 |
| V3 clusters | **9** | 0 |
| L1 mean r | **0.770** | 0.764 |
| L2 mean r | **0.732** | 0.669 |
| L3 mean r | **0.671** | 0.548 |
| L2 detected by | V1+V2 mix | mostly V1 |
| L3 detected by | V1 | V1 |

| Predictive α=0.01 | 91 | 9 | 0 | 0.762 | 0.648 | 0.484 |

**Key finding:** Predictive columns produce almost no hierarchy — 91-92/100 clusters
are V1, only 8-9 V2, zero V3. Conscience creates 38 V1 → 53 V2 → 9 V3. Adding
base class alpha to predictive barely helps (9 vs 8 V2) because alpha only rotates
the output softmax — it doesn't affect encoder training. In conscience, theta
affects BOTH which prototype gets updated (learning) and which output wins,
creating deep churn that propagates through the feedback loop. In predictive,
the encoder trains identically regardless of alpha, so feedback signals stay
stable and don't separate into distinct clusters. Hierarchy requires churn in
the learning dynamics, not just the output.

### Forage: conscience max_k=4 vs max_k=2 (1M ticks, 14×14, m=400)

| Metric | max_k=2 | max_k=4 |
|--------|---------|---------|
| **Collections** | 1119 | **2126** |
| Dense phase | 124 | **262** |
| Sparse phase | 995 | **1864** |
| Clusters alive | 54/400 | 74/400 |
| Total jumps | 23.9M | **39.9M** |
| Total switches | 4.3M | **13.2M** |
| Splits | 146K | **132K** |
| hunger r | 0.84 | **0.95** |
| dir_xp r | — | **0.88** |

max_k=4 nearly doubled food collection. The ring buffer tracks each neuron's
4 most recent cluster memberships, but only the primary (pointer) cluster gets
the neuron wired to its column. The deeper ring means neurons switch primary
cluster more freely — the streaming update checks all ring entries before
deciding where a neuron belongs, creating more migration opportunities.

The 40M jumps (vs 110K) drive massive feedback signal variation which is what
produces exploration. Only 74/400 clusters survive because neurons spread thin,
but survivors are well-connected. Hunger correlation 0.95 — the system tracks
internal state very well despite (because of?) the churn.

### Forage: all-ring wiring (1M ticks, 14×14, m=400)

Wire neurons to ALL ring clusters, not just primary. With max_k=2, a neuron
in clusters [5, 12] feeds signal to both columns instead of only cluster 5.

| Metric | k=2 primary | k=2 all-ring | k=4 primary | k=4 all-ring | k=8 all-ring |
|--------|-------------|--------------|-------------|--------------|--------------|
| **Collections** | 1119 | **2026** | 2126 | 2002 | 1890 |
| Dense phase | 124 | 223 | 262 | 204 | 220 |
| Sparse phase | 995 | 1803 | 1864 | 1798 | 1670 |
| Clusters alive | 54/400 | 72/400 | 74/400 | 68/400 | 68/400 |
| Initial wirings | — | 1796 | — | 1796 | 1796 |
| Total jumps | 23.9M | 47.0M | 39.9M | 45.2M | 28.1M |
| Total switches | 4.3M | 10.1M | 13.2M | 11.5M | 6.5M |
| Splits | 146K | 168K | 132K | 215K | 259K |
| Contiguity | — | 0.668 | — | 0.549 | 0.695 |

All-ring k=2 (2026) nearly matches primary-only k=4 (2126) — the sweet spot.
Deeper rings hurt with max_inputs=8: k=4 all-ring drops to 2002, k=8 to 1890.

### All-ring with larger columns (mi=40, cap=40, 1M ticks, 14×14, m=400)

Test whether max_inputs=8 was the bottleneck for deeper rings.

| Metric | k=8 ar mi=8 | k=8 ar mi=40 | k=4 ar mi=8 | k=4 ar mi=40 |
|--------|-------------|--------------|-------------|--------------|
| **Collections** | 1890 | 1815 | 2002 | **2084** |
| Clusters alive | 68/400 | 92/400 | 68/400 | 76/400 |
| Total jumps | 28.1M | 17.5M | 45.2M | 7.4M |
| Total switches | 6.5M | 5.6M | 11.5M | 1.9M |
| Splits | 259K | 211K | 215K | 98K |
| Contiguity | 0.695 | 0.769 | 0.549 | 0.704 |
| Diameter | 3.4 | 2.2 | — | 3.2 |

Larger columns don't help — k=8 mi=40 dropped further (1815 vs 1890).
k=4 mi=40 improved slightly (2084 vs 2002) but with dramatically fewer jumps
(7.4M vs 45.2M). The bigger columns absorb more neurons so fewer get evicted,
reducing churn. Still doesn't beat k=2 all-ring mi=8 (2026) or k=4 primary mi=8
(2126). The deeper ring itself suppresses exploration by keeping neurons
connected to old clusters too long.

### Forage: hybrid column comparison (1M ticks, 14×14, m=400)

| Metric | Conscience | Hybrid (pred+recon) | Hybrid (recon only) |
|--------|-----------|---------------------|---------------------|
| Collections | **1119** | 126 | 373 |
| Clusters alive | 54/400 | 84/400 | 86/400 |
| Stability | 0.12 | 0.998 | 1.000 |
| Total jumps | 23.9M | 14K | 26K |
| pos_x r | **0.81** | 0.66 | 0.47 |
| hunger r | **0.84** | 0.56 | 0.47 |

**Key finding:** the transformer encoder hurts in the forage setting. It
stabilizes representations so completely that exploration dies. Plain
conscience with simple cosine similarity produces more churn, more
exploration, more food. The encoder solves the wrong problem — stable
consistent categories — when what the system needs is variation and
exploration through the feedback loop.

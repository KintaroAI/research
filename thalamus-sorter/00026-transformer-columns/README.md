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

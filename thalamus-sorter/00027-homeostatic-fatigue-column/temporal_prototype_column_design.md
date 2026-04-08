# Temporal Prototype Column — Design Document

Pseudocode design for the next-generation column, building on
`ConscienceHomeostaticFatigueColumn` with targeted additions:
loser repulsion, rolling decorrelation, future prediction.

## 1. Column state

```python
class TemporalPrototypeColumn:
    def __init__(
        self,
        n_in,           # number of inputs
        n_out,          # number of outputs / prototypes
        window,         # temporal window length
        d_desc,         # descriptor / latent dimension
        k_active,       # max active outputs
        corr_window,    # history window for decorrelation
    ):
        # -----------------------------
        # Streaming input state
        # -----------------------------
        self.x_hist = RingBuffer(shape=(window, n_in))
        self.prev_desc = zeros(d_desc)
        self.prev_output = zeros(n_out)

        # -----------------------------
        # Descriptor / latent stage
        # Option A: hand-crafted descriptor
        # Option B: learned temporal encoder
        # For now keep both as interchangeable blocks
        # -----------------------------
        self.use_learned_encoder = False
        self.encoder = TinyTemporalEncoder(...)   # optional later

        # -----------------------------
        # Prototypes / output units
        # Each output owns one prototype vector
        # -----------------------------
        self.prototypes = l2_normalize(random(n_out, d_desc), axis=1)

        # -----------------------------
        # Homeostasis / anti-monopoly
        # -----------------------------
        self.usage = zeros(n_out)          # slow running usage
        self.theta = zeros(n_out)          # homeostatic threshold / bias
        self.fatigue = zeros(n_out)        # fast suppression after firing

        # -----------------------------
        # Output history for decorrelation
        # -----------------------------
        self.a_hist = RingBuffer(shape=(corr_window, n_out))

        # -----------------------------
        # Predictor: from current state/output to next descriptor
        # This can be tiny linear/MLP
        # -----------------------------
        self.predictor = NextStateHead(
            in_dim=d_desc + n_out,
            out_dim=d_desc,
        )

        # prediction made at previous tick, to be checked now
        self.pending_pred = None

        # -----------------------------
        # Running losses / diagnostics
        # -----------------------------
        self.last_pred_error = 0.0
        self.last_decorrelation = 0.0
        self.last_margin_error = 0.0
```

## 2. Descriptor block

This is where current input history becomes the latent vector the
prototypes see.

### Version close to current HF column

```python
def build_descriptor_from_history(x_hist):
    # x_hist: [window, n_in]

    current = x_hist[-1]                   # [n_in]
    mean_   = mean(x_hist, axis=0)         # [n_in]
    delta   = x_hist[-1] - x_hist[-2]      # [n_in]  if enough history else zeros

    # normalize each part separately
    current = l2_normalize(current)
    mean_   = l2_normalize(mean_)
    delta   = l2_normalize(delta)

    desc = concat([current, mean_, delta])  # [3 * n_in]
    desc = l2_normalize(desc)
    return desc
```

### Optional later version

```python
def build_descriptor_learned(x_hist):
    # Tiny temporal encoder over the window
    # Example: MLP over flattened window, or small 1D temporal conv, or GRU
    desc = encoder(x_hist)                 # [d_desc]
    desc = l2_normalize(desc)
    return desc
```

## 3. Per-tick forward pass

```python
def step(self, x_t):
    # -----------------------------------
    # A. ingest new input
    # -----------------------------------
    self.x_hist.push(x_t)

    if not self.x_hist.is_full_enough():
        return zeros(self.n_out)  # not enough context yet

    # -----------------------------------
    # B. build descriptor / latent
    # -----------------------------------
    if self.use_learned_encoder:
        desc_t = build_descriptor_learned(self.x_hist)
    else:
        desc_t = build_descriptor_from_history(self.x_hist)

    # -----------------------------------
    # C. check last tick's prediction
    # -----------------------------------
    pred_loss = 0.0
    if self.pending_pred is not None:
        pred_loss = mse(self.pending_pred, desc_t)
        self.last_pred_error = pred_loss

    # -----------------------------------
    # D. compare descriptor to prototypes
    # cosine similarity
    # -----------------------------------
    sims = cosine_similarity(self.prototypes, desc_t)   # [n_out]

    # -----------------------------------
    # E. apply homeostasis + fatigue
    # higher theta/fatigue -> harder to activate
    # -----------------------------------
    scores = sims - self.theta - self.fatigue           # [n_out]

    # -----------------------------------
    # F. sparse multi-output activation
    # keep this abstract so you can swap:
    # - top-k masked softmax
    # - sparsemax
    # - entmax
    # -----------------------------------
    a_t, active_idx = sparse_activation(scores, k=self.k_active)
    # a_t: [n_out], sparse-ish, sums to ~1
    # active_idx: active outputs

    # -----------------------------------
    # G. identify confusing losers
    # only losers that were close contenders
    # not all non-winners
    # -----------------------------------
    loser_idx = select_confusing_losers(scores, active_idx)
    # e.g. top 1-3 losers near the winning set
```

## 4. Emit output and schedule next prediction

```python
    # -----------------------------------
    # H. emit output
    # -----------------------------------
    y_t = a_t

    # -----------------------------------
    # I. create a prediction for next step
    # predict next descriptor from current desc + current output
    # -----------------------------------
    pred_input = concat([desc_t, a_t])          # [d_desc + n_out]
    self.pending_pred = self.predictor(pred_input)  # [d_desc]
```

## 5. Online updates

### 5.1 Pull winners toward current descriptor

```python
    # -----------------------------------
    # J. Hebbian pull for active prototypes
    # winners move toward current descriptor
    # -----------------------------------
    for j in active_idx:
        pull = a_t[j] * (desc_t - self.prototypes[j])
        self.prototypes[j] += lr_pos * pull
```

### 5.2 Repel only confusing losers

```python
    # -----------------------------------
    # K. targeted loser repulsion
    # only prototypes that almost matched but lost
    # -----------------------------------
    for j in loser_idx:
        repel = (desc_t - self.prototypes[j])
        self.prototypes[j] -= lr_neg * loser_weight(scores[j]) * repel
```

### 5.3 Optional pairwise prototype repulsion

```python
    # -----------------------------------
    # L. optional pairwise prototype repulsion
    # if two prototypes become too similar, push apart slightly
    # -----------------------------------
    for i in range(n_out):
        for j in range(i + 1, n_out):
            sim_ij = cosine(self.prototypes[i], self.prototypes[j])
            if sim_ij > proto_sim_threshold:
                delta = self.prototypes[i] - self.prototypes[j]
                self.prototypes[i] += lr_proto_repulse * delta
                self.prototypes[j] -= lr_proto_repulse * delta
```

### 5.4 Renormalize prototypes

```python
    self.prototypes = l2_normalize(self.prototypes, axis=1)
```

## 6. Homeostasis and fatigue updates

```python
    # -----------------------------------
    # M. update fast fatigue
    # recent activity temporarily suppresses reuse
    # -----------------------------------
    self.fatigue *= fatigue_decay
    self.fatigue += fatigue_gain * a_t

    # -----------------------------------
    # N. update slow usage statistics
    # -----------------------------------
    self.usage = usage_decay * self.usage + (1 - usage_decay) * a_t

    # -----------------------------------
    # O. homeostatic threshold update
    # push outputs toward target usage
    # target could be 1/n_out or something looser
    # -----------------------------------
    target = ones(n_out) / n_out
    self.theta += lr_homeo * (self.usage - target)
```

Interpretation:

- **fatigue** says: "you just fired, cool down a bit"
- **theta** says: "over long time, don't dominate"

## 7. Rolling decorrelation on output activity

Homeostasis says: "each unit should be used."
Decorrelation says: "units should not carry the same signal."

```python
    # -----------------------------------
    # P. store output history
    # -----------------------------------
    self.a_hist.push(a_t)

    # -----------------------------------
    # Q. decorrelation over rolling window
    # -----------------------------------
    decor_loss = 0.0
    if self.a_hist.is_full():
        A = self.a_hist.read()              # [Tcorr, n_out]

        # center each output dimension
        A_centered = A - mean(A, axis=0)

        # covariance / correlation between outputs
        C = covariance_matrix(A_centered)   # [n_out, n_out]

        # penalize off-diagonal correlations
        decor_loss = offdiag_l2(C)

        self.last_decorrelation = decor_loss

        # conceptual update:
        # apply a small nudge so correlated outputs separate
        apply_decorrelation_update(
            prototypes=self.prototypes,
            recent_outputs=A,
            lr=lr_decor,
        )
```

## 8. Optional margin objective

Instead of only "winner should be high," also require:
**winner score > runner-up score by margin**

```python
    # -----------------------------------
    # R. margin between best winner and best loser
    # -----------------------------------
    best_active = max(scores[active_idx])
    best_loser  = max(scores[loser_idx]) if len(loser_idx) > 0 else -inf

    margin_error = relu(required_margin - (best_active - best_loser))
    self.last_margin_error = margin_error

    if margin_error > 0:
        apply_margin_update(
            prototypes=self.prototypes,
            desc=desc_t,
            active_idx=active_idx,
            loser_idx=loser_idx,
            lr=lr_margin,
        )
```

## 9. Dead unit reseeding / maintenance

```python
    # -----------------------------------
    # S. dead-unit maintenance
    # -----------------------------------
    for j in range(n_out):
        if self.usage[j] < dead_usage_floor and unit_is_dead_for_long(j):
            self.prototypes[j] = reseed_from(desc_t, noise_scale)
            self.theta[j] = 0.0
            self.fatigue[j] = 0.0
```

## 10. Return output

```python
    self.prev_desc = desc_t
    self.prev_output = a_t
    return y_t
```

## Full flow in compact form

```python
def step(x_t):
    x_hist.push(x_t)

    desc_t = build_descriptor(x_hist)                 # hand-crafted or learned

    if pending_pred exists:
        pred_loss = mse(pending_pred, desc_t)         # temporal grounding

    sims   = cosine(prototypes, desc_t)
    scores = sims - theta - fatigue

    a_t, active_idx = sparse_activation(scores, k_active)
    loser_idx = select_confusing_losers(scores, active_idx)

    # output now
    y_t = a_t

    # predict future
    pending_pred = predictor(concat(desc_t, a_t))

    # winners move toward descriptor
    for j in active_idx:
        prototypes[j] += lr_pos * a_t[j] * (desc_t - prototypes[j])

    # confusing losers move away
    for j in loser_idx:
        prototypes[j] -= lr_neg * (desc_t - prototypes[j])

    # optional prototype-prototype repulsion
    repel_similar_prototypes()

    prototypes = normalize(prototypes)

    # fast anti-monopoly
    fatigue = fatigue_decay * fatigue + fatigue_gain * a_t

    # slow anti-monopoly
    usage = usage_decay * usage + (1 - usage_decay) * a_t
    theta += lr_homeo * (usage - target_usage)

    # redundancy reduction
    a_hist.push(a_t)
    if a_hist full:
        decorrelate_outputs_over_window(a_hist, prototypes)

    # dead unit rescue
    reseed_dead_units_if_needed()

    return y_t
```

## Mental picture of what each block does

| Block | Purpose |
|-------|---------|
| Descriptor / encoder | Turns recent stream into a compact state vector |
| Prototype matching | Asks: "which outputs think this state looks like them?" |
| Homeostasis + fatigue | Prevents one output from taking over the ecosystem |
| Sparse activation | Allows multiple outputs, but not flat equal soup |
| Winner pull | Lets active categories absorb the current pattern |
| Loser repulsion | Separates nearly-confused categories |
| Decorrelation | Stops two outputs from learning the same thing and just alternating |
| Future prediction | Adds "where is this going next?" so the column learns dynamics, not only snapshots |

## Recommended implementation order (revised after review)

Original plan was: repulsion → decorrelation → prediction → encoder →
sparsemax. Revised after review to avoid stacking overlapping mechanisms
blindly:

1. **Current HF column** (done — ts-00027 runs 001-009)
2. **Targeted loser repulsion** (~15 lines in `_update_prototypes`)
   - Only near-winners get pushed away, not all losers
   - Test in forage: does it improve collections or feature tracking?
3. **Rolling output-correlation measurement ONLY** (~20 lines, ring
   buffer + covariance, diagnostic only, no update)
   - Measure whether loser repulsion already reduces output redundancy
   - If correlation drops after step 2 → decorrelation update may be
     unnecessary
4. **Prediction-error-modulated Hebbian learning**
   - Tiny linear predictor: `concat(desc_t, a_t)` → predicted next
     descriptor
   - Surprise = MSE(prediction, actual next descriptor)
   - Modulate prototype LR by surprise:
     `effective_lr = base_lr * (1 + alpha * clamp(surprise))`
   - This is the missing gradient path from prediction to prototypes
   - High surprise → learn faster → prototypes reorganize until
     prediction improves
   - Low surprise → slow down → stable specialization
5. **Decorrelation update on prototypes** — only if step 3 shows
   correlation persists after steps 2+4. If outputs i and j are
   correlated: push `prototypes[i]` and `prototypes[j]` apart
   proportional to off-diagonal covariance entry.
6. **Sparsemax / entmax activation** — replace top-k masked softmax.
   Not a learning mechanism, but a readout refinement. Only worth
   testing once the score-producing system (prototypes, repulsion,
   prediction) is already decent. Sparsemax gives naturally sparse
   probs with variable active count (no hard k cap). Entmax gives a
   continuum between softmax and sparsemax.

   Test criteria (compare vs top-k masked softmax):
   - Output entropy
   - Active-output count distribution (does it vary per tick?)
   - Winner diversity
   - Inter-output correlation
   - Forage collections / feature correlations

   Rationale for late placement: if prototypes are redundant,
   sparsemax just gives sparser redundancy. If scores are poorly
   shaped, sparsemax won't invent meaningful categories. The current
   top-k mask has an artificial property (exactly k outputs survive
   even when data supports fewer or more), but fixing that only
   matters once scores are meaningful.

7. Only later try **learned encoder**

### Key design risks to watch

- **Pairwise prototype repulsion (section 5.3) is O(n_out^2).** Fine
  for 4-16 outputs. For larger n_out, vectorize as a full cosine sim
  matrix + threshold + batch nudge. Keep optional and mild.

- **Decorrelation update is underspecified.** Measuring output
  correlation is easy; turning it into a prototype gradient is the hard
  part. Bridge: if outputs i,j are correlated AND both activate on
  similar descriptors → push prototypes[i] and prototypes[j] apart.

- **Margin objective may fight homeostasis.** Homeostasis rotates
  winners; margin pushes current winner away from runner-up. Can
  create oscillation. Add last, test carefully. Note: applying margin
  on raw similarity (before theta) doesn't help because theta
  dominates scores at high h (±2.5 vs sim range [-1,1]).

- **Prediction loss has no gradient path to prototypes** in the naive
  design. The prediction-error-modulated LR (step 4) is the cheapest
  fix. For stronger coupling, would need prediction conditioned on
  prototype activations so useful categories get rewarded.

- **corr_window must match behavioral timescale.** In forage, direction
  changes every ~5-50 ticks. corr_window=50 sees one behavioral cycle.
  Too short → decorrelation punishes noise. Too long → averages over
  distinct regimes.

- **Loser repulsion and decorrelation may partially overlap.** This is
  why we measure correlation (step 3) before adding decorrelation
  updates (step 5) — to see if repulsion already solves the problem.

### Additional review notes

**Surprise modulation safeguard:** don't use raw surprise directly —
one noisy transition can spike learning and reshuffle prototypes. Smooth
and normalize:

```python
surprise_ema = beta * surprise_ema + (1 - beta) * surprise
norm_surprise = clamp(surprise / (eps + surprise_ema))
effective_lr = base_lr * (1 + alpha * norm_surprise)
```

Streaming systems need smoothed surprise, not raw MSE spikes.

**Loser repulsion design:**

Core principle: **only repel when the loss looks representational, not
regulatory.** Under strong homeostatic rotation, "loser" can just mean
"temporarily suppressed by theta," not "wrong prototype." Repulsion
should not punish a unit for losing due to control dynamics.

Selection rule: repel the **runner-up only** (the single highest-scoring
non-winner), and only if it was within a similarity band of the winner:

```python
runner_up_is_confusing = (score_runner_up > score_winner - margin_band)
```

For larger k_active, scale the number of repelled losers:

```python
n_losers_to_repel = max(1, k_active)  # match winners: k winners, k losers
# select top-k losers by score (nearest non-winners)
# only those within margin_band of the weakest winner
```

This keeps the ratio balanced: with k=1, repel 1 loser. With k=4,
repel up to 4 losers. Always filtered by the similarity band — distant
prototypes are never repelled.

Hunger-gated repulsion: scale lr_neg by satiation so repulsion is
strongest when homeostasis is relaxed (stable phase) and weakest when
homeostasis is driving fast rotation (hungry/exploration phase):

```python
lr_neg_eff = base_lr_neg * (1 - hunger_level)
```

This gives clean division of labor:
- **Hungry phase:** homeostasis drives exploration, repulsion backs off
- **Satiated phase:** homeostasis relaxes, repulsion sharpens category
  boundaries

Alternative finer-grained gates (can combine):

```python
winner_is_stable = (same_winner_for_last_N_ticks >= 2)
homeostasis_not_dominating = (abs(theta[winner] - theta[runner_up]) < theta_band)
repulsion_gate = runner_up_is_confusing and winner_is_stable
```

Diagnostic logging by regime (to verify repulsion is helping):
- low-hunger vs high-hunger repulsion events
- stable-winner vs rapidly-rotating winner
- small theta-gap vs large theta-gap

**Correlation diagnostics should log both:**
- Raw output correlation (do outputs co-activate?)
- Correlation after centering by each unit's running mean usage
  (separates "both popular" from "truly co-activate in same situations")

**Margin: skip for now.** At high homeostasis rates (h=0.5), theta
dominates score dynamics (±2.5 vs cosine [-1,1]). Margin on either
raw sim or rotated scores is either irrelevant or fights homeostasis.
Revisit only after the prediction-modulated system is stable.

### Implementation plan (concrete first step)

When ready to implement:
1. Add **loser repulsion** to `_update_prototypes` in
   `conscience_homeostatic_fatigue_column.py`
2. Add **correlation diagnostics** (ring buffer of recent outputs +
   covariance logging) — measure-only, no update
3. Do NOT add decorrelation updates, prediction, or margin yet
4. Test on forage benchmark at best config (4out, k=1, h=0.5, fs=0.001)
5. Compare collections + feature correlations + output correlation vs
   run 008/009 baselines

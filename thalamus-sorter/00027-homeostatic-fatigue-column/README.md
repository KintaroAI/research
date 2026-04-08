# ts-00027: Homeostatic-Fatigue Column

**Date:** 2026-04-05
**Status:** In progress
**Source:** `exp/ts-00027`
**Depends on:** ts-00026 (transformer columns, forage benchmark, all-ring sizes fix)

## Goal

Design a conscience-style column that preserves exploration churn while
preventing winner collapse. Test whether separating anti-collapse pressure
into two timescales (fast fatigue + slow homeostasis) produces better
forage performance than plain conscience.

## Motivation

ts-00026 established two things:

1. **Stable categories hurt forage** — the transformer encoder collapsed
   churn almost completely (stability ≥ 0.998 vs conscience's 0.12), which
   killed food collection. Exploration requires ongoing reassignment.
2. **ConscienceColumn uses a single threshold mechanism** — a per-output
   theta that drifts toward target usage. This couples "prevent monopoly"
   and "prevent dominance" into one slow signal, and reads the mean input
   only (no temporal structure).

The hypothesis: a column that (a) sees temporal structure and (b) responds
to both short-term activity bursts and long-term usage imbalance can
maintain exploration without collapsing.

## Biological motivation

Three distinct mechanisms, three different timescales, three different roles:

| Mechanism | Biology | Role | Timescale |
|-----------|---------|------|-----------|
| **Competition** | divisive normalization, inhibitory pool (V1) | who wins *now* | per tick |
| **Fast adaptation** | spike-frequency adaptation, short-term synaptic depression | recent winners get temporarily penalized | tens of ticks |
| **Slow homeostasis** | synaptic scaling, homeostatic plasticity | keep firing rates near a target set-point | thousands of ticks |

Plain hard-WTA has only competition → one strong early detector monopolizes
learning. Adding fatigue alone gives a **rotating dictatorship**: winners
cycle but specialization is fragile. Adding homeostasis alone gives
**equalized usage but blurry detectors**: units participate fairly but
don't differentiate sharply. Combining all three gives:

- local competition (who wins now)
- temporary suppression of recent winners (fatigue prevents short-term
  monopoly)
- slow pressure toward a target activity rate (homeostasis prevents
  chronic hogging and chronic silence)
- stable specialization once a unit actually matches a recurring pattern

Additional design consequences from biology:

- **Don't freeze losers.** Completely blocking learning on losing units
  creates permanent dead units. Biological homeostasis helps silent
  neurons re-enter the pool. We implement this with usage-scaled LR:
  underused units get a higher learning rate (up to `max_lr_scale`).
- **Near-winners should learn weakly.** Top-k > 1 soft competition lets
  gradient flow to multiple active units per tick — closer to cortical
  population coding than one-hot WTA.
- **Specialization is a consequence, not a constraint.** Prototypes
  differentiate because competition + fatigue force them to, not because
  we hand-label categories.

## Design: conscience_homeostatic_fatigue column

File: `dev/conscience_homeostatic_fatigue_column.py`

```
Per column (m independent instances, pure numpy):

1. DESCRIPTOR — 3-part temporal state from signal window
   X: (m, max_inputs, window)
   current   = X[:, :, -1]                   # last frame
   mean      = X.mean(axis=2)                # window mean
   delta     = X[:, :, -1] - X[:, :, -2]     # last-frame derivative
   each part is mean-centered + L2-normalized, then weighted and concatenated
   descriptor: (m, 3 * max_inputs), final L2-normalized

2. SIMILARITY
   prototypes: (m, n_outputs, 3 * max_inputs), L2-normalized per output
   sim = prototypes · descriptor              # (m, n_outputs)

3. COMPETITION — fully owned by this class (no ColumnBase rotation)
   scores = sim - theta - fatigue_strength * fast_fatigue
   logits = softmax(scores / temperature)
   top-k mask with k_active outputs kept, rest zeroed and renormalized

4. PROTOTYPE UPDATE — Hebbian nudge toward descriptor
   learn = p_active (or one-hot winner if learn_from_probs=False)
   lr_effective = proto_lr * lr_scale(usage) * learn
   prototypes += lr_effective * (descriptor - prototypes)  → renormalize
   lr_scale: underused outputs get higher LR (min_lr_scale..max_lr_scale)

5. FAST FATIGUE — short-term activity EMA
   fast_fatigue = fatigue_decay * fast_fatigue + fatigue_rate * p_active

6. SLOW HOMEOSTASIS — long-term usage tracking + theta drift
   usage = usage_decay * usage + (1 - usage_decay) * p_active
   theta += homeostasis_rate * (usage - target_usage)
   theta = clip(theta, ±theta_clip)

7. DEAD-UNIT RESEEDING — reset prototypes that stay silent
   dead = (tick - last_won > reseed_after) & (usage < dead_usage_threshold)
   dead prototypes ← current descriptor; theta, fatigue ← 0
```

### Key design decisions

- **[current, mean, delta] descriptor** vs conscience's mean-only. Captures
  temporal derivative; same total input dimensionality but 3× feature space.
- **Two-timescale anti-collapse**: fast_fatigue reacts within tens of ticks
  (decay=0.90), homeostatic theta drifts over thousands (usage_decay=0.995).
  The two are additive in the competition: `scores = sim - theta - λ·fatigue`.
- **Usage-driven LR scaling** — underused prototypes learn faster. Fights
  the "stale prototype never wins" failure mode without hard reseeding.
- **Top-k soft competition** — k_active outputs remain active per tick.
  Between hard-WTA (k=1) and unrestricted softmax (k=n). Gradient flows to
  multiple winners without full distribution spreading.
- **Learn from distribution** (`learn_from_probs=True`) — all k_active
  winners update toward the descriptor. Faster prototype migration.
- **Fully detached from ColumnBase rotation** — overrides
  `apply_output_rotation`, `save_rotation_state`, `load_rotation_state`.
  Base class contributes only wiring infrastructure (slot_map,
  `_gather_input`, `apply_wta`).

### Parameters (factory defaults)

| Parameter | Default | Notes |
|-----------|---------|-------|
| `n_outputs` | 4 | raised to 16 for first forage run |
| `k_active` | 2 | top-k outputs kept after softmax |
| `temperature` | 0.45 | softmax sharpness |
| `lr` / `proto_lr` | 0.05 | prototype nudge rate |
| `usage_decay` | 0.995 | slow usage EMA |
| `homeostasis_rate` | 0.02 | theta drift rate |
| `target_usage` | 1/n_outputs | per-output usage target |
| `theta_clip` | 2.5 | theta magnitude cap |
| `fatigue_decay` | 0.90 | fast fatigue EMA |
| `fatigue_rate` | 0.75 | fatigue accumulation |
| `fatigue_strength` | 1.0 | fatigue weight in scores |
| `current_weight` | 1.0 | descriptor part weights |
| `mean_weight` | 0.7 | |
| `delta_weight` | 0.8 | |
| `min_lr_scale` | 0.75 | usage-driven LR range |
| `max_lr_scale` | 1.75 | |
| `reseed_after` | 1000 | dead-unit reseed trigger (ticks) |
| `dead_usage_threshold` | 0.02 | dead-unit usage gate |

### CLI integration

- `--column-type conscience_homeostatic_fatigue`
- `--column-outputs N` (n_outputs)
- `--column-k-active K` (k_active, new flag)
- `--column-temperature T`, `--column-max-inputs MI`, `--column-window W`,
  `--column-wta MODE`, `--column-feedback`, `--lateral-inputs`,
  `--lateral-input-k K`, `--column-reseed-after N`

Other homeostatic-fatigue-specific parameters (fatigue decay/rate/strength,
homeostasis rate, target usage, LR scaling) still rely on factory defaults;
CLI flags to be added if tuning is needed.

## Hypothesis

With 16 outputs + top-k=4 soft competition + two-timescale anti-collapse:

1. **H1** — Forage collections ≥ plain conscience baseline (2284 @ 4 outputs).
2. **H2** — Stability stays < 0.5 (exploration preserved, unlike
   transformer columns at 0.998).
3. **H3** — Feature correlations (pos_x/pos_y/dir/hunger) remain ≥ 0.8,
   showing 16 outputs carry meaningful state distinctions.
4. **H4** — Column winner distribution across 16 outputs is balanced
   (no collapse), measured by winner-entropy / max-entropy.

## Method

### Baseline command (to run)

14×14 forage, 1M ticks, conscience_homeostatic_fatigue column, 16 outputs,
k_active=4, max_inputs=40, cluster cap=40, lateral inputs enabled, all-ring
wiring (default), viz enabled.

```bash
python3 main.py word2vec --mode correlation -W 14 -H 14 \
    --signal-source forage --forage-pois-dense 100 --forage-phase-ticks 5000 \
    --use-deriv-corr --threshold 0.5 --max-hit-ratio 0.1 \
    --dims 8 --lr 0.001 --k-sample 200 --batch-size 256 --anchor-batches 4 \
    --signal-T 1000 \
    --cluster-m 400 --cluster-max-k 2 --cluster-max-size 40 \
    --column-type conscience_homeostatic_fatigue \
    --column-outputs 16 --column-k-active 4 \
    --column-max-inputs 40 --column-window 10 --column-temperature 0.45 \
    --column-feedback --lateral-inputs --lateral-input-k 4 \
    --cluster-report-every 5000 --log-every 99999 --save-every 99999 \
    --viz-address 192.168.0.206:9100 --field-address 192.168.0.206:9101 --viz-every 1 \
    --save-model -f 1000000 \
    -o $(python3 output_name.py 27 homeo_fatigue_16out_k4_1m)
```

**Note on neuron count:** With `n_outputs=16` and `column-feedback`, total
feedback neurons = 400 × 16 = 6400, giving n_total = 196 + 6400 = 6596
(vs 1796 in ts-00026 baseline).

### Comparison targets

| Run | Column | Outputs | Collections |
|-----|--------|---------|-------------|
| ts-00026 / 102 | conscience | 4 | 2284 (baseline, no lateral) |
| ts-00026 / 103 | conscience | 4 | 2284 (with lateral) |
| ts-00026 / 115 | conscience (all-ring) | 4 | 2172 |
| ts-00026 / 101 | conscience_override (γ=2) | 4 | 2494 (best) |
| ts-00027 / 001 | homeo_fatigue | 16 k=4 | TBD |

## Log

### 2026-04-05 — Column design + CLI wiring complete

- File created: `dev/conscience_homeostatic_fatigue_column.py`
- Factory entry added in `cluster_manager.py`
- CLI `--column-type conscience_homeostatic_fatigue` + `--column-k-active`
- All ColumnBase rotation logic detached (overrides for
  `apply_output_rotation`, `save_rotation_state`, `load_rotation_state`)
- Smoke-tested: forward pass produces valid outputs, theta/fatigue track
  correctly, state persistence roundtrips (int64 last_won preserved)

### 2026-04-05 — patch_column_v1 tuning sweep (garden.png, 10 outputs, 10k frames)

Ran HF column in standalone patch task to probe calibration before full
forage run.

| Config | Entropy↑ | Inter-r↓ | Dominance↑ | Balance |
|--------|----------|----------|------------|---------|
| conscience (baseline)            | 0.677 | -0.083 | 0.192 | 9.1-10.7% |
| HF defaults (k=2, fs=1.0)        | 0.275 | -0.111 | 0.115 | 9.8-10.1% |
| HF k=5, fs=1.0                   | 0.627 | -0.108 | 0.135 | 9.2-10.5% |
| HF k=5, fs=0.25                  | 0.651 | -0.106 | 0.164 | 9.2-11.1% |
| HF k=5, fs=0.1                   | 0.649 | -0.105 | 0.170 | 9.3-11.0% |
| HF k=5, fs=0.0                   | 0.642 | -0.102 | 0.175 | 9.2-11.6% |
| HF k=10, fs=0.0                  | 0.792 | -0.098 | 0.200 | 8.2-11.3% |
| **HF descriptor-only (k=10, fs=0, h=0)** | **0.791** | -0.100 | **0.212** | 8.7-11.4% |

Metrics: `Entropy` is Shannon entropy of output probabilities (normalized
to [0,1] by log(n_outputs)). `Dominance` is mean of per-patch
max-winner-fraction (how reliably each patch picks the same output).
`Inter-r` is average pairwise correlation between outputs within a patch
(more negative = more differentiated).

Findings:

1. **HF descriptor-only beats conscience on the patch task** (0.791 ent
   / 0.212 dom vs 0.677 / 0.192). The 3-part `[current, mean, delta]`
   descriptor yields sharper per-patch specialization than conscience's
   mean-only descriptor.

2. **Fatigue is 10× miscalibrated.** Steady-state fatigue magnitude is
   `fatigue_rate / (1 - fatigue_decay) = 0.75 / 0.10 = 7.5` — much larger
   than the cosine-similarity range [−1, 1]. At `fs=1.0` this produces
   the rotating-dictatorship effect predicted in the biological motivation:
   dominance drops from 0.200 → 0.135 as fs sweeps 0 → 1.0.

3. **Top-k masking hurts on static patches.** `k_active=2` with 10
   outputs collapses entropy to 0.275 (only 2 outputs ever nonzero per
   tick).

4. **Homeostasis is nearly inert here.** Sweeping `homeostasis_rate`
   from 0.001 to 0.02 produced identical results — no natural collapse
   pressure to counteract in the static-patch task.

Implication for forage: anti-collapse mechanisms target collapse that
happens *because of feedback loops and temporal dependencies*, not
because of static visual patterns. The forage test is where fatigue +
homeostasis should actually earn their keep — that's what run 001 will
settle.

Calibration fix for future runs: drop `fatigue_rate` 10× (to ~0.075) or
`fatigue_strength` 10× (to ~0.1), OR use faster decay (0.5 instead of
0.9). Current defaults `(fatigue_rate=0.75, fatigue_decay=0.90,
fatigue_strength=1.0)` are too hot.

### 2026-04-06 — Forage runs: infrastructure fixes + 16-out failure + 4-out sweep

**Infrastructure added during run iteration:**
- `--forage-spasm-base` / `--forage-spasm-halflife` — tunable spasm strength
  and persistence (agent was stuck with old defaults)
- `--forage-motor-wiring random` — randomly distribute all column outputs
  across 4 motor directions (max reduction). Fixes 16-output columns where
  outputs 0-3 are rarely in top-k active set.
- `--forage-move-threshold` — configurable motor force gate (lowered to 0.001
  from 0.01)
- `--forage-clocks` — fill spare sensory neurons with clock oscillators
- `--column-fatigue-strength` / `--column-homeostasis-rate` — CLI tuning

**16-output runs (all frozen, failed to navigate):**

| Run | Config | Collections | Stability | hunger \|r\| |
|-----|--------|-------------|-----------|-------------|
| 001 | HF 16out k=4, fs=1.0, h=0.02 | 14 | 1.000 | 0.010 |
| 002 | HF 16out k=16, fs=0, h=0.001 | 1 | 0.999 | 0.802 |
| 003 | **conscience 16out** | **1363** | **0.077** | **0.979** |

16 outputs only work with conscience's hard-WTA + theta rotation.
HF's softmax produces smooth distributions that stabilize the feedback
loop too quickly — good representation (hunger 0.80 in run 002) but
zero navigation. Conscience's churn (stability 0.077) is what drives
motor exploration.

**4-output k=1 ablation (HF with hard WTA via top-k mask):**

All runs: 4 outputs, k_active=1, temp=0.45, spasm_base=0.25,
halflife=100k, random motor wiring, clocks, move_threshold=0.001.

| Run | fs | h | Collections | pos_x r | hunger r | prox r | Stability | Jumps |
|-----|----|---|-------------|---------|----------|--------|-----------|-------|
| 004 | 0.0 | 0.001 | 558 | 0.550 | 0.865 | 0.875 | 0.087 | 12.2M |
| 005 | 0.01 | 0.1 | 2379 | 0.605 | 0.779 | 0.874 | 0.987 | 23k |
| **006** | **0.01** | **0.5** | **2692** | **0.496** | **0.429** | **0.485** | **1.000** | **3.3k** |
| 007 | 0.01 | 1.0 | 763 | 0.441 | 0.264 | 0.328 | 1.000 | 2.6k |

Inverted-U: too little homeostasis → no motor diversity (558). Too much →
frantic jittering in place (763). Peak at h=0.5 (2692).

### Fatigue sweep at h=0.5 (4 outputs, k=1)

| fs | Collections | hunger r | pos_x r | Contiguity | Jumps |
|------|-------------|----------|---------|------------|-------|
| 0.001 | **2969** | 0.418 | 0.519 | **0.781** | 2.4k |
| 0.01 | 2692 | 0.429 | 0.496 | 0.712 | 3.3k |

Fatigue barely matters at optimal homeostasis — reducing from 0.01 to
0.001 gained only +10%. The system runs on homeostasis alone.

### Hunger-modulated homeostasis (run 009)

Added hunger modulation: `h = h_base/10 + hunger * h_base * 9/10`.
With `--column-homeostasis-rate 0.5`: satiated → h=0.05, hungry → h=0.5.
Implemented in forage.py tick_fn — one-line modulation, captures base
rate on first tick.

| Run | Config | Collections | pos_x r | hunger r | prox r | Stability | Jumps |
|-----|--------|-------------|---------|----------|--------|-----------|-------|
| 008 | h=0.5 fixed, fs=0.001 | **2969** | 0.519 | 0.418 | 0.524 | 1.000 | 2.4k |
| **009** | **h=0.05→0.5 hunger-mod**, fs=0.001 | **2642** | **0.632** | **0.801** | **0.890** | **0.991** | **15.7k** |

**Run 009: best balance of collections + representation.** Slightly fewer
collections than fixed h=0.5 (2642 vs 2969, −11%) but feature correlations
recovered dramatically: hunger 0.80 (was 0.42), proximity 0.89 (was 0.52),
pos_x 0.63 (was 0.52). When satiated, outputs stabilize and encode features
cleanly. When hungry, outputs cycle fast for exploration. More dynamic
(15.7k jumps vs 2.4k).

**Run 008 is still the collections champion (2969, +30% over conscience).**
Run 009 is the "best of both worlds" — strong collections AND meaningful
feature tracking.

### 2026-04-07 — Wiring structure analysis (run 005)

Motor columns (0-7) have **zero direct sensory input**. All motor
information flows through intermediate columns:

```
Sensory → [cluster] → Column → [feedback] → Motor column
                        1 hop
```

Min hops from sensory signal types to any motor column:
- **1 hop**: dir_xp, dir_xn, dir_yp, hunger, proximity, pos_y,
  target_x, contract_*, tire_*, clock, rest_dyp
- **2 hops**: pos_x, dir_yn, target_y, rest_dxn, rest_dxp, rest_dyn

Most navigation signals (direction, hunger, proximity) reach motors
in 1 hop. This 2-layer structure (sensory→intermediate→motor) emerged
from clustering alone — no explicit wiring design.

Motor columns receive feedback from 4-17 unique intermediate columns
each, which collectively fan out to 28-155 columns at hop 2.

## Results

### Metrics

| Metric | **HF best (008)** | **HF balanced (009)** | HF (005) | Conscience 4-out | Conscience 16-out |
|--------|--------------|---------------|----------|-----------------|------------------|
| Collections | **2969** | **2642** | 2379 | 2284 | 1363 |
| pos_x \|r\| | 0.519 | 0.632 | 0.605 | 0.75 | 0.708 |
| pos_y \|r\| | 0.515 | 0.890 | 0.877 | — | 0.894 |
| hunger \|r\| | 0.418 | **0.801** | 0.779 | 0.91 | 0.979 |
| proximity \|r\| | 0.524 | **0.890** | 0.874 | 0.77 | 0.899 |
| Stability | 1.000 | 0.991 | 0.987 | 0.12 | 0.077 |
| Total jumps | 2.4k | 15.7k | 23k | 46M | 27.7M |
| Contiguity | 0.781 | 0.770 | 0.601 | — | — |
| Splits | 4 | 123 | 255 | 113k | 2464 |
| Runtime | 10343s | 10488s | 10434s | 14237s | 15399s |

### Homeostasis sweep (4 outputs, k=1, fs=0.01)

| h | Collections | hunger r | Stability | Jumps |
|------|-------------|----------|-----------|-------|
| 0.001 | 558 | 0.865 | 0.087 | 12.2M |
| 0.1 | 2379 | 0.779 | 0.987 | 23k |
| 0.5 | 2692 | 0.429 | 1.000 | 3.3k |
| 1.0 | 763 | 0.264 | 1.000 | 2.6k |

Inverted-U: too little homeostasis → no motor diversity (558). Too much →
frantic jittering in place (763). Peak at h=0.5 (2692).

### Fatigue sweep at h=0.5 (4 outputs, k=1)

| fs | Collections | hunger r | pos_x r | Contiguity | Jumps |
|------|-------------|----------|---------|------------|-------|
| 0.001 | **2969** | 0.418 | 0.519 | **0.781** | 2.4k |
| 0.01 | 2692 | 0.429 | 0.496 | 0.712 | 3.3k |

**Run 008 (fs=0.001, h=0.5): new best at 2969 collections (+30% over
conscience baseline).** Fatigue barely matters — reducing from 0.01 to
0.001 gained +10%. The system runs on homeostasis alone. Contiguity
0.781 (highest seen) indicates strong topographic organization.

## Analysis

Two viable strategies for foraging:

1. **Conscience: explore-exploit via churn.** Hard WTA + fast theta rotation
   → constant winner cycling → cluster reassignment → motor output varies
   → agent wanders → finds food. High jumps, low stability.

2. **HF: stable navigation policy.** Strong homeostasis forces balanced
   outputs → consistent motor signals → directional movement → collects
   food. Very low jumps, high stability. The 3-part descriptor
   [current, mean, delta] captures temporal dynamics that the mean-only
   conscience descriptor misses.

The HF column achieves 30% more collections with 19000× less cluster
churn and 27% faster runtime. The stability vs churn tradeoff suggests
the forage benchmark rewards either extreme — the dangerous middle
ground is "stable enough to not explore but not stable enough to
navigate" (runs 001-002 with 16 outputs).

**Collections vs feature tracking trade-off:** stronger homeostasis
drives more food collection but weaker per-output feature correlations.
At h=0.5 (best collections), hunger r drops to 0.43 vs 0.87 at h=0.001.
The system navigates effectively without clean feature encoding —
the motor outputs are "messy but functional." This challenges the
assumption that good representation precedes good behavior.

**Fatigue calibration matters:** default `fatigue_strength=1.0` produces
steady-state penalty of 7.5 (vs cosine range [-1,1]), destroying
specialization. At `fs=0.01` (steady-state 0.075), fatigue acts as a
gentle tiebreaker without causing rotation.

**Homeostasis is the key mechanism for HF in forage.** The sweep shows
an inverted-U: h=0.001 (558) → h=0.5 (2969) → h=1.0 (763). Strong
theta pressure forces all 4 outputs to participate, which translates
to motor diversity across all 4 directions. But too much (h=1.0) causes
frantic jittering — direction changes so fast the agent can't sustain
movement. Peak at h=0.5.

**Fatigue is nearly irrelevant at the optimal homeostasis.** At h=0.5,
reducing fs from 0.01 to 0.001 gained only +10% (2692→2969). The
system runs on homeostasis alone — fatigue adds marginal tiebreaking
at best.

**Behavioral observation from viz:** with increasing h, direction changes
happen much faster — the agent switches heading more frequently rather
than committing to long straight paths. Movement also concentrates
increasingly toward the center of the field, as the rapid direction
switching prevents the agent from drifting to edges. This "jittery
center-seeking" behavior collects more food than conscience's wider
wandering because POIs respawn uniformly — staying central maximizes
proximity to the next spawn.

## Next Steps

- [x] Run baseline 1M forage with 16 outputs / k=4
- [x] Ablate fatigue_strength and homeostasis_rate at 4 outputs
- [x] Add CLI flags for homeostasis/fatigue params
- [x] Compare against conscience 16-out and 4-out baselines
- [ ] Run full 1M with best HF config (005) without viz for clean timing
- [ ] Test h=0.1, fs=0.01 at 16 outputs (strong homeostasis may now help)
- [ ] Compare against conscience_override (best ts-00026 result: 2494)
- [ ] Sweep temperature (0.2 vs 0.45 vs 1.0) at best config
- [ ] Test without clocks / without increased spasms to isolate their effect
- [ ] Consider whether the "stable policy" strategy generalizes beyond forage

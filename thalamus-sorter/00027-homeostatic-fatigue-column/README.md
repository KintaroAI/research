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

### 2026-04-07 — Rotating line benchmark + multi-output R² analysis

New benchmark: `benchmarks/rotating_line.py`. A line on a 7×7 grid
rotates continuously with periodic direction reversal and thickness
pulsation. Column outputs are correlated against ground-truth generative
factors (angle, sin/cos angle, direction, angular velocity, thickness,
thickness velocity). No supervision — factors must emerge from the
column's unsupervised learning.

**Key methodological finding:** single-output Pearson |r| has a
statistical ceiling that shrinks with more outputs. With k_active=1,
each output fires 1/n_out of the time (sparse binary), capping |r| at
~sqrt(p/(1-p)). This hid the real representational quality of
16-output columns. Multi-output R² (linear regression from all outputs
jointly to each factor) removes this ceiling.

**Rotating line results (20k training, reversal=200, speed_mod=0.3):**

Multi-output R² per factor:

| Factor | Consc 8 | TP 8 k=1 | TP 8 k=2 | TP 16 k=1 | **TP 16 k=2** |
|--------|---------|----------|----------|-----------|------------|
| angle | 0.220 | 0.160 | 0.101 | 0.258 | **0.455** |
| sin_angle | 0.071 | 0.166 | 0.105 | 0.285 | **0.441** |
| cos_angle | 0.048 | 0.289 | 0.257 | **0.506** | 0.376 |
| direction | 0.060 | 0.104 | 0.208 | 0.278 | **0.520** |
| angular_vel | 0.077 | 0.097 | 0.203 | 0.289 | **0.526** |
| thickness | **0.560** | 0.196 | 0.189 | 0.138 | 0.123 |
| thickness_vel | 0.142 | 0.091 | 0.124 | 0.174 | 0.287 |
| **R²>0.3 count** | **1/7** | **0/7** | **0/7** | **1/7** | **5/7** |

Findings:

1. **TP 16-out k=2 discovers 5 of 7 factors** (R²>0.3). Best overall
   representation by a wide margin. Direction (R²=0.52) and angular
   velocity (R²=0.53) are temporal dynamics that require the delta
   descriptor — conscience can't see them (R²=0.06-0.08).

2. **k=2 massively helps multi-output R².** Going from k=1 to k=2 at
   16 outputs: 1/7 → 5/7 coverage. Two active outputs produce
   continuous-valued signals (not sparse binary), giving linear
   regression much more to work with.

3. **More outputs + k=2 is the representation sweet spot.** 16 outputs
   with 2 active gives enough capacity for 7 factors while keeping
   outputs non-degenerate.

4. **Conscience wins only on thickness** (R²=0.56) — a static spatial
   feature. Its mean-only descriptor handles static structure but
   misses all temporal dynamics.

5. **Single-output |r| was hiding the story.** TP 16 k=2 has max
   single-output |r| of ~0.3 but multi-output R² of 0.52. The sparse
   binary ceiling was masking real quality.

Scaling with proportional k (n_out/k ≈ 8):

| Factor | 8/k=1 | 16/k=2 | 32/k=4 |
|--------|-------|--------|--------|
| angle | 0.160 | 0.455 | **0.672** |
| sin_angle | 0.166 | 0.441 | **0.805** |
| cos_angle | 0.289 | 0.376 | **0.705** |
| direction | 0.104 | 0.520 | **0.784** |
| angular_vel | 0.097 | 0.526 | **0.782** |
| thickness | 0.196 | 0.123 | **0.571** |
| thickness_vel | 0.091 | 0.287 | **0.536** |
| **R²>0.3** | **0/7** | **5/7** | **7/7** |

At 32/k=4, all 7 factors discovered with R²>0.5. The ratio n_out/k≈8
scales cleanly — each doubling of capacity with proportional k gives
a large jump.

**Output specialization (32-out k=4 analysis):**

Outputs are distributed, not one-per-factor:
- angular_velocity: 8 outputs specialize to it
- sin_angle: 7 outputs
- angle: 5, cos_angle: 5, direction: 4
- thickness: only 1 output
- thickness_velocity: 2 outputs

Outputs sharing the same best factor are **not redundant** — mean
pairwise r within groups is near 0 (-0.05 to +0.06). They respond at
different phases or polarities (e.g., sin_angle: o23 at +0.48, o31 at
-0.40). Global output uniqueness: mean pairwise r = -0.032, only 1%
of pairs have |r|>0.5.

This is distributed representation — closer to how word2vec encodes
meaning across dimensions than to clean one-output-per-factor
specialization. Multi-output R² captures this; single-output |r| misses
it.

**Output smoothness (32-out k=4):**

Outputs change **gradually**, not hash-like:
- Cosine autocorrelation: 0.979 at lag=1, 0.88 at lag=5, 0.72 at
  lag=10, near-zero at lag=50
- Winner persists for mean 11 ticks (~20° rotation per winner) —
  each prototype owns an angular sector
- Tick-to-tick L2 change: mean 0.063 (small gradual shifts)
- Angle-sorted outputs are 9% smoother than random sort

The column rotates through prototypes as the line rotates, with smooth
probability transitions between winners. Not a hash function — a
continuous state tracker with discrete but overlapping sectors.

**Window size sweep (32-out k=4):**

| Factor | w=1 | w=2 | w=3 | w=5 | w=10 | w=20 | w=50 |
|--------|-----|-----|-----|-----|------|------|------|
| angle | 0.621 | 0.636 | **0.694** | 0.638 | 0.672 | 0.689 | 0.568 |
| sin_angle | 0.795 | 0.636 | 0.798 | 0.688 | **0.805** | 0.795 | 0.719 |
| cos_angle | **0.840** | 0.693 | 0.605 | 0.776 | 0.705 | 0.741 | 0.767 |
| direction | 0.720 | 0.676 | 0.668 | 0.762 | **0.784** | 0.761 | 0.761 |
| angular_vel | 0.704 | 0.663 | 0.674 | 0.738 | **0.782** | 0.785 | 0.745 |
| thickness | **0.625** | 0.526 | 0.578 | **0.643** | 0.571 | 0.513 | 0.576 |
| thickness_vel | 0.414 | 0.475 | 0.545 | 0.477 | **0.536** | 0.495 | 0.392 |
| **Mean R²** | 0.674 | 0.615 | 0.652 | 0.675 | **0.694** | 0.684 | 0.661 |

All window sizes achieve 7/7 factor coverage. w=10 has the best mean
R² (0.694). The tradeoff:

- **Short windows (w=1-3)**: see snapshots better. cos_angle peaks at
  0.840 (w=1), thickness at 0.625 (w=1). No temporal blur → static
  factors are sharpest. But at w=1, descriptor has no mean or delta
  (only current frame).
- **Long windows (w=20-50)**: mean component blurs across multiple
  prototypes, hurting dynamic factors (thickness_velocity drops from
  0.536 to 0.392 at w=50).
- **w=10 sweet spot**: 10 ticks ≈ 18° rotation ≈ one winner sector.
  Descriptor temporal scope matches prototype angular coverage.

Surprising: direction (0.720) and angular_velocity (0.704) remain
strong even at w=1 (no temporal info). Prototypes distinguish angle
positions that correlate with direction regime — reversal boundaries
are at predictable angles.

Implication for forage: the 16-output TP column that failed in forage
(runs 001-002) may have been a motor-wiring problem, not a
representation problem. The rotating line shows 16-out k=2 has
excellent factor discovery — the missing piece was translating that
representation into motor commands. This motivates revisiting 16
outputs in forage with better motor design.

### 2026-04-07 — Multi-scale descriptor experiment

Tested replacing `[current, mean, delta]` with alternatives:

| Descriptor | angle | sin | cos | dir | ang_vel | thick | thick_vel | Mean R² |
|------------|-------|-----|-----|-----|---------|-------|-----------|---------|
| [cur, mean, delta_1] (default) | 0.672 | **0.805** | 0.705 | **0.784** | **0.782** | **0.571** | **0.536** | **0.694** |
| [cur, mean, delta_1, delta_5] (4-part) | 0.712 | 0.841 | 0.710 | 0.765 | 0.749 | 0.479 | 0.322 | 0.654 |
| [cur, delta_1, delta_5] (no mean) | 0.555 | 0.534 | 0.706 | 0.737 | 0.717 | 0.552 | 0.396 | 0.600 |

The mean component carries real signal — dropping it loses sin_angle
(0.805→0.534). The 4-part descriptor is 33% larger but doesn't improve
mean R². `delta_half` is redundant with `delta_1`. Default 3-part wins.

### 2026-04-07 — Patch column v1 benchmark: homeostasis vs specialization

Tested TP column on static visual patches (garden.png saccades). This
task has NO temporal dynamics — just spatial pattern matching.

**8 outputs k=1 — homeostasis sweep:**

| Config | Dominant winner↑ | Balance range | Inter-r↓ |
|--------|-----------------|---------------|----------|
| Conscience 8 | **0.221** | 11.5-14.3% | -0.112 |
| TP 8 k=1 **h=0** | **0.196** | 12.2-12.8% | **-0.140** |
| TP 8 k=1 h=0.001 | 0.131 | 12.4-12.6% | -0.143 |
| TP 8 k=1 h=0.01 | 0.132 | 12.4-12.6% | -0.143 |
| TP 8 k=1 h=0.5 | 0.128 | 12.5-12.5% | -0.143 |

Any homeostasis (even h=0.001) crushes patch specialization from 0.196
to ~0.13. Bare TP (h=0) gets closest to conscience (0.196 vs 0.221),
with even better output differentiation (inter-r -0.140 vs -0.112).

**32 outputs k=4 — homeostasis vs bare:**

| Config | Dominant winner↑ | Inter-r |
|--------|-----------------|---------|
| Conscience 32 | **0.086** | -0.021 |
| TP 32 k=4 **bare** (h=0, fs=0) | **0.098** | -0.018 |
| TP 32 k=4 h=0.001 | 0.053 | -0.032 |
| TP 32 k=4 h=0.5 | 0.050 | -0.032 |

Bare TP at 32 outputs beats conscience on dominant winner (0.098 vs
0.086). Prototypes naturally separate through Hebbian competition
alone on static patterns — homeostasis is counterproductive.

**Key insight: homeostasis role depends on task type.**

| Task | Best homeostasis | Why |
|------|-----------------|-----|
| **Forage** (feedback loop) | h=0.5 (2969 collections) | Forces motor diversity via output cycling |
| **Rotating line** (temporal) | h=0.5 works (factor R² unaffected) | Temporal dynamics drive differentiation regardless |
| **Static patches** (spatial) | **h=0** (0.196 dom winner) | No dynamics to drive differentiation; homeostasis just rotates randomly |

Homeostasis is a **motor/behavioral mechanism**, not a general
representation mechanism. For pure feature learning on static inputs,
Hebbian competition alone produces better specialization.

**Multi-scale descriptor on static patches (8out k=1):**

| h | Default dom | Multi-scale dom |
|------|------------|----------------|
| 0.0 | **0.196** | 0.184 |
| 0.001 | **0.205** | 0.200 |
| 0.01 | **0.175** | 0.167 |
| 0.5 | 0.147 | 0.146 |

Multi-scale slightly worse — on static patches with slow saccades,
`delta_1` and `delta_half` are both near-zero. Extra descriptor
dimension adds noise without new information.

### 2026-04-07 — Rotating line: homeostasis sweep (32out k=4)

| Factor | h=0 | h=0.001 | h=0.01 | **h=0.1** | h=0.5 | h=1.0 |
|--------|-----|---------|--------|-----------|-------|-------|
| angle | 0.383 | 0.361 | 0.363 | 0.468 | **0.672** | 0.643 |
| sin_angle | 0.218 | 0.183 | 0.185 | 0.397 | **0.805** | 0.756 |
| cos_angle | 0.112 | 0.115 | 0.152 | 0.330 | 0.705 | **0.832** |
| direction | 0.562 | 0.648 | 0.818 | **0.874** | 0.784 | 0.783 |
| angular_vel | 0.539 | 0.670 | 0.804 | **0.887** | 0.782 | 0.783 |
| thickness | **0.709** | **0.826** | **0.860** | 0.737 | 0.571 | 0.577 |
| thick_vel | 0.290 | 0.366 | 0.270 | 0.435 | 0.536 | **0.546** |
| **R²>0.3** | 4/7 | 5/7 | 4/7 | **7/7** | **7/7** | **7/7** |

Different factors peak at different h values:
- **Spatial/static** (thickness): peaks at low h (0.01→0.860). Stable
  prototypes hold position.
- **Temporal/dynamic** (direction, angular_vel): peaks at medium h
  (0.1→0.874/0.887). Homeostasis cycling encodes motion.
- **Circular phase** (sin/cos angle): needs high h (0.5-1.0). Many
  active prototypes needed to tile the full circle.

h=0.1 is the overall sweet spot for factor discovery (7/7, best on
dynamics). Different from forage optimal h=0.5 — the rotating line
rewards balanced representation, forage rewards motor diversity.

**Summary: optimal h depends on what matters in the task.**

| Task | Optimal h | Why |
|------|-----------|-----|
| Static patches | 0 | No dynamics; homeostasis rotates randomly |
| Rotating line (all factors) | 0.1 | Balances spatial + dynamic factors |
| Rotating line (phase only) | 0.5-1.0 | Full circle tiling needs many prototypes |
| Forage (collections) | 0.5 | Motor diversity drives food collection |

### 2026-04-08 — Temporal prototype column: loser repulsion + forage runs

Removed hunger-gated repulsion — learning should not be driven by
hunger. Repulsion now fires at constant `lr_neg` regardless of hunger.

**TP forage results (4out k=1):**

| Run | h | lr_neg | hunger-gated | Collections | hunger r | prox r | Stability |
|-----|------|--------|-------------|-------------|----------|--------|-----------|
| 005 (HF) | 0.1 | — | — | 2379 | 0.779 | 0.874 | 0.987 |
| 008 (HF best) | 0.5 | — | — | **2969** | 0.418 | 0.524 | 1.000 |
| 010 (TP) | 0.5 | 0.01 | yes | 2932 | 0.416 | 0.467 | 1.000 |
| 011 (TP) | 0.1 | 0.01 | no | 2148 | **0.819** | **0.890** | 0.993 |
| Conscience | — | — | — | 2284 | 0.91 | 0.77 | 0.12 |

Loser repulsion is **neutral at h=0.5** (2932 vs 2969 — homeostasis
dominates) and **slightly negative at h=0.1** (2148 vs 2379 — constant
repulsion destabilizes already-separated prototypes).

Run 011 has the best feature tracking of any TP run (hunger 0.82,
prox 0.89, pos_y 0.89) — nearly matching conscience on representation
while still +28% over HF baseline at same h.

The collections-vs-representation tradeoff remains consistent:
h=0.5 → 2969 collections / weak features.
h=0.1 → 2148 collections / strong features.

### 2026-04-08 — Step 3+4: correlation persistence + prediction-modulated learning

Implemented in `temporal_prototype_column.py`:
- **Correlation diagnostics** now persist to `column_corr_diagnostics.npz`
  (raw + centered covariance matrices). Summary printed on save.
- **Predictor head**: linear map `(d_model + n_out) → d_model`, predicts
  next tick's descriptor from current descriptor + output.
- **Surprise EMA**: `surprise_ema = β * surprise_ema + (1-β) * MSE`.
  Normalized surprise = `MSE / surprise_ema` (clipped to [0, 5]).
- **LR modulation**: `effective_lr = base_lr * (1 + α * norm_surprise)`.
  Higher surprise → faster prototype learning.
- Predictor trained via proper gradient descent (stored last input for
  backprop through linear layer).

CLI: `--column-surprise-alpha` (default 0.5, 0=disabled),
`--column-surprise-beta` (default 0.99).

**Rotating line benchmark (32out k=4, h=0.1, 20k frames):**

| Factor | TP without prediction | **TP with prediction** | Conscience |
|--------|---------------------|----------------------|------------|
| angle | 0.468 | 0.476 | 0.418 |
| sin_angle | 0.397 | 0.328 | 0.311 |
| cos_angle | 0.330 | 0.309 | 0.279 |
| direction | 0.874 | **0.914** | 0.144 |
| angular_vel | 0.887 | **0.895** | 0.184 |
| thickness | **0.737** | 0.670 | **0.901** |
| thickness_vel | **0.435** | 0.306 | 0.296 |
| **R²>0.3** | **7/7** | **7/7** | 3/7 |

Prediction improved temporal dynamics (direction 0.874→0.914) but
slightly hurt spatial factors (thickness 0.737→0.670). The surprise-
modulated LR may destabilize spatial prototypes when prediction error
spikes at direction reversals.

**Patch v1 (8out k=1, h=0, 10k frames):** prediction has no effect on
static patches (dominant winner 0.194 vs 0.196 without). Expected — no
temporal signal to predict.

**Forage run 012** (h=0.5, prediction enabled) running — will show
whether prediction helps the motor feedback loop.

## Next Steps

- [x] Run baseline 1M forage with 16 outputs / k=4
- [x] Ablate fatigue_strength and homeostasis_rate at 4 outputs
- [x] Add CLI flags for homeostasis/fatigue params
- [x] Compare against conscience 16-out and 4-out baselines
- [x] Rotating line benchmark — factor discovery analysis
- [x] Multi-output R² metric (removes sparse-binary ceiling)
- [x] Loser repulsion implementation + forage test
- [ ] Run full 1M with best HF config (008) without viz for clean timing
- [ ] Revisit 16 outputs in forage with k=2 + better motor design
- [ ] Compare against conscience_override (best ts-00026 result: 2494)
- [ ] Sweep temperature (0.2 vs 0.45 vs 1.0) at best config
- [ ] Test without clocks / without increased spasms to isolate their effect
- [ ] Consider whether the "stable policy" strategy generalizes beyond forage
- [x] Implement prediction-error-modulated learning (step 4 of TP plan)
- [x] Add correlation diagnostics persistence to save()
- [ ] Test adaptive h: low h early (learn features), ramp h later (drive motor)
- [x] Multi-output R² metric (removes sparse-binary ceiling)
- [ ] Implement loser repulsion (step 2 of temporal prototype plan)
- [ ] Add correlation diagnostics (step 3)
- [ ] Add prediction-error-modulated learning (step 4)

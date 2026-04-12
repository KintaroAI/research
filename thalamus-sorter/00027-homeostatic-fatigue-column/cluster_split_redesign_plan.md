# Cluster Split Redesign — Implementation Plan

## Problem

Current split policy (`split_largest_cluster_gpu` in `cluster_experiments.py`):

1. **Trigger**: any empty cluster → split biggest cluster
2. **Selection**: biggest cluster by raw member count
3. **Action**: k-means(2) on all members, half moves to empty slot
4. **Centroid seeding**: mean of each half's embeddings

### Why this is wrong

- **Raw size is a poor proxy for "needs splitting."** A cluster full of
  dead neurons (clock oscillators, spare sensory, noisy feedback) has
  high count but no useful signal. Splitting it wastes capacity.
- **Empty-trigger = death-spiral thrash.** "Cluster died → split biggest"
  means a cluster can be split immediately after one of its children
  dies, creating repeated-split patterns. Forage runs show 100k+ splits
  in 1M ticks — most are likely wasteful.
- **No pressure model.** Splits happen because of deaths, not because
  of actual capacity pressure from active clusters.
- **No sink for noise.** Clock neurons, spare slots, and noisy feedback
  need somewhere to live — they shouldn't compete with meaningful
  signal clusters for representation.

## Core principle

**Split clusters that are both busy AND stretched. Leave clusters that
are merely occupied alone.**

Two different notions of "big":
1. **Occupancy big**: many members (current rule)
2. **Useful big**: much live mass, much distortion (what we want)

Only the second should drive splitting.

## What we already have

- `split_every` (default 10): how often the split check runs
- `_split()` call in `ClusterManager.tick()`: triggered if any empty
- Per-neuron `pointers`, `cluster_ids`, `last_used` arrays
- `sizes` array (count of wired neurons per cluster)
- Skip-gram pair generation in `DriftSolver.tick_correlation()` — knows
  which neurons participated in positive pairs each tick

## Implementation plan (minimal → expand only if needed)

### Step 1: Split cooldown (~10 lines, 0 hyperparameters to tune)

The cheapest, highest-impact change. Adds a single per-cluster field.

```python
# New state in ClusterManager
self.last_split_tick = np.zeros(self.m, dtype=np.int64)

# In tick() split section, after computing n_to_split:
cooldown = 200  # ticks — simple constant, not a tunable
eligible = np.where(
    (self.sizes > 0) & (global_tick - self.last_split_tick > cooldown)
)[0]
# Pass eligible to _split() as candidate parent pool
```

**Effect**: prevents the same cluster (or its children) from being
split repeatedly in a 200-tick window. Should eliminate most of the
thrash pattern.

**Success criterion**: total splits drops significantly (e.g., 100k → 10k)
with collections staying within run-to-run noise.

### Step 2: Per-neuron activity EMA (~15 lines)

Track how often each neuron participates in positive skip-gram pairs.
This is the "live mass" signal.

```python
# New state in ClusterManager
self.activity_ema = np.zeros(self.n, dtype=np.float32)
self.activity_decay = 0.995  # ~200 tick window, matches cooldown

# In tick(), after DriftSolver returns pairs:
participated = np.zeros(self.n, dtype=np.float32)
if pairs is not None:
    anchors, partners = pairs
    participated[anchors.cpu().numpy()] = 1.0
    participated[partners.cpu().numpy()] = 1.0
self.activity_ema = (
    self.activity_decay * self.activity_ema +
    (1.0 - self.activity_decay) * participated
)
```

**Rationale**: in our skip-gram setup, pair participation is the direct
signal of "this neuron has useful correlation structure." A neuron that
never appears in pairs has no learning signal and no meaningful embedding.

**Source choice (from feedback's options)**:
- ✓ Positive-pair participation: directly tied to what makes embeddings meaningful
- ✗ Derivative norm: would favor any noisy signal
- ✗ KNN reuse: indirect, already downstream of pairs
- ✗ Generic "participation": ambiguous

### Step 3: Live-weighted size (~5 lines)

Replace raw count with activity-weighted count when picking parents.

```python
# In split logic, instead of np.argmax(sizes):
most_recent = cluster_ids[np.arange(n), pointers]
live_mass = np.bincount(
    most_recent[most_recent >= 0],
    weights=activity_ema[most_recent >= 0],
    minlength=m,
).astype(np.float32)
# Pick parent from eligible clusters by live_mass
live_mass[~eligible_mask] = 0
largest = np.argmax(live_mass)
```

**Effect**: clusters full of dead neurons become invisible to the
split selector. Clusters with many actively-participating neurons
become the preferred split targets.

### Step 4: Graveyard cluster (~20 lines)

Reserve cluster index 0 (or the first N) as a sink that:
- Accepts any neuron via cluster reassignment
- Never gets split
- Never counts in "biggest" selection
- Not eligible as a knn2 neighbor for other clusters

```python
# New param in ClusterManager.__init__
self.n_graveyards = 1  # hardcoded for now

# Mark graveyard indices
self.is_graveyard = np.zeros(self.m, dtype=bool)
self.is_graveyard[:self.n_graveyards] = True

# In split selection: exclude graveyards from eligible pool
eligible_mask &= ~self.is_graveyard

# In streaming update: low-activity neurons bias toward graveyard
# (one line: extend the candidate set to include graveyard[0] for
#  neurons with activity_ema < threshold)
```

**Effect**: clock neurons, spare sensory, and noisy feedback get pooled
in the graveyard instead of competing for real clusters.

### Step 5: Pressure-based trigger (~10 lines)

Split only when active clusters are genuinely overloaded, not just
because something died.

```python
# In tick() split section:
target_live = (self.activity_ema.sum() /
               max(1, (~self.is_graveyard).sum()))
max_live = live_mass.max()
overloaded = max_live > 1.5 * target_live  # 50% over target

n_empty = (self.sizes == 0).sum()
should_split = n_empty > 0 and overloaded
```

**Effect**: deaths alone don't trigger splits. Only when an active
cluster is pulling more than its fair share of live mass do we
redistribute capacity.

## What we are NOT doing (yet)

Explicitly deferring these to avoid over-engineering:

- **Composite score with 6 weighted terms** (α, β, γ, λ, etc.) — too
  many hyperparameters. Start with simple product, add terms only
  if simple version fails.
- **Child-survival penalty** — worth measuring need first. If step 1
  (cooldown) kills the thrash, this becomes moot.
- **PCA / eigenvalue multi-lobed detection** — interesting but
  expensive. Current k-means(2) on active members is good enough.
- **Dispersion-weighted splitting** — tempting, but we have no
  cheap dispersion metric today. Adding one is its own project.
- **Per-neuron age, churn_ema, centroid_velocity tracking** — extra
  state, unclear signal-to-noise. Revisit if steps 1-5 aren't enough.

## Success metrics

For each step, measure on forage 1M (h=0.1, decoupled, 4out k=1,
baseline run 019 with 3154 collections):

| Metric | Current | Target |
|--------|---------|--------|
| Total splits | ~3-5k | drop significantly (step 1) |
| Collections | 3154 | stay within ±5% |
| Contiguity | 0.69 | maintained or better |
| hunger r | 0.71 | maintained or better |
| Runtime | 11100s | not worse |

If collections drop significantly (>5%) at any step, that step is
doing real harm and needs revisiting before moving on.

## Implementation order

1. **Step 1 (cooldown)** — minimal, test, measure. Should reduce
   splits without affecting collections.
2. **Step 2 (activity EMA)** — add state, verify it decays sensibly
   (clock neurons should have low ema after warmup).
3. **Step 3 (live-weighted size)** — use activity EMA in selection.
   Measure which clusters get picked vs current rule.
4. **Step 4 (graveyard)** — add the sink. Verify dead neurons pool
   there.
5. **Step 5 (pressure trigger)** — gate split frequency.

After each step, commit + run forage. Don't advance if a step regresses.

## Key mental model

**Occupancy ≠ purpose.** The current rule says "biggest means most
important." This is wrong when a cluster can be "big" by accumulating
noise. We want "important means most useful learning signal" —
measured by actual skip-gram pair participation, which is the ground
truth for "this neuron carries information."

Everything else flows from this: cooldown prevents thrash, graveyard
absorbs noise, pressure gating prevents reflex-splits, activity weighting
redirects splits to where real signal lives.

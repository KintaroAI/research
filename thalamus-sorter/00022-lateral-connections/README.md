# ts-00022: Lateral Connections Between Columns

**Date:** 2026-03-20
**Status:** In progress
**Source:** `exp/ts-00022`
**Depends on:** ts-00021 (feedback loop, column wiring, motor control)

## Motivation

ts-00021 demonstrated two fundamental limitations of per-cluster columns:

### 1. Cross-cluster non-linear features are unreachable

The XOR benchmark (ts-00021, runs 012-015) showed that columns can only
detect features **local to one cluster**. A column over the XOR region
cannot compute XOR because it never sees neurons from the A or B regions —
those are in separate clusters (uncorrelated signals → different clusters).

Max |r| between any column output and the XOR feature was 0.17 across all
configurations (lr, anchor batches, training length). This is a fundamental
architectural limitation, not a hyperparameter issue.

### 2. Isolated feedback loops

With feedback neurons (ts-00021, runs 026-028), up to 106 clusters became
completely self-referential: pure-feedback clusters whose member neurons
come from columns whose own clusters are also pure feedback. These circuits
have no sensory grounding — they process their own output in a closed loop.

Lateral connections would break this isolation by routing information from
sensory-grounded columns into feedback-dominated columns.

### 3. Biological analogy

Cortical columns in the brain are densely connected laterally — horizontal
connections span millimeters, linking columns with similar or complementary
response properties. These connections serve multiple functions:

- **Context modulation:** A column's response depends on what surrounding
  columns are detecting (surround suppression, figure-ground)
- **Feature binding:** Combining features from different receptive fields
  into coherent object representations
- **Predictive coding:** Lateral predictions about what neighboring columns
  should be seeing, with error signals when expectations are violated

Our system currently has NO horizontal information flow. Each column is an
isolated processor. Adding lateral connections is the minimal architectural
change that enables cross-cluster computation.

## Architecture

Each column receives two types of input:
1. **Local input** (existing): signal window from cluster's wired neurons
2. **Lateral input** (new): outputs from ALL other columns (previous tick)

```
                    ┌── lateral outputs (prev tick) ──┐
                    ▼                                  │
Column A: [local_signal | lateral_all] → SoftWTA → outputs ──┤
Column B: [local_signal | lateral_all] → SoftWTA → outputs ──┤
Column C: [local_signal | lateral_all] → SoftWTA → outputs ──┘
```

### Separate weight matrices

The column learns local and lateral features independently:

```python
local_sim  = variance(prototypes_local @ signal_window)   # existing path
lateral_sim = prototypes_lateral @ all_column_outputs      # new path
total_sim  = local_sim + lateral_sim
probs = softmax(total_sim / temperature)
```

- `prototypes_local`: (n_outputs, max_inputs) — learns which signal patterns
  among own neurons to respond to
- `prototypes_lateral`: (n_outputs, m * n_outputs) — learns which combinations
  of other columns' states to respond to

### Why full connectivity

For M=42 (16×16 grid), each column receives 42 × 4 = 168 lateral values.
This is tiny. Full connectivity:
- Guarantees every column CAN reach every other (critical for XOR)
- No routing decisions — weights learn what to ignore
- Can sparsify later based on learned weight magnitudes
- At M=1066 (80×80), lateral input is 4264 — still manageable

### XOR detection path

With lateral connections, a column in the XOR region could learn:
```
lateral_weights[xor_output] ≈ +w_A_col * A_output - w_B_col * B_output
```
This fires when A and B disagree — exactly XOR. The local signal provides
the variance/match quality, while lateral input provides the cross-region
non-linear feature.

## Implementation

### Changes to ColumnManager

1. Add `lateral_prototypes: (m, n_outputs, m * n_outputs)` tensor
2. Store previous tick's column outputs as `prev_outputs: (m, n_outputs)`
3. In `tick()`:
   - Compute local similarity (existing)
   - Compute lateral similarity: `lateral_protos @ prev_outputs.flatten()`
   - Combine: `total_sim = local_sim + lateral_sim`
   - Same Hebbian update for lateral weights
4. CLI: `--column-lateral` flag (on/off)

### Verification

1. XOR benchmark: max |r| for XOR feature should rise from ~0.17 to >0.5
2. Garden image: check if isolated feedback loops decrease
3. Motor control: check if motor column responds to richer features

## Results

### Run 001: XOR + lateral, naive Hebbian, 10k ticks

Config: `--signal-source xor --column-lateral --lr 0.01 -f 10000`
XOR max|r|=0.188 — still at noise floor. Lateral connections didn't help.

**Root cause:** The Hebbian target for lateral weights is `prev_outputs` —
the same vector for ALL columns every tick. Every column pulls its winner's
lateral proto toward the same direction. Result: all lateral prototypes
converge toward similar weights, no differentiation.

Lateral weight analysis: all norms=1.0, std=0.07, max|w|=0.30. The
weights barely moved from initialization because the uniform target
provides no per-column learning signal.

**Fix:** Per-column modulated target. Scale the lateral input by each
column's local match strength. Columns with strong local variance response
reinforce the lateral pattern that co-occurred with their local activation.
Columns with weak local response don't update. This creates differentiation:
each column learns which lateral patterns predict its own local state.

```python
local_strength = sim[column, winner_output]
scaled_input = lateral_input * (local_strength / mean_strength)
```

### Run 002: XOR + lateral, modulated Hebbian, 10k ticks

XOR max|r|=0.175 — still noise floor. Modulated target didn't help.

**Deeper issue:** Hebbian learning pulls lateral weights toward co-occurring
lateral patterns. But XOR requires detecting **anti-correlation** — "A high
AND B low, OR A low AND B high." The Hebbian rule sees both cases and
averages them out, learning nothing.

The column needs to learn that specific lateral *combinations* predict its
local state, not just co-occurrence. This requires either:

1. **Contrastive learning:** Pull winner's lateral weights toward the
   lateral input, push loser's weights away. Different outputs would lock
   onto different lateral patterns (e.g., output 0 → "A high, B low",
   output 1 → "A low, B high").
2. **Error-driven learning:** Use the mismatch between predicted and actual
   local signal as the learning signal for lateral weights.
3. **Multi-output decomposition:** If 4 outputs × lateral diversity creates
   enough variety, different outputs might naturally specialize — but the
   Hebbian rule doesn't push toward this.

Next step: try contrastive lateral update — winner pulls, losers push.

### Run 003: XOR + contrastive lateral, hold=5, 10k ticks

XOR max|r|=0.160 — still noise floor. Contrastive push does differentiate
lateral weights (60% negative cosine pairs, mean=-0.03) but the signal
is too noisy. Even A and B features have weak correlation (0.17, 0.29).

The issue: hold=5 means the XOR signal flips every 5 ticks. With streaming
variance over the window, columns only see variance at transitions. The
column outputs never stabilize enough for lateral weights to learn from.

### Run 004: XOR + contrastive lateral, hold=50, 10k ticks — BREAKTHROUGH

Config: `--xor-hold 50 --column-lateral -f 10000`

| Feature | hold=5 | hold=50 |
|---------|--------|---------|
| A       | 0.17   | **0.40** |
| B       | 0.29   | **0.64** |
| XOR     | 0.16   | **0.64** |
| AND     | 0.27   | **0.52** |

**XOR max|r|=0.641** — column 9, output 0 detects XOR with r=0.64.
This is a genuine non-linear feature detection through lateral connections.

The short hold time was the bottleneck, not the architecture. With hold=50,
each A/B state persists long enough for columns to produce stable outputs,
which the lateral weights can then learn to combine for XOR detection.

Column 9 likely has lateral weights that respond to "A-column output high
AND B-column output low" (or vice versa) — exactly the XOR pattern.

# 00010 — Dynamic Input Channels

**Status:** Complete
**Source:** `exp/00010` (`833524a`)

## Goal

Test whether input channels can be added and removed from a live, trained cell
without disrupting existing categorization.

## Method

Added `extend_inputs(n_new)` and `remove_inputs(indices)` methods to SoftWTACell.

- **Extend:** pads prototypes with zeros in new dimensions — zero influence until
  Hebbian learning incorporates the new channel
- **Remove:** drops columns from prototypes, re-normalizes

Five scenarios tested (10 inputs, 4 clusters, 4 outputs):

1. **Extend + no retrain** — add 4 channels, verify zero disruption
2. **Extend + useful channel** — new channel correlates with cluster, cell learns it
3. **Extend + noise channel** — new channel is random, cell ignores it
4. **Remove channel** — drop one channel, measure impact and recovery
5. **Replace broken** — zero out a channel's prototype weight, retrain

## Results

| Scenario | Before | After (no retrain) | After retrain |
|---|---|---|---|
| Extend +4 channels | 1.000 | 1.000 | — |
| Extend +1 useful | 1.000 | — | 1.000 |
| Extend +1 noise | 1.000 | — | 1.000 |
| Remove 1 channel | 1.000 | 1.000 | 1.000 |
| Replace broken | 1.000 | 1.000 | 1.000 |

**Prototype weights on new channels:**
- Useful channel: weight = 0.491 (incorporated into prototypes)
- Noise channel: weight = 0.010 (effectively ignored)

## Analysis

**Zero disruption on extend.** Padding with zeros preserves the dot product exactly —
the new channel contributes nothing until learned. This is a natural property of
the prototype architecture: similarity = dot product, and zeros don't contribute.

**Selective incorporation.** The cell learns to use useful new channels (weight 0.49)
while ignoring noisy ones (weight 0.01). Hebbian learning pulls prototypes toward
the input, but noise averages out while consistent signal accumulates.

**Graceful removal.** Dropping a channel and re-normalizing causes minimal disruption
because 4 clusters in 10D have redundant signal — losing one dimension doesn't lose
much information.

**This is a strong architectural property.** Unlike neural networks where adding/removing
neurons requires restructuring weight matrices, prototype cells handle dynamic inputs
naturally. This makes them suitable for systems where sensors come and go.

## Commands

```bash
cd dev
python benchmark_dynamic_inputs.py -o $(python output_name.py 10 dynamic_inputs)
```

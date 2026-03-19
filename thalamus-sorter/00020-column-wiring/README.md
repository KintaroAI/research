# ts-00020: Column Wiring — Thalamus-to-Cortex Connection

**Date:** 2026-03-18
**Status:** In progress
**Source:** `exp/ts-00020`
**Commit:** `27c4b50`

## Goal

Wire the thalamus-sorter's cluster output to the column (cortical minicolumn)
model as input. Each cluster gets its own SoftWTACell column. Neurons wire/unwire
to their cluster's column as they enter/leave clusters via the ring buffer.

## Architecture

```
saccade crop → N neurons → M clusters
    each cluster → 1 SoftWTACell(n_inputs=max_inputs, n_outputs=configurable)
        input = raw signal of wired neurons (empty slots = 0)
        output = soft-WTA probabilities
```

The SoftWTACell code is copied from `column/dev/column.py` into the
thalamus-sorter codebase (no cross-repo import).

## Implementation Plan

### Data structures

```python
# Per-cluster column
columns: list[SoftWTACell]  # M columns, n_outputs each

# Wiring map: which neuron is in which input slot of which column
# slot_map[cluster_c, slot_s] = neuron_id (-1 = empty)
slot_map: np.ndarray  # (m, max_inputs), int64, init -1

# Pre-allocate max_inputs per column — fixed size, no resizing for now
```

### Wiring rules

**Wire (neuron enters cluster):** Triggered in `streaming_update_v3_gpu` when
`not in_ring` (new cluster entry, LRU eviction path):
1. Find lowest empty slot in `slot_map[best]` (first -1)
2. If no empty slot: skip (cluster full — no wiring)
3. Set `slot_map[best, slot] = anchor`
4. The evicted cluster (old value in LRU slot) loses this neuron

**Unwire (cluster evicted from ring):** Same event as wire — the LRU eviction
replaces one cluster with another. Old cluster loses neuron, new cluster gains it.

**In-ring switch (primary change):** No wiring change. Neuron stays connected
to both clusters in its ring. Only the primary pointer moves.

**Split:** When split moves neuron from `largest` to `dead`:
- Unwire from `largest` column
- Wire to `dead` column

### Column forward + learn (every tick)

Each tick, for each cluster:
1. Build input vector: `input[s] = signal[slot_map[c, s]]` for valid slots, 0 for empty
2. `probs = column.forward(input)`
3. `column.update(input)`

### CLI

```
--column-outputs N         # 0=disabled (default), 4=enable with 4 outputs per cluster
--column-max-inputs N      # pre-allocated input slots per column (default: 20)
--column-lr F              # column learning rate (default: 0.05)
--column-temperature F     # softmax temperature (default: 0.5)
```

## Implementation Summary

### New file: `thalamus-sorter/dev/column_manager.py`

- `SoftWTACell` — copied from `column/dev/column.py`, stripped to instantaneous
  mode only (no temporal/correlation/streaming modes needed here)
- `ColumnManager(m, n_outputs, max_inputs, temperature, lr)` — manages M columns:
  - `wire(cluster_id, neuron_id)` — find lowest empty slot, assign neuron
  - `unwire(cluster_id, neuron_id)` — find neuron in slot_map, clear it
  - `tick(signal)` — for each cluster with wired neurons, build input vector
    from signal values, run forward + Hebbian update
  - `get_outputs()` — return `(m, n_outputs)` array of column probabilities
  - `save(output_dir)` — save slot_map (.npy) + column state_dicts (.pt)

### Modified: `thalamus-sorter/dev/cluster_experiments.py`

- `streaming_update_v3_gpu` — returns 6th value `wiring_events`: list of
  `(neuron, old_cluster, new_cluster)` tuples captured at LRU eviction
- `split_largest_cluster_gpu` — returns `(splits_done, wiring_events)` tuple
  instead of just `splits_done`

### Modified: `thalamus-sorter/dev/main.py`

- `_ClusterManager.__init__` — accepts column params, creates `ColumnManager`
  when `column_outputs > 0`
- `_ClusterManager.tick()` — unpacks wiring events from stream update and split,
  processes wire/unwire calls, calls `column_mgr.tick(signal)` each tick
- `_ClusterManager.report()` — prints column stats (wired column count, winner
  distribution across output units)
- `_ClusterManager.save()` — saves column state alongside cluster state
- `set_signals` now also called when columns are enabled (columns need the
  signal tensor for input)
- CLI: 4 new arguments for column configuration

## Verification

1. Run 1k ticks with `--column-outputs 4` — check columns initialize, wire/unwire works
2. Check slot_map consistency: every neuron with primary cluster C has a slot in column C
3. Run 10k warm-start — observe column outputs stabilize as clusters settle
4. Visualize column winner distribution — are the 4 outputs used roughly equally?

## Results

### Smoke run: 1k ticks, 80x80, m=100, max_k=2, 4 outputs, window=4

```
python main.py word2vec --preset gray_80x80_saccades -f 1000 \
    --cluster-m 100 --cluster-max-k 2 --cluster-report-every 200 \
    --column-outputs 4 --column-window 4
```

**Commits:** `27c4b50` (initial impl), `54510f7` (streaming+window), `20322f7` (assert fix)

- ~95 ms/tick — acceptable for 6400 neurons + 100 columns
- 100/100 columns wired, 2000 total wirings (20 per column, capped at max_inputs)
- Consistency assert (no stale slot_map entries) passed every 200 ticks
- Winner distribution stays balanced, no collapse:
  ```
  tick  200: 25/22/26/27
  tick  400: 26/28/22/24
  tick  600: 24/23/29/24
  tick  800: 21/28/30/21
  tick 1000: 24/21/32/23
  ```
- Columns use streaming variance mode (temporal_mode='streaming', decay=0.5, window=4)
- max_inputs=20 caps wiring at ~30% of neurons per cluster (64 avg members)

### Warm-start: 10k ticks, 80x80, m=100, max_k=2, 4 outputs, window=4

```
python main.py word2vec --preset gray_80x80_saccades -f 10000 \
    --cluster-m 100 --cluster-max-k 2 --cluster-report-every 1000 \
    --column-outputs 4 --column-window 4 \
    --warm-start exp_00019/001_signal_mk2_20k/model.npy
```

- Warm-start from 20k-tick pre-trained model (contiguity=1.000 from tick 1)
- ~76 ms/tick (faster than cold-start — fewer reassignments with settled clusters)
- Consistency assert passed all 10 report intervals — no stale wirings
- Clusters stable: contiguity=1.000, diam=11.4, stability=0.86-0.92
- Winner distribution over time:
  ```
  tick  1000: 19/24/29/28
  tick  3000: 20/23/29/28
  tick  5000: 21/24/28/27
  tick  7000: 21/27/26/26
  tick 10000: 19/22/31/28
  ```
- Distribution balanced but not sharply discriminating — output 2 consistently
  wins slightly more. Columns may need more outputs, lower temperature, or
  more diverse input to develop sharp categories
- eval: K10 mean=1.75, <3px=99.3% — embedding quality excellent

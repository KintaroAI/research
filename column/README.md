# column

Self-organizing competitive categorization cell — a cortical minicolumn model.

## What

A small unsupervised module that maps a low-dimensional input to a probability
output with one dominant winner, learning online from the input stream using
local Hebbian update rules. No backpropagation.

## Setup

```bash
cd dev/
make setup              # creates venv, installs numpy + torch
source venv/bin/activate
make test               # 27 tests
```

## Usage

### Basic — create a cell, feed it data, get categories

```python
import torch
from column import SoftWTACell

# Create: 16 inputs, 8 output categories
cell = SoftWTACell(n_inputs=16, n_outputs=8)

# Feed one sample — returns probabilities over outputs
x = torch.randn(16)
probs = cell.forward(x)         # (8,) probability vector, sums to 1
winner = probs.argmax().item()  # dominant category

# Learn from the sample (Hebbian update)
winner, match_quality = cell.update(x)  # forward + update in one call
```

### Parameters

```python
cell = SoftWTACell(
    n_inputs=16,            # input dimensionality
    n_outputs=8,            # number of output categories
    temperature=0.5,        # softmax sharpness (lower = sharper winner)
    lr=0.05,                # Hebbian learning rate
    match_threshold=0.5,    # below this, recruit a dormant unit instead of updating winner
    usage_decay=0.99,       # EMA decay for usage counters (anti-collapse)
    temporal_mode=None,     # None, 'correlation', or 'streaming'
    streaming_decay=0.95,   # EMA decay for streaming mode projection stats
)
```

### Three temporal modes

**Instantaneous** (`temporal_mode=None`) — default, input is a single vector:
```python
cell = SoftWTACell(16, 8)
x = torch.randn(16)            # (n,)
winner, mq = cell.update(x)
```

**Correlation** (`temporal_mode='correlation'`) — input is an `(n, T)` trace,
similarity based on covariance structure. Best quality at small n, O(n²T):
```python
cell = SoftWTACell(16, 8, temporal_mode='correlation', match_threshold=0.1)
trace = torch.randn(16, 10)    # (n, T) — 10 timesteps of 16 channels
winner, mq = cell.update(trace)
```

**Streaming** (`temporal_mode='streaming'`) — accepts both `(n,)` and `(n, T)`.
Uses projection variance instead of covariance matrix. O(mn) per step, scales
to any n. Recommended for n > 30 or real-time use:
```python
cell = SoftWTACell(16, 8, temporal_mode='streaming', match_threshold=0.1,
                   streaming_decay=0.5)

# Single sample (needs temporal continuity — consecutive from same source)
x = torch.randn(16)            # (n,)
winner, mq = cell.update(x)

# Or a trace (project-first, no covariance matrix)
trace = torch.randn(16, 10)    # (n, T)
winner, mq = cell.update(trace)
```

**Which mode to use:**

| Condition | Mode |
|---|---|
| No temporal structure needed | Instantaneous |
| n ≤ T, best quality needed | Correlation |
| n > T, or real-time, or large n | Streaming |

**Recommended settings** (tested for 20 inputs, 4 outputs — scales to similar configs):

| Parameter | Instantaneous | Streaming (T=10) | Streaming (T=20) |
|---|---|---|---|
| temperature | 0.3 | 0.5 | 0.3 |
| lr | 0.05 | 0.05 | 0.05 |
| match_threshold | 0.5 | 0.1 | 0.1 |
| usage_decay | 0.99 | 0.99 | 0.99 |
| streaming_decay | — | 0.5 | 0.8 |

Rule of thumb: `streaming_decay ≈ 1 - 2/T` (effective window ≈ T/2).

**Sparse inputs are fine.** If you allocate 20 inputs but only 4 carry signal (rest
are zero), performance is identical to a 4-input cell. The dead channels contribute
nothing to the dot product. However, high-amplitude noise on unused channels will
hurt — keep them at zero or very low noise.

### Dynamic inputs — add/remove channels live

```python
cell = SoftWTACell(10, 4)
# ... train for a while ...

cell.extend_inputs(2)       # 10 → 12 inputs, zero disruption
cell.remove_inputs([0, 5])  # drop channels 0 and 5, re-normalize
```

### Save / load

```python
import torch

# Save
torch.save(cell.state_dict(), 'cell.pt')

# Load
state = torch.load('cell.pt')
cell = SoftWTACell.from_state_dict(state)
```

### CLI — train on synthetic data

```bash
# Instantaneous clusters
python main.py --n-inputs 16 --n-outputs 8 --frames 10000 \
  -o $(python output_name.py 1 baseline_16x8)

# Temporal correlation mode
python main.py --n-inputs 16 --n-outputs 4 --frames 5000 \
  --temporal-mode correlation --temporal-window 10 --match-threshold 0.1 \
  -o $(python output_name.py 2 temporal_test)
```

### Benchmarks

```bash
python benchmark.py -o output_dir          # Standard SQM (4 scenarios)
python benchmark_3d.py -o output_dir       # 3D movement directions
python benchmark_hierarchy.py              # Two-cell pipeline
python benchmark_multiscale.py             # Multi-scale temporal
python benchmark_residual.py               # Residual stacking
python benchmark_recurrent.py              # Recurrent ring
python benchmark_receptive.py              # Receptive field tiling
python benchmark_noise.py                  # Noise robustness
python benchmark_dynamic_inputs.py         # Dynamic input channels
python benchmark_composition.py            # Compositional logic
```

### Stacking cells

Cells can be composed in different architectures. The **wiring operation** between
cells determines what's solvable:

```python
# Transition matrix (exp 00005) — sequential patterns
winners_per_segment = [cell1.update(segment)[0] for segment in segments]
trans = compute_transition_matrix(winners_per_segment)
pattern = cell2.update(torch.from_numpy(trans.flatten()))

# Circular convolution (exp 00007) — modular arithmetic
pa = cell_a.forward(input_a)
pb = cell_b.forward(input_b)
conv = circular_conv(pa, pb)   # P(a+b mod n)
result = cell_c.update(conv)

# Multi-scale (exp 00014) — different time constants
cell_fast = SoftWTACell(n, m, temporal_mode='streaming', streaming_decay=0.3)
cell_slow = SoftWTACell(n, m, temporal_mode='streaming', streaming_decay=0.98)
# Both see same input; combined output separates multi-timescale patterns

# Receptive fields (exp 00016) — local features + combinations
local_cells = [SoftWTACell(group_size, k) for _ in range(n_groups)]
combo_cell = SoftWTACell(n_groups * k, m)
local_outs = [cell.forward(x[i*g:(i+1)*g]) for i, cell in enumerate(local_cells)]
result = combo_cell.update(torch.cat(local_outs))
```

## How it works

1. **Prototypes:** each output unit stores a prototype vector on the unit sphere
2. **Competition:** cosine similarity → softmax(temperature) → probability output
3. **Hebbian learning:** winner's prototype pulled toward input
4. **Anti-collapse:** usage counters (EMA) gate learning rate — frequent winners learn slower
5. **Recruitment:** poor matches redirect update to least-used unit at full LR
6. **Temporal:** streaming mode tracks projection variance via EMA; correlation mode
   computes full covariance. Both detect which input channels co-vary over time.

## Performance

| Mode | Time/step | Memory |
|---|---|---|
| Instantaneous | ~190 us (any size) | O(mn) — 272B to 256KB |
| Streaming | ~190 us (any size) | O(mn + m) |
| Correlation | 258 us (n=16) → 28ms (n=256) | O(mn + n²) |

Pure CPU, PyTorch. ~5,000 steps/s for instantaneous and streaming.

## Known limitations

- **Noise fragile:** high-amplitude noise on even 1-2 channels can collapse separation (exp 00009). Root cause: input normalization amplifies noisy channels.
- **Catastrophic forgetting:** switching tasks overwrites prototypes (exp 00008). Retraining is 10x faster though.
- **Relational composition:** can't learn "is a == b?" or sum mod 4 from prototype matching alone (exp 00006). Needs task-specific wiring (exp 00007).
- **match_threshold** must be calibrated per temporal mode: ~0.5 for instantaneous (cosine range), ~0.1 for correlation/streaming (variance-explained range).

## Benchmark results

### Standard (16 inputs, 4 outputs, 4 clusters)

| Scenario | NMI | Lock-in |
|---|---|---|
| Instantaneous clusters | 0.968 | 2000 frames |
| Temporal co-variation | 0.965 | 1000 frames |
| Distribution shift | 0.897 | 200 frames to recover |

### Stacking architectures (experiments 00013–00016)

| Architecture | Best NMI | vs Single cell | When useful |
|---|---|---|---|
| **Multi-scale temporal** | **0.276** | **+2.4x** (0.115) | Multi-timescale patterns |
| **Receptive fields** | **0.993** | **+2.0x** (0.500) | High-dim local structure |
| Residual stacking | 0.822 | +16% (0.711) | Capacity-constrained |
| Recurrent ring | 0.800 | -16% (0.950) | Negative result |

## Experiments

| # | Name | Key result |
|---|---|---|
| 00001 | Soft-WTA cell | Baseline implementation, NMI=0.875 on 8 clusters |
| 00002 | Temporal context | Correlation mode NMI=0.822 where instantaneous gets 0.003 |
| 00003 | SQM benchmark | Full evaluation suite, temporal converges 2x faster |
| 00004 | 3D movement | Direction categorization from position traces, NMI=0.70–0.78 |
| 00005 | Hierarchical stack | Two-cell pipeline for movement patterns, NMI=0.565 vs 0.328 alone |
| 00006 | Compositional logic | Negative result: relational tasks defeat prototype matching with outer product |
| 00007 | Composition wiring | Circular convolution + compare stats partially rescue composition (NMI 0.24–0.38) |
| 00008 | Catastrophic forgetting | Task B overwrites A (0.857→0.289), but retraining is 10x faster (500 vs 5000 frames) |
| 00009 | Noise robustness | 2 noisy channels (12%) drops NMI 1.0→0.14; low-amplitude noise (0.3) at 50% channels OK |
| 00010 | Dynamic inputs | Add/remove channels live — zero disruption, useful channels learned, noise ignored |
| 00011 | Performance | ~5k steps/s instant, ~4k corr (n≤16), 272B–256KB memory, pure CPU |
| 00012 | Streaming mode | O(mn) temporal via EMA projection variance — 9x faster than correlation at n=64 |
| 00013 | Residual stacking | 2×4 residual beats single m=4 by 16%, but single m=16 still wins if capacity available |
| 00014 | Multi-scale temporal | Two decay rates (0.3 + 0.98) give 2.4x NMI over best single scale |
| 00015 | Recurrent ring | Negative result: feedback loop amplifies errors, single cell wins |
| 00016 | Receptive fields | RF 8×4→4 gets 0.993 vs single 0.500 on 32D conjunctions; fragile configs |

## Repository structure

```
column/
├── README.md                  # this file
├── CLAUDE.md                  # conventions and project instructions
├── REQUIREMENTS.md            # 16 functional requirements
├── PLANNING.md                # design analysis, algorithm comparison
├── EXPERIMENT_PROTOCOL.md     # workflow for running experiments
├── dev/                       # active development
│   ├── column.py              # SoftWTACell implementation
│   ├── main.py                # CLI entry point
│   ├── metrics.py             # Separation Quality Metrics (SQM)
│   ├── benchmark*.py          # benchmark scripts
│   ├── output_name.py         # auto-generate output directory paths
│   ├── Makefile               # setup, test, clean
│   └── requirements.txt       # numpy, torch
├── tasks/                     # implementation plans
└── NNNNN-experiment-name/     # archived experiment snapshots
```

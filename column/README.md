# column

Self-organizing competitive categorization cell — a cortical minicolumn model.

## What

A small unsupervised module that takes a low-dimensional input vector (~10–20 values), maps it to a low-dimensional probability output (~4–20 units) with one dominant winner, and learns online from the input stream using local update rules. No backpropagation.

The core mechanism is soft winner-take-all (WTA) competition over learned prototypes:
- Each output unit holds a prototype vector
- Input similarity is computed via dot product or negative MSE, then passed through softmax with tunable temperature
- The winning unit's prototype moves toward the input (Hebbian pull)
- Usage counters gate plasticity — frequent winners learn slower, giving other units a chance
- A match-quality threshold controls stability vs plasticity — poor matches recruit dormant units rather than forcing assimilation into existing categories

## Why

Building block for larger architectures that need local, online, unsupervised categorization. The cell should be stable when the input distribution is stationary (categories lock in) and plastic when it shifts (new categories emerge). This is the classic stability-plasticity dilemma — the module draws on ideas from SOMs, ART, and competitive learning to address it.

Connects to prior work in thalamus-sorter (derivative correlation buffers, temporal similarity measures) and llm-fundamentals (grokking, category formation).

## Key properties

- **Probability outputs** with one clear winner, not hard argmax
- **Anti-collapse** — no single unit monopolizes; white noise yields uniform low output
- **Online learning** — update per example or small batch, no offline training loop
- **Biologically plausible** — local competition, local updates, no global error signal
- **Temporal context** (optional) — input can be an `(n, T)` matrix of recent traces, enabling correlation-based similarity

## Benchmarks

All implementations are evaluated with the **Separation Quality Metrics (SQM)** suite
(`dev/benchmark.py`, `dev/benchmark_3d.py`). Run `make test` (19 tests) to verify.

### Standard benchmark (synthetic clusters, 16 inputs, 4 outputs)

| Scenario | NMI | Consistency | Lock-in frame |
|---|---|---|---|
| Instantaneous clusters | 0.968 | 0.993 | 2000 |
| Temporal co-variation | 0.965 | 0.993 | 1000 |
| Temporal data + instant mode (control) | 0.001 | 0.270 | — |
| Distribution shift adaptation | 0.897 | 0.968 | 200 frames to recover |

### 3D movement direction benchmark (3 inputs, position traces)

Can the cell learn which direction an object is moving from x,y,z position traces?

| Directions | NMI | Consistency | Control (instantaneous) |
|---|---|---|---|
| 6 cardinal (±x,±y,±z) | 0.702 | 0.865 | 0.044 |
| 8 diagonal (±x±y±z) | 0.776 | 0.967 | — |
| 12 icosahedron | 0.770 | 0.813 | — |

Instantaneous mode gets chance-level (NMI=0.04) because random starting positions
hide direction — only temporal co-variation reveals it.

## Experiments

| # | Name | Key result |
|---|---|---|
| 00001 | Soft-WTA cell | Baseline implementation, NMI=0.875 on 8 clusters |
| 00002 | Temporal context | Correlation mode NMI=0.822 where instantaneous gets 0.003 |
| 00003 | SQM benchmark | Full evaluation suite, temporal converges 2x faster |
| 00004 | 3D movement | Direction categorization from position traces, NMI=0.70–0.78 |

## Repository structure

```
column/
├── README.md                  # this file
├── CLAUDE.md                  # conventions and project instructions
├── REQUIREMENTS.md            # detailed functional requirements (15 reqs)
├── PLANNING.md                # design analysis and prototype plan
├── EXPERIMENT_PROTOCOL.md     # workflow for running experiments
├── dev/                       # active development
│   ├── main.py                # CLI entry point — train on synthetic data
│   ├── column.py              # SoftWTACell implementation
│   ├── metrics.py             # Separation Quality Metrics (SQM)
│   ├── benchmark.py           # SQM benchmark — 4-scenario evaluation
│   ├── benchmark_3d.py        # 3D movement direction benchmark
│   ├── output_name.py         # auto-generate output directory paths
│   ├── Makefile               # setup, test, clean
│   └── requirements.txt       # numpy, torch
├── tasks/                     # implementation plans
└── NNNNN-experiment-name/     # archived experiment snapshots
```

Experiments follow the protocol in `EXPERIMENT_PROTOCOL.md` — plan, run, finalize, tag.

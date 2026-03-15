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
│   ├── output_name.py         # auto-generate output directory paths
│   ├── Makefile               # setup, test, clean
│   └── requirements.txt       # numpy, torch
├── tasks/                     # implementation plans
└── NNNNN-experiment-name/     # archived experiment snapshots
```

Experiments follow the protocol in `EXPERIMENT_PROTOCOL.md` — plan, run, finalize, tag.

# Benchmarks

Modular test suite for column/lateral feature detection and sensorimotor
capabilities. Each benchmark is a Python module with `make_signal()` and
`analyze()`. Auto-discovered via `--signal-source <name>`.

## Perception benchmarks

Test what the system can detect/represent.

| Benchmark | Tests | Key metric | Baseline | With lateral |
|-----------|-------|------------|----------|-------------|
| **xor** | 2-input non-linear | XOR max\|r\| | 0.22 | **0.77** |
| **majority** | 3-input non-linear | MAJ max\|r\| | 0.44 | 0.40 |
| **oddball** | novelty detection | odd_idx max\|r\| | 0.41 | 0.39 |
| **sequence** | temporal memory | SEQ max\|r\| | 0.32 | 0.42 |
| **match** | spatial comparison | EQ max\|r\| | 0.36 | 0.25 |

## Capability benchmarks

Test specific perception-to-action capabilities.

| Benchmark | Tests | Key result |
|-----------|-------|------------|
| **echo** | Temporal prediction (delayed signal) | voice r=0.29 |
| **mirror** | Action-consequence (output→input) | curr_out r=**0.98** |
| **lever** | Operant conditioning (stimulus→action→reward) | 0 presses (WIP) |
| **seek** | Directional action-reward | accuracy 28.6% (random=25%) |

## Sensorimotor benchmark

Full integrated test.

| Benchmark | Tests | Key result |
|-----------|-------|------------|
| **forage** | Navigate, collect POIs, manage hunger/fatigue | Best: 117 sparse collections |

Forage features: retina vision, 8-fiber muscles with independent
restlessness/tiredness/contraction, hunger-modulated lr, spasm-based
exploration, gated motor force, wrap-around or walled field.

## Regression benchmark

Quick smoke test for core perception capabilities.

```bash
python main.py word2vec --signal-source regression --knn-track 20 \
  -W 16 -H 16 --use-deriv-corr --threshold 0.5 --max-hit-ratio 0.1 \
  --dims 8 --lr 0.01 --k-sample 50 --signal-T 100 \
  --cluster-neurons-per 10 --column-outputs 4 --column-feedback \
  -f 5000
```

Checks:
- **TOPO**: spatial accuracy > 30% within 5px
- **COLUMN_DIFF**: > 30% of columns differentiated (max prob > 0.4)
- **SEPARATION**: < 10% mixed sensory/feedback clusters
- **STABILITY**: > 80% cluster assignment stability

## Key parameters

| Parameter | Effect |
|-----------|--------|
| `--predictive-shift 1` | Causal prediction (helps temporal tasks) |
| `--predictive-mix 0.1` | 90% spatial + 10% causal (best for mixed) |
| `--column-lateral` | Enable lateral connections |
| `--lateral-k 2` | Connections per column (2 = optimal) |
| `--column-temperature 0.2` | Peakier softmax (better differentiation) |

## Adding a new benchmark

Create `benchmarks/mytest.py` with:

```python
name = 'mytest'
description = 'One-line description'

def add_args(parser):     # CLI flags (optional)
def make_signal(w, h, args):  # returns (tick_fn, metadata)
def analyze(metadata, cluster_mgr, signals, tick_counter, T, output_dir):
```

Auto-discovered via `--signal-source mytest`.

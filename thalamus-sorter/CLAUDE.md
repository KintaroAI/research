# Thalamus Sorter

Topographic map formation via temporal correlation — neurons that fire together become spatial neighbors. No supervision, no ground truth layout.

## Repository Structure

- **`dev/`** — Active development (work here)
- **`NNNNN-experiment-name/`** — Archived experiment snapshots with README documenting results
- Experiments use `ts-NNNNN` naming. Latest: ts-00013 (RGB multi-channel)

## Setup & Run

```bash
cd dev/
make setup              # venv + deps
source venv/bin/activate

# Using presets (recommended):
python main.py word2vec --preset gray_80x80_saccades -f 50000 -o output_run
python main.py word2vec --preset rgb_80x80_garden --dims 16 -f 500000 -o output_run

# CLI args override preset values. See presets/ directory for all configs.
```

## Key Design Decisions

- **Dual-vector dot product skip-gram** (W and C vectors), not Euclidean distance. Captures richer structure than spatial proximity alone — channel identity, correlation strength, multi-scale neighborhoods, temporal dynamics.
- **Embeddings are the real output**, not the 2D rendered grid. PCA/UMAP rendering is lossy visualization. D=8-16 embeddings encode everything meaningful about a neuron's activity pattern.
- **MSE-based neighbor discovery** from rolling saccade buffer with `--max-hit-ratio 0.1` safety net against global signal flickering.
- **k_sample scales linearly with n** (~3% fraction). Dead anchor rate should be 10-15%.

## Important Files

- `main.py` — CLI entry point, all algorithms
- `solvers/drift_torch.py` — DriftSolver: skip-gram training (tick_correlation, tick_sentence)
- `render_embeddings.py` — project() and render() for visualization
- `analyze_channels.py` — post-hoc multi-channel structure analysis
- `DESIGN.md` — Full design principles, parameter tradeoffs, correlation requirements
- `METRICS.md` — All evaluation metrics with interpretation tables and benchmarks
- `presets/` — Reusable parameter configurations (JSON)

## Conventions

- No co-authored-by lines in commit messages
- Experiment READMEs are technical reports (hypothesis, method, results, analysis)
- Don't store large output directories in git — use /tmp for runs, save only model.npy and info.json if needed
- Always include `--max-hit-ratio 0.1` and `--eval` in production runs

### Output directory naming

Format: `output_{experiment}_{run}_{short_desc}`

- `{experiment}` — experiment number (e.g. `13`)
- `{run}` — autoincrementing run number, zero-padded to 3 digits (e.g. `001`)
- `{short_desc}` — brief description using underscores (e.g. `rgb_garden_50k_d16`)

Examples:
```
output_13_001_gray_80x80_50k
output_13_002_rgbg_saccades_50k
output_13_003_rgb_garden_500k_d16
output_14_001_baseline
```

Use `-o $(python -c "from output_name import next_output; print(next_output(13, 'rgb_garden_50k'))")` or just pass `-o` manually following the convention. Directories sort chronologically within each experiment.

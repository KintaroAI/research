# Column

Self-organizing competitive categorization cell — a cortical minicolumn model.

## Repository Structure

- **`dev/`** — Active development (work here)
- **`NNNNN-experiment-name/`** — Archived experiment snapshots with README documenting results
- Experiments follow `EXPERIMENT_PROTOCOL.md`

## Setup & Run

```bash
cd dev/
make setup              # venv + deps
source venv/bin/activate
make test               # verify imports and basic functionality
```

## Conventions

- No co-authored-by lines in commit messages
- Experiment READMEs are technical reports (hypothesis, method, results, analysis)
- Output directories go to `~/data/research/column/`, not in the repo
- Do not store large log files in the repo — sample key lines or use head/tail excerpts
- Before implementing a plan, save it as a numbered markdown file in `tasks/` following the format of existing task files

### Output directory naming

Format: `~/data/research/column/exp_NNNNN/{run}_{short_desc}`

Use `output_name.py` to auto-generate the next path:
```bash
python main.py -o $(python output_name.py 1 baseline_16x8)
```

## Important Files

- `REQUIREMENTS.md` — Detailed functional requirements (15 requirements across I/O, learning, competition, stability)
- `PLANNING.md` — Design analysis, literature mapping, proposed prototype architecture
- `dev/output_name.py` — Auto-generate experiment output directory paths

# Column — dev

Active development directory for the cortical minicolumn module.

## Setup

```bash
make setup          # creates venv, installs deps
source venv/bin/activate
```

## Usage

```bash
# Run tests
make test

# Train a column cell
python main.py --n-inputs 16 --n-outputs 8 --frames 10000 -o output_run

# With auto-generated output path:
python main.py --n-inputs 16 --n-outputs 8 --frames 10000 \
  -o $(python output_name.py 1 baseline_16x8)
```

## Code layout

```
dev/
├── main.py              # CLI entry point — train on synthetic data
├── column.py            # SoftWTACell implementation
├── metrics.py           # Separation Quality Metrics (SQM)
├── output_name.py       # Auto-generate output directory paths
├── Makefile
└── requirements.txt
```

## Output directory naming

Format: `~/data/research/column/exp_NNNNN/{run}_{short_desc}`

Use `output_name.py` to auto-generate the next path:
```bash
python main.py -o $(python output_name.py 1 baseline_16x8)
```

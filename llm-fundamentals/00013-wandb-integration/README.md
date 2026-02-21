# Experiment 13: W&B Integration

**Date:** 2026-02-21
**Status:** In Progress
**Source:** *tagged on completion as `exp/00013`*

## Goal

Add real-time experiment tracking via Weights & Biases without modifying the CUDA training code.

## Hypothesis

A Python wrapper that launches the existing `./train` binary via subprocess and parses its structured stdout can capture all training metrics (train loss, val loss, test loss, throughput) and log them to W&B in real-time. This gives us cloud dashboards, config tracking, and run comparison with zero C code changes.

## Method

### Approach

The CUDA binary already prints structured output:
- Config table: `| parameter | value |` rows
- Per-step: `step  4/1000: train loss 5.123456 (12.3 ms, 84000 tok/s)`
- Validation: `val loss 4.567890 (avg_train 5.123, gap +0.456)`
- Test: `test loss 4.321`

`wandb_train.py` wraps the binary, parses each line with regexes, and calls `wandb.log()`. All stdout passes through to the terminal unchanged.

### Setup

- `pip install wandb` (added to dev/Makefile venv target)
- One-time: `wandb login` with API key from https://wandb.ai/authorize

### Validation Run

Short modular arithmetic run to confirm end-to-end integration:

```bash
cd llm-fundamentals/dev
python wandb_train.py --project gpt2-cuda --name "wandb-test" -- \
    ./train -e model_grok.bin \
    -i data/modular/train.bin -j data/modular/val.bin -x data/modular/test.bin \
    -t 8 -b 256 -n 5000 -v 500 -s 0 \
    -l 0.001 -w 1.0 -a 0.98 -q 1337
```

Note: B=256 (not full-batch) because `gen_modular_data.py` produces a 3-way split
(4704/2352/2353), so val/test shards are too small for B=4703.

## Log

- 2026-02-21: Created `wandb_train.py` wrapper, experiment README, reproduction Makefile. Added `wandb` to dev/Makefile venv target.
- 2026-02-21: Validation run successful (5000 steps, B=256). All metrics logged to W&B dashboard.

## Results

### Metrics

| Metric | Logged? | Notes |
|--------|---------|-------|
| train_loss | yes | Per-step, 5000 data points |
| val_loss | yes | Every 500 steps, 11 data points |
| avg_train_loss | yes | Running average between val evals |
| generalization_gap | yes | val - avg_train |
| test_loss | yes | End of run |
| step_time_ms | yes | Per-step |
| tok_per_sec | yes | Per-step |
| config params | yes | 25 params auto-parsed from config table |

### Validation Run Results

5000 steps, B=256, mod-97 addition, wd=1.0, lr=0.001:
- Final train loss: 1.33
- Final val loss: 1.38 (near theoretical minimum of 1.307 for all-position loss)
- Test loss: 1.38
- Generalization gap: +0.045
- Throughput: ~2.2M tok/s, ~0.94 ms/step

### Graceful Degradation

If `wandb` is not installed, the wrapper prints a warning and runs the command normally — no crash, no missing output.

## Analysis

The wrapper successfully captured all structured output from the CUDA binary without any C code changes. Config parameters were auto-parsed from the table format and appeared correctly in the W&B run config. Step-level metrics (train loss, throughput) and periodic metrics (val loss, gap) were logged at the correct step numbers. The test loss was logged at the final step.

The approach of parsing stdout is robust because the CUDA binary's output format is stable and well-structured (printf format strings with fixed patterns). The regex patterns match cleanly with no false positives.

## Conclusions

- Python subprocess wrapper successfully captures all training metrics from the CUDA binary
- Zero C code changes required — the existing structured stdout is sufficient
- W&B dashboard provides real-time loss curves, config tracking, and run comparison
- Graceful degradation works: missing wandb just prints a warning

## Next Steps

- [x] Validate with a live training run
- [ ] Experiment 00014: HuggingFace Hub artifact export (requires `huggingface_hub` library, separate from wandb)

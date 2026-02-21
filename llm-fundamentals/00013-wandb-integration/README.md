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
    -t 8 -b 4703 -n 5000 -v 500 -s 0 \
    -l 0.001 -w 1.0 -a 0.98 -q 1337
```

## Log

- 2026-02-21: Created `wandb_train.py` wrapper, experiment README, reproduction Makefile. Added `wandb` to dev/Makefile venv target.

## Results

### Metrics

| Metric | Logged? | Notes |
|--------|---------|-------|
| train_loss | | Per-step |
| val_loss | | At val_loss_every intervals |
| avg_train_loss | | Running average between val evals |
| generalization_gap | | val - avg_train |
| test_loss | | End of run (if test set provided) |
| step_time_ms | | Per-step |
| tok_per_sec | | Per-step |
| config params | | Auto-parsed from config table |

### Graceful Degradation

If `wandb` is not installed, the wrapper prints a warning and runs the command normally — no crash, no missing output.

## Analysis

*To be filled after validation run.*

## Conclusions

*To be filled after validation run.*

## Next Steps

- [ ] Validate with a live training run
- [ ] Experiment 00014: HuggingFace Hub artifact export (requires `huggingface_hub` library, separate from wandb)

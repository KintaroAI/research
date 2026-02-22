# Experiment: Three-Way Comparison (Baseline vs Blend vs Hebbian)

**Date:** 2026-02-22
**Status:** In Progress
**Source:** *tagged on completion as `exp/00019`*
**W&B:** *links added on completion*

## Goal

Run baseline, blend-G8, and hebbian-H4 configurations back-to-back with
identical setup for direct side-by-side comparison. Experiments 00017 and 00018
each compared a single variant against baseline in separate runs; this experiment
puts all three on equal footing in one session.

## Hypothesis

Based on prior experiments:
- **Blend-G8** should converge close to baseline (within +0.01 val loss at 50k),
  as the model learns to compensate for the extra layer.
- **Hebbian-H4** should trail baseline by ~+0.06, as the non-learnable pull
  continuously distorts embeddings.
- Final ordering: baseline < blend < hebbian.

## Method

### Training setup

| Parameter | Value |
|-----------|-------|
| Model | 8-layer, 8-head, 512-dim GPT-2 (51M params) |
| Data | TinyStories (906M train tokens, 19M val tokens) |
| Batch size | 8 |
| Seq length | 512 |
| Steps | 50,000 |
| Learning rate | 3e-4 |
| Seed | 0 |

### Runs

All runs use `model_50m.bin` as starting weights, sequential on one GPU.

| # | Name | Extra flags | Checkpoint |
|---|------|-------------|------------|
| 1 | baseline | (none) | `~/data/exp19-baseline.bin` |
| 2 | blend-G8 | `-G 8` | `~/data/exp19-blend-G8.bin` |
| 3 | hebbian-H4 | `-H 4 -u 1e-5` | `~/data/exp19-hebbian-H4.bin` |

### Commands

```bash
# Run 1: Baseline
./venv/bin/python wandb_train.py --name "exp19-baseline" --tags exp19 -- \
    ./train -e model_50m.bin -b 8 -t 512 -n 50000 -q 0 -c ~/data/exp19-baseline.bin

# Run 2: Blend G8
./venv/bin/python wandb_train.py --name "exp19-blend-G8" --tags exp19 -- \
    ./train -e model_50m.bin -b 8 -t 512 -n 50000 -q 0 -G 8 -c ~/data/exp19-blend-G8.bin

# Run 3: Hebbian H4
./venv/bin/python wandb_train.py --name "exp19-hebbian-H4" --tags exp19 -- \
    ./train -e model_50m.bin -b 8 -t 512 -n 50000 -q 0 -H 4 -u 1e-5 -c ~/data/exp19-hebbian-H4.bin
```

## Results

*Pending — will be filled after training completes.*

## Analysis

*Pending.*

## Conclusions

*Pending.*

## Next Steps

*Pending.*

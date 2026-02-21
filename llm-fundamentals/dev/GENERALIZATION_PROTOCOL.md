# Generalization Testing Protocol

## Purpose

Repeatable procedure for comparing architecture variants on generalization.
Eliminates eyeballing val loss curves — instead uses matched-train-loss comparisons,
held-out test evaluation, and multi-seed averaging.

## Dataset Splits

- **Train**: TinyStories shards 2+ (~887M tokens)
- **Val**: TinyStories shard 0 (~19M tokens) — evaluated every N steps
- **Test**: TinyStories shard 1 (~19M tokens) — evaluated only at end

Re-generate splits with:
```bash
python prepare_data.py
```

For Shakespeare experiments:
```bash
python data/tinyshakespeare/tinyshakespeare.py
```

## Standard Configuration

```bash
./train -e model.bin -b 16 -t 256 -n 5000 -v 100 -s 0 \
        -q SEED -o log_VARIANT_sSEED.txt \
        -x data/tinystories/TinyStories_test.bin
```

Key flags:
- `-q SEED` — RNG seed for reproducibility
- `-x <path>` — test data pattern (evaluated at final step)
- `-o <path>` — log file for comparison
- `-s 0` — disable sampling for cleaner benchmarks

## Seed Control

Run each config with 3 seeds: **1337, 42, 7**

Report mean +/- std across seeds.

## Metrics (ranked by importance)

1. **Matched-train-loss val comparison** — generalization at equal training progress
2. **Final test loss** — unbiased held-out performance
3. **Best val loss** — peak generalization
4. **Gap trajectory** — overfitting behavior over time

## Procedure

1. Re-generate data: `python prepare_data.py` (produces train/val/test)
2. For each variant x each seed: run training with `-o` and `-x` flags
3. Compare: `python compare_runs.py log_variant1_s*.txt log_variant2_s*.txt`
4. Report: summary table, matched-train-loss table, gap trajectory

## Interpreting Results

- **Lower val at matched train loss** = better generalization
- **Smaller gap** = less overfitting
- **Lower test loss** = better held-out performance
- Check that train losses reached comparable ranges before drawing conclusions
- If train loss ranges don't overlap, matched-train-loss comparison is unreliable

## Example

```bash
# Baseline (dense) with 3 seeds
for seed in 1337 42 7; do
    ./train -e model.bin -b 16 -t 256 -n 5000 -v 100 -s 0 \
            -q $seed -o log_dense_s${seed}.txt \
            -x data/tinystories/TinyStories_test.bin
done

# Banded sparsity variant with 3 seeds
for seed in 1337 42 7; do
    ./train -e model.bin -b 16 -t 256 -n 5000 -v 100 -s 0 \
            -1 256 -2 256 -q $seed -o log_banded_s${seed}.txt \
            -x data/tinystories/TinyStories_test.bin
done

# Compare
python compare_runs.py log_dense_s*.txt log_banded_s*.txt
```

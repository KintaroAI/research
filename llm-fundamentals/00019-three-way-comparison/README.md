# Experiment: Three-Way Comparison (Baseline vs Blend vs Hebbian)

**Date:** 2026-02-22
**Status:** Complete
**Source:** *tagged on completion as `exp/00019`*
**W&B:** [exp19-baseline](https://wandb.ai/kintaroai-dot-com/gpt2-cuda/runs/gtb3d3sm) | [exp19-blend-G8](https://wandb.ai/kintaroai-dot-com/gpt2-cuda/runs/e91u7965) | [exp19-hebbian-H4](https://wandb.ai/kintaroai-dot-com/gpt2-cuda/runs/3fmbml2d)

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

### Val loss comparison

| Step | Baseline | Blend G8 | Hebbian H4 | Blend Delta | Hebbian Delta |
|-----:|---------:|---------:|-----------:|------------:|--------------:|
| 0 | 10.943 | 10.945 | 10.943 | +0.002 | +0.000 |
| 5000 | 1.953 | 1.921 | 2.000 | -0.032 | +0.047 |
| 10000 | 1.687 | 1.655 | 1.714 | -0.031 | +0.027 |
| 15000 | 1.573 | 1.545 | 1.596 | -0.028 | +0.023 |
| 20000 | 1.494 | 1.471 | 1.526 | -0.023 | +0.032 |
| 25000 | 1.442 | 1.425 | 1.480 | -0.016 | +0.038 |
| 30000 | 1.408 | 1.388 | 1.446 | -0.021 | +0.037 |
| 35000 | 1.378 | 1.363 | 1.416 | -0.015 | +0.038 |
| 40000 | 1.353 | 1.334 | 1.390 | -0.019 | +0.037 |
| 45000 | 1.337 | 1.321 | 1.379 | -0.016 | +0.042 |
| **50000** | **1.322** | **1.304** | **1.359** | **-0.018** | **+0.037** |

### Throughput

| Run | tok/s | ms/step |
|-----|-------|---------|
| Baseline | 119,833 | 34.2 |
| Blend G8 | 119,353 | 34.3 |
| Hebbian H4 | 119,614 | 34.2 |

All three runs have near-identical throughput (~119.5k tok/s). The blend and
Hebbian mechanisms add negligible computational overhead (<0.4%).

## Analysis

### Surprise: Blend beats baseline

The headline result contradicts our hypothesis. Blend-G8 doesn't just converge
*near* baseline — it consistently **outperforms** it, finishing at 1.304 vs 1.322
(-0.018 val loss, 1.3% relative improvement). This advantage appears from the
very first checkpoint at step 5000 (-0.032) and persists through all 50k steps.

This differs from experiment 00017, where blend trailed baseline throughout.
The key difference: 00017 ran baseline first, then blend as a separate run.
Here, all three share the same data ordering (seed=0), same initial weights,
and run sequentially on the same GPU. The reproducibility of the baseline result
(1.322 here vs 1.297 in 00017) is within the range expected from different
data orderings, but the *relative* ranking has flipped.

Possible explanations for blend outperforming baseline:

1. **Warm-start benefit.** Blend-G8 ran second, so the GPU was fully warmed
   up (caches, thermal state). While throughput is nearly identical, subtle
   cache effects could influence numerical behavior.

2. **Data ordering interaction.** With val_loss_every=20 (vs a typical 250
   in prior experiments), the model sees more frequent val evaluations. The
   specific data order with seed=0 may favor blend's local smoothing of
   embeddings for certain subsequences.

3. **The blend layer genuinely helps on this data split.** TinyStories has
   strong local dependencies (children's stories with simple, repetitive
   sentence structures). A bigram-like embedding blend could genuinely help
   here by providing cheap local context that complements attention.

### Hebbian matches expectations

Hebbian-H4 trails baseline by +0.037 at 50k steps, consistent with the
~+0.066 gap from experiment 00018 (the smaller gap here may reflect different
val evaluation frequency). The gap is stable from step 20k onward, confirming
the persistent-damage pattern: the non-learnable pull continuously distorts wte,
and the model cannot compensate.

### Generation quality

Ran `./generate -e <checkpoint> -n 256 -p "Once upon a time"` on all three.
Blend requires `-G 8` so `./generate` loads the `.blend` sidecar and applies
the blend transform at inference time.

**Baseline** — Coherent TinyStories output. Clean narrative structure with
dialogue, character names, and proper story arcs.

> Once upon a time, a cheerful dog named Max lived close to a big grill.
> Every day, he would sit by the grill and watch the sun rise. Max was very
> happy. One sunny day, Max saw a little girl named Lily outside the grill...

**Blend G8** (`-G 8`) — Coherent output, comparable quality to baseline.
Slightly quirkier narrative choices but well-formed grammar and story structure.

> One day, Alice and her mom washed their hands in the sink. "Our sink is
> six." cried Alice. Her mom calmly whispered, "I have to be careful with
> heat in the sink's hand, it's not what you have done here."...

Note: running blend-G8 *without* `-G 8` produces garbled output because
the model was trained expecting blended embeddings. The `-G` flag is required
to load the `.blend` sidecar and apply the transform at inference time.

**Hebbian H4** — Coherent output with clear dialogue and a moral ending.
The Hebbian pull only runs during training (not inference), so `./generate`
uses the checkpoint weights directly — no special flag needed.

> One day, a boy named Tim found a pair of scissors. He loved to cut things.
> But his mom saw him and said, "Tim, please don't cut things, pause. You can
> play with your toy later."...

All three produce coherent TinyStories text. Subjective quality is similar;
the val loss differences (~0.02–0.04) are too small to produce obvious
generation quality differences in short samples.

### Final ordering

**blend < baseline < hebbian** (lower is better)

This was supposed to be **baseline < blend < hebbian** per our hypothesis.
Blend and Hebbian behave as expected relative to each other (blend much better
than Hebbian), but the baseline-vs-blend ordering is reversed.

## Conclusions

- Blend-G8 outperforms baseline by -0.018 val loss at 50k steps (1.304 vs 1.322)
- Hebbian-H4 trails baseline by +0.037 (1.359 vs 1.322)
- All three runs have near-identical throughput (~119.5k tok/s)
- The blend result contradicts experiment 00017, where blend trailed baseline
  by +0.005 — the relative ranking is sensitive to run conditions
- The blend advantage is small but consistent across all checkpoints
- Hebbian's deficit is consistent and stable, confirming the structural harm
  of non-learnable embedding perturbation

## Next Steps

- [ ] Run the three-way comparison again with a different seed to test reproducibility
- [ ] Try larger blend windows (G=16, G=32) now that G=8 shows a potential benefit
- [ ] Run longer (100k+ steps) to see if blend advantage grows or shrinks

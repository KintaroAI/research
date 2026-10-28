# vm-00038: Two Stacked Boxes, Backprop Cut Between Them

**Date:** 2026-10-28
**Status:** Complete (pending tag)
**Source:** *tagged on completion as `exp/vm-00038`* (code: `visual-motor-agent/dev/reference/stacked_local.py`)
**Outputs:** `~/data/research/visual-motor-agent/exp_00038/`
**Task plan:** [`../tasks/00045-local-learning-stack.md`](../tasks/00045-local-learning-stack.md)

## Goal

The owner's question: think of the universal model as a box with x inputs and y outputs (x = y), and stack two:
input → bottom → top → output. If backprop is cut between them, can the stack learn? The signal at the cut has no
target of its own; the aim is to pass something through the cut that gives the bottom one. A PyTorch prototype in the
text world, five ways of training the bottom, and the owner's follow-up: one matrix for the bottom's output map and
the top's input map, so that the top shapes what the bottom outputs.

## Method

- **Data:** Tiny Shakespeare letters (`T_text_B.json`'s train split, 65 letters), random 512-letter windows, 8 per
  update, 30k updates; val nats per letter on eval_predict's windows (as `char_gpt.py`).
- **A box** (`reference/stacked_local.py`): 65 numbers per position in and out; `W_in` (128 × 65) plus a learned
  position, the universal model's Transformer (2 layers, d 128, 4 heads), `W_out` (65 × 128); 479k parameters. The
  bottom reads the letters one-hot; its output h is the top's input; the top's output is the next letter's logits.
- **Arms** (seeds 1–3 each):
  - **e2e:** backprop through both (the ceiling).
  - **frozen:** the bottom stays random (the floor).
  - **local:** the bottom trains on its own next-letter loss (h read as logits).
  - **dtp** (difference target propagation): an inverse box g learns to undo the top (on h plus noise σ = 0.1); the
    bottom's target is h* = h + g(t) − g(ŷ), t the top's output nudged toward the right letter (β = 1).
  - **dni** (synthetic gradients, conditioned): a box reads [h, the next letter] over the window and predicts the top
    loss's gradient at h; the bottom follows it; the predictor trains on the true gradient at the cut.
  - In dtp and dni the bottom learns at 0.1 × the learning rate (AdamW makes every step about lr in size, so the
    untrained helper's noise moved the bottom at full speed: on a toy sequence neither learned at 1 ×).
  - **+tie** (all five arms): the bottom's `W_out` and the top's `W_in` are one matrix, learning from every loss that
    reaches it (in the cut arms the top's loss reaches it, the bottom's own layers not).
  - **dtp+norm:** h layer-normalized at the cut (no parameters), β = 1 and β = 0.1, after dtp diverged.
- **Diagnostics:** dtp: the inverse's reconstruction error and the cosine between the bottom's target step (h* − h)
  and the true descent direction −∂L/∂h; dni: the predictor's error and the cosine between predicted and true gradient.
- Tests (`tests/test_stacked_local.py`, 18): only e2e passes gradient through the cut (and with the tie, only the
  shared matrix receives the top's loss); the difference target with an exact (or biased) inverse reaches its output
  target; the gradient at the cut against finite differences; each arm but dni learns a toy sequence; dni's predictor
  learns the true gradient with the boxes held fixed; cut-norm bounds h.
- 18–26 min per run (dtp ~1 h: the inverse and diagnostics), 15 at once on two GPUs.

## Results

### R1. Val nats per letter, best (and at 30k), mean of 3 seeds

| arm | untied: best | untied: 30k | tied: best | tied: 30k |
|---|---|---|---|---|
| e2e | **1.550** | 1.598 | 1.559 | 1.615 |
| local | **1.593** | 1.608 | 1.604 | 1.609 |
| frozen | 1.623 | 1.624 | 1.624 | 1.625 |
| dni | 1.636 | 1.638 | 1.647 | 1.649 |
| dtp | 2.082 | 2.191 | 2.259 | 2.287 |
| dtp + norm, β = 1 | 1.780 | 1.802 | | |
| dtp + norm, β = 0.1 | 1.764 | 1.774 | | |

Per seed (best): e2e 1.572 / 1.545 / 1.533 (at 15–17k, then overfitting); local 1.593 / 1.589 / 1.598 (the bottom
alone 1.61–1.62); frozen 1.629 / 1.620 / 1.621; dni 1.643 / 1.620 / 1.644 (still improving at 30k); dtp 2.051 /
2.068 / 2.126. All runs: `reference/stacked_summary.py exp_00038/0??_*`.

### R2. What crossed the cut (dtp, dni)

| arm | 5k | 10k | 20k | 30k |
|---|---|---|---|---|
| dtp: cosine of target step and true descent | 0.02–0.08 | 0.001 | 0.000 | 0.000 |
| dtp: inverse's reconstruction error | 40–130 | 270–400 | 620–2,200 | 1,600–6,500 |
| dtp + norm (β = 1): cosine | 0.03–0.04 | 0.01 | 0.002–0.005 | 0.001–0.004 |
| dtp + norm: reconstruction error (h's power: 65) | 25–40 | 42–50 | 50–57 | 45–58 |
| dni: cosine of predicted and true gradient | 0.33–0.37 | 0.27–0.30 | 0.27–0.28 | 0.28–0.29 |
| dni + tie: cosine | 0.39–0.40 | 0.33 | 0.31 | 0.31–0.32 |

## Analysis

- **The stack learns with the cut; the best way across it is the bottom's own loss.** A random bottom already lets
  the top reach 1.623 (it reads 65 random projections of a random transformer's view of the text); the bottom
  trained on its own next-letter loss gives 1.593, about 40% of the way to end to end (1.550). The top improves on the
  bottom's own prediction by ~0.02 nats (1.61 → 1.59): a refiner, not a second layer of features.
- **End to end overfits; the cut arms much less.** e2e peaks at 15–17k and loses 0.05 by 30k; local loses 0.015,
  frozen and dni nothing (dni is still improving).
- **Synthetic gradients carry a weak signal.** The predictor reaches a cosine of ~0.28 with the true gradient and
  stays there; the bottom moves (its own val 4.2 → 5.2) but the top gains nothing over a random bottom (1.636 against
  1.623). Predicting the gradient of a 33k-number signal from h and the next letter is harder than the toy suggested.
- **Target propagation fails, and the reason is the inverse.** Unbounded, h grows without limit (each target pushes
  it outward, the inverse trained near the old h lags, the next target is worse) and the target's alignment with the
  true descent falls to 0 within 10k updates. Bounded (layer-normalized at the cut), nothing explodes but the inverse
  still explains only 10–30% of h (error 45–58 of 65) and the target is uncorrelated with the descent (cosine ≤ 0.006):
  val 1.76–1.78, worse than a random bottom. A box that maps a 512-letter context to next-letter logits discards most
  of its input, so it cannot be inverted; DTP assumes layers that nearly can. Two-layer transformer boxes are far from
  that.
- **Tying the cut does not help.** With a random bottom the top choosing which 65 projections cross the cut is worth
  nothing over reading 65 random ones through its own map (1.624 against 1.623); where the bottom also learns, the
  shared matrix serves two objectives and every arm is slightly worse (local 1.604 against 1.593, e2e 1.559 against
  1.550, dni 1.647 against 1.636).

## Conclusions

1. Two boxes with backprop cut between them learn; the bottom's own prediction loss is the best of the four ways
   tried to train it, recovering ~40% of what end-to-end training adds over a random bottom.
2. Signals derived from the top (target propagation's targets, synthetic gradients) did not beat a random bottom
   here: DTP's inverse cannot invert a transformer box (its targets carry no descent direction), and the gradient
   predictor stays at a cosine of ~0.28.
3. Sharing the cut's matrix between the boxes (the top shaping the bottom's output) changes nothing or costs a little.

## Next Steps

- A top that keeps what it receives (a residual path from h to the output, or an inverse conditioned on h itself,
  g(t, h)), so that target propagation has something to invert.
- The local loss with a different job for the bottom (two letters ahead, or predicting the top's next input), and the
  local loss combined with synthetic gradients.
- A wider cut than the 65 letters (x ≠ y), to see whether the bottleneck limits the stack.
- The same comparison in the universal model in a world (the bottom predicting the next frame, the top refining it).

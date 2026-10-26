# vm-00027: The One-Hot Control

**Date:** 2026-10-26
**Status:** Complete
**Source:** `exp/vm-00027` (code: `visual-motor-agent/dev/`)
**Outputs:** `~/data/research/visual-motor-agent/exp_00027/`
**Task plan:** [`../tasks/00033-one-hot-control.md`](../tasks/00033-one-hot-control.md)

## Goal

vm-00026: with one letter per token (a letter as one grey level), the universal model reached 1.80 nats per letter, the
classical GPT of the same size 1.615. The suspected cause was the linear input map: letters drawn as grey levels lie on
one line in the embedding space. Here the world draws each letter as a one-hot glyph, so that `W_in` becomes an
embedding table, with no change to the model.

## Hypothesis

If the input map causes the gap, the one-hot arm comes close to the classical GPT (1.62).

## Method

- Arm C (`configs/T_text_C.json`): one letter per token, each a 9 × 9 one-hot glyph (81 values, pixel k bright for
  letter k of 65), 72 × 72 pages of 64 letters, `patch` 9, `bins` 2 (the values are 0 or 1); otherwise as vm-00026's B
  (d 128, 2 layers, ctx 8 = 512 letters, a lifetime of 100k ticks with memory U), seeds 1–3; 157–172 ticks/s.
- Scoring (`eval_predict.py`): the same 208 val windows as vm-00026; nats per letter by glyph (each letter's
  factorized probability, normalized over the 65 letters: for one-hot glyphs each letter's odds over the sum of odds).
- What else changes with one-hot glyphs: the output is 65 yes/no predictions instead of one softmax, and a token's
  loss is the mean over 81 mostly-zero values (the letters weigh less in training than in B).

## Log

### 2026-10-26
- **Implemented** glyph tables in the text world (`env.encoding`: level | onehot) and glyph scoring in
  `eval_predict.py` (task 00033).
- **Ran** the 3 lifetimes 16:05–16:16; scoring until 16:20.

## Results

Val nats per letter (C: by glyph; B and GPT from vm-00026):

| Run | 10k | 20k | 30k | 50k | 70k | 100k | best |
|---|---|---|---|---|---|---|---|
| C s1 (one-hot) | 2.056 | 1.891 | 1.820 | 1.796 | 1.771 | 1.769 | **1.769** (100k) |
| C s2 | 2.054 | 1.825 | 1.803 | 1.780 | 1.789 | 1.791 | **1.780** (50k) |
| C s3 | 2.055 | 1.858 | 1.842 | 1.798 | 1.803 | 1.792 | **1.792** (100k) |
| B (grey levels), 3 seeds | 2.39–2.40 | 2.03–2.10 | 1.94–1.98 | 1.82–1.89 | 1.84–1.85 | 1.80–1.82 | 1.796–1.804 |
| classical GPT, 3 seeds | 1.67–1.69 | 1.615–1.635 | 1.62–1.63 | 1.66–1.70 | 1.71–1.78 | 1.80–1.92 | 1.615–1.617 |

## Analysis

- **The embedding table speeds up learning but does not close the gap** (hypothesis rejected): one-hot letters learn
  much faster early (2.05 at 10k against B's 2.40; 1.83–1.89 at 20k against 2.03–2.10), so the linear input map does
  cost B something, but they end only 0.01–0.03 better (1.77–1.79 against 1.80). The gap to the classical GPT, 0.16
  nats per letter, remains.
- The one-hot output (65 yes/no predictions) and the light weight of letter tokens in the loss could hold C back; even
  so, C ends at the same place as B, whose output is a true softmax. Both universal arms plateau near 1.78–1.80; the
  classical GPT passes 1.62 at 20k updates.
- **What else differs from the classical GPT,** each a candidate for the gap:
  1. **The training data:** GPT draws each window from a random letter offset (about 1M distinct windows); the
     lifetime trains on the newest window and on chunks replayed from memory, all aligned to pages of its stream (the
     stream's alignment shifts by 334 letters per pass over the text, so about 2,000 distinct chunks per pass and a
     dozen thousand in memory). B's train loss 1.57 against val 1.80 suggests that it memorises those chunks.
  2. The lifetime's own tokens: a body and a move token per step (the move sampled from the policy, noise to predict).
  3. The 256-bin output (B) against 65 classes; in C the yes/no output.

## Conclusions

1. The linear input map is a real cost (slower learning), but not the cause of the universal model's gap to a
   classical GPT on text. My hypothesis was wrong.
2. The next suspect is the training data: page-aligned windows from a stream and its memory against random windows.
   A direct test: the classical GPT trained on the lifetime's aligned windows, and the universal model trained on
   random windows (offline).

## Next Steps

- The data test above (both directions).
- Then decide between the input fix (learned embedding selection, or a learned patch vocabulary: VQ / FSQ) and a
  change in how the lifetime samples its memory.

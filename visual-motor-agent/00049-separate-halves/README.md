# vm-00049: Two Separate Transformers That See Every Summary

**Date:** 2026-10-30
**Status:** Complete (pending tag)
**Source:** `exp/vm-00049` (code: `visual-motor-agent/dev/`)
**Outputs:** `~/data/research/visual-motor-agent/exp_00049/`
**Task plan:** [`../tasks/00057-separate-halves.md`](../tasks/00057-separate-halves.md)

## Goal

The owner's design: two completely separate transformers, one per half of the view, each attending fully to its own
half's patches and to every summary, never to the other half's patches. No shared weights. vm-00047 (VS) was the same
rule as a mask inside one model; here the halves are two models, and the only thing that passes between them is the
summaries' thumbnails (each transformer has its own copy of every summary token, option 1 of the two weighed). This is
the form a split takes when it is meant to save compute or run on two devices.

## Method

- **`model.band_n = S`, `band_k`, `band_cols`** (task 00057; `vm_uni_has_patch` / `vm_uni_layout` in `vm_agent.h`,
  `agent_lib.has_patch`, `torch_reference.uni_has_patch`): the model is band k of S. Its patch tokens are only its
  band's (rows, or with `band_cols` columns); every summary, the body and the move stay where they are; attention is
  the usual block-causal one, with no mask. Header slots 54–56. In the 2× text world a band model has 215 tokens per step
  (128 patches, 85 summaries, body, move) against 343, and 688,642 parameters.
- **What crosses between the halves** is only the thumbnails: a summary's input is its block of the frame,
  box-averaged. In this world a B1 is exactly one letter, so every finished letter of the other half arrives exactly.
  Test: the right half's pixels shuffled within 2 × 2 squares (every thumbnail kept) leave the left model's outputs
  equal to the bit (C++ and GPU), and change the right and full models'.
- **Tests** (`tests/test_separate_halves.py`, 28): layouts, the shuffle, C++ and GPU losses and gradients against the
  reference, checkpoints, invalid configs. Full suite: 1,521 passed, 30 skipped.
- **Runs:** `configs/T_text_B2.json` with `sum_only = 0`, `uni_pyr = 4` (vm-00046's NB3), `band_n = 2`; seed 1,
  50k ticks, checkpoints every 10k. Each half is its own lifetime on the same text (the text world ignores the moves)
  with its own memory. **V0, V1:** vertical halves (`band_cols = 1`); **H0, H1:** horizontal (`band_cols = 0`). All
  four on the 3090 at once: 7.1 ticks/s each.
- **Measure:** next-letter cross-entropy on the val text (`letter_ce.py --bands`: each letter scored by the half that
  owns its top-left patch, all 64 letters), against NB3 (no split) and VS (the mask), scored the same way.

## Results

### R1. Next-letter cross-entropy on the val text

| run | 10k | 20k | 30k | 40k | 50k |
|---|---|---|---|---|---|
| NB3 (one model, no split) | 1.903 | 1.760 | **1.737** | 1.741 | 1.799 |
| **H0 + H1** (horizontal, separate) | 1.917 | 1.760 | **1.719** | 1.721 | 1.759 |
| VS (vertical, mask, shared weights) | 2.135 | 1.972 | 1.904 | **1.857** | 1.868 |
| **V0 + V1** (vertical, separate) | 2.406 | 2.202 | 2.128 | **2.093** | 2.098 |

vm-00046's NB (the normal world, 66 tokens): 1.721 at its best.

### R2. By letter column at 30k (V: 1–4 the left transformer, 5–8 the right)

| run | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|
| H0 + H1 | 1.950 | 1.698 | 1.687 | 1.683 | 1.676 | 1.690 | 1.678 | 1.691 |
| VS | 1.856 | 1.678 | 1.692 | 1.652 | 1.971 | 2.479 | 2.072 | 1.830 |
| V0 + V1 | 1.950 | 1.701 | 1.698 | 1.681 | **3.421** | 2.585 | 2.108 | 1.881 |

By line, H is level with NB3 everywhere (line 5, the bottom transformer's first, is 1.85 in H and ~1.88–2.26 in NB3
across checkpoints: a bump of the text or the layout, not of the cut).

## Analysis

- **Horizontal halves lose nothing.** Two transformers with no shared weights, exchanging only thumbnails, predict as
  one model that sees everything: 1.719 against 1.737 at the best point (and level with NB, 1.721). The lead is within
  one seed's noise; the claim is "no loss". The reading order crosses the cut once per page, and by then every line
  above it is finished, so its B1s carry it exactly.
- **Vertical halves lose the right half's line start, more than the mask did.** The left transformer predicts as NB3
  (1.68–1.70). In the right one, letter 5's predictor is the last right token of the row above, which has seen nothing
  of its line (in VS it was the last left patch, which reads letters 1–4): 3.42 against 1.97. Letters 6–8 recover as
  they read more of their line (2.59, 2.11, 1.88). Best 2.093 against VS's 1.857.
- **Both splits overfit later.** At 50k H holds 1.759 where NB3 is at 1.799, as VS held against NB3 in vm-00047. Each
  transformer memorizes less of a page it sees only half of in full. One seed: a hint.
- **Each half also predicts the other half's letters**, through its copies of the B1 summaries (a B1 target is a
  letter), but only after the letter's bottom patch row: that is too late to stand in for letter 5's context.

## Conclusions

1. Locality is free when the cut follows the order of the information: what each half needs from the other is then
   already summarized when it needs it. Separate weights do not cost quality here.
2. Where a half needs the other half's content before any summary of it exists (the right half's line start), a
   separate transformer has nothing to read, and loses more than a masked one, whose predictor at the cut still saw
   the left patches.
3. With every summary in every transformer, the summaries are ~N/3 tokens per transformer: fine for two halves, but it
   does not scale to many pieces. A logarithmic version reads only a neighbourhood of summaries per level.

## Next Steps

- Quarters, horizontal (`band_n = 4`): does the result hold when each piece has two lines?
- Summaries by neighbourhood (O(log N) per transformer instead of all), the scalable form.
- A combined dream: assemble the frame from each half's patch predictions and recompute the thumbnails.
- A rule for the move (adding the halves' move logits, or one half owns it) before a visual world.

# Task 00057: Two Separate Transformers That See Every Summary (experiment vm-00049)

**Date:** 2026-10-29
**Status:** Complete

## Context

- The owner's design (2026-10-29): two completely separate transformers, one per half of the view. Each attends fully
  to its own half's patches and to every summary, never to the other half's patches. No weights shared.
- vm-00047 (VS, task 00055) was the same rule as a mask inside one model: shared weights, and one copy of each summary
  that every token reads. Its left half predicted as the unsplit model (1.69), its right half was blind to its own
  line's start (2.03): in the 2× text world a letter's B1 completes a patch row after the right half needs it.
- Option 1 of the two the owner weighed (the other: shared summary tokens, each computed once by its home half and
  read by the other, which couples the two through gradients): **each transformer has its own copy of every summary
  token.** A summary's input is its block's thumbnail (`sum_thumb = 1`), data from the frame, not something computed;
  so the halves exchange only downscaled pixels. In the 2× text world a B1 is exactly one letter, so every finished
  letter of the other half reaches a transformer exactly; B2–B4 are blurrier.

## Design

- **`model.band_n = S`, `model.band_k = k`, `model.band_cols`** (universal, `uni_pyr` ≥ 1, `sum_thumb = 1`): the
  model is band k of S. Its patch tokens are only those of band k (rows `band_cols = 0`, columns `band_cols = 1`); every
  summary, the body and the move stay, at the same places in the step (a summary after its block's last patch, present
  or not). Attention is the usual block-causal one over these tokens: no mask.
  - It is `vm_uni_layout`'s filter, as `sum_only` (task 00054) is: everything else reads the slot table (patch values
    by their index, the 2-D position codes by row and column, the targets by the next slot).
  - Checkpoint header slots 54–56. `agent_lib.uni_layout`, `torch_reference.uni_slot_table` and `uni_values` follow.
  - Invalid with `sum_only`, `attn_split` > 1, `attn_win`, `uni_sum`, or S not dividing the grid.
- **A pair is two runs**: two lifetimes, one per band, on the same text (the text world ignores the moves; same seed),
  each with its own memory. Nothing passes between them.
- **The move:** each transformer predicts one; in the text world it does nothing. For a visual world a rule is needed
  (later: add their move logits, or one transformer owns the move).
- **`letter_ce.py --bands`**: the runs given are the bands of one model; each letter is scored by the run whose patch
  tokens hold its top-left patch, and the score is over all 64 letters.

## Verification

- `tests/test_separate_halves.py`: the layout is the full one without the other bands' patches (rows and columns,
  C++ and reference); the token values at the kept slots equal the full model's (C++ and GPU); C++ and GPU losses and
  gradients against the reference; checkpoint round trip; invalid configs.

## Experiment vm-00049

- The 2× text world (`configs/T_text_B2.json` with `sum_only = 0`, `uni_pyr = 4`, as NB3 and VS), seed 1, 50k ticks,
  checkpoints every 10k (text lifetimes peak at ~30k):
  - **V0, V1:** vertical halves (`band_n = 2`, `band_cols = 1`; `band_k` 0, 1), against VS (the mask).
  - **H0, H1:** horizontal halves (`band_cols = 0`): the reading order crosses the cut once per page, and by then the
    lines above are finished.
- **Measure:** next-letter cross-entropy on the val text (`letter_ce.py --bands`), by column and line, against NB3
  (1.737 at 30k) and VS (1.857 at best). Hypotheses: H ≈ NB3; V ≈ VS (the right half blind to its line's start).
  Each transformer has half the patch targets of NB3, but every summary's: a B1 target is the next letter too.

## Results

- `tests/test_separate_halves.py` (28): the layouts; the right half's pixels shuffled within 2 × 2 squares (every
  thumbnail the same) leave the left model's outputs equal to the bit (C++ and GPU) and change the right and full
  models'; C++ and GPU losses and gradients against the reference; checkpoints; invalid configs. Full suite: 1,521
  passed, 30 skipped.
- `letter_ce.py --bands` reproduces VS at 30k (1.9038). Smoke pair (2k ticks): 2.69.
- A band model: 215 tokens per step (128 patches, 85 summaries, body, move) against NB3's 343; 688,642 parameters.
  Four lifetimes on the 3090 together: 7.1 ticks/s each (two together: 14.4).

### Next-letter cross-entropy (val), by letter column

| run | ticks | all | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|---|---|
| NB3 (no split) | 10k | 1.903 | 2.484 | 1.820 | 1.835 | 1.811 | 1.826 | 1.827 | 1.814 | 1.808 |
| VS (mask, shared weights) | 10k | 2.135 | 2.308 | 1.901 | 1.882 | 1.865 | 2.369 | 2.645 | 2.163 | 1.944 |
| **H0 + H1** | 10k | 1.917 | 2.456 | 1.855 | 1.846 | 1.826 | 1.839 | 1.846 | 1.838 | 1.830 |
| **V0 + V1** | 10k | 2.406 | 2.451 | 1.918 | 1.900 | 1.878 | **3.782** | 2.923 | 2.321 | 2.074 |
| NB3 | 20k | 1.760 | 2.148 | 1.714 | 1.716 | 1.693 | 1.697 | 1.708 | 1.700 | 1.707 |
| VS | 20k | 1.972 | 1.982 | 1.744 | 1.750 | 1.712 | 2.091 | 2.521 | 2.104 | 1.869 |
| **H0 + H1** | 20k | 1.760 | 2.093 | 1.726 | 1.728 | 1.695 | 1.706 | 1.711 | 1.709 | 1.711 |
| **V0 + V1** | 20k | 2.202 | 2.044 | 1.728 | 1.718 | 1.702 | **3.542** | 2.713 | 2.195 | 1.974 |
| NB3 | 30k | 1.737 | | | | | | | | |
| VS | 30k | 1.904 | 1.856 | 1.678 | 1.692 | 1.652 | 1.971 | 2.479 | 2.072 | 1.830 |
| **H0 + H1** | 30k | 1.719 | 1.950 | 1.698 | 1.687 | 1.683 | 1.676 | 1.690 | 1.678 | 1.691 |
| **V0 + V1** | 30k | 2.128 | 1.950 | 1.701 | 1.698 | 1.681 | **3.421** | 2.585 | 2.108 | 1.881 |
| NB3 | 40k | 1.741 | | | | | | | | |
| VS | 40k | 1.857 | | | | | | | | |
| **H0 + H1** | 40k | 1.721 | 1.903 | 1.698 | 1.693 | 1.695 | 1.682 | 1.702 | 1.695 | 1.703 |
| **V0 + V1** | 40k | 2.093 | 1.892 | 1.708 | 1.703 | 1.701 | 3.331 | 2.498 | 2.065 | 1.845 |
| NB3 | 50k | 1.799 | 1.943 | 1.781 | 1.793 | 1.772 | 1.764 | 1.781 | 1.768 | 1.788 |
| VS | 50k | 1.868 | | | | | | | | |
| **H0 + H1** | 50k | 1.759 | 1.894 | 1.749 | 1.726 | 1.747 | 1.721 | 1.752 | 1.738 | 1.748 |
| **V0 + V1** | 50k | 2.098 | 1.891 | 1.747 | 1.728 | 1.740 | 3.304 | 2.480 | 2.062 | 1.830 |

- H is level with NB3 (by line too: line 5 is 2.26–2.30 in both, a property of the text or of the layout, not of the
  cut).
- V's letter 5 is the worst: in the right transformer its predictor is the last right token of the row above, which
  has seen none of its line (in VS it was the last left patch, which reads letters 1–4).

- **Write-up:** [`../00049-separate-halves/README.md`](../00049-separate-halves/README.md). Horizontal halves lose
  nothing (1.719 at best against NB3's 1.737); vertical halves lose the right half's line start (2.093 against VS's
  1.857). Both splits overfit later than NB3.

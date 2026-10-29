# vm-00042: Summary Tokens With Content: Thumbnails of Their Block

**Date:** 2026-10-28
**Status:** In progress (image analysis running)
**Source:** *to be tagged on completion as `exp/vm-00042`* (code: `visual-motor-agent/dev/`)
**Outputs:** `~/data/research/visual-motor-agent/exp_00042/`
**Task plan:** [`../tasks/00049-summary-thumbnails.md`](../tasks/00049-summary-thumbnails.md)

## Goal

Summary tokens (vm-00040, vm-00041) have no values: their content comes only from attention, and the token before one
has no target. Every arm with summaries lost about half the food, even with full attention (vm-00040's ER), and the
suspect was a content-free token predicting the move. The owner's idea: give each summary content, its block of
patches downsampled to one patch (an image pyramid inside the token sequence). Does a move predictor with content
bring behaviour back? Does it help text?

## Method

- **Thumbnails** (`model.sum_thumb = 1`; tests in `tests/test_summary_thumbs.py`): a summary's values are its block's
  pixels box-averaged down to one patch (ps × ps, the same V values as a patch). A level-ℓ pyramid summary averages
  2^ℓ × 2^ℓ squares of pixels, so level 3 (the global summary) is the whole 64 × 64 view at 8 × 8.
  - The block ends right before its summary, so a thumbnail holds only pixels already seen.
  - Every token has a target again: the token before a summary predicts its thumbnail, and the loss is the mean over
    all tokens.
  - C++ `uni_values`, CUDA `uni_value_` (the slot table gains the block's origin and pooling factors) and the
    reference compute the thumbnails from the frame; `text_dream.py` recomputes them from the letters after each
    generated letter. Checkpoint header slot 52.
  - Tests: thumbnails against a numpy box average (pyramid levels 1–3 and row summaries), no leakage, C++ and GPU
    against the reference (loss and every gradient), the instinct's candidates, the lookahead, self-forcing, the
    dream step.
- **A GPU memory leak found and fixed** on the way: `vmc_free` missed several buffers (1.3 GB per model in the test
  suite, 14 GB peak), and a failed `cudaMalloc` went unreported. Buffers now free themselves (a destructor) and an
  allocation failure prints "out of device memory". A test closes a model and checks the memory comes back.
- **Arms**, 100k ticks, seeds 1–3:
  - **EPT:** egocentric touch world, `attn_win = 1`, `uni_pyr = 3`, `sum_thumb = 1` (vm-00041's EP with thumbnails).
  - **EFT:** `uni_pyr = 3`, `sum_thumb = 1`, full attention.
  - **EF:** `uni_pyr = 3`, full attention, no thumbnails (vm-00040's ER with the pyramid).
  - **TPT:** text world, `attn_win = 1`, `uni_pyr = 3`, `sum_thumb = 1` (vm-00041's TP with thumbnails).
  - Controls: vm-00036's EB (full attention, no summaries), vm-00041's EP and TP.
- **Measures** as in vm-00041. The attention mask is unchanged by thumbnails, so the cost is vm-00041's.

## Results

### R1. Text: val nats per letter

| arm | best, s1 / s2 / s3 (mean) | at 10k |
|---|---|---|
| full attention (vm-00030) | 1.717 / 1.736 / 1.745 (1.733) | 1.88–1.97 |
| TH: window + line summaries (vm-00040) | 1.773 / 1.759 / 1.763 (1.765) | 2.16–2.18 |
| TP: window + pyramid (vm-00041) | 1.922 / 1.925 / 1.916 (1.921) | 2.24–2.28 |
| **TPT: window + pyramid with thumbnails** | 1.930 / 1.919 / 1.933, all at 100k (**1.927**) | 2.41–2.50 |

By the predicted letter's column and line (mean of the seeds at 100k):

```
column:        0     1     2     3     4     5     6     7
TP (pyramid) 1.85  2.37  2.06  1.81  1.82  1.83  1.81  1.82
TPT (thumbs) 1.89  2.36  2.06  1.81  1.87  1.81  1.83  1.80

line:          0     1     2     3     4     5     6     7
TP (pyramid) 1.91  1.97  1.92  1.93  1.84  1.97  1.91  1.91
TPT (thumbs) 1.90  1.95  1.91  1.95  1.87  1.98  1.91  1.94
```

### R2. Text: generation (seed 1 at 100k, 32 val windows, 4 pages, T = 0.8)

| model | samples' nats per letter under GPT | mean line length | lines over 60 letters |
|---|---|---|---|
| real text | 1.756 | 28.8 | 0% |
| TH (window + line summaries, vm-00040) | 1.775 | 35.1 | 7% |
| TP (window + pyramid, vm-00041) | 2.365 | 34.9 | 19% |
| **TPT (with thumbnails)** | **2.292** | 34.7 | 13% |

The torch forward agrees with the reference (0.0003). From the same context as before ("Hearest thou, Biondello?"):

```
TPT: COMINIUS:
     I tell is name matter last you.

     Second Citizen:
```

### R3–R5. Image (frozen tests, life, prediction, dream)

*Analysis running: two 4-step tests were lost to out-of-memory errors and are being re-run.*

## Analysis

*Pending the image results.* Note (task 00052): these models were born with misplaced position codes (see
vm-00041's caveat); vm-00045 reruns EF and EFT with the codes fixed.

## Conclusions

*Pending.*

## Next Steps

- vm-00043 (square poison) and vm-00044 (poison with a 2 × 2 checker): worlds where poison stays distinguishable in a
  coarse view, to test whether thumbnails help once they can show the difference.

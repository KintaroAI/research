# Task 00049: Summary Tokens With Content: Thumbnails of Their Block (experiment vm-00042)

**Date:** 2026-10-28
**Status:** In progress

## Context

- Summary tokens (tasks 00047, 00048) have no values. Their input is b_in plus their slot and step embeddings, so
  their content comes only from attention, and the token before one has no target, the one exception to "every
  token predicts the next token's values".
- vm-00040 found that behaviour drops when a summary predicts the move, even with full attention (ER: 60% of the
  food). The suspect is a content-free token as the move's predictor.
- The owner's idea (2026-10-28): give a summary content. Downsample its block of patches into one patch-sized
  thumbnail and use that as its values, an image pyramid (mip-map) built into the token sequence.

## Design

- **`model.sum_thumb = 1`** (needs summaries: `uni_sum` or `uni_pyr`). A summary's values are its block's pixels,
  box-averaged down to one patch (ps × ps pixels, the same channels, so the same V values as a patch).
  - Pyramid level ℓ: a 2^ℓ·ps square, each thumbnail pixel the mean of a 2^ℓ × 2^ℓ square of pixels.
  - Row summary (`uni_sum` R): a band of R·ps rows by the full width, each thumbnail pixel the mean of an R × g
    rectangle.
  - The block ends right before the summary in the sequence, so a thumbnail contains only pixels already seen: no
    leakage.
- **Every token has values again.** The token before a summary predicts its thumbnail; the loss is the mean over
  all tokens, as before task 00047. The summaries still attend as before: a thumbnail is their input, and attention
  adds the rest.
- **What it means per world:**
  - Image: coarse views of 16 × 16, 32 × 32 and 64 × 64 pixels. Fine texture (poison's checker) averages out, so
    brightness and shape remain.
  - Text (letters are 2 × 2 blocks of one grey): a level-1 thumbnail keeps the block's 4 letters exactly, one pixel
    each. Coarser levels average different letters' grey levels, which carry little meaning. It is the same rule, and
    the model learns what to use.
- **Implementation:**
  - The values come from the frame wherever patch values do: C++ `uni_values`, CUDA `uni_value_` (the slot table
    gains each summary's block origin and pooling factors), the reference `uni_values`.
  - `text_dream.py` builds token values from letters itself, so it recomputes a step's thumbnails from its letters
    after each generated letter.
  - Tools that rebuild frames (the lookahead, self-forcing, the dream) are unchanged: the model derives the
    thumbnails from the frames.
  - Loss normalization: the number of tokens without a target is the number of summaries, or 0 with thumbnails.
  - Checkpoint header slot 52.

## Verification

- Thumbnails against a direct numpy box-average of the frame, for pyramid levels 1..3 and for row summaries.
- No leakage: a summary's values do not change when a pixel outside its block, or a later patch, changes.
- C++ and GPU against the reference: loss and every gradient, image and text configs, pyramid and rows, with and
  without the window. The instinct's candidates, the lookahead, self-forcing, the dream step. Checkpoints keep the
  flag.
- `text_dream.py`'s forward against the reference with thumbnails, and its generated windows' thumbnails consistent
  with their letters.
- With `sum_thumb = 0` everything is unchanged.

## Experiment vm-00042

- **Image** (`configs/U_touch_ego.json`, the 1-step instinct), 100k ticks, seeds 1–3:
  - **EPT:** `attn_win = 1`, `uni_pyr = 3`, `sum_thumb = 1`. Against vm-00041's EP (the same without thumbnails).
  - **EFT:** `uni_pyr = 3`, `sum_thumb = 1`, full attention.
  - **EF:** `uni_pyr = 3`, full attention, without thumbnails.
  - EF against EFT is the test of vm-00040's suspicion: does a move predictor with content, the whole-view
    thumbnail, bring behaviour back to EB's? EB (vm-00036) is the control without summaries.
- **Text** (`configs/T_text_B.json`), 100k ticks, seeds 1–3:
  - **TPT:** `attn_win = 1`, `uni_pyr = 3`, `sum_thumb = 1`. Against vm-00041's TP.
- **Measures** as in vm-00041.

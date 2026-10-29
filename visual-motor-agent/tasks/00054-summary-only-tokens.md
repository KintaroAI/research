# Task 00054: Tokens From the Summaries Alone, Text at Twice the Scale (experiment vm-00046)

**Date:** 2026-10-29
**Status:** In progress

## Context

- The owner's test of the pyramid's premise (2026-10-29): a level-1 summary's thumbnail is its block, downscaled. If
  the world is drawn at twice the scale (letters too), a level-1 thumbnail is exactly what the normal world's patch
  holds. A model whose tokens are those thumbnails sees the same information, in the same order, as the normal model,
  and should learn as well.
- Text world only. Two runs, compared by their training loss; no frozen tests or evaluations afterwards.

## Design

- **`model.sum_only = 1`** (needs `uni_pyr` ≥ 1 and `sum_thumb = 1`; not with `uni_sum`, `attn_win` or regions): the
  patches are not tokens. A step is [body, the summaries in the pyramid's order, move]; the patches are read only
  to compute the thumbnails. With `uni_pyr = 1` the summaries are B1 row by row (B1(r, c) follows patch
  (2r + 1, 2c + 1) in the full layout, so the order is row-major on B1's grid).
  - `vm_uni_layout` leaves the patches out; everything else reads the slot table (task 00048), so the thumbnails,
    targets, loss, position codes (block centres) and the GPU need no other change.
  - Checkpoint header slot 53. Python: `agent_lib.uni_layout`, `torch_reference.uni_slot_table`.
- **The scaled text world** (`configs/T_text_B2.json`): `env.size = 32`, `env.cell = 4` (every letter a 4 × 4 block
  of its grey, the same 64 letters per page, the same reading order), model `img = 32`, `patch = 2`, `uni_pyr = 1`,
  `sum_thumb = 1`, `sum_only = 1`. A step is 66 tokens of 4 values each, as in `T_text_B`.

## Verification

- `tests/test_summary_only.py`: the layout ([body, 64 B1 row-major, move] for a 16 × 16 patch grid); the scaled
  world's frame is the normal one upscaled 2× (`np.kron`); with the same parameters, a `T_text_B` model on a frame and
  a `T_text_B2` model on its 2× upscale give the same outputs (C++ and GPU); invalid configs refused; the header keeps
  the flag.

## Experiment vm-00046

- 100k ticks, seed 1, full attention, no window:
  - **NB:** `configs/T_text_B.json` (the normal model).
  - **NB2:** `configs/T_text_B2.json` (the 2× world, tokens = B1 thumbnails).
- **Measure:** the training loss in `metrics.jsonl` (`uni_ce_patch`, per value). Expected: the same curve within seed
  noise. No evaluations afterwards (the owner's instruction).

## Results

- `tests/test_summary_only.py` (9): with the same parameters, the normal model on a frame and the B1-only model on its
  2× upscale give the same values and move probabilities (C++ and GPU, within 2e-5: a thumbnail is 4v / 1020 against
  v / 255).
- vm-00046 (seed 1, 100k ticks, 160 ticks/s each): the training cross-entropy per value of the letter tokens.

  | ticks | NB (normal) | NB2 (2× world, B1 only) | difference |
  |---|---|---|---|
  | 0–5k | 1.639 | 1.642 | +0.003 |
  | 5–10k | 1.237 | 1.240 | +0.003 |
  | 10–20k | 1.230 | 1.229 | −0.001 |
  | 20–40k | 1.170 | 1.163 | −0.007 |
  | 40–60k | 1.092 | 1.084 | −0.008 |
  | 60–80k | 1.042 | 1.034 | −0.008 |
  | 80–100k | 1.012 | 1.003 | −0.009 |

  The same curve within 1% throughout (one seed each; the remaining difference is the init: the slot codes sit at the
  B1 block centres instead of the cells, and the random draws differ). The premise holds: a level-1 thumbnail carries
  exactly what the normal world's token does, and a model fed only those learns the same.

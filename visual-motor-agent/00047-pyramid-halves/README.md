# vm-00047: Two Halves That Share the Summaries

**Date:** 2026-10-29
**Status:** Complete
**Source:** `exp/vm-00047` (code: `visual-motor-agent/dev/`)
**Outputs:** `~/data/research/visual-motor-agent/exp_00047/`
**Task plan:** [`../tasks/00055-pyramid-halves.md`](../tasks/00055-pyramid-halves.md)

## Goal

The owner's design: split the model into two transformers, one per vertical half of the view. Each attends to its own
patches and to every summary token; the patches are private to a half, and the summaries are what the halves exchange.
This is a step toward breaking attention's quadratic cost by locality: if a half needs only its own patches plus the
summaries, its cost grows with its own size, and the summaries carry what is far away. Here it is a mask (the full
QKᵀ is still computed), so it measures the quality, not the saving.

## Method

- **The rule** (`model.attn_split = S` with `uni_pyr` ≥ 1, `attn_cols = 1` for vertical bands; `vm_attn_allow` in
  `vm_agent.h`, mirrored in `torch_reference.attn_allow`): causally, as always, every token attends to every summary
  (all levels, both halves), the body and the move; it attends to a patch only if the patch is in one of its own
  regions. A patch is in its band's region; a summary in every region its block spans; the body and the move in all.
  - In the 2× text world (16 × 16 patches, 8 × 8 letters of 2 × 2 patches): the left half is patch columns 0–7
    (letters 1–4 of each line), the right half columns 8–15 (letters 5–8). B1 (one letter), B2 (2 × 2 letters) and B3
    (a 4 × 4-letter quarter) each lie in one half; B4 (the page) spans both and reads every patch. It is the token
    that predicts the move, so the move needs no special case.
  - 62% of a step's causal token pairs stay allowed. Left patches read no right patch, and vice versa (0 of 16,384
    pairs); every patch reads every summary.
- **Tests** (`tests/test_pyramid_halves.py`, 14): the table (rows and columns); information flow (a change to right
  patch (0, 2) of the last step does not reach the left patches of row 1, which come before any summary that reads
  it, and does without the split); C++ and GPU losses and gradients against the reference; invalid configs. Full
  suite: 1,488 passed, 30 skipped.
- **Arm VS:** vm-00046's NB3 (`configs/T_text_B2.json` with `sum_only = 0`, `uni_pyr = 4`: 343 tokens per step, full
  attention, 705,026 parameters) with `attn_split = 2`, `attn_cols = 1`; seed 1, 100k ticks, checkpoints every 10k;
  24 ticks/s on the 4090 (as NB3: a mask saves nothing).
- **Measure:** next-letter cross-entropy on the val text (`dev/letter_ce.py`, as vm-00046), at every 10k, and by
  letter column.

## Results

### R1. Training loss against NB3 (the same tokens and targets; mostly copies, so low)

| ticks | 0–5k | 5–10k | 10–20k | 20–30k |
|---|---|---|---|---|
| NB3 | 0.565 | 0.303 | 0.289 | 0.274 |
| VS | 0.630 | 0.409 | 0.401 | 0.394 |

### R2. Next-letter cross-entropy on the val text

| run | 20k | 30k | 40k | 50k | 60k | 70k | 80k | 90k | 100k |
|---|---|---|---|---|---|---|---|---|---|
| NB3 (no split) | 1.760 | **1.737** | 1.741 | | | | | | 1.926 |
| **VS** (halves) | 1.972 | 1.904 | **1.857** | 1.868 | 1.859 | 1.864 | 1.882 | 1.864 | 1.872 |

vm-00046's NB, NB2, NB3a: 1.720–1.725 at their best (30k), 1.97 at 100k.

### R3. By letter column (1–4 the left half, 5–8 the right)

| run | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|
| NB3, 20k | 2.148 | 1.714 | 1.716 | 1.693 | 1.697 | 1.708 | 1.700 | 1.707 |
| NB (normal world, no split), 30k | 1.784 | 1.728 | 1.716 | 1.699 | 1.711 | 1.706 | 1.709 | 1.712 |
| VS, 20k | 1.982 | 1.744 | 1.750 | 1.712 | 2.091 | 2.521 | 2.104 | 1.869 |
| VS, 60k | 1.789 | 1.645 | 1.697 | 1.633 | 1.814 | **2.418** | 2.062 | 1.816 |

## Analysis

- **The left half is as good as an unsplit model; the right half is not.** At 60k the left columns average 1.69, the
  right 2.03. Column 6 stays at 2.40–2.43 from 50k to 90k: the loss does not shrink with training.
- **The cause is when the summaries arrive, not that they are shared.** A letter is predicted at its top patch row,
  but its B1 is complete only after its bottom row. While the right half predicts letters 5–8 of a line, the B1s of
  letters 1–4 of that same line do not exist yet, and their patches are the other half's. So the right half is blind
  to the start of its own line: column 6 sees letter 5 of its line only, 7 sees 5–6, 8 sees 5–7, in that order of
  loss. The left half reads its line's start as patches, and every earlier line of both halves through the B1s
  (one letter each, exact), so it loses nothing.
- **Column 5 is also worse** (1.81 at 60k against ~1.70), although its predictor (the last left patch) sees letters
  1–4 of its line and every earlier B1. Not explained by what it can see; perhaps the shared weights serving a right
  half that predicts without its context.
- **The split overfits less.** VS holds 1.86–1.88 from 40k to 100k where NB3 climbs to 1.93; at 100k it is the best
  of all runs. One seed: a hint, not a finding. A plausible reading: each half memorizes less of a page it sees only
  half of.

## Conclusions

1. Halves that share only summaries keep everything the summaries already hold: the left half, which needs only
   finished summaries and its own patches, predicts as well as the unsplit model.
2. They lose what no summary holds yet: the current line's other half. In this world a letter's summary completes one
   patch row after the right half needed it, so the right half predicts its letters without their line's start.
3. The cost of the split here is a property of the layout (letters two patch rows tall, summaries after the block's
   last patch), not of locality itself.

## Next Steps

- The same split where a letter is one patch row (vm-00046's NB2 layout, B1 tokens only; or the normal world with
  the pyramid): every summary is then complete right after its letter. Expected: both halves as the left one here.
- Horizontal halves (`attn_cols = 0`): the reading order crosses the split once per page, not once per line.
- A real compute saving (per-half attention kernels), once a split keeps quality.

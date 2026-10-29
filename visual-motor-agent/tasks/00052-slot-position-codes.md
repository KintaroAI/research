# Task 00052: Position Codes by the Slot Layout (experiment vm-00045)

**Date:** 2026-10-29
**Status:** In progress

## Context

- At birth, every patch slot's learned vector (`uni_slot`, noise of 0.02) gets the 2-D sine-cosine code of its cell
  added (strength 0.1), the universal model's only built-in geometry.
- `model_init_params` adds cell k's code to slot k + 1. That was right for the layout [body, patches, move], but since
  task 00047 summaries are interleaved, and slot k + 1 is no longer patch k:
  - in a vm-00042 EF checkpoint at birth (`uni_pyr = 3`, 87 slots), the first summary carries cell 10's code, 40 of
    64 patches carry another cell's code (shifted by the number of summaries before them), and the last 16 slots (the
    last patches, the global summary) carry none;
  - training does not repair it: at 100k the misplaced codes have grown (projection 1.00 → 1.48), and a patch's
    projection onto its own cell's code is 0.90, against 1.55 in EB (no summaries).
- So every arm with summary tokens (vm-00040's `uni_sum`, vm-00041–00044's `uni_pyr`) started with a scrambled
  position frame. Models without summaries (EB, SB, CB, the split and checker arms) are unaffected. It is a candidate
  cause of the finding of vm-00040 to vm-00044: every arm with summaries lost about half the food, even with full
  attention, while prediction (which neighbouring codes support) held.
- The owner's request (2026-10-29): fix it and run the training.

## Design

- **The code is placed by the slot table** (`M->uslot`, `vm_uni_layout`), in cell coordinates (row, column):
  - a patch: its cell (r, c), as before;
  - a pyramid summary at level ℓ, block (r, c): its block's centre, ((r + ½)·2^ℓ − ½, (c + ½)·2^ℓ − ½). B1(0, 0) sits
    at (0.5, 0.5), the global summary of the 8 × 8 grid at (3.5, 3.5);
  - a row summary (`uni_sum` R), band r: ((r + ½)·R − ½, (g − 1) / 2);
  - the body and the move: none, as before.
  - A summary then starts as a patch at its block's centre; its own learned vector tells the levels apart. The same
    rule in every world (text included).
- `vm_uni_center(c, s, &y, &x)` in `vm_agent.h` gives a slot's centre; `model_init_params` uses it. Models without
  summaries are initialized exactly as before (same draws, same codes).
- Only the initialization changes: no checkpoint field, the forward and backward passes are unchanged, and old
  checkpoints load and run as they were trained.

## Verification

- `tests/test_slot_positions.py`: for image and text configs without summaries, with `uni_sum` and with `uni_pyr`
  1–3, each slot's vector at birth projects onto its expected code (≈ 1) and onto no other cell's or block centre's
  code; the body and the move carry none.
- Without summaries, the parameters at birth are bit-identical to the previous build's (checked once against a
  scratch build of the old init: EB's config identical, EF's and TH's differ only in `uni_slot`).
- The full test suite.

## Results so far

- `tests/test_slot_positions.py` (21 cases, image 8 × 8 and 4 × 4, text, rows and pyramid levels 1–3): the old build
  fails all 17 cases with summaries, the new one passes. Full suite: 1,449 passed, 30 skipped.
- vm-00045's birth checkpoints: every slot's projection onto its own centre's code 0.93–1.00 (the old EF: −0.07 to
  1.05, mean 0.68).

## Experiment vm-00045

- The arms of vm-00042 that isolate the question, with the fix, 100k ticks, seeds 1–3:
  - **XF:** `configs/U_touch_ego.json`, `uni_pyr = 3`, full attention (against vm-00042's EF and vm-00036's EB).
  - **XFT:** the same with `sum_thumb = 1` (against EFT).
  - **XTP:** `configs/T_text_B.json`, `attn_win = 1`, `uni_pyr = 3` (against vm-00041's TP, 1.921 nats per letter).
  - **XB:** `configs/U_touch_ego.json` without summaries: vm-00036's EB on the current code. Added on 2026-10-29: the
    controls of vm-00043 and vm-00044 (SB, CB, the same model in slightly changed worlds) forage at about half of EB's
    rate in life, as much as the summary arms lose. Every summary arm since vm-00040 was compared with EB, run on
    vm-00036's code, so a change in the code since then would look like a cost of summaries.
- **Measures** as in vm-00042.
- **Question:** does food come back toward EB's (36.6 / 8.8 with the 1-step instinct, 44.4 / 4.1 with the 4-step
  pair lookahead) once the positions are right? If it does, vm-00040–00044's summary arms must be re-read.
- The analyses stopped for this (vm-00042's and vm-00044's 4-step tests, vm-00043's remaining runs) are re-run after
  or beside the lifetimes, at most two 4-step tests per GPU (~10 GB each).

# Task 00051: Poison With a 2 × 2 Checker, the Summary Experiment Repeated (experiment vm-00044)

**Date:** 2026-10-28
**Status:** In progress

## Context

- Poison is a checkered circle whose checker has 1-pixel cells. A box average over 2 × 2 pixels (a level-1 summary
  thumbnail, task 00049) turns it into a flat half-bright disc, so coarse views cannot tell it from food by texture.
- Square poison (task 00050, vm-00043) kept the circles' 1-pixel checker and changed the shape. It made the task
  harder for every arm, including full attention without summaries (half the food in life).
- The owner's request (2026-10-28): poison back to a circle, with the checker's cells 2 × 2 pixels. A 2 × 2 box
  average then keeps a 1 × 1 checker (blurred only at the edge), so a level-1 thumbnail still shows the texture.
  Repeat the experiment.

## Design

- **`env.hazard_checker_cell = 2`** (`world/circle_world.py`; default 1, the pixel checker, unchanged): poison pixels
  lit where (row // 2 + column // 2) is even, in view coordinates. The cells are aligned with the patches and with
  the 2 × 2 boxes of a level-1 thumbnail, so each box falls inside one cell. Level 2 (4 × 4 boxes) and level 3
  (8 × 8) still average the checker to half brightness. Poison is drawn and touched as a circle, as before.
- **Config `configs/U_touch_ego_c2.json`:** `U_touch_ego.json` with `hazard_checker_cell = 2`.

## Verification

- `tests/test_world_ext.py`: the drawn poison equals the disc masked by the checker of the given cell size (1 and
  2); inside the disc, the 2 × 2 box averages of the 2-pixel checker are an exact 1 × 1 checker; invalid cell sizes
  are refused.
- A rendered frame and its 2 × 2 box averages: the checker survives the average (checked by eye).

## Experiment vm-00044

- vm-00042's image arms in the 2-pixel-checker world, 100k ticks, seeds 1–3, the 1-step instinct:
  - **CB:** full attention, no summaries (the control in this world).
  - **CF:** `uni_pyr = 3`, full attention.
  - **CFT:** `uni_pyr = 3`, `sum_thumb = 1`, full attention.
  - **CPT:** `attn_win = 1`, `uni_pyr = 3`, `sum_thumb = 1`.
- **Measures** as in vm-00042. Compared with the circle world (EB, EF, EFT, EPT) and the square world (SB, SF, SFT,
  SPT).
- **Questions:**
  - Does a texture that survives the level-1 average make thumbnails useful (CFT against CF)?
  - Does a coarser texture change the control (CB against EB)?
- Started when vm-00043's first runs free the 3090, so that the GPUs are not overloaded.

# Task 00050: Square Poison, the Summary Experiment Repeated (experiment vm-00043)

**Date:** 2026-10-28
**Status:** In progress

## Context

- In the touch worlds, food is a solid grey circle and poison a checkered grey circle (task 00024). They differ only
  in texture, a pixel-level pattern. A box average removes it (vm-00042's thumbnails see poison as a dimmer disc), and
  so, plausibly, does any coarse summary.
- In vm-00040 to vm-00042, every arm with summary tokens lost about half the food against full attention without
  them, and poison avoidance varied.
- The owner's request (2026-10-28): make poison a checkered square and repeat the last experiment. Poison and food
  then differ in shape too, which survives downsampling.

## Design

- **`env.hazard_shape = "square"`** (`world/circle_world.py`; default `"circle"`, unchanged). A hazard is an
  axis-aligned square with half-side r·√π/2, the same area as the circle of radius r it replaces. It is drawn (still
  checkered with `hazard_pattern = checker`) and touched as a square: contact when |dx| and |dy| are both within the
  half-side, on the torus. Respawns, motion and poison effects are unchanged.
- **Config `configs/U_touch_ego_sq.json`:** `U_touch_ego.json` with `hazard_shape = square`.
- The browser replica does not have the option (the equivalence checks run on the default circles).

## Verification

- `tests/test_world_ext.py`: a point in a square's corner (outside the equal-area disc) touches the square, not the
  circle; a point on the axis between the half-side and r touches the circle, not the square. The drawn pixels equal
  the square (or disc) masked by the checker. The areas are equal. Both on the plain and the wrapping world.

## Experiment vm-00043

- vm-00042's image arms in the square world, 100k ticks, seeds 1–3, the 1-step instinct:
  - **SB:** full attention, no summaries (the control, as vm-00036's EB).
  - **SF:** `uni_pyr = 3`, full attention.
  - **SFT:** `uni_pyr = 3`, `sum_thumb = 1`, full attention.
  - **SPT:** `attn_win = 1`, `uni_pyr = 3`, `sum_thumb = 1`.
- **Measures** as in vm-00042: the frozen forage tests, held-out prediction, the dream. Each arm is compared with its
  circle-world counterpart (EB, EF, EFT, EPT).
- **Questions:**
  - Does the summaries' cost in food shrink when poison is distinguishable at a coarse scale?
  - Do thumbnails help once they can show the difference?
  - Does the control itself change, with poison easier to tell apart?
- Started after vm-00042's runs finish, so that both GPUs are free (12 lifetimes).

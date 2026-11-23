# vm-00106: The Blind Spot in the Targets Too — Can Motion Teach the Model to Fill It?

**Date:** 2026-11-23
**Status:** Running (since 2026-11-23 05:13)
**Source:** `dev/` (pending tag `exp/vm-00106`)
**Outputs:** `~/data/research/visual-motor-agent/exp_00106/`
**Task plan:** [`../tasks/00118-blind-spot.md`](../tasks/00118-blind-spot.md)

## Goal

The owner (2026-11-22): "second experiment - we need to figure out how we can train on a image with blind spot as well but teach model to restore image back nevertheless".

In vm-00105 the model saw a gap but learned against the whole frame. The eye never gets the whole frame: the gap is in everything it learns from. Here the gap is in the targets too.

- **The null (G0):** with the gap black in the targets, the model learns to predict black there.
- **The teacher (GB):** the view moves with the body, so the scene slides under the fixed gap. What the gap hides now was in plain view a moment before. Can a target built only from what was seen elsewhere, at other moments, teach the model to fill the gap?

## Method

- **The agent and the gap:** vm-00105's: the default agent (`configs/U_agent.json`) with `online.blind = [38, 16, 12, 12]`, the same for all four transformers.
- **The arms**, 12 seeds each, 60k ticks:
  - **G0** (`online.blind_target = black`): the targets have the gap as the model sees it, black.
  - **GB** (`online.blind_target = motion`): each step's target is the frame as seen, with its gap filled by `GapTeacher`.
    - The teacher keeps a canvas of the world (128 × 128, wrap-around) per body, painted only from the visible part of each frame. The gap's pixels are never read.
    - The gap of each target is read from the canvas at the gap's world pixels: the last grey seen there. Pixels never seen take the background.
    - The body's position comes from the world (`env.cursor`). It is a training signal only, never an input.
    - It's built when the step is stored, so it uses only the past.
- **The teacher is imperfect.** Objects are drawn at sub-pixel positions, the poison's checker is fixed to the screen, and 32 grey dots move on their own. On random walks (before the runs) the teacher's gap is off by 3.7 grey levels against the background guess's 4.8; with an object in the gap, 8.0 against 11.5. In the life it is logged every 1k ticks (`teach_err`, `teach_filled`).
- **The cards:** seeds 1–4 of both arms on the RTX 3090, 5–12 on the RTX 4090, under MPS.
- **Comparisons, not rerun:** vm-00105's BL (the gap in the input, whole targets: the most the gap could be learned) and TT (no gap).
- **Not run: the masked loss** (no target in the gap at all). The CUDA loss has no per-value mask (`set_step_mask` masks whole steps and their keys); it's left for later.
- **New code:**
  - `online.blind_target`, `GapTeacher` and the memory field `timg` (`online.py`, `lifetime.py`);
  - tests in `tests/test_blind.py`.

## Measures

`eval_blind.py`, as vm-00105, on frozen lives with the gap in (`make frozen`: 20k, 40k, 60k, seed 100 + s; the same test worlds as vm-00105's):

- the gap's error, against the background baseline;
- objects in the gap and by ticks since entry;
- seen or not (hits, false alarms, d′);
- the surround;
- lived food and poison.

## Readings, set before the results

1. **Plumbing:** the tests pass; all 24 lives run 60k ticks.
2. **The teacher teaches:** GB_gap's gap error at 60k below G0_gap's, paired by seed.
3. **Better than a guess:** GB_gap's gap error at 60k below the background baseline's, paired by seed.
4. **How much of whole-frame learning it recovers (descriptive):** GB_gap against vm-00105's BL_gap and the teacher's own error.
5. **The cost (descriptive):** lived food and poison of G0 and GB against BL and TT.

## Results

Not started.

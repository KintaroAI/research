# vm-00080: Delayed Copies

**Date:** 2026-11-03
**Status:** Running: 18 lives since 18:55
**Source:** `dev/` (not yet tagged)
**Outputs:** `~/data/research/visual-motor-agent/exp_00080/`
**Task plan:** [`../tasks/00091-delayed-copies.md`](../tasks/00091-delayed-copies.md)

## Goal

The owner's idea: build the view so that it helps the model predict. The frame is four copies of the view: now, and 1 to 16 ticks ago. Each copy has its own four transformers, and every one of them is trained to output the present. Do the delayed transformers learn to dead-reckon, and does the body forage better for seeing the past? The design, arms and readings set before the results are in the task plan.

## How it works

| Quadrant of the 128 x 128 frame | Shows | Transformers | Target |
|---|---|---|---|
| Top-left | now | 4 | now |
| Top-right | d1 ticks ago | 4 | now |
| Bottom-left | d2 ticks ago | 4 | now |
| Bottom-right | d3 ticks ago | 4 | now |

- Each transformer is a default quadrant transformer in size: 4 x 4 patches of 8 pixels, the same network. It holds 36 to 48 tokens per step, against 32 in the default model, because the pyramid has one more level.
- Every transformer's body token is the present body, and all 16 vote by the gain rule.
- In training, the stored composite is the input and the present, tiled four times, is the target: the same split as damage and input noise (`update_rollout(inputs=...)`).
- Before tick d, the first frame stands in for the missing past.

## Method

- **Arms:** D0 (delays 0, 0, 0, 0), D1 (0, 1, 2, 3), DX (0, 1, 4, 16); K4, log summaries, the default world, six seeds, 60k ticks; seeds 1–4 of each arm on the 4090 and 5–6 on the 3090, all 18 at once. Nine per card at first gave 4.8 and 2.6 ticks/s, so the split was changed after 6 minutes, the first start discarded, for the cards to finish together.
- **Reference:** vm-00073's K4, one present frame and 4 transformers, not rerun.
- **Frozen tests (`eval_delay.py`):** `error`, the prediction of the present by copy against its stale input; `chain`, foraging as seen and with the delayed copies blanked.
- **Checks passed before the start:** `tests/test_delay.py`:
  - the composite's quarters are the frames 0, 1, 4 and 16 ticks back over 40 ticks, against a plain world in lockstep;
  - drives, touches and cursor are the plain world's;
  - the targets are the present tiled;
  - bad delays are refused;
  - a short life with 16 transformers learns in every one.
  The world tests are unchanged.

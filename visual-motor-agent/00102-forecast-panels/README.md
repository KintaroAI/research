# vm-00102: Forecast Panels — the Model's Own Forecasts Fed Back Beside the Real Frame

**Date:** 2026-11-21
**Status:** Running
**Source:** `dev/` (pending tag `exp/vm-00102`)
**Outputs:** `~/data/research/visual-motor-agent/exp_00102/`
**Task plan:** [`../tasks/00115-prediction-panels.md`](../tasks/00115-prediction-panels.md)

## Goal

The body builds the frame the model sees from four panels, one per quarter transformer:
- panel 1 is the real view;
- panels 2 to 4 are transformers 0 to 2's forecasts of their own panel from the tick before, slightly noised;
- panel i is trained against the real frame i ticks after panel 1's, so the model learns to look 1 to 4 ticks ahead, each panel reading the forecast before it.

Do the forecasts help it predict and forage, against the same agent with noise in panels 2 to 4? The overnight pilot (task 00115) showed the forecasts are learned and used, but not whether they help. This is the first run with enough seeds to tell.

## Method

| Arm | Panels 2–4 | Seeds |
|---|---|---|
| PB | random grey noise, drawn anew each tick (the baseline) | 1–5 |
| PN | the forecasts (pair decoding) with Gaussian noise σ .03 | 1–5 |

- **Both arms:**
  - `configs/U_view.json`, 4 quarter transformers of 8 layers, `model.img = 128` (each transformer's quarter is one 64 × 64 panel), tolerance .002, context 8, the ring;
  - RD's label (vm-00100: `model.body_label = 1`, `online.label = meal`, `label_gamma` .95, `full_dose`, `label_refresh`), inertia 4, the candidates at temperature 2 (vm-00101);
  - none of these is a default; this run uses them as the best agent so far (the owner).
- **The owner's choices (2026-11-21):**
  - σ .03: at the pilot's .1 the forecasts were hidden under the noise;
  - no lag arm, so the run asks one question;
  - memory unchanged (100k steps), though it replays stale forecasts: "we need to see if it learn new stuff even with frozen predictions".
- **Card:** the RTX 3090 alone, 10 lives at once under MPS (about 2.2 GB each). The pilot's 8 lives ran 0.8 ticks/s each, so 60k ticks take about a day.
- **Controls linked:** TT (vm-00101, 64 × 64, the same settings without panels), seeds 1–5.

## Measures

- **Lived** by 5k and at 40–60k: food, poison, mean hunger, overrides; the prediction loss per panel in training batches (`uni_ce_patch_pK`).
- **Frozen** at 10k to 60k (`make frozen`): each life 600 ticks with learning off in new worlds, the prediction loss per panel; PN also with its forecasts replaced by noise.
- **A video** (`make video`, `dev/panel_video.py`): the panels as the model sees them beside the real frames the forecasts aim at.

## Readings, set before the results

1. **Plumbing:** all 10 lives run 60k ticks.
2. **The forecasts help prediction:** frozen, PN − PB loss on panels 2–4 at 40k–60k, paired by seed, below 0.
3. **They are used:** frozen, PN with its forecasts replaced by noise predicts worse than PN as it lived.
4. **Foraging (descriptive with 5 seeds):** PN − PB food at 40–60k, paired by seed; both against TT.

## Results

Running since 2026-11-21.

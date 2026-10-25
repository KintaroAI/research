# vm-00017: Poison Circles: Does the Prediction Get Both Kinds of Click Right?

**Date:** 2026-10-25
**Status:** Complete
**Source:** `exp/vm-00017` (code: `visual-motor-agent/dev/`)
**Outputs:** `~/data/research/visual-motor-agent/exp_00017/`
**Task plan:** [`../tasks/00021-poison-click.md`](../tasks/00021-poison-click.md)

## Goal

Check the prediction itself, not behavior. Hazard contact (vm-00011–16) was too small an effect to learn; here a click
inside a hazard sets hunger to 1 (poison), as large an effect as eating the red circle (food). Does each kind of model
predict what a food click, a poison click and a click on nothing do to its hunger and to the frame?

## Hypothesis

1. Every model's drives head learns all three click outcomes: next hunger about 0 after a food click, 1 after a poison
   click, unchanged (+0.001) after a click on nothing or no click.
2. The frame prediction shows the eaten circle vanishing, and nothing changing on a poison click.
3. The planner's latent hunger head, which reads features trained by frame prediction only, learns the food click
   (vm-00002: it did) but not the poison click, unless the features tell the two kinds of circle apart.

## Method

- **World** (`env.hazard_click=1`, task 00021): the continual torus with 2 hazards (blue, drifting), slow hunger
  (0.001 per step), no contact cost; a poison click sets hunger to 1, the hazard stays; inside both, the poison wins.
- **Arms** (seeds 1–3, on both GPUs): PD (DMI, the CNN basic agent, 60k), PP (8×8 patch tokens, move loss detached,
  60k), PL (the planner agent L, 40k).
- **Prediction test** (`analyze_hazards.py click`): 1,536 placed states with hunger 0.3 (a third in the circle, a third
  in a hazard, a third on the background); for "stay and click" and "stay, no click": the drives head's next hunger, the
  predicted redness (R − B) of the circle's pixels, and (PL) the latent hunger head over one imagined step, each against
  the world's. Every 10k steps.

## Log

### 2026-10-25
- **Implemented** `env.hazard_click` (tests: a poison click sets hunger, food still relieves, the poison wins inside
  both, off by default) and `analyze_hazards.py click`. A 1,000-step smoke run of each arm: the test's world truths are
  right (next hunger 0.000 / 1.000 / 0.301; the circle's redness 180 → −3 when eaten); the models do not yet predict
  them.
- **Launched** 14:05 (PD on the RTX 3090, PP on the RTX 4090, PL split); all finished by 14:14 (110–150 steps/s);
  the analysis ran 14:14–14:24.

## Results

### R1. Click outcomes: predicted next hunger, "click" minus "stay", from the same placed state (seeds 1, 2, 3)

| Arm | food (world −0.301) | poison (world +0.699) | background (world 0.000) |
|---|---|---|---|
| PD (CNN), 60k | −0.040, +0.008, +0.007 | −0.010, +0.016, +0.020 | −0.012, +0.016, +0.021 |
| PP (8×8 patches), 60k | **−0.172, −0.320, −0.116** | **+0.167, +0.066, +0.209** | −0.113, −0.161, −0.064 |
| PL (planner): drives head, 40k | −0.015, +0.007, +0.011 | −0.012, +0.006, +0.008 | −0.010, +0.005, +0.007 |
| PL: latent hunger, imagined | −0.179, −0.053, −0.123 | −0.163, −0.047, −0.121 | −0.165, −0.057, −0.116 |

PP over training (poison click, absolute predicted next hunger; world 1.0): 10k 0.25–0.27, 20k 0.17–0.19, 40k
0.16, 0.20, 0.44, 60k 0.50, 0.41, 0.50. Without a click every model predicts the steady rise well (0.28–0.38 against
0.301).

### R2. The frame: the circle's predicted redness (R − B) on its pixels, at the end

| Arm | food click (world −3) | stay (world 180) |
|---|---|---|
| PD | 98, 106, 96 | 170, 174, 170 |
| PP | 18, −7, −14 | 156, 160, 156 |
| PL | 83, 167, 167 | 159, 169, 166 |

### R3. Online (last 10k steps) and the frozen tests

Hits per 1,000 steps PD 7.2, 6.0, 6.3; PP 6.4, 7.2, 10.6; PL 6.6, 9.3, 7.9 (vm-00015's SL without poison: 17–18).
Hunger at the cap of 1: PD 0.23–0.34, PP 0.33–0.57, PL 0.40–0.53 of steps. Episodic test at 60k: PD 0.86, 0.71, 0.76;
PP 0.84, 0.84, 0.87.

## Analysis

- **The patch model's prediction works for food, partly for poison** (hypotheses 1 and 2, partly supported). After a
  food click its drives head predicts most of the drop (0.12–0.32 of 0.30), and its frame prediction erases the eaten
  circle almost exactly (redness −14 to 18 against −3). The poison click is learned slowly and only in part (+0.07 to
  +0.21 of +0.70, still rising at 60k). It over-generalizes the click: a click on the background also predicts relief
  (−0.06 to −0.16), so its drives head has learned "a click lowers hunger" plus a correction for where.
- **The CNN agent's drives head learns no click outcome**, and its frame shows the eaten circle only half gone. Its
  one fused token per step must carry the cursor-to-circle relation through the Transformer to the drives head; in
  the patch model the token under the cursor holds it directly.
- **The planner's imagination learned "clicking relieves", not where** (hypothesis 3, worse than expected): its latent
  hunger head lowers the imagined hunger for any click by the same amount, food, poison or background. Its features,
  trained by frame prediction only, do not separate a click on the circle from a click elsewhere well enough for the
  hunger head. This is vm-00002's edge-click exploit in another form, and it makes the planner click everywhere (its
  hits fell from 17–18 to 7–9 per 1,000 steps).
- **The cap returns:** a poison click leaves hunger at 1 until the next hit, so hunger sat at the cap on 23–57% of
  steps, where a further poison click changes nothing. The poison signal is rarer than the clicks.

## Conclusions

1. **The 8×8 patch model predicts the consequences of its clicks**: food relieves hunger and the circle vanishes in its
   predicted frame; poison raises hunger, partly and late; a click on nothing is partly confused with food.
2. **The CNN model and the planner do not**: the CNN's drives head learns no click outcome, and the planner's
   imagined hunger falls for every click alike.
3. **Poison that sets hunger to 1 brings back the hunger cap** and starves its own signal.

## Next Steps

- Poison that adds to hunger (e.g. +0.5) instead of setting it to 1, so it never parks hunger at the cap.
- Longer lifetimes for the patch model, whose poison prediction was still rising at 60k.
- For the planner: let its hunger loss reach the features (`dyn_h_backbone_scale`), so that what is under the cursor
  enters its latent state.


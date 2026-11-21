# Task 00115: Forecast Panels — the Model's Own Noised Forecasts Fed Back Beside the Real Frame (to explore later)

**Date:** 2026-11-20
**Status:** Proposed, for later. The owner, 2026-11-20, during vm-00101: "there are 2 another ideas for later". Clarified the same evening:

> imagine you have a "delay" mechanism in a body - so tick 1 - F1 is "live", there is no F2 and F3 or it's just a noise (first 3 ticks of life), on a second tick F1 = F1(2), F2 = F1(1), F3=noise, tick 3 F1 = F1(3), F2 = F1(2), F3 = F2(1) ... F1(t) = F1(t), F2(t) = F1(t-1), F3(t) = F2(t-1) or F1(t-2) so basically imagine like in neurones - first layer has delay so second layer sees some sort of outdated image (one spike away from layer 2)

The noise idea, in the owner's words:

> F0->P1->O1(trained on F1, add noise to O1 - get NO1)->NO1->P2->O2(trained against F2)->NO2->P3->O3(trained against F3) - so it's naturally trains to predict 2 steps into the future and the whole model sees few frames at once. Adding noise will allow prediction to remain composed and not blur over time. Then we will be able to use this sort of pipeline to dream about future and predict the outcomes better for possible moves

No experiment number yet. The design below was settled with the owner on 2026-11-20.

## Context

- **What the model does now:**
  - each tick is a step of tokens: body, the view's patches (and summaries), move;
  - each patch token predicts the next frame's patch, one tick ahead;
  - the context holds the last 8 steps, all real.
- **Looking further ahead loses the picture:**
  - **vm-00023:** imagined frames (Jacobi decoding of the model's own predictions) lose the moving cursor within a few steps. A 4-step lookahead ranked moves no better than 1 step on them (correlation .10), though the true 4-step future ranked them twice as well (.23).
  - **vm-00096:** in the world with inertia, holding each candidate for 4 imagined ticks still gave 47% more food than the one-move instinct, at 2.4 times the cost per tick.
  - **vm-00032 (self-forcing):** training on the model's own predictions as inputs kept the dream's cursor 2–4 times closer to the truth over 16 steps.
  - **vm-00080 (delayed copies):** a frame made of the view now and the view d ticks ago works, but seeing the past did not change foraging. The context already holds the past.
- **Blur:**
  - a predicted value is a distribution over 32 bins;
  - its mean averages the futures the model is unsure of, so chained means blur.

## The design: the panels are a body property, the model is unchanged (the owner, 2026-11-20)

> we need no special treatment for panels. Imagine we currently have 64x64 field. now imagine we create 128x128 field where 3 quarters are these panels - they appear "at ones" for the model on each tick and 4th quarter is empty ... so we store 128x128 image and construct the "original" reference ... we will be training against. So it's ok for model to have 8 frames and it will take it 8 ticks to proccess. All "panels" stuff is a virtual thing and more like a body/world property which forms 128x128 image

Earlier versions of this task chained the panels within a tick, which needed an order of computing. They are superseded: here the body fills the panels before the tick, from the model's outputs of the tick before, as it feeds back the move.

- **The frame is 128 × 128, four 64 × 64 panels, one per quarter transformer** (the owner: "it should be 4 transformers - let it be 4 panels as well so we do not waste one quarter"). It is built by the body, a world wrapper as `env.delays` built vm-00080's 2 × 2 composite.
- **With the lag L, write R = t − L** for the newest moment sensed. At tick t:

| Panel (transformer) | Shows (filled by the body before the tick) | Trained against (built from stored real frames) |
|---|---|---|
| 1 | F(R), the real frame | F(R + 1) |
| 2 | N(O1 of tick t − 1): last tick's forecast of F(R), noised | F(R + 2) |
| 3 | N(O2 of tick t − 1): last tick's forecast of F(R + 1), noised | F(R + 3) |
| 4 | N(O3 of tick t − 1): last tick's forecast of F(R + 2), noised | F(R + 4) |

- **In general:** panel i is trained against F(R + i), and panel i + 1 shows panel i's output of the tick before, noised.
- **What the model sees each tick:**
  - the real frame;
  - its own expectation of that same moment, made a tick earlier: its prediction error is in view, as in predictive coding;
  - its expectations of the two ticks after.
- **What it learns:** 1, 2, 3 and 4 ticks ahead of the real frame, every tick, against reality. Panels 2 to 4 train it to work from its own noised forecasts.
- **No copying:** every target is newer than F(R), the newest real frame anywhere in the input, its context included.
- **Nothing special in the model:**
  - one pass per tick;
  - the context of 8 ticks as now;
  - every token may see every other, as now.
- **The panels are fed back like the move:**
  - the move is the model's output, and the body feeds it back as the next tick's input;
  - replay does not recompute past moves, and it does not recompute past panels either;
  - memory stores the 128 × 128 frame the agent saw.
- **The targets are built at training time** (as `online.delay_targets` builds vm-00080's): panel i's target is the real frame i ticks after F(R). That is panel 1 of the step i ticks later.
- **Memory:**
  - long-term memory keeps 4 steps past each chunk (`ChunkStore(extra=n)`, as for `value_n`);
  - the newest trainable window ends 4 ticks back;
  - the lag changes only what the body puts in panel 1, so memory is the same for any L.
- **Stale panels are fine:** they were made by an older model, but they are what the agent saw, like its moves, and the targets are real. vm-00099's problem was stale targets.
- **The lag is a body setting; L = 2 (the owner's) and L = 0 are the two arms:**
  - the frame and the body values reach the model L ticks late;
  - moves act at once (an efference copy).
- **Acting:** each tick the model's predictions for panels 1 to 3 are decoded and noised by the body to fill next tick's panels 2 to 4: a readout of the patch predictions it computes anyway.
- **The first ticks of life:** panels not yet filled are noise.
- **The quarter transformers:**
  - each of the 4 has one panel, and the summaries of all;
  - transformer 1 forecasts from reality, and transformers 2 to 4 from the forecasts, further out.
- **The body's target per transformer (to decide):**
  - all predict the next body (hunger, label) as now;
  - or each predicts the body at its own horizon, R + i. Then the move rule (`gain`) would compare labels at different horizons.
- **Moves and candidates:**
  - the step's move is the move made now, as today;
  - panels 2 to 4 predict 2 to 4 ticks after the real frame, so a candidate's effect shows in what the model predicts there;
  - the instinct can score candidates by that prediction and its label, as it scores the next body today.
- **Dreaming:** fill panel 1 from O1 as well, and the model runs on its own forecasts, which it has seen in panels 2 to 4 on every tick of its life.

## Noise

- **Why:**
  - a panel that always sees a slightly wrong input learns to clean it up and commit to a sharp frame, instead of averaging and passing the blur on;
  - the same trick keeps learned game engines from drifting over long rollouts: GameNGen (Valevski 2024) adds Gaussian noise to its context frames in training, and DIAMOND does the same.
- **Kinds of noise to try:**
  - **Gaussian** on the values, σ fixed or drawn per tick;
  - **sampled:** each value drawn from the model's own predicted bins instead of its mean. The noise is then the model's own uncertainty, sharp by construction;
  - **pixel noise** as `online.noise_in` (vm-00077).

## What it is for

- **Dreaming that stays composed:**
  - the model is trained every tick on inputs that are its own noised predictions, so a rollout that feeds its outputs back is what it has always seen;
  - to dream, the output goes back in as the next input, as in vm-00023, but now into a model trained for it.
- **Scoring moves further out:** each candidate's forecast and label read from panel 3 within the tick, as vm-00096's horizon (which helped under inertia), on a chain trained against reality.
- **The label's horizon:** vm-00100's label is right within about 10 ticks of food and flat beyond. A dream that stays sharp for a few ticks lets the label be read further out.
- **The frame-level form of task 00114 Part A:** imagined outcomes of moves not made.

## Stages

1. **Panels 2 to 4 empty:** the 128 × 128 frame with only the real panel, and the four horizons' targets. This gives the cost, and the baseline for each horizon without the model's own forecasts.
2. **Forecast panels without noise**, L = 0 and L = 2.
3. **With noise:** Gaussian, then sampled.
4. **Dreaming and acting:** the dream curve (`eval_dream.py`), then the instinct scoring candidates by their three-tick forecast, in the world with inertia, against RD (vm-00100) and vm-00096's IH.

Stages 1–3 can be compared frozen before any lifetime.

## Measures

- **Each panel against its target,** on held-out real windows: nats per value at 1, 2 and 3 ticks ahead, against a plain Jacobi chain (vm-00023) and the teacher-forced baseline.
- **The dream:** O3 fed back into panel 1, over 3–16 ticks (`eval_dream.py`: the cursor's distance, or lost).
- **Ranking moves:** a candidate's forecast (panel 3, its label) against its real outcome on copies of the world (vm-00023's test).
- **Sharpness:** the cursor in the dream (`eval_dream.py`: distance to the real cursor, or lost), and the predicted bins' entropy.
- **Lived (stage 4):** as vm-00100 (food, poison, hunger, overrides, smoothness, the label monitors).
- **Cost:**
  - a 128 × 128 frame has 4 times the patch tokens of 64 × 64, so attention costs up to 16 times per step (less with the quarter pieces, each on one panel); one pass per tick;
  - with 4 quarter transformers it may need smaller panels (32 × 32) or fewer lives per card;
  - memory keeps K − 1 more frames per step.

## Related work

- **GameNGen** (Valevski 2024) and **DIAMOND** (Alonso 2024): noise on context frames against autoregressive drift.
- **Scheduled sampling** (Bengio 2015): feeding the model its own predictions in training.
- **PlaNet's overshooting** (Hafner 2019): a latent model trained on predictions several steps ahead.
- **Predictive coding:** cortical layers predicting delayed signals from the layer below (Rao & Ballard 1999), the owner's neuron picture.

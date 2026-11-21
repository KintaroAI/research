# Task 00115: A Chain of Forecasts Laid Out as Panels, with Noise (to explore later)

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

## The design: a chain of forecasts laid out as panels (the owner, 2026-11-20)

Its history in one evening:
1. **Layers handing their outputs on with a one-tick delay,** trained against [F(t + 1)][F(t)][F(t − 1)]. Their past targets were in view (panel 2's F(t) in panel 1), so panels 2 and 3 would have copied them.
2. **The owner's fix:**

   > what if we keep everything the same but we will make them to predict future frames and not the past ones? so layer 1 sees F(t-2) -> O1 (F(t-1)) -> O2 (F(t)) -> O3 (F(t+1)) so it predicts into future but with some lag so model sees only F(t-2) but trained on [F(t-1)][F(t)][F(t+1)]

| Panel | Sees | Output | Trained against |
|---|---|---|---|
| 1 | the newest real frame, F(t − L) | O1 | F(t − L + 1), real |
| 2 | N(O1), noised | O2 | F(t − L + 2), real |
| 3 | N(O2), noised | O3 | F(t − L + 3), real |

- **L is the lag** (the owner's L = 2: panel 1 sees F(t − 2), the targets are F(t − 1), F(t), F(t + 1)).
- **No copying, so no mask against it** (the owner: "we do not need to mask anything since it's predicts future which is not yet accessible"): the newest real frame anywhere in the input is F(t − L), and every target is newer.
- **Within one tick, in order:**
  - each panel steps one tick ahead of the one before, so panel 2 needs this tick's O1, not last tick's;
  - (handed over a tick late, the clock would move with the wave and all three panels would land on the same moment);
  - with earlier panels' keys and values kept, the three ordered pieces are about one pass of work.
- **The one rule is the order of computing:**
  - panel k sees panels 1 to k, at this tick and every earlier one, plus the body and move tokens;
  - it does not see the higher panels, which are made from its output;
  - this is not about cheating. It keeps a replayed window to three pieces: all of panel 1 over the 8 steps, then all of panel 2, then panel 3.
  - If panel 1 also read last tick's panels 2 and 3, each step would wait for the one before: 24 pieces per window.
  - Nothing is lost: everything in panels 2 and 3 is made from real frames that panel 1 already sees in its context.
  - Acting uses the same rule, so acting and training stay equal.
- **The quarter transformers:** each of the 4 sees its quarter of every panel, and the summaries of all panels.
- **Moves (an efference copy):**
  - the agent knows its own moves without delay;
  - each panel's forecast gets the move that leads to it: a(t − 2) and a(t − 1) are known, and the forecast of F(t + 1) needs a(t), the move being chosen now;
  - so the instinct can score each candidate by panel 3's forecast and its body (hunger, label): a lookahead within the tick.
- **The body (hunger, label) is delayed with the frame,** so no sense is fresher than F(t − L).
- **The lag is one setting, and both values are arms** (the owner: "make delay as a setting ... I'm totally for checking out both arms"):
  - **L = 2 (the owner's):** the agent senses the world two ticks late and acts on its forecast of now, like a nervous system with conduction delays;
  - **L = 0:** panel 1 sees F(t), and the chain forecasts F(t + 1), F(t + 2), F(t + 3). It is the same chain shifted by two ticks, and the agent senses the present;
  - the frame and the body reach the model L ticks late; moves act at once.
- **Memory, one rule for any lag:**
  - memory keeps real frames in world time, as now;
  - a step at tick t takes its input from frame t − L and its targets from frames t − L + 1 to t − L + 3;
  - so a stored chunk needs L frames before it and 3 − L after it: L = 2 needs 2 before and 1 after (the one after as now), L = 0 needs 3 after;
  - the "after" margin exists already (`ChunkStore(extra=n)`: long-term memory keeps n steps past each chunk for `value_n` and `label_refresh`); the "before" margin is the new part;
  - the newest trainable window ends 3 − L ticks back;
  - a replayed window recomputes the chain with the current model, so nothing stale is stored.
- **The first ticks of life:** panel 1 is noise until a real frame L ticks old exists.
- **Dreaming:** the chain is a dream three ticks long, trained against reality every tick. A longer dream feeds O3 back into panel 1, on a model that has always seen its own noised forecasts.

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

1. **Panels 2 and 3 fed the real frames** (teacher-forced, no chain): the cost of a 3-wide frame, and the baseline for each horizon.
2. **The chain without noise**, L = 0 and L = 2.
3. **With noise:** Gaussian, then sampled.
4. **Dreaming and acting:** the dream curve (`eval_dream.py`), then the instinct reading a few dreamed ticks per candidate, in the world with inertia, against RD (vm-00100) and vm-00096's IH.

Stages 1–3 can be compared frozen before any lifetime.

## Measures

- **Each panel against its target,** on held-out real windows: nats per value at 1, 2 and 3 ticks ahead, against a plain Jacobi chain (vm-00023) and the teacher-forced baseline.
- **The dream:** O3 fed back into panel 1, over 3–16 ticks (`eval_dream.py`: the cursor's distance, or lost).
- **Ranking moves:** a candidate's forecast (panel 3, its label) against its real outcome on copies of the world (vm-00023's test).
- **Sharpness:** the cursor in the dream (`eval_dream.py`: distance to the real cursor, or lost), and the predicted bins' entropy.
- **Lived (stage 4):** as vm-00100 (food, poison, hunger, overrides, smoothness, the label monitors).
- **Cost:**
  - 3 panels make 3 times the patch tokens, so attention costs up to K² times per step; three ordered pieces per tick, about one pass of work;
  - with 4 quarter transformers it may need smaller panels (32 × 32) or fewer lives per card;
  - memory keeps K − 1 more frames per step.

## Related work

- **GameNGen** (Valevski 2024) and **DIAMOND** (Alonso 2024): noise on context frames against autoregressive drift.
- **Scheduled sampling** (Bengio 2015): feeding the model its own predictions in training.
- **PlaNet's overshooting** (Hafner 2019): a latent model trained on predictions several steps ahead.
- **Predictive coding:** cortical layers predicting delayed signals from the layer below (Rao & Ballard 1999), the owner's neuron picture.

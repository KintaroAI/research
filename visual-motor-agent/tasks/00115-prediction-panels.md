# Task 00115: Forecast Panels — the Model's Own Noised Forecasts Fed Back Beside the Real Frame (to explore later)

**Date:** 2026-11-20
**Status:** In progress: the full run vm-00102 (from 2026-11-21, after the overnight proof of concept below) (the owner, 2026-11-20, late: "you do panels (#5) - run so it finishes by around 9 am. The goal is to see numbers and test if poc works - use re-labeled label and temperature that makes sense out of #1 but dont do those as default settings yet ... do it in small iterations ... so we fail fast"). Proposed the same evening, during vm-00101: "there are 2 another ideas for later". Clarified the same evening:

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
- **The body, as now** (the owner, 2026-11-20):
  - every transformer predicts the next sensed body (hunger, label) from what it sees, trained against the real body at R + 1;
  - no special treatment, as if the model did not know there are delays;
  - the move rule (`gain`) and the instinct are unchanged.
- **Later, not in this task** (the owner): choosing moves by their projected outcome further out:
  - an additional model that learns on chains of commands and the projected states (the panels' horizon) to choose the best ones;
  - or simply the horizon read with a heuristic.
  - "For now let's keep it simple."
- **Stale forecasts in memory (found in the pilot; the owner, 2026-11-21):**
  - **The problem:**
    - memory stores the composite as it was shown, so a replayed window's panels 2 to 4 are the forecasts of the model as it was then, not of the current one;
    - the model trains partly on old, worse forecasts. The owner expects this to slow learning, not to break it.
  - **The plan for now: no recalculation.** The owner sees memory as a sliding window of more or less recent events, so the staleness stays bounded.
    - As built, long-term memory holds up to 100k steps (`online.replay`, the oldest evicted first). In a 60k life that is the whole life, so the oldest forecasts are as old as the life.
    - A smaller `replay`, or `ltm_evict` choices, bound the staleness without new code.
    - **The owner's decision (2026-11-21):** keep the memory size as it is for now: "we need to see if it learn new stuff even with frozen predictions".
  - **A viable option (later): a warm-up with the latest weights before training.**
    - Before an update, re-forecast the replayed window's panels with the current weights, then train on it.
    - The panels chain: panel 3 at step j is transformer 1's forecast from step j − 1, which read panel 2, transformer 0's forecast from step j − 2. So the warm-up runs the window forward step by step from a few steps before it, each step's forecasts filling the next step's panels. That costs about one forward pass per step, or a few Jacobi passes over the window (as `uni_lookahead`).
    - It is the forecasts' counterpart of re-labeling the label at every replay (vm-00100), which fixed the same kind of staleness for the label.
- **Moves:** the step's move is the move made now, as today. Panels 2 to 4 predict 2 to 4 ticks after the real frame, so the moves' effects further out are in view; the instinct does not use them yet (see "Later").
- **Dreaming:** fill panel 1 from O1 as well, and the model runs on its own forecasts, which it has seen in panels 2 to 4 on every tick of its life.

## Noise

- **Why:**
  - a panel that always sees a slightly wrong input learns to clean it up and commit to a sharp frame, instead of averaging and passing the blur on;
  - the same trick keeps learned game engines from drifting over long rollouts: GameNGen (Valevski 2024) adds Gaussian noise to its context frames in training, and DIAMOND does the same.
- **Kinds of noise to try:**
  - **Gaussian** on the values, σ fixed or drawn per tick;
  - **sampled:** each value drawn from the model's own predicted bins instead of its mean. The noise is then the model's own uncertainty, sharp by construction;
  - **pixel noise** as `online.noise_in` (vm-00077).
- **Real pixels mixed into the noise (for later; the owner, 2026-11-21: "instead adding noise pixel we can partially add noise and partially add sampled pixels from real frame in case of a lag when real frames already exists - in this way we can control amount of info from real frames model sees and tune it in and out when we need real forecast and "noise" pixels are not available, or we can slowly replace proportion of real pixels "noise" with just random noise over time keeping the whole noise level at the same level"):**
  - **Needs a lag:** with a sensing lag the frames panels 2 to 4 forecast may already exist. The body holds them; the model has not seen them yet.
  - **The mix:** of the pixels a forecast panel replaces, a share r takes the real future frame's pixel and a share q random noise; the rest stays the forecast.
  - **A dial on real information:**
    - r tunes how much of the real future leaks into the panel, and can be turned up or down;
    - with r = 0 the panel is the pure forecast, as it must be when no real frame exists yet (no lag, or further out than the lag).
  - **A schedule, like scheduled sampling:**
    - start with a large r, so the panels carry mostly real future pixels and their targets are easy;
    - over the life move the share from real pixels to random noise (r down, q up), keeping r + q constant, so the panel's total corruption stays at one level while its real information fades to zero;
    - the model then ends on its own forecasts, having learned the panel layout and the chain while they were easy.
  - **To check:** that the model does not learn to copy the real pixels, so its loss on them stays honest. One option is to score only the pixels that are not real; another is to keep r small enough that copying does not pay.

## What it is for

- **Dreaming that stays composed:**
  - the model is trained every tick on inputs that are its own noised predictions, so a rollout that feeds its outputs back is what it has always seen;
  - to dream, the output goes back in as the next input, as in vm-00023, but now into a model trained for it.
- **Scoring moves further out, later:** the forecasts 2 to 4 ticks out are what a horizon or a separate chooser would read (see "Later"), as vm-00096's horizon (which helped under inertia), on forecasts trained against reality.
- **The label's horizon:** vm-00100's label is right within about 10 ticks of food and flat beyond. A dream that stays sharp for a few ticks lets the label be read further out.
- **The frame-level form of task 00114 Part A:** imagined outcomes of moves not made.

## Stages

1. **Panels 2 to 4 empty:** the 128 × 128 frame with only the real panel, and the four horizons' targets. This gives the cost, and the baseline for each horizon without the model's own forecasts.
2. **Forecast panels without noise**, L = 0 and L = 2.
3. **With noise:** Gaussian, then sampled.
4. **Dreaming and acting:** the dream curve (`eval_dream.py`), then the agent living with the panels (the instinct as now), in the world with inertia, against RD (vm-00100) and vm-00096's IH.

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

## The overnight proof of concept (2026-11-20/21)

**Goal:** numbers by 9 am showing whether the panels work. These are pilot runs, not an experiment: few seeds, no defaults changed. A run with statistics follows after the owner's review.

**The agent:**
- RD's settings (vm-00100: the re-labeled meal label in its own channel, full dose, inertia 4);
- the candidates' temperature if vm-00101 supports it (T = 2);
- `model.img = 128` with the four quarter transformers, one 64 × 64 panel each.

**Implementation:**
- **`world/panel_world.py`, `PanelWorld`:**
  - wraps the world. Its frame is the 2 × 2 composite: panel 1 (top left) the view sensed with the lag, panels 2–4 the forecasts set by `set_forecasts` before each step (noise before the first);
  - with the lag, the drives, `terminal_drives` and `hunger_cut` are delayed with the frame.
- **Options** (`online.panels = 1`; `panel_lag` L, default 0; `panel_noise` σ on [0, 1] values, default 0; `panel_decode` mean or pair; `panel_blank = 1`, panels 2–4 random grey noise drawn anew each tick: the no-forecast baseline):
  - need the universal model on the GPU, `band_n = 4`, `band_cols = 2`, `model.img` twice `env.view`;
  - refused with `env.delays`, the state, self-forcing, clog, `noise_in` and `off`.
- **Acting:**
  - after the move, each transformer's predicted values for its own panel at the newest step come from one `forward_uni_step` over the last ctx steps (pair or mean decoded);
  - transformers 1–3's forecasts, noised, become next tick's panels 2–4.
- **Memory:**
  - `GatedMemory(fut=4)`: long-term memory keeps 4 steps past each chunk (`ChunkStore(extra=max(vn, 4))`), and windows carry `f_img`, panel 1 of their steps and the 4 after;
  - the newest window ends 4 ticks back.
- **Training:**
  - the stored composites are the inputs (`update_rollout(inputs=...)`, as for `env.delays`);
  - the target composite at step j is [P1(j + 1), P1(j + 2); P1(j + 3), P1(j + 4)] (`online.panel_targets`): panel i against the real frame i ticks after panel 1's.
- **Measures:**
  - per-panel prediction loss (`uni_ce_patch_pK`, logged per transformer);
  - lived food, poison and hunger against TT (vm-00101) at the same ticks;
  - strips of composites saved every 5k ticks (`panels_TICK.png`, `online.panel_snap`).

**Iterations** (decided as results come in, logged here):
1. Smoke and timing: a few hundred ticks, the plumbing and the speed.
2. Short arms side by side:
   - PB (panels 2–4 blank);
   - PF (forecasts, no noise);
   - PN (forecasts with noise).
3. The longest affordable runs of the promising arms, and L = 2 if time allows.

**What ran (interim, 2026-11-21 07:30):**
- **Smoke and timing:** 7 tests (`tests/test_panels.py`). One full-size life sharing the card with 12 lives ran 1.6 ticks/s, of which the forecasts cost 55 ms a tick.
- **The pilot** (`pilot_00115/`, from 01:14):
  - started after vm-00101's runs, with T = 2 by vm-00101's rule (TT − RD food at 40–60k +16.06 ± 7.39);
  - 8 lives at once under MPS, seeds 1–2: PB, PF (pair decoding), PN (pair, σ .1), PL (PN with lag 2);
  - 0.5–0.8 ticks/s each, so about 20k ticks by 08:45, when they are stopped.
- **What the panels look like at 5k:**
  - in PF, panel 2 (one tick ahead) is a recognisable scene: food, poison, the cursor; panels 3–4 are fragments;
  - with σ .1 (PN, PL) the panels are mostly snow, the forecast faint under it: σ .1 is too strong.
- **Prediction loss in training batches** (mean of the 2 seeds, 10–14k):

  | Arm | Panel 1 | Panel 2 | Panel 3 | Panel 4 |
  |---|---|---|---|---|
  | PB | .846 | .895 | .892 | .898 |
  | PF | .840 | .911 | .921 | .938 |
  | PN | .847 | .895 | .911 | .924 |
  | PL | .832 | .894 | .908 | .916 |

  - Panel 1 is slightly better with forecasts; panels 2–4 are worse.
- **A frozen test** (`lifetime.py --init` with the pieces' checkpoints, `ppo.lr = 0`, 600 ticks in new worlds, seeds 101–102, the loss after tick 200; `--init` now loads the stem of the pieces' `_pK.bin`):

  | 5k checkpoint, seeds 1 / 2 | Panel 1 | Panel 2 | Panel 3 | Panel 4 |
  |---|---|---|---|---|
  | PB, blank (as it lived) | .936 / .935 | 1.024 / 1.075 | 1.026 / 1.065 | 1.020 / 1.064 |
  | PF, its own forecasts | .885 / .892 | .962 / .972 | .984 / 1.014 | 1.004 / 1.016 |
  | PF, forecasts blanked | .948 / .941 | 1.102 / 1.124 | 1.105 / 1.104 | 1.120 / 1.132 |

  - **Frozen, PF predicts better than PB on every panel,** by .04–.10, in both seeds.
  - **It uses its forecasts:** blanked, it falls below PB.
- **Why training batches said the opposite (a reading, not tested):**
  - memory keeps the panels as they were shown, so replayed windows carry forecasts from an older, worse model;
  - PF trains partly on its own stale forecasts, while the frozen test has fresh ones;
  - re-forecasting at replay (as the label is re-labeled) would remove this.
- **The frozen test at 10k,** PF with its forecasts minus PB (below 0: PF predicts better):

  | Checkpoint, seed | Panel 1 | Panel 2 | Panel 3 | Panel 4 |
  |---|---|---|---|---|
  | 5k, s1 | −.051 | −.062 | −.043 | −.016 |
  | 5k, s2 | −.043 | −.102 | −.051 | −.048 |
  | 10k, s1 | +.054 | −.041 | +.018 | +.006 |
  | 10k, s2 | +.023 | −.028 | −.036 | −.024 |

  - At 10k panel 2 is still better in both seeds, panel 1 worse in both, panels 3–4 split.
  - PF blanked at 10k is worse than with its forecasts on every panel (by up to .16): it still uses them.
  - **The test is noisy:** each frozen run acts for itself, so PF and PB see different frames, and the two seeds differ by .03–.05 within an arm. A clean test scores both on the same frames (the same moves replayed) with more seeds.

**The pilot's end (stopped 08:45 at 18–19k ticks):**

| Arm | Food 15–19k, s1 / s2 | Poison 15–19k | Panel 1 loss | Panel 2 | Panel 3 | Panel 4 |
|---|---|---|---|---|---|---|
| PB | 35.2 / 28.5 | 19.8 / 24.2 | .836 | .892 | .891 | .898 |
| PF | 19.0 / 21.0 | 26.0 / 22.7 | .833 | .906 | .917 | .938 |
| PN | 22.0 / 32.3 | 23.3 / 25.0 | .836 | .888 | .908 | .918 |
| PL | 13.7 / 19.0 | 27.7 / 25.0 | .825 | .887 | .903 | .914 |
| TT (img 64), s1 / s2 (s1–12) | 24.0 / 41.8 (40.1) | 18.8 / 14.8 (15.4) | | | | |

(the losses: training batches, mean of the 2 seeds.)

- **The forecasts at 15k** (`panels_0015001.png`):
  - in PF panel 2 is a sharp scene, panel 3 has parts of the objects, panel 4 a few;
  - in PN the forecasts under the noise are whole blobs, food and poison, out to panel 3. That is the owner's reason for the noise, though only by eye.
- **Training-batch loss:**
  - with noise (PN, PL) panel 2 is level with PB and panels 3–4 come closer to it than PF's; PF stays the worst on 3–4;
  - PL, sensing 2 ticks late, has the lowest panel-1 loss and the lowest food.
- **Food:** PB forages best at 15–19k; PF and PL the least. With 2 seeds the arms overlap.

**Readings of the pilot:**
1. **The plumbing works:** the composite, the lag, the targets 1–4 ticks ahead and the memory's 4 extra steps ran 18–19k ticks in 8 lives without an error.
2. **The forecasts are learned and used:** panel 2 is a recognisable scene by 5k, and blanking the forecasts raises PF's loss on every panel at 5k and 10k.
3. **Whether they help is not settled:**
   - frozen, PF predicted better than PB on every panel at 5k, but at 10k only panel 2 stayed ahead in both seeds;
   - in training batches, which replay stale forecasts, PF is worse on panels 2–4;
   - in training batches the noise (σ .1) brings panel 2 level with PB and panels 3–4 closer to it (PN .908 and .918 against PB .891 and .898).
4. **Cost:** at 128 × 128 a life runs at about a quarter of the 64 × 64 agent's speed when sharing the card (0.8 ticks/s against TT's 3.0 with 12 lives), so 60k ticks need about a day.

**For the owner's review (next steps to choose from):**
- a statistically sound run of PB against PF or PN, scored on the same frames, with 6–12 seeds; at this speed 60k ticks take about a day;
- σ smaller than .1 (for example .03), since .1 hides the forecasts;
- the warm-up re-forecast (above) if stale forecasts in memory prove to be the brake.

## The full run: vm-00102 (the owner, 2026-11-21: "lets do the full run and reduce noiise to the level you suggested")

- **Arms:** PB (panels 2–4 noise) against PN (forecasts, pair, σ .03), 5 seeds each, 60k ticks, 10 lives at once on the RTX 3090 (the 4090 was not visible). The owner chose 5 + 5 at 60k over 12 + 12 at 30k, and no lag arm.
- **Measures:** as the pilot, plus the frozen test at every 10k checkpoint for all 10 lives (`make frozen`), and the video (`make video`).
- **New for it:** `online.panel_record = N` (body 0's composites of the first N ticks in `panel_frames.npy`) and `dev/panel_video.py` (the panels as seen beside the real frames they aim at).
- **README:** [`../00102-forecast-panels/README.md`](../00102-forecast-panels/README.md). Breakout (task 00111) is renumbered vm-00103.

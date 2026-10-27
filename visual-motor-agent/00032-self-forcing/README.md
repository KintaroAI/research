# vm-00032: Self-Forcing — Training on the Model's Own Predictions

**Date:** 2026-10-27
**Status:** Complete (pending tag)
**Source:** *tagged on completion as `exp/vm-00032`* (code: `visual-motor-agent/dev/`)
**Outputs:** `~/data/research/visual-motor-agent/exp_00032/`
**Task plan:** [`../tasks/00039-self-forcing.md`](../tasks/00039-self-forcing.md)

## Goal

The universal model learns only from real windows, so run on its own output (the dream) it drifts: with pair
decoding (task 00038) the still scene survives, but the cursor is lost or displaced within a few steps. Self-forcing:
some training windows get their last steps from the model's own predictions, with real targets, so the model learns
to correct its own errors. Does the dream stay true longer, and at what cost to prediction and behaviour?

## Hypothesis

1. The dream stays closer to the truth, the cursor most of all.
2. One-step prediction is not worse.
3. With a truer imagination the 4-step lookahead (task 00029, `pair`) gains more over the 1-step instinct.

## Method

- **Self-forcing** (task 00039): in each update, 8 replayed windows (of 33) get their last 1–3 steps' inputs (frame
  and body) from the model's own predictions (`online.dream_inputs`: the body from the previous move token, the frame
  by 2 Jacobi iterations from the previous frame, `pair` decoding); moves stay real, targets stay real
  (`vmc_set_uni_inputs`). The same loss and the same rule for every token. Tested against the PyTorch reference
  (loss and every gradient with dreamed inputs); full suites pass (548).
- **The dream curve** (`eval_dream.py`): the frozen agent forages in 16 fresh worlds; from 4 points of each life the
  model dreams 16 steps with the moves that were really made next (exact sequential decoding, `pair`), scored against
  the real frames: pixels exactly right, the cursor (the share of its white pixels drawn white; the distance between
  centroids), the hunger error.
- **Runs:** the default universal lifetime (`configs/U_universal.json`, bin embeddings, vm-00030's U) with
  self-forcing (`sf_windows=8, sf_steps=3, sf_jacobi=2`), 100k ticks, seeds 1–3, against vm-00030's three runs (the
  same without). 29 ticks/s per lifetime (the baseline ~63): the dreams cost 9 extra forward passes per update.
- **Also:** the frozen tests at 25k/50k/100k (eval_self, forage with the instinct on 32 worlds; the policy alone and
  the click test on real windows at 100k); the forage test on 16 worlds with the 1-step instinct and the 4-step
  `pair` lookahead.

## Log

### 2026-10-26/27
- **Implemented** task 00039 and `eval_dream.py`; ran 22:49–00:20.
- **Corrected** the dream curve's cursor rule: "lost below 10 white pixels" counted the self-forcing models' smaller
  cursor (about 9 white pixels against 12–13) as lost; replaced by the share of the real cursor's white pixels drawn
  white, lost below 3 (all numbers below use it).
- The 4-step forage tests of SF seeds 1 and 2 failed once (`forward_uni_step failed`, while other jobs were starting
  on that GPU) and were rerun.

## Results

### R1. The dream curve at 100k (pixels right / cursor pixels right / cursor distance, px / hunger error)

| Run | 1 step | 4 steps | 8 steps | 16 steps |
|---|---|---|---|---|
| base s1 | 0.977 / 0.42 / 1.1 / 0.002 | 0.962 / 0.15 / 3.5 / 0.011 | 0.940 / 0.07 / 6.4 / 0.018 | 0.912 / 0.01 / 18.6 / 0.031 |
| **SF s1** | 0.980 / 0.47 / 0.8 / 0.017 | 0.967 / 0.28 / 1.4 / 0.030 | 0.949 / 0.16 / 2.4 / 0.032 | 0.926 / 0.07 / **4.2** / 0.053 |
| base s2 | 0.975 / 0.49 / 1.0 / 0.011 | 0.951 / 0.19 / 2.4 / 0.025 | 0.932 / 0.08 / 5.4 / 0.036 | 0.904 / 0.02 / 10.9 / 0.040 |
| **SF s2** | 0.977 / 0.35 / 0.8 / 0.008 | 0.961 / 0.15 / 1.7 / 0.004 | 0.937 / 0.08 / 3.3 / 0.004 | 0.914 / 0.03 / **5.2** / 0.014 |
| base s3 | 0.972 / 0.68 / 0.5 / 0.008 | 0.941 / 0.33 / 1.6 / 0.037 | 0.922 / 0.17 / 3.4 / 0.040 | 0.888 / 0.05 / 7.6 / 0.067 |
| **SF s3** | 0.981 / 0.41 / 0.7 / 0.005 | 0.967 / 0.21 / 1.3 / 0.010 | 0.943 / 0.12 / 2.5 / 0.008 | 0.907 / 0.06 / **3.6** / 0.033 |

The cursor's distance after 16 dreamed steps over the life:

| | 25k | 50k | 100k |
|---|---|---|---|
| base s1 / s2 / s3 | 26.0 / 25.5 / 19.9 | 16.1 / 17.4 / 11.2 | 18.6 / 10.9 / 7.6 |
| SF s1 / s2 / s3 | 5.6 / 8.2 / 8.0 | 6.0 / 5.8 / 6.8 | 4.2 / 5.2 / 3.6 |

### R2. Behaviour

| Run | eval_self 25k / 50k / 100k | forage at 100k: food / poison (cap) | policy alone | click test: food / poison right, nothing flat | forage (16 worlds): 1-step → 4-step pair | online 80–100k: food / poison (cap) |
|---|---|---|---|---|---|---|
| base s1 | 0.74 / 0.75 / 0.74 | 6.4 / 1.6 (0.01) | 2.8 / 2.1 | 0.66 / 0.85 / 0.82 | 7.2 / 1.5 → 6.9 / 0.5 | 5.2 / 1.0 (0.01) |
| SF s1 | 0.51 / 0.38 / 0.38 | 2.7 / 2.0 (0.02) | 2.8 / 3.0 | 0.30 / 0.75 / 0.81 | 2.6 / 0.6 → 2.8 / 0.9 | 3.2 / 4.3 (0.10) |
| base s2 | 0.86 / 0.94 / 0.94 | 10.5 / 1.8 (0.02) | 6.9 / 3.8 | 0.69 / 0.83 / 0.89 | 10.3 / 1.6 → 13.4 / 3.2 | 8.7 / 4.0 (0.05) |
| SF s2 | 0.99 / 0.98 / 0.95 | 10.4 / 3.2 (0.01) | 7.2 / 5.5 | 0.46 / 0.90 / 0.79 | 8.9 / 1.1 → 11.6 / 1.7 | 9.6 / 2.1 (0.02) |
| base s3 | 0.82 / 0.67 / 0.87 | 11.0 / 4.4 (0.03) | 5.8 / 5.9 | 0.51 / 0.81 / 0.91 | 10.4 / 8.4 → 13.7 / 7.2 | 5.0 / 7.5 (0.07) |
| SF s3 | 0.88 / 0.86 / 0.90 | 8.8 / 2.8 (0.03) | 3.2 / 2.3 | 0.87 / 0.60 / 0.84 | 9.3 / 2.9 → 8.4 / 3.8 | 5.4 / 11.3 (0.13) |

## Analysis

- **The dream stays truer, the cursor's position most** (hypothesis 1, supported): after 16 dreamed steps the cursor
  is 3.6–5.2 px from the real one against 7.6–18.6 in every seed, and from 25k on (5.6–8.2 against 19.9–26.0); the
  whole frame is more often exactly right at every horizon in every seed (0.907–0.926 against 0.888–0.912 at 16
  steps). The cursor's *drawing* is not better: the share of its white pixels drawn white is about the same (8 steps:
  0.08–0.16 against 0.07–0.17). The self-forcing models draw a smaller, more cautious cursor in the right place (the
  dream videos: a mostly black blob, the cursor's outline, rather than a crisp white cross). Dreamed hunger is better
  in two seeds of three.
- **One-step prediction is not worse** (hypothesis 2, supported): pixels exactly right one step ahead 0.977–0.981
  against 0.972–0.977. (The online cross-entropies are not comparable: a quarter of the windows had dreamed inputs.)
- **The lookahead does not gain more** (hypothesis 3, not supported): from the 1-step instinct to the 4-step `pair`
  lookahead the baseline gains food in two seeds (10.3 → 13.4, 10.4 → 13.7), self-forcing in one (8.9 → 11.6).
- **Behaviour is no better, and seed 1 is worse:** SF seed 1 is a weak agent (eval_self 0.38, forage 2.7 food per
  1,000, online at the cap 10% of the time); seed 2 is as good as its baseline, seed 3 a little worse (online poison
  11.3 per 1,000, at the cap 13%). The frozen click test is mixed (food right 0.30–0.87 against 0.51–0.69). Seeds of
  this lifetime vary widely, so three cannot tell a small effect, but nothing points to better behaviour.

## Conclusions

1. **Self-forcing works for what it targets:** with the same loss on the model's own predictions as inputs, the
   dream's cursor stays 2–4× closer to the truth over 16 steps, and one-step prediction does not suffer.
2. It does not (yet) make the agent better: the 1-step instinct does not use the dream, and the 4-step lookahead did
   not gain more from it. At 2–3× the cost per tick it is not a default.
3. A better dream is now available when acting needs one (a longer lookahead, planning); its remaining weakness is
   the cursor's drawing, not its place.

## Next Steps

- Self-forcing in the text world (task 00041, vm-00034).
- Longer dreamed spans in training (sf_steps 6) if longer dreams are needed.

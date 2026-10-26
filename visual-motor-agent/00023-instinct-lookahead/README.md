# vm-00023: The Instinct Looks 4 Steps Ahead

**Date:** 2026-10-25
**Status:** Complete
**Source:** `exp/vm-00023` (code: `visual-motor-agent/dev/`)
**Outputs:** `~/data/research/visual-motor-agent/exp_00023/` (the frozen tests on vm-00021: its run directories' `eval/`)
**Task plan:** [`../tasks/00029-instinct-lookahead.md`](../tasks/00029-instinct-lookahead.md)

## Goal

The instinct compares candidate moves by the model's prediction of the next hunger, one step ahead. Only a click changes
hunger in one step, so it is a click reflex: it cannot steer toward food. Here each candidate move is held for 4 steps
and the model imagines the 3 steps in between (its own predicted frames and drives): does the hunger it predicts after
4 steps choose better moves?

## Hypothesis

1. The model's 4-step prediction ranks moves better than its 1-step prediction: a path that crosses food while clicking
   shows relief, one that crosses poison a rise.
2. With it, the agent eats more and clicks poison less, frozen (vm-00021's checkpoints) and in its own lifetime.

## Method

- **Lookahead** (`online.instinct_h = 4`, task 00029): M = 16 candidates sampled from the model's move prediction, each
  held for 4 steps. The window keeps its last 5 real steps, then 3 imagined ones: an imagined step's drives are the
  previous move token's predicted next drives, its patches the model's own predictions, found by Jacobi iteration
  (J = 2 forward passes per imagined step, from the current frame) and decoded as each pixel's expected value
  (`instinct_decode = mean`) or most likely bin (`mode`). Score: the predicted hunger after the 4th move; only the first
  move is made. 7 forward passes of the 16 windows per tick.
- **Imagination diagnostics** (vm-00021 seed 3 at 200k, 16 forage worlds × 150 steps): each candidate's imagined
  hunger change against the world's (a copy of the world steps the held move).
- **Frozen:** the forage test (16 continual worlds × 1,000 steps) with the 1-step instinct and the 4-step lookahead on
  vm-00021's six seeds at 100k and 200k; the lookahead with mode decoding at 200k.
- **Lifetimes:** `configs/U_universal.json` with `instinct_h = 4`, 100k ticks, seeds 1–3 on the 4090 (18 ticks/s each,
  against 56–81 with the 1-step instinct), against vm-00022's U arm (the same configuration and seeds, 1 step).

## Log

### 2026-10-25
- **Implemented** the lookahead (task 00029): `online.uni_lookahead`, `vmc_forward_uni_step` (one step's predictions
  of a full forward pass, mean or mode decoding), frozen-test options for the forage test alone.
- **Ran** the three lifetimes 23:28–01:04; the frozen tests until 02:02.

### 2026-10-26
- **Fixed** the frozen tests' GPU model (`_GpuEval`), which lacked `forward_uni_step`: the 4-step tests had used full
  forward passes (the same values, slower) and the mode tests failed; reran the mode tests.

## Results

### R1. Imagination: the 4-step prediction against the world

Each candidate's imagined hunger change, by what the true held path clicks (true change in brackets):

| | no click event | food on the path | poison on the path | correlation |
|---|---|---|---|---|
| 1 step (the instinct) | +0.024 (+0.001) | −0.055 (−0.265) | +0.051 (+0.257) | 0.11 |
| 4 steps, imagined (mean, J = 2) | +0.048 (+0.004) | −0.074 (−0.350) | **−0.020** (+0.402) | 0.10 |
| 4 steps, imagined (mean, J = 8) | +0.048 | −0.100 | −0.027 | 0.11 |
| 4 steps, imagined (mode, J = 2) | +0.041 | −0.114 | −0.014 | 0.19 |
| 4 steps, the real frames (drives imagined) | +0.055 | −0.138 | +0.099 | 0.23 |

### R2. Frozen: vm-00021's checkpoints, the forage test

Food / poison clicks per 1,000 steps (time at the hunger cap):

| Seed | 1 step, 100k | 4 steps, 100k | 1 step, 200k | 4 steps, 200k | 4 steps (mode), 200k |
|---|---|---|---|---|---|
| 1 | 10.9 / 5.1 (0.02) | 11.6 / 6.7 (0.03) | 10.9 / 4.0 (0.02) | 16.8 / 7.2 (0.04) | 13.5 / 4.1 (0.01) |
| 2 | 4.2 / 5.4 (0.06) | 4.2 / 2.6 (0.02) | 3.4 / 6.1 (0.06) | 3.8 / 3.2 (0.02) | 4.6 / 2.1 (0.02) |
| 3 | 20.7 / 11.4 (0.05) | 28.2 / 14.9 (0.05) | 15.9 / 5.2 (0.01) | 14.2 / 7.3 (0.04) | 14.1 / 8.6 (0.04) |
| 4 | 27.9 / 27.2 (0.11) | 28.2 / 25.6 (0.12) | 25.6 / 24.5 (0.08) | 25.8 / 26.4 (0.10) | 30.9 / 24.9 (0.08) |
| 5 | 14.8 / 21.2 (0.10) | 16.4 / 30.5 (0.16) | 13.6 / 21.2 (0.13) | 15.7 / 29.2 (0.18) | 16.3 / 31.7 (0.16) |
| 6 | 5.9 / 6.3 (0.06) | 4.8 / 4.5 (0.02) | 3.9 / 3.2 (0.02) | 3.9 / 3.6 (0.03) | 3.6 / 4.2 (0.03) |

### R3. Lifetimes: online, against vm-00022's U (1 step), same seeds

| Run | food / poison per 1k: 0–20k | 40–60k | 80–100k | at the cap: 0–20k, 40–60k, 80–100k | instinct overrides the policy (80–100k) |
|---|---|---|---|---|---|
| U s1 | 14.0 / 22.4 | 10.2 / 16.3 | 8.6 / 6.8 | 0.14, 0.10, 0.05 | 0.06 |
| U s2 | 9.3 / 17.7 | 3.9 / 4.0 | 4.3 / 3.1 | 0.19, 0.03, 0.01 | 0.02 |
| U s3 | 25.8 / 37.2 | 22.8 / 24.6 | 20.1 / 12.8 | 0.20, 0.13, 0.04 | 0.08 |
| H4 s1 | 17.1 / 22.4 | 12.6 / 16.9 | 14.6 / 18.9 | 0.14, 0.09, 0.09 | 0.19 |
| H4 s2 | 3.6 / 4.2 | 4.2 / 4.5 | 2.6 / 4.5 | 0.03, 0.04, 0.05 | 0.03 |
| H4 s3 | 25.7 / 61.9 | 27.6 / 41.0 | 26.2 / 29.6 | 0.37, 0.17, 0.14 | 0.22 |

Cross-entropy at 80–100k: drives 0.24–0.28 and patches 0.72 in both arms; the move token 0.47–0.88 (H4) against
0.39–0.67 (U).

### R4. Lifetimes frozen: the forage test (16 worlds)

Food / poison clicks per 1,000 steps (time at the hunger cap); each model acts with the instinct it lived with, and the
H4 models at 100k also with the 1-step instinct:

| Run | 25k | 50k | 100k | 100k, 1-step instinct |
|---|---|---|---|---|
| H4 s1 | 14.1 / 17.3 (0.09) | 11.8 / 9.8 (0.06) | 17.3 / 14.9 (0.07) | 14.8 / **5.6** (0.01) |
| H4 s2 | 3.6 / 3.9 (0.03) | 5.8 / 6.4 (0.04) | 3.8 / 6.8 (0.04) | 4.3 / **4.0** (0.03) |
| H4 s3 | 27.1 / 39.7 (0.20) | 38.1 / 33.1 (0.15) | 25.9 / 19.2 (0.07) | 23.6 / **11.9** (0.04) |
| U s1 (1 step) | 13.9 / 26.0 (0.19) | 9.0 / 5.7 (0.03) | | 8.1 / 4.2 (0.03) |
| U s2 | 9.2 / 12.8 (0.09) | 4.5 / 5.7 (0.08) | | 4.9 / 3.8 (0.01) |
| U s3 | 25.8 / 28.9 (0.16) | 28.1 / 32.6 (0.13) | | 26.9 / 13.4 (0.04) |

## Analysis

- **The model can use 4 steps, but not with its own imagined frames.** Given the real future frames, its 4-step
  prediction ranks the held moves twice as well as the 1-step one (correlation with the truth 0.23 against 0.11), with
  every sign right. With imagined frames it is no better than 1 step (0.10), and a poison path is predicted to *relieve*
  hunger. The imagined frames keep the scene but lose the cursor: each patch's pixels are predicted independently, so
  where the cursor could be in one of a few places, neither the expected value (a smear) nor each pixel's most likely
  value (fragments) draws one cursor. Without a clear cursor the model cannot tell that the path reaches the poison.
- **Frozen, the lookahead does not beat the 1-step instinct** (hypothesis 2 rejected): over vm-00021's six seeds at
  100k and 200k, poison clicks are lower in 2 seeds (2 and 6 at 100k), about the same in one and higher in three (seed
  5: 21.2 → 29–31 per 1k); food clicks rise slightly in most. Decoding the imagined pixels by their mode instead of the mean
  (200k) raises food clicks in four seeds of six (seed 4: 25.6 → 30.9), as the diagnostics predicted (sharper frames
  show food relief better), but poison clicks are lower only in seed 2 (6.1 → 2.1) and higher in three.
- **In a lifetime it is worse:** against vm-00022's U (same seeds), more poison clicks in all three seeds at 80–100k
  (18.9 / 4.5 / 29.6 against 6.8 / 3.1 / 12.8) and more time at the hunger cap (0.09 / 0.05 / 0.14 against
  0.05 / 0.01 / 0.04); the lookahead overrides the policy's move 3× as often (0.19 and 0.22 in seeds 1 and 3).
- **The harm is in the choice, not in what the model learned:** the H4 lifetimes' models at 100k, given the 1-step
  instinct instead, click poison 2–3× less than with their own lookahead (5.6 / 4.0 / 11.9 against 14.9 / 6.8 / 19.2),
  about as U's models do (4.2 / 3.8 / 13.4). Living with the lookahead did not damage the model's prediction.
- **Cost:** 18 ticks/s against 56–81: 7 forward passes of 16 windows per tick.

## Conclusions

1. **Looking further ahead needs a coherent imagined future.** The model's 4-step prediction is good on real frames and
   poor on its own: per-pixel decoding of independent patch predictions cannot draw the one thing that matters, the
   cursor. Its imagined poison paths look like relief, so the lookahead clicks poison more.
2. **The 1-step instinct stays the default.** The lookahead code stays (`online.instinct_h`, default 1), with mean or
   mode decoding.
3. **What would make imagination work:** sampling one coherent frame instead of per-pixel decoding (sequential
   sampling patch by patch, or a token for the cursor's position that the patches condition on), or a lookahead that
   does not imagine frames at all (the model predicting the hunger several moves ahead from the real window, trained on
   real futures).

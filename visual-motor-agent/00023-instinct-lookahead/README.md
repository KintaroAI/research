# vm-00023: The Instinct Looks 4 Steps Ahead

**Date:** 2026-10-25
**Status:** In progress (analysis running)
**Source:** *tagged on completion as `exp/vm-00023`* (code: `visual-motor-agent/dev/`)
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
- **Ran** the three lifetimes 23:28–01:04; the frozen tests until (pending).

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
| 1 | 10.9 / 5.1 (0.02) | 11.6 / 6.7 (0.03) | 10.9 / 4.0 (0.02) | 16.8 / 7.2 (0.04) | (pending) |
| 2 | 4.2 / 5.4 (0.06) | 4.2 / 2.6 (0.02) | 3.4 / 6.1 (0.06) | 3.8 / 3.2 (0.02) | (pending) |
| 3 | 20.7 / 11.4 (0.05) | 28.2 / 14.9 (0.05) | 15.9 / 5.2 (0.01) | 14.2 / 7.3 (0.04) | (pending) |
| 4 | 27.9 / 27.2 (0.11) | 28.2 / 25.6 (0.12) | 25.6 / 24.5 (0.08) | 25.8 / 26.4 (0.10) | (pending) |
| 5 | 14.8 / 21.2 (0.10) | 16.4 / 30.5 (0.16) | 13.6 / 21.2 (0.13) | 15.7 / 29.2 (0.18) | (pending) |
| 6 | 5.9 / 6.3 (0.06) | 4.8 / 4.5 (0.02) | 3.9 / 3.2 (0.02) | 3.9 / 3.6 (0.03) | (pending) |

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

### R4. Lifetimes frozen: the forage test

| Run | 25k | 50k | 100k | 100k, 1 step |
|---|---|---|---|---|
| H4 s1 (4 steps) | 14.1 / 17.3 (0.09) | 11.8 / 9.8 (0.06) | 17.3 / 14.9 (0.07) | 14.8 / 5.6 (0.01) |
| H4 s2 | (pending) | | | |
| H4 s3 | (pending) | | | |
| U s1–s3 (1 step) | (pending) | | | |

## Analysis

(pending)

## Conclusions

(pending)

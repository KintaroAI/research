# vm-00031: A Long Life with Bin-Embedding Input

**Date:** 2026-10-26
**Status:** Complete
**Source:** `exp/vm-00031` (code: `visual-motor-agent/dev/`)
**Outputs:** `~/data/research/visual-motor-agent/exp_00031/`
**Task plan:** [`../tasks/00037-long-life-bin-input.md`](../tasks/00037-long-life-bin-input.md)

## Goal

vm-00030 made bin-embedding input (`model.in_bins`) the universal model's input: better prediction in both worlds,
but a narrower policy in the circle world (the move token's cross-entropy 0.17–0.23 against 0.39–0.67 at 100k). In
vm-00024/25 a narrowing policy was how traps began. Here vm-00025's long life (500k ticks, a 100k-step memory) is
rerun with bin embeddings, now the default: does surprise-gated memory still keep every life working, and is the memory
that keeps the most recent steps trapped sooner?

## Hypothesis

1. Surprise-gated memory (S) keeps all three lives off the hunger cap through 500k ticks, as in vm-00025.
2. The most recent steps (U) trap the lives as in vm-00025, and a narrower policy may trap them sooner.

## Method

- `in_bins = 1` is now set in every universal config (`U_universal.json`, `T_text_A/B/C.json`; task 00037).
- `configs/U_universal.json` (the default universal model with bin embeddings, the 1-step instinct), `online.replay =
  100,000` steps, 500k ticks, seeds 1–3, two of vm-00025's memories: **S**, the most surprising quarter
  (`store_surprise = 0.25`, `store_base = 0`), and **U**, every chunk, the oldest replaced. Compared with vm-00025's S
  and U (the same configuration, seeds and memories, with the linear input).
- S ran on the 4090 (99 ticks/s each), U on the 3090 (62), 18:37–20:52; the analysis until 21:02. The same measures as
  vm-00025: online clicks and time at the hunger cap per 20k window; the move token's cross-entropy over the life; the
  frozen tests at 200k and 500k (eval_self and the forage test with the instinct on 32 worlds; the policy alone at
  500k); the click test on real windows at 200k and 500k (8 steps of history, as vm-00025).

## Log

### 2026-10-26
- **Ran** the six lives 18:37–20:52. The first analysis queue passed the run directories as a glob made before they
  existed (each command then got all three); the frozen tests failed and the click tests ran early for S. Rerun one
  directory at a time, 20:46–21:02 (the S click tests, run after all S lives ended, were kept).

## Results

### R1. Online

Food / poison clicks per 1,000 steps in 20k windows, the share of time at the hunger cap in the same windows, the move
token's cross-entropy, the share of relief chunks in the memory. vm-00025 (the linear input) below each arm:

| Run | 80–100k | 180–200k | 280–300k | 380–400k | 480–500k | at the cap (same windows) | move cross-entropy 180–200k → 380–400k → 480–500k | relief share at 200k, 500k |
|---|---|---|---|---|---|---|---|---|
| **S s1** | 2.6 / 8.1 | 3.1 / 2.6 | 5.0 / 2.0 | 5.0 / 12.1 | 6.2 / 4.4 | 0.18 0.06 0.02 0.04 0.13 | 0.14 → 0.21 → 0.23 | 0.09, 0.16 |
| **S s2** | 12.4 / 6.2 | 18.9 / 6.5 | 20.1 / 4.7 | 23.0 / 5.6 | 24.6 / 4.1 | 0.02 0.02 0.01 0.02 0.00 | 0.12 → 0.18 → 0.19 | 0.33, 0.44 |
| **S s3** | 12.4 / 18.6 | 15.0 / 15.8 | 18.9 / 12.9 | 19.7 / 9.8 | 19.1 / 13.2 | 0.17 0.10 0.06 0.04 0.06 | 0.13 → 0.18 → 0.19 | 0.39, 0.41 |
| S, vm-00025 | 5.2–17.7 / 7.7–25.1 | 5.2–15.9 / 9.1–15.3 | 5.1–18.4 / 4.1–25.4 | 5.8–15.9 / 5.8–12.8 | 8.2–16.1 / 3.9–9.6 | 0.02–0.05 at the end | 0.23–0.30 → 0.33–0.46 → 0.36–0.51 | 0.22–0.42, 0.23–0.42 |
| **U s1** | 7.1 / 2.5 | 13.6 / 1.9 | 13.4 / 2.0 | 12.7 / 2.9 | 17.6 / 3.0 | 0.01 0.01 0.00 0.02 0.02 | 0.22 → 0.10 → 0.07 | 0.18, 0.25 |
| **U s2** | 2.4 / 1.9 | 2.1 / 0.7 | 0.5 / 0.3 | 0.0 / 0.0 | 0.0 / 0.0 | 0.06 0.31 0.77 **1.00 1.00** | 0.06 → 0.00 → 0.00 | 0.04, 0.00 |
| **U s3** | 19.2 / 12.2 | 22.5 / 12.7 | 17.0 / 17.1 | 7.5 / 38.9 | 0.0 / 0.0 | 0.08 0.09 0.20 0.56 **1.00** | 0.11 → 0.02 → 0.02 | 0.30, 0.01 |
| U, vm-00025 | 3.8–7.3 / 4.0–6.3 | 2.9–7.8 / 1.6–6.7 | 2.7–2.9 / 0.8–13.1 | 0.0–1.8 / 0.5–58.6 | 0.0 / 44.3–84.8 | **1.00** at the end, all three | 0.25–0.35 → 0.07 → 0.03–0.05 | 0.06–0.12, 0.00–0.02 |

The time at the cap per 20k window over the whole life:

| Run | 0 – 500k in 20k windows |
|---|---|
| S s1 | 0.2 0.4 0.6 0.1 0.2 0.4 0.7 0.1 0.1 0.1 0.1 0.3 0.0 0.1 0.0 0.0 0.1 0.1 0.1 0.0 0.0 0.0 0.0 0.0 0.1 |
| S s2 | 0.0 0.0 0.0 0.1 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 |
| S s3 | 0.2 0.3 0.1 0.1 0.2 0.1 0.1 0.1 0.2 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.0 0.1 0.1 0.1 0.1 0.1 |
| U s1 | 0.1 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.2 0.0 0.0 |
| U s2 | 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.3 0.9 0.7 0.7 1.0 0.8 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 |
| U s3 | 0.2 0.1 0.3 0.1 0.1 0.1 0.1 0.0 0.1 0.1 0.1 0.1 0.1 0.1 0.2 0.3 0.4 0.5 0.6 0.6 0.8 1.0 1.0 1.0 1.0 |

The patches' cross-entropy over the last 20k: 0.708–0.711 for S (vm-00025's S: 0.717–0.718); the body's 0.231–0.246
(0.242–0.260). S's chunks were made surprising by the move token (12,631–13,480), the body token (1,489–2,105) and
patches (661–980).

### R2. Frozen

| Run | eval_self success 200k / 500k | forage with the instinct, 200k: food / poison (cap) | 500k | policy alone, 500k | click test 200k: food / poison right, nothing flat | 500k |
|---|---|---|---|---|---|---|
| **S s1** | 0.57 / 0.89 | 4.8 / 3.2 (0.06) | 8.3 / 5.8 (0.04) | 6.8 / 10.1 (0.10) | 0.59 / 0.70 / 0.76 | 0.69 / 0.86 / 0.83 |
| **S s2** | 0.99 / 1.00 | 22.0 / 8.3 (0.02) | 27.3 / 5.4 (0.02) | 25.8 / 13.0 (0.06) | 0.74 / 0.86 / 0.91 | 0.79 / 0.91 / 0.93 |
| **S s3** | 0.99 / 1.00 | 20.1 / 14.6 (0.07) | 24.5 / 13.6 (0.05) | 23.0 / 25.8 (0.17) | 0.72 / 0.92 / 0.90 | 0.75 / 0.93 / 0.95 |
| S, vm-00025 | 0.96–1.00 / 0.98–0.99 | 9.7–19.4 / 9.2–12.8 | 12.7–23.1 / 4.9–11.9 (0.02–0.05) | 9.8–19.7 / 9.2–22.3 | 0.51–0.68 / 0.65–0.85 / 0.68–0.94 | 0.56–0.63 / 0.82–0.91 / 0.87–0.96 |
| **U s1** | 0.83 / 0.92 | 12.9 / 1.3 (0.00) | 20.0 / 2.7 (0.01) | 13.8 / 2.6 (0.03) | 0.72 / 0.83 / 0.91 | 0.72 / 0.90 / 0.90 |
| **U s2** | 0.65 / 0.16 | 3.3 / 1.5 (0.02) | 0.7 / 0.4 (0.19) | 0.3 / 0.1 (0.19) | 0.65 / 0.68 / 0.87 | 0.60 / **0.17** / **0.48** |
| **U s3** | 1.00 / 0.91 | 24.2 / 6.7 (0.02) | 11.3 / 18.3 (0.30) | 10.9 / 30.3 (0.49) | 0.58 / 0.93 / 0.94 | 0.50 / 0.69 / 0.94 |
| U, vm-00025 | 0.71–1.00 / 0.66–0.82 | 4.2–13.4 / 2.0–5.2 | 3.3–6.5 / 1.3–11.8 (0.02–0.28) | 2.4–10.9 / 1.7–31.8 | 0.60–0.82 / 0.63–0.83 / 0.86–0.90 | 0.17–0.78 / 0.59–0.83 / 0.58–0.91 |

## Analysis

- **Surprise-gated memory still keeps every life working** (hypothesis 1, supported): no S life was trapped, and all
  three improved over the life. Seeds 2 and 3 forage more than any vm-00025 life (food 19–25 per 1,000 in the last 20k,
  against 8–16) and are at the cap 0.00–0.06 of the time; seed 2 never spent more than a tenth of a window there.
  Seed 1 started badly (at the cap 0.4–0.7 in four windows before 140k, eval_self 0.57 at 200k) and recovered, but
  stays the weakest forager (food 6.2, poison 4.4 per 1,000 at the end).
- **The policy is narrower than vm-00025's S, and it still widens:** the move token's cross-entropy rises over the
  life in every S run (0.12–0.14 at 200k → 0.19–0.23 at 500k), as it did in vm-00025 (0.23–0.30 → 0.36–0.51), from
  half the level. Most of what S keeps is still the agent's unexpected moves (12,631–13,480 chunks by the move token);
  with bin embeddings the model predicts its own moves better, and the surprise gate still finds the ones it does not.
  A narrower policy did not trap S here.
- **Keeping the most recent steps still traps** (hypothesis 2, partly): two U lives of three were trapped, s2 from
  about 200k (at the cap 0.7–1.0 from 200k, 1.0 from 300k) and s3 from about 340k (1.0 from 420k), each with a
  collapsed policy (move cross-entropy 0.00–0.02), a memory without relief (0.00–0.01) and, once trapped, no clicks
  at all (s2) or poison first (s3: 38.9 per 1,000 at 380–400k). In vm-00025 all three were trapped (from 340–480k).
  Seed 2 was trapped sooner than any vm-00025 life; seed 1 was never trapped and forages well to the end (food 17.6,
  poison 3.0, at the cap 0.02). Three seeds cannot say whether bin embeddings trap U sooner or less often.
- **Trapped models are damaged:** frozen in fresh worlds, U s2 at 500k barely acts (eval_self 0.16, food 0.7 per
  1,000) and predicts poison clicks right in only 17% of cases; its body cross-entropy in the life fell to 0.000 (it
  predicts a hunger stuck at the cap perfectly). U s3 frozen clicks poison 18–30 per 1,000 and sits at the cap
  0.30–0.49 of the time. vm-00025's trapped models were mostly sound in fresh worlds (at the cap 0.02–0.04, except s1).
- **Prediction:** bin embeddings keep vm-00030's gain over a long life (patch cross-entropy 0.708–0.711 against
  0.717–0.718 for S), and the click test at 500k is as good or better (S: food right 0.69–0.79 against 0.56–0.63,
  poison 0.86–0.93 against 0.82–0.91).

## Conclusions

1. **Bin-embedding input and surprise-gated memory work together:** over 500k ticks no S life was trapped; the best
   two forage more than any linear-input life, and the model predicts better. The narrower policy that bin embeddings
   bring still widens over the life under the surprise gate.
2. **Keeping the most recent experience is still a trap** (two of three lives, one of them sooner than with the linear
   input), and with bin embeddings a trapped model is damaged in fresh worlds too.
3. Bin embeddings stay the default input. Surprise gating (`store_surprise = 0.25`, `store_base = 0`) should become the
   default memory, as vm-00025 proposed.

## Next Steps

- Make surprise gating the default memory in `configs/U_universal.json`, with a test that the default memory is gated.
- Seed 1's slow start (at the cap up to 0.7 of a window before 140k) is the remaining weakness of S: the gate is idle
  until the memory first fills; a longer or more varied warmup may help.
